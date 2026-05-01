#!/usr/bin/env python3
"""
chat_gui.py — Ask a Coast Guard Lawyer  (GUI Application)
==========================================================

Windowed GUI chatbot for the CG Legal RAG system.
Built with Tkinter + Pillow. Preserves all retrieval logic from chat.py.

Launch:
    python src/chat_gui.py
    python src/chat_gui.py --llm "C:\\path\\to\\mistral-7b-instruct.Q4_K_M.gguf"
    python src/chat_gui.py --retriever hybrid --rerank

Requires:
    pip install Pillow
    (all other deps same as chat.py)
"""

import json
import sys
import os
import re
import argparse
import threading
import textwrap
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import tkinter.font as tkfont

# ── Pillow for background image ───────────────────────────────────────────────
try:
    from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageDraw
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    print("[WARN] Pillow not found. Run: pip install Pillow")
    print("       Background image will not be shown.")

# ── Core RAG system defaults ──────────────────────────────────────────────────
INDEX_DIR  = Path("data/index")
INDEX_NAME = "pio_rag"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 8

# ── Color palette (derived from the background image) ────────────────────────
NAVY_DARK   = "#1a2d45"   # deep navy panels
NAVY_MID    = "#273b54"   # mid navy
GOLD        = "#c9a84c"   # brass/gold from the scales
GOLD_LIGHT  = "#dfc06a"   # lighter gold for hover
GOLD_DARK   = "#a8873c"   # darker gold for borders
CREAM       = "#f0eeea"   # near-white for chat bg
OFF_WHITE   = "#e8e6e1"   # panel background
RED_ACCENT  = "#8b1a1a"   # deep military red
TEXT_DARK   = "#0e1a26"   # near-black text
TEXT_MID    = "#2c3e50"   # body text
CHAT_USER   = "#1e3a5f"   # user message header color
CHAT_BOT_BG = "#f7f5f0"   # bot message background
CHAT_USER_BG= "#e8f0f8"   # user message background
DISCLAIMER_BG="#fef9ec"   # warm cream for disclaimer box
BORDER      = "#c9a84c"   # gold border

DISCLAIMER = (
    "⚠  NOT LEGAL ADVICE — This system summarises official USCG materials only. "
    "Consult your chain of command or legal office for guidance on specific cases."
)

DEMO_QUESTIONS = [
    # MCM — Article-specific (article filter activates; very reliable output)
    "What are the elements of Article 92 failure to obey an order?",
    "What is the maximum punishment for larceny under Article 121?",
    "What does Article 112a prohibit regarding wrongful use of controlled substances?",
    "What are the elements required to prove assault under Article 128?",
    # CG Conduct Manual — policy questions
    "What is the Coast Guard policy on hazing and what conduct does it prohibit?",
    "What fraternization is prohibited between officers and enlisted members?",
    # Separations and Substance Abuse
    "What happens to a member's career after a confirmed drug incident finding?",
    "What are the separation consequences for a second alcohol incident?",
    # Cross-document / procedure
    "What rights does a member have during a Coast Guard administrative investigation?",
    "Under what conditions can an enlisted member be separated for drug-related misconduct?",
]

# ── Optional RAG imports ──────────────────────────────────────────────────────
try:
    from hybrid_retrieval import load_bm25_index, hybrid_retrieve, detect_article_filter
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False
    def detect_article_filter(q): return None

try:
    from reranker import Reranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# RAG ENGINE  (all logic preserved from chat.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_index(index_dir: Path, index_name: str):
    import faiss
    idx  = faiss.read_index(str(index_dir / f"{index_name}.faiss"))
    with open(index_dir / f"{index_name}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta


def dense_retrieve(query, index, meta, model, top_k, article_filter):
    import numpy as np
    vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    k_pool  = min(len(meta), max(top_k * 20, 200))
    scores, indices = index.search(vec, k_pool)
    ranked  = [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]

    if article_filter:
        art = str(article_filter).lstrip("0").lower()
        results = [
            {**meta[i], "score": s}
            for i, s in ranked
            if str(meta[i].get("article_number", "")).lstrip("0").lower() == art
        ][:top_k]
        if results:
            return results, "filtered"
        return [{**meta[i], "score": s} for i, s in ranked[:top_k]], "fallback"

    return [{**meta[i], "score": s} for i, s in ranked[:top_k]], "unfiltered"


def _norm_text(text: str) -> str:
    text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
    lines = [l for l in text.split("\n") if l.strip()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def aggregate_chunks(chunks):
    seen, passages = set(), []
    for c in chunks:
        cid = c.get("chunk_id", "")
        if cid in seen:
            continue
        seen.add(cid)
        art = c.get("article_number", "")
        sec = c.get("section_number", "")
        key = art if art else sec
        merged = False
        if key and passages:
            last = passages[-1]
            if (last.get("article_number") == art and art) or \
               (last.get("section_number") == sec and sec):
                last["text"] += "\n\n" + c["text"]
                last.get("chunk_ids", [last.get("chunk_id", "")]).append(cid)
                merged = True
        if not merged:
            p = dict(c)
            p["chunk_ids"] = [cid]
            passages.append(p)
    return passages


def format_citation(c: dict, rank: int) -> str:
    src   = c.get("source", "Unknown")
    art   = c.get("article_number", "")
    sec   = c.get("section_number", "")
    title = (c.get("article_title") or c.get("section_title") or "").strip()
    hp    = c.get("heading_path", [])
    p1    = c.get("page_start", "")
    p2    = c.get("page_end",   "")
    score = c.get("ce_score", c.get("score", 0.0))
    cids  = c.get("chunk_ids", [c.get("chunk_id", "?")])

    if art:
        loc = f"Art. {art}" + (f" — {title}" if title else "")
    elif sec:
        loc = f"§ {sec}" + (f" {title}" if title
                             else (f" {hp[2]}" if isinstance(hp, list) and len(hp) >= 3 else ""))
    else:
        loc = " > ".join(str(h) for h in hp[-2:]) if hp else "?"

    pages     = f"pp.{p1}–{p2}" if p1 and p2 else (f"p.{p1}" if p1 else "")
    chunk_str = ", ".join(cids[:2]) + ("…" if len(cids) > 2 else "")
    return f"[{rank}]  {src}  |  {loc}  |  {pages}  |  {chunk_str}  |  score: {score:.3f}"


def run_llm(query: str, passages: list, llm_path: str) -> str:
    try:
        from llama_cpp import Llama
    except ImportError:
        return "[LLM unavailable — install llama-cpp-python]"

    context_parts = []
    for i, p in enumerate(passages[:6], 1):
        art = p.get("article_number", "")
        sec = p.get("section_number", "")
        src = p.get("source", "")
        loc = f"Art. {art}" if art else (f"§ {sec}" if sec else src)
        context_parts.append(f"--- Source [{i}]: {src} | {loc} ---\n{p['text'].strip()}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a legal information assistant for U.S. Coast Guard personnel. "
        "Your role is to summarise official USCG legal and policy documents. "
        "You do NOT provide legal advice. You only summarise what official sources say. "
        "Every factual claim must be grounded in the provided source excerpts. "
        "If information is not in the excerpts, say so explicitly."
    )
    user_prompt = (
        f"Using ONLY the following source excerpts, answer this question:\n\n"
        f"QUESTION: {query}\n\n"
        f"SOURCE EXCERPTS:\n{context}\n\n"
        "Provide a clear, structured answer with:\n"
        "1. A plain-language summary (2–4 sentences)\n"
        "2. Key elements or conditions (if applicable)\n"
        "3. Important exceptions or limitations\n"
        "4. Citations for each claim (reference Source [N] numbers above)\n\n"
        "End with: '⚠ This is a summary of official sources only, not legal advice.'"
    )
    try:
        llm = Llama(model_path=llm_path, n_ctx=8192, n_threads=4, verbose=False)
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1, max_tokens=1500, repeat_penalty=1.1,
        )
        return out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# GUI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class CGLegalApp:
    """Main GUI application window."""

    # ── Initialization ────────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk, args):
        self.root       = root
        self.args       = args
        self.demo_idx   = 0
        self.thinking   = False

        # RAG engine state (loaded in background thread)
        self.index       = None
        self.meta        = None
        self.embed_model = None
        self.bm25_data   = None
        self.reranker    = None
        self.llm_mode    = (args.llm is not None)
        self.retriever_label = "Loading..."

        self._setup_window()
        self._setup_fonts()
        self._setup_background()
        self._build_ui()
        self._start_loading()

    # ── Window setup ──────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title("Ask a Coast Guard Lawyer — CG Legal RAG System")
        self.root.geometry("1280x860")
        self.root.minsize(900, 650)
        self.root.configure(bg=NAVY_DARK)

        # Center window on screen
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = (sw - 1280) // 2
        y  = (sh - 860)  // 2
        self.root.geometry(f"1280x860+{x}+{y}")
        self.root.lift()

    def _setup_fonts(self):
        """Define all fonts used in the interface."""
        self.font_title    = tkfont.Font(family="Georgia",     size=18, weight="bold")
        self.font_subtitle = tkfont.Font(family="Georgia",     size=10, slant="italic")
        self.font_label    = tkfont.Font(family="Georgia",     size=9)
        self.font_user     = tkfont.Font(family="Georgia",     size=10, weight="bold")
        self.font_body     = tkfont.Font(family="Courier New", size=10)
        self.font_citation = tkfont.Font(family="Courier New", size=9,  slant="italic")
        self.font_input    = tkfont.Font(family="Georgia",     size=13)
        self.font_btn      = tkfont.Font(family="Georgia",     size=10, weight="bold")
        self.font_status   = tkfont.Font(family="Courier New", size=8)
        self.font_disclaimer = tkfont.Font(family="Georgia",   size=9,  slant="italic")

    # ── Background image ──────────────────────────────────────────────────────
    def _setup_background(self):
        """Load and prepare the background image."""
        self.bg_photo = None
        self._bg_img_path = Path(__file__).parent.parent / "assets" / "CG_Legal_RAG_GUI_Image.png"

        # Also check next to the script itself, and common relative paths
        candidates = [
            self._bg_img_path,
            Path("assets/CG_Legal_RAG_GUI_Image.png"),
            Path("CG_Legal_RAG_GUI_Image.png"),
            Path(__file__).parent / "CG_Legal_RAG_GUI_Image.png",
        ]
        self._bg_source = None
        for p in candidates:
            if p.exists():
                self._bg_source = p
                break

    def _render_background(self, width: int, height: int):
        """Resize + darken the BG image to fill the window. Called on resize."""
        if not _PIL_AVAILABLE or not self._bg_source:
            return

        try:
            img = Image.open(self._bg_source).convert("RGB")

            # Scale to fill window maintaining aspect
            img_ratio  = img.width / img.height
            win_ratio  = width / height

            if img_ratio > win_ratio:
                new_h = height
                new_w = int(height * img_ratio)
            else:
                new_w = width
                new_h = int(width / img_ratio)

            img = img.resize((new_w, new_h), Image.LANCZOS)

            # Crop to exact window size from center
            left = (new_w - width)  // 2
            top  = (new_h - height) // 2
            img  = img.crop((left, top, left + width, top + height))

            # Darken slightly so text remains readable
            enhancer = ImageEnhance.Brightness(img)
            img      = enhancer.enhance(0.75)

            self.bg_photo = ImageTk.PhotoImage(img)
            self.bg_label.configure(image=self.bg_photo)
            self.bg_label.image = self.bg_photo
        except Exception as e:
            print(f"[WARN] Background image failed: {e}")

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        """Build all UI components."""
        # ── Full-window background canvas ────────────────────────────────────
        self.bg_label = tk.Label(self.root, bg=NAVY_DARK)
        self.bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ── Main content column (centered, 820px wide) ───────────────────────
        self.main_frame = tk.Frame(self.root, bg="")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center",
                               relwidth=0.68, relheight=0.90)
        self.main_frame.configure(bg=NAVY_DARK)  # fallback if no image

        # ── Header ───────────────────────────────────────────────────────────
        self._build_header()

        # ── Mode / status bar ─────────────────────────────────────────────────
        self._build_status_bar()

        # ── Chat display ──────────────────────────────────────────────────────
        self._build_chat_area()

        # ── Input area ───────────────────────────────────────────────────────
        self._build_input_area()

        # ── Disclaimer strip ──────────────────────────────────────────────────
        self._build_disclaimer()

        # ── Resize handler ────────────────────────────────────────────────────
        self.root.bind("<Configure>", self._on_resize)
        self.root.after(100, lambda: self._on_resize(None))

    def _build_header(self):
        """Gold-bordered header with title and anchor emblem."""
        header = tk.Frame(
            self.main_frame,
            bg=NAVY_DARK,
            bd=0, relief="flat",
        )
        header.pack(fill="x", padx=0, pady=(0, 4))

        # Gold top border line
        top_line = tk.Frame(header, bg=GOLD, height=2)
        top_line.pack(fill="x")

        # Inner header content
        inner = tk.Frame(header, bg=NAVY_DARK, pady=12, padx=18)
        inner.pack(fill="x")

        # Left: anchor symbol
        tk.Label(
            inner, text="⚓", font=("Georgia", 28),
            fg=GOLD, bg=NAVY_DARK,
        ).pack(side="left", padx=(0, 14))

        # Center: title block
        title_block = tk.Frame(inner, bg=NAVY_DARK)
        title_block.pack(side="left", fill="x", expand=True)

        tk.Label(
            title_block,
            text="COAST GUARD LEGAL ASSISTANT",
            font=self.font_title,
            fg=GOLD, bg=NAVY_DARK,
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            title_block,
            text="AI-Powered Legal Reference  ·  Sources: MCM 2024, Conduct Manual, Investigations, Separations, Substance Abuse",
            font=self.font_subtitle,
            fg="#9ab0c8", bg=NAVY_DARK,
            anchor="w",
        ).pack(anchor="w")

        # Right: scales of justice symbol
        tk.Label(
            inner, text="⚖", font=("Georgia", 28),
            fg=GOLD, bg=NAVY_DARK,
        ).pack(side="right", padx=(14, 0))

        # Gold bottom border
        tk.Frame(header, bg=GOLD, height=2).pack(fill="x")

    def _build_status_bar(self):
        """Mode toggle, retriever info, demo button."""
        bar = tk.Frame(self.main_frame, bg=NAVY_MID, pady=5, padx=14)
        bar.pack(fill="x", pady=(0, 3))

        # Left: status text
        self.status_var = tk.StringVar(value="⏳  Loading system …")
        self.status_lbl = tk.Label(
            bar, textvariable=self.status_var,
            font=self.font_status, fg=GOLD_LIGHT, bg=NAVY_MID,
            anchor="w",
        )
        self.status_lbl.pack(side="left")

        # Right: buttons
        btn_frame = tk.Frame(bar, bg=NAVY_MID)
        btn_frame.pack(side="right")

        self.mode_btn = self._make_btn(
            btn_frame, "[ Source View Mode ]", self._toggle_mode,
            width=16,
        )
        self.mode_btn.pack(side="left", padx=(0, 6))

        self._make_btn(btn_frame, "Demo ▸", self._insert_demo, width=8).pack(side="left", padx=(0, 6))
        self._make_btn(btn_frame, "Clear", self._clear_chat, width=6).pack(side="left")

    def _build_chat_area(self):
        """Scrollable chat display with gold border."""
        # Outer bordered frame
        border_frame = tk.Frame(
            self.main_frame,
            bg=GOLD_DARK, bd=1, relief="flat",
        )
        border_frame.pack(fill="both", expand=True, pady=(0, 4))

        # Inner chat widget with scrollbar
        inner = tk.Frame(border_frame, bg=CREAM)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        self.chat = tk.Text(
            inner,
            wrap="word",
            font=self.font_body,
            bg=CREAM,
            fg=TEXT_DARK,
            bd=0,
            relief="flat",
            state="disabled",
            cursor="xterm",
            spacing1=2,   # px above each line
            spacing3=2,   # px below each line
            padx=14,
            pady=10,
            selectbackground=GOLD,
            selectforeground=TEXT_DARK,
        )
        self.chat.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(inner, command=self.chat.yview, bg=NAVY_DARK, troughcolor=NAVY_MID)
        scroll.pack(side="right", fill="y")
        self.chat.configure(yscrollcommand=scroll.set)

        # ── Text tags for rich styling ────────────────────────────────────────
        self.chat.tag_configure("user_header",
            font=self.font_user, foreground=NAVY_DARK,
            background=CHAT_USER_BG, spacing1=8, spacing3=4,
            lmargin1=10, lmargin2=10, rmargin=10,
        )
        self.chat.tag_configure("user_body",
            font=self.font_input, foreground=TEXT_MID,
            background=CHAT_USER_BG, spacing3=8,
            lmargin1=10, lmargin2=10, rmargin=10,
        )
        self.chat.tag_configure("bot_header",
            font=self.font_user, foreground=NAVY_DARK,
            background=CHAT_BOT_BG, spacing1=8, spacing3=4,
            lmargin1=10, lmargin2=10, rmargin=10,
        )
        self.chat.tag_configure("bot_body",
            font=self.font_body, foreground=TEXT_DARK,
            background=CHAT_BOT_BG, spacing3=4,
            lmargin1=10, lmargin2=10, rmargin=10,
        )
        self.chat.tag_configure("section_header",
            font=tkfont.Font(family="Georgia", size=10, weight="bold"),
            foreground=NAVY_DARK, background=CHAT_BOT_BG,
            spacing1=6, spacing3=2,
            lmargin1=10, lmargin2=10,
        )
        self.chat.tag_configure("citation_header",
            font=tkfont.Font(family="Georgia", size=9, weight="bold"),
            foreground=GOLD_DARK, background="#f0ece0",
            spacing1=8, spacing3=2,
            lmargin1=10, lmargin2=10,
        )
        self.chat.tag_configure("citation_line",
            font=self.font_citation,
            foreground="#555", background="#f0ece0",
            spacing3=2,
            lmargin1=18, lmargin2=28, rmargin=10,
        )
        self.chat.tag_configure("divider",
            font=self.font_status, foreground=GOLD_DARK,
            background=CREAM, spacing1=4, spacing3=4,
            lmargin1=10,
        )
        self.chat.tag_configure("thinking",
            font=tkfont.Font(family="Courier New", size=10, slant="italic"),
            foreground=GOLD_DARK, background=CREAM,
            spacing1=10, spacing3=10, lmargin1=14,
        )
        self.chat.tag_configure("article_tag",
            font=tkfont.Font(family="Georgia", size=9, weight="bold"),
            foreground=CREAM, background=NAVY_MID,
            spacing1=6, spacing3=2,
            lmargin1=10, lmargin2=10,
        )
        self.chat.tag_configure("filter_note",
            font=self.font_status, foreground=GOLD_DARK,
            background=CREAM, spacing1=4, lmargin1=14,
        )
        self.chat.tag_configure("error",
            font=self.font_body, foreground="#8b1a1a",
            background="#fff0f0",
            lmargin1=10, lmargin2=10, spacing3=6,
        )

        # Show welcome message
        self._append_welcome()

    def _build_input_area(self):
        """Text entry + Send/Demo buttons at the bottom."""
        input_frame = tk.Frame(self.main_frame, bg=NAVY_MID, pady=8, padx=10)
        input_frame.pack(fill="x", pady=(0, 3))

        # Gold border on top
        tk.Frame(self.main_frame.master if False else input_frame,
                 bg=GOLD, height=1).pack(fill="x", side="top")

        # "You:" label
        tk.Label(
            input_frame, text="Ask a legal question:",
            font=tkfont.Font(family="Georgia", size=11, weight="bold"),
            fg=GOLD, bg=NAVY_MID,
        ).pack(anchor="w", padx=4, pady=(0, 4))

        # Input row
        row = tk.Frame(input_frame, bg=NAVY_MID)
        row.pack(fill="x")

        # Text entry (expands)
        entry_border = tk.Frame(row, bg=GOLD_DARK, bd=1, relief="flat")
        entry_border.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.input_var = tk.StringVar()
        self.entry = tk.Entry(
            entry_border,
            textvariable=self.input_var,
            font=self.font_input,
            bg=CREAM, fg=TEXT_DARK,
            relief="flat",
            bd=6,
            insertbackground=NAVY_DARK,
        )
        self.entry.pack(fill="x")
        self.entry.bind("<Return>",       lambda e: self._send())
        self.entry.bind("<KP_Enter>",     lambda e: self._send())
        self.entry.bind("<FocusIn>",      self._on_entry_focus)

        # Buttons
        self.send_btn = self._make_btn(row, "  Send  ▸", self._send, width=9)
        self.send_btn.pack(side="left", padx=(0, 6))

    def _build_disclaimer(self):
        """Bottom disclaimer strip."""
        disc = tk.Frame(self.main_frame, bg=NAVY_DARK, pady=4, padx=10)
        disc.pack(fill="x")

        tk.Frame(disc, bg=GOLD_DARK, height=1).pack(fill="x", pady=(0, 4))

        tk.Label(
            disc,
            text=DISCLAIMER,
            font=self.font_disclaimer,
            fg="#9ab0c8", bg=NAVY_DARK,
            wraplength=800, justify="center",
        ).pack()

    # ── Buttons ───────────────────────────────────────────────────────────────
    def _make_btn(self, parent, text, command, width=None):
        """Styled gold-on-navy button with hover effect."""
        kwargs = dict(
            text=text, command=command,
            font=self.font_btn,
            fg=NAVY_DARK, bg=GOLD,
            activeforeground=NAVY_DARK, activebackground=GOLD_LIGHT,
            relief="flat", bd=0,
            padx=10, pady=4,
            cursor="hand2",
        )
        if width:
            kwargs["width"] = width
        btn = tk.Button(parent, **kwargs)
        btn.bind("<Enter>", lambda e: btn.configure(bg=GOLD_LIGHT))
        btn.bind("<Leave>", lambda e: btn.configure(bg=GOLD))
        return btn

    # ── Welcome message ───────────────────────────────────────────────────────
    def _append_welcome(self):
        self._chat_insert("divider",
            "─" * 72 + "\n")
        self._chat_insert("bot_header",
            "  ⚓  COAST GUARD LEGAL ASSISTANT  —  Session Started\n")
        self._chat_insert("bot_body",
            "  Type a legal question and press Enter or click Send.\n"
            "  Use the Demo button to cycle through example questions.\n"
            "  All answers cite official USCG sources.\n")
        self._chat_insert("divider",
            "─" * 72 + "\n\n")

    # ── Resize handler ────────────────────────────────────────────────────────
    def _on_resize(self, event):
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w > 10 and h > 10:
            self._render_background(w, h)

    def _on_entry_focus(self, event):
        """Flash gold border on focus."""
        pass  # styling already applied via tag

    # ── Text insertion helpers ────────────────────────────────────────────────
    def _chat_insert(self, tag: str, text: str):
        """Append styled text to the chat widget (must be called from main thread)."""
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _chat_insert_many(self, segments: list[tuple[str, str]]):
        """Insert multiple (tag, text) segments at once."""
        self.chat.configure(state="normal")
        for tag, text in segments:
            self.chat.insert("end", text, tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    # ── System loading (background thread) ───────────────────────────────────
    def _start_loading(self):
        """Load the RAG system in a background thread so UI stays responsive."""
        t = threading.Thread(target=self._load_system, daemon=True)
        t.start()

    def _load_system(self):
        """Background: load embedding model, FAISS index, BM25, reranker."""
        def status(msg):
            self.root.after(0, lambda m=msg: self.status_var.set(m))

        try:
            status("⏳  Loading embedding model …")
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.args.model)

            status("⏳  Loading FAISS index …")
            self.index, self.meta = load_index(self.args.index_dir, self.args.index_name)

            if self.args.retriever == "hybrid" and _HYBRID_AVAILABLE:
                status("⏳  Loading BM25 index …")
                self.bm25_data = load_bm25_index(self.args.index_dir, self.args.index_name)

            if self.args.rerank and _RERANKER_AVAILABLE:
                status("⏳  Loading cross-encoder reranker …")
                self.reranker = Reranker()
                self.reranker._load()

            # Build retriever label
            if self.args.retriever == "hybrid" and self.bm25_data is not None:
                self.retriever_label = "Hybrid BM25+Dense"
            else:
                self.retriever_label = "Dense"
            if self.reranker:
                self.retriever_label += " + Cross-Encoder"

            n_chunks = len(self.meta)
            status("✓  System Ready")

            # Show ready message in chat
            self.root.after(0, self._show_ready)

        except FileNotFoundError as e:
            msg = (
                f"Index not found.\n"
                f"Run: python src/build_index.py\n\n{e}"
            )
            self.root.after(0, lambda: self._chat_insert("error", f"\n  ⚠  {msg}\n"))
            status("⚠  Index not found — see chat for details")
        except Exception as e:
            self.root.after(0, lambda: self._chat_insert("error", f"\n  ⚠  Load error: {e}\n"))
            status(f"⚠  Load error: {e}")

    def _show_ready(self):
        mode_str = "LLM Summary (AI-generated)" if self.llm_mode else "Retrieval-Only (verbatim sources)"
        self._chat_insert_many([
            ("bot_header", "  ✓  System Ready\n"),
            ("bot_body",   f"  Mode: {mode_str}\n"
                           "  Ask a question or click Demo ▸ to see an example.\n"),
            ("divider",    "─" * 72 + "\n\n"),
        ])
        self._update_mode_btn()
        self.entry.focus_set()

    # ── Mode toggle ───────────────────────────────────────────────────────────
    def _toggle_mode(self):
        if self.index is None:
            return  # not loaded yet
        if self.llm_mode:
            self.llm_mode = False
            self._chat_insert("filter_note", "  ⇄  Switched to Retrieval-Only mode.\n\n")
        else:
            if not self.args.llm:
                messagebox.showinfo(
                    "LLM Not Available",
                    "LLM mode requires starting the app with:\n\n"
                    "  --llm /path/to/mistral.gguf\n\n"
                    "Restart the application with that argument to enable summaries."
                )
                return
            self.llm_mode = True
            self._chat_insert("filter_note", "  ⇄  Switched to LLM Summary mode.\n\n")
        self._update_mode_btn()

    def _update_mode_btn(self):
        if self.llm_mode:
            self.mode_btn.configure(text="[  AI Summary Mode  ]")
        else:
            self.mode_btn.configure(text="[ Source View Mode ]")

    # ── Demo question ─────────────────────────────────────────────────────────
    def _insert_demo(self):
        q = DEMO_QUESTIONS[self.demo_idx % len(DEMO_QUESTIONS)]
        self.demo_idx += 1
        self.input_var.set(q)
        self.entry.icursor("end")
        self.entry.focus_set()

    # ── Clear chat ────────────────────────────────────────────────────────────
    def _clear_chat(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._append_welcome()
        if self.index is not None:
            self._show_ready()

    # ── Send query ────────────────────────────────────────────────────────────
    def _send(self):
        """Read input, display user message, dispatch retrieval in background."""
        if self.thinking:
            return  # block double-sends while processing

        query = self.input_var.get().strip()
        if not query:
            return
        if self.index is None:
            self._chat_insert("error", "  ⚠  System still loading. Please wait.\n\n")
            return

        self.input_var.set("")
        self.thinking = True
        self.send_btn.configure(state="disabled", text="  …  ")

        # Display user turn
        self._chat_insert_many([
            ("user_header", f"  You  ─────────────────────────────────────────\n"),
            ("user_body",   f"  {query}\n\n"),
        ])

        # Show thinking indicator
        self._thinking_id = self.root.after(0, self._show_thinking)

        # Retrieve in background thread
        t = threading.Thread(
            target=self._retrieve_and_respond, args=(query,), daemon=True
        )
        t.start()

    def _show_thinking(self):
        self._chat_insert("thinking", "  ⏳  Retrieving …\n\n")
        self._thinking_line = self.chat.index("end-3l")

    def _remove_thinking(self):
        """Remove the '⏳ Retrieving …' line."""
        try:
            self.chat.configure(state="normal")
            # Remove the last thinking line
            idx = self._thinking_line
            self.chat.delete(f"{idx} linestart", f"{idx} lineend+1c")
            # Also remove the blank line after it
            self.chat.delete(f"{idx} linestart", f"{idx} lineend+1c")
            self.chat.configure(state="disabled")
        except Exception:
            pass

    # ── Retrieval + display (background thread) ───────────────────────────────
    def _retrieve_and_respond(self, query: str):
        """Run retrieval in background; post result back to main thread."""
        try:
            article_filter = detect_article_filter(query) if _HYBRID_AVAILABLE else None
            top_k_fetch    = self.args.top_k * 3 if self.reranker else self.args.top_k

            if self.args.retriever == "hybrid" and self.bm25_data is not None:
                chunks, ret_mode = hybrid_retrieve(
                    query, self.index, self.meta, self.bm25_data,
                    top_k=top_k_fetch,
                    article_filter=article_filter,
                    model_name=self.args.model,
                )
            else:
                chunks, ret_mode = dense_retrieve(
                    query, self.index, self.meta, self.embed_model,
                    top_k=top_k_fetch,
                    article_filter=article_filter,
                )

            if self.reranker and chunks:
                chunks = self.reranker.rerank(query, chunks, top_k=self.args.top_k)
                reranked = True
            else:
                chunks = chunks[:self.args.top_k]
                reranked = False

            passages = aggregate_chunks(chunks)

            if self.llm_mode and self.args.llm and passages:
                llm_text = run_llm(query, passages, self.args.llm)
            else:
                llm_text = None

            # Post result back to UI thread
            self.root.after(0, lambda: self._display_response(
                query, passages, ret_mode, reranked, article_filter, llm_text
            ))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda: self._display_error(str(e), tb))

    # ── Response rendering ────────────────────────────────────────────────────
    def _display_response(
        self, query, passages, ret_mode, reranked,
        article_filter, llm_text
    ):
        self._remove_thinking()

        segments = []

        # Bot turn header
        segments.append(("bot_header",
            "  Legal Assistant  ──────────────────────────────────────────\n"))

        # Article filter note
        if article_filter:
            segments.append(("filter_note",
                f"  ▸ Article filter applied: Art. {article_filter.upper()}\n"))

        if not passages:
            segments.append(("bot_body",
                "  No relevant passages found. Try rephrasing or including an article number.\n\n"))
        elif llm_text:
            # LLM summary mode
            segments.append(("section_header", "  Summary\n"))
            for line in llm_text.split("\n"):
                segments.append(("bot_body", f"  {line}\n"))
            segments.append(("bot_body", "\n"))
            # Citations
            segments.append(("citation_header", "  Sources\n"))
            for i, p in enumerate(passages, 1):
                segments.append(("citation_line", f"  {format_citation(p, i)}\n"))
            segments.append(("bot_body", "\n"))
        else:
            # Retrieval-only mode
            for i, passage in enumerate(passages, 1):
                # Section label
                art   = passage.get("article_number", "")
                sec   = passage.get("section_number", "")
                hp    = passage.get("heading_path", [])
                title = (passage.get("article_title") or passage.get("section_title") or "").strip()
                hp_list = hp if isinstance(hp, list) else []

                if art:
                    label = f"Art. {art}" + (f"  —  {title}" if title else "")
                elif hp_list:
                    label = " › ".join(str(h) for h in hp_list[-3:])
                elif sec:
                    label = f"§ {sec}" + (f"  {title}" if title else "")
                else:
                    label = passage.get("source", "")[:60]

                segments.append(("article_tag", f"  [{i}]  {label}\n"))

                # Passage text (cleaned, wrapped)
                text = _norm_text(passage.get("text", ""))
                for para in text.split("\n\n"):
                    para = para.strip()
                    if para:
                        # Wrap long paragraphs
                        wrapped = textwrap.fill(para, width=90,
                                                initial_indent="  ",
                                                subsequent_indent="  ")
                        segments.append(("bot_body", wrapped + "\n"))
                segments.append(("bot_body", "\n"))

            # Citations block
            segments.append(("citation_header", "  Citations\n"))
            for i, p in enumerate(passages, 1):
                segments.append(("citation_line", f"  {format_citation(p, i)}\n"))
            segments.append(("bot_body", "\n"))

        # Divider
        segments.append(("divider", "─" * 72 + "\n\n"))

        self._chat_insert_many(segments)

        # Restore UI
        self.thinking = False
        self.send_btn.configure(state="normal", text="  Send  ▸")
        self.entry.focus_set()

    def _display_error(self, msg: str, tb: str):
        self._remove_thinking()
        self._chat_insert_many([
            ("error", f"\n  ⚠  Error: {msg}\n"),
            ("citation_line", tb + "\n"),
            ("divider", "─" * 72 + "\n\n"),
        ])
        self.thinking = False
        self.send_btn.configure(state="normal", text="  Send  ▸")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CG Legal RAG — GUI Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/chat_gui.py
  python src/chat_gui.py --retriever hybrid --rerank
  python src/chat_gui.py --llm "C:\\models\\mistral-7b-instruct.Q4_K_M.gguf"
  python src/chat_gui.py --retriever hybrid --rerank --llm "C:\\models\\mistral.gguf"

Place the background image at: assets/CG_Legal_RAG_GUI_Image.png
(or next to chat_gui.py)
        """,
    )
    parser.add_argument("--index-dir",  type=Path, default=INDEX_DIR)
    parser.add_argument("--index-name", type=str,  default=INDEX_NAME)
    parser.add_argument("--model",      type=str,  default=MODEL_NAME)
    parser.add_argument("--top-k",      type=int,  default=TOP_K)
    parser.add_argument("--retriever",  type=str,  default="hybrid",
                        choices=["dense", "hybrid"])
    parser.add_argument("--rerank",     action="store_true",
                        help="Apply cross-encoder reranking (recommended, ~1s extra)")
    parser.add_argument("--llm",        type=str,  default=None,
                        help="Path to .gguf model for LLM summaries")
    parser.add_argument("--mode",       type=str,  default="auto",
                        choices=["retrieval", "llm", "auto"])
    args = parser.parse_args()

    # Resolve starting mode
    if args.mode == "auto":
        args._llm_start = args.llm is not None
    else:
        args._llm_start = (args.mode == "llm" and args.llm is not None)

    root = tk.Tk()
    app  = CGLegalApp(root, args)

    # Set starting mode
    if args._llm_start:
        app.llm_mode = True

    root.mainloop()


if __name__ == "__main__":
    main()
