# Offline / VPN Installation Guide

Use this guide when installing on a network that restricts PyPI access,
such as a CGA VPN or air-gapped government network.

---

## Strategy

Download all packages on an unrestricted machine first, transfer to the
target machine, then install from local files. No internet required after
the initial download step.

---

## Step 1 — Download all wheels (on unrestricted machine)

```bash
mkdir pio-rag-wheels
cd pio-rag-wheels

# Download everything pio-rag needs, including all transitive dependencies
pip download -r /path/to/pio-rag/requirements.txt \
    --platform win_amd64 \
    --python-version 3.12 \
    --only-binary=:all: \
    -d .
```

**If the target machine is not Windows**, change `--platform`:
| Target OS | Flag |
|---|---|
| Windows 64-bit | `--platform win_amd64` |
| macOS Apple Silicon | `--platform macosx_14_0_arm64` |
| macOS Intel | `--platform macosx_12_0_x86_64` |
| Linux 64-bit | `--platform manylinux2014_x86_64` |

This will download approximately **1.5–2 GB** total.
The largest single package is PyTorch (~800 MB), which sentence-transformers
pulls in as a dependency.

---

## Step 2 — Transfer wheels

Copy the `pio-rag-wheels/` folder to the target machine via USB drive,
shared drive, or whatever transfer method is available.

---

## Step 3 — Install from local wheels (on target machine)

```bash
conda create -n pio-rag python=3.12 -y
conda activate pio-rag

# Install everything from the local folder — no internet needed
pip install \
    --no-index \
    --find-links /path/to/pio-rag-wheels \
    -r requirements.txt
```

---

## Step 4 — LLM model (Mistral 7B)

The Mistral GGUF model file is not installed via pip — it is a single
large file (~4 GB) that you download separately and place anywhere on disk.

**Download on unrestricted machine:**
- Model: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- Source: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

**Transfer** the `.gguf` file to the target machine and note the full path.

**Install llama-cpp-python offline:**
```bash
# Download the pre-built wheel on an unrestricted machine:
pip download llama-cpp-python \
    --platform win_amd64 \
    --python-version 3.12 \
    --only-binary=:all: \
    -d pio-rag-wheels/

# Install on target:
pip install \
    --no-index \
    --find-links /path/to/pio-rag-wheels \
    llama-cpp-python
```

> **Note:** If a pre-built wheel is not available for your platform,
> `llama-cpp-python` requires a C++ compiler to build from source.
> On Windows this means Visual Studio Build Tools. On Mac/Linux,
> `xcode-select --install` or `gcc` respectively.

---

## Step 5 — Embedding model and reranker

`sentence-transformers` downloads model weights from HuggingFace on first use.
On a restricted network this will fail silently or time out.

**Pre-download on unrestricted machine:**

```python
# Run this script once on a machine with internet access:
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

This saves the model files to your HuggingFace cache:
- Windows: `C:\Users\<you>\.cache\huggingface\`
- Mac/Linux: `~/.cache/huggingface/`

**Transfer** the entire `~/.cache/huggingface/` folder to the same path
on the target machine.

**Tell the system not to go online:**
```bash
# Set this environment variable before running any pio-rag script:
set HF_DATASETS_OFFLINE=1      # Windows
export HF_DATASETS_OFFLINE=1   # Mac/Linux

# Or set it permanently in the conda environment:
conda activate pio-rag
conda env config vars set HF_DATASETS_OFFLINE=1
conda env config vars set TRANSFORMERS_OFFLINE=1
```

---

## Verification

After installation, verify everything is working:

```bash
conda activate pio-rag

# Check core imports
python -c "
import faiss
import numpy
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from PIL import Image
print('All core imports OK')
"

# Check embedding model loads offline
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
v = m.encode(['test'], normalize_embeddings=True)
print(f'Embedding model OK — vector shape: {v.shape}')
"

# Check reranker loads offline
python -c "
from sentence_transformers import CrossEncoder
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
score = ce.predict([('test query', 'test passage')])
print(f'Reranker OK — score: {score[0]:.3f}')
"

# Check index exists and loads
python -c "
import faiss, json
idx = faiss.read_index('data/index/pio_rag.faiss')
meta = json.load(open('data/index/pio_rag_meta.json'))
print(f'Index OK — {idx.ntotal} vectors, {len(meta)} metadata entries')
"

# Launch GUI (no LLM)
python src/chat_gui.py --retriever hybrid --rerank
```

---

## Package size reference

| Package | Approximate size |
|---|---|
| torch (via sentence-transformers) | ~800 MB |
| sentence-transformers | ~50 MB |
| transformers | ~100 MB |
| faiss-cpu | ~50 MB |
| all-MiniLM-L6-v2 model weights | ~90 MB |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | ~90 MB |
| Mistral 7B Q4_K_M GGUF | ~4.1 GB |
| All other packages combined | ~200 MB |
| **Total (without Mistral)** | **~1.4 GB** |
| **Total (with Mistral)** | **~5.5 GB** |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip download` fails for faiss-cpu | Try `--only-binary=:all:` without a platform flag first |
| llama-cpp-python wheel not found | Download from https://github.com/abetlen/llama-cpp-python/releases |
| Model not found on target machine | Verify HuggingFace cache path matches exactly |
| `TRANSFORMERS_OFFLINE=1` still tries internet | Also set `HF_HUB_OFFLINE=1` |
| tkinter missing | On Linux: `sudo apt install python3-tk`. On Windows/Mac it ships with Python. |
