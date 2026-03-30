# 🎬 CineRAG — Multimodal Hierarchical RAG for Movie Trailer QA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)]()
[![RAG](https://img.shields.io/badge/Approach-RAG-orange.svg)]()
[![Multimodal](https://img.shields.io/badge/Multimodal-Text+Vision+Audio-purple.svg)]()

---
## 👥 Authors

- Maria Goicoechea Elio
- Joaquín Orradre Berdusán
- Paula Pina Alcántara

## 🧠 What is CineRAG?

CineRAG is a **multimodal Retrieval-Augmented Generation (RAG) system** for answering questions about **movie trailers**.

It combines (typical setup):
- 🎤 **Whisper** → speech-to-text (ASR)
- 🖼️ **BLIP** (or similar) → visual scene captions
- 🔍 **Vector DB (e.g., Qdrant)** → semantic search with hierarchical retrieval
- 💬 **Local LLM (e.g., LLaMA GGUF)** → answer generation

### 🔥 Key idea: Hierarchical Retrieval

Instead of a flat index, CineRAG uses:
1. **🎬 Trailer-level summary** → global context
2. **🎞️ Temporal windows** → scene-level retrieval
3. **✂️ Segments** → precise answers

---

## ⚙️ Run with Docker

```bash
git clone https://github.com/MariaGoico/NLP_Final_Project.git
cd NLP_Final_Project

docker compose build
docker compose up
# or:
docker compose up --build
```

Then open:
- **http://localhost:8501**

---

## 🎬 How it works (high-level)

1. Select or add a trailer (e.g., YouTube link)
2. Click **“Process trailer”**
3. Pipeline typically runs:
   - Download (e.g., `yt-dlp` + `ffmpeg`)
   - Transcription (Whisper)
   - Frame extraction
   - Captioning (BLIP)
   - Indexing (vector DB)
4. Ask questions in natural language 💬

### Example queries
- *“What is this movie about?”*
- *“What does the main character say?”*
- *“Are there any comedic moments?”*

👉 The system returns (depending on configuration):
- ✅ Generated answer
- ⏱️ Supporting timestamps

---

## 📊 Benchmark

Evaluate the system using the built-in benchmark:

```bash
pip install requests sentence-transformers numpy
python benchmark/run_benchmark.py
python benchmark/evaluate.py
```

✔ Generates:
- Quantitative metrics
- HTML report

Metrics (may include):
- Keyword Hit Rate (KHR)
- Semantic Similarity
- Level Accuracy
- Trap Accuracy (hallucination control)

---

## 🖥️ Requirements

- Docker + Docker Compose
- (Optional) GPU recommended for faster Whisper/vision/LLM inference

---
