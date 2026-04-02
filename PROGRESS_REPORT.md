# Progress report

**Project:** m3docrag (M3-DocVQA multimodal RAG pipeline). **Last updated:** 2026-04-02.

This file is the living **progress report** for the repo: it combines a **technical walkthrough** (ColPali page embeddings, FAISS, Weaviate) with **current integration status**, **limitations**, and **evaluation metrics**. It replaces the former `COLPALI_PAGE_EMBEDDINGS_M3DOCVQA.md`.

**Contents:** §§1–6 — indexing and storage; §7 — MoE router + Qwen2-VL + low-memory FAISS path; §8 — run commands; §9 — limitations; §10 — latest dev eval numbers; §11 — changelog + ready-to-run commands.

---

## Dual indexing at a glance

- **Visual index**: ColPali page embeddings stored in a **FAISS** index.
- **Text and keyword index**: BGE-small embeddings + BM25 in **Weaviate**.

---

## 1. Data (input)

- Repo root: `/home/user/m3docrag`
- Dataset root: `/home/user/m3docrag/m3-docvqa`

### Visual inputs (PDFs)

The ColPali embedding script loads PDFs from:

- `/home/user/m3docrag/m3-docvqa/splits/pdfs_dev/*.pdf`  (`--split=dev`).

Each file is named by `doc_id`:

- For example: `.../pdfs_dev/<doc_id>.pdf`

### Text inputs (MMQA wiki passages)

The Weaviate indexing script reads:

- `/home/user/m3docrag/m3-docvqa/multimodalqa/MMQA_texts.jsonl`

Each line is JSON with fields like:

- `id` (this is the **doc_id** key will retrieve later)
- `title`, `url`, `text`

---

## 2. Environment setup

### a. Activate the venv

```bash
cd /home/user/m3docrag
source /home/user/m3docrag/m3docvqa/.venv/bin/activate
```

### b. Export local path variables

These are used by multiple scripts to find the dataset, embeddings, and models. If they are missing, the code may fall back to `/job/...` and crash.

```bash
export LOCAL_DATA_DIR=/home/user/m3docrag
export LOCAL_EMBEDDINGS_DIR=/home/user/m3docrag/embeddings
export LOCAL_MODEL_DIR=/home/user/m3docrag/models
```

---

## 3. Visual index (ColPali): PDF to images to embeddings (`.safetensors`)

### Output of this process:

- One file per doc:
  - `embeddings/colpali_m3docvqa_dev/<doc_id>.safetensors`
- Inside each file: tensor `embeddings` with shape:
[[n_pages, n_tokens,128]]

### File runs

- **Runner script**: `examples/run_page_embedding.py`
- **Dataset / PDF to image**: `src/m3docrag/datasets/m3_docvqa/dataset.py`
  - PDF pages are converted to images via `m3docrag.utils.pdfs.get_images_from_pdf(...)` (uses `pdf2image`)
- **Embedding model wrapper**: `src/m3docrag/retrieval/colpali.py` via `ColPaliRetrievalModel.encode_images(...)`

### Command to run (resume-safe)

This job can be killed and resumed because the script skips documents whose `.safetensors` already exists.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python examples/run_page_embedding.py \
  --use_retrieval \
  --retrieval_model_type=colpali \
  --data_name=m3-docvqa \
  --split=dev \
  --loop_unique_doc_ids=True \
  --output_dir="/home/user/m3docrag/embeddings/colpali_m3docvqa_dev" \
  --retrieval_model_name_or_path=colpaligemma-3b-pt-448-base \
  --retrieval_adapter_model_name_or_path=colpali-v1.2 \
  --per_device_eval_batch_size=1
```

### Flow

For each `doc_id` in `dev_doc_ids.json`:

1. Load `.../pdfs_dev/<doc_id>.pdf`
2. Convert **every PDF page** into a **PIL image**
3. Feed images into ColPali → page embeddings
4. Stack into `[n_pages, n_tokens, 128]`
5. Save to `embeddings/colpali_m3docvqa_dev/<doc_id>.safetensors`

### Progress checking

Count completed docs:

```bash
ls -1 /home/user/m3docrag/embeddings/colpali_m3docvqa_dev/*.safetensors 2>/dev/null | wc -l
```

---

## 4. Visual index (FAISS): `.safetensors` → `index.bin`

### output of this process

- FAISS index file:
  - `embeddings/faiss_colpali_m3docvqa_dev/index.bin`

This FAISS index stores the **flattened token vectors** (not just one vector per document).

### Which code runs

- **Runner script**: `examples/run_indexing_m3docvqa.py`
- It loads embeddings using:
  - `src/m3docrag/datasets/m3_docvqa/dataset.py` to `load_all_embeddings()`
- Then it flattens page/token vectors and builds:
  - `IndexIVFFlat` (default in args) over `d=128`

### Command

```bash
python examples/run_indexing_m3docvqa.py \
  --data_name=m3-docvqa \
  --split=dev \
  --retrieval_model_type=colpali \
  --embedding_name=colpali_m3docvqa_dev \
  --faiss_index_type=ivfflat \
  --output_dir="/home/user/m3docrag/embeddings/faiss_colpali_m3docvqa_dev"
```

### How to verify it worked

```bash
ls -lh /home/user/m3docrag/embeddings/faiss_colpali_m3docvqa_dev/index.bin
```

---

## 5. Text + keyword index (Weaviate): `MMQA_texts.jsonl` to collection `MmqTextChunk`

This stage gives **two retrieval modes** from the same stored objects:

- **Dense text retrieval**: BGE-small vectors (manual vectors)
- **Keyword retrieval**: BM25 over the `text` field

### Which code runs

- **Docker**: `docker-compose.weaviate.yml`
- **Indexing script**: `examples/run_weaviate_index_mmqa.py`
- **Weaviate helpers**: `src/m3docrag/retrieval/weaviate_mmqa.py`
  - `chunk_text(...)` splits each wiki passage into chunks (default ~1200 chars)
  - The script encodes each chunk with `sentence-transformers`:
    - `BAAI/bge-small-en-v1.5`
  - Then inserts each chunk as a Weaviate object with:
    - `doc_id`, `chunk_id`, `title`, `url`, `text`
    - plus a vector for dense search

### Step 5.1 Start Weaviate (Docker)

This repo maps ports to avoid collisions:

- HTTP: host `8090` → container `8080`
- gRPC: host `50052` → container `50051`

Start:

```bash
cd /home/user/m3docrag
docker compose -f docker-compose.weaviate.yml up -d
```

Set env vars (so Python can connect):

```bash
export WEAVIATE_HTTP_HOST=localhost
export WEAVIATE_HTTP_PORT=8090
export WEAVIATE_GRPC_PORT=50052
```

Ready check:

```bash
curl -s http://localhost:8090/v1/.well-known/ready
```

### Step 5.2. Install deps (if needed)

```bash
python -m pip install -U weaviate-client sentence-transformers
```

### Step 5.3. Run indexing

#### Smoke test (fast)

```bash
python examples/run_weaviate_index_mmqa.py \
  --texts_path /home/user/m3docrag/m3-docvqa/multimodalqa/MMQA_texts.jsonl \
  --max_docs 50 \
  --recreate
```

#### Full index (real run)

```bash
python examples/run_weaviate_index_mmqa.py \
  --texts_path /home/user/m3docrag/m3-docvqa/multimodalqa/MMQA_texts.jsonl \
  --collection MmqTextChunk \
  --embed_model BAAI/bge-small-en-v1.5 \
  --recreate
```

### How to verify Weaviate has the data

Count objects in the collection:

```bash
curl -sS "http://localhost:8090/v1/graphql" \
  -H 'Content-Type: application/json' \
  -d '{"query":"{ Aggregate { MmqTextChunk { meta { count } } } }"}'
```

---

## 6. What is stored

### Visual (ColPali)

- **Per-doc embeddings**: `embeddings/colpali_m3docvqa_dev/<doc_id>.safetensors`
- **FAISS index**: `embeddings/faiss_colpali_m3docvqa_dev/index.bin`

### Text + keyword (Weaviate)

- **Weaviate collection**: `MmqTextChunk`
  - **BM25** over `text`
  - **BGE-small vectors** stored per chunk (manual vectors you uploaded)

---

## 7. Router + generation integration (current status)

This repository now includes an MoE-style routing path in the RAG flow:

- Router experts: `visual`, `text`, `keyword`
- Router implementation: `src/m3docrag/routing/moe_router.py`
- Plan builder: `src/m3docrag/routing/rag_plan.py`
- Main execution script: `examples/run_rag_m3docvqa.py`

### How routing currently works

- `visual` route is triggered by structural visual cues (e.g., `page`, `figure`, `table`).
- `text` and `keyword` routes can call Weaviate to get top document IDs, then pass those IDs as filters before ColPali retrieval.
- Router diagnostics are saved in output JSON:
  - `router_expert`
  - `router_reason`
  - `router_features`
  - `router_doc_ids_filter`

### Generation model

- VQA generation uses `Qwen2-VL-7B-Instruct` via `src/m3docrag/vqa/qwen2.py`.
- Low-memory path is enabled with:
  - `--bits=4` (bitsandbytes)
  - retrieval model moved to CPU during VQA stage
  - reduced visual token budget and shorter max generation tokens (to reduce VRAM pressure)

### Low-memory retrieval (FAISS without full token matrix)

- When a FAISS index exists, the RAG path can avoid materializing the full flattened `all_token_embeddings` tensor.
- It builds `token2pageuid` from safetensor **shapes** only and ranks pages using **FAISS inner-product scores** (`src/m3docrag/rag/base.py`).
- This is what makes **full dev** eval feasible on a single GPU without huge CPU RAM for the embedding matrix.

---

## 8. End-to-end run command (current working baseline)

```bash
cd /home/user/m3docrag
source /home/user/m3docrag/m3docvqa/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_DATA_DIR=/home/user/m3docrag
export LOCAL_EMBEDDINGS_DIR=/home/user/m3docrag/embeddings
export LOCAL_MODEL_DIR=/home/user/m3docrag/models
export WEAVIATE_HTTP_HOST=127.0.0.1

python examples/run_rag_m3docvqa.py \
  --output_dir=./outputs \
  --data_name=m3-docvqa \
  --embedding_name=colpali_m3docvqa_dev \
  --faiss_index_type=ivfflat \
  --data_len=2 \
  --use_weaviate_router \
  --n_retrieval_pages=1 \
  --bits=4 \
  --model_name_or_path=Qwen2-VL-7B-Instruct
```

For a **full dev** eval, omit `--data_len` (and expect long runtime). Partial `--data_len` runs still evaluate against the full gold set unless you change eval settings, so metrics will look artificially bad.

---

## 9. Current limitations (for now)

- **GPU OOM risk is reduced but not fully eliminated.**
  - Large/long runs can still exceed VRAM depending on page/image complexity.
- **Startup is slow.**
  - Loading FAISS and document embeddings can take significant time before first predictions.
- **Router is rule-based.**
  - Good for controllability and debugging, but not perfect for all query types.
- **Weaviate availability is required for text/keyword narrowing.**
  - If Weaviate is down/misconfigured, routed filtering falls back/weakens.
- **Throughput is not optimized yet.**
  - Current setup prioritizes stability on limited GPUs, not max speed.

---

## 10. Latest evaluation metrics (M3DocVQA dev)

**Stack:** ColPali v1.2, FAISS `ivfflat`, `n_retrieval_pages=1`, Qwen2-VL-7B-Instruct (4-bit in the recorded run), `--use_weaviate_router`, full dev split (~2441 questions).

**Output files**

- Predictions: `outputs/colpali-v1.2_ivfflat_ret1_Qwen2-VL-7B-Instruct_2026-04-01_22-23-21.json`
- Summary JSON: `outputs/colpali-v1.2_ivfflat_ret1_Qwen2-VL-7B-Instruct_2026-04-01_22-23-21_eval_results.json`

### Overall and retrieval

| Metric | Value |
|--------|------:|
| list EM | **21.63%** |
| list F1 | **25.38%** |
| Average recall @1 | **49.99%** |
| Average recall @2, 4, 5, 10 | **49.99%** (same as @1 in this configuration) |

### By modality

| Modality | n | list EM | list F1 |
|----------|--:|--------:|--------:|
| image | 533 | 15.76% | 17.45% |
| table | 860 | 17.79% | 21.17% |
| text | 1048 | 27.77% | 32.85% |

### By hop type

| Type | n | EM | F1 |
|------|--:|---:|---:|
| Multi-hop | 980 | 16.73% | 18.85% |
| Single-hop | 1461 | 24.91% | 29.75% |

### By question type (from eval console; counts sum to dev size)

| Question type | n | EM (%) | F1 (%) |
|---------------|--:|-------:|-------:|
| Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ)) | 15 | 0.0 | 0.8 |
| Compare(Compose(TableQ,ImageQ),TableQ) | 104 | 38.46 | 41.03 |
| Compare(TableQ,Compose(TableQ,TextQ)) | 64 | 45.31 | 48.41 |
| Compose(ImageQ,TableQ) | 142 | 16.20 | 16.20 |
| Compose(ImageQ,TextQ) | 20 | 10.0 | 10.0 |
| Compose(TableQ,ImageListQ) | 195 | 5.64 | 7.33 |
| Compose(TableQ,TextQ) | 82 | 7.32 | 8.45 |
| Compose(TextQ,ImageListQ) | 46 | 15.22 | 17.30 |
| Compose(TextQ,TableQ) | 216 | 15.74 | 18.06 |
| ImageListQ | 141 | 5.67 | 10.09 |
| ImageQ | 230 | 22.17 | 23.39 |
| Intersect(ImageListQ,TableQ) | 44 | 2.27 | 8.02 |
| Intersect(ImageListQ,TextQ) | 3 | 33.33 | 33.33 |
| Intersect(TableQ,TextQ) | 49 | 20.41 | 27.02 |
| TableQ | 369 | 23.58 | 27.46 |
| TextQ | 721 | 30.24 | 36.80 |

**Eval JSON caveat:** `overall`, `modalities`, and `average_recall_at_k` in `_eval_results.json` match the tables above. Per-`q_type` (and similarly structured) entries in that file may duplicate a single F1 value across keys; for question-type splits, prefer the **console breakdown** (or fix the exporter) as source of truth.

### Additional run (ret3 / 2026-04-02_01-24-43)

**Stack:** ColPali v1.2, FAISS `ivfflat`, `n_retrieval_pages=3`, Qwen2-VL-7B-Instruct (4-bit in the recorded run), `--use_weaviate_router`, full dev split (~2441 questions).

**Output files**

- Predictions: `outputs/colpali-v1.2_ivfflat_ret3_Qwen2-VL-7B-Instruct_2026-04-02_01-24-43.json`
- Summary JSON: `outputs/colpali-v1.2_ivfflat_ret3_Qwen2-VL-7B-Instruct_2026-04-02_01-24-43_eval_results.json`

### Overall and retrieval

| Metric | Value |
|--------|------:|
| list EM | **24.78%** |
| list F1 | **29.35%** |
| Average recall @1 | **54.79%** |
| Average recall @2 | **62.60%** |
| Average recall @4, 5, 10 | **66.50%** |

### By modality

| Modality | n | list EM | list F1 |
|----------|--:|--------:|--------:|
| image | 533 | 17.26% | 20.28% |
| table | 860 | 19.77% | 24.33% |
| text | 1048 | 32.73% | 38.08% |

### By hop type

| Type | n | EM | F1 |
|------|--:|---:|---:|
| Multi-hop | 980 | 17.86% | 20.99% |
| Single-hop | 1461 | 29.43% | 34.95% |

### By question type (from eval console; counts sum to dev size)

| Question type | n | EM (%) | F1 (%) |
|---------------|--:|-------:|-------:|
| Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ)) | 15 | 0.00 | 0.60 |
| Compare(Compose(TableQ,ImageQ),TableQ) | 104 | 36.54 | 39.03 |
| Compare(TableQ,Compose(TableQ,TextQ)) | 64 | 43.75 | 50.30 |
| Compose(ImageQ,TableQ) | 142 | 21.13 | 21.13 |
| Compose(ImageQ,TextQ) | 20 | 20.00 | 22.50 |
| Compose(TableQ,ImageListQ) | 195 | 7.69 | 10.36 |
| Compose(TableQ,TextQ) | 82 | 8.54 | 9.66 |
| Compose(TextQ,ImageListQ) | 46 | 8.70 | 15.07 |
| Compose(TextQ,TableQ) | 216 | 15.74 | 19.61 |
| ImageListQ | 141 | 4.96 | 11.86 |
| ImageQ | 230 | 22.17 | 24.72 |
| Intersect(ImageListQ,TableQ) | 44 | 2.27 | 8.98 |
| Intersect(ImageListQ,TextQ) | 3 | 33.33 | 33.33 |
| Intersect(TableQ,TextQ) | 49 | 26.53 | 32.63 |
| TableQ | 369 | 28.18 | 33.68 |
| TextQ | 721 | 37.17 | 43.39 |

**Eval JSON caveat:** `overall`, `modalities`, and `average_recall_at_k` in `_eval_results.json` match the tables above. Per-`q_type` (and similarly structured) entries in that file may duplicate a single F1 value across keys; for question-type splits, prefer the **console breakdown** (or fix the exporter) as source of truth.

---

## 11. Document changelog (insert-only; newest first)

### 2026-04-02 — FAISS MaxSim neighbor count + IVF `nprobe` + retrieval defaults

- **`src/m3docrag/rag/base.py`:** FAISS `index.search` now uses **`nn_k`** token neighbors per query token (default `max(64, n_retrieval_pages × 16)`, never less than `n_retrieval_pages`). The final list is still the top **`n_retrieval_pages` pages**. Previously `k` for search equaled page count, which starved MaxSim.
- **`src/m3docrag/utils/args.py`:** New **`--faiss_search_k`** (optional; overrides auto `nn_k`). New **`--faiss_nprobe`** (default **16**; IVF query probes, ignored for `flatip`). **`--n_retrieval_pages`** default is now **3** (was 1); use **`--n_retrieval_pages=1`** for tight VRAM smoke tests.
- **`examples/run_rag_m3docvqa.py`:** After `faiss.read_index`, sets **`index.nprobe`** when the loaded index exposes it and `faiss_index_type` is not `flatip`.

### 2026-04-02 — Ready-to-run commands (when paths and artifacts exist)

Prerequisites: `embeddings/colpali_m3docvqa_dev/*.safetensors`, `embeddings/faiss_colpali_m3docvqa_dev/index.bin` (or `embeddings/colpali_m3docvqa_dev_pageindex_ivfflat/index.bin`), Qwen2-VL + ColPali weights under `LOCAL_MODEL_DIR`, venv activated. For `--use_weaviate_router`, Weaviate running (e.g. `docker compose -f docker-compose.weaviate.yml up -d`, HTTP **8090**, gRPC **50052**).

**Smoke (2 questions, 1 page, router on):**

```bash
cd /home/user/m3docrag
source /home/user/m3docrag/m3docvqa/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_DATA_DIR=/home/user/m3docrag
export LOCAL_EMBEDDINGS_DIR=/home/user/m3docrag/embeddings
export LOCAL_MODEL_DIR=/home/user/m3docrag/models
export WEAVIATE_HTTP_HOST=127.0.0.1
export WEAVIATE_HTTP_PORT=8090
export WEAVIATE_GRPC_PORT=50052

python examples/run_rag_m3docvqa.py \
  --output_dir=./outputs \
  --data_name=m3-docvqa \
  --embedding_name=colpali_m3docvqa_dev \
  --faiss_index_type=ivfflat \
  --data_len=2 \
  --use_weaviate_router \
  --n_retrieval_pages=1 \
  --faiss_nprobe=16 \
  --bits=4 \
  --model_name_or_path=Qwen2-VL-7B-Instruct
```

**Full dev eval (long; 3 pages + router; no `data_len`):**

```bash
cd /home/user/m3docrag
source /home/user/m3docrag/m3docvqa/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_DATA_DIR=/home/user/m3docrag
export LOCAL_EMBEDDINGS_DIR=/home/user/m3docrag/embeddings
export LOCAL_MODEL_DIR=/home/user/m3docrag/models
export WEAVIATE_HTTP_HOST=127.0.0.1
export WEAVIATE_HTTP_PORT=8090
export WEAVIATE_GRPC_PORT=50052

python examples/run_rag_m3docvqa.py \
  --output_dir=./outputs \
  --data_name=m3-docvqa \
  --embedding_name=colpali_m3docvqa_dev \
  --faiss_index_type=ivfflat \
  --use_weaviate_router \
  --n_retrieval_pages=3 \
  --faiss_nprobe=16 \
  --bits=4 \
  --model_name_or_path=Qwen2-VL-7B-Instruct
```

**Weaviate-off ablation:** remove `--use_weaviate_router`; Weaviate env vars optional.
