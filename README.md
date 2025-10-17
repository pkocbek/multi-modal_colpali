# Multi-Modal ColPali Experiments

End-to-end workflows for building multi-modal vector stores and evaluating large
vision-language models on the Glycan multiple-choice benchmark. The repository
contains two experiment tracks:

- **Experiment 01** – compares no-RAG, text-RAG, multimodal RAG, and ColPali
  retrieval across multiple generators.
- **Experiment 02** – benchmarks GPT-5 model family variants against several
  late-interaction visual retrievers (ColPali / ColFlor / ColQwen).

The instructions below aim for full reproducibility and make it easy to port the
setup to new domains.

---

## 1. Repository Layout & Required Assets

```
multi-modal_colpali/
├── 00_run_docker_containers.sh            # Infrastructure bootstrap
├── 01_create_context_qdrant.py            # Context creation pipeline
├── 02_experiment01.py                     # Async evaluation driver
├── 03_experiment01_run.py                 # Batch runner (permuted / non-permuted)
├── 04_experiment01_eval.py                # Experiment 01 summariser
├── 05_experiment02.py                     # Experiment 02 evaluator
├── 06_experiment02_eval.py                # Experiment 02 summariser
├── functions.py                           # Shared helpers (Docling, Qdrant, ColPali)
├── prompts_used.pkl                       # Prompt templates for summarisation/eval
├── data/Glycans_q_a_v5.xlsx               # MCQ benchmark (place under ./data)
├── papers/                                # Source PDFs (PAPERS_DIR)
└── results/                               # Evaluation artefacts
```

- Keep `.env_sample` as a reference. Populate your own `.env` alongside it.
- Leave `benchmark_placeholder.csv` untouched; it is a scaffold for future data.
- Ensure `prompts_used.pkl`, the benchmark Excel file, and all PDFs exist before
  running any pipeline.

---

## 2. Environment & Credentials

1. **Create a Python environment**
   ```bash
   pyenv install 3.12.3   # or your preferred 3.12 build
   pyenv virtualenv 3.12.3 mm-colpali
   pyenv local mm-colpali
   pip install --upgrade pip
   pip install poetry
   poetry install
   ```
   GPU users should confirm CUDA/cuDNN are available (Flash Attention 2 is
   leveraged when present).

2. **Populate `.env`**

   | Key | Purpose |
   | --- | --- |
   | `OPENAI_API_KEY` | Required for hosted GPT-* models |
   | `VLLM_API_KEY` | Token used by local VLLM server |
   | `QDRANT_API_KEY` | Secures the Qdrant instance |
   | `DOCLING_API_KEY` | Optional: Docling SaaS |
   | `HUGGING_FACE_HUB_TOKEN` | Needed for downloading private/multi-modal models |
   | `HF_DIR` | Hugging Face cache directory (e.g. `/mnt/cache/hf/`) |
   | `VD_DIR` | Persistent storage root for Qdrant (mounted in Docker) |
   | `PAPERS_DIR` (optional) | Defaults to `./papers` if unset |

   Use `.env_sample` as a template.

3. **Load environment variables**
   ```bash
   source .env
   ```

4. **(Optional) Hugging Face login**
   ```
   huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
   ```

---

## 3. Infrastructure Bootstrap

1. **Grant Docker permissions without `sudo`**
   ```bash
   newgrp docker
   ```

2. **Launch services**
   ```bash
   ./00_run_docker_containers.sh
   ```

   - Qdrant exposes `http://localhost:6333`.
   - VLLM (Gemma 3 27B) is available via `http://localhost:8006/v1`.
   - Qdrant persists data under `${VD_DIR}` (`storage/`, `custom_config.yaml`).

3. **Health checks**
   - Visit `http://localhost:6333/dashboard` (if enabled) or use the REST API.
   - `curl http://localhost:8006/v1/models` to confirm the VLLM endpoint.

---

## 4. Context Creation (Experiment 01)

1. **Prepare PDFs**
   - Place domain-specific PDFs inside `${PAPERS_DIR}` (defaults to `./papers`).
   - Optional: provide a line-delimited DOI/URL file to align metadata.

2. **Generate context collections**
   ```bash
   python 01_create_context_qdrant.py \
     --huggingface-login \
     --models-config configs/models_exp01.json  # optional override
   ```

   Key actions:
   - Parses PDFs with Docling, storing text, tables, and images into `${VD_DIR}`.
   - Summarises images for each configured model using prompts from `prompts_used.pkl`.
   - Uploads documents to Qdrant:
     - Text-only collection (shared).
     - Multimodal collections (one per generator).
     - ColPali late-interaction pages stored as dense vectors.

3. **Adapting to new domains**
   - Swap PDFs, update DOIs as needed, and re-run `01_create_context_qdrant.py`.
   - Clear or relocate `${VD_DIR}`/collection names to avoid clashes.

---

## 5. Experiment 01 Evaluation Workflow

### 5.1 Single Run Driver

```bash
python 02_experiment01.py \
  --vllm_port 8006 \
  --model_name "google/gemma-3-27b-it" \
  --filepath_output "./results/eval/eval_gemma_no_RAG_benchmark" \
  --vector_db "" \
  --type "" \
  --perm_quest yes
```

- `--type ""` disables retrieval (no-RAG).
- `--type mm_RAG` + `--vector_db <collection>` for text/multimodal runs.
- `--type colpali` targets late-interaction collections; the script automatically
  permutes answers when `--perm_quest` evaluates truthy.

### 5.2 Batch Runner

Use the orchestrator for 5 repeats per collection, with and without permutation:

```bash
python 03_experiment01_run.py \
  --vllm_port 8006 \
  --model_name "google/gemma-3-27b-it" \
  --model_name_short "Gemma3-27b" \
  --vd_mm_name "MM_07_GEMMA3_27B" \
  --vd_colpali_name "COL_PALI" \
  --vd_text_name "RAG_TEXT" \
  --repeats 5
```

Outputs land in `results/eval/`, following the pattern:
`eval_{model_short}_{retrieval}_{perm|no_perm}_benchmark_{timestamp}[_perm_q].pkl`.

### 5.3 Aggregation

```bash
python 04_experiment01_eval.py \
  --eval-dir results/eval \
  --benchmark-path data/Glycans_q_a_v5.xlsx \
  --summary-path results/eval_results.xlsx \
  --majority-path results/eval_maj_results.xlsx \
  --full-path results/eval_full_results.xlsx
```

This Produces:
- Accuracy by difficulty & retrieval (summary sheet).
- Majority-vote accuracy across repeats.
- Full merged dataset for downstream analysis.

---

## 6. Experiment 02 Workflow

1. **Run evaluations**
   ```bash
   python 05_experiment02.py \
     --qa_path data/Glycans_q_a_v5.xlsx \
     --pdf_dir papers \
     --results_dir results/evals \
     --models gpt-5 gpt-5-mini gpt-5-nano \
     --retrievers vidore/colpali-v1.3-merged ahmed-masry/ColFlor vidore/colqwen2.5-v0.2 \
     --iterations 5 \
     --context
   ```

   - Embedding caches are stored under `data/` to speed up re-runs.
   - `--context` toggles retrieval-augmented prompting; omit for no-context runs.

2. **Summarise results**
   ```bash
   python 06_experiment02_eval.py \
     --results_dir results/evals \
     --output results/summary.xlsx \
     --models gpt-5 gpt-5-mini gpt-5-nano \
     --retrievers vidore/colpali-v1.3-merged vidore/colqwen2.5-v0.2 ahmed-masry/ColFlor
   ```

   Generates an Excel workbook with per-difficulty metrics and raw evaluations.

---

## 7. Utility Module (`functions.py`)

Key helpers centralised for reuse:

- **Docling ingest & chunking**: `doc_conv`, `data_preparation`, `pdf_loader`.
- **Image handling**: `resize_image`, `convert_pdfs_to_images`.
- **Evaluation scaffolding**: `format_msgs`, `response_real_out`, async HTTP utilities.
- **Vector DB operations**: `qdrant_process`, `colpali_qdrant`, `retrieve_colpali`.
- **Experiment 02 support**: `convert_pdf_dir_to_images`, `create_document_embeddings`.

Each helper carries succinct docstrings describing inputs, outputs, and side
effects. Extend them to support new modalities (e.g., audio, video) when needed.

---

## 8. Reproducibility Checklist

1. `source .env`
2. `newgrp docker`
3. `./00_run_docker_containers.sh`
4. `python 01_create_context_qdrant.py`
5. `python 03_experiment01_run.py ...` (or ad-hoc runs with `02_experiment01.py`)
6. `python 04_experiment01_eval.py`
7. `python 05_experiment02.py ...` (optional)
8. `python 06_experiment02_eval.py` (optional)

Verify key artefacts:
- Qdrant collections appear via its REST API.
- `results/eval/*.pkl` and Excel reports exist.
- `results/evals/*.csv` and `results/summary.xlsx` exist for Experiment 02.

To adapt the pipeline to another domain:
- Replace PDFs in `${PAPERS_DIR}` and update DOIs (if available).
- Provide a new MCQ benchmark Excel sheet with the same column schema.
- Update Qdrant collection names if you need to keep multiple domains in the
  same instance.

---

## 9. Troubleshooting

- **Missing CUDA / Flash Attention**: adjust `--device cpu` or install the
  necessary GPU drivers.
- **Hugging Face auth errors**: ensure `HUGGING_FACE_HUB_TOKEN` is valid and you
  ran `huggingface-cli login`.
- **Permission errors writing to `${VD_DIR}`**: confirm the path exists and is
  writable by Docker (bind-mount parent directory if required).
- **Qdrant already contains documents**: delete the collection via the REST API
  or choose new collection names before re-running ingestion.

---

Happy experimenting!
