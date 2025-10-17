#!/usr/bin/env python3
"""
Context creation pipeline for Experiment 01.

This script parses source PDFs, generates multimodal summaries with the
available VLMs, and populates the required Qdrant collections
(text-only, multimodal, and ColPali-style late-interaction indexes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Iterable, List

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from transformers import AutoTokenizer
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import (
    ColFlor,
    ColFlorProcessor,
    ColIdefics3,
    ColIdefics3Processor,
    ColPaliForRetrieval,
    ColPaliProcessor,
    ColQwen2_5,
    ColQwen2_5_Processor,
)

from functions import (
    colpali_qdrant,
    convert_pdfs_to_images,
    pdf_loader,
    process_models,
    qdrant_process,
)


DEFAULT_MODELS = [
    {
        "model_name": "gpt-4o-2024-11-20",
        "model_short": "gpt-4o",
        "port": "9999",
        "text_vd": "RAG_TEXT",
        "mm_vd": "MM_04_GPT_4o",
        "late_inter": "vidore/colpali-v1.3-merged",
        "late_inter_short": "COL_PALI",
    },
    {
        "model_name": "gpt-4o-mini-2024-07-18",
        "model_short": "gpt-4o-mini",
        "port": "9999",
        "text_vd": "RAG_TEXT",
        "mm_vd": "MM_05_GPT_4o_mini",
        "late_inter": "vidore/colpali-v1.3-merged",
        "late_inter_short": "COL_PALI",
    },
    {
        "model_name": "google/gemma-3-27b-it",
        "model_short": "Gemma3-27b",
        "port": "8006",
        "text_vd": "RAG_TEXT",
        "mm_vd": "MM_07_GEMMA3_27B",
        "late_inter": "vidore/colpali-v1.3-merged",
        "late_inter_short": "COL_PALI",
    },
]

EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
EMB_DIM = 768


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the context creation pipeline."""
    parser = argparse.ArgumentParser(description="Create Qdrant context collections.")
    parser.add_argument(
        "--papers-dir",
        default=os.environ.get("PAPERS_DIR", "./papers"),
        help="Directory containing source PDFs.",
    )
    parser.add_argument(
        "--vd-dir",
        default=os.environ.get("VD_DIR"),
        help="Vector DB storage root (defaults to VD_DIR environment variable).",
    )
    parser.add_argument(
        "--prompts-path",
        default="prompts_used.pkl",
        help="Path to prompts pickle file used for image summarisation.",
    )
    parser.add_argument(
        "--models-config",
        default=None,
        help="Optional JSON file describing model configuration overrides.",
    )
    parser.add_argument(
        "--doi-file",
        default=None,
        help="Optional text file with one DOI (or URL) per line.",
    )
    parser.add_argument(
        "--huggingface-login",
        action="store_true",
        help="Login to HuggingFace Hub using the HUGGING_FACE_HUB_TOKEN env var.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant service URL.",
    )
    parser.add_argument(
        "--colpali-cache",
        default=None,
        help="Directory for storing page-level ColPali images (defaults to <vd-dir>/colpali_pages).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override for embeddings and retriever models.",
    )
    return parser.parse_args()


def resolve_device(device_override: str | None) -> str:
    """Resolve the torch device used for embedding models."""
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(config_path: str | None) -> List[dict]:
    """Load model configuration from JSON or return the default profile."""
    if config_path is None:
        return DEFAULT_MODELS
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def read_doi_file(path: str | None, papers: list[str]) -> list[str]:
    """Return DOI/URL list; produce placeholders when not provided."""
    if path is None:
        return [""] * len(papers)
    with open(path, "r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp if line.strip()]
    if len(lines) != len(papers):
        raise ValueError(
            f"DOI file contains {len(lines)} entries, but {len(papers)} PDFs were found."
        )
    return lines


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Ensure directories exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def login_huggingface() -> None:
    """Login to Hugging Face Hub if token is available."""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    login(token=token)


async def build_model_documents(processed_multi, prompts, models):
    """Wrapper so we can await the async helper from synchronous context."""
    return await process_models(processed_multi, prompts, models)


def load_retriever(model_name: str, device: str):
    """Instantiate the requested vision retriever and processor."""
    if model_name == "vidore/colpali-v1.3-merged":
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    elif model_name == "ahmed-masry/ColFlor":
        processor = ColFlorProcessor.from_pretrained(model_name)
        model = ColFlor.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    elif model_name == "vidore/colSmol-500M":
        processor = ColIdefics3Processor.from_pretrained(model_name)
        model = ColIdefics3.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    elif model_name == "vidore/colqwen2.5-v0.2":
        processor = ColQwen2_5_Processor.from_pretrained(model_name)
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    else:
        raise ValueError(f"Unsupported retriever model: {model_name}")
    return model, processor


def create_colpali_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create (if necessary) a dense collection for late-interaction vectors."""
    if client.collection_exists(collection_name=collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        on_disk_payload=True,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
            on_disk=True,
        ),
    )


def main() -> None:
    args = parse_args()
    load_dotenv()

    if args.vd_dir is None:
        raise ValueError("Vector storage directory must be provided (via --vd-dir or VD_DIR).")

    papers_dir = Path(args.papers_dir)
    if not papers_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {papers_dir}")

    prompts_path = Path(args.prompts_path)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    ensure_dirs([Path(args.vd_dir)])

    device = resolve_device(args.device)
    print(f"[setup] Using device: {device}")

    if args.huggingface_login:
        login_huggingface()

    papers = sorted(str(p) for p in papers_dir.glob("*.pdf"))
    if not papers:
        raise FileNotFoundError(f"No PDF files found in {papers_dir}")

    doi_links = read_doi_file(args.doi_file, papers)

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": device},
    )

    print(f"[ingest] Parsing PDFs from {papers_dir}")
    processed_multi, processed_text = pdf_loader(
        papers,
        doi_links,
        [Path(p).name for p in papers],
        args.vd_dir,
        tokenizer,
    )

    with prompts_path.open("rb") as fh:
        prompts = pickle.load(fh)

    models = load_models(args.models_config)
    print(f"[ingest] Generating multimodal summaries for {len(models)} models...")
    model_docs = asyncio.run(build_model_documents(processed_multi, prompts, models))
    model_docs["text_only"] = processed_text

    client = QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))

    text_collections_done = set()
    for model_cfg in models:
        text_collection = model_cfg["text_vd"]
        if text_collection not in text_collections_done:
            qdrant_process(
                model_docs["text_only"],
                client,
                text_collection,
                EMB_DIM,
                embeddings,
                url=args.qdrant_url,
            )
            text_collections_done.add(text_collection)

        mm_collection = model_cfg["mm_vd"]
        qdrant_process(
            model_docs[model_cfg["model_short"]],
            client,
            mm_collection,
            EMB_DIM,
            embeddings,
            url=args.qdrant_url,
        )

    # Late-interaction (ColPali) collections
    page_cache = Path(args.colpali_cache or Path(args.vd_dir) / "colpali_pages")
    ensure_dirs([page_cache])
    dataset = convert_pdfs_to_images(papers, str(page_cache))

    for model_cfg in models:
        col_collection = model_cfg["late_inter_short"]
        retriever_name = model_cfg["late_inter"]

        print(f"[colpali] Populating collection {col_collection} from {retriever_name}")
        retriever_model, retriever_processor = load_retriever(retriever_name, device)

        # Determine vector size dynamically
        sample_batch = retriever_processor.process_images([dataset[0]["image"]]).to(retriever_model.device)
        with torch.no_grad():
            sample_vectors = retriever_model(**sample_batch).embeddings
        vector_size = sample_vectors.shape[-1]

        create_colpali_collection(client, col_collection, vector_size)
        colpali_qdrant(
            dataset,
            papers,
            doi_links,
            retriever_model,
            retriever_processor,
            client,
            col_collection,
        )

    print("[done] Context creation finished successfully.")


if __name__ == "__main__":
    main()
