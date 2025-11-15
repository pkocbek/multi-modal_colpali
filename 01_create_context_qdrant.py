#!/usr/bin/env python3
"""
Experiment 01 context creation pipeline.

Steps:
1. Parse PDFs under PAPERS_DIR using Docling via pdf_loader.
2. Generate multimodal summaries for each configured model using prompts.
3. Upsert text/multimodal documents into Qdrant collections.
4. Build ColPali-style late-interaction collections (page-level embeddings).
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
from huggingface_hub import login, snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from transformers import AutoTokenizer, AutoProcessor, AutoModel
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
    pdf_loader,
    process_models,
    qdrant_process,
    convert_pdfs_to_images,
    colpali_qdrant,
)

load_dotenv()

DEFAULT_MODELS = [
    {"model_name": "gpt-4o-2024-11-20", "model_short": "gpt-4o", "port": "9999", "text_vd": "RAG_TEXT", "mm_vd": "MM_04_GPT_4o", "late_inter": "vidore/colpali-v1.3-merged", "late_inter_short": "COL_PALI"},
    {"model_name": "gpt-4o-mini-2024-07-18", "model_short": "gpt-4o-mini", "port": "9999", "text_vd": "RAG_TEXT", "mm_vd": "MM_05_GPT_4o_mini", "late_inter": "vidore/colpali-v1.3-merged", "late_inter_short": "COL_PALI"},
    {"model_name": "google/gemma-3-27b-it", "model_short": "Gemma3-27b", "port": "8006", "text_vd": "RAG_TEXT", "mm_vd": "MM_07_GEMMA3_27B", "late_inter": "vidore/colpali-v1.3-merged", "late_inter_short": "COL_PALI"},
]

DEFAULT_PROMPT = (
    "You are an expert in glycan biology and you will be querried. Here is the query: {query}\n"
    "Task:\n"
    "Answer clearly and concisely. You will be given Context information, which can be empty.\n"
    "Tone: scientific and concise. Include critical numeric data, significant results, and relevant keywords if relevant.\n"
    "Constraints:\n"
    "Avoid generic answers.\n"
    "Here is the Context information:\n"
)

EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
EMB_DIM = 768
VECTOR_SIZE = 128
#add doi manually!!! or empty list same length as papers
#doi_papers = ["" for _ in papers]
DEFAULT_DOIS = [
    "https://doi.org/10.1038/s41590-024-01916-8",
    "https://doi.org/10.1186/s12967-018-1695-0",
    "https://doi.org/10.1097/hjh.0000000000002963",
    "https://doi.org/10.1186/s12967-018-1616-2",
    "https://doi.org/10.3390%2Fbiom13020375",
    "https://doi.org/10.1016/j.bbagen.2017.06.020",
    "https://doi.org/10.1172%2Fjci.insight.89703",
    "https://doi.org/10.1016%2Fj.isci.2022.103897",
    "https://doi.org/10.1002%2Fart.39273",
    "https://doi.org/10.1016/j.bbadis.2018.03.018",
    "https://doi.org/10.1097/MIB.0000000000000372",
    "https://doi.org/10.1053%2Fj.gastro.2018.01.002",
    "https://doi.org/10.1186/s13075-017-1389-7",
    "https://doi.org/10.1021/pr400589m",
    "https://doi.org/10.1161/CIRCRESAHA.117.312174",
    "https://doi.org/10.2337/dc22-0833",
    "https://doi.org/10.1097%2FMD.0000000000003379",
    "https://doi.org/10.1158/1078-0432.CCR-15-1867",
    "https://doi.org/10.1093/gerona/glt190",
    "https://doi.org/10.1111/imr.13407",
    "https://doi.org/10.1053/j.gastro.2018.05.030",
    "https://doi.org/10.1016/j.csbj.2024.03.008",
    "https://doi.org/10.1016/j.cellimm.2018.07.009",
    "https://doi.org/10.1016/j.biotechadv.2023.108169",
    "https://doi.org/10.4049/jimmunol.2400447",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Experiment 01 context collections.")
    parser.add_argument("--papers-dir", default=os.environ.get("PAPERS_DIR", "./papers"), help="Directory containing source PDFs.")
    parser.add_argument("--vd-dir", default=os.environ.get("VD_DIR"), help="Vector DB storage directory.")
    parser.add_argument("--prompts-path", default="prompts_used.pkl", help="Pickled prompts used for image summarisation.")
    parser.add_argument("--models-config", default=None, help="Optional JSON file describing model configuration overrides.")
    parser.add_argument("--doi-file", default=None, help="Optional text file with one DOI/URL per line.")
    parser.add_argument("--huggingface-login", action="store_true", help="Login to HuggingFace Hub using the env token.")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"), help="Qdrant service URL.")
    parser.add_argument("--device", default=None, help="Override the device used for retrievers (default: auto).")
    return parser.parse_args()


def resolve_device(device_override: str | None) -> str:
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_papers(papers_dir: Path) -> list[Path]:
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
    return sorted([path for path in papers_dir.iterdir() if path.suffix.lower() == ".pdf"])


def read_doi_file(path: str | None, num_papers: int) -> list[str]:
    if path is None:
        return DEFAULT_DOIS[:num_papers] if len(DEFAULT_DOIS) >= num_papers else [""] * num_papers
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if len(lines) != num_papers:
        raise ValueError(f"DOI file contains {len(lines)} entries, but {num_papers} PDFs were found.")
    return lines


def load_prompts(path: str | None):
    if path and Path(path).exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return DEFAULT_PROMPT


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def login_hf_if_needed(do_login: bool) -> None:
    if not do_login:
        return
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HUGGING_FACE_HUB_TOKEN not set.")
    login(token=token)
    snapshot_download(repo_id=DEFAULT_MODELS[-1]["model_name"], cache_dir=os.environ["HF_DIR"] + "hub/")


def load_retriever(model_name: str, device: str):
    if model_name == "vidore/colpali-v1.3-merged":
        processor = ColPaliProcessor.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=dtype,
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
    elif model_name == "ibm-granite/granite-vision-3.3-2b-embedding":
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    elif model_name == "vidore/colqwen2.5-v0.2":
        processor = ColQwen2_5_Processor.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    else:
        raise ValueError(f"Unsupported late-interaction model: {model_name}")
    return model, processor


def ensure_colpali_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(collection_name=name):
        return
    client.create_collection(
        collection_name=name,
        on_disk_payload=True,
        vectors_config=qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
            on_disk=True,
            multivector_config=qmodels.MultiVectorConfig(
                comparator=qmodels.MultiVectorComparator.MAX_SIM,
            ),
        ),
    )


def load_models(config_path: str | None) -> List[dict]:
    if config_path is None:
        return DEFAULT_MODELS
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


async def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    login_hf_if_needed(args.huggingface_login)

    papers_dir = Path(args.papers_dir)
    vd_dir = Path(args.vd_dir or os.environ.get("VD_DIR", "./src/vectordb"))
    ensure_dirs([vd_dir])

    papers = list_papers(papers_dir)
    doi_links = read_doi_file(args.doi_file, len(papers))
    prompts = load_prompts(args.prompts_path)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": device if device != "mps" else "cpu"},
    )
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    processed_multi, processed_text = pdf_loader(
        papers=[str(p) for p in papers],
        doi_links=doi_links,
        filenames=[p.name for p in papers],
        vd_dir=str(vd_dir),
        vd_tokenizer=tokenizer,
    )

    models_cfg = load_models(args.models_config)
    model_outputs = await process_models(processed_multi, prompts, models_cfg)
    model_outputs["text_only"] = processed_text

    qdrant_client = QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))

    # Upsert text and multimodal documents
    text_loaded = False
    for model_cfg in models_cfg:
        if not text_loaded:
            qdrant_process(
                docs=model_outputs["text_only"],
                qdrant_client=qdrant_client,
                vec_db=model_cfg["text_vd"],
                emb_dim=EMB_DIM,
                embeddings=embeddings,
                url=args.qdrant_url,
            )
            text_loaded = True

        qdrant_process(
            docs=model_outputs[model_cfg["model_short"]],
            qdrant_client=qdrant_client,
            vec_db=model_cfg["mm_vd"],
            emb_dim=EMB_DIM,
            embeddings=embeddings,
            url=args.qdrant_url,
        )

    # ColPali collections
    page_cache = vd_dir / "pg_images"
    dataset = convert_pdfs_to_images([str(p) for p in papers], str(page_cache))

    for model_cfg in models_cfg:
        ensure_colpali_collection(qdrant_client, model_cfg["late_inter_short"])
        retriever_model, retriever_processor = load_retriever(model_cfg["late_inter"], device)
        colpali_qdrant(
            dataset,
            [str(p) for p in papers],
            doi_links,
            retriever_model,
            retriever_processor,
            qdrant_client,
            model_cfg["late_inter_short"],
        )

    print("[done] Context creation completed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" in str(exc):
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise
