#!/usr/bin/env python3
"""
Script version of the 01_create_context_qdrant notebook.

The pipeline parses PDF papers, builds multimodal summaries with the configured
LLMs/VLMs, and pushes the outputs plus ColPali-style representations to Qdrant.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

# ColPali relies on custom torch.classes registries – clear stale paths in scripts.
torch.classes.__path__ = []  # type: ignore[attr-defined]

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
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

from functions import pdf_loader, process_models

OPEN_MODEL = "google/gemma-3-27b-it"
OPEN_MODEL_SHORT = "Gemma3-27b"
LI_MODEL = "vidore/colpali-v1.3-merged"
LI_MODEL_SHORT = "ColPali"
EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
EMB_DIM = 768
VECTOR_SIZE = 128

MODELS = [
    {
        "model_name": "gpt-4o-2024-11-20",
        "model_short": "gpt-4o",
        "port": "9999",
        "text_vd": "RAG_TEXT",
        "mm_vd": "MM_04_GPT_4o",
        "late_inter": LI_MODEL,
        "late_inter_short": LI_MODEL_SHORT,
    },
    {
        "model_name": "gpt-4o-mini-2024-07-18",
        "model_short": "gpt-4o-mini",
        "port": "9999",
        "text_vd": "RAG_TEXT",
        "mm_vd": "MM_05_GPT_4o_mini",
        "late_inter": LI_MODEL,
        "late_inter_short": LI_MODEL_SHORT,
    },
    {
        "model_name": OPEN_MODEL,
        "model_short": OPEN_MODEL_SHORT,
        "port": "8006",
        "text_vd": "RAG_TEXT",
        "mm_vd": "MM_07_GEMMA3_27B",
        "late_inter": LI_MODEL,
        "late_inter_short": LI_MODEL_SHORT,
    },
]

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

PROMPTS_PICKLE = Path("prompts_used.pkl")
DICT_OUT_PICKLE = Path("dict_out.pkl")


def ensure_env(name: str) -> str:
    """Fetch a required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set.")
    return value


def determine_device() -> str:
    """Prefer CUDA, fall back to MPS/CPU if unavailable."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_embedding_stack(device: str):
    """Initialise the BGE encoder used for LangChain documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    return embeddings, tokenizer


def login_and_cache_open_model() -> None:
    """Authenticate with HuggingFace Hub and prefetch the Gemma weights."""
    token = ensure_env("HUGGING_FACE_HUB_TOKEN")
    cache_root = Path(ensure_env("HF_DIR")).expanduser()
    cache_dir = cache_root / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    login(token=token)
    snapshot_download(repo_id=OPEN_MODEL, cache_dir=str(cache_dir))


def load_prompts(path: Path):
    """Load the prompt dictionary used for multimodal summarisation."""
    with path.open("rb") as handle:
        return pickle.load(handle)


async def run_model_processing(processed_multi, prompts):
    """Mirror the notebook-level await call."""
    outputs = await process_models(processed_multi, prompts, MODELS)
    return outputs


def align_dois(papers: Sequence[Path]) -> list[str]:
    """Ensure we have one DOI/URL per paper."""
    if len(DEFAULT_DOIS) == len(papers):
        return list(DEFAULT_DOIS)
    if not DEFAULT_DOIS:
        return [""] * len(papers)

    print(
        "Warning: DOI list length does not match number of papers. "
        "Excess entries are dropped; missing entries are padded with blanks."
    )
    dois = list(DEFAULT_DOIS[: len(papers)])
    if len(dois) < len(papers):
        dois.extend([""] * (len(papers) - len(dois)))
    return dois


def list_papers(papers_dir: Path) -> list[Path]:
    """Collect PDF files from PAPERS_DIR."""
    return sorted([path for path in papers_dir.iterdir() if path.suffix.lower() == ".pdf"])


def chunk_points(points: Sequence[qmodels.PointStruct], batch_size: int) -> Iterable[Sequence[qmodels.PointStruct]]:
    """Yield Qdrant points in fixed-size batches for upsert calls."""
    for start in range(0, len(points), batch_size):
        yield points[start : start + batch_size]


def get_ret_model_processor(model_id: str, device: str):
    """Instantiate the requested vision retriever."""
    attn_impl = "flash_attention_2" if is_flash_attn_2_available() else None
    if model_id == "vidore/colpali-v1.3-merged":
        processor = ColPaliProcessor.from_pretrained(model_id)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = ColPaliForRetrieval.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
        ).eval()
    elif model_id == "ahmed-masry/ColFlor":
        processor = ColFlorProcessor.from_pretrained(model_id)
        model = ColFlor.from_pretrained(
            model_id,
            device_map=device,
            attn_implementation=attn_impl,
        ).eval()
    elif model_id == "vidore/colSmol-500M":
        processor = ColIdefics3Processor.from_pretrained(model_id)
        model = ColIdefics3.from_pretrained(
            model_id,
            device_map=device,
            attn_implementation=attn_impl,
        ).eval()
    elif model_id == "ibm-granite/granite-vision-3.3-2b-embedding":
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(
            model_id,
            device_map=device,
            attn_implementation=attn_impl,
        ).eval()
    elif model_id == "vidore/colqwen2.5-v0.2":
        processor = ColQwen2_5_Processor.from_pretrained(model_id)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = ColQwen2_5.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
        ).eval()
    else:
        raise ValueError(f"Unsupported late-interaction model: {model_id}")
    return model, processor


def convert_pdf_dir_to_images(pdf_dir: Path):
    """Load every PDF as a sequence of PIL images."""
    mapping: dict[str, list] = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        mapping[pdf_path.name] = convert_from_path(str(pdf_path))
    return mapping


def to_tensor(output: Any) -> torch.Tensor:
    """Convert a HF model output into a plain tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        return to_tensor(output[0])
    for attr in ("last_hidden_state", "pooler_output", "embeddings"):
        if hasattr(output, attr):
            return getattr(output, attr)
    raise TypeError(f"Unsupported model output type: {type(output)}")


def create_document_embeddings(
    pdf_dir: Path,
    model,
    processor,
    image_root: Path,
    batch_size: int = 4,
) -> list[qmodels.PointStruct]:
    """Produce Qdrant-ready points for every PDF page."""
    all_points: list[qmodels.PointStruct] = []
    documents = convert_pdf_dir_to_images(pdf_dir)

    # Some retriever models expose .device, others require reading from parameters.
    try:
        model_device = model.device  # type: ignore[attr-defined]
    except AttributeError:
        model_device = next(model.parameters()).device  # type: ignore[attr-defined]

    for doc_id, (filename, images) in enumerate(documents.items()):
        dataloader = DataLoader(
            dataset=images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: processor.process_images(batch),
        )
        page_counter = 1
        for batch in tqdm(dataloader, desc=f"Processing {filename}"):
            with torch.no_grad():
                inputs = {key: value.to(model_device) for key, value in batch.items()}
                tensor = to_tensor(model(**inputs))
                cpu_embeddings = list(torch.unbind(tensor.to("cpu")))

            for embedding in cpu_embeddings:
                img_link = image_root / f"{Path(filename).stem}_{page_counter:03d}.png"
                all_points.append(
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={
                            "metadata": {
                                "doc_id": doc_id,
                                "page_id": page_counter,
                                "file_name": filename,
                                "img_link": str(img_link),
                            }
                        },
                    )
                )
                page_counter += 1

    return all_points


def ensure_collection(client: QdrantClient, name: str, size: int, distance=qmodels.Distance.COSINE):
    """Create a dense Qdrant collection if it does not already exist."""
    if client.collection_exists(collection_name=name):
        return
    client.create_collection(
        collection_name=name,
        on_disk_payload=True,
        vectors_config=qmodels.VectorParams(size=size, distance=distance, on_disk=True),
    )


def ensure_multivector_collection(client: QdrantClient, name: str):
    """Create the multi-vector late interaction collection if required."""
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
                comparator=qmodels.MultiVectorComparator.MAX_SIM
            ),
        ),
    )


def main() -> None:
    load_dotenv()
    device = determine_device()
    embeddings, emb_tokenizer = build_embedding_stack(device)
    login_and_cache_open_model()

    papers_dir = Path(os.environ.get("PAPERS_DIR", "./papers")).resolve()
    papers = list_papers(papers_dir)
    if not papers:
        raise RuntimeError(f"No PDF files found in {papers_dir}")

    doi_links = align_dois(papers)
    vd_dir = ensure_env("VD_DIR")
    processed_multi, processed_text = pdf_loader(
        [str(p) for p in papers],
        doi_links,
        [p.name for p in papers],
        vd_dir,
        emb_tokenizer,
    )

    prompts = load_prompts(PROMPTS_PICKLE)
    dict_outputs = asyncio.run(run_model_processing(processed_multi, prompts))
    dict_outputs["text_only"] = processed_text
    with DICT_OUT_PICKLE.open("wb") as handle:
        pickle.dump(dict_outputs, handle)

    qdrant_api_key = ensure_env("QDRANT_API_KEY")
    qdrant_url = os.environ.get("QDRANT_URL", f"http://localhost:{os.environ.get('QDRANT_PORT', '6333')}")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    text_collection_upserted = False
    image_root = Path(vd_dir) / "pg_images"

    for idx, model_cfg in enumerate(MODELS):
        if idx == 0 and not text_collection_upserted:
            ensure_collection(qdrant_client, model_cfg["text_vd"], EMB_DIM)
            QdrantVectorStore.from_documents(
                documents=dict_outputs["text_only"],
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=model_cfg["text_vd"],
                embedding=embeddings,
            )
            text_collection_upserted = True

        ensure_collection(qdrant_client, model_cfg["mm_vd"], EMB_DIM)
        QdrantVectorStore.from_documents(
            documents=dict_outputs[model_cfg["model_short"]],
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=model_cfg["mm_vd"],
            embedding=embeddings,
        )

        ensure_multivector_collection(qdrant_client, model_cfg["late_inter_short"])
        model_li, processor_li = get_ret_model_processor(model_cfg["late_inter"], device)
        points = create_document_embeddings(papers_dir, model_li, processor_li, image_root, batch_size=4)
        for batch in chunk_points(points, batch_size=10):
            qdrant_client.upsert(collection_name=model_cfg["late_inter_short"], points=batch)


if __name__ == "__main__":
    main()
