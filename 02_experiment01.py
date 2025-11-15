#!/usr/bin/env python3
"""
Experiment 01 evaluation pipeline.

This script aligns with the revamped 05_experiment02 flow:
    * Retrieve both text and image snippets from the configured vector store.
    * Build multimodal prompts that embed references inline.
    * Query either OpenAI (for GPT models) or a local vLLM endpoint.
    * Persist detailed pickle outputs compatible with downstream analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import random
from pathlib import Path
from time import time
from typing import Iterable, List

import aiohttp
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from transformers import ColPaliForRetrieval, ColPaliProcessor
from openai.lib._parsing._completions import type_to_response_format_param

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

from functions import (
    retrieve_colpali,
    response_real_out,
    encode_image_to_data_url,
    build_instruction_block,
    build_reference_from_metadata,
    document_to_context_entry,
)

load_dotenv()

BENCHMARK_PATH = Path("./data/Glycans_q_a_v5.xlsx")
DEFAULT_TOP_K = 5


class MCQ(BaseModel):
    """Schema enforcing a single-letter response."""

    answer: str = Field(
        description="Output is the answer of a MCQ with only one of the following categories: A, B, C or D."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 01 evaluation for a single model/RAG mode.")
    parser.add_argument("--vllm_port", type=int, required=True, help="Port of the vLLM server (ignored for GPT models).")
    parser.add_argument("--model_name", required=True, help="Model identifier.")
    parser.add_argument("--filepath_output", required=True, help="Prefix for the pickle output (timestamp appended).")
    parser.add_argument("--vector_db", default="", help="Qdrant collection used for retrieval (if applicable).")
    parser.add_argument(
        "--type",
        default="",
        choices=["", "mm_RAG", "colpali"],
        help="Retrieval type: '' (no RAG), 'mm_RAG', or 'colpali'.",
    )
    parser.add_argument(
        "--perm_quest",
        default="No",
        help="Set to 'Yes' to permute answer order per question.",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of retrieved context items.")
    parser.add_argument(
        "--qa_path",
        default=str(BENCHMARK_PATH),
        help="Path to the Glycans benchmark Excel file.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> pd.DataFrame:
    """Load and shuffle the benchmark rows."""
    return pd.read_excel(path).sample(frac=1).reset_index(drop=True)


def should_permute(flag: str) -> bool:
    return flag.lower() in {"yes", "true", "1"}


class RetrievalManager:
    """Handles retrieval for the different Experiment 01 RAG setups."""

    def __init__(self, retrieval_type: str, vector_db: str, top_k: int):
        self.retrieval_type = retrieval_type
        self.vector_db = vector_db
        self.top_k = top_k
        self.qdrant_client = None
        self.vector_store = None
        self.colpali_model = None
        self.colpali_processor = None

        if retrieval_type == "mm_RAG" and vector_db:
            self._init_qdrant_store(vector_db)
        elif retrieval_type == "colpali" and vector_db:
            self._init_colpali(vector_db)

    def _init_qdrant_store(self, vector_db: str) -> None:
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        api_key = os.environ.get("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(url=url, api_key=api_key)
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5", providers=["CUDAExecutionProvider"])
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=vector_db,
            embedding=embeddings,
        )

    def _init_colpali(self, vector_db: str) -> None:
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        api_key = os.environ.get("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(url=url, api_key=api_key)
        model_id = "vidore/colpali-v1.3-hf"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.colpali_model = ColPaliForRetrieval.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self.colpali_processor = ColPaliProcessor.from_pretrained(model_id)

    def fetch(self, query: str) -> list[dict]:
        if self.retrieval_type == "" or self.top_k <= 0:
            return []
        if self.retrieval_type == "mm_RAG" and self.vector_store is not None:
            docs = self.vector_store.similarity_search_with_score(query, self.top_k)
            return [document_to_context_entry(doc, score) for doc, score in docs]
        if self.retrieval_type == "colpali" and self.colpali_model is not None:
            result = retrieve_colpali(
                query,
                self.colpali_processor,
                self.colpali_model,
                self.qdrant_client,
                "",
                self.vector_db,
                self.top_k,
            )
            entries = []
            for point in result.points:
                payload = point.payload or {}
                metadata = payload.get("metadata", payload)
                entries.append(
                    {
                        "type": "image",
                        "text": "",
                        "image_path": metadata.get("img_link"),
                        "reference": build_reference_from_metadata(metadata),
                        "score": getattr(point, "score", None),
                    }
                )
            return entries
        return []


def build_messages(question: str, answers: list[str], contexts: list[dict]) -> tuple[list[dict], list[str]]:
    """Assemble the multimodal prompt and capture reference labels."""
    instruction = build_instruction_block(question, answers)
    content = [{"type": "text", "text": instruction}]
    references: list[str] = []

    for ctx in contexts:
        reference = ctx.get("reference", "context")
        if ctx.get("image_path") and ctx["type"] == "image":
            data_url = encode_image_to_data_url(ctx["image_path"])
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
                references.append(reference)
        if ctx.get("text"):
            snippet = ctx["text"].strip()
            if snippet:
                content.append({"type": "text", "text": f"[{reference}] {snippet}"})
                if reference not in references:
                    references.append(reference)

    return [{"role": "user", "content": content}], references


async def post_request_with_retries(session, url, headers, data, retries=4, backoff=1):
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=data, timeout=120) as response:
                if response.status == 200:
                    return await response.json()
                text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {text}")
        except Exception as exc:
            if attempt < retries - 1:
                await asyncio.sleep(backoff * (2**attempt))
            else:
                raise exc


async def run_inference(
    model_name: str,
    messages_list: list[list[dict]],
    url: str,
    headers: dict,
    use_schema: bool,
) -> list[dict]:
    connector = aiohttp.TCPConnector(limit=256)
    payloads = []
    for messages in messages_list:
        body = {"model": model_name, "messages": messages}
        if use_schema:
            body["response_format"] = type_to_response_format_param(MCQ)
        payloads.append(body)

    async with aiohttp.ClientSession(connector=connector) as session:
        responses = await asyncio.gather(
            *[post_request_with_retries(session, url, headers, data=body) for body in payloads]
        )
    return responses


def prepare_requests(
    qa_table: pd.DataFrame,
    retrieval: RetrievalManager,
    permute_answers: bool,
) -> tuple[list[list[dict]], list[dict]]:
    messages_list: list[list[dict]] = []
    records: list[dict] = []

    for _, row in qa_table.iterrows():
        answers = [row["A"], row["B"], row["C"], row["D"]]
        if permute_answers:
            perm_idx = random.sample(range(len(answers)), len(answers))
        else:
            perm_idx = list(range(len(answers)))
        shuffled_answers = [answers[i] for i in perm_idx]

        contexts = retrieval.fetch(row["question"])
        messages, context_refs = build_messages(row["question"], shuffled_answers, contexts)

        messages_list.append(messages)
        records.append(
            {
                "Question_nr": row["Question_nr"],
                "question": row["question"],
                "quest_order": perm_idx,
                "context_refs": context_refs,
            }
        )

    return messages_list, records


def extract_answer_text(response_payload: dict) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


async def main() -> None:
    args = parse_args()
    qa_table = load_questions(Path(args.qa_path))
    permute_answers = should_permute(args.perm_quest)
    retrieval = RetrievalManager(args.type, args.vector_db, args.top_k)

    messages_list, records = prepare_requests(qa_table, retrieval, permute_answers)

    if args.model_name.startswith("gpt"):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        use_schema = True
    else:
        url = f"http://localhost:{args.vllm_port}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}",
            "Content-Type": "application/json",
        }
        use_schema = False

    t_start = time()
    responses = await run_inference(args.model_name, messages_list, url, headers, use_schema)

    out_list = []
    for record, raw in zip(records, responses):
        parsed_answer = (
            raw.get("choices", [{}])[0].get("message", {}).get("content", "") if use_schema else extract_answer_text(raw)
        )
        filt_resp, answer_letter = response_real_out(parsed_answer, record["quest_order"])
        out_list.append(
            {
                **record,
                "answer": answer_letter,
                "resp_init": parsed_answer[:50],
                "filt_resp": filt_resp,
            }
        )

    timestamp = pd.Timestamp("now", tz="CET").strftime("%Y%m%d-%H%M%S")
    suffix = "_perm_q" if permute_answers else ""
    eval_results = {
        "model": args.model_name,
        "evaluation": sorted(out_list, key=lambda x: x["Question_nr"]),
        "elapsed_time": time() - t_start,
        "timestamp": timestamp,
        "permuted_answers": permute_answers,
    }

    output_path = Path(f"{args.filepath_output}_{timestamp}{suffix}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(eval_results, fh)

    print(f"Saved evaluation results to {output_path}")


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
