#!/usr/bin/env python3

"""
Experiment 02: async evaluation of GPT models with vision retrievers.
Ported from the original notebook to a script for easier automation.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import pickle
from io import BytesIO
from pathlib import Path
from time import gmtime, strftime
from typing import Iterable, Literal

import backoff
import nest_asyncio
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import (
    ColFlor,
    ColFlorProcessor,
    ColIdefics3,
    ColIdefics3Processor,
    ColPali,
    ColPaliProcessor,
    ColQwen2_5,
    ColQwen2_5_Processor,
)

from functions import convert_pdf_dir_to_images, create_document_embeddings


nest_asyncio.apply()


class MCQ(BaseModel):
    answer: Literal["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Experiment 02 runs."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 02 multi-modal evaluations."
    )
    parser.add_argument(
        "--qa_path",
        default="./data/Glycans_q_a_v5.xlsx",
        help="Path to the benchmark MCQ spreadsheet.",
    )
    parser.add_argument(
        "--pdf_dir",
        default="papers_merge",
        help="Directory containing source PDFs for retrieval.",
    )
    parser.add_argument(
        "--results_dir",
        default="results/evals",
        help="Directory to store CSV evaluation outputs.",
    )
    parser.add_argument(
        "--cache_dir",
        default="data",
        help="Directory used to cache pre-computed embeddings.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        help="List of generator models to evaluate.",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=[
            "vidore/colpali-v1.3-merged",
            "ahmed-masry/ColFlor",
            "vidore/colqwen2.5-v0.2",
        ],
        help="List of vision retrievers to use.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of evaluation repeats per model/retriever pair.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieved pages per query.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=10,
        help="Batch size for retrieval queries.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device override (default: auto-detect).",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="If set, enable retrieval-augmented prompting.",
    )
    return parser.parse_args()


def auto_device(device_override: str | None) -> str:
    """Resolve the torch device, defaulting to CUDA → MPS → CPU."""
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resize_base64_image(base64_string: str, fixed_width: int = 1024) -> str:
    """Resize an image encoded as Base64 to a target width."""
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))

    height = int(fixed_width / (img.width / img.height))
    resized = img.resize((fixed_width, height), resample=Image.LANCZOS)

    buffer = BytesIO()
    resized.save(buffer, format=img.format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_context_messages(context_images: list) -> list[dict]:
    """Build vision message payloads for chat completions."""
    messages = [{"type": "text", "text": "Context information:"}]
    for image in context_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        resized = resize_base64_image(img_str)
        messages.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{resized}"}}
        )
    return messages


def batched(iterable: list[str], n: int) -> Iterable[list[str]]:
    """Yield chunks of length n from a list."""
    for idx in range(0, len(iterable), n):
        yield iterable[idx : idx + n]


def build_query_prompts(table_qa: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Create retrieval and generation prompts for the MCQ table."""
    prompts_for_retrieval = []
    prompts_for_model = []
    for _, row in table_qa.iterrows():
        answers = [row["A"], row["B"], row["C"], row["D"]]
        resp_labels = ["A", "B", "C", "D"]
        question_string = "".join(
            f"{label}. {option}" for label, option in zip(resp_labels, answers)
        )
        base_prompt = (
            f"You are an experienced senior researcher tasked with providing in-depth analysis.\n"
            f"Use all the information at your disposal, such as uploaded files and other sources. "
            f"Think about the following statement or question: {row['question']}\n"
            f"Below are the possible answers, where letters mark each answer. "
            f"First, exclude the unlikely answer or answers, rethink, and select an output from the rest. "
            f"The output is only ONE letter from the list {resp_labels}. "
            f"Check that you return only one letter; if two letters, choose one. No explanations. The answers are:\n"
            f"{question_string}"
        )
        prompts_for_model.append(base_prompt)
        prompts_for_retrieval.append(f"{row['question']} The answers are: {question_string}")
    return prompts_for_retrieval, prompts_for_model


def score_results(
    queries: list[str],
    processor,
    model,
    dataset: list[dict],
    images_per_pdf: dict,
    top_k: int,
) -> list[list[dict]]:
    """Retrieve top-k pages per query with late-interaction scoring."""
    query_embeddings = processor.process_queries(queries).to(model.device)
    with torch.no_grad():
        query_outputs = model(**query_embeddings)

    document_embeddings = torch.stack([entry["embedding"] for entry in dataset])
    scores = processor.score_multi_vector(query_outputs, document_embeddings)

    retrieved = []
    for query_scores in scores:
        score_vals = query_scores.tolist()
        top_indices = query_scores.topk(top_k).indices.tolist()
        results = []
        for idx in top_indices:
            entry = dataset[idx]
            file_name = entry["file_name"]
            page_id = entry["page_id"]
            image = images_per_pdf[file_name][page_id]
            results.append(
                {
                    "doc_id": entry["doc_id"],
                    "page_id": page_id,
                    "file_name": file_name,
                    "image": image,
                    "score": score_vals[idx],
                }
            )
        retrieved.append(results)
    return retrieved


@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
async def get_completion_with_backoff(
    client: AsyncOpenAI, gpt_model: str, prompt_messages: list[dict]
) -> dict:
    """Query the OpenAI API with retry/backoff and parse MCQ responses."""
    completion = await client.beta.chat.completions.parse(
        model=gpt_model,
        messages=prompt_messages,
        response_format=MCQ,
    )
    return json.loads(completion.choices[0].message.content)


async def send_to_model_async(
    gpt_model: str,
    qa_table: pd.DataFrame,
    enable_context: bool,
    topk: int,
    chunk: int,
    processor,
    model,
    dataset: list[dict],
    images_per_pdf: dict,
) -> tuple[list[str], list[list[str]]]:
    """Run the evaluation loop asynchronously for a single model/retriever pair."""
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retrieval_prompts, model_prompts = build_query_prompts(qa_table)

    retrieved_results: list[list[dict]] = []
    info_res: list[list[str]] = []
    prompt_messages: list[list[dict]] = []

    if enable_context:
        for batch in batched(retrieval_prompts, chunk):
            retrieved_results.extend(
                score_results(batch, processor, model, dataset, images_per_pdf, topk)
            )
        for base_prompt, retrieved in zip(model_prompts, retrieved_results):
            info_res.append(
                [f"{entry['file_name'].split('.')[0]}_pg_{entry['page_id']}" for entry in retrieved]
            )
            context_images = [entry["image"] for entry in retrieved]
            prompt_messages.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": base_prompt}]
                        + image_context_messages(context_images),
                    }
                ]
            )
    else:
        info_res = [[] for _ in model_prompts]
        prompt_messages = [
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for prompt in model_prompts
        ]

    tasks = [get_completion_with_backoff(client, gpt_model, msg) for msg in prompt_messages]
    completions = await asyncio.gather(*tasks)
    answers = [completion["answer"] for completion in completions]

    return answers, info_res


def send_to_model(
    gpt_model: str,
    qa_table: pd.DataFrame,
    enable_context: bool,
    topk: int,
    chunk: int,
    processor,
    model,
    dataset: list[dict],
    images_per_pdf: dict,
) -> tuple[list[str], list[list[str]]]:
    """Synchronous wrapper around the async evaluation helper."""
    return asyncio.run(
        send_to_model_async(
            gpt_model,
            qa_table,
            enable_context,
            topk,
            chunk,
            processor,
            model,
            dataset,
            images_per_pdf,
        )
    )


def load_retriever(model_name: str, device: str):
    """Instantiate the requested visual retriever and its processor."""
    if model_name == "vidore/colpali-v1.3-merged":
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
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
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
    else:
        raise ValueError(f"Unsupported retriever: {model_name}")
    return processor, model


def eval_fn(
    model_name: str,
    retriever_name: str,
    device: str,
    qa_data: pd.DataFrame,
    iterations: int,
    topk: int,
    chunk: int,
    results_dir: Path,
    cache_dir: Path,
    pdf_dir: Path,
    enable_context: bool,
) -> None:
    """Evaluate a generator model across multiple iterations for a given retriever."""
    processor, retriever_model = load_retriever(retriever_name, device)

    cache_path = cache_dir / f"{retriever_name.replace('/', '_')}_pdf_emb.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fp:
            dataset = pickle.load(fp)
    else:
        dataset = create_document_embeddings(str(pdf_dir), retriever_model, processor, batch_size=4)
        with cache_path.open("wb") as fp:
            pickle.dump(dataset, fp)

    images_per_pdf = convert_pdf_dir_to_images(str(pdf_dir))

    for iteration in range(iterations):
        print(
            f"Processing iteration {iteration + 1}/{iterations} for "
            f"model {model_name} with retriever {retriever_name}."
        )

        answers, context_info = send_to_model(
            model_name,
            qa_data,
            enable_context,
            topk,
            chunk,
            processor,
            retriever_model,
            dataset,
            images_per_pdf,
        )

        eval_frame = qa_data.copy()
        eval_frame["Model"] = model_name
        eval_frame["Model_ret"] = retriever_name
        eval_frame["Answer"] = answers
        eval_frame["Context_papers"] = context_info
        eval_frame["Cor_answer"] = (eval_frame["Answer"] == eval_frame["Correct"]).astype(int)

        output_name = (
            f"eval_{retriever_name.split('/')[-1].split('-')[0]}_{model_name}_"
            f"{strftime('%Y%m%d%H%M%S', gmtime())}.csv"
        )
        eval_path = results_dir / output_name
        eval_frame.to_csv(eval_path, index=False)
        accuracy = eval_frame["Cor_answer"].mean()
        print(f"Saved results to {eval_path} | Accuracy: {accuracy:.3f}")


def main() -> None:
    args = parse_args()
    load_dotenv()

    pdf_dir = Path(args.pdf_dir)
    results_dir = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)

    ensure_dirs([pdf_dir, results_dir, cache_dir])

    qa_data = pd.read_excel(args.qa_path).sample(frac=1).reset_index(drop=True)

    device = auto_device(args.device)
    print(f"Using device: {device}")

    for generator_model in args.models:
        for retriever_model in args.retrievers:
            eval_fn(
                generator_model,
                retriever_model,
                device,
                qa_data,
                args.iterations,
                args.top_k,
                args.chunk,
                results_dir,
                cache_dir,
                pdf_dir,
                args.context,
            )


if __name__ == "__main__":
    main()
