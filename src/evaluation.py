import asyncio
import json
import os
import pickle
import random
import re
import time
from enum import Enum

import aiohttp
import numpy as np
import pandas as pd
import torch
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import ColPaliForRetrieval, ColPaliProcessor

from .retrieval import retrieve_colpali
from .utils import format_msgs


class MCQ(BaseModel):
    """Pydantic model for multiple choice questions."""

    answer: str = Field(
        description="Output is the answer of a MCQ with only one of the following categories: A, B, C or D."
    )

    @field_validator("answer")
    def answer_must_letter(cls, answer: str) -> str:
        """Validates that the answer is one of A, B, C, or D."""
        if answer[0] not in ["A", "B", "C", "D"]:
            raise ValueError(f"Must be element of A B C D")
        return answer


class MSC_choice(str, Enum):
    """Enum for multiple choice answers."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class MCQ_(BaseModel):
    """Pydantic model for multiple choice questions."""

    query_answer: MSC_choice


async def post_request_with_retries(session, url, headers, data, retries=4, backoff=1):
    """Sends a POST request with exponential backoff.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        url (str): The URL to send the request to.
        headers (dict): The request headers.
        data (dict): The request data.
        retries (int, optional): The number of retries. Defaults to 4.
        backoff (int, optional): The backoff factor. Defaults to 1.

    Returns:
        str: The response text.
    """
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    choices = response_data.get("choices", [{}])
                    text = choices[0].get("message").get("content")
                    return text
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                await asyncio.sleep(backoff * (2**attempt))
            else:
                return f"[error] {response.status}"

def run_benchmark(
    vllm_port,
    model_name,
    filepath_output,
    vector_db,
    type,
    perm_quest,
    benchmark_file="./test/Glycans_q_a_v5.xlsx",
    embed_model_id="BAAI/bge-base-en-v1.5",
    qdrant_port=6333,
    top_k=5,
):
    """Runs the benchmark evaluation.

    Args:
        vllm_port (int): The port of the vLLM server.
        model_name (str): The name of the model to use.
        filepath_output (str): The path to the output file.
        vector_db (str): The name of the vector database.
        type (str): The type of RAG to use.
        perm_quest (str): Whether to permute the questions.
        benchmark_file (str, optional): The path to the benchmark file. Defaults to "./test/Glycans_q_a_v5.xlsx".
        embed_model_id (str, optional): The ID of the embedding model. Defaults to "BAAI/bge-base-en-v1.5".
        qdrant_port (int, optional): The port of the Qdrant service. Defaults to 6333.
        top_k (int, optional): The number of documents to retrieve. Defaults to 5.
    """
    qa_data = pd.read_excel(benchmark_file)
    qa_data = qa_data.sample(frac=1).reset_index(drop=True)

    qdrant_client = QdrantClient(
        url=f"http://localhost:{qdrant_port}", api_key=os.environ["QDRANT_API_KEY"]
    )

    embeddings = FastEmbedEmbeddings(
        model_name=embed_model_id, providers=["CUDAExecutionProvider"]
    )

    filepath_prompts = "./prompts_used.pkl"

    async def main():
        t_start = time.time()

        if model_name.startswith("gpt"):
            url = f"https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            }
        else:
            url = f"http://localhost:{vllm_port}/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}",
                "Content-Type": "application/json",
            }

        with open(filepath_prompts, "rb") as file:
            prompts = pickle.load(file)

        q_perm = []
        for _, row in tqdm(qa_data.iterrows()):
            answers = [row["A"], row["B"], row["C"], row["D"]]
            if perm_quest == "Yes":
                perm_idx = random.sample(range(len(answers)), len(answers))
            else:
                perm_idx = list(range(len(answers)))
            q_perm.append(perm_idx)

        prompts_ = prompt_prep(
            qa_data, prompts, qdrant_client, vector_db, embeddings, top_k, type, q_perm
        )

        final_prompts = []
        output_file = []
        for (_, row), perm_el in zip(qa_data.iterrows(), q_perm):
            question = row["question"]

            answers = [row["A"], row["B"], row["C"], row["D"]]
            answers_ = [answers[i] for i in perm_el]

            question_string = "\n".join(
                [
                    f"{letter}. {option}"
                    for letter, option in zip(["A", "B", "C", "D"], answers_)
                ]
            )

            query_final = f"""
            Generate a JSON with the query_answer, the answer provided behind the letters: A, B, C, and D. These are the values. Additional information if provided in the Context below. If the Context is not empty, analyse it and choose from the letters. MAKE SURE your output is one of the four values stated. 
            Here is the query: {question}. Here are the choices: {question_string} 
            Context:
            """
            conn = aiohttp.TCPConnector(limit=512)

            async with aiohttp.ClientSession(connector=conn) as session:
                tasks = [
                    post_request_with_retries(
                        session,
                        url=url,
                        headers=headers,
                        data={
                            "model": model_name,
                            "messages": msg,
                        },
                    )
                    for msg in prompts_[question]
                ]
                responses = await asyncio.gather(*tasks)

                fin_query = format_msgs(query_final, [], "\n".join(responses))
                output_file.append(
                    {
                        "Question_nr": row["Question_nr"],
                        "question": question,
                        "summary": responses,
                        "final_query": fin_query,
                        "quest_order": perm_el,
                    }
                )

                final_prompts.append(fin_query)

        conn = aiohttp.TCPConnector(limit=512)

        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = [
                post_request_with_retries(
                    session,
                    url=url,
                    headers=headers,
                    data={
                        "model": model_name,
                        "messages": msg,
                        "response_format": type_to_response_format_param(MCQ),
                    },
                )
                for msg in final_prompts
            ]
            answers_fin = await asyncio.gather(*tasks)

        out_list = []
        for quest, answer in zip(output_file, answers_fin):
            filt_resp, answer_ = response_real_out(answer, quest["quest_order"])
            out_list.append(
                {
                    **quest,
                    **{"answer": answer_, "resp_init": answer[:30], "filt_resp": filt_resp},
                }
            )

        timestamp = pd.Timestamp("now", tz="CET").strftime("%Y%m%d-%H%M%S")
        eval_results = {
            "model": model_name,
            "evaluation": sorted(out_list, key=lambda x: x["Question_nr"]),
            "elapsed_time": time.time() - t_start,
            "timestamp": timestamp,
        }

        if perm_quest == "Yes":
            with open(
                filepath_output + "_" + timestamp + "_perm_q.pkl", "wb"
            ) as file:
                pickle.dump(eval_results, file)
        else:
            with open(filepath_output + "_" + timestamp + ".pkl", "wb") as file:
                pickle.dump(eval_results, file)

    asyncio.run(main())

def analyze_results(results_dir, benchmark_file):
    """Analyzes the results of an experiment.

    Args:
        results_dir (str): The directory containing the experiment results.
        benchmark_file (str): The path to the benchmark file.
    """
    eval_results = [
        os.path.join(results_dir, f)
        for f in sorted(os.listdir(results_dir))
        if f.lower().startswith("eval")
    ]

    all_dfs = []
    for el in eval_results:
        with open(el, "rb") as file:
            data = pickle.load(file)
        df = pd.DataFrame(data["evaluation"])
        df["model"] = data["model"]
        df["timestamp"] = data["timestamp"]
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    qa_data = pd.read_excel(benchmark_file)
    merged_df = merged_df.merge(qa_data, on="Question_nr")

    merged_df["Cor_answer"] = 1 * (merged_df["answer"] == merged_df["Correct"])

    def is_paper_id_in_context(row):
        paper_id_val = str(row["Paper_id"])
        if not paper_id_val.startswith("Paper"):
            return np.nan

        paper_id = paper_id_val.lower()
        context_papers = row["summary"]

        if pd.isna(context_papers) or not isinstance(context_papers, list):
            return 0

        for context_paper in context_papers:
            if isinstance(context_paper, str) and paper_id in context_paper.lower():
                return 1
        return 0

    merged_df["is_paper_id_in_context"] = merged_df.apply(
        is_paper_id_in_context, axis=1
    )

    summary_table = merged_df.groupby(["model", "Difficulty"]).agg(
        Cor_answer_mean=("Cor_answer", "mean"),
        Cor_answer_std=("Cor_answer", "std"),
        is_paper_id_in_context_mean=("is_paper_id_in_context", "mean"),
        is_paper_id_in_context_std=("is_paper_id_in_context", "std"),
    ).round(3)

    print(summary_table)

    summary_table.to_excel("results/summary.xlsx")