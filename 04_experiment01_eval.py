#!/usr/bin/env python3
"""
Aggregate Experiment 01 evaluation pickles into accuracy reports.

The script scans `results/eval`, reads every pickle produced by
`02_experiment01.py`, merges the responses with the benchmark sheet,
and exports accuracy summaries (per difficulty, per retrieval setting,
and majority-vote tables).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARK_FILE = "./data/Glycans_q_a_v5.xlsx"
SUMMARY_PATH = Path("results/eval_results.xlsx")
MAJORITY_PATH = Path("results/eval_maj_results.xlsx")
FULL_PATH = Path("results/eval_full_results.xlsx")


FILE_PATTERN = re.compile(
    r"eval_(?P<model_short>[^_]+)_(?P<vd_name>.+)_(?P<perm_flag>perm|no_perm)_benchmark_(?P<timestamp>\d{8}-\d{6})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise Experiment 01 evaluation pickles.")
    parser.add_argument(
        "--eval-dir",
        default="results/eval",
        help="Directory containing pickled evaluation files.",
    )
    parser.add_argument(
        "--benchmark-path",
        default=BENCHMARK_FILE,
        help="Path to the benchmark Excel file.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(SUMMARY_PATH),
        help="Output Excel path for per-difficulty accuracy tables.",
    )
    parser.add_argument(
        "--majority-path",
        default=str(MAJORITY_PATH),
        help="Output Excel path for majority-vote accuracy tables.",
    )
    parser.add_argument(
        "--full-path",
        default=str(FULL_PATH),
        help="Output Excel path for the merged raw evaluations.",
    )
    return parser.parse_args()


def load_evaluation_pickle(path: Path) -> dict:
    import pickle

    with path.open("rb") as fh:
        return pickle.load(fh)


def parse_metadata(path: Path) -> dict:
    stem = path.stem
    perm_suffix = stem.endswith("_perm_q")
    if perm_suffix:
        stem = stem[: -len("_perm_q")]

    match = FILE_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Cannot parse evaluation filename: {path.name}")

    vd_name = match.group("vd_name")
    return {
        "model_short": match.group("model_short"),
        "vd_name": vd_name,
        "perm_label": match.group("perm_flag"),
        "timestamp": match.group("timestamp"),
        "perm_suffix": perm_suffix,
    }


def build_dataframe(eval_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in eval_dir.glob("eval_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No evaluation pickles found in {eval_dir}")

    frames = []
    for pkl_path in files:
        blob = load_evaluation_pickle(pkl_path)
        meta = parse_metadata(pkl_path)

        df = pd.DataFrame(blob["evaluation"])
        df["model"] = blob.get("model")
        df["model_short"] = meta["model_short"]
        df["vd_name"] = meta["vd_name"]
        df["elapsed_time"] = blob.get("elapsed_time")
        df["run_timestamp"] = blob.get("timestamp", meta["timestamp"])
        df["file_timestamp"] = meta["timestamp"]
        df["permuted_answers"] = blob.get("permuted_answers", meta["perm_label"] == "perm")
        df["filepath"] = str(pkl_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def compute_majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["model_short", "model", "vd_name", "permuted_answers", "Question_nr", "Difficulty"],
            observed=True,
        )["Cor_answer"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grouped["Maj_vote"] = (grouped["sum"] >= np.ceil(grouped["count"] / 2)).astype(int)
    pivot = (
        grouped.groupby(["model_short", "model", "vd_name", "permuted_answers"], observed=True)["Maj_vote"]
        .mean()
        .reset_index()
    )
    return pivot


def compute_summary_tables(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.pivot_table(
        df,
        values="Cor_answer",
        index=["model_short", "model", "vd_name", "permuted_answers"],
        columns="Difficulty",
        aggfunc="mean",
        observed=True,
    )
    pivot = pivot.reindex(columns=["Easy", "Medium", "Hard"])
    return pivot


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    benchmark = pd.read_excel(args.benchmark_path)

    evaluation_df = build_dataframe(eval_dir)

    merged = evaluation_df.merge(
        benchmark[["Question_nr", "Correct", "Difficulty"]],
        on="Question_nr",
        how="left",
    )
    merged["Cor_answer"] = (merged["answer"] == merged["Correct"]).astype(int)
    merged["Difficulty"] = pd.Categorical(
        merged["Difficulty"],
        categories=["Easy", "Medium", "Hard"],
        ordered=True,
    )
    merged["vd_name"] = pd.Categorical(
        merged["vd_name"],
        categories=["no_RAG", "text_RAG", "mm_RAG", "colpali"],
        ordered=True,
    )

    # Save full merged dataset
    Path(args.full_path).parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["model_short", "vd_name", "permuted_answers", "Question_nr"]).to_excel(
        args.full_path, index=False
    )

    # Summary accuracies by difficulty
    summary = compute_summary_tables(merged)
    with pd.ExcelWriter(args.summary_path) as writer:
        summary.to_excel(writer, sheet_name="Accuracy")

    # Majority vote statistics
    majority = compute_majority_vote(merged)
    majority.to_excel(args.majority_path, index=False)

    print(f"[done] Summary saved to {args.summary_path}")
    print(f"[done] Majority vote saved to {args.majority_path}")
    print(f"[done] Full evaluations saved to {args.full_path}")


if __name__ == "__main__":
    main()
