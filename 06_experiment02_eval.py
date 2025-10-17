#!/usr/bin/env python3

"""
Experiment 02: summarise evaluation CSV files into a consolidated Excel report.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Experiment 02 aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate Experiment 02 evaluation CSVs.")
    parser.add_argument(
        "--results_dir",
        default="results/evals",
        help="Directory containing experiment CSV outputs.",
    )
    parser.add_argument(
        "--output",
        default="results/summary.xlsx",
        help="Path for the aggregated Excel workbook.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        help="Ordered list of evaluated generator models.",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=[
            "vidore/colpali-v1.3-merged",
            "vidore/colqwen2.5-v0.2",
            "ahmed-masry/ColFlor",
        ],
        help="Ordered list of retrieval models.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV result files in the target directory."""
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    frames = [pd.read_csv(csv_file) for csv_file in csv_files]
    return pd.concat(frames, ignore_index=True)


def parse_context_presence(row: pd.Series) -> float:
    """Return 1 if the gold paper ID appears in the retrieved context list."""
    paper_id_str = str(row.get("Paper_id", ""))
    if not paper_id_str.lower().startswith("paper"):
        return np.nan

    context = row.get("Context_papers")
    if not isinstance(context, str) or not context.startswith("["):
        return 0.0

    try:
        context_list = ast.literal_eval(context)
    except (ValueError, SyntaxError):
        return 0.0

    paper_seed = paper_id_str.lower()
    for item in context_list:
        if str(item).split("_pg_")[0].lower() == paper_seed:
            return 1.0
    return 0.0


def build_summary_table(
    df: pd.DataFrame, models: list[str], retrievers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create difficulty-aware and overall accuracy summaries."""
    df = df.copy()
    df["Model"] = pd.Categorical(df["Model"], categories=models, ordered=True)
    df["Model_ret"] = pd.Categorical(df["Model_ret"], categories=retrievers, ordered=True)
    df["is_paper_id_in_context"] = df.apply(parse_context_presence, axis=1)

    grouped = df.groupby(["Model", "Model_ret", "Difficulty"], observed=True)
    agg = grouped.agg(
        mean_cor=("Cor_answer", "mean"),
        std_cor=("Cor_answer", "std"),
        mean_hit=("is_paper_id_in_context", "mean"),
        std_hit=("is_paper_id_in_context", "std"),
    ).round(3)

    agg["Cor_answer"] = agg["mean_cor"].astype(str) + " (SD=" + agg["std_cor"].astype(str) + ")"
    agg["is_paper_id_in_context"] = (
        agg["mean_hit"].astype(str) + " (SD=" + agg["std_hit"].astype(str) + ")"
    )
    difficulty_summary = agg[["Cor_answer", "is_paper_id_in_context"]].unstack("Difficulty")
    difficulty_summary.columns = [
        f"{metric}_{difficulty}" for metric, difficulty in difficulty_summary.columns
    ]

    grouped_simple = df.groupby(["Model", "Model_ret"], observed=True)
    agg_simple = grouped_simple.agg(
        mean_cor=("Cor_answer", "mean"),
        std_cor=("Cor_answer", "std"),
        mean_hit=("is_paper_id_in_context", "mean"),
        std_hit=("is_paper_id_in_context", "std"),
    ).round(3)
    agg_simple["Cor_answer_summary"] = (
        agg_simple["mean_cor"].astype(str) + " (SD=" + agg_simple["std_cor"].astype(str) + ")"
    )
    agg_simple["is_paper_id_in_context_summary"] = (
        agg_simple["mean_hit"].astype(str) + " (SD=" + agg_simple["std_hit"].astype(str) + ")"
    )

    final_summary = difficulty_summary.merge(
        agg_simple[["Cor_answer_summary", "is_paper_id_in_context_summary"]],
        left_index=True,
        right_index=True,
    )

    return final_summary, df


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    data = load_results(results_dir)
    summary, raw = build_summary_table(data, args.models, args.retrievers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="Summary")
        raw.to_excel(writer, sheet_name="raw_evaluations", index=False)

    print(f"Saved aggregated results to {output_path}")


if __name__ == "__main__":
    main()
