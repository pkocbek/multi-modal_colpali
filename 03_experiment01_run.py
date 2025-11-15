#!/usr/bin/env python3
"""
Batch runner for Experiment 01.

Loops over RAG modes (no_RAG, text_RAG, mm_RAG, ColPali), answer permutations,
and repeat counts, invoking 02_experiment01.py for each setting.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

EVAL_SCRIPT = "02_experiment01.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Experiment 01 runner (supports permutations and multiple RAG settings)."
    )
    parser.add_argument("--vllm_port", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_name_short", required=True, type=str)
    parser.add_argument("--vd_mm_name", required=True, type=str)
    parser.add_argument("--vd_colpali_name", required=True, type=str)
    parser.add_argument("--vd_text_name", required=True, type=str)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument(
        "--top_k",
        default=5,
        type=int,
        help="Number of retrieved context items per question (mm_RAG/ColPali).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path("./results/eval")
    eval_root.mkdir(parents=True, exist_ok=True)

    evaluation_modes = [
        ("no_RAG", "", ""),
        ("text_RAG", args.vd_text_name, "mm_RAG"),
        ("mm_RAG", args.vd_mm_name, "mm_RAG"),
        ("colpali", args.vd_colpali_name, "colpali"),
    ]

    perm_settings = [
        (True, "perm"),
        (False, "no_perm"),
    ]

    t_start0 = time.time()
    for permute, perm_label in perm_settings:
        perm_flag = ["--perm_quest", "Yes"] if permute else []
        for repeat_idx in range(1, args.repeats + 1):
            loop_start = time.time()
            for eval_label, vector_db, eval_type in evaluation_modes:
                print(
                    f"Model={args.model_name} | Port={args.vllm_port} | "
                    f"Mode={eval_label} ({vector_db or 'none'}) | "
                    f"permute={permute} | repeat {repeat_idx}/{args.repeats} | top_k={args.top_k}"
                )
                output_stub = eval_root / f"eval_{args.model_name_short}_{eval_label}_{perm_label}_benchmark"
                cmd = [
                    "python",
                    EVAL_SCRIPT,
                    "--vllm_port",
                    args.vllm_port,
                    "--model_name",
                    args.model_name,
                    "--filepath_output",
                    str(output_stub),
                    "--vector_db",
                    vector_db,
                    "--type",
                    eval_type,
                    "--top_k",
                    str(args.top_k),
                ]
                cmd.extend(perm_flag)
                subprocess.call(cmd)

            print(
                f"Permutation={perm_label} repeat {repeat_idx}/{args.repeats} finished in "
                f"{time.time() - loop_start:.2f}s."
            )

    total_loops = args.repeats * len(perm_settings)
    print(
        f"\nFull evaluation task for model {args.model_name} with {total_loops} "
        f"repeat configurations took {time.time() - t_start0:.2f}s."
    )


if __name__ == "__main__":
    main()
