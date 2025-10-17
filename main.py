import argparse
import json
import os
import time

from qdrant_client import QdrantClient
from transformers import AutoTokenizer

from src.data_processing import pdf_loader
from src.evaluation import analyze_results, run_benchmark
from src.retrieval import qdrant_process


def prepare_data(pdf_dir, benchmark_file):
    """Prepares the data for the experiments by processing the PDF files and creating the vector stores.

    Args:
        pdf_dir (str): The path to the directory containing the PDF files.
        benchmark_file (str): The path to the benchmark file.
    """
    print(f"Preparing data from {pdf_dir} and {benchmark_file}")

    papers = [
        os.path.join(pdf_dir, f)
        for f in sorted(os.listdir(pdf_dir))
        if f.lower().endswith(".pdf")
    ]
    doi_links = ["" for _ in papers]
    filenames = [os.path.basename(p) for p in papers]

    EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
    emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    qdrant_client = QdrantClient(
        url="http://localhost:6333", api_key=os.environ.get("QDRANT_API_KEY")
    )

    processed_multi, processed_text = pdf_loader(
        papers, doi_links, filenames, "./vector_store", emb_tokenizer
    )

    MODELS = [
        {
            "model_name": "gpt-4o-2024-11-20",
            "model_short": "gpt-4o",
            "port": "9999",
            "text_vd": "RAG_TEXT",
            "mm_vd": "MM_04_GPT_4o",
            "late_inter": "vidore/colpali-v1.3-merged",
            "late_inter_short": "ColPali",
        },
        {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_short": "gpt-4o-mini",
            "port": "9999",
            "text_vd": "RAG_TEXT",
            "mm_vd": "MM_05_GPT_4o_mini",
            "late_inter": "vidore/colpali-v1.3-merged",
            "late_inter_short": "ColPali",
        },
        {
            "model_name": "google/gemma-3-27b-it",
            "model_short": "Gemma3-27b",
            "port": "8006",
            "text_vd": "RAG_TEXT",
            "mm_vd": "MM_06_GEMMA3_27B",
            "late_inter": "vidore/colpali-v1.3-merged",
            "late_inter_short": "ColPali",
        },
    ]

    qdrant_process(processed_text, qdrant_client, "RAG_TEXT", 768, None)
    for model in MODELS:
        qdrant_process(processed_multi, qdrant_client, model["mm_vd"], 768, None)


def run_experiment(config_file):
    """Runs an experiment based on a configuration file.

    Args:
        config_file (str): The path to the experiment configuration file.
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    repeats = config.get("repeats", 1)

    for model_config in config["models"]:
        model_name = model_config["model_name"]
        model_name_short = model_config["model_name_short"]
        vllm_port = model_config["vllm_port"]
        vd_mm_name = model_config["vd_mm_name"]
        vd_colpali_name = model_config["vd_colpali_name"]
        vd_text_name = model_config["vd_text_name"]

        t_start0 = time.time()

        for i in range(repeats):
            t_start = time.time()

            for permuted_questions in ["Yes", "No"]:
                print(
                    f"Processing model: {model_name}, port: {vllm_port}, vd_name: no_RAG, permuted_questions: {permuted_questions}.."
                )
                run_benchmark(
                    vllm_port,
                    model_name,
                    str("./results/eval/eval_" + model_name_short + "_no_RAG_benchmark"),
                    "",
                    "",
                    permuted_questions,
                )

                print(
                    f"Processing model: {model_name}, port: {vllm_port}, vd_name: text_RAG ({vd_text_name}), permuted_questions: {permuted_questions}."
                )
                run_benchmark(
                    vllm_port,
                    model_name,
                    str(
                        "./results/eval/eval_" + model_name_short + "_text_RAG_benchmark"
                    ),
                    vd_text_name,
                    "mm_RAG",
                    permuted_questions,
                )

                print(
                    f"Processing model: {model_name}, port: {vllm_port}, vd_name: mm_RAG ({vd_mm_name}), permuted_questions: {permuted_questions}."
                )
                run_benchmark(
                    vllm_port,
                    model_name,
                    str("./results/eval/eval_" + model_name_short + "_mm_RAG_benchmark"),
                    vd_mm_name,
                    "mm_RAG",
                    permuted_questions,
                )

                print(
                    f"Processing model: {model_name}, port: {vllm_port}, vd_name: colpali ({vd_colpali_name}), permuted_questions: {permuted_questions}."
                )
                run_benchmark(
                    vllm_port,
                    model_name,
                    str(
                        "./results/eval/eval_" + model_name_short + "_colpali_benchmark"
                    ),
                    vd_colpali_name,
                    "colpali",
                    permuted_questions,
                )

            print(
                f"Evaluation task for model {model_name}, {int(i) + 1}/{repeats}, loop took {time.time() - t_start} seconds."
            )

        print(
            f"\nFull evaluation task for model {model_name} with {repeats} repeats took {time.time() - t_start0} seconds."
        )


def main():
    """Main function of the CLI tool."""
    parser = argparse.ArgumentParser(
        description="A CLI tool for the multi-modal ColPali project."
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare_data_parser = subparsers.add_parser(
        "prepare-data", help="Prepare data for the experiments."
    )
    prepare_data_parser.add_argument(
        "--pdf_dir",
        type=str,
        required=True,
        help="Path to the directory containing the PDF files.",
    )
    prepare_data_parser.add_argument(
        "--benchmark_file",
        type=str,
        required=True,
        help="Path to the benchmark file.",
    )

    run_experiment_parser = subparsers.add_parser(
        "run-experiment", help="Run an experiment."
    )
    run_experiment_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration file.",
    )

    analyze_results_parser = subparsers.add_parser(
        "analyze-results", help="Analyze the results of an experiment."
    )
    analyze_results_parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory containing the experiment results.",
    )
    analyze_results_parser.add_argument(
        "--benchmark_file",
        type=str,
        required=True,
        help="Path to the benchmark file.",
    )

    args = parser.parse_args()

    if args.command == "prepare-data":
        prepare_data(args.pdf_dir, args.benchmark_file)
    elif args.command == "run-experiment":
        run_experiment(args.config)
    elif args.command == "analyze-results":
        analyze_results(args.results_dir, args.benchmark_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()