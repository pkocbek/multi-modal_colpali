#!/usr/bin/env python3

"""
for deployed model on docker on specific port
"""
import argparse
import subprocess
import time
from pathlib import Path

EVAL_SCRIPT = "02_experiment01.py"

#python run_eval_model_repeat_perm_q.py --vllm_port "8006" --model_name "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8" --model_name_short "qwen3_30b_think_fp8" --vd_mm_name "MM21_qwen3_30b_think_fp8" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model_repeat_perm_q.py --vllm_port "8006" --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8" --model_name_short "qwen3_30b_fp8" --vd_mm_name "MM20_qwen3_30b_fp8" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model.py --vllm_port "8001" --model_name "AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B" --model_name_short "llava" --vd_mm_name "MM_01_Llava" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT"

#python run_eval_model.py --vllm_port #8004# --model_name "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4" --model_name_short "qwen72" --vd_mm_name "MM_06_QWEN_72B" --vd_colpali_name "COL_PALI" -vd_text_name "RAG_TEXT"

#python run_eval_model.py --vllm_port "8005" --model_name "AdaptLLM/biomed-Qwen2-VL-2B-Instruct" --model_name_short "qwen" --vd_mm_name "MM_02_Qwen" --vd_colpali_name "COL_PALI" -vd_text_name "RAG_TEXT"

#python run_eval_model.py --vllm_port "8007" --model_name "ibm-granite/granite-vision-3.2-2b" --model_name_short "ibm_granite" --vd_mm_name "MM_00_Base" --vd_colpali_name "COL_PALI" -vd_text_name "RAG_TEXT"

#python run_eval_model.py --vllm_port "8010" --model_name "AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct" --model_name_short "llama" --vd_mm_name "MM_03_Llama" --vd_colpali_name "COL_PALI" -vd_text_name "RAG_TEXT"

#python run_eval_model.py --vllm_port "9999" --model_name "gpt-4o-2024-11-20" --model_name_short "gpt_4o" --vd_mm_name "MM_04_GPT_4o" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT"

#python run_eval_model.py --vllm_port "9999" --model_name "gpt-4o-mini-2024-07-18" --model_name_short "gpt_4o_mini" --vd_mm_name "MM_05_GPT_4o_mini" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT"

# python run_eval_model.py --vllm_port "8008" --model_name "google/gemma-3-12b-it" --model_name_short "gemma_12b" --vd_mm_name "MM_08_GEMMA3_12B" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT"

#python run_eval_model.py --vllm_port "8009" --model_name "google/gemma-3-4b-it" --model_name_short "gemma_4b" --vd_mm_name "MM_09_GEMMA3_4B" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT"

#python run_eval_model_repeat.py --vllm_port "8012" --model_name "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g" --model_name_short "gemma_27b_q4" --vd_mm_name "MM_10_GEMMA3_27B_Q4" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model_repeat.py --vllm_port "9999" --model_name "gpt-4o-2024-11-20" --model_name_short "gpt_4o" --vd_mm_name "MM_04_GPT_4o" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"



#python run_eval_model_repeat.py --vllm_port "9999" --model_name "gpt-4o-mini-2024-07-18" --model_name_short "gpt_4o_mini" --vd_mm_name "MM_05_GPT_4o_mini" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model_repeat.py --vllm_port "8013" --model_name "gaunernst/gemma-3-27b-it-int4-awq" --model_name_short "gemma_27b_awq" --vd_mm_name "MM_07_GEMMA3_27B" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model_repeat_perm_q.py --vllm_port "8012" --model_name "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g" --model_name_short "gemma_27b_q4" --vd_mm_name "MM_10_GEMMA3_27B_Q4" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model_repeat_perm_q.py --vllm_port "8006" --model_name "google/gemma-3-27b-it" --model_name_short "gemma_27b" --vd_mm_name "MM_07_GEMMA3_27B" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model_repeat_perm_q.py --vllm_port "9999" --model_name "gpt-4o-mini-2024-07-18" --model_name_short "gpt_4o_mini" --vd_mm_name "MM_05_GPT_4o_mini" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"

#python run_eval_model_repeat_perm_q.py --vllm_port "9999" --model_name "gpt-4o-2024-11-20" --model_name_short "gpt_4o" --vd_mm_name "MM_04_GPT_4o" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "5"


#python run_eval_model_repeat_perm_q.py --vllm_port "8006" --model_name "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit" --model_name_short "Mistral_24b_4bit" --vd_mm_name "MM17_Mistral_24B_4bit" --vd_colpali_name "COL_PALI" --vd_text_name "DEP_RAG_TEXT" --repeats "5"


#python run_eval_model_repeat_perm_q.py --vllm_port "9999" --model_name "gpt-5-2025-08-07" --model_name_short "gpt_5" --vd_mm_name "MM18_GPT_5" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "1"

#python run_eval_model_repeat.py --vllm_port "9999" --model_name "gpt-5-mini-2025-08-07" --model_name_short "gpt_5_mini" --vd_mm_name "MM18_GPT_5_mini" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "1"


#python run_eval_model_repeat.py --vllm_port "9999" --model_name "gpt-5-nano-2025-08-07" --model_name_short "gpt_5_nano" --vd_mm_name "MM19_GPT_5_nano" --vd_colpali_name "COL_PALI" --vd_text_name "RAG_TEXT" --repeats "1"



def main():
    parser = argparse.ArgumentParser(
        description="Script that adds 5 variables from CMD"
    )
    parser.add_argument("--vllm_port", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_name_short", required=True, type=str)
    parser.add_argument("--vd_mm_name", required=True, type=str)
    parser.add_argument("--vd_colpali_name", required=True, type=str)
    parser.add_argument("--vd_text_name", required=True, type=str)
    parser.add_argument("--repeats", required=True, type=str)


    args = parser.parse_args()

    vllm_port = args.vllm_port 
    model_name  = args.model_name
    model_name_short  = args.model_name_short
    vd_mm_name = args.vd_mm_name
    vd_colpali_name = args.vd_colpali_name
    vd_text_name = args.vd_text_name
    if model_name.startswith("gpt"):
        port="not_local"
    else:
        port = vllm_port
    repeats  = int(args.repeats)

    eval_root = Path("./results/eval")
    eval_root.mkdir(parents=True, exist_ok=True)

    evaluation_modes = [
        ("no_RAG", "", ""),
        ("text_RAG", vd_text_name, "mm_RAG"),
        ("mm_RAG", vd_mm_name, "mm_RAG"),
        ("colpali", vd_colpali_name, "colpali"),
    ]

    perm_settings = [
        (True, "perm"),
        (False, "no_perm"),
    ]

    t_start0=time.time()

    for permute, perm_label in perm_settings:
        perm_flag = ["--perm_quest", "Yes"] if permute else []
        for i in range(repeats):
            t_start=time.time()
            repeat_idx = i + 1
            for eval_label, vector_db, eval_type in evaluation_modes:
                print(
                    f"Processing model: {model_name}, port: {port}, "
                    f"vd_name: {eval_label} ({vector_db or 'none'}), "
                    f"permute_answers={permute}, repeat {repeat_idx}/{repeats}"
                )
                output_stub = eval_root / f"eval_{model_name_short}_{eval_label}_{perm_label}_benchmark"
                cmd = [
                    "python",
                    EVAL_SCRIPT,
                    "--vllm_port",
                    vllm_port,
                    "--model_name",
                    model_name,
                    "--filepath_output",
                    str(output_stub),
                    "--vector_db",
                    vector_db,
                    "--type",
                    eval_type,
                ]
                cmd.extend(perm_flag)
                subprocess.call(cmd)

            print(
                f"Evaluation loop ({perm_label}) for model {model_name}, "
                f"repeat {repeat_idx}/{repeats} took {time.time() - t_start:.2f} seconds."
            )
    
    total_loops = repeats * len(perm_settings)
    print(
        f"\nFull evaluation task for model {model_name} with {total_loops} "
        f"repeats (including permutations) took {time.time() - t_start0:.2f} seconds."
    )
if __name__ == '__main__':
    main()
