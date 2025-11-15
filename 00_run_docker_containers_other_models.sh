#!/bin/sh

# -----------------------------------------------------------------------------
# Helper script for spinning up additional vLLM endpoints (Experiment 01).
# Uncomment/edit the sections below to launch the models you need.
# -----------------------------------------------------------------------------

export $(xargs < .env)

# Ensure you have docker permissions (run `newgrp docker` if necessary).
docker run --gpus all -it --rm \
    --name biomed_Llama_VL \
    -v "${HF_DIR}:/root/.cache/huggingface" \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" \
    --env TRANSFORMERS_OFFLINE=1 \
    --env VLLM_RPC_TIMEOUT=180000 \
    --env HF_DATASET_OFFLINE=1 \
    -p 8010:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct \
    --gpu-memory-utilization 0.75 --max_model_len 32000 --max_num_seqs 16 \
    --enforce_eager --enable-sleep-mode 
#     --disable-frontend-multiprocessing \
#     --limit_mm_per_prompt 'image=1'

docker run --gpus all -it --rm \
    --name biomed_qwenVL \
    -v "${HF_DIR}:/root/.cache/huggingface" \
    --env VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000 \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_DATASET_OFFLINE=1 \
    -p 8005:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model AdaptLLM/biomed-Qwen2-VL-2B-Instruct  \
    --gpu_memory_utilization 0.7 \
    --max_model_len 32000 --enable-sleep-mode \
    --enforce_eager
#     #kv_cacher for 0.25 is 14,9 GB for cache)
   # --shm-size 3g \

docker run --gpus all -it --rm \
    --name biomed_LLaVA \
    -v "${HF_DIR}:/root/.cache/huggingface" \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" \
    --env "TRANSFORMERS_OFFLINE=1" \
    --env "HF_DATASET_OFFLINE=1" \
    -p 8001:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B \
    --gpu_memory_utilization 0.4 --max_model_len 8192 --enable-sleep-mode
# # # #     # kv_cache for 0.25 is 3,18GBd
