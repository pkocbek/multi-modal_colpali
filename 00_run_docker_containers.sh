#!/bin/sh

export $(xargs < .env)

newgrp docker

docker run \
	--rm \
    --name qdrant_vd \
	--gpus=all \
	-p 6333:6333 \
	-p 6334:6334 \
	-e QDRANT__GPU__INDEXING=1 \
    -e "QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}" \
    --ulimit nofile=65536:65536 \
    -v "${VD_DIR}/storage:/qdrant/storage" \
    -v "${VD_DIR}/custom_config.yaml:/qdrant/config/custom_config.yaml" \
	qdrant/qdrant:gpu-nvidia-latest


#add --pull=always if needed
docker run --gpus all -it \
    --name gemma_27b \
    -v "${HF_DIR}:/root/.cache/huggingface" \
    --env "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" \
    --env TRANSFORMERS_OFFLINE=1 \
    --env VLLM_RPC_TIMEOUT=180000 \
    --env HF_DATASET_OFFLINE=1 \
    -p 8006:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "google/gemma-3-27b-it" \
    --limit_mm_per_prompt '{"image": 10}' \
    --gpu-memory-utilization 0.82 --max_model_len 16000 \
    --enable-sleep-mode 
