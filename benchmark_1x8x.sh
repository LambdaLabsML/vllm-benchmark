#!/bin/sh

HF_TOKEN="<your-hf-token>"
GPU_NAME="A100-80GB-SXM"

# Benchmark 1x
NUM_GPU=1
GPU_DEVICES='"device=0"'
SHM_SIZE="2000g"
docker run --gpus $GPU_DEVICES \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4 \
    -c "huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 benchmark.py --tasks all_tasks.yaml --num_gpus $NUM_GPU --name_gpu $GPU_NAME"

# Benchmark 2x
NUM_GPU=2
GPU_DEVICES='"device=0,1"'
SHM_SIZE="4000g"
docker run --gpus $GPU_DEVICES \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4 \
    -c "huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 benchmark.py --tasks all_tasks.yaml --num_gpus $NUM_GPU --name_gpu $GPU_NAME"

# Benchmark 4x
NUM_GPU=4
GPU_DEVICES='"device=0,1,2,3"'
SHM_SIZE="8000g"
docker run --gpus $GPU_DEVICES \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4 \
    -c "huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 benchmark.py --tasks all_tasks.yaml --num_gpus $NUM_GPU --name_gpu $GPU_NAME"

# Benchmark 8x
NUM_GPU=8
GPU_DEVICES=all
SHM_SIZE="16000g"
docker run --gpus $GPU_DEVICES \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4 \
    -c "huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 benchmark.py --tasks all_tasks.yaml --num_gpus $NUM_GPU --name_gpu $GPU_NAME"