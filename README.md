
# VLLM Benchmark

## Set Up Repo And Dataset
```bash
sudo usermod -aG docker ${USER} && \
newgrp docker

git clone https://github.com/LambdaLabsML/vllm-benchmark.git && \
cd vllm-benchmark && \
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Cache Model
```bash
HF_TOKEN=<your-hf-token>
SHM_SIZE=2000g
docker run --gpus all \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4 \
    -c "huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 cache_model.py"
```

## Benchmark

### Run 1x, 2x, 4x, 8x On A Server

```bash
export HF_TOKEN=<your-hf-token>
export GPU_NAME=A100-80GB-SXM

./benchmark_1x8x.sh
```

### 1xA100-80GB-SXM
```bash
HF_TOKEN=<your-hf-token>
GPU_NAME=A100-80GB-SXM

NUM_GPU=1
GPU_DEVICES='"device=0"'
SHM_SIZE=2000g
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
```

### 2xA100-80GB-SXM
```bash
HF_TOKEN=<your-hf-token>
GPU_NAME=A100-80GB-SXM

NUM_GPU=2
GPU_DEVICES='"device=0,1"'
SHM_SIZE=4000g
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
```

### 4xA100-80GB-SXM
```bash
HF_TOKEN=<your-hf-token>
GPU_NAME=A100-80GB-SXM

NUM_GPU=4
GPU_DEVICES='"device=0,1,2,3"'
SHM_SIZE=8000g
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
```

### 8xA100-80GB-SXM
```bash
HF_TOKEN=<your-hf-token>
GPU_NAME=A100-80GB-SXM

NUM_GPU=8
GPU_DEVICES=all
SHM_SIZE=16000g
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
```
