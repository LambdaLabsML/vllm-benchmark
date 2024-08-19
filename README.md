
# VLLM Benchmark

## Set Up Repo And Dataset
```bash
git clone https://github.com/LambdaLabsML/vllm-benchmark.git && \
cd vllm-benchmark && \
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Cache Model
```bash
python3 cache_model.py
```

## Validation A Specific GPU Configuration

### 1xGPU 80GB 
```bash
docker run --gpus '"device=0"' \
    -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4

huggingface-cli login --token <your-hf-token>

python3 validate.py --tasks all_tasks.yaml --num_gpus 1 --vram 80
```

### 2xGPU 80GB 
```bash
docker run --gpus '"device=1,2"' \
    -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4

huggingface-cli login --token <your-hf-token>

python3 validate.py --tasks all_tasks.yaml --num_gpus 2 --vram 80
```

### 4xGPU 80GB
```bash
docker run --gpus '"device=3,4,5,6"' \
    -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4

huggingface-cli login --token <your-hf-token>

python3 validate.py --tasks all_tasks.yaml --num_gpus 4 --vram 80
```

### 8xGPU 80GB
```bash
docker run --gpus '"device=3,4,5,6"' \
    -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.5.4

huggingface-cli login --token <your-hf-token>

python3 validate.py --tasks all_tasks.yaml --num_gpus 8 --vram 80
```

## Benchmark