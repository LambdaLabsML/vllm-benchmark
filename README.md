
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
SHM_SIZE=1600g
vLLM_BUILD="vllm/vllm-openai:v0.6.0"

docker run --gpus all \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    $vLLM_BUILD \
    -c "huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 cache_model.py"
```

## Benchmark

### Run 1x, 2x, 4x, 8x On A Server

```bash
export HF_TOKEN=<your-hf-token>
export GPU_NAME=H100-80GB-SXM
export vLLM_BUILD="vllm/vllm-openai:v0.6.0"

./benchmark_1x2x4x8x.sh $vLLM_BUILD
```