
# vLLM Benchmark

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
vLLM_BUILD="v0.6.0"

docker run --gpus all \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:$vLLM_BUILD \
    -c "huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 cache_model.py"
```

## Benchmark

```bash
export HF_TOKEN=<your-hf-token>
export TASKS=./tasks/tasks_example.yaml

# 1xH100-80GB-SXM + v0.5.4
# results will be saved to ./results_v0.5.4/1xH100-80GB-SXM
export GPU_NAME=H100-80GB-SXM
export NUM_GPU=1
export GPU_DEVICES='"device=0"'
export SHM_SIZE=1600g
export vLLM_BUILD="v0.5.4"
export NUM_SCHEDULER_STEPS=0
./benchmark.sh

# 8xH100-80GB-SXM + v0.6.0
# results will be saved to ./results_v0.6.0/8xH100-80GB-SXM
export GPU_NAME=H100-80GB-SXM
export NUM_GPU=8
export GPU_DEVICES='"device=0,1,2,3,4,5,6,7"'
export SHM_SIZE=1600g
export vLLM_BUILD="v0.6.0"
export NUM_SCHEDULER_STEPS=10
./benchmark.sh
```
