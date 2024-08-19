#!/bin/sh

# Function to run the benchmark
run_benchmark() {
  NUM_GPU=$1
  GPU_DEVICES=$2
  SHM_SIZE=$3

  for NUM_PROMPTS in 10 20 40 80 160 320
  do
    docker run --gpus $GPU_DEVICES \
      --rm \
      --shm-size=$SHM_SIZE \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v ./:/vllm-workspace \
      --entrypoint /bin/bash \
      vllm/vllm-openai:v0.5.4 \
      -c "huggingface-cli login --token $HF_TOKEN && \
      cd /vllm-workspace && \
      python3 benchmark.py --tasks all_tasks.yaml --num_gpus $NUM_GPU --name_gpu $GPU_NAME --num_prompts $NUM_PROMPTS"
  done
}

# Benchmark configurations
run_benchmark 1 '"device=0"' "2000g"
run_benchmark 2 '"device=0,1"' "4000g"
run_benchmark 4 '"device=0,1,2,3"' "8000g"
run_benchmark 8 "all" "16000g"