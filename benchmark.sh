#!/bin/sh

HF_TOKEN=${HF_TOKEN:-""}  
vLLM_BUILD=${vLLM_BUILD:-"v0.5.4"}
TASKS=${TASKS:-"./tasks/tasks_example.yaml"} 
GPU_NAME=${GPU_NAME:-"H100-80GB-SXM"}
GPU_DEVICES=${GPU_DEVICES:-"device=0"}
NUM_GPU=${NUM_GPU:-1}
SHM_SIZE=${SHM_SIZE:-"1600g"}  
NUM_SCHEDULER_STEPS=${NUM_SCHEDULER_STEPS:-0} 
MAX_NUM_SEQ=${MAX_NUM_SEQ:-256}  

RESULT_PATH="results_${vLLM_BUILD}/${NUM_GPU}x${GPU_NAME}"

for NUM_PROMPTS in 10 20 40 80 160 320 640 1280 5000
do
  docker run --gpus $GPU_DEVICES \
    --rm \
    --shm-size=$SHM_SIZE \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./:/vllm-workspace \
    --entrypoint /bin/bash \
    vllm/vllm-openai:$vLLM_BUILD \
    -c "ulimit -n 65536 && \
    huggingface-cli login --token $HF_TOKEN && \
    cd /vllm-workspace && \
    python3 benchmark.py --tasks $TASKS --num_gpus $NUM_GPU --name_gpu $GPU_NAME --num_prompts $NUM_PROMPTS --num-scheduler-steps $NUM_SCHEDULER_STEPS --result_path $RESULT_PATH --max-num-seqs $MAX_NUM_SEQ"
done
