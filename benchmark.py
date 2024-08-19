import subprocess
import threading
import os
import signal
import yaml
import json
import time
import argparse

message_fail = [
    "Server process terminated unexpectedly",
    "Detect error during server process launch",
    "Server failed to start within the timeout period",
    "A smaller test has failed"
]

def read_output(pipe):
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print(line, end='')  # You can log or process this output as needed
    finally:
        pipe.close()

def run_benchmark(model, max_model_len, num_gpus, output_json, vllm_start_timeout, dataset_name, dataset_path, num_prompts):

    # Start the server and capture its output for error detection
    server_cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--max-model-len", str(max_model_len)]
    if num_gpus > 1:
        server_cmd.append("--tensor-parallel-size")
        server_cmd.append(str(num_gpus))
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    start_time = time.time()
    server_started = False

    while time.time() - start_time < vllm_start_timeout:    
        # Check if the server process has terminated
        if server_process.poll() is not None:
            print(message_fail[0] + "============================")
            stdout, stderr = server_process.communicate()
            print("Stdout:", stdout)
            print("Stderr:", stderr)
            return message_fail[0]
        
        # Check the message from std
        stderr_line = server_process.stderr.readline()
        if stderr_line:
            # Handle specific error detection here if needed
            if "ValueError" in stderr_line or "Traceback" in stderr_line:
                print(message_fail[1] + "============================")
                os.kill(server_process.pid, signal.SIGTERM)
                server_process.wait()  # Wait for the server process to terminate
                return  message_fail[1]
            elif "Application startup complete" in stderr_line:
                print("Server process has been launched ============================")
                server_started = True
                break

    if not server_started:
        print(message_fail[2] + "============================")
        os.kill(server_process.pid, signal.SIGTERM)
        server_process.wait()  # Wait for the server process to terminate
        return  message_fail[2]

    stdout_thread = threading.Thread(target=read_output, args=(server_process.stdout,))
    stderr_thread = threading.Thread(target=read_output, args=(server_process.stderr,))

    stdout_thread.start()
    stderr_thread.start()

    benchmark_process = subprocess.run([
        "python3", "benchmark_serving.py",
        "--backend", "vllm",
        "--dataset-name", dataset_name,
        "--dataset-path", dataset_path,
        "--model", model,
        "--num-prompts", str(num_prompts),
        "--endpoint", "/v1/completions",
        "--tokenizer", model,
        "--save-result",
        "--result-filename", output_json
    ])
    bench_serving_exit_code = benchmark_process.returncode

    # Terminate the server process
    os.kill(server_process.pid, signal.SIGTERM)
    try:
        # Keep the main thread running while subprocess is active
        server_process.wait()
    finally:
        # Ensure threads finish their work before the program exits
        stdout_thread.join()
        stderr_thread.join()
    server_process.wait()  # Wait for the server process to terminate

    return bench_serving_exit_code


def main(args):
    tasks=args.tasks
    num_gpus=args.num_gpus
    name_gpu=args.name_gpu
    num_prompts=args.num_prompts
    dataset_name=args.dataset_name
    dataset_path=args.dataset_path
    vllm_start_timeout=args.vllm_start_timeout

    print(f"Tasks file: {tasks}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Name GPU: {name_gpu}")
    print(f"Number of prompts: {num_prompts}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"VLLM start timeout: {vllm_start_timeout} seconds")

    # Load the YAML file
    with open(tasks, 'r') as file:
        data = yaml.safe_load(file)

    # Extract the list of tasks
    tasks = data.get('tasks', [])

    output_dir = f"results/prompt_{num_prompts}/{num_gpus}x{name_gpu}"
    os.makedirs(output_dir, exist_ok=True)

    flag_failed = False
    last_model = tasks[0]["model"]

    for task in tasks:
        print(f"Running benchmark for model: {task}")
        model = task["model"]
        max_model_len = task["max_model_len"]
        output_json = output_dir + "/" + model.split("/")[1] + "_tp" + str(num_gpus) + "_len" + str(max_model_len) + ".json"
        
        if flag_failed == True:
            if last_model == task["model"]:
                data = {
                    "reason": message_fail[3]
                }
                with open(output_json, 'w') as json_file:
                    json.dump(data, json_file)
                print(f"Skip the task {task} because {message_fail[3]}")
                print(f"Reason of failure: {message_fail[3]} has been written to {output_json}")
                continue
            else:
                print(f"Reset flag_failed to False because a new model {task} is used.")
                last_model = task["model"]
                flag_failed = False
        try:
            exit_code = run_benchmark(model, max_model_len, num_gpus, output_json, vllm_start_timeout, dataset_name, dataset_path, num_prompts)
            print(f"Benchmark test completed with exit code: {exit_code}\n")
            if exit_code != 0:
                flag_failed = True
                print(f"Set flag_failed to True for model {task}")
                data = {
                    "reason": exit_code
                }
                with open(output_json, 'w') as json_file:
                    json.dump(data, json_file)
                print(f"Reason of failure: {exit_code} has been written to {output_json}")
        except Exception as e:
            print(f"An error occurred while running benchmark for model {task}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run benchmark with configurable parameters.")

    parser.add_argument("--tasks", type=str, default='all_tasks.yaml', help="Path to the tasks YAML file.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--name_gpu", type=str, default="A100-80GB-SXM", help="Name of GPU")
    parser.add_argument("--num_prompts", type=int, default=40, help="Number of prompts to use for benchmarking.")
    parser.add_argument("--dataset_name", type=str, default="sharegpt", help="Name of the dataset to use.")
    parser.add_argument("--dataset_path", type=str, default="./ShareGPT_V3_unfiltered_cleaned_split.json", help="Path to the dataset JSON file.")
    parser.add_argument("--vllm_start_timeout", type=int, default=600, help="Timeout in seconds for starting the VLLM server.")

    args = parser.parse_args()

    # Pass parsed arguments to the main function
    main(args)