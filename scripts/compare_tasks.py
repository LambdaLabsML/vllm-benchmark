import json
import os
import matplotlib.pyplot as plt

def plot_benchmark_data(num_prompts, gpu_name, base_task_names, result_path, render_path):
    colors = plt.cm.get_cmap('tab10', len(base_task_names))  # Use a colormap to get distinct colors
    
    plt.figure(figsize=(10, 6))
    
    for idx, task_name in enumerate(base_task_names):
        list_output_throughput = []
        list_median_itl_ms = []
        
        for num_prompt in num_prompts:
            # Construct the file path
            file_path = f"{result_path}/prompt_{num_prompt}/{gpu_name}/{task_name}.json"
            
            # Check if the file exists
            if os.path.exists(file_path):
                # Read the JSON file
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
                    # Extract the required metrics
                    output_throughput = data.get('output_throughput')
                    median_itl_ms = data.get('median_itl_ms')
                    
                    # Append the metrics to the lists
                    if output_throughput is not None and median_itl_ms is not None:
                        list_output_throughput.append(output_throughput)
                        list_median_itl_ms.append(median_itl_ms)
                    else:
                        print(f"Warning: Missing metrics in {file_path}")
            else:
                print(f"Warning: File {file_path} does not exist")

        # Plot each task's data with a different color
        plt.plot(list_median_itl_ms, list_output_throughput, marker='o', color=colors(idx), label=task_name)
    
    # model_name = base_task_names[0].split("_tp")[0]
    plt.xlabel('Median ITL (ms)')
    plt.ylabel('Output Throughput')
    # plt.title(f'Benchmark Results for {gpu_name}_{model_name}')
    plt.title(f'Benchmark Results for {gpu_name}')
    plt.grid(True)
    plt.legend(title="Task Names")
    
    # Save the figure
    # plt.savefig(f'{render_path}/{gpu_name}_{model_name}.png')
    plt.savefig(f'{render_path}/{gpu_name}.png')

# Example usage
num_prompts = [10, 20, 40, 80, 160, 320]  # Replace with your actual values
# gpu_name = "1xA100-80GB-SXM"  # Replace with your actual GPU name
# base_task_names = [
#     "Meta-Llama-3.1-8B-FP8_tp1_len2000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len2000",
#     "Mistral-7B-Instruct-v0.3_tp1_len2000",
#     "Mixtral-8x7B-Instruct-v0.1_tp1_len2000",
#     "Mixtral-8x22B-Instruct-v0.1_tp1_len2000",
#     "Mistral-Nemo-Instruct-2407_tp1_len2000",
#     "Mistral-Large-Instruct-2407_tp1_len2000",
#     "Hermes-3-Llama-3.1-405B-FP8_tp1_len2000",
# ]  # Replace with your actual task names

gpu_name = "2xA100-80GB-SXM"  # Replace with your actual GPU name
base_task_names = [
    "Meta-Llama-3.1-8B-FP8_tp2_len2000",
    "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len2000",
    "Mistral-7B-Instruct-v0.3_tp2_len2000",
    "Mixtral-8x7B-Instruct-v0.1_tp2_len2000",
    "Mixtral-8x22B-Instruct-v0.1_tp2_len2000",
    "Mistral-Nemo-Instruct-2407_tp2_len2000",
    "Mistral-Large-Instruct-2407_tp2_len2000",
    "Hermes-3-Llama-3.1-405B-FP8_tp2_len2000",
]  # Replace with your actual task names


# gpu_name = "1xA100-80GB-SXM"  # Replace with your actual GPU name
# base_task_names = [
#     "Meta-Llama-3.1-8B-FP8_tp1_len2000",
#     "Meta-Llama-3.1-8B-FP8_tp1_len4000",
#     "Meta-Llama-3.1-8B-FP8_tp1_len8000",
#     "Meta-Llama-3.1-8B-FP8_tp1_len16000",
#     "Meta-Llama-3.1-8B-FP8_tp1_len32000",
#     "Meta-Llama-3.1-8B-FP8_tp1_len64000",
#     "Meta-Llama-3.1-8B-FP8_tp1_len128000",
# ]  # Replace with your actual task names

# gpu_name = "1xA100-80GB-SXM"  # Replace with your actual GPU name
# base_task_names = [
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len2000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len4000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len8000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len16000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len32000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len64000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp1_len128000",
# ]  # Replace with your actual task names

# gpu_name = "2xA100-80GB-SXM"  # Replace with your actual GPU name
# base_task_names = [
#     "Meta-Llama-3.1-8B-FP8_tp2_len2000",
#     "Meta-Llama-3.1-8B-FP8_tp2_len4000",
#     "Meta-Llama-3.1-8B-FP8_tp2_len8000",
#     "Meta-Llama-3.1-8B-FP8_tp2_len16000",
#     "Meta-Llama-3.1-8B-FP8_tp2_len32000",
#     "Meta-Llama-3.1-8B-FP8_tp2_len64000",
#     "Meta-Llama-3.1-8B-FP8_tp2_len128000",
# ]  # Replace with your actual task names

# gpu_name = "2xA100-80GB-SXM"  # Replace with your actual GPU name
# base_task_names = [
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len2000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len4000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len8000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len16000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len32000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len64000",
#     "Meta-Llama-3.1-70B-Instruct-FP8_tp2_len128000",
# ]  # Replace with your actual task names

result_path = "./results"
render_path = "./renders"

plot_benchmark_data(num_prompts, gpu_name, base_task_names, result_path, render_path)
