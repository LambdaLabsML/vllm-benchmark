import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_benchmark_data(num_prompts, gpu_names, base_task_name, result_path, render_path):
    # Define color maps for A100 and H100 groups
    a100_colors = LinearSegmentedColormap.from_list("a100", ["#ADD8E6", "#0000FF"], N=8)  # Light to dark blue for A100
    h100_colors = LinearSegmentedColormap.from_list("h100", ["#FFB6C1", "#FF0000"], N=8)  # Light to dark red for H100
    
    plt.figure(figsize=(10, 6))
    
    for idx, gpu_name in enumerate(gpu_names):
        list_output_throughput = []
        list_median_itl_ms = []

        # Extract the number of GPUs from gpu_name (e.g., "8x" gives us 8)
        num_gpus = int(gpu_name.split('x')[0])

        # Modify the task_name to include the correct "tp" value based on number of GPUs
        task_name = base_task_name.replace("_tp", f"_tp{num_gpus}")
        
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
        
        # Assign color based on whether the GPU is A100 or H100, adjusting the shade based on the number of GPUs
        if "A100" in gpu_name:
            color = a100_colors(num_gpus - 1)  # Use a lighter color for fewer GPUs and darker for more
        else:
            color = h100_colors(num_gpus - 1)

        # Plot each GPU's data with a different color and marker
        plt.plot(list_median_itl_ms, list_output_throughput, marker='o', color=color, label=gpu_name)
    
    # Modify the base_task_name to remove the "_tp" part for the title
    title_task_name = base_task_name.replace("_tp", "").replace(f"_tp{num_gpus}", "")
    
    plt.xlabel('Median ITL (ms)')
    plt.ylabel('Output Throughput (tokens/s)')
    plt.title(f'Benchmark Results for {title_task_name}')
    plt.grid(True)
    plt.legend(title="GPU Names", loc="best")
    
    # Save the figure
    plt.savefig(f'{render_path}/{title_task_name}.png')

# Example usage
num_prompts = [10, 20, 40, 80, 160, 320]
gpu_names = [
    "1xA100-80GB-SXM", "2xA100-80GB-SXM", "4xA100-80GB-SXM", "8xA100-80GB-SXM",
    "1xH100-80GB-SXM", "2xH100-80GB-SXM", "4xH100-80GB-SXM", "8xH100-80GB-SXM"
]
base_task_names = [
    "Meta-Llama-3.1-8B-FP8_tp_len2000",
    "Meta-Llama-3.1-70B-Instruct-FP8_tp_len2000",
    "Mistral-7B-Instruct-v0.3_tp_len2000",
    "Mixtral-8x7B-Instruct-v0.1_tp_len2000",
    "Mixtral-8x22B-Instruct-v0.1_tp_len2000",
    "Mistral-Nemo-Instruct-2407_tp_len2000",
    "Mistral-Large-Instruct-2407_tp_len2000",
    "Hermes-3-Llama-3.1-405B-FP8_tp_len2000",
]

result_path = "./results_v0"
render_path = "./renders_v0"

for base_task_name in base_task_names:
    plot_benchmark_data(num_prompts, gpu_names, base_task_name, result_path, render_path)
