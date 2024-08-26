import json
import os
import matplotlib.pyplot as plt

def plot_benchmark_data(num_prompts, gpu_names, base_task_name, result_path, render_path):
    colors = plt.cm.get_cmap('tab10', len(gpu_names))  # Use a colormap to get distinct colors
    
    plt.figure(figsize=(10, 6))
    
    for idx, gpu_name in enumerate(gpu_names):
        list_output_throughput = []
        list_median_itl_ms = []

        # Extract the "tp" value from the gpu_name
        tp_value = gpu_name.split('x')[0]  # Get the number before 'x'

        # Modify the task_name to include the correct "tp" value
        task_name = base_task_name.replace("_tp", f"_tp{tp_value}")
        
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

        if idx == 0:
            list_output_throughput = [t * 2 for t in list_output_throughput[1:]]
            list_median_itl_ms = list_median_itl_ms[1:]
            # label = "DP_" + str(2*int(gpu_name.split("x")[0])) + "x" + gpu_name.split("x")[1]
            label = "DP_2x_TP_" + gpu_name
        else:
            list_output_throughput = list_output_throughput[:-1]
            list_median_itl_ms = list_median_itl_ms[:-1]
            label = "TP_" + gpu_name

        # Plot each GPU's data with a different color
        plt.plot(list_median_itl_ms, list_output_throughput, marker='o', color=colors(idx), label=label)
    
    # Modify the base_task_name to remove the "_tp" part for the title
    title_task_name = "scale_" + base_task_name.replace("_tp", "").replace(f"_tp{tp_value}", "")
    
    plt.xlabel('Median ITL (ms)')
    plt.ylabel('Output Throughput')
    plt.title(f'Scaling Test Results for {title_task_name}')
    plt.grid(True)
    plt.legend(title="GPU Names")
    
    # Save the figure
    plt.savefig(f'{render_path}/{title_task_name}.png')

# Example usage
num_prompts = [10, 20, 40, 80, 160, 320, 640, 1280]  # Replace with your actual values
# gpu_names = [
#     "1xH100-80GB-SXM",
#     "2xH100-80GB-SXM",
# ]  # Replace with your actual GPU names

# base_task_names = [
#     "Mistral-7B-Instruct-v0.3_tp_len2000",
# ]


# gpu_names = [
#     "2xH100-80GB-SXM",
#     "4xH100-80GB-SXM",
# ]  # Replace with your actual GPU names

# base_task_names = [
#     "Mixtral-8x7B-Instruct-v0.1_tp_len2000",
# ]

gpu_names = [
    "4xH100-80GB-SXM",
    "8xH100-80GB-SXM",
]  # Replace with your actual GPU names

base_task_names = [
    "Mixtral-8x22B-Instruct-v0.1_tp_len2000",
]


result_path = "./results"
render_path = "./renders"

for base_task_name in base_task_names:
    plot_benchmark_data(num_prompts, gpu_names, base_task_name, result_path, render_path)
