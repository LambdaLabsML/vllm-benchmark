import json
import csv

path_results = "./results"
list_gpus = [
    "1xA100-80GB-SXM",
    "2xA100-80GB-SXM",
    "4xA100-80GB-SXM",
    "8xA100-80GB-SXM",
    "1xH100-80GB-SXM",
    "2xH100-80GB-SXM",
    "4xH100-80GB-SXM",
    "8xH100-80GB-SXM"
]

num_prompt = 320

max_model_len = 2000

list_models = [
    "neuralmagic/Meta-Llama-3.1-8B-FP8",
    "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
    "NousResearch/Hermes-3-Llama-3.1-405B-FP8",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-Large-Instruct-2407"
]

def check_for_reason_key(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    if "reason" in data:
        return "N/A"
    else:
        # Extract the required metrics
        output_throughput = data.get('output_throughput')
        median_itl_ms = data.get('median_itl_ms')
        return f"{round(output_throughput, 2)}/{round(median_itl_ms, 2)}"
    


# Initialize a dictionary to hold the data
results = {gpu: {model: "N/A" for model in list_models} for gpu in list_gpus}

# Populate the results dictionary with max_model_len values
for gpu in list_gpus:
    for model in list_models:
        model_name = model.split("/")[1]
        num_gpu = gpu.split("x")[0]
        file_path = f"{path_results}/prompt_{num_prompt}/{gpu}/{model_name}_tp{num_gpu}_len{max_model_len}.json"
        results[gpu][model] = check_for_reason_key(file_path)
        print(results[gpu][model])

# Write the results to a CSV file
with open('results_sweetspot.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow([''] + list_models)
    # Write the rows
    for gpu in list_gpus:
        row = [gpu] + [results[gpu][model] for model in list_models]
        writer.writerow(row)

print("CSV file 'results_sweetspot.csv' created successfully.")
