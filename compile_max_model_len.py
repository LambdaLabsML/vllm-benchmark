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

num_prompt = 10

list_max_model_len = [2000, 4000, 8000, 16000, 32000, 64000, 128000]

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
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        if "reason" in data:
            return 1
        else:
            return 0
    
    except json.JSONDecodeError:
        print("Error decoding JSON file.")
        return 1
    except FileNotFoundError:
        print("File not found.")
        return 1

# Initialize a dictionary to hold the data
results = {gpu: {model: 0 for model in list_models} for gpu in list_gpus}

# Populate the results dictionary with max_model_len values
for gpu in list_gpus:
    for model in list_models:
        for max_model_len in reversed(list_max_model_len):
            model_name = model.split("/")[1]
            num_gpu = gpu.split("x")[0]
            file_path = f"{path_results}/prompt_{num_prompt}/{gpu}/{model_name}_tp{num_gpu}_len{max_model_len}.json"
            
            if not check_for_reason_key(file_path):
                results[gpu][model] = max_model_len
                break

# Write the results to a CSV file
with open('results_max_model_len.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow([''] + list_models)
    # Write the rows
    for gpu in list_gpus:
        row = [gpu] + [results[gpu][model] for model in list_models]
        writer.writerow(row)

print("CSV file 'results_max_model_len.csv' created successfully.")
