from transformers import AutoModel, AutoTokenizer

# list_models = [
#   "neuralmagic/Meta-Llama-3.1-8B-FP8",
#   "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
#   "mistralai/Mistral-7B-Instruct-v0.3",
#   "mistralai/Mixtral-8x7B-Instruct-v0.1",
#   "mistralai/Mixtral-8x22B-Instruct-v0.1",
#   "mistralai/Mistral-Nemo-Instruct-2407",
#   "mistralai/Mistral-Large-Instruct-2407",
#   "NousResearch/Hermes-3-Llama-3.1-405B-FP8"
# ]

# list_models = [
#   "neuralmagic/Meta-Llama-3.1-8B-FP8",
#   "mistralai/Mistral-7B-Instruct-v0.3"
# ]

list_models = [
  "meta-llama/Meta-Llama-3-8B-Instruct",
  "meta-llama/Meta-Llama-3-70B-Instruct",
  "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
  "mistralai/Mixtral-8x7B-Instruct-v0.1",
  "mistralai/Mixtral-8x22B-Instruct-v0.1",
  "mistralai/Mistral-Nemo-Instruct-2407",
  "mistralai/Mistral-Large-Instruct-2407",
  "NousResearch/Hermes-3-Llama-3.1-405B-FP8"
]


for model_name in list_models:
    try:
      print("--------------------------------------------------------------------------------")
      print(f"Caching model {model_name} ...")
      print("--------------------------------------------------------------------------------")
      # This will download and cache the model and tokenizer
      model = AutoModel.from_pretrained(model_name)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error caching model {model_name}\n")