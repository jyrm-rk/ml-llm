from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Cognitive-Lab/Kannada-Instruct-dataset")

print(len(ds))