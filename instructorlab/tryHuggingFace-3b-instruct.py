# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="ibm-granite/granite-20b-code-base")


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-20b-code-base")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-20b-code-base")