# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b", cache_dir='./')
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b", cache_dir='./')

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**input)
print(tokenizer.decode(tokens[0]))