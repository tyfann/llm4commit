# Use a pipeline as a high-level helper
from transformers import pipeline
import json
import os

os.environ['HF_HOME'] = './'
with open('../data/final_preprocessed_data/js_baseline_test_data_300.json') as f:
    data = json.load(f)

pipe = pipeline("text2text-generation", model="google/flan-t5-small")



diffs = []
for commit in data:
    diff = commit['diff']
    diffs.append(diff)

prompt = f"""
The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly.
{diffs[0]}
According to the diff, the commit message should be:
"""
print(pipe(prompt))