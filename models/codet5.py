# Use a pipeline as a high-level helper
from transformers import pipeline
import json
import os
import random

os.environ['HF_HOME'] = './'
with open('../data/chronicle/chronicle_data.json') as f:
    data = json.load(f)

pipe = pipeline("text2text-generation", model="JetBrains-Research/cmg-codet5-with-history")



diffs = []
for commit in data:
    diff = commit['diff']
    diffs.append(diff)

prompt = f"""
The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly.
{diffs[0]}
According to the diff, the commit message should be:
"""
print(pipe('prompt'))