import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import datetime
import json

# 读取JSON文件
with open('data/msg_nngen_nmt_codebert_chatgpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = []

for item in data:
    diff = "Here is a code diff: \n" +  item['diff'] + "\nGenerate the commit message based on git diff(with 30 words):"
    prompts.append(diff)
 
MODEL_NAME = "TheBloke/Llama-2-7B-GPTQ"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, cache_dir="models/")
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda", cache_dir="models/")
 
generation_config = GenerationConfig.from_pretrained(MODEL_NAME, cache_dir="models/")
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15
 
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.5})

counter = 0
for prompt in prompts:
    if counter == 1:
        break
    try:
        result = llm(prompt)
        print(result)
        print(datetime.datetime.now())
    except Exception as e:
        print(e)
        break
    counter += 1
# result = llm(
#     """
#     The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. 
# \ndiff --git a/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java b/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java\nindex e8d986667..ce243ca1f 100644\n--- a/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java\n+++ b/src/gwt/src/org/rstudio/studio/client/workbench/views/source/editors/text/TextEditingTargetRenameHelper.java\n@@ -44,7 +44,7 @@ public class TextEditingTargetRenameHelper\n       \n       editor_.setCursorPosition(position);\n       \n-      // Validate that we're looking at an R identifier (TODO: refactor strings?)\n+      // Validate that we're looking at an R identifier\n       String targetValue = cursor.currentValue();\n       String targetType = cursor.currentType();\n       \n@@ -104,7 +104,6 @@ public class TextEditingTargetRenameHelper\n       }\n       \n       // Otherwise, just rename the variable within the current scope.\n-      // TODO: Do we need to look into parent scopes?\n       return renameVariablesInScope(\n             editor_.getCurrentScope(),\n             targetValue,\n
# \nAccording to the diff, the commit message should be:
#     """
# )
# print(result)
