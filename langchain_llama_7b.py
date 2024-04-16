from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline
import json

model_id = 'meta-llama/Llama-2-7b-chat-hf'
# CUDA_RUNTIME_LIBS: list = ['libcudart.so', 'libcudart.so.12', 'libcudart.so.12.2.140'] 

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    cache_dir='models/'
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth,
    cache_dir='models/'
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    cache_dir='models/'
)

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.5,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


llm = HuggingFacePipeline(pipeline=generate_text)

with open('data/msg_nngen_nmt_codebert_chatgpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = []

with open('llama2-7b-output2.txt', 'w', encoding='UTF-8') as file:
    for index, item in enumerate(data):
        if index < 1641: 
            continue
        try:
            message = llm(prompt=f"Here is a code diff: \n{item['diff']}\nGenerate the commit message based on git diff (with 30 words):")
            file.write(f"index: {index}\n")
            file.write(message + '\n')
        except Exception as e:
            file.write(str(e) + '\n')
            break