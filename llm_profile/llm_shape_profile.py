from huggingface_hub import login
login(token="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")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import time, argparse, os
from typing import Optional
import pickle


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
     "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
)
parser.add_argument(
     "--dump_result", action='store_true', help="Whether dump the model config"
)
args = parser.parse_args()
model_str = args.model
dump_result = args.dump_result

torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
model_config = AutoConfig.from_pretrained(model_str).to_dict()

layer_config = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        layer_config[n] = list(m.weight.shape)
        print(f'Module name:  {n}')
        print(f'Module shape: {m.weight.shape}')
        print()

print('\n\n')

model_name_dict = {
    "facebook/opt-1.3b": "opt_1_point_3", 
    "facebook/opt-6.7b": "opt_6_point_7", 
    "microsoft/phi-2": "phi_2",
    "01-ai/Yi-6B": "yi_6",
    "meta-llama/Llama-2-7b-hf": "llama_2_7", 
    "meta-llama/Llama-2-13b-hf": "llama_2_13", 
    "meta-llama/Meta-Llama-3-8B": "llama_3_8", 
}
base_path = '/home/yc2367/llm/BitMoD/llm_profile/model_shape_config'
file_path = f'{base_path}/{model_name_dict[model_str]}.pickle'
if dump_result:
    with open(file_path, 'wb') as f:
        pickle.dump((model_config, layer_config), f)
