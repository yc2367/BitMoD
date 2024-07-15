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
     "--group_size", "-gs", type=int, default=128, help="Quantization group size"
)
args = parser.parse_args()
model_str = args.model
group_size = args.group_size

torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
model_config = AutoConfig.from_pretrained(model_str).to_dict()

tensor_outlier_config  = {}
channel_outlier_config = {}
group_outlier_config   = {}
tensor_range_config    = {}
channel_range_config   = {}
group_range_config     = {}
model_config = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        wg = m.weight.to(torch.float32).reshape(-1, group_size)

        group_max = torch.amax(wg, dim=-1, keepdim=True)
        group_min = torch.amin(wg, dim=-1, keepdim=True)
        group_max[group_max == 0] = 1e-8
        group_min[group_min == 0] = 1e-8
        is_more_neg   = torch.lt(group_max.abs(), group_min.abs()) & torch.lt(group_min, 0) & torch.gt(group_max, 0)

        asymmetry_pos = group_max.abs() / group_min.abs()
        asymmetry_neg = group_min.abs() / group_max.abs()
        asymmetry     = torch.zeros_like(asymmetry_pos)
        asymmetry[is_more_neg] = asymmetry_neg[is_more_neg]
        asymmetry[~is_more_neg] = asymmetry_pos[~is_more_neg]
        
        asymmetricity = torch.mean(asymmetry).item()

        print(f'Module name:   {n}')
        print(f'Module shape:  {m.weight.shape}')
        print(f'Asymmetricity: {asymmetricity}')

        print('\n')

'''
model_name_dict = {
    "facebook/opt-1.3b": "opt_1_point_3", 
    "facebook/opt-6.7b": "opt_6_point_7", 
    "microsoft/phi-2": "phi_2",
    "01-ai/Yi-6B": "yi_6",
    "meta-llama/Llama-2-7b-hf": "llama_2_7", 
    "meta-llama/Llama-2-13b-hf": "llama_2_13", 
    "meta-llama/Meta-Llama-3-8B": "llama_3_8", 
}
base_path = '/home/yc2367/llm/llm_profile/model_asymmetry_config'
file_path = f'{base_path}/{model_name_dict[model_str]}.pickle'
with open(file_path, 'wb') as f:
    pickle.dump(model_config, f)
'''

