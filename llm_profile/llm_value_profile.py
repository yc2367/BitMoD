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
        w = m.weight

        tensor_max   = torch.amax(w)
        tensor_min   = torch.amin(w)
        tensor_mean  = torch.mean(w)
        tensor_std   = torch.std(w)
        tensor_norm  = (w - tensor_mean) / tensor_std
        norm_tensor_range = ((tensor_max - tensor_min) / tensor_std).item()
        norm_tensor_max = torch.amax(tensor_norm.abs()).item()

        channel_max   = torch.amax(w, dim=-1, keepdim=True)
        channel_min   = torch.amin(w, dim=-1, keepdim=True)
        channel_mean  = torch.mean(w, dim=-1, keepdim=True)
        channel_std   = torch.std(w, dim=-1, keepdim=True)
        channel_norm  = (w - channel_mean) / channel_std
        norm_channel_range = torch.mean((channel_max - channel_min) / channel_std).item()
        norm_channel_max = torch.amax(channel_norm.abs(), dim=-1).mean().item()

        wg = m.weight.reshape(-1, group_size)

        group_max  = torch.amax(wg, dim=-1, keepdim=True)
        group_min  = torch.amin(wg, dim=-1, keepdim=True)
        group_mean = torch.mean(wg, dim=-1, keepdim=True)
        group_std  = torch.std(wg, dim=-1, keepdim=True)
        group_norm = (wg - group_mean) / group_std
        norm_group_range = torch.mean((group_max - group_min) / group_std).item()
        norm_group_max = torch.amax(group_norm.abs(), dim=-1).mean().item()

        tensor_outlier_config[n]  = norm_tensor_max
        channel_outlier_config[n] = norm_channel_max
        group_outlier_config[n]   = norm_group_max
        tensor_range_config[n]    = norm_tensor_range
        channel_range_config[n]   = norm_channel_range
        group_range_config[n]     = norm_group_range
        print(f'Module name:  {n}')
        print(f'Module shape: {m.weight.shape}')
        print(f'Normalized tensor max:     {norm_tensor_max}')
        print(f'Normalized channel max:    {norm_channel_max}')
        print(f'Normalized group max:      {norm_group_max}')
        print()
        print(f'Normalized tensor range:   {norm_tensor_range}')
        print(f'Normalized channel range:  {norm_channel_range}')
        print(f'Normalized group range:    {norm_group_range}')

        print('\n')

model_config['tensor_outlier_config']   = tensor_outlier_config
model_config['channel_outlier_config']  = channel_outlier_config
model_config['group_outlier_config']    = group_outlier_config
model_config['tensor_range_config']     = tensor_range_config
model_config['channel_range_config']    = channel_range_config
model_config['group_range_config']      = group_range_config
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
base_path = '/home/yc2367/llm/BitMoD/llm_profile/model_value_config'
file_path = f'{base_path}/{model_name_dict[model_str]}.pickle'
with open(file_path, 'wb') as f:
    pickle.dump(model_config, f)


