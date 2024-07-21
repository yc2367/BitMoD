from huggingface_hub import login
login(token="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")

import torch
torch.set_grad_enabled(False)
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import time, argparse, os
from typing import Optional
import pickle


def quant_datatype(w_fp16, datatype: str="", group_size: Optional[int]=None):
    DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT

    assert datatype in DATATYPE_MAPPING, "unexpected data type."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)

    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    tensor = w_fp16_new / scale_fp

    q_tensor = torch.zeros_like(tensor)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(tensor <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(tensor > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < tensor) & (tensor <= mid_value[i]), data, 0)
    w_fp16_new = q_tensor * scale_fp 

    del tensor
    del q_tensor

    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


def search_datatype(w_fp16, group_size: Optional[int]=None):
    datatype_list = ['fp3_sp_pos', 'fp3_sp_neg', 'fp3_sr_pos', 'fp3_sr_neg']

    K, C = w_fp16.size() # output channel, input channel
    if (group_size is None) or (group_size <= 0):
        group_size = C
    NUM_GROUP = C // group_size
    w_fp16 = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    q_tensor = torch.zeros_like(w_fp16)
    
    error = torch.full([K, NUM_GROUP], 1e3, dtype=torch.float16, device=w_fp16.device)
    for datatype in datatype_list:
        w_fp16_new = quant_datatype(w_fp16, datatype=datatype, group_size=None)
        quant_error = (w_fp16_new - w_fp16).abs().pow(2).mean(-1)
        update_mask = torch.lt(quant_error, error)
        error[update_mask] = quant_error[update_mask]
        q_tensor[update_mask] = w_fp16_new[update_mask]

        del w_fp16_new
        del quant_error
        del update_mask
        
    return q_tensor.reshape(K, C)
    


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
     "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
)
parser.add_argument(
     "--group_size", "-gs", type=int, default=128, help="Quantization group size"
)
parser.add_argument(
     "--outlier", "-ol", type=float, default=6, help="The special value added to FP3"
)
args = parser.parse_args()
model_str = args.model
group_size = args.group_size
outlier = args.outlier

FP3_SP_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_SP_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SR_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0] + [abs(outlier)]
FP3_SR_NEG = [-abs(outlier)] + [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

DATATYPE_MAPPING_3_BIT = {
    'fp3_sp_pos': FP3_SP_POS, 'fp3_sp_neg': FP3_SP_NEG, 
    'fp3_sr_pos': FP3_SR_POS, 'fp3_sr_neg': FP3_SR_NEG, 
}

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
model_size_config = {}
model_error_config = {}
#################### Quant Model ####################
print(f"Applying quantization with bits: {3}, mixed datatypes, group size: {group_size}, outlier value: {outlier}")
time.sleep(2)
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and ('head' not in n):
        #print(m.qweight._data.unique())
        print(n)
        model_size_config[n] = torch.numel(m.weight)
        
        w = m.weight.data
        wq = search_datatype(w, group_size=group_size)
        model_error_config[n] = (wq - w).pow(2).sum().item()

        print(model_error_config[n] / model_size_config[n])
        print('\n')


model_config = [model_size_config, model_error_config]
model_name_dict = {
    "facebook/opt-1.3b": "opt_1_point_3", 
    "facebook/opt-6.7b": "opt_6_point_7", 
    "microsoft/phi-2": "phi_2",
    "01-ai/Yi-6B": "yi_6",
    "meta-llama/Llama-2-7b-hf": "llama_2_7", 
    "meta-llama/Llama-2-13b-hf": "llama_2_13", 
    "meta-llama/Meta-Llama-3-8B": "llama_3_8", 
}
base_path = '/home/yc2367/llm/BitMoD/llm_profile/model_fp3_cv_profile'
file_path = f'{base_path}/{model_name_dict[model_str]}_outlier_{int(outlier)}.pickle'
with open(file_path, 'wb') as f:
    pickle.dump(model_config, f)


