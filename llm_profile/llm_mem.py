import torch
import numpy as np
import time, argparse, os
from typing import Optional, Dict, List
import pickle


def calc_mem(model_config: Dict, layer_config: Dict, context_length: int=512, gen_length: int=1):
    layer_names         = layer_config.keys()
    num_hidden_layers   = model_config['num_hidden_layers']
    hidden_size         = model_config['hidden_size']
    num_attention_heads = model_config['num_attention_heads']
    if 'num_key_value_heads' in model_config.keys():
        num_key_value_heads = model_config['num_key_value_heads']
    else:
        num_key_value_heads = num_attention_heads

    weight_mem = 0
    act_mem = 0

    ################ Prefill Stage ################
    # FFN memory
    for name in layer_names:
        weight_shape = layer_config[name]
        weight_mem += np.prod(weight_shape)
        act_mem += (context_length * weight_shape[0])
    
    # Attention memory
    for _ in range(num_hidden_layers):
        act_mem += ((context_length ** 2) + (context_length * hidden_size))
    
    ################ Generation Stage ################
    for gen_idx in range(gen_length):
        # FFN memory
        for name in layer_names:
            weight_shape = layer_config[name]
            weight_mem += np.prod(weight_shape)
            act_mem += (1 * weight_shape[0])
        
        # Attention memory
        for _ in range(num_hidden_layers):
            kv_cache_mem = (context_length + gen_idx) * hidden_size * 2
            attn_mem = ((context_length + gen_idx) + (1 * hidden_size))
            act_mem += (kv_cache_mem + attn_mem)

    kv_cache_mem_total = (context_length + gen_length) * hidden_size * 2 * num_hidden_layers
    return weight_mem, act_mem, kv_cache_mem_total


model_list = ["facebook/opt-1.3b", "microsoft/phi-2", "01-ai/Yi-6B", "facebook/opt-6.7b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
model_name_dict = {
    "facebook/opt-1.3b": "opt_1_point_3", 
    "facebook/opt-6.7b": "opt_6_point_7", 
    "microsoft/phi-2": "phi_2",
    "01-ai/Yi-6B": "yi_6",
    "meta-llama/Llama-2-7b-hf": "llama_2_7", 
    "meta-llama/Llama-2-13b-hf": "llama_2_13", 
    "meta-llama/Meta-Llama-3-8B": "llama_3_8", 
}
layer_name_keys = {
    "facebook/opt-1.3b": ["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"], 
    "facebook/opt-6.7b": ["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"], 
    "microsoft/phi-2": ["k_proj", "v_proj", "q_proj", "dense", "fc1", "fc2"], 
    "01-ai/Yi-6B": ["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    "meta-llama/Llama-2-7b-hf": ["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    "meta-llama/Llama-2-13b-hf": ["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    "meta-llama/Meta-Llama-3-8B": ["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
}
base_path = '/home/yc2367/llm/BitMoD/llm_profile/model_config'


for model_str in model_list:
    file_path = f'{base_path}/{model_name_dict[model_str]}.pickle'
    with open(file_path, 'rb') as f:
        model_config, layer_config = pickle.load(f)
    
    weight_mem, act_mem, kv_cache_mem = calc_mem(model_config, layer_config, context_length=256, gen_length=10)

    print(model_str)
    print(f'Weight memory:     {weight_mem * 2 / 1024**3} GB')
    print(f'Activation memory: {act_mem * 2 / 1024**3} GB')
    print(f'KV Cache memory:   {kv_cache_mem * 2 / 1024**3} GB')
    print('\n')
    
