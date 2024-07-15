from typing import List
import numpy as np
import math, pickle
import torch
import torch.nn as nn


class PE(OperationalUnit):
    ## The class constructor
    # @param i_prec:    The input precision.
    # @param w_prec:    The weight precision.
    # @param dp_size:   The dot-product size of the PE.
    # @param energy:    The energy cost of PE.
    # @param area:      The area of PE.
    def __init__(
            self, 
            i_prec: int=16, 
            w_prec: int=8, 
            dp_size: int=1,
            energy: float=0, 
            area: float=0
    ):
        assert len(input_precision) != 2, "ERROR! You must provide precision for 2 input operands of a bit-serial PE."
        assert energy_cost == 0, "ERROR! You must provide the energy cost of a PE."

        self.i_prec  = i_prec
        self.w_prec  = w_prec
        self.dp_size = dp_size
        self.energy  = energy
        self.area    = area


class Accelerator:
    ### Global variable
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, 
                 i_prec: int=16, 
                 w_prec: int=8, 
                 pe_dp_size: int=1,
                 pe_energy: float=0, 
                 pe_area: float=0,
                 pe_array_dim: List[int],
                 model_name: str):
        self.model_name     = model_name
        self.pe             = PE(i_prec, w_prec, pe_dp_size, pe_energy, pe_area)
        self.pe_array_dim   = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        self.total_pe_count = np.prod(pe_array_dim)

        self._init_model_profiler(model_name)
    
    def _init_model_profiler(self, model_name, context_length: int=256):
        model_name_dict = {
            "facebook/opt-1.3b": "opt_1_point_3", 
            "facebook/opt-6.7b": "opt_6_point_7", 
            "microsoft/phi-2": "phi_2",
            "01-ai/Yi-6B": "yi_6",
            "meta-llama/Llama-2-7b-hf": "llama_2_7", 
            "meta-llama/Llama-2-13b-hf": "llama_2_13", 
            "meta-llama/Meta-Llama-3-8B": "llama_3_8", 
        }
        file_path = f'./model_shape_config/{model_name_dict[model_name]}.pickle'
        with open(file_path, 'rb') as f:
            model_config, layer_config = pickle.load(f)
        
        ########## FFN Dimension ##########
        weight_dim = {}
        input_dim  = {}
        output_dim = {}
        for name, shape in layer_config.items():
            weight_dim[name] = shape
            input_dim[name]  = [context_length, shape[1]]
            output_dim[name] = [context_length, shape[0]]

        ########## Attention Dimension ##########
        num_hidden_layers   = model_config['num_hidden_layers']
        hidden_size         = model_config['hidden_size']
        num_attention_heads = model_config['num_attention_heads']
        if 'num_key_value_heads' in model_config.keys():
            num_key_value_heads = model_config['num_key_value_heads']
        else:
            num_key_value_heads = num_attention_heads
        
        for l_idx in range(num_hidden_layers):
            op_name = f'model.layers.{l_idx}.self_attn.attn_qk'
            weight_dim[op_name] = [context_length, hidden_size] # query dimension
            input_dim[op_name]  = [context_length, hidden_size / num_attention_heads * num_key_value_heads] # key dimension
            output_dim[op_name] = [num_attention_heads * context_length, context_length] # score dimension

            op_name = f'model.layers.{l_idx}.self_attn.attn_v'
            weight_dim[op_name] = [num_attention_heads * context_length, context_length] # score dimension
            input_dim[op_name]  = [context_length, hidden_size / num_attention_heads * num_key_value_heads] # value dimension
            output_dim[op_name] = [context_length, hidden_size] # output dimension

        self.weight_dim = weight_dim
        self.input_dim  = input_dim
        self.output_dim = output_dim
        

    def _init_mem(self):
        raise NotImplementedError('ERROR! No implementation of function _init_mem()')
