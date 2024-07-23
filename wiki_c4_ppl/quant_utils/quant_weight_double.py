import torch
import torch.nn as nn
from typing import Optional
import time

#################################  3-bit Datatypes  #################################
INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SP_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_SP_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SM_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 5.0]
FP3_SM_NEG = [-5.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_SR_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
FP3_SR_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


#################################  4-bit Datatypes  #################################
INT4 = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FLINT4 = [-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]

FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_SP_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
FP4_SP_NEG = [-12.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

FP4_SM_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_SM_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

FP4_SR_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_SR_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

APOT4 = [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
APOT4_SP_POS = [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
APOT4_SP_NEG = [-10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

APOT4_SR_POS = [-10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 16.0]
APOT4_SR_NEG = [-16.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0., 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]


#################################  5-bit Datatypes  #################################
INT5 = [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
FLINT5 = [-64.0, -32.0, -24.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 32.0, 64.0]
FP5_E2M2 = [-28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0]
FP5_E3M1 = [-192.0, -128.0, -96.0, -64.0, -48.0, -32.0, -24.0, -16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]


#################################  6-bit Datatypes  #################################
INT6 = [-31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]
FP6_E2M3 = [-60.0, -56.0, -52.0, -48.0, -44.0, -40.0, -36.0, -32.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0]
FP6_E3M2 = [-448.0, -384.0, -320.0, -256.0, -224.0, -192.0, -160.0, -128.0, -112.0, -96.0, -80.0, -64.0, -56.0, -48.0, -40.0, -32.0, -28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0, 320.0, 384.0, 448.0]


DATATYPE_MAPPING_3_BIT = {'int3': INT3, 'fp3': FP3, 
                        'fp3_sp_pos': FP3_SP_POS, 'fp3_sp_neg': FP3_SP_NEG, 
                        'fp3_sm_pos': FP3_SM_POS, 'fp3_sm_neg': FP3_SM_NEG, 
                        'fp3_sr_pos': FP3_SR_POS, 'fp3_sr_neg': FP3_SR_NEG, }

DATATYPE_MAPPING_4_BIT = {'int4': INT4, 'fp4': FP4_E2M1, 'flint4': FLINT4,
                        'fp4_sp_pos': FP4_SP_POS, 'fp4_sp_neg': FP4_SP_NEG, 
                        'fp4_sm_pos': FP4_SM_POS, 'fp4_sm_neg': FP4_SM_NEG, 
                        'fp4_sr_pos': FP4_SR_POS, 'fp4_sr_neg': FP4_SR_NEG, 
                        'apot4': APOT4,
                        'apot4_sp_pos': APOT4_SP_POS, 'apot4_sp_neg': APOT4_SP_NEG, 
                        'apot4_sr_pos': APOT4_SR_POS, 'apot4_sr_neg': APOT4_SR_NEG,                         
                        }

DATATYPE_MAPPING_5_BIT = {'int5': INT5, 'fp5': FP5_E2M2, 'flint5': FLINT5,
                        'fp5_e2m2': FP5_E2M2, 'fp5_e3m1': FP5_E3M1}

DATATYPE_MAPPING_6_BIT = {'int6': INT6, 'fp6': FP6_E2M3, 
                        'fp6_e2m3': FP6_E2M3, 'fp6_e3m2': FP6_E3M2}



@torch.no_grad()
def quant_int(w_fp16, wq_bits:int=4, group_size: Optional[int]=None, scale_bits: Optional[int]=None):
    """
        Symmetric INT quantization.
    """
    print(f"Symmetric INT quantization, w_bis: {wq_bits}, group_size: {group_size}, scale_bits: {scale_bits}")
    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = 2 ** (wq_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax

    assert torch.isnan(scale_fp).sum() == 0

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp), min=qmin, max=qmax)
    print(q_tensor.unique())
    print()

    ### Second-level scaling factor quantization
    scale_max = torch.amax(scale_fp.abs(), dim=1, keepdim=True)
    qscale_max = 2**scale_bits - 1
    scale_channel = scale_max / qscale_max
    q_scale = torch.clamp(torch.round(scale_fp / scale_channel), min=0, max=qscale_max)

    w_fp16_new = q_tensor * scale_channel * q_scale 
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_int_asym(w_fp16, wq_bits:int=4, group_size: Optional[int]=None, scale_bits: Optional[int]=None):
    """
        Asymmetric INT quantization.
    """
    print(f"Asymmetric INT quantization, w_bis: {wq_bits}, group_size: {group_size}, scale_bits: {scale_bits}")
    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = -(2 ** (wq_bits - 1))
    qmax = 2 ** (wq_bits - 1) - 1
    scale_fp = (rmax - rmin) / (qmax - qmin)
    zeropoint = torch.round(-rmin / scale_fp).to(torch.int8)

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zeropoint, min=0, max=2**(wq_bits)-1)
    print(q_tensor.unique())
    print()
    
    ### Second-level scaling factor quantization
    scale_max = torch.amax(scale_fp.abs(), dim=1, keepdim=True)
    qscale_max = 2**scale_bits - 1
    scale_channel = scale_max / qscale_max
    q_scale = torch.clamp(torch.round(scale_fp / scale_channel), min=0, max=qscale_max)

    w_fp16_new = (q_tensor - zeropoint) * scale_channel * q_scale 
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_datatype(w_fp16, wq_bits:int=4, datatype: str="", group_size: Optional[int]=None, scale_bits: Optional[int]=None):
    print(f"w_bis: {wq_bits}, group_size: {group_size}, scale_bits: {scale_bits}, scale_bits: {scale_bits}")
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit, 5-bit and 6-bit quantization, not {wq_bits}-bit")

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
    print(q_tensor.unique())
    print()

    ### Second-level scaling factor quantization
    scale_max = torch.amax(scale_fp.abs(), dim=1, keepdim=True)
    qscale_max = 2**scale_bits - 1
    scale_channel = scale_max / qscale_max
    q_scale = torch.clamp(torch.round(scale_fp / scale_channel), min=0, max=qscale_max)
    w_fp16_new = q_tensor * scale_channel * q_scale 

    del tensor
    del q_tensor

    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def search_datatype(w_fp16, wq_bits:int=4, datatype: str='mixed', group_size: Optional[int]=None, scale_bits: Optional[int]=None):
    if wq_bits == 3:
        if datatype == 'mixed':
            # for facebook/opt-1.3b: datatype_list = ['fp3_sm_pos', 'fp3_sm_neg', 'fp3_sr_pos', 'fp3_sr_neg']
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg', 'fp3_sr_pos', 'fp3_sr_neg']
        elif datatype == 'mixed_sp':
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg']
        elif datatype == 'mixed_sm':
            datatype_list = ['fp3_sm_pos', 'fp3_sm_neg']
        elif datatype == 'mixed_sr':
            datatype_list = ['fp3_sr_pos', 'fp3_sr_neg']
        elif datatype == 'mixed_sp_sm':
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg', 'fp3_sm_pos', 'fp3_sm_neg']
        elif datatype == 'mixed_sm_sr':
            datatype_list = ['fp3_sm_pos', 'fp3_sm_neg', 'fp3_sr_pos', 'fp3_sr_neg']
        elif datatype == 'mixed_sp_sr':
            datatype_list = ['fp3_sp_pos', 'fp3_sp_neg', 'fp3_sr_pos', 'fp3_sr_neg']
    elif wq_bits == 4:
        if datatype == 'mixed':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg', 'fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_sp':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg']
        elif datatype == 'mixed_sm':
            datatype_list = ['fp4_sm_pos', 'fp4_sm_neg']
        elif datatype == 'mixed_sr':
            datatype_list = ['fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_sp_sm':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg', 'fp4_sm_pos', 'fp4_sm_neg']
        elif datatype == 'mixed_sm_sr':
            datatype_list = ['fp4_sm_pos', 'fp4_sm_neg', 'fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_sp_sr':
            datatype_list = ['fp4_sp_pos', 'fp4_sp_neg', 'fp4_sr_pos', 'fp4_sr_neg']
        elif datatype == 'mixed_apot4':
            datatype_list = ['apot4_sp_pos', 'apot4_sp_neg', 'apot4_sr_pos', 'apot4_sr_neg']
    elif wq_bits == 5:
        datatype_list = ['int5', 'fp5_e2m2', 'fp5_e3m1']
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit and 5-bit quantization, not {wq_bits}-bit")

    K, C = w_fp16.size() # output channel, input channel
    if (group_size is None) or (group_size <= 0):
        group_size = C
    NUM_GROUP = C // group_size
    w_fp16 = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    q_tensor = torch.zeros_like(w_fp16)
    
    error = torch.full([K, NUM_GROUP], 1e3, dtype=torch.float16, device=w_fp16.device)
    
    for datatype in datatype_list:
        w_fp16_new = quant_datatype(w_fp16, wq_bits=wq_bits, datatype=datatype, group_size=None, scale_bits=scale_bits)
        quant_error = (w_fp16_new - w_fp16).abs().pow(2).mean(-1)
        update_mask = torch.lt(quant_error, error)

        print(datatype, torch.sum(update_mask))
        print()
        error[update_mask] = quant_error[update_mask]
        q_tensor[update_mask] = w_fp16_new[update_mask]

        del w_fp16_new
        del quant_error
        del update_mask
    
    return q_tensor.reshape(K, C)


def quant_model(model, wq_bits: Optional[int]=None, wq_datatype: Optional[str]=None, wq_groupsize: Optional[int]=None, scale_bits: Optional[int]=None):
    if (wq_datatype is None) or (wq_datatype in ["fp16", "fp32"]):
        print("Not applying quantization")
        time.sleep(2)
    elif ("int" in wq_datatype) and ("asym" in wq_datatype):
        print(f"Applying asymmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                #print(m.qweight._data.unique())
                print(n)
                m.weight.data = quant_int_asym(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize, scale_bits=scale_bits)
    elif ("int" in wq_datatype) and ("asym" not in wq_datatype):
        print(f"Applying symmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                #print(m.qweight._data.unique())
                print(n)
                m.weight.data = quant_int(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize, scale_bits=scale_bits)
    elif ("mixed" not in wq_datatype):
        print(f"Applying quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                #print(m.qweight._data.unique())
                print(n)
                m.weight.data = quant_datatype(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize, scale_bits=scale_bits)
    else:
        print(f"Applying quantization with bits: {wq_bits}, mixed datatypes, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                #print(m.qweight._data.unique())
                print(n)
                m.weight.data = search_datatype(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize, scale_bits=scale_bits)
                