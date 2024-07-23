from huggingface_hub import login
login(token="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")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import random, time, argparse, os
from tqdm import tqdm
from typing import Optional

from quant_utils.quant_weight_double import quant_model
from quant_utils.write_results_double import write_results


def get_dataset(model, dataset_path, dataset_name):
    if 'c4' in dataset_path:
        dataset_split = 'validation'
        column = 'text'
    elif 'ptb' in dataset_path:
        dataset_split = 'test'
        column = 'sentence'
    else:
        raise ValueError(f'Only support dataset C4 and PTB, not {dataset}')
    
    testdata = load_dataset(dataset_path, dataset_name, split=dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer(" ".join(testdata[column]), return_tensors='pt')
    return testenc


DATASET_PATH_MAPPING = {'c4': 'allenai/c4', 'ptb': 'ptb_text_only'}
DATASET_NAME_MAPPING = {'c4': 'realnewslike', 'ptb': 'penn_treebank'}


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
     "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
)
parser.add_argument(
    "--dataset", default="c4", type=str, help="Dataset path, e.g. c4",
)
parser.add_argument(
    "--wq_datatype", type=str, default="", help="The weight datatype for weight-only quantization",
)
parser.add_argument(
    "--wq_bits", type=int, default=4, help="The weight precision for weight-only quantization",
)
parser.add_argument(
    "--wq_groupsize", type=int, default=None, help="The quantization group size for weight-only quantization",
)
parser.add_argument(
    "--scale_bits", type=int, default=None, help="The precision for per-group scaling factors",
)
args = parser.parse_args()

model_str = args.model
dataset_path = DATASET_PATH_MAPPING[args.dataset]
dataset_name = DATASET_NAME_MAPPING[args.dataset]
wq_bits = args.wq_bits
wq_groupsize = args.wq_groupsize
wq_datatype = args.wq_datatype
scale_bits = args.scale_bits


# Added by Yuzong Chen (yc2367@cornell.edu)
if 'c4' in dataset_path:
    tile_num = 4
else:
    tile_num = 1

torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)
seqlen = 1024

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
quant_model(model, wq_bits, wq_datatype, wq_groupsize, scale_bits)

input_tok = get_dataset(model_str, dataset_path, dataset_name)['input_ids']
nsamples = input_tok.numel() // seqlen
input_tok = input_tok[0, :(seqlen * nsamples)].view(nsamples, seqlen)
print(input_tok.shape)

loss_fct = torch.nn.CrossEntropyLoss().cuda()
acc_loss = 0.0

with tqdm(range(nsamples // tile_num)) as progress:
    for ii in progress:
        input = input_tok[ii, :].cuda().view(1, -1)
        output = model(input,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False)[0]
        shift_logits = output[:, :-1, :].contiguous()
        shift_labels = input[:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        acc_loss += loss.item()
        progress.set_description(f"Evaluating")

avg_loss = acc_loss / (nsamples // tile_num)

ppl = torch.exp(torch.tensor(avg_loss)).item()
print(f'perplexity: {ppl}')

write_results(ppl, model_str, args.dataset, wq_bits, wq_datatype, wq_groupsize, scale_bits)
