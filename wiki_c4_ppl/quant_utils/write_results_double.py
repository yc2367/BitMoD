import os 

def write_results(ppl: float, model: str, dataset: str, wq_bits: int, wq_datatype: str, wq_groupsize: int, scale_bits: int):
    BASE_PATH = "/home/yc2367/llm/test_llm_eval_c4_wikitext/results_llm_double"
    if wq_groupsize <= 0:
        wq_groupsize = "none"
    
    if wq_datatype == "fp16":
        wq_datatype = "base_fp16"

    model_info = model.split('/')
    if len(model_info) > 1:
        model_dir = f"{model_info[0]}__{model_info[1]}"
    else:
        model_dir = f"{model_info[0]}__"
    dataset_dir = f"{dataset}"
    wq_dir = f"wbits_{wq_bits}_gs_{wq_groupsize}"
    dtype_file = f"{wq_datatype}_s{scale_bits}.txt"

    if not os.path.exists(f'{BASE_PATH}/{model_dir}'):
        os.mkdir(f'{BASE_PATH}/{model_dir}')
    
    if not os.path.exists(f'{BASE_PATH}/{model_dir}/{dataset_dir}'):
        os.mkdir(f'{BASE_PATH}/{model_dir}/{dataset_dir}')
    
    if (wq_datatype != 'base_fp16') and (not os.path.exists(f'{BASE_PATH}/{model_dir}/{dataset_dir}/{wq_dir}')):
        os.mkdir(f'{BASE_PATH}/{model_dir}/{dataset_dir}/{wq_dir}')
    
    if (wq_datatype == 'base_fp16'):
        result_file = f'{BASE_PATH}/{model_dir}/{dataset_dir}/{dtype_file}'
    else:
        result_file = f'{BASE_PATH}/{model_dir}/{dataset_dir}/{wq_dir}/{dtype_file}'

    with open(result_file, 'w') as f:
        f.writelines(f'Perplexity: {ppl} \n')
    
    print('Successfully written results. \n\n')