from accelerator import Accelerator 

model_list = ["facebook/opt-1.3b", "microsoft/phi-2", "01-ai/Yi-6B", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Meta-Llama-3-8B"]
#model_list = ["meta-llama/Llama-2-7b-hf"]

if __name__ == "__main__":
    is_generation = False
    is_lossless = False
    if is_generation:
        pe_array_dim = [64, 16]
        if is_lossless:
            w_prec = 6.0625
        else:
            w_prec = 3.0625
    else:
        pe_array_dim = [32, 32]
        if is_lossless:
            w_prec = 6.0625
        else:
            w_prec = 4.0625
    
    total_energy_list = [[0, 0] for _ in model_list]

    for idx, model_name in enumerate(model_list):
        acc = Accelerator(
            model_name=model_name, 
            i_prec=16,
            w_prec=w_prec,
            is_bit_serial=True,
            pe_dp_size=4,
            pe_energy=0.56,
            pe_area=1507.7,
            pe_array_dim=pe_array_dim,
            context_length=256,
            is_generation=is_generation,
        )

        total_cycle    = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy    = acc.calc_dram_energy() / 1e6
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

        print_energy = True
        print(f'model: {model_name}')
        print(f'total cycle:        {total_cycle}')

        if print_energy:
            print(f'pe array area:      {acc.pe_array_area / 1e6} mm2')
            print(f'weight buffer area: {acc.w_sram.area} mm2')
            print(f'input buffer area:  {acc.i_sram.area} mm2')
            print(f'compute energy:     {compute_energy} uJ')
            print(f'sram rd energy:     {sram_rd_energy} uJ')
            print(f'sram wr energy:     {sram_wr_energy} uJ')
            print(f'dram energy:        {dram_energy} uJ')
            print(f'on-chip energy:     {onchip_energy} uJ')
            print(f'total energy:       {total_energy} uJ')
            total_energy_list[idx][0] = round(onchip_energy)
            total_energy_list[idx][1] = round(total_energy)
        
        print('\n')

    print(total_energy_list)
    