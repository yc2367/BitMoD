set tsmc28 "/opt/PDKs/TSMC/28nm/Std_Cell_Lib/tcbn28hpcplusbwp30p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp30p140_180a"
set workdir /home/yc2367/Research/BitMoD/hardware/pe
set_app_var target_library "$tsmc28/tcbn28hpcplusbwp30p140ssg0p9vm40c.db"
set_app_var link_library   "* $target_library"

set func  3 ;

if {$func == 0} {
    analyze -format sverilog $workdir/exp_match.v
    elaborate exp_match
} elseif {$func == 1} {
    analyze -format sverilog $workdir/bs_mul.v
    elaborate bs_mul_clk
} elseif {$func == 2} {
    analyze -format sverilog $workdir/grp_accum.v
    elaborate grp_accum_clk
} elseif {$func == 3} {
    analyze -format sverilog $workdir/dequant.v
    elaborate dequant_clk
} 

check_design
create_clock clk -name ideal_clock1 -period 1
compile

# Generate structural verilog netlist
write_file -hierarchy -format verilog -output "./work/post.28nm.syn.v"
write_parasitics -output "./work/post.28nm.spef.gz"

# Generate timing constraints file
write_sdc "./work/post.syn.28nm.sdc"

report_resources -nosplit -hierarchy
report_timing -nosplit -transition_time -nets -attributes
report_area -nosplit -hierarchy
report_power -nosplit -hierarchy

exit