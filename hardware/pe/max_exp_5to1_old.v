`ifndef __max_exp_5to1_V__
`define __max_exp_5to1_V__

module max_exp_5to1
#(
    parameter DATA_WIDTH = 6
) (
    input  logic [DATA_WIDTH-1:0]  v_exp   [3:0],        
    input  logic [DATA_WIDTH-1:0]  acc_exp,        
    output logic [DATA_WIDTH-1:0]  out
);

    logic [DATA_WIDTH-1:0]  cmp_level_one [2:0];
    assign cmp_level_one[2] = acc_exp;
    assign cmp_level_one[1] = (v_exp[3] > v_exp[2]) ? v_exp[3] : v_exp[2];
    assign cmp_level_one[0] = (v_exp[1] > v_exp[0]) ? v_exp[1] : v_exp[0];
    
    logic [DATA_WIDTH-1:0]  cmp_level_two [1:0];
    assign cmp_level_two[1] = cmp_level_one[2];
    assign cmp_level_two[0] = (cmp_level_one[1] > cmp_level_one[0]) ? cmp_level_one[1] : cmp_level_one[0];

    assign out = (cmp_level_two[1] > cmp_level_two[0]) ? cmp_level_two[1] : cmp_level_two[0];

endmodule

`endif