`ifndef __exp_match_V__
`define __exp_match_V__


module max_exp_4to1
#(
    parameter DATA_WIDTH = 6
) (
    input  logic [DATA_WIDTH-1:0]  v_exp   [3:0],        
    output logic [DATA_WIDTH-1:0]  out
);

    logic [DATA_WIDTH-1:0]  cmp_level_one [1:0];
    assign cmp_level_one[1] = (v_exp[3] > v_exp[2]) ? v_exp[3] : v_exp[2];
    assign cmp_level_one[0] = (v_exp[1] > v_exp[0]) ? v_exp[1] : v_exp[0];
    
    assign out = (cmp_level_one[1] > cmp_level_one[0]) ? cmp_level_one[1] : cmp_level_one[0];
    
endmodule


module exp_match
#(
    parameter A_EXP_WIDTH    = 5,
    parameter W_EXP_WIDTH    = 2,
    parameter ACC_EXP_WIDTH  = 6,
	parameter VEC_LENGTH     = 4
) (
	input  logic  [A_EXP_WIDTH-1:0]    a_exp      [VEC_LENGTH-1:0],  // activation exponent
	input  logic  [W_EXP_WIDTH-1:0]    w_exp      [VEC_LENGTH-1:0],  // weight exponent

    input  logic                       a_sign     [VEC_LENGTH-1:0],  // activation sign
	input  logic                       w_sign     [VEC_LENGTH-1:0],  // weight sign

	output logic                       y_sign     [VEC_LENGTH-1:0],  // product sign
	output logic  [ACC_EXP_WIDTH-1:0]  delta_exp  [VEC_LENGTH-1:0],  // exponent difference
	output logic  [ACC_EXP_WIDTH-1:0]  max_exp                       // max exponent 
);
    genvar i;

    logic [ACC_EXP_WIDTH-1:0]  v_exp [VEC_LENGTH-1:0];
	generate
		for (i = 0; i < VEC_LENGTH; i = i + 1) begin
			assign y_sign[i] = a_sign[i] ^ w_sign[i];
            assign v_exp[i]  = a_exp[i] + w_exp[i];
		end
	endgenerate

    logic [ACC_EXP_WIDTH-1:0]  exp_out;
    max_exp_4to1 #(ACC_EXP_WIDTH) exp_compare (.out(exp_out), .*);
    generate
		for (i = 0; i < VEC_LENGTH; i = i + 1) begin
			assign delta_exp[i] = exp_out - v_exp[i];
		end
	endgenerate

    assign max_exp = exp_out;

endmodule

`endif