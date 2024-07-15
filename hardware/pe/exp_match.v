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



module exp_match_clk
#(
    parameter A_EXP_WIDTH    = 5,
    parameter W_EXP_WIDTH    = 2,
    parameter ACC_EXP_WIDTH  = 6,
	parameter VEC_LENGTH     = 4
) (
    input  logic  clk,
	input  logic  reset,
	input  logic  en,
    
	input  logic  [A_EXP_WIDTH-1:0]    a_exp_t      [VEC_LENGTH-1:0],  // activation exponent
	input  logic  [W_EXP_WIDTH-1:0]    w_exp_t      [VEC_LENGTH-1:0],  // weight exponent

    input  logic                       a_sign_t     [VEC_LENGTH-1:0],  // activation sign
	input  logic                       w_sign_t     [VEC_LENGTH-1:0],  // weight sign

	output logic                       y_sign_t     [VEC_LENGTH-1:0],  // product sign
	output logic  [ACC_EXP_WIDTH-1:0]  delta_exp_t  [VEC_LENGTH-1:0],  // exponent difference
	output logic  [ACC_EXP_WIDTH-1:0]  max_exp_t                       // max exponent 
);
    genvar i;
    
    logic  [A_EXP_WIDTH-1:0]    a_exp      [VEC_LENGTH-1:0];
	logic  [W_EXP_WIDTH-1:0]    w_exp      [VEC_LENGTH-1:0];

    logic                       a_sign     [VEC_LENGTH-1:0];
	logic                       w_sign     [VEC_LENGTH-1:0];

    logic                       y_sign     [VEC_LENGTH-1:0];
	logic  [ACC_EXP_WIDTH-1:0]  delta_exp  [VEC_LENGTH-1:0];
	logic  [ACC_EXP_WIDTH-1:0]  max_exp;

	generate
		for (i = 0; i < VEC_LENGTH; i = i + 1) begin
			always @(posedge clk) begin
				if (reset) begin
					a_exp[i]     <= 0;
					w_exp[i]     <= 0;
					a_sign[i]    <= 0;
					w_sign[i]    <= 0;

                    y_sign_t[i]  <= 0;
                    delta_exp_t[i] <= 0;
				end else if	(en) begin
					a_exp[i]     <= a_exp_t[i];
					w_exp[i]     <= w_exp_t[i];
					a_sign[i]    <= a_sign_t[i];
					w_sign[i]    <= w_sign_t[i];

                    y_sign_t[i]  <= y_sign[i];
                    delta_exp_t[i] <= delta_exp[i];
				end
			end
		end
	endgenerate

    always @(posedge clk) begin
        if (reset) begin
            max_exp_t <= 0;
        end else if	(en) begin
            max_exp_t <= max_exp;
        end
    end
		
	exp_match #(A_EXP_WIDTH, W_EXP_WIDTH, ACC_EXP_WIDTH, VEC_LENGTH) dut (.*);

endmodule
`endif