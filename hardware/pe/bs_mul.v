`ifndef __bs_mul_V__
`define __bs_mul_V__

`include "pos_neg_select.v"

module shifter #(
	parameter IN_WIDTH   = 11,
	parameter OUT_WIDTH  = 14,
	parameter EXP_WIDTH  = 6
) (
	input  logic [IN_WIDTH-1:0]   in,
	input  logic [EXP_WIDTH-1:0]  shift_sel,	
	output logic [OUT_WIDTH-1:0]  out
);	
	localparam TMP_WIDTH = IN_WIDTH + OUT_WIDTH;
	logic [TMP_WIDTH-1:0]  out_tmp;
	always_comb begin 
		case (shift_sel)
			6'd0  : out_tmp = { in, 14'd0 };
			6'd1  : out_tmp = { in, 13'd0 };
			6'd2  : out_tmp = { in, 12'd0 };
			6'd3  : out_tmp = { in, 11'd0 };
			6'd4  : out_tmp = { in, 10'd0 };
			6'd5  : out_tmp = { in, 9'd0 };
			6'd6  : out_tmp = { in, 8'd0 };
			6'd7  : out_tmp = { in, 7'd0 };
			6'd8  : out_tmp = { in, 6'd0 };
			6'd9  : out_tmp = { in, 5'd0 };
			6'd10 : out_tmp = { in, 4'd0 };
			6'd11 : out_tmp = { in, 3'd0 };
			6'd12 : out_tmp = { in, 2'd0 };
			6'd13 : out_tmp = { in, 1'd0 };
			6'd14 : out_tmp = { in };

			default: out_tmp = 0;
		endcase
	end

	assign out = out_tmp[TMP_WIDTH-1 : IN_WIDTH];
endmodule


module bs_mul
#(
    parameter A_MAN_WIDTH    = 11,
    parameter ACC_EXP_WIDTH  = 6,
	parameter VEC_LENGTH     = 4,
	parameter OUT_WIDTH      = 17
) (
	input  logic  [A_MAN_WIDTH-1:0]     a_man      [VEC_LENGTH-1:0],  // activation mantissa
	input  logic                        w_man      [VEC_LENGTH-1:0],  // weight mantissa
	input  logic  [ACC_EXP_WIDTH-1:0]   delta_exp  [VEC_LENGTH-1:0],  // exponent difference
	input  logic                        y_sign     [VEC_LENGTH-1:0],  // sign of product between weight and activation

	output logic signed [OUT_WIDTH-1:0] bs_out                        // bit-serial output
);
    genvar i;

    logic        [A_MAN_WIDTH-1:0]  bs_prod         [VEC_LENGTH-1:0]; // bit-serial product 
    logic        [A_MAN_WIDTH+2:0]  bs_prod_aligned [VEC_LENGTH-1:0]; // bit-serial product after shifting by the delta exponent
	logic signed [A_MAN_WIDTH+3:0]  bs_prod_2s      [VEC_LENGTH-1:0]; // bit-serial product after negation block
	generate
		for (i = 0; i < VEC_LENGTH; i = i + 1) begin
			assign bs_prod[i] = a_man[i] & {A_MAN_WIDTH{w_man[i]}};

			shifter #(
				.IN_WIDTH(A_MAN_WIDTH),  .OUT_WIDTH(A_MAN_WIDTH+3),  .EXP_WIDTH(ACC_EXP_WIDTH)
			) mantissa_align (
				.in(bs_prod[i]), .shift_sel(delta_exp[i]), .out(bs_prod_aligned[i])
			);

			pos_neg_select #(A_MAN_WIDTH+3) twos_complement (.in(bs_prod_aligned[i]), .sign(y_sign[i]), .out(bs_prod_2s[i]));
		end
	endgenerate

	logic signed [A_MAN_WIDTH+4:0]  psum_level_one  [VEC_LENGTH/2-1:0];
	logic signed [A_MAN_WIDTH+5:0]  psum_level_two;
	generate
		for (i = 0; i < VEC_LENGTH/2; i = i + 1) begin
			assign psum_level_one[i] = bs_prod_2s[2*i] + bs_prod_2s[2*i+1];
		end
	endgenerate
	assign psum_level_two = psum_level_one[1] + psum_level_one[0];
	assign bs_out = psum_level_two;

endmodule



module bs_mul_clk
#(
    parameter A_MAN_WIDTH    = 11,
    parameter ACC_EXP_WIDTH  = 6,
	parameter VEC_LENGTH     = 4,
	parameter OUT_WIDTH      = 17
) (
	input  logic  clk,
	input  logic  reset,
	input  logic  en,
	
	input  logic  [A_MAN_WIDTH-1:0]     a_man_t      [VEC_LENGTH-1:0],  // activation mantissa
	input  logic                        w_man_t      [VEC_LENGTH-1:0],  // weight mantissa
	input  logic  [ACC_EXP_WIDTH-1:0]   delta_exp_t  [VEC_LENGTH-1:0],  // exponent difference
	input  logic                        y_sign_t     [VEC_LENGTH-1:0],  // sign of product between weight and activation

	output logic signed [OUT_WIDTH-1:0] bs_out_t                        // bit-serial output
);
    genvar i;

    logic  [A_MAN_WIDTH-1:0]     a_man      [VEC_LENGTH-1:0];
    logic                        w_man      [VEC_LENGTH-1:0];
    logic  [ACC_EXP_WIDTH-1:0]   delta_exp  [VEC_LENGTH-1:0];
    logic                        y_sign     [VEC_LENGTH-1:0];
    logic signed [OUT_WIDTH-1:0] bs_out;
	generate
		for (i = 0; i < VEC_LENGTH; i = i + 1) begin
			always @(posedge clk) begin
				if (reset) begin
					a_man[i]     <= 0;
					w_man[i]     <= 0;
					delta_exp[i] <= 0;
					y_sign[i]    <= 0;
				end else if	(en) begin
					a_man[i]     <= a_man_t[i];
					w_man[i]     <= w_man_t[i];
					delta_exp[i] <= delta_exp_t[i];
					y_sign[i]    <= y_sign_t[i];
				end
			end
		end
	endgenerate

	always @(posedge clk) begin
		if (reset) begin
			bs_out_t <= 0;
		end else if	(en) begin
			bs_out_t <= bs_out;
		end
	end

	bs_mul #(A_MAN_WIDTH, ACC_EXP_WIDTH, VEC_LENGTH, OUT_WIDTH) dut (.*);

endmodule


`endif