`ifndef __grp_accum_V__
`define __grp_accum_V__

`include "pos_neg_select.v"

module max_2to1 #(
	parameter DATA_WIDTH = 6
) (
	input  logic [DATA_WIDTH-1:0]  in_1,
	input  logic [DATA_WIDTH-1:0]  in_2,
	output logic [DATA_WIDTH-1:0]  out
);
	assign out = (in_1 > in_2) ? in_1 : in_2;
endmodule


module shifter_left #(
	parameter IN_WIDTH   = 17,
	parameter OUT_WIDTH  = 23,
	parameter EXP_WIDTH  = 3
) (
	input  logic signed [IN_WIDTH-1:0]   in,
	input  logic        [EXP_WIDTH-1:0]  shift_sel,	
	output logic signed [OUT_WIDTH-1:0]  out
);	
	always_comb begin 
		case (shift_sel)
			3'd0  : out = in;
			3'd1  : out = in <<< 3'd1;
			3'd2  : out = in <<< 3'd2;
			3'd3  : out = in <<< 3'd3;
			3'd4  : out = in <<< 3'd4;
			3'd5  : out = in <<< 3'd5;
			3'd6  : out = in <<< 3'd6;

			default: out = {OUT_WIDTH{1'bx}};
		endcase
	end
endmodule


module shifter_right #(
	parameter IN_WIDTH   = 15,
	parameter OUT_WIDTH  = 15,
	parameter EXP_WIDTH  = 6
) (
	input  logic signed [IN_WIDTH-1:0]   in,
	input  logic        [EXP_WIDTH-1:0]  shift_sel,	
	output logic signed [OUT_WIDTH-1:0]  out
);	
	always_comb begin 
		case (shift_sel)
			6'd0  : out = in;
			6'd1  : out = in >>> 4'd1;
			6'd2  : out = in >>> 4'd2;
			6'd3  : out = in >>> 4'd3;
			6'd4  : out = in >>> 4'd4;
			6'd5  : out = in >>> 4'd5;
			6'd6  : out = in >>> 4'd6;
			6'd7  : out = in >>> 4'd7;
			6'd8  : out = in >>> 4'd8;
			6'd9  : out = in >>> 4'd9;
			6'd10 : out = in >>> 4'd10;
			6'd11 : out = in >>> 4'd11;
			6'd12 : out = in >>> 4'd12;
			6'd13 : out = in >>> 4'd13;

			default: out = 0;
		endcase
	end
endmodule


module normalize #(
	parameter IN_WIDTH       = 23,
	parameter EXP_OUT_WIDTH  = 4,
	parameter MAN_OUT_WIDTH  = 15
) (
	input  logic signed [IN_WIDTH-1:0]       in,

	output logic                             sign_out,
	output logic        [EXP_OUT_WIDTH-1:0]  exp_out,
	output logic signed [MAN_OUT_WIDTH-1:0]  man_out_signed
);	
	logic signed [IN_WIDTH-1:0] in_unsigned;
	always_comb begin
		if (in[IN_WIDTH-1]) begin // change from negative to positive
			in_unsigned = ~in + 1'b1;
			sign_out = 1'b1;
		end else begin // maintain positive
			in_unsigned = in;
			sign_out = 1'b0;
		end
	end

	always_comb begin
		casez (in_unsigned)
            23'b01?????????????????????:  exp_out = 4'b1000;
            23'b001????????????????????:  exp_out = 4'b0111;
            23'b0001???????????????????:  exp_out = 4'b0110;
            23'b00001??????????????????:  exp_out = 4'b0101;
            23'b000001?????????????????:  exp_out = 4'b0100;
            23'b0000001????????????????:  exp_out = 4'b0011;
            23'b00000001???????????????:  exp_out = 4'b0010;
            23'b000000001??????????????:  exp_out = 4'b0001;
            23'b0000000001?????????????:  exp_out = 4'b0000;

            default: exp_out = 4'b0000;
        endcase
	end

	logic signed [IN_WIDTH-1:0]  man_tmp; 
	assign man_tmp = in >>> exp_out;
	assign man_out_signed = man_tmp[MAN_OUT_WIDTH-1:0];

endmodule



module grp_accum
#(
    parameter IN_WIDTH       = 17,
    parameter ACC_EXP_WIDTH  = 6,
	parameter ACC_MAN_WIDTH  = 15,
	parameter ACC_WIDTH      = 23
) (
	input  logic clk,
	input  logic reset,
	input  logic en,

	input  logic signed [IN_WIDTH-1:0]        bs_dp,      // bit-serial dot product
	input  logic        [ACC_EXP_WIDTH-1:0]   a_max_exp,  // max exponent of activations
	input  logic        [2:0]                 w_sig,      // bit-significance of the current weight mantissa

	output logic        [ACC_EXP_WIDTH-1:0]   acc_exp,    // accumulator exponent
	output logic        [ACC_MAN_WIDTH-1:0]   acc_man,    // accumulator mantissa
	output logic                              acc_sign    // accumulator sign
);	

	logic  [ACC_EXP_WIDTH-1:0]  acc_exp_tmp;  
	logic  [ACC_MAN_WIDTH-1:0]  acc_man_tmp;  
	logic                       acc_sign_tmp;  

	logic  [ACC_EXP_WIDTH-1:0]  max_exp;    // maximum exponent between accumulator and activation
	logic  [ACC_EXP_WIDTH-1:0]  delta_acc_exp;  // exponent difference between max_exp and acc_exp
	max_2to1 #(ACC_EXP_WIDTH) compare_exp (.in_1(acc_exp_tmp), .in_2(a_max_exp), .out(max_exp));
	assign delta_acc_exp = max_exp - acc_exp_tmp;

	logic signed [ACC_WIDTH-1:0] acc_in_1;
	shifter_left #(
		.IN_WIDTH(IN_WIDTH), .OUT_WIDTH(ACC_WIDTH), .EXP_WIDTH(3)
	) l_shift (
		.in(bs_dp), .out(acc_in_1), .shift_sel(w_sig)
	);

	logic signed [ACC_MAN_WIDTH-1:0] acc_in_2;
	shifter_right #(
		.IN_WIDTH(ACC_MAN_WIDTH), .OUT_WIDTH(ACC_MAN_WIDTH), .EXP_WIDTH(6)
	) r_shift (
		.in(acc_man_tmp), .out(acc_in_2), .shift_sel(delta_acc_exp)
	);

	logic signed [ACC_WIDTH-1:0] acc_sum;
	assign acc_sum = acc_in_1 + acc_in_2;

	logic  [3:0]                 acc_exp_diff;
	logic  [ACC_EXP_WIDTH-1:0]   acc_result_exp;
	logic  [ACC_MAN_WIDTH-1:0]   acc_result_man;
	logic                        acc_result_sign;  
	normalize #(
		.IN_WIDTH(ACC_WIDTH), .EXP_OUT_WIDTH(4), .MAN_OUT_WIDTH(ACC_MAN_WIDTH)
	) norm_fp (
		.in(acc_sum), .exp_out(acc_exp_diff), .sign_out(acc_result_sign), .man_out_signed(acc_result_man)
	);
	assign acc_result_exp = acc_exp_diff + max_exp;

	always @(posedge clk) begin
		if (reset) begin
			acc_man_tmp  <= 0;
			acc_exp_tmp  <= 0;
			acc_sign_tmp <= 0;
		end else if	(en) begin
			acc_man_tmp  <= acc_result_man;
			acc_exp_tmp  <= acc_result_exp;
			acc_sign_tmp <= acc_result_sign;
		end
	end

	assign acc_man  = acc_man_tmp;
	assign acc_exp  = acc_exp_tmp;
	assign acc_sign = acc_sign_tmp;
endmodule



module grp_accum_clk
#(
    parameter IN_WIDTH       = 17,
    parameter ACC_EXP_WIDTH  = 6,
	parameter ACC_MAN_WIDTH  = 15,
	parameter ACC_WIDTH      = 23
) (
	input  logic clk,
	input  logic reset,
	input  logic en,

	input  logic signed [IN_WIDTH-1:0]        bs_dp_t,      // bit-serial dot product
	input  logic        [ACC_EXP_WIDTH-1:0]   a_max_exp_t,  // max exponent of activations
	input  logic        [2:0]                 w_sig_t,      // bit-significance of the current weight mantissa

	output logic        [ACC_EXP_WIDTH-1:0]   acc_exp,    // accumulator exponent
	output logic        [ACC_MAN_WIDTH-1:0]   acc_man,    // accumulator mantissa
	output logic                              acc_sign    // accumulator sign
);	
	logic signed [IN_WIDTH-1:0]        bs_dp; 
	logic        [ACC_EXP_WIDTH-1:0]   a_max_exp;
	logic        [2:0]                 w_sig;

	always @(posedge clk) begin
		if (reset) begin
			bs_dp      <= 0;
			a_max_exp  <= 0;
			w_sig      <= 0;
		end else if	(en) begin
			bs_dp      <= bs_dp_t;
			a_max_exp  <= a_max_exp_t;
			w_sig      <= w_sig_t;
		end
	end

	grp_accum #(IN_WIDTH, ACC_EXP_WIDTH, ACC_MAN_WIDTH, ACC_WIDTH) dut (.*);
endmodule


`endif