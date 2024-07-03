`ifndef __dequant_V__
`define __dequant_V__


module dequant
#(
    parameter IN_EXP_WIDTH   = 6,
	parameter IN_MAN_WIDTH   = 15,
	parameter OUT_EXP_WIDTH  = 6,
	parameter OUT_MAN_WIDTH  = 21
) (
	input  logic clk,
	input  logic reset,
	input  logic en_acc,     // enable bit-serial accumulator
	input  logic start_acc,  // start new bit-serial accumulation

	input  logic                               scale_bit,  // bit-significance of the current weight mantissa
	input  logic                               scale_sign, // bit-significance of the current weight mantissa

	input  logic         [IN_EXP_WIDTH-1:0]    in_exp,   // input exponent
	input  logic  signed [IN_MAN_WIDTH-1:0]    in_man,   // input mantissa
	input  logic                               in_sign,  // input sign

	output logic         [OUT_EXP_WIDTH-1:0]   out_exp,  // output exponent
	output logic  signed [OUT_MAN_WIDTH-1:0]   out_man,  // output mantissa
	output logic                               out_sign  // output sign
);	

    logic signed [IN_MAN_WIDTH-1:0]   bs_prod; // bit-serial product 
	assign bs_prod = in_man & {IN_MAN_WIDTH{scale_bit}};

	logic signed [OUT_MAN_WIDTH-1:0]  out_man_tmp, acc_in;
	always_comb begin
		if (start_acc) begin
			acc_in = 0;
		end begin
			acc_in = out_man_tmp <<< 1'b1;
		end
	end
	always @(posedge clk) begin
		if (reset) begin
			out_man_tmp  <= 0;
		end else if	(en_acc) begin
			out_man_tmp  <= acc_in + bs_prod;
		end
	end
	assign out_man = out_man_tmp;

	always @(posedge clk) begin
		if (reset) begin
			out_sign <= 0;
			out_exp  <= 0;
		end else if	(start_acc) begin
			out_sign <= scale_sign ^ in_sign;
			out_exp  <= in_exp;
		end
	end
endmodule



module dequant_clk
#(
    parameter IN_EXP_WIDTH   = 6,
	parameter IN_MAN_WIDTH   = 15,
	parameter OUT_EXP_WIDTH  = 6,
	parameter OUT_MAN_WIDTH  = 21
) (
	input  logic clk,
	input  logic reset,
	input  logic en_latch_input, 
	input  logic en_acc,     // enable bit-serial accumulator
	input  logic start_acc,  // start new bit-serial accumulation

	input  logic                               scale_bit_t,  // bit-significance of the current weight mantissa
	input  logic                               scale_sign_t, // bit-significance of the current weight mantissa

	input  logic         [IN_EXP_WIDTH-1:0]    in_exp_t,   // input exponent
	input  logic  signed [IN_MAN_WIDTH-1:0]    in_man_t,   // input mantissa
	input  logic                               in_sign_t,  // input sign

	output logic         [OUT_EXP_WIDTH-1:0]   out_exp,  // output exponent
	output logic  signed [OUT_MAN_WIDTH-1:0]   out_man,  // output mantissa
	output logic                               out_sign  // output sign
);

	logic                               scale_bit;  // bit-significance of the current weight mantissa
	logic                               scale_sign; // bit-significance of the current weight mantissa

	logic         [IN_EXP_WIDTH-1:0]    in_exp;   // input exponent
	logic  signed [IN_MAN_WIDTH-1:0]    in_man;   // input mantissa
	logic                               in_sign;  // input sign 

	always @(posedge clk) begin
		if (reset) begin
			scale_bit   <= 0;
			scale_sign  <= 0;
			in_exp      <= 0;
			in_man      <= 0;
			in_sign     <= 0;
		end else if	(en_latch_input) begin
			scale_bit   <= scale_bit_t;
			scale_sign  <= scale_sign_t;
			in_exp      <= in_exp_t;
			in_man      <= in_man_t;
			in_sign     <= in_sign_t;
		end
	end

	dequant #(IN_EXP_WIDTH, IN_MAN_WIDTH, OUT_EXP_WIDTH, OUT_MAN_WIDTH) dut (.*);
endmodule


`endif