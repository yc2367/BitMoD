`ifndef __pos_neg_select_V__
`define __pos_neg_select_V__

module pos_neg_select #(
	parameter DATA_WIDTH = 14
) (
	input  logic        [DATA_WIDTH-1:0] in,
	input  logic                         sign,	
	output logic signed [DATA_WIDTH:0]   out
); 
	always_comb begin
		if (sign) begin
			out = ~{1'b0, in} + 1'b1;
		end else begin
			out = {1'b0, in};
		end
	end
endmodule

`endif