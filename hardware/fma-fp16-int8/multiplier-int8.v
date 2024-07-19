`ifndef _MULTIPLIER_V__
`define _MULTIPLIER_V__

module mulCore 
(
    input   logic [7:0] in8,
    input   logic [10:0] in11,
    output  logic [18:0] out
);

    assign out = in8 * in11;

endmodule

module abs8(
    input  logic [7:0] in,
    output logic [7:0] out
);

always_comb begin
    if (in[7] == 1'b1)
        out = ~in + 1;
    else
        out = in;
end

endmodule


module intMultiplier
(
    input   logic [7:0]     int8, // int 8 input
    input   logic [15:0]    act,

    output  logic           sign_out,
    output  logic [4:0]     exp_out,
    output  logic [18:0]    man_out_int
);

    // PREPARATION activation =========================================

    // decompose activation to sign, exponent, mantissa
    logic       act_sign;
    logic [4:0] act_exp;
    logic [9:0] act_man;
    assign act_sign = act[15];
    assign act_exp = act[14:10];
    assign act_man = act[9:0];
    
    logic [10:0] act_man_ext;
    assign act_man_ext = { act_exp == 0 ? 1'b0 : 1'b1, act_man};

    // PREPARATION int8 ================================================

    // decompose input to int8
    logic int8_sign;
    logic [7:0] int8val;
    assign int8_sign = int8[7];
    abs8 abs8_1(.in(int8), .out(int8val));

    // multiply

    localparam EXP_ACT_CALC_OFFSET = 15; // TODO this offset is the exponent offset of fp16, and should involved in calculation, but I have not decided whether to wrap it in this module or not.
    assign sign_out = int8_sign ^ act_sign;
    assign exp_out = act_exp - EXP_ACT_CALC_OFFSET;

    mulCore mc(.in8(int8val), .in11(act_man_ext), .out(man_out_int));
endmodule


`endif