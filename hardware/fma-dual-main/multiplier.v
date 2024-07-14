`ifndef __MULTIPLIER_V__
`define __MULTIPLIER_V__

// convert fp4 exponent and mantissa to two parts, one for multiplication with mantissa and other for addition with exponent
// TODO implement the real funciton, here is now a place holder.
module cvt34
(
    input   logic [2:0] in,
    output  logic [3:0] mul,  // 4 bit fixed point in 2.2 format (2 bits integer, and 2 bits fraction)
    output  logic [2:0] add   // 3 bit number for directly add to exponent, should consider offset in this module
);


    assign mul = {1'b0, in}; // TODO: placeholder for now
    assign add = in[2]; // TODO: placeholder for now
endmodule


module Mux2 #(
    parameter WIDTH = 16
)
(
    input   logic [WIDTH-1:0] in1,
    input   logic [WIDTH-1:0] in2,
    input   logic             sel,
    output  logic [WIDTH-1:0] out
);

    assign out = sel ? in2 : in1;
endmodule

// simple multiplier core, now just a * b
module mulCore 
(
    input   logic [3:0] in4,
    input   logic [10:0] in11,
    output  logic [14:0] out
);

    assign out = in4 * in11;

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


module dualMultiplier
(
    input   logic [15:0] act, // fp16 activation
    input   logic [7:0]  in, // 8-bit input in int8/ 2fp4 format
    input   logic        mode, // 0: int8, 1: 2fp4

    // 2fp4 output
    output  logic           sign_out1,
    output  logic           sign_out2,
    output  logic [5:0]     exp_out1, // 6-bit exponent for considering overflow
    output  logic [5:0]     exp_out2, // 
    output  logic [14:0]    man_out1, // 3.12 format
    output  logic [14:0]    man_out2, // 3.12 format

    // int8 output
    output  logic           sign_out_int,
    output  logic [4:0]     exp_out_int,
    output  logic [18:0]    man_out_int // 9.10 format
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
    // PREPARATION fp4 ================================================

    // decompose input to two fp4
    logic [3:0] fp41, fp42;
    assign fp41 = in[7:4];
    assign fp42 = in[3:0];

    // decompose fp4 to sign, add part (effective exp), mul part (effective mantissa)
    logic fp41_sign, fp42_sign;
    logic [2:0] fp4_add1, fp4_add2;
    logic [3:0] fp4_mul1, fp4_mul2;
    assign fp41_sign = fp41[3];
    assign fp42_sign = fp42[3];
    cvt34 cv1(.in(fp41[2:0]), .mul(fp4_mul1), .add(fp4_add1));
    cvt34 cv2(.in(fp42[2:0]), .mul(fp4_mul2), .add(fp4_add2));

    // PREPARATION int8 ================================================

    // decompose input to int8
    logic int8_sign;
    logic [7:0] int8val;
    assign int8_sign = in[7];
    abs8 abs8_1(.in(in), .out(int8val));

    // decompose int8 magnitude to upper 4 and lower 4 bits
    logic [3:0] int8_upper, int8_lower;
    assign int8_upper = int8val[7:4];
    assign int8_lower = int8val[3:0];

    // CALCULATION ====================================================

    assign sign_out1 = act_sign ^ fp41_sign;
    assign sign_out2 = act_sign ^ fp42_sign;
    assign sign_out_int = act_sign ^ int8_sign;

    localparam EXP_ACT_CALC_OFFSET = 15; // TODO this offset is the exponent offset of fp16, and should involved in calculation, but I have not decided whether to wrap it in this module or not.
    assign exp_out1 = act_exp + fp4_add1 - EXP_ACT_CALC_OFFSET;
    assign exp_out2 = act_exp + fp4_add2 - EXP_ACT_CALC_OFFSET;
    assign exp_out_int = act_exp - EXP_ACT_CALC_OFFSET;

    // mode mux for int8 or 2fp4
    logic   [3:0] mul_in0, mul_in1;
    Mux2 #(4) mux0 (.in1(int8_upper), .in2(fp4_mul1), .sel(mode), .out(mul_in0));
    Mux2 #(4) mux1 (.in1(int8_lower), .in2(fp4_mul2), .sel(mode), .out(mul_in1));

    // multiplication
    mulCore mul0 (.in4(mul_in0), .in11(act_man_ext), .out(man_out1));
    mulCore mul1 (.in4(mul_in1), .in11(act_man_ext), .out(man_out2));

    // shift and add two ints.
    logic [18:0] temp1, temp2;
    assign temp1 = {man_out1, 4'b0}; // == man_out1 << 4 (higher 4 bits)
    assign temp2 = {4'b0, man_out2}; // == man_out2 (lower 4 bits)
    assign man_out_int = temp1 + temp2;

endmodule


`endif