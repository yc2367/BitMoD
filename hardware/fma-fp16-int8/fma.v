`ifndef __FMA_V__
`define __FMA_V__

`include "multiplier-int8.v"
`include "normalizer.v"
`include "fp_adder.v"


module fma_int8
(
    input   logic        clk,
    input   logic        reset,

    input   logic [15:0] act, // fp16 activation
    input   logic [7:0]  in, // 8-bit input in int8
    input   logic [15:0] acc, // fp16 accumulation 1

    output  logic [15:0] acc_out // fp16 accumulation 1
);

    // input pipeline registers

    logic [15:0] act_reg;
    logic [7:0]  in_reg;
    logic [15:0] acc_reg0;
    assign act_reg = act;
    assign in_reg = in;
    assign acc_reg0 = acc;

    // multiplier

    logic           sign_out;
    logic [4:0]     exp_out;
    logic [18:0]    man_out_int;

    intMultiplier intMul (
        .int8(in_reg),
        .act(act_reg),
        .sign_out(sign_out),
        .exp_out(exp_out),
        .man_out_int(man_out_int)
    );

    // register from multiplier
    logic           sign_out_reg;
    logic [4:0]     exp_out_reg;
    logic [18:0]    man_out_int_reg;
    // register from previous preseve stage
    logic [15:0]    acc_reg;

    // pipeline registers for multiplier
    always_ff @(posedge clk) begin
        if (reset) begin
            sign_out_reg <= 1'b0;
            exp_out_reg <= 5'b0;
            man_out_int_reg <= 19'b0;
            acc_reg <= 16'b0;
        end else begin
            sign_out_reg <= sign_out;
            exp_out_reg <= exp_out;
            man_out_int_reg <= man_out_int;
            acc_reg <= acc_reg0;
        end
    end

    localparam EXP_ACT_CALC_OFFSET = 15;

    logic           acc_sign;
    logic [4:0]     acc_exp_real;
    logic [10:0]    acc_man_ext;
    assign acc_sign = acc_reg[15];
    assign acc_exp_real = acc_reg[14:10] - EXP_ACT_CALC_OFFSET;
    assign acc_man_ext = { (acc_reg[14:10]) == 0 ? 1'b0 : 1'b1, acc_reg[9:0]};

    logic           acc_sign_out;
    logic [4:0]     acc_exp_out;
    logic [19:0]    acc_man_out;
    // adder
    fp_adder_core #(5, 11, 19, 1, 9) acc_adder (
        .sign1(acc_sign),
        .sign2(sign_out_reg),
        .exp1real(acc_exp_real),
        .exp2real(exp_out_reg),
        .man1(acc_man_ext),
        .man2(man_out_int_reg),

        .sign_out(acc_sign_out),
        .exp_out(acc_exp_out),
        .man_out(acc_man_out)
    );

    logic [4:0] norm_exp_out;
    logic [9:0] norm_man_out;

    normalizer #(5, 20, 10, 9) nrmlz (
        .exp_in(acc_exp_out),
        .man_in(acc_man_out),
        .exp_out(norm_exp_out),
        .man_out(norm_man_out)
    );

    logic [15:0] acc_out_reg;

    assign acc_out_reg = {acc_sign_out, norm_exp_out, norm_man_out};

    always_ff @(posedge clk) begin
        if (reset) begin
            acc_out <= 16'b0;
        end else begin
            acc_out <= acc_out_reg;
        end
    end

endmodule


module fma_int8_clk
(
    input   logic        clk,
    input   logic        reset,

    input   logic [15:0] act_t, // fp16 activation
    input   logic [7:0]  in_t,  // 8-bit input in int8/ 2fp4 format
    input   logic [15:0] acc_t, // fp16 accumulation 2

    output  logic [15:0] acc_out // fp16 accumulation 1
);

    logic [15:0] act;
    logic [7:0]  in;
    logic [15:0] acc;

    always_ff @(posedge clk) begin
        if (reset) begin
            act <= 16'b0;
            in <= 8'b0;
            acc <= 16'b0;
        end else begin
            act <= act_t;
            in <= in_t;
            acc <= acc_t;
        end
    end

    fma_int8 dut (.*);
endmodule

`endif