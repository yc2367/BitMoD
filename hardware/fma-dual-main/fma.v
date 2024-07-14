`ifndef __FMA_V__
`define __FMA_V__

`include "multiplier.v"
`include "normalizer.v"
`include "fp_adder.v"

/*  dual function FMA module, FP16 + FP16 * INT8/2FP4 -> FP16
    mode 0: acc2_out = acc2 + act * in
            acc1 does not care.

    mode 1: acc1_out = acc1 + act * in[7:4] (fp4 format)
            acc2_out = acc2 + act * in[3:0] (fp4 format)
    
*/
module dualFMA
(
    input   logic        clk,
    input   logic        reset,

    input   logic [15:0] act, // fp16 activation
    input   logic [7:0]  in, // 8-bit input in int8/ 2fp4 format
    input   logic        mode, // 0: int8, 1: 2fp4
    input   logic [15:0] acc1, // fp16 accumulation 1
    input   logic [15:0] acc2, // fp16 accumulation 2

    output  logic [15:0] acc1_out, // fp16 accumulation 1
    output  logic [15:0] acc2_out // fp16 accumulation 2
);
    
    // pipeline registers

    logic [15:0] act_reg;
    logic [7:0]  in_reg;
    logic        mode_reg0;
    logic [15:0] acc1_reg0;
    logic [15:0] acc2_reg0;

    always_ff @(posedge clk) begin
        if (reset) begin
            act_reg <= 16'b0;
            in_reg <= 8'b0;
            mode_reg0 <= 1'b0;
            acc1_reg0 <= 16'b0;
            acc2_reg0 <= 16'b0;
        end else begin
            act_reg <= act;
            in_reg <= in;
            mode_reg0 <= mode;
            acc1_reg0 <= acc1;
            acc2_reg0 <= acc2;
        end
        
    end

    // dual multiplier 
    logic           sign_out1;
    logic           sign_out2;
    logic [5:0]     exp_out1;
    logic [5:0]     exp_out2;
    logic [14:0]    man_out1;
    logic [14:0]    man_out2;
    logic           sign_out_int;
    logic [4:0]     exp_out_int;
    logic [18:0]    man_out_int;

    
    dualMultiplier dM (
        .act(act_reg),
        .in(in_reg),
        .mode(mode_reg0),

        .sign_out1(sign_out1),
        .sign_out2(sign_out2),
        .exp_out1(exp_out1),
        .exp_out2(exp_out2),
        .man_out1(man_out1),
        .man_out2(man_out2),

        .sign_out_int(sign_out_int),
        .exp_out_int(exp_out_int),
        .man_out_int(man_out_int)
    );


    logic           sign_out1_reg;
    logic           sign_out2_reg;
    logic [5:0]     exp_out1_reg;
    logic [5:0]     exp_out2_reg;
    logic [14:0]    man_out1_reg;
    logic [14:0]    man_out2_reg;
    logic           sign_out_int_reg;
    logic [4:0]     exp_out_int_reg;
    logic [18:0]    man_out_int_reg;

    logic [15:0] acc1_reg;
    logic [15:0] acc2_reg;

    logic        mode_reg;
    // second stage pipeline

    always_ff @(posedge clk) begin
        if (reset) begin
            sign_out1_reg <= 1'b0;
            sign_out2_reg <= 1'b0;
            exp_out1_reg <= 6'b0;
            exp_out2_reg <= 6'b0;
            man_out1_reg <= 15'b0;
            man_out2_reg <= 15'b0;
            sign_out_int_reg <= 1'b0;
            exp_out_int_reg <= 5'b0;
            man_out_int_reg <= 19'b0;

            acc1_reg <= 16'b0;
            acc2_reg <= 16'b0;

            mode_reg <= 1'b0;
        end else begin
            sign_out1_reg <= sign_out1;
            sign_out2_reg <= sign_out2;
            exp_out1_reg <= exp_out1;
            exp_out2_reg <= exp_out2;
            man_out1_reg <= man_out1;
            man_out2_reg <= man_out2;
            sign_out_int_reg <= sign_out_int;
            exp_out_int_reg <= exp_out_int;
            man_out_int_reg <= man_out_int;

            acc1_reg <= acc1_reg0;
            acc2_reg <= acc2_reg0;

            mode_reg <= mode_reg0;
        end
    end

    // decompose accumulation to sign, exponent, mantissa

    localparam EXP_ACT_CALC_OFFSET = 15;

    logic           acc1_sign;
    logic [4:0]     acc1_exp_real;
    logic [10:0]    acc1_man_ext;
    assign acc1_sign = acc1_reg[15];
    assign acc1_exp_real = acc1_reg[14:10] - EXP_ACT_CALC_OFFSET;
    assign acc1_man_ext = { (acc1_reg[14:10]) == 0 ? 1'b0 : 1'b1, acc1_reg[9:0]};

    logic           acc2_sign;
    logic [4:0]     acc2_exp_real;
    logic [10:0]    acc2_man_ext;
    assign acc2_sign = acc2_reg[15];
    assign acc2_exp_real = acc2_reg[14:10] - EXP_ACT_CALC_OFFSET;
    assign acc2_man_ext = { (acc2_reg[14:10]) == 0 ? 1'b0 : 1'b1, acc2_reg[9:0]};


    // prepare input for accumulate adder route 1
    
    logic sign_adder_route_1;
    logic [4:0] exp_adder_route_1;
    logic [14:0] man_adder_route_1;

    assign sign_adder_route_1 = sign_out1_reg;
    assign exp_adder_route_1 = exp_out1_reg;
    assign man_adder_route_1 = man_out1_reg;

    // prepare input for accumulate adder route 2

    logic sign_adder_route_2;
    logic [4:0] exp_adder_route_2;
    logic [20:0] man_adder_route_2;

    Mux2 #(1) mux_sign_route_2 (
        .in1(sign_out_int_reg),
        .in2(sign_out2_reg),
        .sel(mode_reg),
        .out(sign_adder_route_2)
    );

    Mux2 #(5) mux_exp_route_2 (
        .in1(exp_out_int_reg),
        .in2(exp_out2_reg[4:0]), // TODO: truncated, need extra modification
        .sel(mode_reg),
        .out(exp_adder_route_2)
    );

    Mux2 #(21) mux_man_route_2 (
        .in1({man_out_int_reg, 2'd0}),
        .in2({6'd0, man_out2_reg}),
        .sel(mode_reg),
        .out(man_adder_route_2)
    );

    // accumulate adders
    

    logic           acc1_sign_out;
    logic [4:0]     acc1_exp_out;
    logic [15:0]    acc1_man_out;


    fp_adder_core #(5, 11, 15, 1, 3) acc1_adder (
        .sign1(acc1_sign),
        .sign2(sign_adder_route_1),
        .exp1real(acc1_exp_real),
        .exp2real(exp_adder_route_1),
        .man1(acc1_man_ext),
        .man2(man_adder_route_1),

        .sign_out(acc1_sign_out),
        .exp_out(acc1_exp_out),
        .man_out(acc1_man_out)
    );



    logic           acc2_sign_out;
    logic [4:0]     acc2_exp_out;
    logic [21:0]    acc2_man_out;

    fp_adder_core #(5, 11, 21, 1, 9) acc2_adder (
        .sign1(acc2_sign),
        .sign2(sign_adder_route_2),
        .exp1real(acc2_exp_real),
        .exp2real(exp_adder_route_2),
        .man1(acc2_man_ext),
        .man2(man_adder_route_2),

        .sign_out(acc2_sign_out),
        .exp_out(acc2_exp_out),
        .man_out(acc2_man_out)
    );

    // final normalizer

    logic [15:0] acc1_out_assemble;
    logic [15:0] acc2_out_assemble;

    logic [4:0] final_acc1_out_exp;
    logic [9:0] final_acc1_out_man;
    normalizer #(5, 16, 10, 3) norm1 (
        .exp_in(acc1_exp_out),
        .man_in(acc1_man_out),
        .exp_out(final_acc1_out_exp),
        .man_out(final_acc1_out_man)
    );
    assign acc1_out_assemble = {acc1_sign_out, final_acc1_out_exp, final_acc1_out_man};

    logic [4:0] final_acc2_out_exp;
    logic [9:0] final_acc2_out_man;
    normalizer #(5, 22, 10, 9) norm2 (
        .exp_in(acc2_exp_out),
        .man_in(acc2_man_out),
        .exp_out(final_acc2_out_exp),
        .man_out(final_acc2_out_man)
    );
    assign acc2_out_assemble = {acc2_sign_out, final_acc2_out_exp, final_acc2_out_man};

    // output

    always_ff @(posedge clk) begin
        if (reset) begin
            acc1_out <= 16'b0;
            acc2_out <= 16'b0;
        end else begin
            acc1_out <= acc1_out_assemble;
            acc2_out <= acc2_out_assemble;
        end
    end

endmodule
`endif