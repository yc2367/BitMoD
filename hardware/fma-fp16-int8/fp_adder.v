`ifndef __FP_ADDER_V__
`define __FP_ADDER_V__


/*  arbitrary float point adder core module
    input:  sign1, exp1real, man1 (with parameterized integer part lengt)
            sign2, exp2real, man2 (with parameterized integer part lengt)
        require the man2 total length and integer part length is equal or larger than man1 
    output  sign, expreal, man (integer part length is of length 2)

*/

module fp_adder_core #(
    parameter EXP_LEN = 6,
    parameter MAN_LEN_1 = 11,
    parameter MAN_LEN_2 = 15,

    parameter INT_LEN_1 = 1,
    parameter INT_LEN_2 = 3
)(
    input   logic sign1,
    input   logic sign2,
    input   logic [EXP_LEN-1:0] exp1real,
    input   logic [EXP_LEN-1:0] exp2real,
    input   logic [MAN_LEN_1-1:0] man1,
    input   logic [MAN_LEN_2-1:0] man2,

    output  logic sign_out,
    output  logic [EXP_LEN-1:0] exp_out,
    output  logic [MAN_LEN_2:0] man_out  // 1 more bit for overflow, so it contains (INT_LEN_2+1) bits of integer, and remaining bits of fraction
);

    // padding mantissa 1 to match the place of mantissa 2:  000 man1 000
    logic [MAN_LEN_2-1:0] man1_padded;
    assign man1_padded = {{(MAN_LEN_2 - MAN_LEN_1 - (INT_LEN_2 - INT_LEN_1)){1'b0}}, man1, {(INT_LEN_2 - INT_LEN_1){1'b0}}};

    // align mantissa
    logic [MAN_LEN_2-1:0] man1_aligned, man2_aligned;
    logic [EXP_LEN-1:0] exp_aligned;

    // shift mantissa based on the number.
    always_comb begin
        if (exp1real > exp2real) begin
            man1_aligned = man1_padded >> (exp1real - exp2real);
            man2_aligned = man2;
            exp_aligned = exp1real;
        end else begin
            man1_aligned = man1_padded;
            man2_aligned = man2 >> (exp2real - exp1real);
            exp_aligned = exp2real;
        end
    end
    
    logic [MAN_LEN_2:0] man1_aligned_ext;
    assign man1_aligned_ext = {1'b0, man1_aligned};
    logic [MAN_LEN_2:0] man2_aligned_ext;
    assign man2_aligned_ext = {1'b0, man2_aligned};


    // add or subtract based on the sign
    logic diff;
    assign diff = sign1 ^ sign2;
    logic x2_bigger;
    assign x2_bigger = man2_aligned > man1_aligned;
    
    always_comb begin
        exp_out = exp_aligned;
        if (diff) begin // differen sign, subtraction
            if (x2_bigger) begin
                sign_out = sign2;
                man_out = man2_aligned_ext - man1_aligned_ext;
            end else begin
                sign_out = sign1;
                man_out = man1_aligned_ext - man2_aligned_ext;
            end
        end else begin // same sign, addition
            sign_out = sign1;
            man_out = man1_aligned_ext + man2_aligned_ext;
        end
    end


endmodule

`endif