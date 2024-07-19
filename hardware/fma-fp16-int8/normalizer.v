`ifndef __NORMALIZER_V__
`define __NORMALIZER_V__

/*  find the index of most left 1 in the bit string.
    10000 -> 0
    00101 -> 2
    01111 -> 1
    00001 -> 4
    00000 -> 5
*/
module leftmost_one_index #(
    parameter int WIDTH = 8 // Default width
)(
    input  logic [WIDTH-1:0] in,
    output logic [$clog2(WIDTH+1)-1:0] out // the width+1 is accounted for the case then the input is all 0
);

always_comb begin
    out = WIDTH; // Initialize output
    for (int i = WIDTH-1; i >= 0; i--) begin
        if (in[i] == 1'b1) begin
            out = WIDTH-1-i;
            break;
        end
    end
end

endmodule


// round module with tie to even
module round2even #(
    parameter WIDTH_IN = 8, // input width
    parameter WIDTH_OUT = 4 // output width
)(
    input   logic [WIDTH_IN-1:0] in,
    output  logic [WIDTH_OUT-1:0] out
);
    localparam guard_idx = WIDTH_IN - WIDTH_OUT - 1; // index of the guard bit
    localparam sticky_idx = guard_idx - 1; // index of the sticky bit
    localparam round_idx = sticky_idx - 1; // index of the round bit
    // logic guard, round, sticky;
    generate
        if (WIDTH_OUT > WIDTH_IN) begin // output is more than input, padding with 0 at the right end
            assign out = {in, {(WIDTH_OUT-WIDTH_IN){1'b0}}};
        end else if (WIDTH_OUT == WIDTH_IN) begin // output is the same as input
            assign out = in;
        end else if (WIDTH_OUT == (WIDTH_IN - 1)) begin // 1 bit difference, only guard
            logic [WIDTH_OUT-1:0] temp_out;
            assign temp_out = in[WIDTH_IN-1:WIDTH_IN-WIDTH_OUT];
            logic guard;
            assign guard = in[guard_idx];
            always_comb begin
                if (guard == 1'b0) begin
                    out = temp_out;
                end else begin
                    out = temp_out + 1'b1;
                end
            end
        end else if (WIDTH_OUT == (WIDTH_IN - 2)) begin // 2 bits difference, guard and round
            logic [WIDTH_OUT-1:0] temp_out;
            assign temp_out = in[WIDTH_IN-1:WIDTH_IN-WIDTH_OUT];
            logic guard, round;
            assign guard = in[guard_idx];
            assign round = in[round_idx];
            always_comb begin
                if (guard == 1'b0) begin
                    out = temp_out;
                end else if (guard == 1'b1 && round == 1'b0) begin
                    out = temp_out;
                end else begin
                    out = temp_out + 1'b1;
                end
            end
        end else begin // 3 or more bits difference, guard, round, and sticky
            logic [WIDTH_OUT-1:0] temp_out;
            assign temp_out = in[WIDTH_IN-1:WIDTH_IN-WIDTH_OUT];
            logic guard, round, sticky;
            assign guard = in[guard_idx];
            assign round = in[round_idx];
            assign sticky = in[sticky_idx];
            always_comb begin
                if (guard == 1'b0) begin
                    out = temp_out;
                end else if (guard == 1'b1 && ((round == 1'b1) || (sticky == 1'b1)) ) begin
                    out = temp_out + 1'b1;
                end else begin // tie to even
                    if (temp_out[0] == 1'b1) begin
                        out = temp_out + 1'b1;
                    end else begin
                        out = temp_out;
                    end
                end
            end
        
        
        end
        

    endgenerate


endmodule

/*  Normalize the input mantissa to the specified width.
    The output mantissa will be shifted left or right to make the most significant bit 1, and the 1 will be removed like ieee754.
    The output exponent will be adjusted accordingly.

    This module does NOT handle denormalized numbers. denormalized numbers will output 0.
*/
module normalizer #(
    parameter EXP_WIDTH = 5,
    parameter MAN_IN_WIDTH = 15, // mantissa input
    parameter MAN_OUT_WIDTH = 10, // mantissa output
    parameter INT_LEN = 4 // integer part length
)(
    input   logic [EXP_WIDTH-1:0] exp_in,
    input   logic [MAN_IN_WIDTH-1:0] man_in,
    output  logic [EXP_WIDTH-1:0] exp_out,
    output  logic [MAN_OUT_WIDTH-1:0] man_out
);

    logic [$clog2(MAN_IN_WIDTH+1)-1:0] left_idx; // Index of the most left 1 in the input mantissa

    leftmost_one_index #(MAN_IN_WIDTH) loi (
        .in(man_in),
        .out(left_idx)
    );

    logic [MAN_IN_WIDTH-1:0] shifted_man_in; // Shifted input mantissa
    assign shifted_man_in = (man_in << left_idx) << 1; // extra shift one because the most left 1 is removed

    logic [MAN_OUT_WIDTH-1:0] rounded_shifted_man;
    round2even #(MAN_IN_WIDTH,MAN_OUT_WIDTH) r2e (
        .in(shifted_man_in),
        .out(rounded_shifted_man)
    );

    logic [EXP_WIDTH:0] ext_exp; // extended exponent, to check if overflow.
    always_comb begin
        // assign default values
        ext_exp = {1'b0, exp_in};
        exp_out = exp_in;
        man_out = man_in;

        if (left_idx < (INT_LEN - 1)) begin // input is larger and exp need increase
            ext_exp = exp_in + (INT_LEN - 1 - left_idx);
            // check overflow
            if (ext_exp[EXP_WIDTH] == 1'b1) begin
                exp_out = {EXP_WIDTH{1'b1}};
                man_out = 0;
            end else begin
                exp_out = ext_exp[EXP_WIDTH-1:0];
                man_out = rounded_shifted_man;
            end
        end else if (left_idx == (INT_LEN - 1)) begin // input is just right
            exp_out = exp_in;
            man_out = rounded_shifted_man;
        end else begin // input is smaller and exp need decrease
            ext_exp = exp_in - (left_idx - INT_LEN + 1);
            // check underflow
            if (ext_exp[EXP_WIDTH] == 1'b1) begin // 1'b1 means the most significant bit is 1 and underflow
                exp_out = 0;
                man_out = 0;
            end else begin
                exp_out = ext_exp[EXP_WIDTH-1:0];
                man_out = rounded_shifted_man;
            end
        end
    end

endmodule



`endif