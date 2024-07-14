`timescale 1ns / 1ps

`include "normalizer.v"
module testbench;
    // Parameters
    parameter EXP_WIDTH = 5;
    parameter MAN_IN_WIDTH = 15;
    parameter MAN_OUT_WIDTH = 10;
    parameter INT_LEN = 4;

    // Inputs and Outputs
    logic [EXP_WIDTH-1:0] exp_in;
    logic [MAN_IN_WIDTH-1:0] man_in;
    logic [EXP_WIDTH-1:0] exp_out;
    logic [MAN_OUT_WIDTH-1:0] man_out;

    // Instantiate the module
    normalizer #(EXP_WIDTH, MAN_IN_WIDTH, MAN_OUT_WIDTH, INT_LEN) uut (
        .exp_in(exp_in),
        .man_in(man_in),
        .exp_out(exp_out),
        .man_out(man_out)
    );

    initial begin
        // Test cases
        $display("Starting test...");

        // Test Case 1
        exp_in = 5'b01010;
        man_in = 15'b101010101010101;
        #10;
        $display("Input: exp_in = %d (%b), man_in = %b", exp_in, exp_in, man_in);
        $display("Output: exp_out = %d (%b), man_out = %b", exp_out, exp_out, man_out);

        // Test Case 2
        exp_in = 5'b00100;
        man_in = 15'b110011001100110;
        #10;
        $display("Input: exp_in = %d (%b), man_in = %b", exp_in, exp_in, man_in);
        $display("Output: exp_out = %d (%b), man_out = %b", exp_out, exp_out, man_out);

        // Test Case 3
        exp_in = 5'b11111;
        man_in = 15'b111111111111111;
        #10;
        $display("Input: exp_in = %d (%b), man_in = %b", exp_in, exp_in, man_in);
        $display("Output: exp_out = %d (%b), man_out = %b", exp_out, exp_out, man_out);

        // Test Case 4
        exp_in = 5'b00000;
        man_in = 15'b000000000000001;
        #10;
        $display("Input: exp_in = %d (%b), man_in = %b", exp_in, exp_in, man_in);
        $display("Output: exp_out = %d (%b), man_out = %b", exp_out, exp_out, man_out);

        // Test Case 5
        exp_in = 5'b10111;
        man_in = 15'b001000000000110;
        #10;
        $display("Input: exp_in = %d (%b), man_in = %b", exp_in, exp_in, man_in);
        $display("Output: exp_out = %d (%b), man_out = %b", exp_out, exp_out, man_out);

        $display("Test completed.");
        $finish;
    end
endmodule


