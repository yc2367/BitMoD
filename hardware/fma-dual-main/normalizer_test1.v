`timescale 1ns / 1ps

`include "normalizer.v"
module testbench;
    // Parameters
    parameter int WIDTH = 5;
    
    // Inputs and Outputs
    logic [WIDTH-1:0] in;
    logic [$clog2(WIDTH)-1:0] out;
    
    // Instantiate the module
    leftmost_one_index #(WIDTH) uut (
        .in(in),
        .out(out)
    );
    
    initial begin
        // Test cases
        $display("Starting test...");

        in = 5'b10000; #10;
        $display("Input: %b, Output: %d", in, out);

        in = 5'b00101; #10;
        $display("Input: %b, Output: %d", in, out);

        in = 5'b01111; #10;
        $display("Input: %b, Output: %d", in, out);

        in = 5'b00001; #10;
        $display("Input: %b, Output: %d", in, out);

        in = 5'b01000; #10;
        $display("Input: %b, Output: %d", in, out);

        in = 5'b00000; #10;
        $display("Input: %b, Output: %d", in, out);
        
        $display("Test completed.");
        $finish;
    end
endmodule
