`ifndef __mux_9to1_with_val_V__
`define __mux_9to1_with_val_V__

module mux_9to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [8:0], 
    input  logic [3:0]             sel,     
    input  logic                   val,     

    output logic [DATA_WIDTH-1:0]  out
);
    logic [DATA_WIDTH-1:0]  out_tmp;
    always_comb begin
        case (sel) // synopsys infer_mux
            4'b0000: out_tmp = vec[0];
            4'b0001: out_tmp = vec[1];
            4'b0010: out_tmp = vec[2];
            4'b0011: out_tmp = vec[3];
            4'b0100: out_tmp = vec[4];
            4'b0101: out_tmp = vec[5];
            4'b0110: out_tmp = vec[6];
            4'b0111: out_tmp = vec[7];
            4'b1000: out_tmp = vec[8];
            default:  out_tmp = {DATA_WIDTH{1'bx}};
        endcase
    end

    always_comb begin
        if ( val ) begin
            out = out_tmp;
        end else begin
            out = 0;
        end
    end
endmodule

`endif