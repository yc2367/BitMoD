// 3 bits integer to booth encoding
// sign-exp-man: 1-1-1
module intbooth
(
    input   logic  [2:0] in,
    output  logic  [2:0] out
);

    always_comb begin
        case (in)
            3'b000: out = 3'b000;
            3'b001: out = 3'b001;
            3'b010: out = 3'b001;
            3'b011: out = 3'b011;
            3'b100: out = 3'b111;
            3'b101: out = 3'b101;
            3'b110: out = 3'b101;
            3'b111: out = 3'b000;
        endcase
    end

endmodule

// convert fp4 to fixed point int
// 3 bits input (fp4 exp-man) no sign
// 4 bits output: 3 bits integer, 1 bit fraction
module fpcvt
(
    input    logic  [2:0] in,
    output   logic  [3:0] out
);


    always_comb begin
        case (in)
            3'b000: out = 4'b0000; // 0 
            3'b001: out = 4'b0001; // 0.5
            3'b010: out = 4'b0010; // 1
            3'b011: out = 4'b0011; // 1.5
            3'b100: out = 4'b0100; // 2
            3'b101: out = 4'b0110; // 3
            3'b110: out = 4'b1000; // 4
            3'b111: out = 4'b1100; // 6
        
        endcase
    end


endmodule

module LOD4
(
    input   logic  [3:0]  in,
    output  logic  [1:0]  exp,
    output  logic         man
);
    always_comb begin
        if (in[3]) exp = 0;
        else if (in[2]) exp = 1;
        else if (in[1]) exp = 2;
        else if (in[0]) exp = 3;
        else exp = 0;
    
        man = in[0];
    end

endmodule





module serial_reader 
(
    input   logic         clk,
    input   logic         reset,
    input   logic         go,

    input   logic  [1:0]  mode, // 0: int8, 1: int6, 2: fp4
    input   logic  [7:0]  data,

    output  logic         sign,
    output  logic  [1:0]  exp,
    output  logic         mantissa,
    output  logic  [2:0]  bsig
);

    typedef enum logic [$clog2(4)-1:0] {
        STATE_IDLE, // waiting for start
        STATE_INT8, // int8 output loop
        STATE_INT6, // int6 output loop
        STATE_FP4   // fp4 output loop
    } state_t;

    // data counter state registers
    logic   [8:0]       data_ext_reg, data_ext_reg_next;
    logic   [8:0]       data_cvt;
    logic   [1:0]       counter, counter_next;
    state_t state, state_next;

    // state transition registers
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= STATE_IDLE;
            data_ext_reg <= 0;
            counter <= 0;
        end else begin
            state <= state_next;
            data_ext_reg <= data_ext_reg_next;
            counter <= counter_next;
        end
    end

    // logic for state machine transition
    always_comb begin
        state_next = state;
        data_ext_reg_next = data_ext_reg;
        counter_next = counter;

        case (state)
            STATE_IDLE: begin
                if (go) begin
                    // reset counter and fill data
                    counter_next = 0;
                    data_ext_reg_next = data_cvt;
                    // select mode
                    if (mode == 0) begin
                        state_next = STATE_INT8;
                    end else if (mode == 1) begin
                        state_next = STATE_INT6;
                    end else if (mode == 2) begin
                        state_next = STATE_FP4;
                    end 
                end
            end
            STATE_INT8: begin
                counter_next = counter + 1;
                if (counter == 3) begin // in total 4 stages
                    state_next = STATE_IDLE;
                end
            end

            STATE_INT6: begin
                counter_next = counter + 1;
                if (counter == 3) begin // in total 3 stages
                    state_next = STATE_IDLE;
                end
            end

            STATE_FP4: begin
                counter_next = counter + 1;
                if (counter == 2) begin // in total 2 stages
                    state_next = STATE_IDLE;
                end
            end

        endcase
    end


    logic [3:0] fixed_point;
    fpcvt cvt (
        .in(data[2:0]),
        .out(fixed_point)
    );


    logic [2:0] booth_in;
    logic [2:0] booth_out;
    intbooth booth (
        .in(booth_in),
        .out(booth_out)
    );

    always_comb begin 
        if (counter == 2'd0) booth_in = data_ext_reg[2: 0];
        else if (counter == 2'd1) booth_in = data_ext_reg[4: 2];
        else if (counter == 2'd2) booth_in = data_ext_reg[6: 4];
        else if (counter == 2'd3) booth_in = data_ext_reg[8: 6];
        else booth_in = 3'bxxx;
    end

    logic [3:0] lod_in;
    logic [1:0] lod_exp;
    logic       lod_man;
    LOD4 loddd (
        .in(lod_in),
        .exp(lod_exp),
        .man(lod_man)
    );
    always_comb begin 
        if (counter == 2'd0) lod_in = data_ext_reg[3: 0];
        else if (counter == 2'd1) lod_in = data_ext_reg[4: 1];
        else lod_in = 4'bxxxx;
    end

    // combinational logic for output data
    always_comb begin
        // encoding value:
        case (mode)
            2'b00: begin // int8
                data_cvt = {data, 1'b0}; // pad zero
            end
            2'b01: begin // int6
                data_cvt = {data, 1'b0}; // pad zero
            end
            2'b10: begin // fp4
                if (data == 8'b0000_1000) begin
                    data_cvt = 9'b0_0001_0000; // dec: 8, special value
                end else begin
                    data_cvt = {data[3], 4'd0, fixed_point}; // sign, 4-bit padding, 4 bit value
                end
            end
            default: begin
                data_cvt = 0;
            end
        endcase
    

        // output data
        sign = 0;
        exp = 0;
        mantissa = 0;
        bsig = 0;
        
        case (state)
        STATE_INT8: begin
            sign = booth_out[2];
            exp = {1'b0, booth_out[1]};
            mantissa = booth_out[0];
            bsig = 2 * counter;
        end

        STATE_INT6: begin
            sign = booth_out[2];
            exp = {1'b0, booth_out[1]};
            mantissa = booth_out[0];
            bsig = 2 * counter;
        end

        STATE_FP4: begin
            sign = data_ext_reg[8];
            exp = lod_exp;
            mantissa = lod_man;
            bsig = counter;
        end
        endcase
    
    end




endmodule