// Unit test for armse_skew_mem: rotation/de-rotation round trip and conflict monitor.
`timescale 1ns/1ps
module tb_skew_mem;
    localparam NB = 8, LOGNB = 3, DEPTH = 64, AW = 12;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg [3:0] w_en = 4'd0;
    reg [4*LOGNB-1:0] w_bank = 0;
    reg [4*AW-1:0] w_addr = 0;
    reg [31:0] w_data = 0;
    reg r_en = 1'b0;
    reg [AW-1:0] r_addr = 0;
    reg [LOGNB-1:0] r_par = 0;
    wire [8*NB-1:0] r_data;
    wire [31:0] conflict_cnt;
    armse_skew_mem #(.NB(NB), .LOGNB(LOGNB), .DEPTH(DEPTH), .AW(AW)) dut (
        .clk(clk), .w_en(w_en), .w_bank(w_bank), .w_addr(w_addr), .w_data(w_data),
        .r_en(r_en), .r_addr(r_addr), .r_par(r_par), .r_data(r_data), .conflict_cnt(conflict_cnt));
    integer row, k, l, errs;
    reg [7:0] val;
    initial begin
        errs = 0;
        // store a 5-row x 16-col tile element-by-element, one element per cycle, skewed mapping
        for (row = 0; row < 5; row = row + 1)
            for (k = 0; k < 16; k = k + 1) begin
                @(negedge clk);
                w_en = 4'b0001;
                w_bank[LOGNB-1:0] = row + k;
                w_addr[AW-1:0] = row * 2 + (k >> LOGNB);
                w_data = {24'd0, 8'(row * 16 + k + 1)};
            end
        @(negedge clk); w_en = 4'd0;
        // read back lanes for each row/group and compare
        for (row = 0; row < 5; row = row + 1)
            for (k = 0; k < 2; k = k + 1) begin
                @(negedge clk); r_en = 1'b1; r_addr = row * 2 + k; r_par = row;
                @(negedge clk); r_en = 1'b0;
                for (l = 0; l < NB; l = l + 1) begin
                    val = r_data[l*8 +: 8];
                    if (val !== 8'(row * 16 + k * NB + l + 1)) begin
                        errs = errs + 1;
                        $display("ERROR row %0d g %0d lane %0d got %0d", row, k, l, val);
                    end
                end
            end
        // conflict monitor: two writes to the same bank must be counted, distinct banks must not
        @(negedge clk); w_en = 4'b0011; w_bank = {{(4*LOGNB-2*LOGNB){1'b0}}, 3'd2, 3'd2};
        @(negedge clk); w_en = 4'b0011; w_bank = {{(4*LOGNB-2*LOGNB){1'b0}}, 3'd3, 3'd2};
        @(negedge clk); w_en = 4'd0;
        @(negedge clk);
        if (conflict_cnt !== 32'd1) begin errs = errs + 1; $display("ERROR conflict_cnt=%0d expected 1", conflict_cnt); end
        if (errs == 0) $display("RESULT tb_skew_mem: PASS"); else $display("RESULT tb_skew_mem: FAIL (%0d)", errs);
        $finish;
    end
endmodule
