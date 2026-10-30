// Self-checking testbench for armse_post. Vectors: python3 -m armse.quant  (post_vectors/pNN/{in.hex,exp.hex,meta.vh}).
`timescale 1ns/1ps
`include "meta.vh"
module tb_post;
    reg clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;
    reg in_valid = 1'b0;
    reg signed [31:0] in_data = 32'sd0;
    wire out_valid;
    wire signed [7:0] out_data;

    armse_post dut (.clk(clk), .rst(rst), .cfg_mult(`M_P), .cfg_shift(`S_P), .cfg_zp(`ZP_P), .cfg_relu(`RELU_P),
                    .in_valid(in_valid), .in_data(in_data), .out_valid(out_valid), .out_data(out_data));

    reg [31:0] xin  [0:`N_P-1];
    reg [7:0]  xexp [0:`N_P-1];
    integer i, got, errs, cyc, first_in, first_out;

    initial begin
        $readmemh("in.hex", xin);
        $readmemh("exp.hex", xexp);
        got = 0; errs = 0; cyc = 0; first_in = -1; first_out = -1;
        repeat (4) @(posedge clk);
        @(negedge clk); rst = 1'b0;
        for (i = 0; i < `N_P; i = i + 1) begin
            @(negedge clk); in_valid = 1'b1; in_data = xin[i];
            if (i % 7 == 6) begin @(negedge clk); in_valid = 1'b0; end      // idle bubbles: valid must be tracked per sample
        end
        @(negedge clk); in_valid = 1'b0;
        repeat (6) @(posedge clk);
        if (got != `N_P) begin $display("ERROR %s: got %0d outputs, expected %0d", `TEST_NAME, got, `N_P); errs = errs + 1; end
        if (first_out - first_in != 2) begin $display("ERROR %s: latency %0d, expected 2", `TEST_NAME, first_out - first_in); errs = errs + 1; end
        if (errs == 0) $display("RESULT %s: PASS", `TEST_NAME); else $display("RESULT %s: FAIL (%0d errors)", `TEST_NAME, errs);
        $finish;
    end

    always @(posedge clk) begin
        cyc = cyc + 1;
        if (in_valid && first_in < 0) first_in = cyc;
        if (out_valid) begin
            if (first_out < 0) first_out = cyc;
            if (got < `N_P && out_data !== xexp[got]) begin
                if (errs < 8) $display("ERROR %s: sample %0d in=%h got %0d expected %0d", `TEST_NAME, got, xin[got], out_data, $signed(xexp[got]));
                errs = errs + 1;
            end
            got = got + 1;
        end
    end
endmodule
