// ARMSE post-processing unit: INT32 -> INT8 requantisation with optional ReLU (streaming, 2-cycle latency, no backpressure).
//   p = in_data * cfg_mult            (32-bit signed x 16-bit unsigned, exact)
//   r = (p + 2^(shift-1)) >>> shift   (shift = 0 -> r = p)
//   y = r + cfg_zp
//   out = clamp(y, relu ? cfg_zp : -128, 127)
// Bit-exact with armse/quant.py. Standalone module: NOT yet wired into the DMA writeback path (see docs).
// Timing: in_valid/in_data presented in cycle t  ->  out_valid/out_data visible in cycle t+2.
`timescale 1ns/1ps
module armse_post #(
    parameter SW = 5
)(
    input  wire                clk,
    input  wire                rst,
    input  wire [15:0]         cfg_mult,
    input  wire [SW-1:0]       cfg_shift,
    input  wire signed [7:0]   cfg_zp,
    input  wire                cfg_relu,
    input  wire                in_valid,
    input  wire signed [31:0]  in_data,
    output reg                 out_valid,
    output reg  signed [7:0]   out_data
);
    wire signed [16:0] mult_s = {1'b0, cfg_mult};
    reg                 v1;
    reg signed [48:0]   prod;

    always @(posedge clk) begin
        if (rst) begin
            v1        <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            v1        <= in_valid;
            out_valid <= v1;
        end
        prod <= in_data * mult_s;
    end

    reg signed [48:0] half, rnd, sh;
    reg signed [49:0] y, lo, hi;
    always @* begin
        half = (cfg_shift == 0) ? 49'sd0 : (49'sd1 <<< (cfg_shift - 1));
        rnd  = prod + half;
        sh   = rnd >>> cfg_shift;
        y    = sh + cfg_zp;
        lo   = cfg_relu ? cfg_zp : -50'sd128;
        hi   = 50'sd127;
    end

    always @(posedge clk) begin
        if (y < lo)      out_data <= lo[7:0];
        else if (y > hi) out_data <= hi[7:0];
        else             out_data <= y[7:0];
    end
endmodule
