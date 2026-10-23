// ARMSE output / partial-sum buffer: 32-bit words.
//  compute port : 1 synchronous read + 1 write per cycle
//  DMA port     : 1 asynchronous read (LUTRAM style; a BRAM implementation would need one more
//                 pipeline stage in the DMA, which is NOT modelled here)
`timescale 1ns/1ps
module armse_obuf #(
    parameter DEPTH = 1024,
    parameter AW    = 14
)(
    input  wire          clk,
    input  wire          c_re,
    input  wire [AW-1:0] c_raddr,
    output reg  [31:0]   c_rdata,
    input  wire          c_we,
    input  wire [AW-1:0] c_waddr,
    input  wire [31:0]   c_wdata,
    input  wire [AW-1:0] d_raddr,
    output wire [31:0]   d_rdata
);
    reg [31:0] mem [0:DEPTH-1];
    integer z;
    initial begin
        c_rdata = 32'd0;
        for (z = 0; z < DEPTH; z = z + 1) mem[z] = 32'd0;
    end
    always @(posedge clk) begin
        if (c_we) mem[c_waddr] <= c_wdata;
        if (c_re) c_rdata <= mem[c_raddr];
    end
    assign d_rdata = mem[d_raddr];
endmodule
