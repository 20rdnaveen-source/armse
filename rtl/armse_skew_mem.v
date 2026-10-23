// ARMSE skewed banked buffer (used for both the A and the B operand buffers).
//
// Logical element (row r, col c) lives in bank (r + c) mod NB.  The DMA engine computes the
// bank and word address of every element it writes (up to 4 per cycle); compute reads ONE word
// address from ALL banks in a single cycle and the module de-rotates the NB bytes into lane order.
//   - reads : NB elements (consecutive c, fixed r) hit NB different banks   -> conflict free
//   - writes: 4 consecutive elements of a row or a column hit 4 different banks -> conflict free
// conflict_cnt is a run-time monitor: cycles in which two enabled write elements target one bank
// (must stay 0; the mapping guarantees it, the counter proves it in simulation).
// Every bank has 1 synchronous read port + 1 write port (simple dual port -> maps to BRAM).
`timescale 1ns/1ps
module armse_skew_mem #(
    parameter NB    = 8,
    parameter LOGNB = 3,
    parameter DEPTH = 256,      // words per bank (all slots)
    parameter AW    = 12
)(
    input  wire                 clk,
    input  wire [3:0]           w_en,
    input  wire [4*LOGNB-1:0]   w_bank,
    input  wire [4*AW-1:0]      w_addr,
    input  wire [31:0]          w_data,     // 4 bytes, element q = byte q
    input  wire                 r_en,
    input  wire [AW-1:0]        r_addr,     // same word address presented to every bank
    input  wire [LOGNB-1:0]     r_par,      // rotation = row index mod NB
    output wire [8*NB-1:0]      r_data,     // lane l in bits [8l+7:8l], valid one cycle after r_en
    output reg  [31:0]          conflict_cnt
);
    wire [8*NB-1:0] bank_rd;
    reg  [LOGNB-1:0] par_q;

    initial begin
        conflict_cnt = 32'd0;
        par_q        = {LOGNB{1'b0}};
    end

    always @(posedge clk) if (r_en) par_q <= r_par;

    genvar b;
    generate
        for (b = 0; b < NB; b = b + 1) begin : g_bank
            reg [7:0] mem [0:DEPTH-1];
            reg [7:0] rd;
            integer q, z;
            initial begin
                rd = 8'd0;
                for (z = 0; z < DEPTH; z = z + 1) mem[z] = 8'd0;
            end
            always @(posedge clk) begin
                for (q = 0; q < 4; q = q + 1)
                    if (w_en[q] && (w_bank[q*LOGNB +: LOGNB] == b))
                        mem[w_addr[q*AW +: AW]] <= w_data[q*8 +: 8];
                if (r_en) rd <= mem[r_addr];
            end
            assign bank_rd[b*8 +: 8] = rd;
        end
    endgenerate

    // lane l was stored in bank (l + r) mod NB
    genvar l;
    generate
        for (l = 0; l < NB; l = l + 1) begin : g_lane
            wire [LOGNB-1:0] bsel = par_q + l;
            assign r_data[l*8 +: 8] = bank_rd[bsel*8 +: 8];
        end
    endgenerate

    integer ci, cj;
    reg conf;
    always @* begin
        conf = 1'b0;
        for (ci = 0; ci < 4; ci = ci + 1)
            for (cj = ci + 1; cj < 4; cj = cj + 1)
                if (w_en[ci] && w_en[cj] &&
                    (w_bank[ci*LOGNB +: LOGNB] == w_bank[cj*LOGNB +: LOGNB]))
                    conf = 1'b1;
    end
    always @(posedge clk) if (conf) conflict_cnt <= conflict_cnt + 32'd1;
endmodule
