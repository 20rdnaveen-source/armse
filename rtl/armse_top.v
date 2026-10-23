// ARMSE top level: program memories + DMA engine + compute engine + banked buffers + counters.
//
// Host usage:  1) pulse rst  2) write both programs through h_* 3) hold n_dma/n_cmp  4) pulse go
//              5) wait done  6) read counters.   `cycles` = cycle index (go pulse = cycle 0) at which
//              both programs had completed (matches the Python model's `cycles`).
`timescale 1ns/1ps
`include "armse_fields.vh"
module armse_top #(
    parameter NB         = 8,
    parameter LOGNB      = 3,
    parameter A_DEPTH    = 256,     // words per bank, A buffer (all slots)
    parameter B_DEPTH    = 256,
    parameter O_DEPTH    = 1024,    // 32-bit words, output buffer (all slots)
    parameter PROG_DEPTH = 4096     // descriptors per program memory
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 go,
    input  wire [`CNT_W-1:0]    n_dma,
    input  wire [`CNT_W-1:0]    n_cmp,
    // host program-write port
    input  wire                 h_we,
    input  wire                 h_sel,        // 0 = DMA program, 1 = compute program
    input  wire [15:0]          h_addr,
    input  wire [127:0]         h_wdata,
    output wire                 busy,
    output wire                 done,
    output wire [31:0]          cycles,
    // external memory port
    output wire                 x_cmd_valid,
    input  wire                 x_cmd_ready,
    output wire                 x_cmd_write,
    output wire [31:0]          x_cmd_addr,
    output wire [7:0]           x_cmd_beats,
    input  wire                 x_rvalid,
    input  wire [31:0]          x_rdata,
    output wire                 x_wvalid,
    output wire [31:0]          x_wdata,
    input  wire                 x_wready,
    // counters
    output wire [31:0]          cmp_busy,
    output wire [31:0]          cmp_wait,
    output wire [31:0]          dma_busy,
    output wire [31:0]          dma_wait,
    output wire [31:0]          bytes_rd,
    output wire [31:0]          bytes_wr,
    output wire [31:0]          n_xfer,
    output wire [31:0]          bank_conflicts
);
    localparam AAW = 12, OAW = 14;

    reg [127:0] dprog [0:PROG_DEPTH-1];
    reg [127:0] cprog [0:PROG_DEPTH-1];
    always @(posedge clk) if (h_we) begin
        if (h_sel) cprog[h_addr] <= h_wdata; else dprog[h_addr] <= h_wdata;
    end

    wire [`CNT_W-1:0] d_idx, c_idx, dma_done, cp_done;
    wire [127:0] d_ent = dprog[d_idx];
    wire [127:0] c_ent = cprog[c_idx];

    reg run, fin;
    reg [31:0] cyc;
    wire done_w = run && (dma_done == n_dma) && (cp_done == n_cmp);
    always @(posedge clk) begin
        if (rst) begin run <= 1'b0; fin <= 1'b0; cyc <= 32'd0; end
        else if (go) begin run <= 1'b1; fin <= 1'b0; cyc <= 32'd1; end
        else if (run && !fin) begin
            if (done_w) fin <= 1'b1; else cyc <= cyc + 32'd1;
        end
    end
    assign busy = run && !fin;
    assign done = fin;
    assign cycles = cyc;

    // buffer wires
    wire [3:0] a_w_en, b_w_en;
    wire [4*LOGNB-1:0] a_w_bank, b_w_bank;
    wire [4*AAW-1:0] a_w_addr, b_w_addr;
    wire [31:0] a_w_data, b_w_data;
    wire a_r_en, b_r_en;
    wire [AAW-1:0] a_r_addr, b_r_addr;
    wire [LOGNB-1:0] a_r_par, b_r_par;
    wire [8*NB-1:0] a_r_data, b_r_data;
    wire o_re, o_we;
    wire [OAW-1:0] o_raddr, o_waddr, od_raddr;
    wire [31:0] o_rdata, o_wdata, od_rdata;
    wire [31:0] conf_a, conf_b;

    armse_skew_mem #(.NB(NB), .LOGNB(LOGNB), .DEPTH(A_DEPTH), .AW(AAW)) u_abuf (
        .clk(clk), .w_en(a_w_en), .w_bank(a_w_bank), .w_addr(a_w_addr), .w_data(a_w_data),
        .r_en(a_r_en), .r_addr(a_r_addr), .r_par(a_r_par), .r_data(a_r_data), .conflict_cnt(conf_a));
    armse_skew_mem #(.NB(NB), .LOGNB(LOGNB), .DEPTH(B_DEPTH), .AW(AAW)) u_bbuf (
        .clk(clk), .w_en(b_w_en), .w_bank(b_w_bank), .w_addr(b_w_addr), .w_data(b_w_data),
        .r_en(b_r_en), .r_addr(b_r_addr), .r_par(b_r_par), .r_data(b_r_data), .conflict_cnt(conf_b));
    armse_obuf #(.DEPTH(O_DEPTH), .AW(OAW)) u_obuf (
        .clk(clk), .c_re(o_re), .c_raddr(o_raddr), .c_rdata(o_rdata),
        .c_we(o_we), .c_waddr(o_waddr), .c_wdata(o_wdata),
        .d_raddr(od_raddr), .d_rdata(od_rdata));

    armse_dma #(.NB(NB), .LOGNB(LOGNB), .AAW(AAW), .OAW(OAW)) u_dma (
        .clk(clk), .rst(rst), .clr(go), .run(run), .n_dma(n_dma), .cp_done(cp_done),
        .dma_done(dma_done), .p_idx(d_idx), .p_data(d_ent),
        .x_cmd_valid(x_cmd_valid), .x_cmd_ready(x_cmd_ready), .x_cmd_write(x_cmd_write),
        .x_cmd_addr(x_cmd_addr), .x_cmd_beats(x_cmd_beats), .x_rvalid(x_rvalid), .x_rdata(x_rdata),
        .x_wvalid(x_wvalid), .x_wdata(x_wdata), .x_wready(x_wready),
        .a_w_en(a_w_en), .a_w_bank(a_w_bank), .a_w_addr(a_w_addr), .a_w_data(a_w_data),
        .b_w_en(b_w_en), .b_w_bank(b_w_bank), .b_w_addr(b_w_addr), .b_w_data(b_w_data),
        .o_raddr(od_raddr), .o_rdata(od_rdata),
        .bytes_rd(bytes_rd), .bytes_wr(bytes_wr), .n_jobs(n_xfer),
        .busy_cnt(dma_busy), .wait_cnt(dma_wait));

    armse_compute #(.NB(NB), .LOGNB(LOGNB), .AAW(AAW), .OAW(OAW)) u_cmp (
        .clk(clk), .rst(rst), .clr(go), .run(run), .n_cmp(n_cmp), .dma_done(dma_done),
        .cp_done(cp_done), .p_idx(c_idx), .p_data(c_ent),
        .a_r_en(a_r_en), .a_r_addr(a_r_addr), .a_r_par(a_r_par), .a_r_data(a_r_data),
        .b_r_en(b_r_en), .b_r_addr(b_r_addr), .b_r_par(b_r_par), .b_r_data(b_r_data),
        .o_re(o_re), .o_raddr(o_raddr), .o_rdata(o_rdata),
        .o_we(o_we), .o_waddr(o_waddr), .o_wdata(o_wdata),
        .busy_cnt(cmp_busy), .wait_cnt(cmp_wait));

    assign bank_conflicts = conf_a + conf_b;
endmodule
