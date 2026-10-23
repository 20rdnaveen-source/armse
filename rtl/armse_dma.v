// ARMSE transfer engine + DMA-program sequencer.
//
// Entry types: LOAD_A (ext -> A buffer), LOAD_B (ext -> B buffer), WRITEBACK (O buffer -> ext).
// An entry starts when cp_done >= DEP.  Each tile row is one burst: command cycle, ROW_LAT wait
// cycles (provided by the external memory), then ceil(cols/4) data beats (writeback: cols beats).
// Timing (hand-derived; the testbench reports measured values):
//   job duration = 2 + rows * (1 + ROW_LAT + beats_per_row)         (model: job_ovh = 2, row_fix = ROW_LAT+1)
// Load data path: each beat carries 4 consecutive external bytes = 4 consecutive columns of a tile row.
//   A tile : element (i=row, k=col)  -> buffer row i, col k   bank (i+k)%NB, word i*stride + k/NB
//   B tile : element (k=row, j=col)  -> buffer row j, col k   bank (j+k)%NB, word j*stride + k/NB
//   stride = ceil(tk/NB) for both.  (External rows may start at any byte address: idealised bus.)
`timescale 1ns/1ps
`include "armse_fields.vh"
module armse_dma #(
    parameter NB    = 8,
    parameter LOGNB = 3,
    parameter AAW   = 12,
    parameter OAW   = 14
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 clr,
    input  wire                 run,
    input  wire [`CNT_W-1:0]    n_dma,
    input  wire [`CNT_W-1:0]    cp_done,
    output reg  [`CNT_W-1:0]    dma_done,
    output wire [`CNT_W-1:0]    p_idx,
    input  wire [127:0]         p_data,
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
    // A buffer write port
    output wire [3:0]           a_w_en,
    output wire [4*LOGNB-1:0]   a_w_bank,
    output wire [4*AAW-1:0]     a_w_addr,
    output wire [31:0]          a_w_data,
    // B buffer write port
    output wire [3:0]           b_w_en,
    output wire [4*LOGNB-1:0]   b_w_bank,
    output wire [4*AAW-1:0]     b_w_addr,
    output wire [31:0]          b_w_data,
    // output buffer DMA read port (asynchronous)
    output wire [OAW-1:0]       o_raddr,
    input  wire [31:0]          o_rdata,
    // performance counters
    output reg  [31:0]          bytes_rd,
    output reg  [31:0]          bytes_wr,
    output reg  [31:0]          n_jobs,
    output reg  [31:0]          busy_cnt,
    output reg  [31:0]          wait_cnt
);
    localparam [2:0] S_IDLE = 3'd0, S_CMD = 3'd1, S_LD = 3'd2, S_WR = 3'd3, S_DONE = 3'd4;

    reg [2:0] state;
    reg [`CNT_W-1:0] idx;
    assign p_idx = idx;

    wire [1:0]        e_type   = p_data[`DMA_TYPE_HI:`DMA_TYPE_LO];
    wire [`CNT_W-1:0] e_dep    = p_data[`DMA_DEP_HI:`DMA_DEP_LO];
    wire [7:0]        e_rows   = p_data[`DMA_ROWS_HI:`DMA_ROWS_LO];
    wire [7:0]        e_cols   = p_data[`DMA_COLS_HI:`DMA_COLS_LO];
    wire [31:0]       e_eaddr  = p_data[`DMA_EXT_ADDR_HI:`DMA_EXT_ADDR_LO];
    wire [15:0]       e_estr   = p_data[`DMA_EXT_STRIDE_HI:`DMA_EXT_STRIDE_LO];
    wire [15:0]       e_base   = p_data[`DMA_BASE_HI:`DMA_BASE_LO];
    wire [7:0]        e_stride = p_data[`DMA_STRIDE_HI:`DMA_STRIDE_LO];
    wire [8:0]        e_nbeats = (e_type == `T_WB) ? {1'b0, e_cols} : ({1'b0, e_cols} + 9'd3) >> 2;

    wire have   = run && (idx < n_dma);
    wire dep_ok = (cp_done >= e_dep);
    wire start  = (state == S_IDLE) && have && dep_ok;

    reg [1:0]  typ;
    reg [7:0]  rows_r, cols_r, stride_r, row_r, beat_r, nbeats_r;
    reg [31:0] eaddr_row;
    reg [15:0] estr_r, base_r, rb_r, jb_r;

    wire [15:0] st   = {8'd0, stride_r};
    wire [7:0]  col0 = {beat_r[5:0], 2'b00};                  // loads: first column of this beat
    wire [8:0]  rem  = {1'b0, cols_r} - {1'b0, col0};
    wire [3:0]  nvalid = (rem >= 9'd4) ? 4'd4 : rem[3:0];

    assign x_cmd_valid = (state == S_CMD);
    assign x_cmd_write = (typ == `T_WB);
    assign x_cmd_addr  = eaddr_row;
    assign x_cmd_beats = nbeats_r;
    assign x_wvalid    = (state == S_WR);
    assign x_wdata     = o_rdata;
    assign o_raddr     = rb_r[OAW-1:0] + {6'd0, beat_r};

    // per-element bank / address generation for load beats
    integer q;
    integer elem;
    reg [3:0]          wen_c;
    reg [4*LOGNB-1:0]  wbank_c;
    reg [4*AAW-1:0]    waddr_a, waddr_b;
    always @* begin
        wen_c = 4'd0; wbank_c = {4*LOGNB{1'b0}}; waddr_a = {4*AAW{1'b0}}; waddr_b = {4*AAW{1'b0}};
        for (q = 0; q < 4; q = q + 1) begin
            elem = col0 + q;
            wen_c[q] = (state == S_LD) && x_rvalid && (q < nvalid);
            wbank_c[q*LOGNB +: LOGNB] = row_r + elem;
            waddr_a[q*AAW +: AAW]     = rb_r + (elem >> LOGNB);
            waddr_b[q*AAW +: AAW]     = jb_r + q * st + (row_r >> LOGNB);
        end
    end
    assign a_w_en   = (typ == `T_LA) ? wen_c : 4'd0;
    assign b_w_en   = (typ == `T_LB) ? wen_c : 4'd0;
    assign a_w_bank = wbank_c;
    assign b_w_bank = wbank_c;
    assign a_w_addr = waddr_a;
    assign b_w_addr = waddr_b;
    assign a_w_data = x_rdata;
    assign b_w_data = x_rdata;

    wire beat_xfer = ((state == S_LD) && x_rvalid) || ((state == S_WR) && x_wready);
    wire last_beat = (beat_r == nbeats_r - 8'd1);
    wire last_row  = (row_r == rows_r - 8'd1);

    always @(posedge clk) begin
        if (rst || clr) begin
            state <= S_IDLE; idx <= {`CNT_W{1'b0}}; dma_done <= {`CNT_W{1'b0}};
            bytes_rd <= 32'd0; bytes_wr <= 32'd0; n_jobs <= 32'd0;
            busy_cnt <= 32'd0; wait_cnt <= 32'd0;
        end else begin
            if (state != S_IDLE)       busy_cnt <= busy_cnt + 32'd1;
            else if (have && !dep_ok)  wait_cnt <= wait_cnt + 32'd1;

            case (state)
                S_IDLE: if (start) begin
                    typ <= e_type; rows_r <= e_rows; cols_r <= e_cols; stride_r <= e_stride;
                    nbeats_r <= e_nbeats[7:0]; eaddr_row <= e_eaddr; estr_r <= e_estr;
                    base_r <= e_base; rb_r <= e_base; jb_r <= e_base;
                    row_r <= 8'd0; beat_r <= 8'd0;
                    state <= S_CMD;
                end
                S_CMD: if (x_cmd_ready) begin
                    beat_r <= 8'd0;
                    state  <= (typ == `T_WB) ? S_WR : S_LD;
                end
                S_LD, S_WR: if (beat_xfer) begin
                    if (state == S_LD) bytes_rd <= bytes_rd + {28'd0, nvalid};
                    else               bytes_wr <= bytes_wr + 32'd4;
                    if (last_beat) begin
                        if (last_row) begin
                            state <= S_DONE;
                        end else begin
                            row_r     <= row_r + 8'd1;
                            eaddr_row <= eaddr_row + {16'd0, estr_r};
                            rb_r      <= rb_r + st;
                            jb_r      <= base_r;
                            beat_r    <= 8'd0;
                            state     <= S_CMD;
                        end
                    end else begin
                        beat_r <= beat_r + 8'd1;
                        jb_r   <= jb_r + {st[13:0], 2'b00};     // 4 * stride per beat (LOAD_B)
                    end
                end
                S_DONE: begin
                    dma_done <= dma_done + {{(`CNT_W-1){1'b0}}, 1'b1};
                    idx      <= idx + {{(`CNT_W-1){1'b0}}, 1'b1};
                    n_jobs   <= n_jobs + 32'd1;
                    state    <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
