// ARMSE compute engine + compute-program sequencer.
//
// Executes the compute program in order.  An entry starts when its dependency counter is met
// (dma_done >= DEP).  One output element pair (i,j) is reduced over K with NB parallel MAC lanes:
// every issue cycle reads NB k-values of A row i and of B column j and the NB products are summed.
//
// Timing (derived by hand, see docs/ARCHITECTURE.md section 6; the testbench reports the measured value):
//   start seen in IDLE at cycle s -> first issue s+1 -> last issue s+N, N = tm*tn*ceil(tk/NB)
//   -> completion becomes visible in cp_done at cycle s+N+4        (model: cmp_ovh = 4)
`timescale 1ns/1ps
`include "armse_fields.vh"
module armse_compute #(
    parameter NB    = 8,
    parameter LOGNB = 3,
    parameter AAW   = 12,
    parameter OAW   = 14
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 clr,        // 1-cycle pulse with go: clears counters
    input  wire                 run,
    input  wire [`CNT_W-1:0]    n_cmp,
    input  wire [`CNT_W-1:0]    dma_done,
    output reg  [`CNT_W-1:0]    cp_done,
    output wire [`CNT_W-1:0]    p_idx,
    input  wire [127:0]         p_data,
    // A / B buffer read ports
    output wire                 a_r_en,
    output wire [AAW-1:0]       a_r_addr,
    output wire [LOGNB-1:0]     a_r_par,
    input  wire [8*NB-1:0]      a_r_data,
    output wire                 b_r_en,
    output wire [AAW-1:0]       b_r_addr,
    output wire [LOGNB-1:0]     b_r_par,
    input  wire [8*NB-1:0]      b_r_data,
    // output buffer compute port
    output wire                 o_re,
    output wire [OAW-1:0]       o_raddr,
    input  wire [31:0]          o_rdata,
    output wire                 o_we,
    output wire [OAW-1:0]       o_waddr,
    output wire [31:0]          o_wdata,
    // performance counters
    output reg  [31:0]          busy_cnt,
    output reg  [31:0]          wait_cnt
);
    localparam [2:0] S_IDLE = 3'd0, S_RUN = 3'd1, S_D1 = 3'd2, S_D2 = 3'd3, S_DONE = 3'd4;

    reg [2:0]  state;
    reg [`CNT_W-1:0] idx;
    assign p_idx = idx;

    // ---- entry decode ----
    wire [7:0]            e_tm    = p_data[`CMP_TM_HI:`CMP_TM_LO];
    wire [7:0]            e_tn    = p_data[`CMP_TN_HI:`CMP_TN_LO];
    wire [7:0]            e_tk    = p_data[`CMP_TK_HI:`CMP_TK_LO];
    wire                  e_first = p_data[`CMP_FIRST_HI:`CMP_FIRST_LO];
    wire [`CNT_W-1:0]     e_dep   = p_data[`CMP_DEP_HI:`CMP_DEP_LO];
    wire [15:0]           e_abase = p_data[`CMP_ABASE_HI:`CMP_ABASE_LO];
    wire [15:0]           e_bbase = p_data[`CMP_BBASE_HI:`CMP_BBASE_LO];
    wire [15:0]           e_obase = p_data[`CMP_OBASE_HI:`CMP_OBASE_LO];
    wire [15:0]           kg_new  = ({8'd0, e_tk} + NB - 1) >> LOGNB;

    wire have   = run && (idx < n_cmp);
    wire dep_ok = (dma_done >= e_dep);
    wire start  = (state == S_IDLE) && have && dep_ok;

    // ---- latched job ----
    reg [7:0]  tm_r, tn_r, tk_r, kg_r;
    reg        first_r;
    reg [15:0] bbase_r;
    reg [7:0]  i_r, j_r, g_r;
    reg [15:0] a_row, b_col, o_ptr;

    wire issuing = (state == S_RUN);
    assign a_r_en   = issuing;
    assign a_r_addr = a_row + {8'd0, g_r};
    assign a_r_par  = i_r[LOGNB-1:0];
    assign b_r_en   = issuing;
    assign b_r_addr = b_col + {8'd0, g_r};
    assign b_r_par  = j_r[LOGNB-1:0];
    assign o_re     = issuing && (g_r == 8'd0) && !first_r;
    assign o_raddr  = o_ptr[OAW-1:0];

    // ---- pipeline ----
    reg        v1, g0_1, gl_1;
    reg [7:0]  g1;
    reg [15:0] optr1;
    reg        v2, g0_2, gl_2;
    reg [15:0] optr2;
    reg signed [31:0] dot_r, part_r, acc;

    integer l;
    integer kidx;
    reg signed [31:0] dot_c;
    always @* begin
        dot_c = 32'sd0;
        for (l = 0; l < NB; l = l + 1) begin
            kidx = g1 * NB + l;
            if (kidx < tk_r)
                dot_c = dot_c + $signed(a_r_data[l*8 +: 8]) * $signed(b_r_data[l*8 +: 8]);
        end
    end

    wire signed [31:0] acc_next = (g0_2 ? part_r : acc) + dot_r;
    assign o_we    = v2 && gl_2;
    assign o_waddr = optr2[OAW-1:0];
    assign o_wdata = acc_next;

    initial begin
        acc = 32'sd0; dot_r = 32'sd0; part_r = 32'sd0;
    end

    always @(posedge clk) begin
        if (rst || clr) begin
            state <= S_IDLE; idx <= {`CNT_W{1'b0}}; cp_done <= {`CNT_W{1'b0}};
            v1 <= 1'b0; v2 <= 1'b0; busy_cnt <= 32'd0; wait_cnt <= 32'd0;
        end else begin
            // data pipeline
            v1    <= issuing;
            g0_1  <= (g_r == 8'd0);
            gl_1  <= (g_r == kg_r - 8'd1);
            g1    <= g_r;
            optr1 <= o_ptr;
            v2    <= v1;
            g0_2  <= g0_1;
            gl_2  <= gl_1;
            optr2 <= optr1;
            dot_r  <= dot_c;
            part_r <= first_r ? 32'd0 : o_rdata;
            if (v2) acc <= acc_next;

            // counters
            if (state != S_IDLE)              busy_cnt <= busy_cnt + 32'd1;
            else if (have && !dep_ok)         wait_cnt <= wait_cnt + 32'd1;

            // control FSM
            case (state)
                S_IDLE: if (start) begin
                    tm_r <= e_tm; tn_r <= e_tn; tk_r <= e_tk; kg_r <= kg_new[7:0];
                    first_r <= e_first; bbase_r <= e_bbase;
                    i_r <= 8'd0; j_r <= 8'd0; g_r <= 8'd0;
                    a_row <= e_abase; b_col <= e_bbase; o_ptr <= e_obase;
                    state <= S_RUN;
                end
                S_RUN: begin
                    if (g_r == kg_r - 8'd1) begin
                        g_r   <= 8'd0;
                        o_ptr <= o_ptr + 16'd1;
                        if (j_r == tn_r - 8'd1) begin
                            j_r   <= 8'd0;
                            b_col <= bbase_r;
                            if (i_r == tm_r - 8'd1) begin
                                state <= S_D1;
                            end else begin
                                i_r   <= i_r + 8'd1;
                                a_row <= a_row + {8'd0, kg_r};
                            end
                        end else begin
                            j_r   <= j_r + 8'd1;
                            b_col <= b_col + {8'd0, kg_r};
                        end
                    end else begin
                        g_r <= g_r + 8'd1;
                    end
                end
                S_D1:   state <= S_D2;
                S_D2:   state <= S_DONE;
                S_DONE: begin
                    cp_done <= cp_done + {{(`CNT_W-1){1'b0}}, 1'b1};
                    idx     <= idx + {{(`CNT_W-1){1'b0}}, 1'b1};
                    state   <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
