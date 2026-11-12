/*
 * Copyright (c) 2026 Eric Bhandari
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

/* verilator lint_off DECLFILENAME */

// =================================================================================================
// Trit-TPU (TTPU): ternary tile processing unit (Tiny Tapeout GF180)
//
// An N x N (default 8 x 8) weight-stationary systolic array with ternary weights {-1,0,+1} and
// ternary activation digits, plus per-column carry-save accumulators that
//   * recombine multi-digit ("effort mode") activations:  x = sum_d 3^d * x_d,  d = 0..3 (LSD first)
//   * accumulate across tiles (the K dimension), so the host reads one result per output block.
// "Fast mode" is a one-digit activation; "effort mode" sends up to four digits (81 levels).
// The pins, commands and timing rules are described in docs/info.md.
//
// Ternary trit code (2 bits): bit0 = nonzero, bit1 = negative   00 = 0, 01 = +1, 11 = -1
//
// ---- Pins -------------------------------------------------------------------------------------
//   ui_in[7:0]      DATA byte (command-dependent)
//   uio_in[2:0]     CMD
//   uio_in[7:3]     FIELD: 5 more data bits in W8 and XP* clocks; in every other clock the read-port address
//                   (so a single read needs no command; streams use RDS)
//   uo_out[7:0]     registered read data for the address, 4 clocks after it; it keeps its value for a clock whose field pins
//                   carry data (W8 / XP* outside a read stream), so it only changes for reads that were asked for
//   uio_out/oe      unused (all bidirectional pins are inputs)
//
// ---- 8 trits per clock ------------------------------------------------------------------------
//   W8 and XP* carry eight ternary values in 13 pin bits: DATA = 5 trits packed base-3
//   (b = t0 + 3 t1 + 9 t2 + 27 t3 + 81 t4, t = 0,1,2 for 0,+1,-1; legal 0..242) and FIELD = 3 trits packed base-3
//   (f = t5 + 3 t6 + 9 t7, legal 0..26). An illegal DATA (243..255) or FIELD (27..31) gives zero trits for that
//   part and sets ERR. A weight tile (N = 8) is 8 W8 commands, an activation plane one XP command.
//
// ---- Commands (CMD) ---------------------------------------------------------------------------
//   0 NOP
//   1 W8    eight weight trits for the next group of the write pointer (diagonal PE order, see wpos): group g
//           holds the PEs with stream position 8g..8g+7. A tile is N*N/8 groups (8 for N = 8). The pointer wraps.
//   2 WRST  reset the weight pointer to group 0 and clear ERR; data 0xD3 / 0x5D switches DDR mode on / off
//   3 CFG   DATA = ternarize threshold tau (0..255)
//   4 XP    one digit plane of the activation vector (trit r = activation row r), next digit of the vector
//   5 XPF   like XP, but the first plane of a new vector (restarts the digit counter)
//   6 XPS   like XPF, and the accumulators are cleared before this vector's first digit is added
//   7 RDS   read stream: the address in FIELD is read as usual, and DATA[5:0] more reads follow at the
//           next addresses, one per clock (the address pins are ignored meanwhile, so W8/XP* can go on)
//   A vector is D digit planes (D = 1..4, least-significant digit first).
//   The host picks D per vector by how many planes it sends: there is no mode register.
//   A fifth plane would be scaled by 3^0 again (the plane counter wraps).
//
// ---- Read map (address) -----------------------------------------------------------------------
//   0x00 ID (0x3E)    0x01 {3'd4, N[4:0]}    0x02 ACCW    0x03 STATUS {4'b0, DDR, ERR, 1'b0, READY}
//   0x04 tau          0x05 protocol version (3 with the DDR lane, 2 without)    0x08+g ternarized byte g: columns 4g..4g+3, column 4g+j in [2j+1:2j]
//   0x10+c raw column c, low byte      0x10+N+c raw column c, high byte (two's complement ACCW): the low bytes come first
//   because they settle earlier, so one read stream of 2N bytes can start sooner
//   Ternary results are valid when READY = 1.  Ternary: +1 if acc > tau, -1 if acc < -tau.
//   The addresses 0x06, 0x07 and 0x0A .. 0x0F read 0.
//   Raw bytes of column c are valid from t0 + RAWBASE + c (high byte; the low byte (ACCW-8)/4 clocks earlier), earlier than READY.
//
// ---- Timing (clocks; t0 = clock in which the last XP* of a tile is on the pins) -----------------------
//   * the plane reaches the array at t0+7 (LX = 7: 5-stage unpack, load) and column c adds it into its
//     accumulator at the end of clock t0+LX+N+PEL+RPIPE+1+c   (the +1 is the column output register stage; PEL = 1 for
//     the default single-clock PE, 2 with TREG = 1; RPIPE = 1 with the pipelined table)
//   * a read whose address is on the pins at t0+1+TSETTLE or later returns READY = 1 and settled ternary results;
//     TSETTLE = 3N + ACCW/4 + PEL + RPIPE + 13 (= 42 for N = 8): all carries have rippled out and the ternarizer
//     has scanned every column
//   * the raw bytes of column c are binary from the address time t0 + RAWBASE + c, RAWBASE = LX + N + PEL + RPIPE + ACCW/4 - 1
//     (= 19 for N = 8): 24 clocks before READY for column 0 (24 - c for column c)
//   * reads: the data for an address appears 4 clocks later;  tau/status follow the same pipeline
//   * weights: a W8 command changes the weights of its group 7 clocks after its pins (LW = 7); PE (r,c) reads its
//     weight in clock X+LX+r+c for a plane in clock X. A group may land after the last plane has passed its PEs and
//     must land before the next tile's first plane reaches them
//   * the next block's XPS clears column c at the end of clock t(XPS)+LX+N+PEL+RPIPE+1+c, so the previous block's
//     raw bytes must have been read by then (the tests check this at the bound)
//
// ---- Reset ------------------------------------------------------------------------------------
//   Only control state is reset (pointers, counters, tau, event valid bits, scan ring). The datapath
//   (PE array, weights, accumulators) has no reset: weights must be written before use, and the
//   first vector of an output block must be an XPS. Verified from 40 random power-up states: a reset of
//   3 clocks was always enough. Commands in the clock rst_n rises are accepted.
// =================================================================================================

// ---- Processing element: add, subtract or skip, no multiplier -----------------------------------
// TREG = 0 (default): term and add in one clock, the smaller PE; it closes timing with more margin than the
// registered variant at the 15 ns clock. TREG = 1: the product term is registered
// one clock before it is added, so the add stage is a bare increment/decrement (2 more flops per PE; in the flow this
// variant closed with less margin, so it is not the default).
// Either way the sum is built as a selection among psum + 1, psum - 1 and psum, so the late signal (the term, decoded from
// the weight and the incoming activation) only drives a 3:1 select and not a carry chain.
module tern_pe #(
    parameter PW   = 5,
    parameter TREG = 0
) (
    input  wire                 clk,
    input  wire           [1:0] w,         // stationary ternary weight
    input  wire           [1:0] x_in,      // from the left
    input  wire signed [PW-1:0] psum_in,   // from above
    output reg            [1:0] x_out,     // to the right
    output wire    [PW-1:0]     psum_out   // to below: the registered sum
);
  wire nz  = w[0] & x_in[0];            // product is nonzero only if both are
  wire neg = w[1] ^ x_in[1];            // signs differ: negative

  // The sum is psum + 1, psum - 1 or psum. psum + 1 and psum - 1 depend on the incoming sum only, so they are formed
  // while the term (nz, neg) is being decoded, and the term merely selects.
  wire up_sel, dn_sel;                  // add one, subtract one
  generate
    if (TREG != 0) begin : g_treg
      reg t_nz, t_neg;                  // registered term: 0, +1 or -1 (t_neg implies t_nz)
      always @(posedge clk) begin
        t_nz  <= nz;
        t_neg <= nz & neg;
      end
      assign up_sel = t_nz & ~t_neg;
      assign dn_sel = t_neg;
    end else begin : g_inline
      assign up_sel = nz & ~neg;
      assign dn_sel = nz & neg;
    end
  endgenerate

  wire [PW-1:0] sum_up  = psum_in + {{(PW-1){1'b0}}, 1'b1};
  wire [PW-1:0] sum_dn  = psum_in - {{(PW-1){1'b0}}, 1'b1};
  wire [PW-1:0] sum_nxt = dn_sel ? sum_dn : up_sel ? sum_up : psum_in;
  always @(posedge clk) x_out <= x_in;

  reg [PW-1:0] q;
  always @(posedge clk) q <= sum_nxt;
  assign psum_out = q;
endmodule

// ---- NxN array. y[c] = sum over r of W[r][c] * x[r]; row r of x is delayed r clocks -------------
module systolic #(
    parameter N    = 8,
    parameter PW   = 5,
    parameter TREG = 0
) (
    input  wire                clk,
    input  wire [2*N*N-1:0]    w_flat,   // N*N weights x 2 bits, PE(r,c) at [2*(N*r+c) +: 2]
    input  wire [2*N-1:0]      x_flat,   // N rows x 2 bits (row r already delayed r clocks)
    output wire [PW*N-1:0]     y_flat    // N columns x PW bits (column c arrives c clocks late)
);
  wire        [1:0]    x_w [0:N-1][0:N];
  wire signed [PW-1:0] p_w [0:N-1][0:N-1];   // sum entering row r, column c

  genvar r, c;
  generate
    for (r = 0; r < N; r = r + 1) begin : g_row
      assign x_w[r][0] = x_flat[2*r +: 2];
      for (c = 0; c < N; c = c + 1) begin : g_col
        if (r == 0) begin : g_top
          assign p_w[0][c] = {PW{1'b0}};
        end
        wire [PW-1:0] po;
        tern_pe #(.PW(PW), .TREG(TREG)) u_pe (
            .clk      (clk),
            .w        (w_flat[2*(N*r + c) +: 2]),
            .x_in     (x_w[r][c]),
            .psum_in  (p_w[r][c]),
            .x_out    (x_w[r][c+1]),
            .psum_out (po)
        );
        if (r == N - 1) begin : g_out
          assign y_flat[PW*c +: PW] = po;
        end else begin : g_down
          assign p_w[r+1][c] = po;
        end
      end
    end
  endgenerate
endmodule

// ---- Base-3 unpack: byte b = 81 t4 + 27 t3 + 9 t2 + 3 t1 + t0 gives five trits ---------------------
// plus the 5-bit field f = 9 t7 + 3 t6 + t5 gives three more trits (eight trits per command).
// A feed-forward chain with one digit per register stage (compare with two constants, subtract the
// matching multiple, pass the remainder on); the first digit is split in two stages (compare, then
// subtract) because compare + subtract + select of an 8-bit value is the longest step. A single-stage
// divide-by-9 is 18+ gates deep. The field is delayed as a raw 5-bit number for three stages
// (cheaper than delaying its decoded trits) and decoded in the last two, so both parts leave together.
// Latency: trits_* appear 5 clocks after b and f; bad4 appears 4 clocks after them (one clock earlier).
module tpu_unpack (
    input  wire        clk,
    input  wire [7:0]  b,
    input  wire [4:0]  f,
    output wire        bad4,      // b was illegal (> 242), aligned with the fourth stage
    output wire        badf4,     // f was illegal (> 26), aligned with the fourth stage
    output reg  [15:0] trits_a,   // trit i in [2i+1:2i]; illegal parts give zeros (byte: trits 0-4, field: 5-7)
    output reg  [15:0] trits_b    // identical copy (halves the fanout of the weight-group write)
);
  function [1:0] tc;             // digit 0, 1, 2 gives trit code 00, 01, 11
    input [1:0] dg;
    tc = (dg == 2'd0) ? 2'b00 : (dg == 2'd1) ? 2'b01 : 2'b11;
  endfunction

  // stage 1: compare with 162 and 81, keep the byte
  reg [7:0] b_1;
  reg       g162_1, g81_1, bad_1;
  always @(posedge clk) begin
    b_1    <= b;
    g162_1 <= (b >= 8'd162);
    g81_1  <= (b >= 8'd81);
    bad_1  <= (b >= 8'd243);
  end

  // stage 2: t4 = b / 81, r4 = b - 81 t4
  wire [7:0] r4c = g162_1 ? (b_1 - 8'd162) : g81_1 ? (b_1 - 8'd81) : b_1;
  reg  [1:0] t4_2;
  reg  [6:0] r4_2;
  reg        bad_2;
  always @(posedge clk) begin
    t4_2  <= g162_1 ? 2'd2 : g81_1 ? 2'd1 : 2'd0;
    r4_2  <= r4c[6:0];
    bad_2 <= bad_1;
  end

  // stage 3: t3 = r4 / 27, r3 = r4 - 27 t3
  wire       g54 = (r4_2 >= 7'd54);
  wire       g27 = (r4_2 >= 7'd27);
  wire [6:0] r3c = g54 ? (r4_2 - 7'd54) : g27 ? (r4_2 - 7'd27) : r4_2;
  reg  [1:0] t4_3, t3_3;
  reg  [5:0] r3_3;
  reg        bad_3;
  always @(posedge clk) begin
    t4_3  <= t4_2;
    t3_3  <= g54 ? 2'd2 : g27 ? 2'd1 : 2'd0;
    r3_3  <= r3c[5:0];
    bad_3 <= bad_2;
  end

  // stage 4: t2 = r3 / 9, r2 = r3 - 9 t2
  wire       g18 = (r3_3 >= 6'd18);
  wire       g9  = (r3_3 >= 6'd9);
  wire [5:0] r2c = g18 ? (r3_3 - 6'd18) : g9 ? (r3_3 - 6'd9) : r3_3;
  reg  [1:0] t4_4, t3_4, t2_4;
  reg  [4:0] r2_4;
  reg        bad_4;
  always @(posedge clk) begin
    t4_4  <= t4_3;
    t3_4  <= t3_3;
    t2_4  <= g18 ? 2'd2 : g9 ? 2'd1 : 2'd0;
    r2_4  <= r2c[4:0];
    bad_4 <= bad_3;
  end
  assign bad4 = bad_4;

  // stage 5: t1 = r2 / 3, t0 = r2 - 3 t1 (r2 <= 8 for legal bytes), gate illegal bytes to zero
  wire       g6  = (r2_4 >= 5'd6);
  wire       g3  = (r2_4 >= 5'd3);
  wire [4:0] t0c = g6 ? (r2_4 - 5'd6) : g3 ? (r2_4 - 5'd3) : r2_4;
  wire [1:0] t1c = g6 ? 2'd2 : g3 ? 2'd1 : 2'd0;
  wire [9:0] tr  = bad_4 ? 10'b0 : {tc(t4_4), tc(t3_4), tc(t2_4), tc(t1c), tc(t0c[1:0])};

  // the field: raw for three stages, then t7 = f / 9 and the remainder (stage 4), then t6, t5 (stage 5)
  reg [4:0] f_1, f_2, f_3;
  always @(posedge clk) begin
    f_1 <= f;
    f_2 <= f_1;
    f_3 <= f_2;
  end
  wire       h18 = (f_3 >= 5'd18);
  wire       h9  = (f_3 >= 5'd9);
  wire [4:0] fr  = h18 ? (f_3 - 5'd18) : h9 ? (f_3 - 5'd9) : f_3;
  reg  [1:0] t7_4;
  reg  [3:0] fr_4;
  reg        badf_4;
  always @(posedge clk) begin
    t7_4   <= h18 ? 2'd2 : h9 ? 2'd1 : 2'd0;
    fr_4   <= fr[3:0];
    badf_4 <= (f_3 >= 5'd27);
  end
  assign badf4 = badf_4;
  wire       k6  = (fr_4 >= 4'd6);
  wire       k3  = (fr_4 >= 4'd3);
  wire [3:0] f0c = k6 ? (fr_4 - 4'd6) : k3 ? (fr_4 - 4'd3) : fr_4;
  wire [1:0] f1c = k6 ? 2'd2 : k3 ? 2'd1 : 2'd0;
  wire [5:0] trf = badf_4 ? 6'b0 : {tc(t7_4), tc(f1c), tc(f0c[1:0])};

  (* keep *) always @(posedge clk) trits_a <= {trf, tr};
  (* keep *) always @(posedge clk) trits_b <= {trf, tr};

  wire _unused = &{1'b0, r4c[7], r3c[6], r2c[5], t0c[4:2], fr[4], f0c[3:2]};
endmodule

// ---- One output column: digit scaling and segmented carry-save accumulator -----------------------
// acc = s + sum over segments k >= 1 of cc[k] << (SEG*k)  (mod 2^W). The word is cut into segments of SEG bits; inside a
// segment the new value mq (zero when idle) is added by an ordinary ripple adder, between segments only the carry is
// kept, in a register. Every clock each segment adds the carry that its neighbour produced in the clock before, so the
// carries ripple out by themselves within W/SEG - 1 idle clocks (3 for W = 16, SEG = 4), the loop is SEG full adders
// deep, and the carry word has W/SEG - 1 bits instead of W - 1. An XPS vector clears (acc := mq) instead.
// The scale p * 3^d is formed from small lookup tables of p (one per multiple, built at elaboration
// time, 5-input functions) selected by a registered one-hot digit code. The one-hot register also
// carries the valid bit: an idle clock selects nothing and mq is zero.
// RPIPE = 1 registers the three multiples (3p, 9p, 27p) in a first stage and selects in a second stage:
// every stage is then 3-4 gates deep (28 more flops per column). RPIPE = 0 does table and select in one
// clock (6-7 gates, the longest path of the design).
module tpu_acc #(
    parameter PW = 5,       // signed width of the array column output
    parameter MW = 9,       // signed width of the scaled digit result p * 3^d
    parameter W  = 16,      // accumulator width
    parameter NMAX = 8,     // largest |p| that can occur (the array size N)
    parameter RPIPE = 0,    // 0: table and select in one clock (the default), 1: pipeline the table lookup (see above)
    parameter SEG = 4,      // bits per carry segment (W must be a multiple of it)
    parameter NCOPY = 3     // copies of the column output (2 or 3)
) (
    input  wire          clk,
    input  wire [NCOPY*PW-1:0] pc, // copies of the array column output (registers): [0 +: PW] feeds the p and 3p path,
                                 // [PW +: PW] the 9p table, [(NCOPY-1)*PW +: PW] the 27p table (the same copy as 9p when NCOPY = 2)
    input  wire          v0,     // an event is one clock ahead of the select stage ...
    input  wire [1:0]    d0,     // ... and this is its digit index (scale 3^d0)
    input  wire          clr1,   // the digit result entering mq starts a new accumulation
    output wire [W-1:0]  s_o     // accumulator (binary when settled)
);
  wire [PW-1:0] p   = pc[0 +: PW];
  wire [PW-1:0] p9  = pc[PW +: PW];
  wire [PW-1:0] p27 = pc[(NCOPY-1)*PW +: PW];
  // registered one-hot digit select, zero when no event is arriving
  reg [3:0] dsel;
  always @(posedge clk) dsel <= {v0 & (d0 == 2'd3), v0 & (d0 == 2'd2), v0 & (d0 == 2'd1), v0 & (d0 == 2'd0)};

  localparam NP = 32'd1 << PW;
  /* verilator lint_off UNUSEDSIGNAL */
  function automatic [MW-1:0] mult_entry;     // p * 3^d for the two's complement code i
    input integer i;
    input integer d;
    integer mul, v, pr;
    begin
      mul = (d == 0) ? 1 : (d == 1) ? 3 : (d == 2) ? 9 : 27;
      v   = (i >= (NP >> 1)) ? (i - NP) : i;
      pr  = v * mul;
      mult_entry = pr[MW-1:0];
    end
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */

  // |p| <= N always (a sum of N terms of +-1), so the table entries for larger magnitudes are
  // don't-cares: the synthesis tool is free to choose them to make the logic smaller and faster.
  wire [MW-1:0] mt [0:3][0:NP-1];
  genvar mi, ri;
  generate
    for (mi = 0; mi < 4; mi = mi + 1) begin : g_mul
      for (ri = 0; ri < NP; ri = ri + 1) begin : g_rom
        if (ri <= NMAX || ri >= NP - NMAX) begin : g_used
          assign mt[mi][ri] = mult_entry(ri, mi);
        end else begin : g_dc
          assign mt[mi][ri] = {MW{1'bx}};
        end
      end
    end
  endgenerate

  wire [MW-1:0] rv;
  wire          rsgn;     // sign of rv: every multiple of p has the sign of p, so the sign flops do not wait for the table
  generate
    if (RPIPE != 0) begin : g_rpipe
      // stage 1: the multiples of p, registered (p itself is only delayed)
      reg [MW-1:0] m3q, m9q, m27q;
      reg [PW-1:0] p1;
      always @(posedge clk) begin
        p1   <= p;
        m3q  <= mt[1][p];
        m9q  <= mt[2][p9];
        m27q <= mt[3][p27];
      end
      // stage 2: select (the upper bits of m3q / m9q are sign copies and cost no flops: unused flops vanish)
      assign rv = ({MW{dsel[0]}} & {{(MW-PW){p1[PW-1]}}, p1}) | ({MW{dsel[1]}} & m3q)
                | ({MW{dsel[2]}} & m9q) | ({MW{dsel[3]}} & m27q);
      assign rsgn = (|dsel) & p1[PW-1];
    end else begin : g_rdirect
      assign rv = ({MW{dsel[0]}} & mt[0][p]) | ({MW{dsel[1]}} & mt[1][p])
                | ({MW{dsel[2]}} & mt[2][p9]) | ({MW{dsel[3]}} & mt[3][p27]);
      assign rsgn = (|dsel) & p[PW-1];
    end
  endgenerate

  localparam NSEG = W / SEG;
  reg [MW-1:0]    mq;
  reg [W-1:0]     s;
  reg [NSEG-1:1]  cc;            // cc[k]: carry into segment k, produced by segment k-1 in the clock before
  always @(posedge clk) mq <= rv;

  // sign extension of mq: the sign bit would drive ~24 loads, so it is registered in three copies that each
  // drive a third of the extension bits
  wire [2:0] sgn;
  genvar si;
  generate
    for (si = 0; si < 3; si = si + 1) begin : g_sgn
      reg q;
      (* keep *) always @(posedge clk) q <= rsgn;
      assign sgn[si] = q;
    end
  endgenerate
  wire [W-1:0] mx;
  assign mx[MW-1:0] = mq;
  genvar xi;
  generate
    for (xi = MW; xi < W; xi = xi + 1) begin : g_ext
      assign mx[xi] = sgn[((xi - MW) * 3) / (W - MW)];
    end
  endgenerate
  wire [W-1:0]    sm;            // sum of every segment, with the carry its neighbour registered
  wire [NSEG-1:1] cy;            // carry out of segment k-1, registered into cc[k]
  genvar sk;
  generate
    for (sk = 0; sk < NSEG; sk = sk + 1) begin : g_seg
      wire cin;
      if (sk == 0) begin : g_c0
        assign cin = 1'b0;
      end else begin : g_cn
        assign cin = cc[sk];
      end
      wire [SEG:0] t = {1'b0, s[sk*SEG +: SEG]} + {1'b0, mx[sk*SEG +: SEG]} + {{SEG{1'b0}}, cin};
      assign sm[sk*SEG +: SEG] = t[SEG-1:0];
      if (sk < NSEG - 1) begin : g_co
        assign cy[sk+1] = t[SEG];
      end else begin : g_top
        wire _unused = t[SEG];       // the carry out of the top segment is dropped (the sum wraps modulo 2^W)
      end
    end
  endgenerate

  // the clear is registered locally in four copies (one per quarter of the accumulator): it would
  // otherwise be a 19-load net (16 sum bits and 3 carry bits)
  wire [3:0] clrq;
  genvar ci;
  generate
    for (ci = 0; ci < 4; ci = ci + 1) begin : g_clr
      reg q;
      (* keep *) always @(posedge clk) q <= clr1;
      assign clrq[ci] = q;
    end
  endgenerate

  integer b;
  always @(posedge clk) begin
    for (b = 0; b < W; b = b + 1) s[b]  <= clrq[(b * 4) / W] ? mx[b] : sm[b];
    for (b = 1; b < NSEG; b = b + 1) cc[b] <= clrq[(b * SEG * 4) / W] ? 1'b0 : cy[b];
  end
  assign s_o = s;
endmodule

// ---- Ternarizer: one shared comparator scans the N columns round robin -----------------------------
// tern[c] = +1 if acc[c] > tau, -1 if acc[c] < -tau, else 0 (code 01 / 11 / 00). A rotating one-hot ring
// selects a column, three register stages split the value, compare it with tau and decide, and the result
// is written back to the column's register. Each column is refreshed every N clocks, so after the
// accumulators settle the ternary results are valid within N + 5 clocks. The compare is kept to one
// 8-bit comparator pair, so tau has a fanout of 2 instead of 16+.
//   negative s:  |s| > tau  <=>  ~s >= tau
module tpu_tern #(
    parameter N = 8,
    parameter W = 16
) (
    input  wire           clk,
    input  wire           rqa,         // registered reset, active low (two ports, both driven by rq[4] for now)
    input  wire           rqb,
    input  wire [N*W-1:0] acc_flat,
    input  wire [7:0]     tau,
    output wire [2*N-1:0] tern_flat
);
  genvar i;
  wire [N-1:0] ra, rb;                 // two rotating one-hot rings (low / high accumulator bits)
  generate
    for (i = 0; i < N; i = i + 1) begin : g_ring
      reg qa, qb;
      (* keep *) always @(posedge clk) begin
        if (!rqa) qa <= (i == 0);
        else      qa <= ra[(i + N - 1) % N];
      end
      (* keep *) always @(posedge clk) begin
        if (!rqb) qb <= (i == 0);
        else      qb <= rb[(i + N - 1) % N];
      end
      assign ra[i] = qa;
      assign rb[i] = qb;
    end
  endgenerate

  // stage 1: select the column under the ring
  reg [7:0]   lo_c;
  reg [W-9:0] hi_c;
  integer     j;
  always @(*) begin
    lo_c = 8'h00;
    hi_c = {(W-8){1'b0}};
    for (j = 0; j < N; j = j + 1) begin
      lo_c = lo_c | ({8{ra[j]}}     & acc_flat[W*j +: 8]);
      hi_c = hi_c | ({(W-8){rb[j]}} & acc_flat[W*j + 8 +: W-8]);
    end
  end
  reg [W-1:0] selv;
  always @(posedge clk) selv <= {hi_c, lo_c};

  // stage 2: sign, "any upper bit set" and the low byte, complemented for negative values
  reg       sgn1, hi1;
  reg [7:0] lo1;
  always @(posedge clk) begin
    sgn1 <= selv[W-1];
    hi1  <= |(selv[W-2:8] ^ {(W-9){selv[W-1]}});
    lo1  <= selv[7:0] ^ {8{selv[W-1]}};
  end

  // stage 3: compare the low byte with tau
  reg sgn2, hi2, gt2, ge2;
  always @(posedge clk) begin
    sgn2 <= sgn1;
    hi2  <= hi1;
    gt2  <= (lo1 > tau);
    ge2  <= (lo1 >= tau);
  end

  // stage 4: decide
  reg [1:0] tn;
  always @(posedge clk) begin
    if (!sgn2) tn <= (hi2 | gt2) ? 2'b01 : 2'b00;
    else       tn <= (hi2 | ge2) ? 2'b11 : 2'b00;
  end

  // write back: the column selected 4 clocks ago is the one whose ring bit was 3 positions earlier
  generate
    for (i = 0; i < N; i = i + 1) begin : g_wr
      reg       we;
      reg [1:0] tq;
      always @(posedge clk) we <= ra[(i + 3) % N];
      always @(posedge clk) if (we) tq <= tn;
      assign tern_flat[2*i +: 2] = tq;
    end
  endgenerate
endmodule

// ---- Top module (the ports must stay as-is; the name must start with tt_um_ and match top_module in info.yaml) ----
module tt_um_ericbhandari_trit_tpu #(
    parameter N    = 8,     // array size: 4 or 8
    parameter ACCW = 16,    // accumulator width, 12 or 16 (a multiple of SEG = 4)
    parameter TREG = 0,     // 0: single-clock PE (smaller, default), 1: PE product term registered (2 more flip-flops per PE, closed with less margin in the flow)
    parameter RPIPE = 0,    // 0: digit-scale table and select in one clock (smaller, the default: the DDR lane needs the area), 1: pipelined in two stages (28 more flip-flops per column)
    parameter DDR  = 1      // 1: adds the double-data-rate lane (see "DDR lane" below); 0: protocol 2 exactly
) (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  localparam PW  = $clog2(N + 1) + 1;        // signed width of a column sum (range -N..+N)
  localparam MW  = $clog2(N * 27 + 1) + 1;   // signed width of p * 3^d
  localparam G   = N / 4;                    // ternary result bytes (4 columns each)
  localparam NG  = (N * N) / 8;              // 8-trit weight groups (8 for N = 8, 2 for N = 4)
  localparam PEL = (TREG != 0) ? 2 : 1;      // clocks from x_vec to the column output, per PE hop
  localparam TP  = N - 1 + PEL + 1;          // column c output is on y_raw at x_vec + TP + c (+1: the output register stage)
  localparam NS  = 2 * N + PEL + 1;          // event pipeline depth (stages 0 .. NS-1)
  localparam NR  = (DDR != 0) ? 7 : 6;        // copies of the registered reset (the seventh serves the DDR lane)
  localparam SEG  = 4;                       // bits per carry segment of the accumulators
  localparam YCP  = 3;                       // copies of every column output register (2 or 3)
  localparam NSEG = ACCW / SEG;
  localparam [31:0] TS32 = 3 * N + NSEG + PEL + 13 + RPIPE;   // clocks from the last plane to READY: the address bound is t0 + 1 + TS32 (the tests check that READY rises exactly there, and that the data are settled earlier)
  localparam        SW = $clog2(TS32 + 1);   // bits needed to count TS32 clocks (6 for N = 8)
  localparam        SL = SW + 1;             // width of the settle LFSR: its period 2^SL - 1 is longer than TS32
  function [SL-1:0] lfsr_step;               // x^SL + x^(SL-1) + 1 (primitive for SL = 6 and 7)
    input [SL-1:0] st;
    lfsr_step = {st[SL-2:0], st[SL-1] ^ st[SL-2]};
  endfunction
  function [SL-1:0] lfsr_after;              // the state n steps after st
    input [SL-1:0] st;
    input integer n;
    integer i;
    begin
      lfsr_after = st;
      for (i = 0; i < n; i = i + 1) lfsr_after = lfsr_step(lfsr_after);
    end
  endfunction
  localparam [SL-1:0] LF0 = {{(SL-1){1'b0}}, 1'b1};      // loaded by a plane command
  localparam [SL-1:0] LFT = lfsr_after(LF0, TS32);       // reached TS32 clocks later (= TSETTLE): READY
  localparam [31:0] NB32 = N, AW32 = ACCW;
  localparam [4:0]  NB = NB32[4:0];          // N and ACCW as constants for the info bytes
  localparam [7:0]  AW = AW32[7:0];

  // Position of PE (r, c) in the weight stream: PEs are written in diagonal order (r + c, then r), 8 per W8
  // command. The wave of one tile leaves the array diagonal by diagonal, so the next tile's weights can follow
  // right behind it.
  function integer wpos;
    input integer r;
    input integer c;
    integer dg, k, cnt, rlo;
    begin
      dg  = r + c;
      cnt = 0;
      for (k = 0; k < dg; k = k + 1) cnt = cnt + ((k < N) ? (k + 1) : (2 * N - 1 - k));
      rlo = (dg > N - 1) ? (dg - (N - 1)) : 0;
      wpos = cnt + (r - rlo);
    end
  endfunction

  genvar gi, gj;

  // ---- Registered reset, replicated so no copy drives more than ~20 loads -------------------------
  // rst_n must be synchronous to clk (it is on the Tiny Tapeout board: both come from the same microcontroller);
  // a release that violates setup/hold could leave the copies one clock apart. Seven copies (six without the DDR lane) keep the load on the rst_n pin
  // (and its hold-time fix) small; each drives at most 20 loads. Consumers (active low):
  //   rq[0]   weight-pointer ring                                   rq[3]  event valid bits
  //   rq[1]   tau and its read copy, CFG decode                      rq[4]  ternarizer scan rings
  //   rq[2]   XP* / RDS decode, digit counter, ERR, read stream      rq[5]  W8 / WRST decode, settle counter, last stage of the
  //                                                                         weight flags, next-to-last of the plane flags
  //   rq[6]   (DDR = 1 only) DDR mode flag, lane 0 decode and flag pipeline
  wire [NR-1:0] rq;
  generate
    for (gi = 0; gi < NR; gi = gi + 1) begin : g_rq
      reg q;
      (* keep *) always @(posedge clk) q <= rst_n;
      assign rq[gi] = q;
    end
  endgenerate

  // ---- Stage S1: command decode straight from the pins ---------------------------------------------
  // The raw decode is gated one stage later by the registered reset (rq): a command is accepted
  // exactly when rst_n was high in the clock it was on the pins, and the rst_n pin only drives the
  // reset flops (the pin has a 3 ns input-delay budget).
  wire [2:0] cmd_p = uio_in[2:0];
  reg        wgt1r, wrst1r, cfg1r, xc1r, xf1r, xs1r, rds1r;
  reg  [7:0] d1w, d1x;          // two copies of the data byte (weight / plane path, config / stream count)
  reg  [4:0] d1f;               // the field (three more trits)
  always @(posedge clk) begin
    wgt1r  <= (cmd_p == 3'd1);
    wrst1r <= (cmd_p == 3'd2);
    cfg1r  <= (cmd_p == 3'd3);
    xc1r   <= (cmd_p == 3'd4);
    xf1r   <= (cmd_p == 3'd5);
    xs1r   <= (cmd_p == 3'd6);
    rds1r  <= (cmd_p == 3'd7);
    d1f    <= uio_in[7:3];
  end
  wire wgt1  = wgt1r  & rq[5];
  wire wrst1 = wrst1r & rq[5];
  wire cfg1  = cfg1r  & rq[1];
  wire xc1   = xc1r   & rq[2];
  wire xf1   = xf1r   & rq[2];
  wire xs1   = xs1r   & rq[2];
  wire rds1  = rds1r  & rq[2];
  (* keep *) always @(posedge clk) d1w <= ui_in;
  (* keep *) always @(posedge clk) d1x <= ui_in;

  // ---- DDR lane (DDR = 1) ----------------------------------------------------------------------------------
  // In DDR mode every clock carries a pair of command words: lane 0 is sampled on the falling edge before the rising edge
  // that samples lane 1 (the ordinary word, ui_in / uio_in as in SDR mode); lane 0 reaches stage S1 one clock later (see
  // below), so the pair that reaches S1 together is lane 1 of cycle t and lane 0 of the falling edge before cycle t - 1. Lane 0 may
  // carry W8, XP, XPF, XPS (or anything else, which is ignored); its W8 commands write the first half of the weight groups, the W8
  // commands of lane 1 the second half, each lane with its own pointer ring, and a plane may come on either lane (at most
  // one per pair, else ERR). DDR mode is switched on by a WRST command whose data byte is 0xD3 and off by one whose
  // data byte is 0x5D (any other WRST leaves the mode alone); it takes effect for the pairs from the second clock after
  // that command. In SDR mode lane 0 is ignored; its unpack chain mirrors lane 1 so that the first half of the groups is written
  // too, the pointer passing from the first ring to the second.
  wire        ddr_m;                         // DDR mode is on (constant 0 when DDR = 0)
  wire        wgt0, xc0, xf0, xs0;           // lane 0 commands at stage S1 (0 outside DDR mode)
  wire        zw0, zw1;                      // lane 0 W8 flag at the last stage (ring and write-enable copies)
  wire        bad40, badf40;                 // lane 0 unpack chain: illegal byte / field
  wire [15:0] tq0;                           // lane 0 trits
  generate
    if (DDR != 0) begin : g_l0
      // Falling-edge capture of lane 0 with nothing but wires around it: the pins reach these flops through the input buffers
      // (which carry the flow's hold-fix delay) with about half a period to spare, and the path from them to the rising-edge
      // copy has half a period too. The command is decoded only after the copy, so a lane-0 word reaches stage S1 one clock
      // later than the lane-1 word of the pair it was sampled with: the lane-0 word sampled on the falling edge before rising
      // edge t acts like a lane-1 word of cycle t + 1 (its "effective" cycle; ttpu_model.Prog drives it one slot earlier).
      reg [7:0] un;
      reg [4:0] fn;
      reg [2:0] cn;
      always @(negedge clk) begin
        un <= ui_in;
        fn <= uio_in[7:3];
        cn <= uio_in[2:0];
      end
      reg [7:0] rw;
      reg [4:0] rf;
      reg [2:0] rc;
      reg       dm;
      reg [7:0] d0wq;
      reg [4:0] d0fq;
      reg       wr0, xcr0, xfr0, xsr0;
      always @(posedge clk) begin
        rw <= un;
        rf <= fn;
        rc <= cn;
        if (!rq[6])                         dm <= 1'b0;
        else if (wrst1 && d1x == 8'hD3)     dm <= 1'b1;
        else if (wrst1 && d1x == 8'h5D)     dm <= 1'b0;
        d0wq <= dm ? rw : ui_in;
        d0fq <= dm ? rf : uio_in[7:3];
        wr0  <= dm & (rc == 3'd1);
        xcr0 <= dm & (rc == 3'd4);
        xfr0 <= dm & (rc == 3'd5);
        xsr0 <= dm & (rc == 3'd6);
      end
      assign ddr_m = dm;
      assign wgt0  = wr0  & rq[6];
      assign xc0   = xcr0 & rq[6];
      assign xf0   = xfr0 & rq[6];
      assign xs0   = xsr0 & rq[6];
      reg zwA, zwB, zwC, zwC0, zwC1;
      always @(posedge clk) begin
        zwA <= wgt0;  zwB <= zwA;  zwC <= zwB;
      end
      (* keep *) always @(posedge clk) zwC0 <= zwC & rq[6];
      (* keep *) always @(posedge clk) zwC1 <= zwC & rq[6];
      assign zw0 = zwC0;
      assign zw1 = zwC1;
      wire [15:0] tq0b_unused;
      tpu_unpack u_unpack0 (.clk(clk), .b(d0wq), .f(d0fq), .bad4(bad40), .badf4(badf40), .trits_a(tq0), .trits_b(tq0b_unused));
    end else begin : g_nol0
      assign ddr_m = 1'b0;
      assign wgt0  = 1'b0;
      assign xc0   = 1'b0;
      assign xf0   = 1'b0;
      assign xs0   = 1'b0;
      assign zw0   = 1'b0;
      assign zw1   = 1'b0;
      assign bad40 = 1'b0;
      assign badf40 = 1'b0;
      assign tq0   = 16'h0000;
    end
  endgenerate

  // ---- Weight path: unpack chain, write-pointer ring, group write enables --------------------------
  // d1w, d1f go through 4 pipeline stages into the trits registers (tqa/tqb). The command flags travel alongside.
  reg wgtA, wgtB, wgtC, wrstA, wrstB, wrstC, wrstD;
  reg wgtC0, wgtC1;                         // two copies of the last stage: ring, wen
  always @(posedge clk) begin
    wgtA  <= wgt1;  wgtB  <= wgtA;  wgtC  <= wgtB;
    wrstA <= wrst1; wrstB <= wrstA; wrstC <= wrstB; wrstD <= wrstC;
  end
  // the last stage is gated by the registered reset: whatever a scrambled power-up left in the earlier
  // stages of this pipeline cannot reach the pointer or the write enables once a short reset has ended
  (* keep *) always @(posedge clk) wgtC0 <= wgtC & rq[5];
  (* keep *) always @(posedge clk) wgtC1 <= wgtC & rq[5];

  wire        bad4, badf4;
  wire [15:0] tqa, tqb;
  tpu_unpack u_unpack (.clk(clk), .b(d1w), .f(d1f), .bad4(bad4), .badf4(badf4), .trits_a(tqa), .trits_b(tqb));

  wire [NG-1:0] ring;                       // one-hot: next weight group to write (DDR = 1: two one-hot rings, one per half of the groups)
  reg  [NG-1:0] wen;
  generate
    if (DDR == 0) begin : g_ptr_sdr
      for (gi = 0; gi < NG; gi = gi + 1) begin : g_ring
        reg q;
        always @(posedge clk) begin
          if (!rq[0] || wrstD) q <= (gi == 0);
          else if (wgtC0)                  q <= ring[(gi + NG - 1) % NG];
        end
        assign ring[gi] = q;
      end

      for (gi = 0; gi < NG; gi = gi + 1) begin : g_wen
        always @(posedge clk) wen[gi] <= wgtC1 & ring[gi];
      end
    end else begin : g_ptr_ddr
      // The weight groups 0 .. NG/2 - 1 are written through lane 0, the groups NG/2 .. NG - 1 through lane 1 (each lane has its
      // own pointer ring, so the two sequences are independent; the halves rather than the even / odd groups keep the
      // diagonals that one lane has to visit close together, which is what lets consecutive tiles overlap).
      // In SDR mode the W8 commands (lane 1) fill the first ring and then the second one, which is the order of protocol 2.
      localparam NH = NG / 2;
      reg hf;                                    // SDR mode: the second half is being written
      wire vAr = ddr_m ? zw0   : (wgtC0 & ~hf);
      wire vBr = ddr_m ? wgtC0 : (wgtC0 & hf);
      wire vAw = ddr_m ? zw1   : (wgtC1 & ~hf);
      wire vBw = ddr_m ? wgtC1 : (wgtC1 & hf);
      always @(posedge clk) begin
        if (!rq[0] || wrstD)                                        hf <= 1'b0;
        else if (!ddr_m & wgtC0 & (hf ? ring[NG-1] : ring[NH-1]))   hf <= ~hf;
      end
      for (gi = 0; gi < NH; gi = gi + 1) begin : g_ra
        reg q;
        always @(posedge clk) begin
          if (!rq[0] || wrstD) q <= (gi == 0);
          else if (vAr)        q <= ring[(gi + NH - 1) % NH];
        end
        assign ring[gi] = q;
        always @(posedge clk) wen[gi] <= vAw & ring[gi];
      end
      for (gi = NH; gi < NG; gi = gi + 1) begin : g_rb
        reg q;
        always @(posedge clk) begin
          if (!rq[0] || wrstD) q <= (gi == NH);
          else if (vBr)        q <= ring[NH + ((gi - NH + (NG - NH) - 1) % (NG - NH))];
        end
        assign ring[gi] = q;
        always @(posedge clk) wen[gi] <= vBw & ring[gi];
      end
    end
  endgenerate

  // ---- Activation planes: the same unpack chain; the flags travel as a 2-bit code ---------------------
  //   pk: 0 none, 1 XP, 2 XPF (restart digit counter), 3 XPS (restart and clear)
  wire [1:0] pk1 = xs1 ? 2'd3 : xf1 ? 2'd2 : xc1 ? 2'd1 : 2'd0;
  wire [1:0] pk0 = xs0 ? 2'd3 : xf0 ? 2'd2 : xc0 ? 2'd1 : 2'd0;     // lane 0 (0 outside DDR mode)
  wire [1:0] pkm = (pk1 != 2'd0) ? pk1 : pk0;                      // a plane on both lanes: lane 1 counts, ERR is set
  wire       pcf = (pk1 != 2'd0) && (pk0 != 2'd0);
  wire       sel0 = (pk1 == 2'd0) && (pk0 != 2'd0);                // the plane came on lane 0
  reg  [1:0] pkA, pkB, pkC, pkD, pkE;
  reg        plA, plB, plC, plD, plE;                              // the same, for the lane select
  always @(posedge clk) begin
    plA <= sel0;  plB <= plA;  plC <= plB;  plD <= plC;  plE <= plD;
    pkA <= pkm;
    pkB <= pkA;
    pkC <= pkB;
    pkD <= pkC & {2{rq[5]}};               // gated like the weight flags: junk of a scrambled power-up stops here
    pkE <= pkD;
  end
  reg ldA, ldB;                             // load enables of the two halves of the plane register
  (* keep *) always @(posedge clk) ldA <= |pkD;
  (* keep *) always @(posedge clk) ldB <= |pkD;

  reg err;                                  // an illegal packed value (byte > 242, field > 26) was written
  wire badp = bad4 | ((N > 4) ? badf4 : 1'b0);   // (for N = 4 the field of a plane is not used)
  wire badq = plD ? (bad40 | ((N > 4) ? badf40 : 1'b0)) : badp;   // the same for a plane that came on lane 0
  always @(posedge clk) begin
    if (!rq[2] || wrstD)                     err <= 1'b0;
    else if (wgtC0 & (bad4 | badf4))         err <= 1'b1;
    else if (zw0 & (bad40 | badf40))         err <= 1'b1;
    else if (pcf)                            err <= 1'b1;
    else if ((|pkD) & badq)                  err <= 1'b1;
  end

  // ---- Weight registers: group g holds the PEs at stream positions 8g..8g+7 ---------------------------
  wire [2*N*N-1:0] w_flat;
  wire [15:0]      wq [0:NG-1];
  generate
    for (gi = 0; gi < NG; gi = gi + 1) begin : g_wg
      reg [15:0] q;
      wire [15:0] tq = (DDR != 0 && gi < NG / 2) ? tq0 : (gi < (NG + 1) / 2) ? tqa : tqb;
      always @(posedge clk) if (wen[gi]) q <= tq;
      assign wq[gi] = q;
    end
    for (gi = 0; gi < N; gi = gi + 1) begin : g_wr
      for (gj = 0; gj < N; gj = gj + 1) begin : g_wc
        localparam WP = wpos(gi, gj);
        assign w_flat[2*(N*gi + gj) +: 2] = wq[WP / 8][2*(WP % 8) +: 2];
      end
    end
  endgenerate

  // ---- Config: ternarize threshold (tau_rd is a separate copy for the read port) ---------------------
  reg [7:0] tau, tau_rd;
  always @(posedge clk) begin
    if (!rq[1])    tau <= 8'd0;
    else if (cfg1) tau <= d1x;
  end
  (* keep *) always @(posedge clk) begin
    if (!rq[1])    tau_rd <= 8'd0;
    else if (cfg1) tau_rd <= d1x;
  end

  // ---- Activation sequencer: digit counter (runs when the plane has been unpacked) -------------------
  wire          xany  = xc1 | xf1 | xs1 | xc0 | xf0 | xs0;      // a plane command is on stage S1 (settle counter)
  wire          anyE  = |pkE;
  wire          start = pkE[1];               // XPF or XPS
  reg  [1:0]    dcnt;                         // next expected digit index
  wire [1:0]    di_n  = start ? 2'd0 : dcnt;
  always @(posedge clk) begin
    if (!rq[2])      dcnt <= 2'd0;
    else if (anyE)   dcnt <= di_n + 2'd1;
  end

  // ---- The plane register x_vec: the N trits of one digit plane ------------------------------------
  // (loaded from the unpack chain 6 clocks after the pins: x_vec is visible 7 clocks after them, LX = 7)
  reg [2*N-1:0] xh;
  always @(posedge clk) begin
    if (ldA) xh[N-1:0]   <= plE ? tq0[N-1:0] : tqa[N-1:0];
    if (ldB) xh[2*N-1:N] <= plE ? tq0[2*N-1:N] : tqb[2*N-1:N];
  end
  wire [2*N-1:0] x_vec = xh;

  // ---- Event pipeline: follows the digit through the array to the accumulators ---------------------
  // Stage k is visible k clocks after x_vec changes. Column c's output is on y_raw at stage TP+c; the
  // one-hot digit select is registered from stage TP+c-1, and the clear flag is registered from stage TP+c,
  // so both line up with the scaled value entering mq.
  reg [NS-1:0]   ev_v, ev_c;
  reg [2*NS-1:0] ev_dq;                  // digit index of every stage, stage k at [2*k +: 2]
  integer        k;
  always @(posedge clk) begin
    for (k = 0; k < NS; k = k + 1) begin
      if (!rq[3]) ev_v[k] <= 1'b0;
      else if (k == 0)           ev_v[k] <= anyE;
      else                       ev_v[k] <= ev_v[k-1];
      if (k == 0) begin
        ev_c[k] <= (pkE == 2'd3);
        ev_dq[2*k +: 2] <= di_n;
      end else begin
        ev_c[k] <= ev_c[k-1];
        ev_dq[2*k +: 2] <= ev_dq[2*(k-1) +: 2];
      end
    end
  end

  // ---- Input skew: row r is delayed r clocks so data meets the partial sums --------------------------
  wire [2*N-1:0] x_skew;
  generate
    for (gi = 0; gi < N; gi = gi + 1) begin : g_skew
      if (gi == 0) begin : g_none
        assign x_skew[1:0] = x_vec[1:0];
      end else begin : g_dly
        reg [2*gi-1:0] sh;                // gi stages of two bits, the newest in the low bits
        if (gi == 1) begin : g_one
          always @(posedge clk) sh <= x_vec[2*gi +: 2];
        end else begin : g_more
          always @(posedge clk) sh <= {sh[2*gi-3:0], x_vec[2*gi +: 2]};
        end
        assign x_skew[2*gi +: 2] = sh[2*gi-1 -: 2];
      end
    end
  endgenerate

  // ---- Array --------------------------------------------------------------------------------------
  wire [PW*N-1:0] y_one;
  systolic #(.N(N), .PW(PW), .TREG(TREG)) u_array (
      .clk    (clk),
      .w_flat (w_flat),
      .x_flat (x_skew),
      .y_flat (y_one)
  );

  // ---- Column output registers: YCP copies of every column sum, one per group of consumers (the 1p/3p table, the 9p and 27p
  // tables of the accumulator). The last PE row drives one register only; the copies, which the placer puts next to their
  // consumers and which are far apart, hang off that register, not off the PE adder.
  wire [YCP*PW*N-1:0] y_raw;
  generate
    for (gi = 0; gi < N; gi = gi + 1) begin : g_ycol
      for (gj = 0; gj < YCP; gj = gj + 1) begin : g_ycp
        reg [PW-1:0] q;
        (* keep *) always @(posedge clk) q <= y_one[PW*gi +: PW];
        assign y_raw[YCP*PW*gi + PW*gj +: PW] = q;
      end
    end
  endgenerate

  // ---- Per-column accumulators and the shared ternarizer ---------------------------------------------
  wire [N*ACCW-1:0] acc_flat;
  generate
    for (gi = 0; gi < N; gi = gi + 1) begin : g_col
      tpu_acc #(.PW(PW), .MW(MW), .W(ACCW), .NMAX(N), .RPIPE(RPIPE), .SEG(SEG), .NCOPY(YCP)) u_acc (
          .clk  (clk),
          .pc   (y_raw[YCP*PW*gi +: YCP*PW]),
          .v0   (ev_v[TP-1+RPIPE+gi]),
          .d0   (ev_dq[2*(TP-1+RPIPE+gi) +: 2]),
          .clr1 (ev_v[TP+RPIPE+gi] & ev_c[TP+RPIPE+gi]),
          .s_o  (acc_flat[ACCW*gi +: ACCW])
      );
    end
  endgenerate

  wire [2*N-1:0] tern_flat;
  tpu_tern #(.N(N), .W(ACCW)) u_tern (
      .clk(clk), .rqa(rq[4]), .rqb(rq[4]), .acc_flat(acc_flat), .tau(tau), .tern_flat(tern_flat)
  );

  // ---- Settle tracking: READY = no plane accepted in the last TSETTLE clocks ------------------------
  // The clocks since the last plane are counted by an LFSR, not by a down counter: its next state is a shift and one XOR instead of a
  // borrow chain. A plane command loads LF0, TS32 steps later the state is LFT and READY rises; the reset sets LFT (idle).
  reg [SL-1:0] lf;
  reg          ready_q;
  wire         lf_busy = (lf != LFT);
  always @(posedge clk) begin
    if (!rq[5])        lf <= LFT;
    else if (xany)     lf <= LF0;
    else if (lf_busy)  lf <= lfsr_step(lf);
    ready_q <= !lf_busy;
  end

`ifdef TTPU_ASSERT
  // ---- Simulation-only assertions (compile with -DTTPU_ASSERT; the test harness fails a test if assert_fail is set) ----
  // They state design invariants independently of any test program:
  //   1. the weight-write pointer is one-hot while out of reset,
  //   2. the settle LFSR agrees with a plain down counter (busy exactly while the counter is not zero),
  //   3. once a vector has been accepted, READY implies that every carry-save accumulator is a plain binary number
  //      (carry word zero): the guarantee behind TSETTLE, checked for every column and every data pattern.
  reg assert_fail = 1'b0;
  reg chk_armed   = 1'b0;                   // the settle counter has run since reset (a vector has been accepted)
  reg chk_en      = 1'b1;                   // the harness switches the checks off while it forces unreachable states
  integer ci_chk, ones_chk;
  integer sh_settle = 0;                    // the counter that the LFSR replaces
  always @(posedge clk) begin
    if (!rq[5])                 sh_settle <= 0;
    else if (xany)              sh_settle <= TS32;
    else if (sh_settle != 0)    sh_settle <= sh_settle - 1;
    if (!rq[5]) chk_armed <= 1'b0;
    else if (lf_busy) chk_armed <= 1'b1;                  // a vector was accepted and the settle logic is running
    if (chk_en && rq[0] && rq[5]) begin
      ones_chk = 0;
      for (ci_chk = 0; ci_chk < NG; ci_chk = ci_chk + 1) ones_chk = ones_chk + (ring[ci_chk] === 1'b1);
      if (ones_chk != ((DDR != 0) ? 2 : 1)) begin
        $display("TTPU ASSERTION: weight pointer not one-hot (%0d bits set) at %0t", ones_chk, $time);
        assert_fail <= 1'b1;
      end
    end
    if (chk_en && rq[5] && (lf_busy ? 1'b1 : 1'b0) !== ((sh_settle != 0) ? 1'b1 : 1'b0)) begin
      $display("TTPU ASSERTION: settle LFSR state %0d disagrees with a counter at %0d at %0t", lf, sh_settle, $time);
      assert_fail <= 1'b1;
    end
  end
  generate
    for (gi = 0; gi < N; gi = gi + 1) begin : g_chk
      always @(posedge clk) begin
        // (a carry word with unknown bits is data computed from weights that were never written: not checked)
        if (chk_en && chk_armed && ready_q && rq[5] && (^g_col[gi].u_acc.cc !== 1'bx) && (g_col[gi].u_acc.cc != {(NSEG-1){1'b0}})) begin
          $display("TTPU ASSERTION: carry word of column %0d is not zero while READY (%h) at %0t", gi, g_col[gi].u_acc.cc, $time);
          assert_fail <= 1'b1;
        end
      end
    end
  endgenerate
`endif

  // ---- Read address: the pins (FIELD) in every clock, or the running address of a read stream -------------
  // RDS loads the number of further reads (DATA[5:0]); while it counts down the address increments by one per clock
  // and the pins are ignored, so W8 and XP* commands (whose FIELD carries data, not an address) can be streamed meanwhile.
  reg [4:0] a1;
  reg [5:0] rdc;                            // stream length n, counted down while the stream runs (no decrementer on the load path)
  reg       rd_act;                         // rdc != 0 (registered, so the address mux does not wait for a wide compare)
  reg       cnt_nz1, cnt_gt1;               // stage S1 flags of the stream length on the pins: n != 0, n > 1
  always @(posedge clk) begin
    cnt_nz1 <= |ui_in[5:0];
    cnt_gt1 <= |ui_in[5:1];
  end
  wire      rd_inc = rds1 ? cnt_nz1 : rd_act;
  // rdc has no reset: it is only looked at while rd_act is set, and rd_act is cleared by the reset and every stream loads rdc first
  // (keeping the reset out of the decrement path took it off the critical path)
  always @(posedge clk) begin
    if (rds1)        rdc <= d1x[5:0];                  // n (only used when cnt_gt1 is set, i.e. when the stream has more than one extra read)
    else if (rd_act) rdc <= rdc - 6'd1;
  end
  always @(posedge clk) begin
    if (!rq[2])      rd_act <= 1'b0;
    else if (rds1)   rd_act <= cnt_gt1;
    else if (rd_act) rd_act <= (rdc >= 6'd3);          // rdc - 1 >= 2: at least one more increment after the next one
    a1 <= rd_inc ? a1 + 5'd1 : uio_in[7:3];
  end

  // ---- Read port: one-hot select registers, two AND-OR stages, output register ----------------------
  reg [2*N-1:0] rs_raw;
  reg [G-1:0]   rs_tern;
  reg           rs_id, rs_i0, rs_i1, rs_i2, rs_st, rs_tau;
  integer       qi;
  /* verilator lint_off UNUSEDSIGNAL */
  function [4:0] raw_addr;                  // address of the raw byte with index qi: low bytes at 0x10 + c, high bytes at 0x10 + N + c
    input integer qi_f;
    integer a_f;
    begin
      a_f = 16 + (qi_f % 2) * N + qi_f / 2;
      raw_addr = a_f[4:0];
    end
  endfunction
  /* verilator lint_on UNUSEDSIGNAL */
  always @(posedge clk) begin
    for (qi = 0; qi < 2 * N; qi = qi + 1) rs_raw[qi]  <= (a1 == raw_addr(qi));      // bit 2c: low byte of column c, 2c+1: high byte
    for (qi = 0; qi < G; qi = qi + 1)     rs_tern[qi] <= (a1 == (5'h08 + qi[4:0]));
    rs_id  <= (a1 == 5'h00);
    rs_i0  <= (a1 == 5'h01);
    rs_i1  <= (a1 == 5'h02);
    rs_st  <= (a1 == 5'h03);
    rs_tau <= (a1 == 5'h04);
    rs_i2  <= (a1 == 5'h05);
  end

  reg [7:0]  po_a, po_b, po_c;              // partial ORs (combinational): raw columns 0..N/2-1, N/2..N-1, the rest
  reg [15:0] ax;
  integer    m;
  always @(*) begin
    po_a = 8'h00;
    po_b = 8'h00;
    ax = 16'h0000;
    for (m = 0; m < N; m = m + 1) begin
      ax = acc_flat[ACCW*m +: ACCW];          // zero-extended to 16 bits
      if (m < N / 2) po_a = po_a | ({8{rs_raw[2*m]}} & ax[7:0]) | ({8{rs_raw[2*m+1]}} & ax[15:8]);
      else           po_b = po_b | ({8{rs_raw[2*m]}} & ax[7:0]) | ({8{rs_raw[2*m+1]}} & ax[15:8]);
    end
    po_c = 8'h00;
    for (m = 0; m < G; m = m + 1)
      po_c = po_c | ({8{rs_tern[m]}} & tern_flat[8*m +: 8]);
    po_c = po_c | ({8{rs_id}}  & 8'h3E)
            | ({8{rs_i0}}  & {3'd4, NB})
            | ({8{rs_i1}}  & AW)
            | ({8{rs_i2}}  & ((DDR != 0) ? 8'h03 : 8'h02))
            | ({8{rs_st}}  & {4'b0, ddr_m, err, 1'b0, ready_q})
            | ({8{rs_tau}} & tau_rd);
  end
  reg [7:0] rd_a, rd_b, rd_c;
  always @(posedge clk) begin
    rd_a <= po_a;
    rd_b <= po_b;
    rd_c <= po_c;
  end

  // ---- Output hold: uo_out only follows reads that were asked for -----------------------------------------
  // The address register takes the field pins in every clock outside a read stream, and in a W8 / XP* clock those pins carry data,
  // not an address: the read port would put out whatever byte the data happens to address, a different one nearly every clock
  // (in a matrix-vector product, most of the changes of uo_out). Such a clock is flagged, the flag
  // travels down the read pipeline next to the address, and the output register keeps its byte instead of taking the one of that
  // clock. A clock inside a read stream is never flagged (rd_inc: the address is the running one, whatever the pins carry), and
  // a read has no command, so every clock with a real address is still passed on: idle polling of the status is unchanged.
  // inc1 / ok2 / ok3 are the stages of a1 / rs_* / rd_*. They need no reset: they are rebuilt from the pins within three clocks.
  reg inc1, ok2, ok3;
  always @(posedge clk) begin
    inc1 <= rd_inc;
    ok2  <= inc1 | ~(wgt1r | xc1r | xf1r | xs1r);
    ok3  <= ok2;
  end

  reg [7:0] uo_q;
  always @(posedge clk) if (ok3) uo_q <= rd_a | rd_b | rd_c;
  assign uo_out  = uo_q;
  assign uio_out = 8'b0;
  assign uio_oe  = 8'b0;    // all bidirectional pins are inputs

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, 1'b0};
  wire _unused_l0 = &{1'b0, wgt0, zw1};      // (only the DDR lane uses them)

endmodule
