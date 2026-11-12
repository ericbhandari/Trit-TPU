`default_nettype none
`timescale 1ns / 1ps

/* This testbench just instantiates the module and makes some convenient wires
   that can be driven / tested by the cocotb test.py.

   TTPU_N selects the array size of the RTL (default 8); the gate-level netlist is fixed.
*/
`ifndef TTPU_N
`define TTPU_N 8
`endif
`ifndef TTPU_TREG
`define TTPU_TREG 0
`endif
`ifndef TTPU_RPIPE
`define TTPU_RPIPE 0
`endif
`ifndef TTPU_ACCW
`define TTPU_ACCW 16
`endif
`ifndef TTPU_DDR
`define TTPU_DDR 1
`endif

module tb ();

  // Dump the signals to a FST file. You can view it with gtkwave or surfer.
  initial begin
    $dumpfile("tb.fst");
    $dumpvars(0, tb);
    #1;
  end

  // Wire up the inputs and outputs:
  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  reg [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;
`ifdef GL_TEST
  wire VPWR = 1'b1;
  wire VGND = 1'b0;
`endif

  // The top module of the project (info.yaml: top_module):
  tt_um_ericbhandari_trit_tpu
`ifndef GL_TEST
      #(.N(`TTPU_N), .ACCW(`TTPU_ACCW), .TREG(`TTPU_TREG), .RPIPE(`TTPU_RPIPE), .DDR(`TTPU_DDR))
`endif
      user_project (

      // Include power ports for the Gate Level test:
`ifdef GL_TEST
      .VPWR(VPWR),
      .VGND(VGND),
`endif

      .ui_in  (ui_in),    // Dedicated inputs
      .uo_out (uo_out),   // Dedicated outputs
      .uio_in (uio_in),   // IOs: Input path
      .uio_out(uio_out),  // IOs: Output path
      .uio_oe (uio_oe),   // IOs: Enable path (active high: 0=input, 1=output)
      .ena    (ena),      // enable - goes high when design is selected
      .clk    (clk),      // clock
      .rst_n  (rst_n)     // not reset
  );

endmodule
