# Requires iverilog + vvp (Icarus Verilog >= 10) for rtl-test / skew-test.  NOT executed in the authoring
# environment (no HDL simulator was available) - see README "Verification status".
IVERILOG ?= iverilog
VVP      ?= vvp
RTL      := rtl/armse_skew_mem.v rtl/armse_obuf.v rtl/armse_dma.v rtl/armse_compute.v rtl/armse_top.v

.PHONY: all py-test vectors rtl-test rtl-model-test skew-test post-test lint clean
all: py-test vectors rtl-model-test

py-test:
	python3 -m unittest discover -s tests -t .

vectors:
	python3 -m armse.rtlexport . --random 40
	python3 -m armse.quant .

rtl-model-test: vectors
	python3 -m armse.rtlmodel rtl_vectors

skew-test:
	$(IVERILOG) -g2012 -o /tmp/tb_skew.vvp tb/tb_skew_mem.v rtl/armse_skew_mem.v
	$(VVP) /tmp/tb_skew.vvp | tee /tmp/tb_skew.log
	@grep -q "RESULT tb_skew_mem: PASS" /tmp/tb_skew.log

# Fails (non-zero exit) on compile errors, on any ERROR line, and unless every case prints RESULT ... PASS.
rtl-test: vectors skew-test post-test
	@fail=0; for d in rtl_vectors/t*; do \
	  out=$$(cd $$d && $(IVERILOG) -g2005 -I../../rtl -I. -o sim.vvp ../../tb/tb_armse.v $(addprefix ../../,$(RTL)) 2>&1 && $(VVP) sim.vvp 2>&1) \
	    || { echo "$$out"; echo "COMPILE/RUN FAILED: $$d"; fail=1; continue; }; \
	  echo "$$out" | grep -E "RESULT|CYCLES|ERROR"; \
	  echo "$$out" | grep -q "RESULT .*: PASS" || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then echo "RTL TESTS: FAILURES"; exit 1; else echo "RTL TESTS: ALL PASS"; fi

post-test: vectors
	@fail=0; for d in post_vectors/p*; do \
	  out=$$(cd $$d && $(IVERILOG) -g2005 -I. -o post.vvp ../../tb/tb_post.v ../../rtl/armse_post.v 2>&1 && $(VVP) post.vvp 2>&1) \
	    || { echo "$$out"; echo "COMPILE/RUN FAILED: $$d"; fail=1; continue; }; \
	  echo "$$out" | grep -E "RESULT|ERROR"; \
	  echo "$$out" | grep -q "RESULT .*: PASS" || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then echo "POST TESTS: FAILURES"; exit 1; else echo "POST TESTS: ALL PASS"; fi

# Optional static check (Verilator): warnings only, never fatal.
lint:
	-verilator --lint-only -Wall -Wno-fatal -Irtl --top-module armse_top $(RTL)

clean:
	rm -f rtl_vectors/*/sim.vvp /tmp/tb_skew.vvp /tmp/tb_skew.log
