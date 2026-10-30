"""INT32 -> INT8 requantisation + optional ReLU: the bit-exact definition shared by the Python model and rtl/armse_post.v.

  p  = x * M                       (x signed 32-bit, M unsigned 16-bit; exact)
  r  = (p + 2^(s-1)) >> s          (s in 0..31; arithmetic shift = floor; s = 0 -> r = p)
  y  = r + zp                      (zp signed 8-bit)
  lo = zp if relu else -128        (ReLU in the quantised domain clamps at the zero point)
  out = clamp(y, lo, 127)          (signed 8-bit)
python3 -m armse.quant [root]  writes post_vectors/pNN/{in.hex,exp.hex,meta.vh} for tb/tb_post.v"""
import os
import sys
import numpy as np

SETS = [  # (M, shift, zp, relu)
    (1, 0, 0, 0), (1, 1, 0, 0), (23170, 15, 0, 0), (16384, 14, 0, 1), (65535, 31, -5, 0),
    (1, 31, 127, 1), (40000, 8, -128, 0), (12345, 12, 10, 1), (65535, 0, 0, 0), (7, 3, -100, 1),
]


def requant(x, M, s, zp, relu):
    x = np.asarray(x, dtype=np.int64)
    p = x * int(M)
    r = (p + (1 << (s - 1))) >> s if s > 0 else p
    y = r + int(zp)
    lo = int(zp) if relu else -128
    return np.clip(y, lo, 127).astype(np.int8)


def sample_inputs(n, seed):
    rng = np.random.default_rng(seed)
    parts = [rng.integers(-2 ** 31, 2 ** 31, n // 3, dtype=np.int64), rng.integers(-5000, 5000, n // 3, dtype=np.int64),
             rng.integers(-300, 300, n - 2 * (n // 3), dtype=np.int64),
             np.array([0, 1, -1, 2 ** 31 - 1, -2 ** 31, 127, 128, -128, -129], dtype=np.int64)]
    return np.concatenate(parts).astype(np.int64)


def export(root=".", n=300):
    out = os.path.join(root, "post_vectors")
    for i, (M, s, zp, relu) in enumerate(SETS):
        d = os.path.join(out, f"p{i:02d}")
        os.makedirs(d, exist_ok=True)
        x = sample_inputs(n, i)
        y = requant(x, M, s, zp, relu)
        open(os.path.join(d, "in.hex"), "w").write("\n".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in x) + "\n")
        open(os.path.join(d, "exp.hex"), "w").write("\n".join(f"{int(v) & 0xFF:02x}" for v in y) + "\n")
        open(os.path.join(d, "meta.vh"), "w").write(
            f"`define M_P {M}\n`define S_P {s}\n`define ZP_P {zp}\n`define RELU_P {relu}\n`define N_P {len(x)}\n`define TEST_NAME \"post_p{i:02d}\"\n")
    return len(SETS)


if __name__ == "__main__":
    print(export(sys.argv[1] if len(sys.argv) > 1 else "."), "post test sets written")
