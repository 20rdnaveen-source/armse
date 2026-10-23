"""Benchmark suite. Conv, attention and MLP are lowered to INT8 GEMM (see docs for limitations)."""
from .config import Workload


def gemm(name, M, N, K, family="gemm"):
    return Workload(name, M, N, K, family)


def conv2d(name, H, W, Cin, Cout, k, stride=1, pad=0):
    """im2col lowering: M = OH*OW, K = Cin*k*k, N = Cout.  Traffic counts the *lowered* A matrix,
    i.e. sliding-window input reuse is NOT modelled (conservative for conv)."""
    oh = (H + 2 * pad - k) // stride + 1
    ow = (W + 2 * pad - k) // stride + 1
    return Workload(name, oh * ow, Cout, Cin * k * k, "conv")


def attention(name, seq, d):
    """QK^T and PV as two GEMMs (softmax and fusion of the score matrix are out of scope)."""
    return [Workload(f"{name}_QKt", seq, seq, d, "attention"),
            Workload(f"{name}_PV", seq, d, seq, "attention")]


def mlp(name, batch, dims):
    return [Workload(f"{name}_L{i}", batch, dims[i + 1], dims[i], "mlp") for i in range(len(dims) - 1)]


def suite():
    s = []
    for n in (16, 32, 64, 128, 192, 256):
        s.append(gemm(f"gemm_sq{n}", n, n, n))
    s += [gemm("gemv_1x256x256", 1, 256, 256, "gemv"),
          gemm("gemv_256x1x256", 256, 1, 256, "gemv"),
          gemm("tall_512x16x64", 512, 16, 64, "skinny"),
          gemm("wide_16x512x64", 16, 512, 64, "skinny"),
          gemm("deepK_32x32x1024", 32, 32, 1024, "skinny"),
          gemm("ragged_37x29x53", 37, 29, 53, "ragged"),
          gemm("ragged_101x67x131", 101, 67, 131, "ragged"),
          gemm("ragged_250x255x249", 250, 255, 249, "ragged")]
    s += mlp("mlp_b1", 1, [784, 256, 128, 10])
    s += mlp("mlp_b32", 32, [784, 256, 128, 10])
    s += [conv2d("conv3x3_16to32_28", 28, 28, 16, 32, 3, 1, 1),
          conv2d("conv1x1_64to64_14", 14, 14, 64, 64, 1),
          conv2d("conv5x5_3to8_32", 32, 32, 3, 8, 5, 1, 2),
          conv2d("conv3x3_32to64_14", 14, 14, 32, 64, 3, 1, 1)]
    s += attention("attn_s64_d32", 64, 32)
    s += attention("attn_s128_d64", 128, 64)
    # capacity-stress workloads: operands far larger than on-chip buffers
    s += [gemm("stress_512x512x64", 512, 512, 64, "stress"),
          gemm("stress_64x64x2048", 64, 64, 2048, "stress")]
    return s


def smoke_suite():
    return [w for w in suite() if w.name in (
        "gemm_sq64", "gemv_1x256x256", "tall_512x16x64", "ragged_37x29x53", "mlp_b32_L0",
        "conv3x3_16to32_28", "attn_s64_d32_QKt", "deepK_32x32x1024")]
