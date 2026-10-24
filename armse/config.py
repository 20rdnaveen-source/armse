"""Hardware configuration, workloads and error types."""
from dataclasses import dataclass
 
 
class ArmseError(Exception):
    pass
 
 
class UnsupportedOp(ArmseError):
    pass
 
 
class ScheduleError(ArmseError):
    pass
 
 
class NoFitError(ArmseError):
    pass
 
 
class DeadlockError(ArmseError):
    pass
 
 
class HazardError(ArmseError):
    pass
 
 
def cdiv(a: int, b: int) -> int:
    return -(-a // b)
 
 
@dataclass(frozen=True)
class HwConfig:
    """Parameterised local-memory + transfer model (see docs/ARCHITECTURE.md).
 
    lanes        MAC lanes along K == number of banks in the A/B buffers (power of two >= 4)
    bus_bytes    external bus bytes per beat (RTL implements 4; model accepts multiples of 4)
    row_lat      cycles between a burst command and its first data beat
    a_depth      words per bank in the A buffer  (A capacity = lanes*a_depth bytes, all slots)
    b_depth      words per bank in the B buffer
    o_words      32-bit words in the output/partial-sum buffer (all slots)
    cmp_ovh      fixed compute-job overhead cycles (pipeline drain + sequencer), derived from RTL
    job_ovh      fixed DMA-job overhead cycles, derived from RTL
    start_off    first cycle at which an engine may start after `go`
    bank_mode    'skew' (implemented in RTL) or 'naive' (row-major, ablation only)
    """
    lanes: int = 8
    bus_bytes: int = 4
    row_lat: int = 12
    a_depth: int = 256
    b_depth: int = 256
    o_words: int = 1024
    cmp_ovh: int = 4
    job_ovh: int = 2
    start_off: int = 1
    bank_mode: str = "skew"
    max_dim: int = 255          # tile fields are 8 bits in the descriptor format
    max_entries: int = 1048575  # dependency counters / descriptor indices are 20 bits
    prog_fetch_block: int = 0   # MODEL ONLY: >0 streams descriptors from external memory in blocks of this many DMA entries
 
    def __post_init__(self):
        if self.lanes < 4 or self.lanes & (self.lanes - 1):
            raise ValueError("lanes must be a power of two >= 4")
        if self.bus_bytes < 4 or self.bus_bytes % 4:
            raise ValueError("bus_bytes must be a positive multiple of 4")
        if self.bank_mode not in ("skew", "naive"):
            raise ValueError("bank_mode must be 'skew' or 'naive'")
        if self.prog_fetch_block < 0:
            raise ValueError("prog_fetch_block must be >= 0")
        if min(self.a_depth, self.b_depth, self.o_words) < 0:
            raise ValueError("buffer sizes must be non-negative")
 
    @property
    def row_fix(self) -> int:
        """Fixed cycles per burst row: 1 command cycle + row_lat wait."""
        return self.row_lat + 1
 
    @property
    def onchip_bytes(self) -> int:
        return self.lanes * (self.a_depth + self.b_depth) + 4 * self.o_words
 
    def scaled(self, f: float) -> "HwConfig":
        """Scale all three buffer capacities by f (rounded down, min 1 word)."""
        g = lambda v: max(1, int(v * f))
        return HwConfig(**{**self.__dict__, "a_depth": g(self.a_depth),
                           "b_depth": g(self.b_depth), "o_words": g(self.o_words)})
 
    def replace(self, **kw) -> "HwConfig":
        return HwConfig(**{**self.__dict__, **kw})
 
 
@dataclass(frozen=True)
class Workload:
    """One INT8 GEMM  C[MxN] (int32) = A[MxK] * B[KxN].  Conv/attention/MLP are lowered to GEMM."""
    name: str
    M: int
    N: int
    K: int
    family: str = "gemm"
    op: str = "gemm"
    out_bytes: int = 4          # bytes per output element written to external memory (4 = int32, 1 = requantised int8)
    weight_bits: int = 8        # MODEL ONLY: external storage width of B (8, or 4 = packed INT4 + per-column 16-bit scale per tile load)
 
    @property
    def macs(self) -> int:
        return self.M * self.N * self.K
 
    @property
    def compulsory_in(self) -> int:
        return self.M * self.K + self.K * self.N
 
    @property
    def compulsory_out(self) -> int:
        return self.out_bytes * self.M * self.N
 