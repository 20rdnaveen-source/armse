# ARMSE diagrams

PNG versions (for slides/reports): `docs/img/armse_block_diagram.png`, `docs/img/armse_flow.png` (regenerate with `python tools/draw_diagrams.py`).
The Mermaid versions below render directly on GitHub. They describe the RTL in `rtl/` and the host software in `armse/`; the green/yellow
blocks exist as Verilog, the blue ones are host software (Python), and the dashed items are model-only (no RTL).

## 1. Block diagram
```mermaid
flowchart LR
  subgraph HOST["Host software (Python, not RTL)"]
    S["scheduler: tile x order x buffering<br/>cost model + calibrated D_v2"]
    V["simulator: deadlock / hazard check"]
    G["descriptor generator"]
    S --> V --> G
    X["model-only: counter-driven re-planning,<br/>deadline / energy-bounded planner"]:::model
  end
  subgraph TOP["armse_top (RTL)"]
    DP["DMA program memory<br/>128-bit descriptors"]
    CP["compute program memory<br/>128-bit descriptors"]
    DMA["armse_dma<br/>sequencer + transfer engine"]
    CMP["armse_compute<br/>sequencer + NB MAC lanes"]
    CNT["dependency counters<br/>dma_done / cp_done"]
    A["A buffer (skew mem)"]
    B["B buffer (skew mem)"]
    O["output / psum buffer"]
    PC["performance counters"]
    DP --> DMA
    CP --> CMP
    DMA <--> CNT
    CMP <--> CNT
    DMA -->|"4 elem/cycle"| A
    DMA -->|"4 elem/cycle"| B
    A -->|"NB lanes"| CMP
    B -->|"NB lanes"| CMP
    CMP <-->|"psum"| O
    O -->|"writeback"| DMA
    DMA -.-> PC
    CMP -.-> PC
  end
  EXT["external memory<br/>(testbench model; DDR/AXI later)"]
  G -->|"h_we/h_sel/h_addr/h_wdata"| DP
  G -->|"h_sel = 1"| CP
  DMA <-->|"x_cmd / x_rdata / x_wdata"| EXT
  classDef model stroke-dasharray: 5 5;
```

## 2. Scheduling and execution flow
```mermaid
flowchart LR
  W["workload (GEMM M,N,K)"] --> E["enumerate candidates<br/>tile x order x buffering"]
  E --> R["reject illegal<br/>capacity / fields / program length"]
  R --> K["rank by cost model<br/>(closed form; calibrated D_v2)"]
  K --> Y["verify with simulator<br/>deadlock + hazards (+ top-k feedback)"]
  Y --> D["emit descriptor programs"]
  D --> H["RTL executes: two counters,<br/>decoupled DMA and compute"]
  R -. "nothing fits" .-> F["minimal 1x1xK tile or NoFitError"]
  H -. "counters (model only)" .-> K
```

## 3. Double-buffered execution (timeline idea)
```mermaid
sequenceDiagram
  participant DMA as DMA engine
  participant CMP as Compute engine
  Note over DMA,CMP: each job starts only when the other engine's counter >= DEP
  DMA->>DMA: load tile 0 (slot 0)
  DMA->>CMP: dma_done = 2 (operands of iteration 0 ready)
  par overlap
    CMP->>CMP: compute iteration 0 (slot 0)
  and
    DMA->>DMA: load tile 1 (slot 1)
  end
  CMP->>DMA: cp_done = 1 (slot 0 free again)
  DMA->>DMA: writeback of finished tile queued behind the next loads
```
