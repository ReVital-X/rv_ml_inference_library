# Bare-Metal C Fixed-Point Neural Network Inference Engine

A lightweight, memory-efficient C neural network inference engine designed for bare-metal microcontrollers and embedded targets (e.g., RISC-V QEMU). Built for high numerical integrity with zero dynamic heap allocation (`malloc`).

---

## Key Technical Features

* **Zero Heap Footprint (`malloc`-Free):** Operates strictly using static `.bss` memory allocation, ideal for deterministic embedded systems and safety-critical execution.
* **Fixed-Point Arithmetic:** `int8_t` weight and activation storage paired with `int32_t` accumulation to prevent register overflow during Multiply-Accumulate (MAC) loops.
* **Ping-Pong Buffer Strategy:** Alternates intermediate layer tensors across two static scratch buffers (`$2 \times \text{max\_layer\_size}$`), minimizing total SRAM consumption across arbitrary network depths.
* **Fused Activation Kernels:** Supports fused `ReLU`, `ReLU6`, and direct linear pass-through directly inside the scale reduction stage.
* **Contiguous Memory Layout:** Weight tensors are stored as `[output_nodes x input_nodes]` to align with PyTorch native memory structures and maximize CPU cache line prefetching.
* **Batch-Aware 2D Kernels:** Implemented with generalized 2D matrix multiplication (`rows x cols`), cleanly degrading to 1D vector processing when `batch_size = 1`.

---

## Repository Structure

```text
.
├── include/
│   └── ml_engine.h         # Core data structures, activation types, and kernel APIs
├── src/
│   └── ml_engine.c         # Kernels: linear_forward, requantize, apply_activation
├── tests/
│   └── test_engine.c       # Host unit testing suite (Unity framework)
├── scripts/                # (Planned) PyTorch quantization & C header exporter
├── main.c                  # (Planned) Bare-metal QEMU target entry point
└── Makefile                # Host test and cross-compilation target build configuration

```

---

## Memory & Execution Model

### 1. Matrix Multiplication Kernel

The core linear kernel computes:


$$\text{Output} = \text{Requantize}\left(\text{Input} \cdot \mathbf{W}^T + \mathbf{B}\right)$$

```text
Input Tensor (R x K)  ×  Weight Matrix (C x K)^T  +  Bias (C)  ──►  Int32 Acc  ──►  Requantize & Activation  ──►  Int8 Output (R x C)

```

### 2. Static Scratch Buffer Reuse (Ping-Ponging)

```text
Layer 0 Input  ──► [ Scratch Buffer 0 ] ──► Layer 1 Execution ──► [ Scratch Buffer 1 ] ──► Layer 2 Execution ──► Output

```

---

## Implementation Status

* [x] Fixed-point `int8` multiply-accumulate (MAC) loop with `int32_t` accumulator
* [x] Round-to-nearest fixed-point `requantize` helper
* [x] Fused non-linear activation functions (`ReLU`, `ReLU6`, `NONE`)
* [x] Static `.bss` ping-pong buffer management in `network_forward`
* [ ] Explicit `EngineStatus` return codes and parameter dimension validation
* [ ] Python PyTorch exporter script (`export_to_c.py`) for automated C header generation
* [ ] Bare-metal RISC-V startup (`crt0.s`), linker script (`linker.ld`), and QEMU integration
* [ ] Automated CI/CD build pipeline via GitHub Actions

---

## Building and Testing (Host Environment)

### Run Unit Tests

To compile and execute the test suite on your local host using GCC:

```bash
make test

```

### Clean Build Artifacts

```bash
make clean

```

```

```