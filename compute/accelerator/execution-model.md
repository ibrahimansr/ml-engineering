# GPU Execution Model and Kernels for ML Engineers

## Prerequisites

- [Computer architecture](../../foundations/computer-architecture.md) - caches, memory bandwidth vs latency, SIMD, and what an instruction pipeline is; this chapter assumes all of it.
- [Concurrency](../../foundations/concurrency.md) - threads, synchronization, barriers, atomics and asynchronous execution, which is how GPU kernels and CUDA streams behave.
- [Linux and the OS](../../foundations/linux-and-os.md) - processes, system calls and drivers, needed for the launch-overhead and profiling sections where the CPU side matters.
- [Transformer architectures](../../models/architectures.md) - attention, GQA, RMSNorm and the MLP block, the ops whose kernels this chapter dissects.

The [Accelerators](README.md) chapter tells you what a GPU is rated at - [TFLOPS](README.md#tflops), [memory bandwidth](README.md#accelerator-memory-size-and-speed), [caches](README.md#caches), and how much of the peak is [actually achievable](README.md#maximum-achievable-and-sustainable-flops). This chapter is about *why* a given kernel (a function compiled for the GPU and launched from the CPU to run across thousands of GPU threads at once) gets the fraction it gets: how the hardware schedules threads (independent streams of instructions, each with its own registers), where the bytes live, what a tensor core instruction looks like, and how the kernels that dominate an LLM step (GEMM, attention, normalization) are structured to hit or miss the roofline (the performance ceiling set by whichever of compute or memory bandwidth runs out first, [defined below](#roofline-per-op)). It ends with the tools that tell you which case you are in.

Two terms used throughout: *HBM* (High Bandwidth Memory) is the GPU's main memory, the tens of GB of DRAM stacked next to the chip where tensors live; the *host* is the CPU and its RAM, as opposed to the *device* (the GPU).

Most of the numbers below are for NVIDIA H100 SXM (Hopper, `sm_90`) with B200 (Blackwell, `sm_100`) differences called out. Every hardware number is either taken from a named NVIDIA/AMD document or derived by arithmetic that is shown. Where a number is not published by the vendor and comes from microbenchmarking literature or from my own inference, it is marked as such.

Nothing in this chapter is a benchmark result. The one kernel in it comes with a benchmark harness but no measured numbers - run it on your hardware.


## Glossary

- CTA: Cooperative Thread Array - the hardware name for a thread block
- DSMEM: Distributed Shared Memory - shared memory of other CTAs in the same cluster
- GPC: Graphics Processing Cluster - a group of SMs; a thread block cluster is co-scheduled within one
- ILP: Instruction-Level Parallelism
- LDS: Local Data Share - AMD's counterpart of shared memory
- MUFU: Multi-Function Unit - the special function unit (`exp2`, `rsqrt`, `sin`, ...)
- SIMT: Single Instruction Multiple Threads
- SOL: Speed Of Light - Nsight Compute's "% of peak" for a hardware unit
- smem: shared memory
- TMA: Tensor Memory Accelerator (Hopper and later)
- TMEM: Tensor Memory (Blackwell `sm_100`)
- warpgroup: 4 consecutive warps (128 threads), the issuing unit of Hopper `wgmma`

For the general accelerator glossary see [the main chapter](README.md#glossary).


## SMs, warps and schedulers

### The hierarchy

A GPU is made of many Streaming Multiprocessors (SMs), each an independent processor with its own registers (the fastest storage, private to each thread), shared memory and schedulers. A kernel launch is a grid of thread blocks (CTAs, groups of up to 1024 threads that can cooperate through shared memory). Each CTA is assigned to one SM for its whole lifetime and never migrates. Inside the SM the CTA's threads are grouped into warps of 32 threads, and the warp - not the thread - is what the hardware schedules.

The full GH100 die (H100 SXM ships with 132 of these 144 SMs enabled):

![full GH100 die with 144 SMs](images/h100-full-gpu-144-sms.png)

([source](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/))

One H100 SM - four partitions, each with its own warp scheduler, register file and tensor core:

![H100 streaming multiprocessor](images/h100-sm.png)

([source](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/))

On Hopper (per the [H100 architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) and the [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)) each SM is split into 4 sub-partitions (SMSPs). Each sub-partition has:

- 1 warp scheduler (the hardware that picks which ready warp runs its next instruction) that can issue 1 instruction per clock from one of its resident warps
- a 16K x 32-bit register file slice (64KB), so 64K registers = 256KB per SM
- 32 FP32 lanes, 16 INT32, 16 FP64 lanes
- 1 tensor core (4 per SM, 528 on the 132-SM H100 SXM - the figure the [TFLOPS derivation](README.md#how-to-calculate-theoretical-tflops) uses)
- special function units; the FlashAttention-3 paper states H100 SXM5 has "989 TFLOPS of FP16 matmul but only 3.9 TFLOPS of special functions", which works out to `3.9e12 / (132 SMs * 1.83e9 Hz) = 16` MUFU ops per clock per SM, i.e. 4 per sub-partition (derived; the per-SM count is consistent with the white paper but I did not find it stated as a single number)

Per-SM limits for compute capability 9.0, from the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications) table:

| resource                          | H100 (`sm_90`) | B200 (`sm_100`) |
| :-------------------------------- | -------------: | --------------: |
| max resident warps per SM         |             64 |              64 |
| max resident threads per SM       |           2048 |            2048 |
| max resident CTAs per SM          |             32 |              32 |
| max threads per CTA               |           1024 |            1024 |
| 32-bit registers per SM           |            64K |             64K |
| max registers per thread          |            255 |             255 |
| max shared memory per SM          |          228KB |           228KB |
| max shared memory per CTA (opt-in)|          227KB |           227KB |
| SMs (product)                     |            132 |             148 |

The B200 column follows the [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), which keeps the Hopper per-SM limits for `sm_100`. The 148 SM count is what the [divisibility section](../../training/performance/README.md#do-more-sms-give-more-tflops) measured on a B200 SKU. If your toolkit's programming guide disagrees with this table, it wins.

### Latency hiding is the whole design

A GPU does not hide memory latency with caches and out-of-order execution as a CPU does. (Latency is the time from issuing a load until its data arrives; bandwidth is how many bytes per second arrive once the pipe is full.) It hides it by having many warps resident and switching between them for free: every resident warp's registers are permanently allocated in the register file, so a "context switch" is the scheduler picking a different warp on the next clock. When warp A issues a global load and its next instruction depends on the result, A stalls (in Nsight Compute vocabulary, on the *long scoreboard*) and the scheduler issues from warp B.

How much parallelism you need is Little's law: `bytes in flight = bandwidth * latency`. NVIDIA does not publish DRAM latency; microbenchmark studies of Hopper (e.g. Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", arXiv:2402.13499 (id unverified)) report several hundred ns. Taking 500ns as an assumption:

```
3.35e12 B/s * 500e-9 s = 1.675 MB in flight across the GPU
1.675 MB / 132 SMs     = ~12.4 KB in flight per SM
```

At one 16-byte vectorized load outstanding per thread, that is ~790 threads (~25 warps) per SM doing nothing but waiting on memory, just to saturate HBM. You get there with some combination of:

- **TLP** - more resident warps (occupancy)
- **ILP** - each thread issues several independent loads before consuming any
- **asynchronous copies** - `cp.async` (Ampere) or TMA (Hopper) move data into shared memory without holding a register or a thread, which is how modern GEMM and attention kernels run at low occupancy and still saturate the memory system

### Occupancy, worked

Occupancy is `resident warps / 64`. It is limited by whichever resource runs out first: registers, shared memory (a small on-chip scratchpad per SM that the threads of one CTA explicitly read and write to share data), the thread limit, or the CTA limit.

Allocation granularities matter. Registers are allocated per warp in units of 256 (so per-thread counts round up to a multiple of 8), and since compute capability 8.0 each CTA also reserves 1KB of shared memory for the system ([CUDA Programming Guide, shared memory section](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x)). The [CUDA Occupancy Calculator](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) (now in Nsight Compute) applies the exact rules; this is the arithmetic it does.

A kernel with 128 threads per CTA, 96 registers per thread, 48KB of shared memory per CTA, on H100:

```
warps per CTA        = 128 / 32                           = 4
registers per warp   = 96 * 32                            = 3072  (already a multiple of 256)
CTAs by registers    = floor(floor(65536 / 3072) / 4)     = floor(21 / 4) = 5
CTAs by smem         = floor(228KB / (48KB + 1KB))        = 4
CTAs by threads      = 2048 / 128                         = 16
CTAs by CTA limit    = 32
resident CTAs        = min(5, 4, 16, 32)                  = 4
occupancy            = 4 * 4 / 64                         = 25%
```

Shared memory is the binding limit. Some variants:

| regs/thread | smem/CTA | CTAs/SM | occupancy | binding limit  |
| ----------: | -------: | ------: | --------: | :------------- |
|          96 |     48KB |       4 |       25% | smem           |
|          96 |     32KB |       5 |     31.3% | registers      |
|          64 |     32KB |       6 |     37.5% | smem           |
|         128 |     48KB |       4 |       25% | registers/smem |
|         168 |        0 |       3 |     18.8% | registers      |

The ceiling for 100% occupancy is `65536 / 2048 = 32` registers per thread, which almost no compute-heavy kernel meets. That is fine. High occupancy is a means (latency hiding) and not a goal: a Hopper GEMM typically runs 1 CTA of 384 threads per SM, 12 of 64 warps, about 19% occupancy, and sits near the tensor core roofline because TMA does the loads and each consumer thread holds a large accumulator tile in registers. Conversely, a memory-bound elementwise kernel at 25% occupancy with one scalar load per thread will not saturate HBM. Read occupancy together with the stall reasons ([Profiling](#profiling-nsys-and-ncu)), not on its own.

In code you control the register count with `__launch_bounds__(maxThreads, minBlocks)` or `-maxrregcount`, in Triton with `num_warps` (and the compiler's register allocation follows from the block size), and the shared memory with the tile sizes and `num_stages`. Forcing registers down below what the kernel needs causes *spills* to local memory, which lives in L1/L2/HBM - check `ptxas -v` output or the Nsight Compute "Launch Statistics" section.

### Divergence

The 32 threads of a warp share an instruction stream (one sequence of instructions). When a branch condition differs across the warp, both sides execute with inactive lanes masked off, so the cost is the sum of both paths. Since Volta each thread has its own program counter ("independent thread scheduling"), which makes divergent code *correct* in more cases (e.g. intra-warp locks), but it does not make it faster - divergent paths are still serialized.

For ML kernels divergence is rarely the bottleneck, because the data-dependent control flow is usually a boundary mask (`if (col < N)`) that diverges in at most one warp per row, or a causal mask in attention. The standard remedy for the latter is structural: FlashAttention skips fully-masked K/V blocks entirely and applies the elementwise mask only on the diagonal block, so only one block per row of blocks pays for the mask. Prefer predication (`tl.where`, masked loads) over branches for anything per-element.


## Memory hierarchy

### Capacities and bandwidths

| level              | H100 SXM                                     | B200                                    | scope                   | source / note                                                                 |
| :----------------- | :------------------------------------------- | :-------------------------------------- | :---------------------- | :---------------------------------------------------------------------------- |
| registers          | 256KB per SM, 33MiB total                    | 256KB per SM                            | per thread (private)    | 64K x 4B; `132 * 256KB = 33MiB`                                              |
| tensor memory      | -                                            | 256KB per SM                            | per CTA (allocated)     | `sm_100` only; 128 lanes x 512 columns x 4B, see [tcgen05](#blackwell-tcgen05-and-tmem) |
| L1 / shared memory | 256KB per SM combined, up to 228KB as smem   | 256KB per SM combined, up to 228KB smem | per CTA (smem) / per SM | Hopper / Blackwell tuning guides                                              |
| L2                 | 50MB                                         | 126MB (GB200 figure)                    | whole GPU (2 partitions)| [caches table](README.md#caches)                                              |
| HBM                | 80GB HBM3, 3.35TBps                          | 180GB HBM3e, 8.0TBps                    | whole GPU               | [memory table](README.md#accelerator-memory-size-and-speed)                   |

Bandwidths NVIDIA does not publish as a single figure:

- **Shared memory**: 32 banks x 4 bytes per clock per SM = 128 B/clk/SM. At 1.83GHz this is `128 * 1.83e9 * 132 = 30.9TBps` aggregate on H100 (derived; `34.7TBps` with 148 SMs on B200 at the same clock, also derived - the B200 clock may differ). That is ~9x HBM.
- **L2**: not published. Microbenchmarks report a few times HBM bandwidth; treat any specific number as unverified. The L2 on H100 and A100 is split into two partitions connected by a crossbar (an on-chip switch connecting every SM to every L2 slice), and data homed in the far partition costs more ([Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)).
- **Registers**: effectively whatever the ALUs can consume - not a bottleneck you design around, but a *capacity* you design around (see the accumulator arithmetic in [GEMM tiling](#gemm-tiling-hierarchy)).

B200 is two reticle-sized dies joined by NVIDIA's 10TBps NV-HBI link and presented to software as one GPU ([Blackwell architecture page](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)). L2 and HBM are physically split between the dies, so locality effects similar to Hopper's two L2 partitions exist at a larger scale; NVIDIA does not publish the cross-die latency penalty.

The ratio that matters for kernel design: H100 does ~4094 dense BF16 FLOP per clock per SM (`989e12 / (132 * 1.83e9)`), but HBM delivers `3.35e12 / (132 * 1.83e9) = 13.9` bytes per clock per SM. Every byte fetched from HBM has to be reused ~295 times in FLOPs for the tensor cores not to starve. Registers and shared memory exist to provide that reuse.

### Coalescing

Global memory (the kernel's view of HBM, readable by every thread) is accessed in 32-byte sectors, grouped into 128-byte cache lines (the unit a cache stores and evicts). When a warp issues a load, the hardware computes the set of sectors touched by the 32 addresses and issues one transaction per sector. If thread `i` reads `float` element `i`, the warp touches `32 * 4 = 128` contiguous bytes = 4 sectors: perfect. If thread `i` reads element `i * stride`:

| stride (float) | sectors touched | bytes moved | bytes used | efficiency |
| -------------: | --------------: | ----------: | ---------: | ---------: |
|              1 |               4 |         128 |        128 |       100% |
|              2 |               8 |         256 |        128 |        50% |
|              8 |              32 |        1024 |        128 |      12.5% |
|            >=8 |              32 |        1024 |        128 |      12.5% |

In ML code this shows up as reading a tensor along the wrong dimension: a kernel that reduces over the last dim of a `(B, T, C)` tensor where the tensor is actually a transposed view. The fix is almost always to assign the contiguous dimension to consecutive threads (in Triton: the innermost `tl.arange` along the stride-1 dim), and to load 16 bytes per thread (8 bf16) so each warp moves 512B per instruction. This needs the base pointer and row stride to be 16-byte aligned - one concrete reason dims should be multiples of 8 for bf16 (see [why multiples](#why-dims-want-multiples-of-8-16-64-128)).

### Bank conflicts

Shared memory has 32 banks (independent memory sub-arrays that can each serve one 4-byte word per clock), each 4 bytes wide; word address `a` lives in bank `(a / 4) % 32`. A warp's shared memory access completes in one cycle if all 32 threads hit distinct banks (or the same word - broadcast). If `k` threads hit different words in the same bank, it takes `k` cycles: a `k`-way conflict.

The canonical case is a transpose through a `float tile[32][32]`: writing rows is conflict-free, reading a column has every thread hitting bank `(row * 32 + c) % 32 = c`, a 32-way conflict. Two fixes:

- **padding**: `float tile[32][33]` shifts each row by one bank. Simple, wastes 3% of the buffer, and breaks the alignment that vectorized and TMA accesses need.
- **swizzling**: permute the column index by XOR-ing it with bits of the row index, so a logical column maps to 32 different banks. This keeps rows aligned and is what CUTLASS, Triton and TMA use. TMA supports 32B, 64B and 128B swizzle modes in hardware, and Hopper `wgmma` / Blackwell `tcgen05` shared memory descriptors expect operands in exactly those swizzled layouts.

`ldmatrix` (Ampere+) loads 8x8 matrix fragments from shared memory into the register layout `mma.sync` expects; feeding it from an un-swizzled row-major tile with a 128B row pitch is a classic 8-way conflict.

### TMA

The Tensor Memory Accelerator (Hopper and later) is a DMA engine (a hardware unit that copies memory on its own while the processor does other work) per SM that copies 1D-5D tiles between global and shared memory. You describe the global tensor once on the host with a *tensor map* (`cuTensorMapEncodeTiled`: base address, dims, strides, box size, swizzle mode, OOB fill), and then a single thread issues `cp.async.bulk.tensor` with tile coordinates. Properties that change how kernels are written:

- **One thread issues a whole tile.** No address arithmetic per thread, no registers held for the data, so the loading warps can be few and lightweight ([warp specialization](#flashattention-3-on-hopper)).
- **Completion via mbarrier with transaction counts.** The consumer waits on an `mbarrier` (a barrier object in shared memory: threads block on it until a set number of arrivals or bytes have been recorded) that tracks the expected byte count, rather than a `__syncthreads()`.
- **Bounds handling in hardware**: out-of-bounds elements are filled (zero or NaN), so tails need no masking code.
- **Constraints**: global strides must be multiples of 16 bytes and the global base address 16-byte aligned ([CUDA Driver API, `cuTensorMapEncodeTiled`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)). A bf16 tensor whose row has an odd number of elements cannot use TMA directly and falls back to slower paths.
- **Multicast**: within a cluster, one TMA load can be written into the shared memory of several CTAs at once.

TMA vs Ampere's per-thread async copies:

![TMA on H100 vs LDGSTS on A100](images/tma-vs-ldgsts.jpg)

([source](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/))

### Thread block clusters and distributed shared memory

Hopper adds a level between CTA and grid: a *cluster* of CTAs guaranteed to be co-scheduled on SMs of the same GPC. Portable cluster size is up to 8 CTAs; H100 allows 16 with the `cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in ([CUDA Programming Guide, thread block clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)). CTAs in a cluster can:

- read, write and perform atomics (read-modify-write operations such as add that no other thread can interleave with) on each other's shared memory (DSMEM) through the SM-to-SM network, faster than going through L2
- synchronize with cluster-wide barriers
- receive TMA multicast loads

Grids of clusters of thread blocks:

![thread block clusters](images/thread-block-clusters.jpg)

([source](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/))

The GEMM use: CTAs computing output tiles in the same row of `C` all need the same `A` tile. With a 2x1 cluster, one TMA multicast fetches `A` once for both CTAs, halving that operand's L2 traffic. CUTLASS exposes this as the `ClusterShape` template parameter. On Blackwell, the 2-SM `tcgen05` MMA builds directly on clusters (below).


## Tensor cores

### Instruction generations

| generation      | instruction                    | issued by                        | operands                                   | accumulator         | bf16 shape (M x N x K)        |
| :-------------- | :----------------------------- | :------------------------------- | :----------------------------------------- | :------------------ | :---------------------------- |
| Ampere `sm_80`  | `mma.sync`                     | 1 warp, synchronous              | A, B in registers                          | registers           | `m16n8k16`                    |
| Hopper `sm_90a` | `wgmma.mma_async`              | 1 warpgroup (4 warps), async     | A in registers or smem, B in smem          | registers           | `m64nNk16`, N = 8..256 step 8 |
| Blackwell `sm_100a` | `tcgen05.mma`              | 1 thread, async                  | A in smem or TMEM, B in smem               | TMEM                | M = 64/128 (1 SM), 128/256 (2 SM); N up to 256 |

Sources: [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) sections on `mma`, `wgmma` and `tcgen05`. The K extent is 32 bytes of the input type throughout: `k16` for 16-bit types, `k32` for FP8, `k8` for TF32. The Blackwell M/N ranges above are from my reading of the PTX ISA and should be checked against your CUDA version's documentation (uncertain on the exact allowed N steps per M).

Note the `a` suffix: `sm_90a` and `sm_100a` are *architecture-specific* targets. `wgmma` and `tcgen05` code compiled for them does not run on any other architecture, including later ones, which is why libraries ship separate kernels per generation and why a Hopper-tuned FlashAttention-3 binary does not run on Blackwell.

### Ampere: `mma.sync`

A warp collectively holds fragments of A (16x16), B (16x8) and C (16x8) in registers, in a fixed thread-to-element mapping, and issues a synchronous MMA. Operands come from shared memory via `ldmatrix`. The whole pipeline - global to shared (`cp.async`), shared to registers (`ldmatrix`), MMA - is driven by the same warps, so register pressure and instruction issue bandwidth both limit how large a tile each warp can own.

### Hopper: `wgmma`

`wgmma.mma_async` is issued by a warpgroup and computes a `64 x N x 16` product per instruction, with B (and optionally A) read *directly from shared memory* through a 64-bit matrix descriptor that encodes the address, leading-dimension byte offset, stride byte offset and swizzle mode. Consequences:

- no `ldmatrix` round trip through registers for B, freeing registers and issue slots
- asynchronous: the warpgroup issues a batch of `wgmma`s, then `wgmma.commit_group`, and later `wgmma.wait_group N` - so it can do other work (softmax, in FA3) while the tensor core runs
- the accumulator (the tile that `C += A @ B` sums into) is still in registers, distributed across the 128 threads. A `64 x 256` FP32 accumulator is `64 * 256 / 128 = 128` registers per thread. This is the register-capacity limit that shapes Hopper kernels.
- for FP8 inputs, `wgmma` requires both operands "K-major" (K contiguous) in shared memory; 16-bit types allow either major-ness for smem operands. This is why FA3's FP8 path transposes V in-kernel.

### Blackwell: `tcgen05` and TMEM

Blackwell datacenter parts (`sm_100`) add **Tensor Memory**: a 256KB per-SM on-chip memory organised as 128 lanes x 512 columns of 32-bit cells, allocated explicitly by a CTA (`tcgen05.alloc` / `tcgen05.dealloc`, in units of columns). The MMA:

- is issued by **a single thread**, not a warp or warpgroup
- writes its accumulator to TMEM rather than registers; the epilogue moves it out with `tcgen05.ld` (TMEM to registers)
- can pair the two SMs of a cluster (`cta_group::2`) to compute a 256-row tile, each SM holding half of the operands and half of the accumulator
- supports block-scaled low-precision formats (MXFP8/MXFP6/MXFP4 and NVFP4) with scale factors also held in TMEM

The practical effect: the accumulator no longer competes with softmax / epilogue state for the register file. A `128 x 256` FP32 accumulator is exactly 128 lanes x 256 columns, half of TMEM, which leaves room for double-buffering accumulators so the epilogue of tile `i` overlaps the MMAs of tile `i+1` without register pressure.

To my knowledge consumer Blackwell (`sm_120`, RTX 50-series) does not have TMEM or `tcgen05` and uses an extended `mma.sync` path instead - check before assuming a datacenter-Blackwell kernel ports.

### Why dims want multiples of 8, 16, 64, 128

There are three distinct mechanisms hiding behind "make it divisible", with different multiples:

1. **Alignment (8 elements for bf16, 16 for FP8)** - 16 bytes. Vectorized 128-bit loads, `cp.async` 16B copies and TMA strides all need 16-byte-aligned rows. Miss it and cuBLAS/cuBLASLt pick a kernel that can handle arbitrary alignment, which is usually much slower. This is the old "multiples of 8 for FP16 tensor cores" rule; tensor cores now technically run any shape, but the fast kernels assume alignment.
2. **Instruction shape (16 / 64)** - the MMA instruction's M/N/K. A K that is not a multiple of 16 (bf16) means padding the last K-slab with zeros inside the kernel; an N not a multiple of 8 wastes part of every `wgmma`.
3. **Tile quantization (64 / 128 / 256)** - the CTA tile. The output is covered by `BM x BN` tiles, and partial tiles cost as much as full ones.

On top of tile quantization sits **wave quantization**: tiles are executed in waves of `#SMs` (or `#SMs * CTAs-per-SM`). Both are covered with plots in the [Tile and wave quantization](../../training/performance/README.md#tile-and-wave-quantization) section and the SKU-dependence in [Do more SMs give more TFLOPS?](../../training/performance/README.md#do-more-sms-give-more-tflops); here is the combined arithmetic for H100 (132 SMs, 1 CTA per SM, `128 x 256` tile - a common Hopper CUTLASS/cuBLAS tile; the real kernel's tile is picked by heuristics and may differ):

```
efficiency = (M*N / (tiles * BM*BN))   # tile quantization
           * (tiles / (waves * SMs))   # wave quantization
tiles = ceil(M/BM) * ceil(N/BN),  waves = ceil(tiles / SMs)
```

| M x N           | tile      | tiles | waves (exact) | tile eff | wave eff | total |
| :-------------- | :-------- | ----: | ------------: | -------: | -------: | ----: |
| 4096 x 4096     | 128 x 256 |   512 |  4 (3.88)     |   100%   |   97.0%  | 97.0% |
| 4096 x 4160     | 128 x 256 |   544 |  5 (4.12)     |   95.6%  |   82.4%  | 78.8% |
| 4096 x 11008    | 128 x 256 |  1376 | 11 (10.42)    |   100%   |   94.8%  | 94.8% |
| 2048 x 2048     | 128 x 128 |   256 |  2 (1.94)     |   100%   |   97.0%  | 97.0% |
| 1536 x 1536     | 128 x 128 |   144 |  2 (1.09)     |   100%   |   54.5%  | 54.5% |
| 4096 x 4096 (B200, 148 SMs) | 128 x 256 | 512 | 4 (3.46) | 100% | 86.5% | 86.5% |

The `4160` row is the instructive one: adding 1.6% to N (64 columns) adds a 17th tile column, pushes the tile count over 4 waves, and costs 25% more time (`5/4` waves) for 1.6% more work. The last row is the same shape on a part with *more* SMs getting *worse* wave efficiency - the staircase from the performance chapter.

And a counterexample to a popular explanation: padding GPT-2's vocab from 50257 to 50304 (a multiple of 64) is often described as a tile-quantization win, but with `M = 4096` tokens and a `128 x 256` tile both shapes need exactly 6304 tiles and 48 waves. The arithmetic says the gain is from mechanism 1: 50257 is odd, so the bf16 rows of the LM head weight are not 16-byte aligned. (This is my derivation, not a profiled finding - verify with the kernel names in a trace, which usually reveal the alignment variant chosen.)

These effects are why `Linear` dimensions in well-sized models are multiples of 128 or 256 and why persistent / stream-K kernels exist ([below](#split-k-stream-k-and-persistent-kernels)).


## Roofline per op

### The model

For a kernel doing `F` FLOPs and moving `B` bytes to/from HBM:

```
time >= max(F / peak_flops, B / peak_bw)
arithmetic intensity  AI = F / B          [FLOP/byte]
ridge point           AI* = peak_flops / peak_bw
```

If `AI < AI*` the kernel is memory-bound and its best possible time is `B / peak_bw`, no matter how clever the arithmetic. Ridge points from the vendor peaks in the [TFLOPS table](README.md#tflops-comparison-table) and [memory table](README.md#accelerator-memory-size-and-speed):

| accelerator | bf16 dense TFLOPS | HBM TBps | bf16 ridge (FLOP/B) | fp8 ridge |
| :---------- | ----------------: | -------: | ------------------: | --------: |
| H100 SXM    |               989 |     3.35 |                 295 |       591 |
| B200        |              2250 |     8.00 |                 281 |       563 |
| MI300X      |              1307 |     5.30 |                 247 |         - |

Use [achievable](README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table) rather than theoretical peaks for planning; the ridge moves left by a similar factor for both, so the classification below does not change.

### GEMM

`C (M, N) = A (M, K) @ B (K, N)`, each element `b` bytes, `C` written once, `A` and `B` read once (the ideal; achieved only if tiling gives enough reuse):

```
F  = 2 * M * N * K                         # one multiply + one add per MAC
B  = b * (M*K + K*N + M*N)
AI = 2*M*N*K / (b * (M*K + K*N + M*N))
```

Cases, bf16 (`b = 2`):

- **Square** `M = N = K = n`: `AI = 2n^3 / (2 * 3n^2) = n/3`. At `n = 4096`, `AI = 1365` - well above the ridge.
- **Skinny** `M << N, K` (decode, or the per-token weight GEMMs at small batch): the `K*N` weight term dominates, and `AI -> 2*M*N*K / (2*K*N) = M`. Intensity equals the number of tokens in the batch.

| M (tokens) | AI for N = K = 4096, bf16 |
| ---------: | ------------------------: |
|          1 |                      1.00 |
|         16 |                      15.9 |
|        128 |                       120 |
|        256 |                       228 |
|        512 |                       410 |
|       1024 |                       683 |

Solving `AI(M) = 295` gives `M = 345`: a `4096 x 4096` bf16 projection on H100 is memory-bound below ~345 tokens. With FP8 weights *and* activations (`b = 1`) both AI and ridge double, so the crossover stays at ~345 tokens; with FP8 weights and bf16 activations the weight bytes halve and the crossover moves down. This is the arithmetic behind continuous batching in inference: a decode step at batch 1 is a GEMV with `AI ~ 1`, bound by reading the weights. For an 8B model in bf16 (16.06GB of weights), that floor is `16.06e9 / 3.35e12 = 4.8ms` per token on H100 regardless of kernel quality.

### Normalization, softmax and elementwise

Per element, bf16 in and out, each element read once and written once (4 bytes total):

| op                          | FLOPs / element (approx.) | AI (FLOP/B) | bound  |
| :-------------------------- | ------------------------: | ----------: | :----- |
| add / mul / cast            |                         1 |        0.25 | memory |
| GELU / SiLU                 |                     ~5-10 |        ~1-3 | memory |
| RMSNorm (`x * rstd * w`)    |                        ~4 |          ~1 | memory |
| LayerNorm                   |                       ~8  |          ~2 | memory |
| softmax (max, sub, exp, sum, div) |                  ~5 |        ~1.3 | memory |

All are two orders of magnitude below the ridge. Their cost is bytes, so the only levers are (a) moving fewer bytes - fusion - and (b) moving them at a higher fraction of peak bandwidth. On H100, a LayerNorm on an `(8192, 4096)` bf16 activation moves `2 * 8192 * 4096 * 2 = 134MB`, a floor of 40us, while the following `4096 x 4096` projection does `2 * 8192 * 4096 * 4096 = 275GFLOP`, a floor of 278us. At the roofline the norm is ~14% of the GEMM; in eager PyTorch with an fp32 upcast it is several separate kernels each paying the round trip, which is how normalization and elementwise ops end up a large fraction of a step for smaller models (see [Anatomy of Model's Operations](../../training/performance/README.md#anatomy-of-models-operations)).

Note softmax has an `exp` per element, which runs on the MUFU at 1/256 the rate of the tensor cores (`3.9 / 989`). That is irrelevant for a standalone softmax (memory-bound) but matters inside fused attention ([FA3](#flashattention-3-on-hopper)).

### Attention

Per head, sequence length `N`, head dim `d`, non-causal forward:

```
F = 2*N*N*d (Q K^T) + 2*N*N*d (P V) = 4 * N^2 * d        # causal: ~half
```

**Naive** (materialize `S` and `P` in HBM, bf16): read Q, K; write S; read S, write P; read P and V; write O:

```
B_naive ~ 2 * (4*N^2 + 4*N*d) bytes
N = 8192, d = 128:  B = 545MB  ->  163us at 3.35TBps
                    F = 34.4GFLOP -> 35us at 989TFLOPS
                    AI = 63  ->  memory-bound by ~4.7x
```

**Fused** (FlashAttention): only Q, K, V, O touch HBM in the ideal, `B = 2 * 4*N*d = 8.4MB`, `AI = N/2 = 4096` - compute-bound. Training-time attention at realistic `N` is compute-bound *if fused*, memory-bound if not.

**Decode attention** (one query row against a KV cache of length `L`): `F = 4*L*d`, `B ~ 2*L*d*2` (read K and V) so `AI ~ 1` per query head. With GQA, `g` query heads share each KV head, so `AI ~ g`. Still far below the ridge: decode attention is bound by reading the KV cache, which is why paged KV layout (storing the cache in fixed-size blocks, like OS memory pages, so a sequence need not be contiguous), KV quantization and MQA/GQA/MLA matter for serving and FLOP-saving attention tricks do not.

### Summary for an LLM step

| op                                  | prefill / training | decode (small batch) |
| :---------------------------------- | :----------------- | :------------------- |
| QKV / O / MLP projections           | compute-bound      | memory-bound (weights) |
| attention core (fused)              | compute-bound      | memory-bound (KV cache) |
| RMSNorm / LayerNorm, residual add   | memory-bound       | memory-bound (and launch-bound) |
| activation (SiLU/GELU), gating mul  | memory-bound       | memory-bound |
| RoPE                                | memory-bound       | memory-bound |
| softmax over vocab / cross-entropy  | memory-bound       | memory-bound |
| optimizer step (AdamW)              | memory-bound       | -                    |

AdamW is worth a line of arithmetic: per parameter it reads param, grad, `m`, `v` and writes param, `m`, `v`; with fp32 master weights and states that is `4*4 + 3*4 = 28` bytes (plus bf16 grad/param copies depending on the setup) for ~10-15 FLOPs. Multi-tensor-apply / fused optimizers win by reducing launch count, not FLOPs.


## GEMM tiling hierarchy

### Block, warp, instruction

A high-performance GEMM is the same loop nest at three levels, each level staging data in a faster memory to get reuse:

```
for each CTA tile (BM, BN) of C:                 # grid level: one CTA per tile (or persistent, below)
    acc = zeros(BM, BN)                          # (BM, BN) fp32, in registers (Hopper) or TMEM (Blackwell)
    for k0 in range(0, K, BK):                   # mainloop over K
        sA = load A[m0:m0+BM, k0:k0+BK]          # (BM, BK) global -> smem
        sB = load B[k0:k0+BK, n0:n0+BN]          # (BK, BN) global -> smem
        for each warp(group) sub-tile (WM, WN):  # warp level
            for kk in range(0, BK, MMA_K):       # instruction level
                acc[warp tile] += mma(sA[..., kk], sB[kk, ...])
    epilogue(acc) -> C[m0:m0+BM, n0:n0+BN]       # scale, bias, activation, cast, store
```

The CUTLASS GEMM hierarchy - global memory to shared-memory block tiles to register warp tiles to per-thread fragments:

![CUTLASS GEMM hierarchy](images/cutlass-gemm-hierarchy.png)

([source](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/))

Reuse per level: each element of `sA` is used `BN` times, each of `sB` `BM` times. The CTA's global-memory intensity is

```
AI_tile = 2*BM*BN*BK / (b * (BM*BK + BK*BN)) = 2*BM*BN / (b * (BM + BN))
128 x 256, bf16:  2*128*256 / (2*384) = 85 FLOP/B
```

85 is below H100's 295 ridge with respect to HBM - which is fine, because most of those reads hit L2 (neighbouring CTAs read the same `A` rows and `B` columns). This is why tile *rasterization* (the order CTAs are launched in, e.g. CUTLASS's swizzled tile schedulers or Triton's `GROUP_SIZE_M` trick) affects GEMM throughput: it controls L2 hit rate. Clusters with TMA multicast push the effective tile larger still.

### Resource arithmetic for a Hopper tile

`BM x BN x BK = 128 x 256 x 64`, bf16, 2 consumer warpgroups + 1 producer warpgroup (384 threads):

```
smem per stage  = (BM*BK + BN*BK) * 2B = (8192 + 16384) * 2 = 48KB
4 stages        = 192KB  (fits in 227KB, leaving room for the epilogue buffer)
accumulator     = 128*256 fp32 / 256 consumer threads = 128 registers/thread
register budget = 65536 / 384 = 170 per thread on average
```

128 accumulator registers plus addresses and fragments do not fit comfortably at 170 per thread for everyone - but the producer warpgroup only issues TMA and needs few registers. `setmaxnreg` (Hopper) lets warpgroups re-balance: e.g. producer at 40 and consumers at 232 gives `128*40 + 256*232 = 64512 <= 65536`. (The exact numbers CUTLASS and FA3 choose vary by kernel; these are illustrative.)

### Software pipelining

The mainloop above is serial as written: load, wait, compute. Pipelining keeps `S` stages of `sA`/`sB` in a ring buffer in shared memory so that while the tensor cores work on stage `s`, loads for stages `s+1 ... s+S-1` are in flight:

- **Ampere**: `cp.async` copies with `cp.async.commit_group` / `wait_group`, issued by the same warps that compute. Triton's `num_stages` controls this.
- **Hopper/Blackwell**: TMA loads completing on per-stage `mbarrier`s; consumers arrive on an "empty" barrier when done with a stage so the producer can refill it. CUTLASS calls this `PipelineTmaAsync`.

Overlapping loads for tile `k+1` with math on tile `k`:

![software pipelining in CUTLASS](images/cutlass-software-pipeline.png)

([source](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/))

Choose `S` so that `S - 1` stages of loads cover the memory latency at the rate the tensor cores consume stages, subject to the shared memory budget above. Too few stages leaves the tensor cores waiting on loads (Nsight Compute shows "long scoreboard" or barrier stalls); too many reduces CTAs per SM or blocks a larger tile.

### Split-K, stream-K and persistent kernels

Wave quantization is worst when there are few output tiles: a GEMM with `M = 64, N = 4096, K = 16384` has only `1 * 16` tiles of `64 x 256` - 16 CTAs on 132 SMs.

- **Split-K**: partition K into `s` chunks, give each to a different CTA, then reduce the `s` partial `C` tiles (through a global workspace and a second reduction kernel, or atomics, or a serialized "fixup" in the last CTA). Raises CTA count by `s` at the cost of `s` extra FP32 tile writes/reads. cuBLAS and Triton (`SPLIT_K`) pick it heuristically for small-`M*N` / large-`K` shapes - common after tensor-parallel sharding (splitting each weight matrix across GPUs, which shrinks N or K per GPU).
- **Stream-K** (Osama et al., "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU", arXiv:2301.03598): launch exactly as many CTAs as SMs and divide the *total* MAC-loop iterations `tiles * ceil(K/BK)` evenly among them, so a CTA may finish one tile's K range and start partway through another. Tiles split across CTAs are fixed up with partial-sum exchange. This removes the tail wave in general rather than per shape. CUTLASS ships stream-K tile schedulers.
- **Persistent kernels**: launch `#SMs` (times CTAs-per-SM) CTAs that loop over tiles from a tile scheduler instead of one CTA per tile. This amortizes prologue (barrier init, TMA descriptor prefetch) over many tiles and, more importantly, lets the epilogue of tile `i` overlap the mainloop of tile `i+1`. On Hopper CUTLASS offers two schedules:
  - *cooperative*: both consumer warpgroups work on the same tile (split along M), epilogue not overlapped
  - *ping-pong*: the two consumer warpgroups work on different tiles and alternate, so one runs its epilogue while the other runs MMAs

  On Blackwell the TMEM double-buffered accumulator plays the role ping-pong plays on Hopper.

### Epilogue fusion

The epilogue is the only point where the output tile is in registers before it hits HBM, so anything elementwise on `C` should happen there: scaling (`alpha`, FP8 dequant scales), bias, activation, residual add, dtype cast, computing `amax` for the next FP8 scale, even writing an auxiliary output (the pre-activation, for backward). Each op fused saves a full `M*N` round trip. Mechanisms:

- cuBLASLt epilogues (`CUBLASLT_EPILOGUE_BIAS`, `_GELU_BIAS`, `_GELU_AUX`, ...) - a fixed menu
- CUTLASS 3.x Epilogue Visitor Trees (EVT) - arbitrary DAGs of elementwise/broadcast/reduction nodes compiled into the epilogue
- Triton - write the epilogue yourself after the K loop; Inductor's `max-autotune` does this with its Triton GEMM templates

What does *not* fuse into a GEMM epilogue: anything needing a full row of `C` when the tile only has `BN` columns of it - e.g. LayerNorm or softmax over N. Those need a separate pass, a row-tile of the full width, or a cross-CTA reduction.

### CUTLASS and CuTe in one screen

CUTLASS 3.x is built on **CuTe**, whose core abstraction is a `Layout = (Shape, Stride)`: a function from logical coordinates to offsets, composable and hierarchical. `((8, 16), 4) : ((1, 32), 8)` is a valid layout; swizzles are layouts; a thread-value partition is a layout. On top of layouts:

- **Atoms**: `MMA_Atom` (one tensor core instruction with its thread/value layout, e.g. `SM90_64x256x16_F32BF16BF16_SS`) and `Copy_Atom` (one copy instruction: `SM90_TMA_LOAD`, `SM75_U32x4_LDSM_N`, ...)
- **TiledMMA / TiledCopy**: atoms tiled over threads and values to cover a CTA tile
- `local_tile` / `local_partition` / `partition_fragment_C`: slicing global/shared/register tensors for this CTA/thread

The GEMM itself is assembled from a `CollectiveMainloop` (pipeline + MMA), `CollectiveEpilogue` and a `TileScheduler` (persistent, stream-K) inside `GemmUniversal`. CUTLASS 4 adds the CuTe DSL, a Python front end over the same abstractions. You rarely need to write CUTLASS for standard GEMMs - cuBLASLt covers them - but it is the tool for fused GEMMs (grouped/MoE GEMM, GEMM + custom epilogue, block-scaled FP8/FP4) and it is what FA3 is written in.


## FlashAttention

### Online softmax

For one query block `Q_i` (`Br` rows) iterate over key/value blocks `K_j, V_j` (`Bc` rows each), keeping per-row running max `m`, running denominator `l` and unnormalized output `O`. Scale `1/sqrt(d)` folded into `S`:

```
init:   m = -inf (Br,),  l = 0 (Br,),  O = 0 (Br, d)

for j in blocks:
    S     = Q_i @ K_j^T * scale              # (Br, Bc)
    m_new = max(m, rowmax(S))                # (Br,)
    P     = exp(S - m_new[:, None])          # (Br, Bc)   unnormalized probabilities
    alpha = exp(m - m_new)                   # (Br,)      rescale factor for old state
    l     = alpha * l + rowsum(P)            # (Br,)
    O     = alpha[:, None] * O + P @ V_j     # (Br, d)
    m     = m_new

finalize:
    O   = O / l[:, None]                     # (Br, d)
    LSE = m + log(l)                         # (Br,)  saved for backward
```

The online-softmax forward pass with K/V split into two blocks:

![FlashAttention forward with K split into two blocks](images/flash-attention-forward-diagram.png)

([source](https://arxiv.org/abs/2307.08691))

Correctness: after processing blocks `1..j`, `l = sum_k exp(s_k - m)` and `O = sum_k exp(s_k - m) v_k` over all keys seen so far, with `m` their max. Replacing `m` by `m_new` multiplies every existing term by `exp(m - m_new)`, which is exactly `alpha`. The final division gives `sum_k softmax(s)_k v_k` exactly - this is not an approximation. (In practice kernels use `exp2` with `scale * log2(e)` folded in, since `ex2` is the native MUFU op.)

The equations above are the FA2 form. FA1 kept `O` normalized at each step (`O = (l_old * alpha * O + P @ V) / l_new`), costing a division per element per block; FA2 defers it to the end.

### IO complexity

The FlashAttention paper (Dao et al., arXiv:2205.14135) counts HBM accesses with an SRAM of size `M` elements, `d <= M <= N*d`:

- standard attention: `Theta(N*d + N^2)`
- FlashAttention: `Theta(N^2 * d^2 / M)`

The argument (FA1 loop order: K/V blocks outer, Q blocks inner): choose `Bc = ceil(M / 4d)` so a K/V block plus a Q/O block plus `S` fit in SRAM. There are `Tc = N / Bc` outer iterations, and each makes one pass over all of `Q` and `O` (`Theta(N*d)` each). Total:

```
Tc * N * d = (N / Bc) * N * d = N^2 * d * 4d / M = Theta(N^2 d^2 / M)
```

Since `d^2 < M` in practice, this beats `N^2`. With `d = 128` and `M = 228KB / 2B = 116,736` bf16 elements, `d^2 / M = 0.14`. The paper also shows (Proposition 3) that no exact attention algorithm can achieve `o(N^2 d^2 / M)` for all `M` in `[d, N*d]`.

Two caveats the asymptotics hide, worth knowing when you reason about real kernels:

- The constants are not small. In the FA2 loop order (Q outer), each Q block loads all of K and V once: `2 * N * d * (N / Br)` elements. At `N = 8192, d = 128, Br = 128` that is 268MB per head - comparable to the naive 545MB. The kernel is still fast because those re-reads mostly hit **L2**, not HBM: one head's K and V are `2 * 8192 * 128 * 2B = 4.2MB`, and the CTAs concurrently working on a head share them. The model counts HBM-SRAM traffic without an L2.
- What the model *does* capture is the thing that matters most: the `N^2` matrices `S` and `P` never exist in HBM, so the memory footprint is `O(N)` and the kernel's AI becomes `~N/2` instead of `~63`.

### FA2 partitioning changes

FlashAttention-2 (Dao, arXiv:2307.08691) changes three things relative to FA1:

1. **Fewer non-matmul FLOPs**: deferred normalization (above), and only `LSE` is saved instead of both `m` and `l`. On A100 the paper notes non-matmul FLOPs are up to 16x more expensive than matmul FLOPs (312 TFLOPS tensor vs 19.5 TFLOPS FP32), so these matter.
2. **Parallelize over the sequence dimension**: the grid is `(N / Br, batch * heads)` rather than `batch * heads`, so long sequences with small batch still fill the SMs. This required swapping the loop order to Q-blocks outer.
3. **Work partitioning between warps**: FA1 split K/V across the 4 warps of a CTA ("split-K"), so each warp held partial `S @ V` products that had to be combined through shared memory. FA2 splits Q across warps, each warp seeing all of K/V for its query rows, so the per-warp results are independent and no inter-warp reduction is needed in the forward.

FA2 splits Q across warps instead of K/V ("split-Q"), removing the shared-memory reduction between warps:

![FlashAttention-2 warp partitioning](images/flash-attention-2-partitioning.png)

([source](https://arxiv.org/abs/2307.08691))

The paper reports up to 73% of theoretical max FLOPs on A100 for the forward pass; I am quoting, not measuring.

### FlashAttention-3 on Hopper

FA2 on H100 reaches only ~35% utilization according to the FA3 paper (Shah et al., arXiv:2407.08608), because it uses `mma.sync` and synchronous loads and so cannot use `wgmma` and TMA. FA3's changes:

- **Warp specialization**: within a CTA, a *producer* warpgroup issues TMA loads of K/V into a shared-memory ring buffer; *consumer* warpgroups issue `wgmma` and do softmax. `setmaxnreg` moves registers from the producer to the consumers. The producer/consumer handshake is `mbarrier`-based, as in a CUTLASS Hopper GEMM.
- **Ping-pong scheduling between warpgroups**: with 2 consumer warpgroups, named barriers force warpgroup 1's GEMMs to run while warpgroup 2 does its softmax, and vice versa. The motivation is the MUFU arithmetic above: per the paper, for `d = 128` there are 512x more matmul FLOPs than exponentials but exponentials have 256x lower throughput, so the `exp` can take ~50% of the cycles of the matmuls. Overlapping hides it.
- **Intra-warpgroup pipelining**: within one warpgroup, issue `S_{j+1} = Q K_{j+1}^T` asynchronously (`wgmma` is async), then compute softmax of `S_j` while it runs, then `P_j V_j`. Costs extra registers for the second `S` tile.
- **FP8**: `wgmma` FP8 requires K-major operands, so V (which is MN-major for `P @ V`) is transposed in shared memory inside the kernel; the register layout of the FP32 accumulator of the first GEMM does not match the operand layout FP8 `wgmma` needs for `P`, so a register permutation is done. To limit quantization error: **block quantization** (one scale per block rather than per tensor) and **incoherent processing** (multiply Q and K by a random orthogonal - Hadamard with random signs - matrix, which leaves `Q K^T` unchanged but spreads outliers).

FA3's ping-pong schedule - one warpgroup's softmax overlaps the other's GEMMs:

![FA3 ping-pong scheduling](images/flash-attention-3-pingpong.png)

([source](https://arxiv.org/abs/2407.08608))

Paper-reported results: 1.5-2.0x over FA2 in FP16, up to 740 TFLOPS (75% of 989), and close to 1.2 PFLOPS in FP8, with 2.6x lower numerical error than baseline per-tensor FP8 attention. On Blackwell the equivalent role is played by `tcgen05`-based kernels (cuDNN's, and FlashAttention's own Blackwell path); the FA3 binary does not run there (`sm_90a`).

### Backward and recomputation

The backward needs `P` but it was never stored. It is recomputed from `Q`, `K` and the saved `LSE`, block by block:

```
D   = rowsum(dO * O)                         # (N,)  precomputed once, cheap
for each (i, j) block pair:
    S   = Q_i @ K_j^T * scale                # (Br, Bc)
    P   = exp(S - LSE_i[:, None])            # (Br, Bc)  exact, already normalized
    dV_j += P^T @ dO_i                       # (Bc, d)
    dP  = dO_i @ V_j^T                       # (Br, Bc)
    dS  = P * (dP - D_i[:, None])            # (Br, Bc)  softmax Jacobian-vector product
    dQ_i += dS @ K_j * scale                 # (Br, d)
    dK_j += dS^T @ Q_i * scale               # (Bc, d)
```

`D_i = rowsum(dO_i * O_i)` equals `rowsum(P * dP)` and is what makes the softmax backward local to a block. The backward does 5 matmuls per block (`S`, `dV`, `dP`, `dQ`, `dK`) versus 2 in the forward, so it is ~2.5x the forward FLOPs, and recomputing `S` is one of those five - recomputation costs ~20% more backward FLOPs but saves `O(N^2)` memory and the HBM traffic of reading `P`, which is a net speed win. FA2's backward parallelizes over K/V blocks (each CTA owns `dK_j, dV_j` and loops over Q blocks), so `dQ` contributions from different CTAs are accumulated with atomic adds in FP32.


## torch.compile internals

The [torch.compile section](../../training/performance/README.md#torchcompile) of the performance chapter covers when it pays off, with measurements. This is what happens under the hood, so you can reason about the failure modes.

### Dynamo: capture

TorchDynamo hooks CPython's frame evaluation (PEP 523, the interpreter API that lets a library intercept each Python function call before it runs). When a compiled function is called, Dynamo symbolically executes its *bytecode* (the low-level instruction list CPython compiles each function into), recording tensor operations into an FX graph and keeping everything else (Python ints, lists, attribute lookups, module structure) as compile-time constants or guarded values.

- **Graph breaks**: when Dynamo hits something it cannot trace - a call into an unsupported C extension, `print`, `.item()` / `.tolist()` used for control flow, data-dependent Python branching on tensor values, some generators - it compiles the graph so far, emits bytecode that runs the offending code in plain Python, and resumes tracing after it. Each break is a fusion boundary and a graph launch boundary. `fullgraph=True` makes breaks errors; `torch._dynamo.explain(fn)(*args)` summarizes them; `TORCH_LOGS="graph_breaks"` prints each one with its source line.
- **Guards**: every assumption baked into the graph becomes a guard checked on each call - tensor dtype, device, rank, sizes and strides (or symbolic constraints on them), `requires_grad`, identity/value of globals and module attributes used, Python scalars that were specialized. Guard evaluation itself costs CPU time per call (a small, but non-zero, term in launch-bound regimes).
- **Recompiles**: a guard failure triggers a new trace for the new specialization, cached alongside the old one, up to `torch._dynamo.config.recompile_limit` entries. `TORCH_LOGS="recompiles"` tells you which guard failed. The usual offenders: a changing sequence length, a Python float hyperparameter that changes (e.g. a scheduled scale passed as `float` rather than a tensor), toggling `model.train()`/`eval()`.
- **Dynamic shapes**: the first compile specializes on exact sizes. When a size changes and causes a recompile, Dynamo by default ("automatic dynamic") recompiles with that dimension symbolic, so the next graph is generic in it. Sizes 0 and 1 are always specialized. `torch._dynamo.mark_dynamic(x, dim)` opts in up front. Symbolic shapes cost some kernel quality (no constant folding of sizes, more masking) - for a handful of fixed buckets, static specialization per bucket is often faster.

### AOTAutograd: joint graph

For training, AOTAutograd traces the forward graph through the autograd engine via `__torch_dispatch__` to produce a single *joint* forward+backward graph of ATen ops, then:

- **functionalizes** it (in-place ops and views rewritten as pure functions plus copy-backs), so the compiler can reorder and fuse freely
- **decomposes** composite ops into a smaller core ATen / prims op set
- **partitions** the joint graph into forward and backward with a min-cut *rematerialization* partitioner: it decides which intermediates to save for backward and which to recompute, and prefers recomputing cheap pointwise ops (whose recompute is fused into backward kernels for free) over saving their outputs. This is automatic, local activation checkpointing; it is one reason compiled models often use less activation memory.

### Inductor: codegen

Inductor lowers the ATen graph to its own loop-level IR, then a scheduler groups nodes into kernels:

- **pointwise and reduction fusion**: chains of elementwise ops, and elementwise ops feeding/following a reduction, become one Triton kernel. RMSNorm becomes one kernel, the RoPE + reshape becomes one, a residual add + norm becomes one. This is the memory-bound win from the [roofline section](#normalization-softmax-and-elementwise).
- **matmuls and convolutions** go by default to `extern` calls (cuBLAS / cuDNN), and the surrounding pointwise ops become separate Triton kernels before/after.
- **`mode="max-autotune"`**: benchmarks Triton GEMM templates (and optionally the CUTLASS backend) against cuBLAS at compile time, per shape, and can fuse pointwise epilogues into the chosen template. It also enables CUDA graphs. Compile time is much longer (see the cold-start numbers in the performance chapter). `max-autotune-no-cudagraphs` exists for when graphs are the problem.
- Triton kernels are themselves autotuned over a small set of block sizes / `num_warps` per kernel; `torch._inductor.config.coordinate_descent_tuning = True` searches further.

To see what you got: `TORCH_LOGS="output_code"` prints the generated Python/Triton; `TORCH_COMPILE_DEBUG=1` dumps the IR and code per graph; `TORCH_TRACE=/tmp/trace` plus `tlparse` gives a browsable report of every graph, guard set, break and recompile. Reading the generated code is the fastest way to answer "did it fuse X into Y?".

### reduce-overhead and CUDA graphs

`mode="reduce-overhead"` wraps each compiled graph in CUDA graphs ("CUDA graph trees" in Inductor): after a warmup call the kernel sequence is captured once and replayed with a single launch per graph. Constraints and gotchas that follow from how CUDA graphs work ([below](#cuda-graphs)):

- inputs are copied into static buffers unless they are already static (parameters, buffers)
- outputs live in a graph-owned memory pool and are **overwritten by the next replay**. Holding a reference across iterations and reading it later gives an error ("accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"); clone it, or call `torch.compiler.cudagraph_mark_step_begin()` at the start of each iteration in non-standard loops
- dynamic shapes mean one graph per distinct shape, each with its own memory; graphs are skipped for graphs with CPU tensors, some mutations, or ops that synchronize


## Writing a Triton kernel: fused RMSNorm

RMSNorm is memory-bound with `AI ~ 1`, so the whole job is to read `x` once and write `y` once at as close to peak bandwidth as possible. Eager PyTorch with the standard fp32 upcast launches a chain of kernels (cast, `pow`, `mean`, `add`+`rsqrt`, `mul`, `mul`, cast), each of which round-trips an `(M, N)` tensor. Rough per-element byte count for that chain (bf16 in, fp32 intermediates):

```
x.float()             read 2, write 4
.pow(2)               read 4, write 4
.mean(-1)             read 4
xf * rstd             read 4, write 4
* w                   read 4, write 4
.to(bf16)             read 4, write 2
total                 ~40 bytes/element   vs   4 bytes/element fused
```

So the fused kernel's floor is ~10x below eager's for large tensors (an arithmetic expectation, not a measurement; the real ratio depends on the shapes and on any fusion eager does internally, and `torch.nn.functional.rms_norm` may already dispatch to a fused kernel in your PyTorch version).

### The kernel

One program per row: a row of up to ~16K bf16 elements fits in registers across the CTA's threads.

```python
# rmsnorm_triton.py
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X, W, Y, RSTD,                 # pointers
    stride_x, stride_y,            # row strides (elements)
    N,                             # row length
    eps,
    BLOCK_N: tl.constexpr,         # power of 2 >= N
):
    row = tl.program_id(0)                          # one program per row
    cols = tl.arange(0, BLOCK_N)                    # (BLOCK_N,)
    mask = cols < N                                 # (BLOCK_N,)  tail of the row

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)  # (BLOCK_N,)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)                   # (BLOCK_N,)

    ms = tl.sum(x * x, axis=0) / N                  # ()  mean of squares, fp32; masked lanes are 0
    rstd = 1.0 / tl.sqrt(ms + eps)                  # ()
    y = x * rstd * w                                # (BLOCK_N,)

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)                      # (1,) per row, for a backward pass


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape                                 # (..., N)
    x2 = x.reshape(-1, shape[-1])                   # (M, N)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M, N = x2.shape
    y = torch.empty_like(x2)                        # (M, N)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)  # (M,)

    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N * x.element_size() > 64 * 1024:
        raise ValueError(f"row of {N} elements too wide for the one-row-per-program kernel")
    # ponytail: fixed num_warps heuristic; use @triton.autotune over num_warps if a shape matters
    num_warps = 4 if BLOCK_N <= 2048 else 8 if BLOCK_N <= 8192 else 16

    rmsnorm_fwd_kernel[(M,)](
        x2, w, y, rstd,
        x2.stride(0), y.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N, num_warps=num_warps,
    )
    return y.view(shape)


def rmsnorm_eager(x, w, eps=1e-6):
    xf = x.float()                                                  # (M, N) fp32
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 8192, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)     # (M, N)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)        # (N,)

    # correctness first: bf16 output, compare with an fp32-internal reference
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_eager(x, w), atol=2e-2, rtol=2e-2)
    # non-power-of-2 width exercises the mask
    x_odd = torch.randn(64, 4000, device="cuda", dtype=torch.bfloat16)
    w_odd = torch.randn(4000, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(rmsnorm(x_odd, w_odd), rmsnorm_eager(x_odd, w_odd), atol=2e-2, rtol=2e-2)

    compiled = torch.compile(rmsnorm_eager)
    candidates = {
        "eager": rmsnorm_eager,
        "F.rms_norm": lambda x, w: torch.nn.functional.rms_norm(x, (x.shape[-1],), w, 1e-6),
        "torch.compile": compiled,
        "triton": rmsnorm,
    }

    peak = torch.cuda.get_device_properties(0)
    peak_gbps = peak.memory_clock_rate * 1e3 * (peak.memory_bus_width / 8) * 2 / 1e9

    print(f"{peak.name}  theoretical {peak_gbps:.0f} GB/s")
    for N in [1024, 2048, 4096, 8192, 16384]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        bytes_moved = 2 * x.numel() * x.element_size() + w.numel() * w.element_size()  # read x, write y, read w
        row = []
        for name, fn in candidates.items():
            ms = triton.testing.do_bench(lambda: fn(x, w), quantiles=[0.5])          # median, ms
            row.append(f"{name}={ms*1e3:7.1f}us ({bytes_moved / (ms * 1e-3) / 1e9:6.0f} GB/s)")
        print(f"N={N:5d}  " + "  ".join(row))
```

This code was written for this chapter and not run while writing it (no GPU was available); treat it as untested until the two `assert_close` calls pass on your machine.

### Choices, and why

- **Block size = whole row** (`next_power_of_2(N)`): the reduction needs the whole row, and keeping it in registers means one read of `x`. For `N = 4096` with 8 warps that is `4096 / 256 = 16` fp32 values of `x` plus 16 of `w` per thread - comfortable. At `N = 16384` with 16 warps it is 32 + 32, still fine; much beyond that, register pressure forces spills and you switch to a two-pass loop over column chunks (reduce pass, then normalize pass), paying a second read of `x` that will usually hit L2.
- **`num_warps`**: enough threads that each one issues a few 16-byte vector loads (8 bf16). With 8 warps and `N = 4096`, each thread loads `4096 / 256 = 16` elements = two 16B loads. Too few warps per program and each program has too little memory parallelism; too many and the reduction (`tl.sum` across warps goes through shared memory) dominates for short rows. Autotune it if the shape matters.
- **Grid = M programs**: `M = 8192` rows on 132 SMs, many programs resident per SM, so per-row latency is hidden across rows. For tiny `M` (decode, `M = batch`) one row per program leaves most SMs idle - but then the op is launch-bound anyway ([next section](#launch-overhead-and-cpu-boundness)).
- **fp32 accumulation** for the mean of squares - bf16 has 8 bits of mantissa and summing thousands of squares in it loses precision.
- **Saving `rstd`** costs `4M` bytes (negligible) and makes the backward a single fused pass.

### Benchmarking notes

- `triton.testing.do_bench` runs warmup, flushes L2 between repetitions by writing a large buffer, synchronizes with CUDA events, and returns the requested quantiles of the timings. Flushing matters: at `M = 8192, N = 1024` the input is 16MB and fits in H100's 50MB L2, so without a flush you would measure L2 bandwidth.
- Report GB/s against the theoretical peak from the device properties (computed as in [How to calculate theoretical memory bandwidth](README.md#how-to-calculate-theoretical-memory-bandwidth)). A good memory-bound kernel on H100 lands at a large fraction of it; NVIDIA does not publish an achievable figure, so measure a plain `y.copy_(x)` with the same harness as the practical ceiling.
- Expect `torch.compile` to be close to the hand-written kernel here: this is exactly the pattern Inductor fuses well. The reason to write Triton yourself is when the fusion crosses something Inductor will not fuse - a GEMM mainloop, a custom backward that recomputes rather than saves, a non-standard reduction layout, or a cross-entropy over a 128K vocab that should never materialize the logits (Liger-Kernel's fused linear cross-entropy is a good public example).


## Launch overhead and CPU-boundness

### What it costs

Every eager PyTorch op runs Python, the dispatcher (PyTorch's C++ layer that picks the implementation for a given dtype/device), the ATen kernel's host-side setup, and a `cudaLaunchKernel` call before the GPU sees anything. These costs are host-CPU dependent and not published as specs; as orders of magnitude (my estimates, measure yours):

- `cudaLaunchKernel` host call: a few us
- full eager op from Python, including dispatch: roughly 5-20us
- a kernel launch replayed from a CUDA graph: on the order of 1us or less per kernel of GPU-side overhead

The GPU runs asynchronously, so overhead only matters when the GPU executes kernels faster than the CPU can enqueue them. Illustrative arithmetic: a decode step of a 1B-parameter bf16 model reads 2GB of weights, a floor of `2e9 / 3.35e12 = 0.6ms` on H100. If that model has 16 layers with ~15 kernels each (norm, QKV, RoPE, attention, O-proj, residual, norm, gate/up, act, mul, down, residual, ...) that is ~240 launches; at an assumed 10us each the CPU needs 2.4ms. The step is CPU-bound by ~4x and a faster kernel would change nothing.

Measure your own launch rate with a trivially small kernel:

```python
import time, torch
x = torch.zeros(1, device="cuda")
for _ in range(100): x.add_(1)                  # warmup
torch.cuda.synchronize()
n = 10000
t0 = time.perf_counter()
for _ in range(n): x.add_(1)                    # each is one tiny kernel
torch.cuda.synchronize()
print(f"{(time.perf_counter() - t0) / n * 1e6:.1f} us per eager op")
```

### Spotting it in a trace

In an Nsight Systems or `torch.profiler` timeline:

- **GPU idle gaps between kernels**, with the CUDA HW row showing short kernels separated by whitespace. If kernels are 5-20us and the gaps are similar, you are launch-bound.
- **The CPU thread is always busy** in the CUDA API row (`cudaLaunchKernel` back to back) while the GPU is not; the launch of kernel `k+1` ends just before kernel `k+1` starts on the GPU (the queue is empty - the GPU is waiting on the CPU). In a healthy GPU-bound trace the CPU runs far ahead and then blocks in a synchronize.
- **Synchronization points** (a CUDA stream is a queue of GPU work executed in order; synchronizing makes the CPU wait until a stream's queue is empty): `cudaStreamSynchronize` / `cudaMemcpy DtoH` from `.item()`, `.cpu()`, `print(tensor)`, `if tensor:`, `torch.nonzero`, boolean-mask indexing. Each one drains the queue and exposes the full launch latency of whatever follows.

Remedies in order of effort: remove syncs; fuse (fewer kernels - `torch.compile`, fused optimizers via `foreach`/`fused=True`); larger batches (more work per launch); CUDA graphs.

### CUDA graphs

A CUDA graph records a sequence of kernel launches (with their arguments, i.e. fixed pointers and fixed shapes) once, then replays the whole sequence with one host call. The constraints follow from "fixed pointers and shapes":

- static shapes - one graph per shape bucket (this is what vLLM and SGLang do for decode batch sizes)
- inputs must be copied into the captured input buffers; outputs are in graph-owned memory
- no CPU synchronization or host-side, data-dependent control flow inside the captured region
- memory allocated during capture comes from a private pool held for the graph's lifetime

Manual use, when you are not going through `torch.compile(mode="reduce-overhead")`:

```python
static_x = torch.randn(B, T, C, device="cuda")              # (B, T, C) capture-time input buffer
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):                                  # warmup on a side stream (lazy init, autotune)
    for _ in range(3):
        static_y = model(static_x)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y = model(static_x)                              # (B, T, V) output lives in the graph pool

for batch in loader:
    static_x.copy_(batch)                                   # write into the captured buffer
    g.replay()
    use(static_y.clone())                                   # clone if it must survive the next replay
```

See the [PyTorch CUDA graphs docs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) for capturing training steps including backward and optimizer.


## Profiling: nsys and ncu

Two different tools, for two different questions. Start with the timeline; only go to the kernel profiler once you know which kernel matters. For `torch.profiler` usage see [Profilers](../../debug/pytorch.md#profilers).

### Nsight Systems: where does the time go

```bash
nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-graph-trace=node \
    -o step_profile --force-overwrite true python train.py
nsys stats --report cuda_gpu_kern_sum step_profile.nsys-rep     # kernel time summary
```

Wrap regions with NVTX (NVIDIA's API for named ranges that show up on the timeline; `torch.cuda.nvtx.range_push("fwd")` / `range_pop()`, or `torch.autograd.profiler.emit_nvtx()`) so the timeline is readable, and limit the capture to a few steady-state steps (`--capture-range=cudaProfilerApi` with `torch.cuda.profiler.start()/stop()`), since the first steps include compilation and allocator warmup.

What to read off it:

| what you see                                                     | what it implies                                                   |
| :--------------------------------------------------------------- | :---------------------------------------------------------------- |
| GPU row dense, CPU blocked in sync                               | GPU-bound: go to kernel-level analysis                            |
| short kernels with gaps, CPU saturated with launches             | launch / CPU-bound: fuse, batch, CUDA graphs                      |
| gaps aligned with `cudaMemcpy`/`cudaStreamSynchronize`           | host syncs; find the `.item()`                                    |
| gaps aligned with CPU work in `osrt` / Python                    | dataloader or Python overhead                                     |
| NCCL (collective communication, e.g. all-reduce = sum a tensor across GPUs) kernels serialized with compute          | communication not overlapped                                      |
| one kernel dominating `cuda_gpu_kern_sum`                        | the candidate for `ncu`                                           |

### Nsight Compute: why is this kernel slow

```bash
ncu --set full -k regex:rmsnorm --launch-skip 10 --launch-count 1 \
    -o rmsnorm_report python rmsnorm_triton.py
```

`ncu` replays each profiled kernel many times to collect all counters, so profile one or a few launches (`-k`, `--launch-skip`, `--launch-count`, or `--nvtx --nvtx-include`). By default it locks clocks to base (`--clock-control base`), so its durations are not directly comparable to an unprofiled run; use `--clock-control none` when you care about absolute time.

The sections and how to read them:

- **GPU Speed Of Light Throughput**: `Compute (SM) Throughput %` and `Memory Throughput %`, each the *maximum* over the sub-units in that category (e.g. memory = max of DRAM, L2, L1, shared memory). The first decision:
  - memory SOL high (say >60-80%), compute low: memory-bound. Check which unit is the high one in Memory Workload Analysis - DRAM means fuse or reduce bytes; L1/shared means bank conflicts or poor access patterns.
  - compute SOL high: compute-bound. Check which pipe (tensor, FMA, ALU, XU/MUFU) in Compute Workload Analysis; if it is not the tensor pipe for a GEMM-like kernel, something is wrong.
  - **both low**: latency-bound - neither unit is busy because warps are stalled. This is the most common state of a bad kernel, and the next sections tell you why.
- **Memory Workload Analysis**: DRAM throughput (`dram__throughput.avg.pct_of_peak_sustained_elapsed`), L1/L2 hit rates, sectors per request (should be 4 for 32-bit coalesced warp loads; higher means uncoalesced), shared memory bank conflicts.
- **Occupancy**: theoretical vs **achieved** occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`) and which limit binds - the [worked calculation](#occupancy-worked) done for you. Achieved much lower than theoretical means tail effects or load imbalance (wave quantization shows up here).
- **Launch Statistics**: grid/block size, registers per thread, shared memory per block, and "waves per SM" - fractional waves are wave quantization.
- **Warp State Statistics**: average cycles between issued instructions per warp, broken down by stall reason. The main ones:

| stall reason          | meaning                                                         | typical fix                                                       |
| :-------------------- | :-------------------------------------------------------------- | :---------------------------------------------------------------- |
| Long Scoreboard       | waiting on an L1TEX (global/local) memory result                | more memory parallelism (occupancy, ILP, async copies); fewer bytes; check for register spills (local memory) |
| Short Scoreboard      | waiting on shared memory or MUFU result                         | bank conflicts; fewer transcendental ops                          |
| MIO Throttle          | shared memory / special-op instruction queue full               | fewer, wider shared memory accesses; reduce bank conflicts        |
| LG Throttle           | local/global memory instruction queue full                      | vectorize (16B per thread), fewer memory instructions             |
| Barrier               | waiting at `__syncthreads`                                      | imbalance between warps; smaller CTAs; async barriers             |
| Math Pipe Throttle    | the execution pipe is busy                                      | good if it is the tensor pipe - you are compute-bound             |
| Wait                  | fixed-latency dependency                                        | more ILP                                                          |
| Not Selected / Selected | warp was ready but another was issued / was issued            | not stalls - plenty of eligible warps                             |
| Membar / Drain        | memory fence / draining stores at exit                          | fewer fences; usually minor                                       |

- **Source Counters**: stalls attributed to SASS (and source) lines, so you can see which load or `exp` is responsible. Compile with `-lineinfo` (Triton kernels carry line info by default) to get source correlation.

For tensor core kernels look at the tensor pipe utilization in Compute Workload Analysis (metric names differ by architecture, e.g. `sm__pipe_tensor_op_hmma_cycles_active` on Ampere; Hopper `wgmma` uses a separate `gmma` pipe counter - check `ncu --query-metrics` on your device rather than trusting a name from this page).


## AMD differences

The same concepts apply to AMD Instinct (CDNA) GPUs with different names and numbers. For tuning specifics see [AMD GPUs Performance](amd/performance.md).

| concept               | NVIDIA (H100)                         | AMD (MI300X, CDNA3)                                       |
| :-------------------- | :------------------------------------ | :-------------------------------------------------------- |
| compute unit          | SM, 132                               | CU, 304 (8 XCDs x 38)                                     |
| SIMT group            | warp, 32 threads                      | wavefront, 64 threads                                     |
| matrix unit           | tensor core, 4/SM                     | Matrix Core, 4/CU (1216 total, per the [TFLOPS table](README.md#how-to-calculate-theoretical-tflops)) |
| matrix instruction    | `mma.sync` / `wgmma`                  | MFMA (`v_mfma_*`), issued by a wavefront                  |
| shared memory         | smem, up to 228KB/SM                  | LDS, 64KB/CU                                              |
| last-level cache      | 50MB L2                               | 4MB L2 per XCD + 256MB Infinity Cache                     |
| HBM                   | 3.35TBps                              | 5.3TBps                                                   |
| bulk async copy       | TMA                                   | no direct TMA equivalent on CDNA3 (to my knowledge)       |

Sources: [AMD CDNA3 ISA and MI300 documentation](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html) and the [MI300X specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html); cache numbers as in the [caches table](README.md#caches).

What changes in practice:

- **Wavefront 64.** In Triton on ROCm `num_warps` counts 64-wide wavefronts, so the same `num_warps` means twice the threads; block sizes tuned for NVIDIA are often wrong. Cross-lane reductions span 64 lanes.
- **Small LDS.** 64KB per CU on MI300X versus 228KB per SM on H100 means smaller GEMM / attention tiles or fewer pipeline stages. CDNA4 (MI350/MI355X) increases LDS to 160KB per CU per AMD's CDNA4 materials (stated with moderate confidence; check the MI355X documentation).
- **Large register file, organized differently.** Each SIMD has its own VGPR file and CDNA3 lets a wavefront use up to 512 VGPRs, unified with the accumulation registers (AGPRs) that MFMA writes (from my reading of the CDNA3 ISA guide; verify). Occupancy is counted in waves per SIMD; the ROCm Triton knob `waves_per_eu` trades it against registers.
- **Chiplets.** MI300X is built from chiplets (several smaller dies in one package); its 8 XCDs each have a private 4MB L2; workgroups are dispatched round-robin across XCDs, so consecutive workgroup IDs land on different L2s. GEMM and attention kernels on MI300X remap workgroup IDs so that tiles sharing operands land on the same XCD - the analogue of tile rasterization, with a much larger penalty for getting it wrong.
- **Ridge point** is `1307 / 5.3 = 247` FLOP/B for bf16 - the same classification of ops as on H100.
- **Tools**: `rocprofv3` (timeline, the nsys analogue) and `rocprof-compute` (formerly Omniperf; the ncu analogue with roofline and per-unit utilization). Libraries: hipBLASLt (cuBLASLt analogue), Composable Kernel (CUTLASS analogue), and AITER / CK-based FlashAttention.


## Further reading

NVIDIA documentation:
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) - execution model, occupancy, clusters, per-architecture limits
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) - coalescing, shared memory, occupancy
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) - `mma`, `wgmma`, `tcgen05`, `cp.async.bulk.tensor`, `mbarrier`
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [NVIDIA H100 Tensor Core GPU Architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
- [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) - tile and wave quantization, alignment
- [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - SOL, stall reasons, metric definitions
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass) and its CuTe documentation (`media/docs/cute/`)

Papers:
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Milakov and Gimelshein, "Online normalizer calculation for softmax", [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) (id unverified)
- Osama et al., "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU", [arXiv:2301.03598](https://arxiv.org/abs/2301.03598) (id unverified)
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 2009
- Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", [arXiv:2402.13499](https://arxiv.org/abs/2402.13499) (id unverified)
- Anthony et al., "The Case for Co-Designing Model Architectures with Hardware", [arXiv:2401.14489](https://arxiv.org/abs/2401.14489)
- Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation", ASPLOS 2024

PyTorch and Triton:
- [torch.compile documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html) and the [troubleshooting guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) - especially fused softmax, matmul, and layer norm
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) - production Triton kernels for RMSNorm, RoPE, SwiGLU, fused linear cross-entropy

AMD:
- [AMD Instinct MI300 architecture (ROCm docs)](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
- [ROCm Triton kernel optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel.html)

Other:
- [Modal GPU Glossary](https://modal.com/gpu-glossary) - concise definitions of most terms in this chapter
