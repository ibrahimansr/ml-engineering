# Parallelism From First Principles: Derivations and Cost Models

## Prerequisites

This chapter assumes you know the transformer math. It does not assume you know systems. If any of the following is new, read it first:

- [Computer architecture](../../foundations/computer-architecture.md) - the GPU memory hierarchy (HBM vs SRAM) and why FLOP/s and bytes/s are separate limits. The roofline argument used throughout this chapter depends on it.
- [Concurrency](../../foundations/concurrency.md) - streams, asynchronous launches and overlap. "Overlap compute with communication" is a concurrency statement, and every cost model below depends on whether the overlap actually happens.
- [Networking](../../foundations/networking.md) - latency vs bandwidth, NVLink vs InfiniBand, RDMA, bisection bandwidth. Every `β` in this chapter is a networking number.
- [Distributed systems](../../foundations/distributed-systems.md) - ranks, process groups, collectives, and why a slow rank stalls everyone.
- [Transformer architectures](../../models/architectures.md) - GQA/MQA, SwiGLU, MoE layers. The head-count constraints and the expert-parallel sections use these details.

You don't need the Linux/OS chapter or the beyond-transformers chapter here.

This chapter is the quantitative companion to the [Model Parallelism survey](README.md). The survey explains what each technique is and links to the implementations. This chapter derives what each technique costs in memory and in bytes moved, and then uses those formulas to choose a layout for two concrete models. The collective algorithms themselves (ring, tree, NVLS, how NCCL chooses between them) are covered in [collectives internals](../../network/collectives-internals.md). FLOPs-per-token and memory arithmetic for a single model are in [training arithmetic](../arithmetic.md). Here I only state those results and use them.

Every number below comes from a short Python computation that I ran. No number is a measured benchmark. All step times are first-order estimates from the stated assumptions, and I note where real systems are known to differ.

## Notation and hardware constants

| Symbol | Meaning |
|---|---|
| `Ψ` | number of model parameters |
| `N` | number of ranks in a collective (a *rank* is one process, normally driving one GPU) |
| `M` | message size in bytes (defined per collective below) |
| `α` | per-message latency (seconds): fixed startup cost of one send |
| `β` | per-rank bandwidth (bytes/s) in one direction on the slowest link the collective crosses |
| `F` | achieved FLOP/s per GPU (not peak) |
| `B, T, h` | micro-batch size (sequences), sequence length, hidden size |
| `L` | number of transformer layers |
| `t, p, d, c, e` | tensor-, pipeline-, data-, context-, expert-parallel degree |
| `m` | number of micro-batches per pipeline per step |

Hardware used in the worked examples. These are the brief's numbers, and they match H100 SXM specs:

- **HBM** (High Bandwidth Memory, the GPU's on-package DRAM, the memory "GPU memory" usually refers to): 80 GB per H100.
- **NVLink** (NVIDIA's GPU-to-GPU link inside a node, routed through NVSwitch chips so that any GPU can reach any other at full speed): 900 GB/s per GPU *bidirectional*, so `β_nvl = 450 GB/s` per direction. Achieved NCCL bus bandwidth is lower. Measure yours with the [network benchmarks](../../network/benchmarks/).
- **InfiniBand (IB)** (the inter-node network; each GPU has its own NIC, a network interface card, and talks to it via **RDMA**, remote direct memory access, which means one machine writes directly into another machine's memory without going through either CPU): 400 Gb/s per GPU, so `β_ib = 400/8 = 50 GB/s` per direction.
- Dense BF16 peak is 989 TFLOP/s. I assume 40% MFU, so `F = 0.4 * 989e12 = 395.6e12 FLOP/s`. This is an assumption, not a measurement.

A **node** here means 8 GPUs sharing one NVLink domain. A **collective** is a communication operation in which every rank of a group takes part, for example all of them summing a tensor together. **NCCL** is NVIDIA's library that implements collectives on GPUs. A **kernel** is a single function launched on the GPU. A **CUDA stream** is an ordered queue of kernels. Kernels on different streams may run concurrently, which is the mechanism behind every "overlap" claim in this chapter.

## Collective cost models (alpha-beta, ring)

The derivations are in [collectives internals](../../network/collectives-internals.md). The results for the ring algorithm on `N` ranks are:

```
all-reduce      (every rank ends with the sum of all ranks' M-byte buffers)
    t_AR = 2(N-1)·α + 2·(N-1)/N · M/β

reduce-scatter  (input M bytes per rank; rank i ends with the summed i-th 1/N slice)
    t_RS = (N-1)·α + (N-1)/N · M/β

all-gather      (rank i holds M/N; every rank ends with the full M bytes)
    t_AG = (N-1)·α + (N-1)/N · M/β

all-to-all      (rank i sends a distinct M/N slice to each rank j; M = rank's total send buffer)
    t_A2A ≈ (N-1)·α + (N-1)/N · M/β      (pairwise exchange, assumes full bisection bandwidth)
```

Three facts are used repeatedly:

1. `AR = RS + AG`, exactly in volume. Every rank sends `2(N-1)/N · M` bytes for an all-reduce and `(N-1)/N · M` for each half. This identity drives ZeRO, FSDP and sequence parallelism.
2. The bandwidth term barely depends on `N`, and the latency term grows linearly in `N`. At large `N` and small `M`, NCCL switches to tree/NVLS algorithms with `O(log N)` latency. The bandwidth formulas above remain a good lower bound for large messages.
3. `β` is the bottleneck link. A ring that spans nodes runs at roughly NIC speed (50 GB/s), not NVLink speed, even though most of its hops are intra-node. NCCL runs one ring per NIC in parallel, which is why the per-GPU NIC bandwidth is the right `β`.

From here on, "volume" means bytes sent per rank, and I often drop the `(N-1)/N` factor when `N` is large.

## Data parallelism (DDP)

**DDP** (DistributedDataParallel) keeps a full replica of the model on each of the `d` ranks, gives each a different slice of the batch, and averages gradients with an all-reduce before the optimizer step.

### Bucketing and overlap

Gradients become available in roughly reverse layer order during backward. PyTorch DDP:

- Assigns parameters to **buckets**: contiguous flat buffers, `bucket_cap_mb=25` by default, filled in reverse `model.parameters()` order. The first bucket is capped smaller (1 MiB, `dist._DEFAULT_FIRST_BUCKET_BYTES`) so that the first all-reduce launches early. I'm fairly but not fully certain of that constant name.
- Registers an autograd hook on each parameter. When every gradient in a bucket is ready, it launches an asynchronous all-reduce of that bucket on a separate NCCL stream while backward keeps computing on the compute stream.
- After the first iteration, rebuilds the buckets in the *observed* gradient-ready order (`_rebuild_buckets`).
- With `gradient_as_bucket_view=True`, `param.grad` is a view into the bucket, which saves one gradient-sized copy.

The point of bucketing is the `α` term. One all-reduce per parameter tensor pays `2(N-1)α` hundreds of times. One all-reduce of the whole model can't start until backward finishes, so nothing overlaps. A bucket of 25 MB is large enough to be bandwidth-bound and small enough to give many overlap opportunities. The exposed communication is the last bucket, plus any part of the total that doesn't fit under backward compute.

Gradient accumulation uses `no_sync()`: skip the all-reduce on the first `k-1` micro-batches. This costs nothing extra in memory, because the gradients exist anyway.

### Communication volume

With `b_g` bytes per gradient element (2 for bf16 gradients, 4 for fp32 gradients as in classic AMP with fp32 params), one all-reduce of all gradients per step sends:

```
V_DDP = 2·(d-1)/d · b_g·Ψ  ≈  2·b_g·Ψ  bytes per rank per step
```

This is independent of batch size and of `d`.

### When is DDP comm-bound?

Let `T_g` be the tokens each GPU processes per optimizer step. Ignoring attention FLOPs, training compute is `6Ψ` FLOPs per token (2 forward, 4 backward). See [arithmetic](../arithmetic.md).

```
t_compute  = 6·Ψ·T_g / F
t_backward = 4·Ψ·T_g / F        # the only window into which the all-reduce can overlap
t_comm     = 2·b_g·Ψ / β
```

Comm is fully hidden only if `t_comm ≤ t_backward`:

```
2·b_g·Ψ/β ≤ 4·Ψ·T_g/F    ⟹    T_g ≥ b_g·F / (2β)
```

**Ψ cancels.** Whether DDP is comm-bound depends only on tokens per GPU per step, achieved FLOP/s and bandwidth, not on model size. Bigger models need more memory, but they are not more comm-bound under DDP.

For bf16 gradients (`b_g = 2`), `F = 395.6 TFLOP/s` and `β = 50 GB/s`:

```
T_g ≥ F/β = 395.6e12 / 50e9 ≈ 7,912 tokens per GPU per step    (overlap into backward only)
T_g ≥ 2F/(3β)              ≈ 5,275 tokens                       (break-even against whole step, no overlap)
```

With fp32 gradient reduction, double these. If the gradient all-reduce stays on NVLink (single node, 450 GB/s), the threshold drops to about 880 tokens. These thresholds assume perfect overlap and full achieved bandwidth. In practice, give yourself 2x headroom.

## ZeRO and FSDP

### Memory per GPU

With Adam and mixed precision, each parameter needs 2 bytes (bf16 weights), 2 bytes (bf16 gradients) and `K = 12` bytes of optimizer state (fp32 master weights, fp32 momentum, fp32 variance), for 16 bytes in total. See [ZeRO](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., arXiv 1910.02054). Sharding over `d` data-parallel ranks gives:

| Stage | Sharded | Bytes / GPU | 70.55B params, `d=64` |
|---|---|---|---|
| DDP | nothing | `16Ψ` | 1128.8 GB |
| ZeRO-1 | optimizer states | `4Ψ + 12Ψ/d` | 295.4 GB |
| ZeRO-2 | + gradients | `2Ψ + 14Ψ/d` | 156.5 GB |
| ZeRO-3 / FSDP | + parameters | `16Ψ/d` | 17.6 GB |

The ZeRO stages and what each shards:

![ZeRO stages](images/parallelism-zero.png)

([source](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/))

These exclude activations, temporary unsharded buffers and fragmentation.

FSDP2's usual setup (`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`) stores the *sharded* parameter in fp32 and uses it as the master copy. The sharded state is then `4 (fp32 param) + 4 (fp32 grad) + 8 (Adam m, v) = 16Ψ/d`, the same total with a different split. The bf16 copy exists only transiently, in the all-gathered (unsharded) buffer.

### Communication volume: why ZeRO-3 is 1.5x DDP

Count bf16 bytes per rank per step, with `b = 2`:

- **DDP**: all-reduce of gradients, `2bΨ`.
- **ZeRO-1/2**: reduce-scatter the gradients (each rank needs only its gradient shard to update its optimizer shard), `bΨ`. Then all-gather the updated bf16 parameters, `bΨ`. Total `2bΨ`, the same as DDP. This is just the `AR = RS + AG` identity with the optimizer step inserted in the middle.
- **ZeRO-3**: parameters are not resident, so:
  - forward: all-gather each layer's params before use, `bΨ`
  - backward: all-gather them again, because they were freed after forward, `bΨ`
  - backward: reduce-scatter gradients, `bΨ`
  - no final all-gather, because each rank updates only its own shard and the next forward will gather
  - total `3bΨ = 1.5 × V_DDP`

Gradient accumulation changes this. ZeRO-3 with `reshard_after_forward=True` repeats the two all-gathers for each micro-batch. If it also reduce-scatters every micro-batch, as it does by default, the cost becomes `3bΨ·k` for `k` micro-batches. With FSDP2 you can trade memory for volume: `reshard_after_forward=False` keeps unsharded params from forward to backward (which is ZeRO-2 behaviour, `2bΨ`), and `model.set_requires_gradient_sync(False)` on all but the last micro-batch accumulates unsharded gradients locally. The [survey's GAS discussion](README.md#inter-node-speed-requirements-to-use-zero) covers the DeepSpeed side.

The comm-bound condition is the same derivation as for DDP, with `3bΨ` in place of `2bΨ`. Forward all-gathers can overlap with forward compute, so the window is the whole step:

```
3·b·Ψ/β ≤ 6·Ψ·T_g/F   ⟹   T_g ≥ b·F/(2β) = F/β ≈ 7,912 tokens/GPU   (bf16, IB)
```

### Prefetching

FSDP overlaps by running layer `i+1`'s all-gather on a separate stream while layer `i` computes:

- **Forward prefetch**: issue AG(i+1) before compute(i). FSDP2 does this implicitly, one layer ahead. `module.set_modules_to_forward_prefetch([...])` prefetches more.
- **Backward prefetch**: issue AG(i-1) before RS(i), so the next layer's parameters arrive while the current layer's gradients are being reduced (FSDP1 `BackwardPrefetch.BACKWARD_PRE`).
- **Memory cost**: about 2 unsharded layers of bf16 params, plus 1 unsharded layer of gradients during backward. For a 70B Llama layer (855.7M params) that is `2·2·855.7e6 ≈ 3.4 GB` plus about 1.7 GB, which is small next to activations.

Prefetch depth is limited by memory. Overlap is limited by the CPU: if the Python thread falls behind (too many small modules, CPU-bound launches), the all-gather is issued late and is exposed. This is the most common reason FSDP underperforms its cost model.

### FSDP1 vs FSDP2

| | FSDP1 (`FullyShardedDataParallel`) | FSDP2 (`fully_shard`) |
|---|---|---|
| Unit of sharding | one `FlatParameter` per wrapped module: all params flattened and concatenated, then split into `d` equal 1-D chunks | each parameter is a `DTensor` sharded on dim 0 (`Shard(0)`) |
| Consequences | a parameter may straddle ranks; per-param attributes (dtype, `requires_grad`, weight decay) need `use_orig_params=True` and are awkward; padding at the end | per-param semantics preserved; frozen params, per-param optimizers, fp8 all-gather are natural |
| Composition with TP | params are already DTensors after TP, so flat-param over DTensor is clumsy | 2-D `DTensor` with placements `(Shard(0) on dp, Shard(k) on tp)` on a 2-D mesh |
| Memory management | `recordStream`-based, which can make memory release non-deterministic | copy-in/copy-out into a per-group flat buffer, deterministic frees |
| Communication | one AG/RS per FlatParameter | one AG/RS per `fully_shard` group (FSDP2 also flattens for the collective, then views back out) |

Communication volume is identical. The differences are in memory determinism, composability and correctness of per-parameter features. Reference: [PyTorch FSDP](https://arxiv.org/abs/2304.11277) (Zhao et al., arXiv 2304.11277) for FSDP1, [TorchTitan](https://arxiv.org/abs/2410.06511) (Liang et al., arXiv 2410.06511) for FSDP2 in a 4D stack.

### HSDP (hybrid sharding)

Shard within a group of `s` ranks and replicate across `r = d/s` groups. Pick `s` = one node so that the AG/RS traffic stays on NVLink:

```
memory           = 16Ψ/s
intra-group comm = 3·b·Ψ·(s-1)/s          over β_nvl  (AG, AG, RS)
inter-group comm = 2·b·(Ψ/s)·(r-1)/r      over β_ib   (all-reduce of the gradient shard)
```

Inter-node bytes drop by `1.5·s` relative to flat ZeRO-3, but memory is only divided by `s`. For a 70B model with `s = 8`, `16·70.55e9/8 = 141 GB`, which doesn't fit. HSDP (shard = 8) is a tool for models where `16Ψ/8 = 2Ψ` bytes fits, roughly `Ψ ≲ 20B` on 80 GB if you leave about half of HBM for activations. For larger models, shrink `Ψ` per rank with TP/PP first.

## Tensor parallelism (Megatron)

Reference: [Megatron-LM](https://arxiv.org/abs/1909.08053) (Shoeybi et al., arXiv 1909.08053). Background is in the [survey](README.md#tensor-parallelism).

### Column and row splits

For a 2-layer MLP `Z = act(X A) B` with `X: (BT, h)`, `A: (h, 4h)`, `B: (4h, h)`:

- **Column-parallel A**: `A = [A_1 … A_t]` split along the output dim. Rank `i` computes `Y_i = act(X A_i)`, shape `(BT, 4h/t)`. The activation is elementwise, so no communication is needed (splitting along rows would have required a sum *before* the nonlinearity).
- **Row-parallel B**: `B = [B_1; …; B_t]` split along the input dim. Rank `i` computes `Z_i = Y_i B_i`, shape `(BT, h)`, a *partial sum*. `Z = Σ_i Z_i` requires one all-reduce.

Attention works the same way: `W_Q, W_K, W_V` are column-parallel *by heads* (rank `i` owns `n_heads/t` query heads and `n_kv/t` KV heads), attention is computed locally per head, and `W_O` is row-parallel. This requires `n_kv % t == 0`, or KV-head replication when `t > n_kv`.

### The f / g conjugate operators

Megatron inserts two autograd functions per block:

```
f:  forward = identity        backward = all-reduce(dX)     # at block input, before the column-parallel GEMM
g:  forward = all-reduce(Z)   backward = identity           # at block output, after the row-parallel GEMM
```

Where `f` and `f̄` sit in a Megatron TP layer (Korthikanti et al.):

![transformer layer with tensor parallelism](images/megatron-tp-f-g-operators.jpg)

([source](https://arxiv.org/abs/2205.05198))

Why `f`'s backward is an all-reduce: `X` feeds every rank's `A_i`, so `dL/dX = Σ_i dY_i A_i^T`, a sum of partials across ranks. Why `g`'s backward is the identity: `Z` is replicated, and each rank already holds the full `dZ`.

### Exactly which all-reduces happen per layer

A transformer layer has two blocks (attention, MLP), and each block has one `f` and one `g`:

| Phase | All-reduces | Where |
|---|---|---|
| forward | 2 | `g` after attention `W_O`, `g` after MLP `B` |
| backward | 2 | `f` before attention QKV, `f` before MLP `A` |
| forward recompute (full activation checkpointing) | +2 | the forward `g`s run again |

Each all-reduce is over an activation of `B·T·h` elements, `M = 2BTh` bytes in bf16. Per layer per rank:

```
V_TP(layer) = 4 · 2(t-1)/t · 2BTh = 16·B·T·h·(t-1)/t  bytes     (no recompute)
            = 24·B·T·h·(t-1)/t                                   (full recompute)
```

### Why TP stays inside NVLink

TP communication is on the critical path. `g`'s output is the next block's input, so unless you use async-TP / collective-matmul decomposition ([survey](README.md#async-tp)), the GPU waits for it. Compare comm time to compute time per layer. For a standard layer with `12h²` params, the per-rank FLOPs are `6·12h²·BT/t` (forward + backward), so:

```
t_comm / t_compute = [16·BTh·(t-1)/t / β] / [72·h²·BT/(t·F)]
                   = 2·(t-1)·F / (9·h·β)
```

B and T cancel. The ratio depends only on `t`, `h` and the ratio of FLOP/s to bandwidth. For `t = 8`, `h = 8192`, `F = 395.6 TFLOP/s`:

| Link | `β` | ratio |
|---|---|---|
| NVLink (per direction) | 450 GB/s | `2·7·395.6e12/(9·8192·450e9)` = **0.167** |
| IB per GPU | 50 GB/s | **1.50** |

The exact computation for Llama-70B layer shapes (855.7M params/layer, GQA, causal attention FLOPs included) at `B=1, T=8192` gives:

- one AR: `2·7/8·2·8192·8192 = 234.9 MB`, which takes 0.52 ms on NVLink and 4.70 ms on IB
- 4 ARs per layer: 2.09 ms on NVLink and 18.8 ms on IB
- compute per layer per sequence on 8 GPUs: 14.3 ms

So TP=8 costs about 15% exposed time on NVLink and about 130% over IB. That is the whole argument. It holds for any `h` you are likely to train: you would need `h ≳ 74k` to bring cross-IB TP down to the NVLink ratio. On NVL72-class systems, the NVLink domain is 72 GPUs, and the same formula says TP can go wider. You still pay the `(t-1)/t` factor and shrinking GEMMs (`h/t` gets small, efficiency drops). That second effect is not in this model, so measure it.

## Sequence parallelism and context parallelism

### Megatron sequence parallelism (SP)

Reference: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) (Korthikanti et al., arXiv 2205.05198).

With SP, `g` (all-gather) and `ḡ` (reduce-scatter) replace the all-reduces, and the norm/dropout regions are sharded along the sequence:

![transformer layer with tensor and sequence parallelism](images/megatron-tp-sp-g-operators.jpg)

([source](https://arxiv.org/abs/2205.05198))

Under plain TP, LayerNorm, dropout and the residual stream are *replicated* on all `t` ranks, which is wasted memory. SP shards those regions along the sequence dimension and rewrites the conjugate operators using `AR = RS + AG`:

```
g  →  forward: reduce-scatter along T  (B, T, h) partials → (B, T/t, h)
      backward: all-gather
f  →  forward: all-gather along T      (B, T/t, h) → (B, T, h)
      backward: reduce-scatter
```

Forward volume per layer: 2 AG + 2 RS at `(t-1)/t · 2BTh` each, which equals 2 ARs. Per-layer volume is unchanged. One implementation detail: Megatron-LM keeps only the *sharded* input of a column-parallel linear and re-all-gathers it in backward to compute `dW`, overlapped with the `dX` GEMM. That is one extra AG per block in backward. I believe this is the current behaviour of `LinearWithGradAccumulationAndAsyncCommunication`, but verify it against your version.

Activation memory per layer (bytes, fp16/bf16 with dropout masks, standard GPT block with a `4h` MLP, `a` = heads), from the paper:

```
no parallelism:  s·b·h·(34 + 5·a·s/h)
TP:              s·b·h·(10 + 24/t + 5·a·s/(h·t))
TP + SP:         s·b·h·(34/t + 5·a·s/(h·t))
```

The `5as/h` term is the materialized `s×s` attention score/softmax/dropout. FlashAttention doesn't store it, so drop the term. At `s = 8192`, `b = 1`, `h = 8192`, `t = 8`:

- no parallelism: `34·s·h` = 2281.7 MB/layer
- TP: `(10 + 3)·s·h` = 872.4 MB/layer
- TP + SP: `34/8·s·h` = 285.2 MB/layer

The non-shardable "10" term disappears. For SwiGLU/GQA models the constants differ, so treat 34 as approximate.

### Context parallelism (CP)

CP shards *all* activations along `T`, including inside attention. Every layer except attention is token-local, so only attention communicates. Q is local. Each query block needs all keys and values, subject to the causal mask.

Ring attention passes KV blocks around the ring while each host computes blockwise attention on its query block:

![ring attention](images/ring-attention-overview.png)

([source](https://arxiv.org/abs/2310.01889))

Ulysses instead uses two all-to-alls to swap the sharded dimension from sequence to heads and back:

![DeepSpeed-Ulysses](images/ulysses-design.png)

([source](https://arxiv.org/abs/2309.14509))

**Ring attention** ([Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889), Liu et al., arXiv 2310.01889). Rank `i` holds `Q_i, K_i, V_i` for `T/c` tokens. In `c-1` steps it passes its current KV block to rank `i+1` (a point-to-point send/recv, not a collective), receives one from `i-1`, and accumulates attention with online softmax:

```
forward  volume/rank = (c-1) · B · (T/c) · 2·d_kv · b    ≈  2·B·T·d_kv·b       (d_kv = n_kv·d_head)
backward volume/rank ≈ 2× forward  (KV circulates again, plus dK, dV circulate back)
```

Overlap condition: step `j` computes a `(T/c)×(T/c)` block, `4·(T/c)²·h` FLOPs, while sending `2·(T/c)·d_kv·b` bytes:

```
4·(T/c)²·h / F  ≥  4·(T/c)·d_kv / β     (b = 2)
⟹  T/c ≥ F·d_kv / (h·β)
```

For the 70B (`h = 8192`, `d_kv = 1024`) over IB: `T/c ≥ 395.6e12·1024/(8192·50e9) ≈ 989` tokens per rank. With MHA (`d_kv = h`) it is `F/β ≈ 7,912`. With GQA, ring CP can cross nodes at modest chunk lengths, and it has no head-count constraint.

**Ulysses** ([DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509), Jacobs et al., arXiv 2309.14509). Before attention, an all-to-all converts sequence-sharding into head-sharding: `(B, T/c, n_heads, d)` → `(B, T, n_heads/c, d)`. Full-sequence attention then runs locally on `n_heads/c` heads, and an all-to-all on O converts back. Per rank, forward:

```
volume = (c-1)/c · B·(T/c) · (h_q + 2·d_kv + h_o) · b       # Q, K, V in; O out; h_q = h_o = h
```

Comparison per layer forward, GQA with `h/d_kv = 8`:

```
Ulysses / Ring = (2h + 2d_kv) / (c · 2d_kv) = (h + d_kv)/(c·d_kv) = 9/c
```

At `T = 128k`, `c = 8`: ring sends 469.8 MB per rank and Ulysses 528.5 MB (ratio 1.125). With strong GQA, ring sends *fewer* bytes until `c > 9`. With MHA (`d_kv = h`) the ratio is `2/c`, and Ulysses wins for any `c > 2`. This is the regime where the "constant comm volume" claim in the Ulysses paper is most favourable. Other trade-offs:

| | Ring | Ulysses |
|---|---|---|
| head constraint | none | `n_heads % c == 0` and `n_kv % c == 0`, otherwise replicate KV heads. 70B GQA: `c ≤ 8` |
| overlap | natural (P2P overlaps the block compute) | all-to-all is a sync point before/after attention, harder to hide |
| fabric | P2P, tolerates crossing IB | all-to-all wants high bisection, best on NVLink |
| kernel | needs a ring/online-softmax wrapper around flash attention | stock flash attention on full `T` |

The two compose (2D/"USP" style): Ulysses inside the node, ring across nodes.

**Causal load balancing.** With a contiguous split, rank `i` attends to `i+1` blocks, so work is proportional to `i + 1/2` and the last rank does `(c-1/2)/(c/2) = 2 - 1/c` times the mean: 1.5x at `c = 2`, 1.875x at `c = 8`. Fix (zigzag): cut T into `2c` chunks and give rank `i` chunks `i` and `2c-1-i`. Work becomes `(i+½) + (2c-1-i+½) = 2c`, the same on every rank. Megatron-CP and the Llama 3 CP use this layout. Llama 3 ([The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), arXiv 2407.21783) all-gathers K/V instead of ringing them, which is simpler and, with GQA, cheap.

## Pipeline parallelism

Split `L` layers into `p` stages, and split the per-replica batch into `m` micro-batches. Let `F`, `B`, `W` denote per-micro-batch per-stage times for forward, backward-for-inputs and backward-for-weights. Normalize to `F = 1`, `B_full = B + W = 2`. The survey's [PP section](README.md#pipeline-parallelism) has the pictures.

### GPipe and 1F1B: the bubble

GPipe ([arXiv 1811.06965](https://arxiv.org/abs/1811.06965)) runs all `m` forwards, then all `m` backwards. Stage `p-1` can't start until micro-batch 0 has passed `p-1` stages, and the drain is symmetric. The critical path is:

```
t_total  = (m + p - 1) · (t_F + t_B)
t_useful =  m          · (t_F + t_B)
bubble fraction (of total)  = (p-1)/(m+p-1)
bubble / ideal              = (p-1)/m
```

The bubble, from the GPipe paper:

![naive model parallelism vs GPipe](images/parallelism-gpipe-bubble.png)

([source](https://arxiv.org/abs/1811.06965))

1F1B (PipeDream-Flush, [arXiv 2006.09503](https://arxiv.org/abs/2006.09503)) reorders the same work. After `p - i` warm-up forwards, stage `i` alternates one forward and one backward. The bubble is *identical*, `(p-1)/(m+p-1)`, because the critical path still contains `p-1` fills and drains. What changes is memory:

| Schedule | Micro-batches of activations in flight on stage 0 |
|---|---|
| GPipe | `m` |
| 1F1B | `p` (stage `i` holds `p - i`) |

So 1F1B lets you raise `m` to shrink the bubble without raising memory.

### Interleaved 1F1B

[Megatron-LM 2021](https://arxiv.org/abs/2104.04473) (Narayanan et al., arXiv 2104.04473). Each device owns `v` non-contiguous chunks of `L/(pv)` layers. A chunk's F/B takes `1/v` of a stage's time, and the fill/drain still crosses `p-1` devices, so:

```
bubble = (p-1)·(t_F+t_B)/v        fraction = (p-1)/(v·m + p - 1)
```

The cost is `v`× more P2P messages, and slightly more memory. Megatron-LM's implementation warms up `2(p-1) + (v-1)·p` chunk-forwards on rank 0. In units of one micro-batch's activations for a full stage, that gives `(2(p-1) + (v-1)p + 1)/v = p + (p-1)/v` in flight. I derived this from the code's warm-up count, not from the paper. It also requires `m % p == 0` in Megatron.

### Zero-bubble: ZB-H1, ZB-H2

[Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) (Qi et al., arXiv 2401.10241) splits backward into `B` (computes `dX`, on the critical path to the previous stage) and `W` (computes `dW`, needed only by the optimizer). `W` can be scheduled into the holes.

- **ZB-H1**: same peak memory as 1F1B (`p` micro-batches). Bubble `(p-1)(t_F + t_B - t_W)`. With `F = B = W = 1` that is `(p-1)` versus 1F1B's `3(p-1)`, a third of the bubble.
- **ZB-H2**: more warm-up forwards fill the start, and Ws fill the end. The bubble is about 0 (it relies on bypassing the optimizer-step sync, with post-hoc validation). Peak memory is about `2p` micro-batches.
- A `W`-deferred micro-batch keeps its inputs and output-grads alive until `W` runs, which is where the extra memory comes from.

### DualPipe

DeepSeek-V3 ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437)) feeds micro-batches from *both ends* of the pipeline. Each device holds 2 stages (stage `i` and `p-1-i`), so parameters are duplicated (2×). It overlaps one chunk's forward with another's backward (`F&B`), so that the expert-parallel all-to-all of one hides under the compute of the other. Their table, using their notation (`B` = full backward, `W` = weight part):

| Schedule | Bubble | Params | Activations |
|---|---|---|---|
| 1F1B | `(p-1)(F+B)` | 1× | `p` |
| ZB1P | `(p-1)(F+B-2W)` | 1× | `p` |
| DualPipe | `(p/2-1)(F&B+B-3W)` | 2× | `p+1` |

DualPipe's bidirectional schedule:

![DualPipe schedule](images/parallelism-pp-dualpipe.png)

([source](https://arxiv.org/abs/2412.19437))

For `p = 16`, `m = 64`, `F = 1`, `B = 2`, `W = 1`, and pessimistically `F&B = F + B = 3` (no overlap gain), with useful work `m·(F+B) = 192` units:

- 1F1B: 45 units, 19.0% of the step
- ZB1P: 15 units, 7.2%
- DualPipe: 14 units, 6.8%
- interleaved with `v = 2`: 22.5 units, 10.5%

DualPipe's real win is not the bubble. It is that the all-to-all disappears under `F&B` overlap, which this bubble table doesn't capture.

### P2P volume

Each stage boundary sends one activation forward and one gradient backward per micro-batch: `2·B·T·h·b` bytes. With TP+SP each TP rank sends only its `1/t` shard. This is tiny compared with DP and TP, which is why PP is the one parallelism that tolerates IB well.

## Expert parallelism

An MoE layer has `E` experts and routes each token to `k` of them. EP places `E/e` experts per rank in a group of `e` ranks. Background is in the [survey](README.md#expert-parallelism).

### Dispatch / combine volume

With `n` tokens on a rank (after attention), forward:

```
dispatch: all-to-all, send n·k·h·b_d bytes   (k copies of each token's hidden state; b_d = 1 for FP8 dispatch, 2 for bf16)
combine:  all-to-all, send n·k·h·b_c bytes   (expert outputs back, b_c = 2)
fraction leaving the rank ≈ (e-1)/e under uniform routing
```

Backward repeats both in reverse, so a layer costs about `2·n·k·h·(b_d + b_c)·(e-1)/e` per rank.

DeepSeek-V3 caps cross-node fan-out with *node-limited routing*. Each token goes to experts on at most `M = 4` nodes, and is sent over IB once per target node, then forwarded over NVLink to the GPUs in that node. IB volume per token per MoE layer is then at most `M·h·b` per direction-pass instead of `k·h·b`.

### Capacity factor

Static-shape implementations (GShard [arXiv 2006.16668](https://arxiv.org/abs/2006.16668), Switch [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)) give each expert a fixed buffer:

```
capacity C = CF · (tokens in EP group · k) / E
```

Tokens beyond `C` are dropped, which means they skip the expert through the residual connection. Buffers are padded to `C`, so the a2a volume and expert FLOPs scale with `CF`. `CF = 1.25` costs 25% more bytes and FLOPs than perfect balance, and still drops tokens whenever an expert's load is more than 1.25x the mean. Dropless implementations (grouped GEMM with variable sizes, as in MegaBlocks, [arXiv 2211.15841](https://arxiv.org/abs/2211.15841)) remove the padding, but then the slowest rank sets the step time. Load balance is a throughput problem, not just a quality problem.

### Aux loss vs aux-loss-free balancing

- **Aux loss (Switch)**: `L_aux = α · E · Σ_i f_i · P_i`, where `f_i` is the fraction of tokens routed to expert `i` and `P_i` is the mean router probability for expert `i`. Switch uses `α = 0.01`. It pushes towards uniform load, but the gradient interferes with the LM objective, and a larger `α` hurts quality.
- **Aux-loss-free bias** ([Wang et al., arXiv 2408.15664](https://arxiv.org/abs/2408.15664), used in DeepSeek-V3). Add a per-expert bias `b_i` to the affinity *only for top-k selection*: `select topk(s_i + b_i)`. Gate values still use the unbiased `s_i`, normalized over the selected experts. After each step, `b_i -= γ` if expert `i` was overloaded and `b_i += γ` if it was underloaded. DeepSeek-V3 uses `γ = 0.001` for most of training, plus a tiny sequence-wise aux loss (`α = 1e-4`) as a guard against extreme per-sequence imbalance. There is no gradient through `b`, so the LM loss is untouched, and no token dropping. These constants come from my reading of the V3 report.

Measured expert load for aux-loss vs aux-loss-free models (DeepSeek-V3):

![expert load, aux-loss vs aux-loss-free](images/deepseek-v3-expert-load.png)

([source](https://arxiv.org/abs/2412.19437))

### EP × TP interactions

- Attention is usually TP+SP, so at the MoE layer each TP rank holds `n/t` tokens and dispatches only those. The a2a volume per GPU is divided by `t`.
- TP *inside* an expert (expert-TP, ETP) splits already-small FFNs (`2048` wide in V3), giving tiny, inefficient GEMMs and an extra AG/RS per expert. The modern default is ETP = 1. The expert mesh `(EP, EDP)` is decoupled from the attention mesh `(TP, CP, DP)`, so that the same GPUs are factored differently for the two halves of the layer. Megatron calls this "MoE Parallel Folding" ([arXiv 2504.14960](https://arxiv.org/abs/2504.14960), id unverified).
- Expert parameters are replicated across `EDP = world / (PP · EP)` ranks. Their gradients are reduced over that EDP group, not over the full DP group.

## Composing parallelisms: device mesh ordering

A **DeviceMesh** is an n-D array of ranks. Each mesh dimension is a process group, and rank coordinates are row-major, so the last dimension changes fastest and holds consecutive ranks, which means the same node. Put the parallelism with the highest bandwidth demand *on the critical path* innermost:

| Rank (innermost first) | Dim | Why |
|---|---|---|
| 1 | TP | ~hundreds of GB/step, synchronous per block (computed above). Must be NVLink. |
| 2 | CP / EP | per-layer traffic. Ring-CP overlaps and tolerates IB, Ulysses/EP all-to-all prefer NVLink. |
| 3 | DP / FSDP shard | per-step (or per-micro-batch) traffic, overlappable with compute |
| 4 | DP replicate (HSDP) | small per-step all-reduce of shards |
| 5 | PP | tiny P2P, tolerates IB best |

TorchTitan's mesh is `("pp", "dp_replicate", "dp_shard", "cp", "tp")`, outermost to innermost. Megatron's order string is `tp-cp-ep-dp-pp`, innermost first. Rank of coordinate `(pp, dpr, dps, cp, tp)`:

```
rank = (((pp·DPR + dpr)·DPS + dps)·CP + cp)·TP + tp
```

Two non-obvious points:

1. FSDP in TorchTitan shards over `dp_shard × cp` jointly, because CP ranks hold identical params. This flattened sub-mesh gets the FSDP collectives.
2. The product of the inner dims assigned to NVLink must not exceed 8 here. `TP = 8` leaves CP and EP crossing IB.

## Worked example 1: 70B dense on 1024 H100

Model (Llama-3-70B shapes): `h = 8192`, `L = 80`, FFN 28672 (SwiGLU), 64 query heads, 8 KV heads, `d_head = 128`, vocab 128256.

- params/layer: attention `h·(64·128) + 2·h·(8·128) + (64·128)·h` = 151.0M; MLP `3·h·28672` = 704.6M; total 855.7M
- total `Ψ = 80·855.7M + 2·128256·8192 (untied embeddings) ≈ 70.55B`

Workload: `T = 8192`, global batch `4·2^20 = 4,194,304` tokens, which is **512 sequences**. FLOPs/token is `6Ψ + 6·L·T·h` (causal attention: QKᵀ and AV are each `2Th` per layer forward, halved by the mask, times 3 for forward and backward):

- `6Ψ` = 423.3 GFLOP; attention 32.2 GFLOP (+7.6%)
- step = `455.5e9 · 4,194,304` = 1.911e18 FLOP
- `t_compute = 1.911e18 / 1024 / 395.6e12` = **4.72 s** per step

The first constraint is easy to miss: **512 sequences < 1024 GPUs**. Pure DP would need half a sequence per GPU, so some model or context parallel degree ≥ 2 is mandatory.

### Candidate A: FSDP over 1024 (CP = 2 intra-node, FSDP over `dp × cp`)

- Memory: states `16Ψ/1024` = 1.10 GB. Activations at 4096 local tokens: about `34·4096·8192` = 1.14 GB/layer, 91 GB for 80 layers. **Doesn't fit without full checkpointing.** Full checkpointing stores `2·4096·8192·80` = 5.4 GB plus one layer's working set. It fits, but adds a recompute forward.
- FSDP comm: `3·2·Ψ` = **422.9 GB/GPU** over IB, **8.46 s**, versus 4.72 s of compute. Even with ZeRO-2 behaviour (`2·2Ψ`, 282 GB), it takes 5.6 s. **Comm-bound by about 1.8x.**
- Sanity check against the DDP/FSDP threshold: 4096 tokens/GPU is below the `F/β ≈ 7,912` threshold for ZeRO-3, which is consistent.
- HSDP with shard = 8 to keep AG/RS on NVLink needs `16Ψ/8` = 141 GB of states. It doesn't fit.

Rejected.

### Candidate B: TP = 8 (+SP) intra-node × FSDP = 128

Per TP rank `Ψ/8` = 8.82B. Each DP rank (one node) gets `512/128 = 4` sequences: micro-batch 1 sequence, 4 accumulation steps, no recompute.

- **FSDP** with `reshard_after_forward=False` and gradient sync only on the last micro-batch, so params are gathered once and gradients reduce-scattered once: `2·2·8.82e9` = 35.3 GB → **0.71 s** over IB, overlappable. With default per-micro-batch resharding it would be `4·3·2·8.82e9` = 212 GB → 4.2 s. That is the trap.
- **TP+SP**: 4 AR-equivalents/layer × 80 layers × 4 micro-batches × 234.9 MB = 300.6 GB → **0.67 s** on NVLink, largely exposed (on the critical path).
- **Memory/GPU**:
  - sharded states 1.10 GB
  - unsharded bf16 params 17.6 GB
  - unsharded bf16 accumulated grads 17.6 GB
  - activations `34/8·s·h·80` = 22.8 GB
  - logits shard (vocab-parallel, fp32) `8192·128256·4/8` = 0.53 GB
  - total ≈ **60 GB**, about 20 GB headroom for fragmentation and temporaries

  An alternative that saves memory is per-micro-batch resharding, at 4.2 s of comm. Don't.

### Candidate C: TP = 8 × PP = 4 × DP = 32 (ZeRO-1)

20 layers/stage, interleaved with `v = 5` (4 layers/chunk). `m = 512/32 = 16` micro-batches.

- **Memory**: params/GPU `Ψ/32` = 2.20B. ZeRO-1 over 32 gives `4·2.20 + 12·2.20/32` = 9.65 GB. Activations `(p + (p-1)/v) = 4.6` micro-batches × 20 layers × 285.2 MB = 26.2 GB. Total ≈ 36 GB.
- **Bubble**: 1F1B `3/19` = 15.8%. Interleaved `3/(5·16+3)` = 3.6%, or 0.18 s.
- **DP comm** (RS + AG): `2·2·2.20e9·31/32` = 8.5 GB → 0.17 s, overlappable.
- **TP comm**: 20 layers × 16 micro-batches × 4 × 234.9 MB = 300.6 GB → 0.67 s. Same as B, because TP volume per GPU is `(L/p)·(seqs/d)·…`, and `p·d` is fixed at 128.
- **PP P2P**: `16·2·2·8192·8192/8` = 0.54 GB, 0.01 s.

### Comparison

| | A: FSDP1024+CP2 | B: TP8 × FSDP128 | C: TP8 × PP4(v5) × DP32 |
|---|---|---|---|
| states + params/grads | 1.1 GB (+ ckpt) | 36 GB | 9.6 GB |
| activations | 5.4 GB + recompute | 22.8 GB | 26.2 GB |
| IB bytes/GPU/step | 423 GB | 35 GB | 8.5 GB |
| IB time (overlappable) | 8.46 s | 0.71 s | 0.17 s |
| NVLink time (exposed) | ~0 | 0.67 s | 0.67 s |
| bubble | 0 | 0 | 0.18 s |
| est. step (compute 4.72 s) | ≥ 8.5 s (+recompute) | ≈ 5.4 s if FSDP overlaps | ≈ 5.6 s + 0.17 s if DP doesn't overlap |

**Pick B.** It has the same exposed TP cost as C, no bubble, no PP code, and its IB traffic is 15% of compute time, so it hides with margin under a 4.7 s step. Its risk is FSDP overlap quality at 128 ranks with 80 layer-groups, which you can check in a profiler trace in 10 minutes. If the trace shows exposed all-gathers, C is the fallback: its IB traffic is 4x lower. Async-TP would cut the exposed 0.67 s in both. A variant to consider is TP4 × FSDP256: TP 0.43 s, FSDP 3-pass 105 GB → 2.1 s, activations 45.6 GB. It is viable only if FSDP overlap is near perfect.

All of these step times are cost-model estimates (40% MFU assumed, nominal `β`). None of them has been run.

## Worked example 2: ~600B MoE (DeepSeek-V3 shapes) on 1024 H100

Model: `h = 7168`, 61 layers (3 dense, 58 MoE), 256 routed experts + 1 shared expert, `k = 8`, expert FFN 2048, 671B total, 37B active.

- one expert: `3·7168·2048` = 44.04M
- routed experts: `256·58·44.04M` = 653.9B
- everything else (MLA attention, dense layers, shared experts, embeddings) ≈ `671 − 653.9` = **17.1B**

Workload (my choice): `T = 4096`, 4096 sequences = 16.78M tokens/step. Compute `6·37e9·16.78e6/1024/395.6e12` = **9.19 s** (active-param FLOPs only, MLA attention ignored).

Both candidates use PP = 16 (`58/16 ≈ 3.6` MoE layers/stage) and 64 attention-DP ranks per stage with ZeRO-1 for the non-expert 17.1B: `17.1/16` = 1.07B/stage, which gives `4·1.07 + 12·1.07/64` = 4.47 GB. Each rank handles `16.78M/64` = 262,144 tokens/step = 64 micro-batches of 4096 tokens. The PP P2P is `64·2·4096·7168·2` = 7.5 GB → 0.15 s. The bubble is 7% with DualPipe or ZB, and 19% with 1F1B (numbers in the PP section).

### Candidate M1: EP = 64 across 8 nodes (DeepSeek-V3's training layout, scaled to 1024 GPUs)

- **Experts/GPU**: `256/64 = 4` per layer, so `3.6·4·44.04M` = 0.64B params, 10.2 GB at 16 B/param (EDP = 1, nothing to shard).
- **a2a** with node-limited `M = 4`, FP8 dispatch and bf16 combine: per token per MoE layer forward+backward `2·(4·7168·1 + 4·7168·2)` = 172 KB. Per GPU per step `262,144 · 3.625 · 172 KB` = **163 GB over IB → 3.27 s**, which is 36% of compute. This has to be hidden (DualPipe's F&B overlap). Without node-limited routing (8 copies, 56/64 remote) it would be 5.7 s.
- **Tokens/expert/micro-batch**: `4096·64·8/256` = 8192. The GEMMs are large and efficient.

DeepSeek reports roughly 1:1 computation-to-communication for cross-node EP on H800, versus my 0.36. The gap plausibly comes from FP8 compute (about 2x my `F`) and achieved a2a bandwidth well below 50 GB/s. I can't verify the split.

### Candidate M2: EP = 8 inside each node, EDP = 8 across nodes

- **Experts/GPU**: 32 per layer, so `3.6·32·44.04M` = 5.11B params. With bf16 params and grads resident and fp32 master and Adam ZeRO-sharded over EDP = 8: `5.11·(2 + 2 + 12/8)` = **28.1 GB**. With 4.5 GB of dense states, that leaves about 45 GB for activations (2x expert params if you use DualPipe: +10 GB).
- **a2a** entirely on NVLink: `262,144 · 3.625 · 2·8·7168·3 · 7/8` = 286 GB → **0.64 s**.
- **Expert gradient RS+AG** over EDP = 8 (IB): `2·2·5.11e9·7/8` = 17.9 GB → 0.36 s, once per step, overlappable with backward.
- **Tokens/expert/micro-batch**: `4096·8·8/256` = 1024. The GEMMs are 8x smaller per expert, with the same total FLOPs.

### Pick

On this hardware (900 GB/s NVLink) the arithmetic favours **M2**:

- 0.64 s NVLink + 0.36 s overlappable IB, versus 3.27 s of IB a2a that only an aggressive schedule hides
- simpler schedule (1F1B/ZB is fine)
- its costs are 3x more HBM per GPU for experts and 1024-row expert GEMMs, which lose some efficiency (unquantified here; measure grouped-GEMM throughput at M=1024 vs 8192 for 7168×2048)

M1 becomes attractive when NVLink is weak (H800: 400 GB/s), when HBM is tight, or when you need the larger per-expert GEMMs. These are some of the conditions DeepSeek actually faced. Both layouts are untested cost-model estimates.

## PyTorch sketch: DeviceMesh + FSDP2 + TP

Untested. There is no GPU or torch in the environment where this was written. APIs as of PyTorch 2.6+ (`torch.distributed.fsdp.fully_shard`; on older versions, `torch.distributed._composable.fsdp`). Launch with `torchrun --nproc-per-node 8 --nnodes N ...`.

```python
import os
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel,
    SequenceParallel, PrepareModuleInput,
)

class Attn(nn.Module):
    def __init__(s, h, nh, nkv):
        super().__init__()
        s.dh = h // nh
        s.wq = nn.Linear(h, nh * s.dh, bias=False)
        s.wk = nn.Linear(h, nkv * s.dh, bias=False)
        s.wv = nn.Linear(h, nkv * s.dh, bias=False)
        s.wo = nn.Linear(nh * s.dh, h, bias=False)

    def forward(s, x):                                      # x: (B, T, h), replicated over tp (AG'd by PrepareModuleInput)
        B, T, _ = x.shape
        # Colwise outputs are local shards: -1 resolves to nh/tp and nkv/tp local heads
        q = s.wq(x).view(B, T, -1, s.dh).transpose(1, 2)    # (B, nh/tp,  T, dh)
        k = s.wk(x).view(B, T, -1, s.dh).transpose(1, 2)    # (B, nkv/tp, T, dh)
        v = s.wv(x).view(B, T, -1, s.dh).transpose(1, 2)    # (B, nkv/tp, T, dh)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)  # (B, nh/tp, T, dh)
        o = o.transpose(1, 2).reshape(B, T, -1)             # (B, T, nh*dh/tp)
        return s.wo(o)                                      # rowwise: partial (B, T, h) -> reduce-scatter -> (B, T/tp, h)

class MLP(nn.Module):
    def __init__(s, h, f):
        super().__init__()
        s.w1, s.w3 = nn.Linear(h, f, bias=False), nn.Linear(h, f, bias=False)
        s.w2 = nn.Linear(f, h, bias=False)

    def forward(s, x):                                      # (B, T, h) replicated
        return s.w2(F.silu(s.w1(x)) * s.w3(x))              # (B, T, f/tp) local -> (B, T/tp, h) after RS

class Block(nn.Module):
    def __init__(s, h, nh, nkv, f):
        super().__init__()
        s.norm1, s.attn = nn.RMSNorm(h), Attn(h, nh, nkv)
        s.norm2, s.mlp = nn.RMSNorm(h), MLP(h, f)

    def forward(s, x):                                      # x: (B, T/tp, h) local, sequence-sharded residual
        x = x + s.attn(s.norm1(x))
        return x + s.mlp(s.norm2(x))

class Model(nn.Module):
    def __init__(s, V, h, nh, nkv, f, L):
        super().__init__()
        s.emb = nn.Embedding(V, h)
        s.layers = nn.ModuleList(Block(h, nh, nkv, f) for _ in range(L))
        s.norm, s.out = nn.RMSNorm(h), nn.Linear(h, V, bias=False)

    def forward(s, idx):                                    # idx: (B, T), same tokens on every tp rank
        x = s.emb(idx)                                      # (B, T/tp, h) after rowwise emb + RS
        for blk in s.layers:
            x = blk(x)
        return s.out(s.norm(x))                             # (B, T, V) replicated (see note on loss_parallel)

dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
TP = 8
mesh = init_device_mesh("cuda", (dist.get_world_size() // TP, TP), mesh_dim_names=("dp", "tp"))  # tp innermost = same node

with torch.device("meta"):                                  # no full-size allocation on any rank
    model = Model(V=128256, h=8192, nh=64, nkv=8, f=28672, L=80)

# 1) TP + SP inside each node
block_plan = {
    "norm1": SequenceParallel(),                            # norm on (B, T/tp, h)
    "attn": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),  # AG over T
    "attn.wq": ColwiseParallel(), "attn.wk": ColwiseParallel(), "attn.wv": ColwiseParallel(),
    "attn.wo": RowwiseParallel(output_layouts=Shard(1)),    # RS over T
    "norm2": SequenceParallel(),
    "mlp": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
    "mlp.w1": ColwiseParallel(), "mlp.w3": ColwiseParallel(),
    "mlp.w2": RowwiseParallel(output_layouts=Shard(1)),
}
for blk in model.layers:
    parallelize_module(blk, mesh["tp"], block_plan)
parallelize_module(model, mesh["tp"], {
    "emb": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
    "norm": SequenceParallel(),
    "out": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),  # AG of logits: simple, memory-heavy
})

# 2) FSDP2 across nodes; params become 2-D DTensors (dp-shard x tp-shard)
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for blk in model.layers:
    fully_shard(blk, mesh=mesh["dp"], mp_policy=mp, reshard_after_forward=False)  # one AG per step under GA (see example 1)
fully_shard(model, mesh=mesh["dp"], mp_policy=mp)

model.to_empty(device="cuda")
# ... init weights here (they are uninitialized after to_empty) ...
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)  # states sharded like params: 12 B/param / (dp*tp)

def step(micro_batches):                                    # list of (idx, tgt), each (B, T); identical across tp ranks
    for i, (idx, tgt) in enumerate(micro_batches):
        model.set_requires_gradient_sync(i == len(micro_batches) - 1)  # RS only on the last micro-batch
        logits = model(idx.cuda())                          # (B, T, V)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), tgt.cuda().flatten())
        (loss / len(micro_batches)).backward()
    opt.step()
    opt.zero_grad()
```

Notes on the sketch:

- **Order matters.** Apply TP first, then FSDP. FSDP then shards the already-TP-sharded DTensor again on dim 0 (internally a strided shard).
- **Output layer.** Replicating the `(B, T, V)` logits costs `8192·128256·2` = 2.1 GB/sequence in bf16 (4.2 GB if upcast to fp32 for the loss). In production, keep `output_layouts=Shard(-1)` and wrap the loss in `torch.distributed.tensor.parallel.loss_parallel()`.
- **Data loading.** All TP ranks must see the same tokens. Shard the data by `mesh["dp"].get_local_rank()`, not by global rank.
- **Missing pieces.** No CP, no PP, no activation checkpointing, no `torch.compile`, no checkpoint save/resume. TorchTitan has all of these and should be your starting point for anything real.

## Facts I'm not certain about

- DDP's first-bucket constant (`_DEFAULT_FIRST_BUCKET_BYTES`, 1 MiB): I'm confident about the behaviour, less so about the exact name.
- Megatron-SP re-gathering the input for `dW` in backward is my reading of the code. It may be configurable or have changed.
- The interleaved-1F1B in-flight memory `p + (p-1)/v` is derived from Megatron's warm-up count, not quoted from the paper.
- The DeepSeek-V3 constants (`γ = 0.001`, sequence-wise `α = 1e-4`, `M = 4` nodes, EP64/PP16 on 2048 H800) are from my reading of arXiv 2412.19437. The DualPipe table is reproduced from it.
- The "MoE Parallel Folding" arXiv id 2504.14960 is unverified.
- The activation-memory constant 34 is for a GPT block with a 4h GELU MLP and dropout. For SwiGLU/GQA models without dropout it is approximate.
- All `β` values are nominal. Achieved all-reduce/all-to-all bus bandwidth is typically noticeably lower and depends on message size. Plug in measured busbw.

## Further reading

- [Model Parallelism survey](README.md) - implementations, pictures, and the ZeRO inter-node analysis
- [Collectives internals](../../network/collectives-internals.md) - ring/tree/NVLS derivations behind the alpha-beta models
- [Training arithmetic](../arithmetic.md) - FLOPs/token, memory accounting, MFU
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, [arXiv 1910.02054](https://arxiv.org/abs/1910.02054)
- Zhao et al., *PyTorch FSDP*, [arXiv 2304.11277](https://arxiv.org/abs/2304.11277)
- Shoeybi et al., *Megatron-LM*, [arXiv 1909.08053](https://arxiv.org/abs/1909.08053)
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, [arXiv 2104.04473](https://arxiv.org/abs/2104.04473)
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, [arXiv 2205.05198](https://arxiv.org/abs/2205.05198)
- Liu et al., *Ring Attention with Blockwise Transformers*, [arXiv 2310.01889](https://arxiv.org/abs/2310.01889)
- Jacobs et al., *DeepSpeed Ulysses*, [arXiv 2309.14509](https://arxiv.org/abs/2309.14509)
- Huang et al., *GPipe*, [arXiv 1811.06965](https://arxiv.org/abs/1811.06965)
- Narayanan et al., *Memory-Efficient Pipeline-Parallel DNN Training* (PipeDream-2BW/Flush), [arXiv 2006.09503](https://arxiv.org/abs/2006.09503)
- Qi et al., *Zero Bubble Pipeline Parallelism*, [arXiv 2401.10241](https://arxiv.org/abs/2401.10241)
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, [arXiv 2408.15664](https://arxiv.org/abs/2408.15664)
- Lepikhin et al., *GShard*, [arXiv 2006.16668](https://arxiv.org/abs/2006.16668)
- Fedus et al., *Switch Transformers*, [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)
- Gale et al., *MegaBlocks*, [arXiv 2211.15841](https://arxiv.org/abs/2211.15841)
- Llama Team, *The Llama 3 Herd of Models*, [arXiv 2407.21783](https://arxiv.org/abs/2407.21783)
- Liang et al., *TorchTitan*, [arXiv 2410.06511](https://arxiv.org/abs/2410.06511)
