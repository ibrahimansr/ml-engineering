# Profiling and Optimizing Training, Hands-On

A training step is a pipeline. The CPU fetches a batch, Python walks the model and hands thousands of kernels to the GPU, the GPU runs them, GPUs exchange gradients over the network, and the optimizer updates the weights. The step can only go as fast as its slowest stage. Optimizing without first finding that stage is guessing. You will usually guess wrong, because the stage you would naturally suspect (the matmuls) is often not the slow one.

This chapter is the practical companion to the theory chapters. It shows how to measure a step correctly, compute tokens/s and MFU from a real loop, read `torch.profiler` and Nsight Systems traces, find what is eating memory, and recognize the four classic pathologies from their traces. It then walks the standard optimization ladder on a small GPT training script, one change at a time, with measured numbers.


## Prerequisites

- [Concurrency, Part 7: GPU concurrency](../foundations/concurrency.md#part-7-gpu-concurrency) explains asynchronous kernel launch, CUDA streams and synchronization points. Everything in this chapter's timing sections depends on that model.
- [GPU execution model](../compute/accelerator/execution-model.md) covers what a kernel is, the roofline, [launch overhead](../compute/accelerator/execution-model.md#launch-overhead-and-cpu-boundness), [CUDA graphs](../compute/accelerator/execution-model.md#cuda-graphs) and the [nsys/ncu reference](../compute/accelerator/execution-model.md#profiling-nsys-and-ncu). This chapter uses those tools and does not re-explain them.
- [Training arithmetic](arithmetic.md) covers FLOPs per token, [MFU and HFU](arithmetic.md#mfu-and-hfu-properly), [activation memory](arithmetic.md#activation-memory) and the [caching allocator](arithmetic.md#pytorch-caching-allocator-internals). This chapter computes those quantities for real runs.
- [Software tune-up for performance](performance/README.md) is the existing reference on batch sizes, gradient checkpointing, DataLoader settings, `torch.compile` and model-shape choices. This chapter links to it rather than repeating it.
- [Model parallelism derivations: DDP](model-parallelism/derivations.md#data-parallelism-ddp), especially [bucketing and overlap](model-parallelism/derivations.md#bucketing-and-overlap) and [ZeRO and FSDP](model-parallelism/derivations.md#zero-and-fsdp). The multi-GPU sections here measure what that chapter derives.
- [Distributed systems: data parallelism](../foundations/distributed-systems.md#data-parallelism) and [networking: rendezvous over TCP](../foundations/networking.md#step-1-rendezvous-over-tcp-the-store) for what `torchrun`, ranks, `--master_addr` and the gloo backend are.
- [Computer architecture: pinned memory](../foundations/computer-architecture.md#why-this-matters-for-gpus-pinned-memory) for why `pin_memory=True` and `non_blocking=True` go together.
- [The toolkit chapter](../foundations/toolkit.md) for the shell, virtual environments and JSON used in the runnable parts.
- Useful but optional: [Debugging PyTorch: profilers and the memory profiler](../debug/pytorch.md#profilers), [collectives internals](../network/collectives-internals.md#compute-communication-overlap) and the [GPU software stack](../compute/gpu-software-stack.md).
- Forward links (later chapters, not needed to follow this one): [decentralized training](decentralized-training.md), [serving hands-on](../inference/serving-hands-on.md), [RL infrastructure](../platform/rl-infrastructure.md).

**Conventions used here.** Everything marked **[ran]** was executed on an Apple M4 laptop (10 CPU cores, 16 GB unified memory, macOS 15.7, Python 3.14, PyTorch 2.15.0) on either the CPU or the Apple GPU through the MPS backend (Metal Performance Shaders, PyTorch's Apple-GPU device `"mps"`). The output shown is real output from that machine. Everything marked **[GPU, untested]** needs an NVIDIA GPU. It follows documented APIs but was not run for this chapter. Every number from the M4 is one machine on one day: it is an anecdote, not a benchmark. Where I repeated a run, I show the spread.

Setup for the runnable parts:

```bash
python3 -m venv /tmp/prof-venv && /tmp/prof-venv/bin/pip install torch numpy
mkdir -p /tmp/prof && cd /tmp/prof
# training data: the bytes of this book's own markdown (look at it before training on it)
cat ~/ml-engineering/training/*.md ~/ml-engineering/foundations/*.md > data.bin   # use the path where you cloned this book
head -c 200 data.bin
```

```text
# Training Arithmetic and Numerics: Going Deeper

## Prerequisites

This chapter assumes you know the math of transformers and backprop. It does not assume any systems background, but a few foun
```


## The methodology

### The loop: measure, classify, change one thing, re-measure

Performance work is an experiment loop with four steps:

1. **Measure** the end-to-end number you care about (tokens/s, step time, MFU) with a correct timer, over enough steps to see the noise.
2. **Classify** the bottleneck. Profile once and assign the lost time to one of five classes (next section). Each class has a different family of fixes, so this step decides what you try next.
3. **Change one thing.** If you change two things and the step gets faster, you do not know which change did it, or whether one change helped while the other hurt.
4. **Re-measure** with the same harness, and check that the loss curve did not change beyond noise. A speedup that changes the math is a bug until proven otherwise.

Stop when the remaining gap to the roofline (the hardware limit for this workload) is smaller than the noise, or when the next fix costs more engineering time than the GPU hours it saves.

Write the result of each iteration down as a row: what you changed, the step time, tokens/s, peak memory and the loss at step N. The case study at the end of this chapter is a table like that.

```text
| change                  | step ms | tok/s  | peak mem | loss@30 | note                        |
| baseline                | 65.6    | 62.4k  |    -     | 3.226   | 3 runs: 65.6-66.9           |
| - per-step loss.item()  | 64.2    | 63.8k  |    -     | 3.245   | within noise on MPS         |
```

### The five bottleneck classes

Every slow step loses time in at least one of these places:

| class | what is happening | how it shows in a trace | first fixes |
| :-- | :-- | :-- | :-- |
| **dataloader** | the GPU waits for the next batch | gap at the start of each step; time inside `next(loader)`; DataLoader worker processes at 100% CPU | more workers, pinned memory, pre-tokenize offline, move augmentation to GPU |
| **CPU overhead** (launch-bound) | Python and the dispatcher cannot enqueue kernels as fast as the GPU runs them | many short kernels with idle gaps between them; the CPU thread is never idle | bigger batches, fewer and fused kernels (`torch.compile`, fused optimizer), CUDA graphs |
| **host-device syncs** | the CPU stops and waits for the GPU, so the queue drains | `cudaStreamSynchronize` / `cudaMemcpy DtoH` on the CPU row followed by a GPU gap | remove `.item()`, `.cpu()`, `print(tensor)`, data-dependent shapes from the hot loop |
| **kernels** | the GPU is busy, but the kernels are slow | the GPU row is dense; a few kernels dominate | bf16, FlashAttention/SDPA, `torch.compile` fusion, shapes divisible by 64/128, then `ncu` |
| **communication** | GPUs wait for collectives (NCCL all-reduce, all-gather and similar) | NCCL kernels running while no compute kernel runs | bucket and overlap gradient reduction, prefetch, faster interconnect, a different parallelism layout |

Memory is a sixth concern that cuts across all five: when you run out, you are forced into smaller micro-batches, recomputation or sharding, and each of those costs speed. Treat "fits with headroom" as a constraint, and treat speed as the objective you optimize subject to it.

### Write down the expected number first

Before profiling, compute what the step *should* cost, so you know whether you are 10% or 10x off. For a dense decoder transformer, [training FLOPs per token](arithmetic.md#flops-of-a-decoder-transformer) are about $6 \cdot N_{\text{nonembedding}} + 12 \cdot n_{\text{layer}} \cdot \text{seq\_len} \cdot d_{\text{model}}$. Divide by a realistic achievable FLOP/s to get a floor on step time:

```python
# expected_step.py - lower bound on step time before you run anything  [ran]
def flop_per_token(n_layer, d_model, seq_len, vocab):
    n_nonemb = 12 * n_layer * d_model**2 + vocab * d_model      # blocks + untied LM head
    return 6 * n_nonemb + 12 * n_layer * seq_len * d_model

f = flop_per_token(n_layer=4, d_model=256, seq_len=256, vocab=256)
tokens_per_step = 16 * 256                                       # B * T
achievable = 3.38e12                                             # measured fp32 matmul ceiling on M4 GPU (below)
print(f"{f/1e6:.1f} MFLOP/token, floor {tokens_per_step * f / achievable * 1e3:.1f} ms/step")
```

```text
22.4 MFLOP/token, floor 27.2 ms/step
```

The 12-per-block approximation gives 22.4 MFLOP/token. The exact count for the test model below, which includes biases and LayerNorms, is 22.5 MFLOP/token. The naive script ran at 65.6 ms/step, so it was 2.4x above the floor. That number tells you how much headroom there is before you start optimizing.


## Timing GPU code correctly

### Why the naive timer lies

PyTorch on a GPU is asynchronous. A call like `c = a @ b` does not compute anything on the CPU. It enqueues a kernel on a CUDA stream (an ordered queue of GPU work) and returns in microseconds. The CPU races ahead while the GPU works through the queue. A `time.perf_counter()` pair around GPU code therefore measures how long the *enqueue* took, unless something forces the CPU to wait. [Concurrency: why naive GPU timing is wrong](../foundations/concurrency.md#why-naive-gpu-timing-is-wrong) walks through the mechanism. Here is the effect, measured on the Apple GPU, which is asynchronous in the same way:

```python
# timing.py - three ways to time GPU work; only two are right  [ran, MPS]
import time, torch
dev = "mps"
a = torch.randn(4096, 4096, device=dev)                 # (4096, 4096)
for _ in range(3): a @ a                                # warmup: shader compile, allocator
torch.mps.synchronize()

t0 = time.perf_counter(); c = a @ a
print(f"no sync      : {(time.perf_counter()-t0)*1e3:7.3f} ms   <- measures the enqueue, not the matmul")

torch.mps.synchronize(); t0 = time.perf_counter(); c = a @ a; torch.mps.synchronize()
print(f"sync, 1 iter : {(time.perf_counter()-t0)*1e3:7.3f} ms")

n = 20; torch.mps.synchronize(); t0 = time.perf_counter()
for _ in range(n): c = a @ a
torch.mps.synchronize()
print(f"sync, {n} iter: {(time.perf_counter()-t0)/n*1e3:7.3f} ms   -> {2*4096**3/((time.perf_counter()-t0)/n)/1e12:.2f} TFLOP/s")

start, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
start.record()
for _ in range(n): c = a @ a
end.record(); end.synchronize()
print(f"events       : {start.elapsed_time(end)/n:7.3f} ms")
```

```text
no sync      :   0.329 ms   <- measures the enqueue, not the matmul
sync, 1 iter :  52.466 ms
sync, 20 iter:  47.809 ms   -> 2.87 TFLOP/s
events       :  45.761 ms
```

The unsynchronized timer is 145x too fast. Taken at face value it would claim 418 TFLOP/s from a laptop. The rules:

- **Synchronize before starting the clock and before stopping it.** The first sync makes sure earlier queued work is not billed to you. The second makes sure your work has actually finished.
- **Warm up.** The first calls pay for one-time costs: kernel or shader compilation, cuBLAS heuristics, allocator growth, and `torch.compile` tracing. Exclude them.
- **Average over many iterations.** A single iteration includes the fixed cost of one sync, about 5 ms above (52.5 vs 47.8).
- **Device events** (`torch.cuda.Event` / `torch.mps.Event` with `enable_timing=True`) timestamp the GPU queue itself. They do not include the CPU's view of the gap, which makes them the right tool for timing a region *inside* a running loop without adding a full sync.

On CUDA the same pattern is: **[GPU, untested]**

```python
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
y = model(x)
end.record()
torch.cuda.synchronize()            # or end.synchronize()
print(start.elapsed_time(end), "ms")

# or let PyTorch handle warmup, sync and repetition:
import torch.utils.benchmark as tb
print(tb.Timer("model(x)", globals={"model": model, "x": x}).blocked_autorange(min_run_time=2))
```

`torch.utils.benchmark.Timer` synchronizes CUDA for you and reports a median with an IQR (interquartile range: the spread between the 25th and 75th percentile runs). Use it for micro-benchmarks of single ops or modules.

### The attribution trap

In a real training loop there is a second, subtler lie. A CPU-side timer around one *line* gets billed for whatever GPU work is queued ahead of the next blocking call. The first version of the training script below wrapped `next(loader)` and the host-to-device copy `x.to(dev)` in one timer. With `loss.item()` called every step (a sync), "data wait" read 0.5%. After the per-step `.item()` was removed, the same timer read 13.3%, although nothing about the data had changed:

```text
[ran, MPS] timer around next(loader) AND x.to(dev):
  log every step : step=65.6ms  data_wait= 0.5%
  log every 10   : step=64.2ms  data_wait=13.3%     <- the copy now blocks behind queued GPU work
[ran, MPS] timer around next(loader) only:
  log every 10   : step=64.6ms  data_wait= 0.3%
```

Once the CPU was free to run ahead, the copy became the first operation that had to wait for the GPU, so the GPU's compute time landed in the "data" bucket. The fix was to time only the CPU work (`next(loader)`) and nothing that touches the device. More generally, **CPU timers measure where the CPU waited, not where the time went.** Use a profiler trace whenever you want attribution.


## Tokens/s and MFU from a real loop

### The instrumented loop

The minimum instrumentation for any training loop:

```python
# inside the training loop (excerpt of gpt.py below)  [ran]
for step in range(warmup + steps):
    if step == warmup:
        sync(); t0 = time.perf_counter(); data_t = 0.0            # start clock after warmup, on a drained queue
    td = time.perf_counter()
    x, y = next(loader)                                           # CPU-side fetch only
    data_t += time.perf_counter() - td
    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    loss = model(x, y); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if step % log_every == 0:
        last = loss.item()                                        # the only sync in the loop
sync(); dt = time.perf_counter() - t0
tok_s = steps * B * T / dt
print(f"tok/s={tok_s:.0f}  achieved={tok_s * flop_per_tok / 1e12:.3f} TFLOP/s  data_wait={data_t/dt:.1%}")
```

`non_blocking=True` lets the host-to-device copy run asynchronously, but only if the source tensor is in [pinned memory](../foundations/computer-architecture.md#why-this-matters-for-gpus-pinned-memory) (the DataLoader's `pin_memory=True`); from ordinary pageable memory the copy is effectively synchronous.

In a multi-GPU job, tokens/s is global (summed over data-parallel ranks), and MFU divides by the GPU count. See [MFU from a measured tokens/s](arithmetic.md#mfu-from-a-measured-tokenss):

```python
def mfu(tokens_per_sec_global, n_gpus, flop_per_token, peak_flops):
    return tokens_per_sec_global / n_gpus * flop_per_token / peak_flops
```

### The denominator: spec peak vs measured ceiling

MFU (model FLOPs utilization: achieved useful FLOP/s divided by peak FLOP/s) needs a peak. The convention for reporting is the vendor's dense spec peak (989.4 TFLOP/s bf16 for H100 SXM). For deciding *how much is left to gain*, it is more useful to measure the largest matmul throughput your device actually reaches. This book calls that MAMF, [maximum achievable matmul FLOPS](../compute/accelerator/README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table). Apple does not publish a FLOP/s figure for the M4 GPU, so for this chapter the measured number is the only denominator available:

```python
# mamf.py - measured matmul ceiling: the denominator you can actually reach on this device  [ran, MPS]
import sys, time, torch
dev = sys.argv[1] if len(sys.argv) > 1 else "mps"
sync = {"mps": torch.mps.synchronize, "cpu": lambda: None}.get(dev, torch.cuda.synchronize)
for dtype in (torch.float32, torch.bfloat16):
    best = 0
    for n in (1024, 2048, 4096):
        a, b = torch.randn(n, n, device=dev, dtype=dtype), torch.randn(n, n, device=dev, dtype=dtype)
        for _ in range(3): a @ b
        sync(); t0 = time.perf_counter(); iters = 20
        for _ in range(iters): a @ b
        sync(); dt = (time.perf_counter() - t0) / iters
        best = max(best, 2 * n**3 / dt)
        print(f"{dev} {str(dtype):15s} n={n:5d} {2*n**3/dt/1e12:6.2f} TFLOP/s")
    print(f"  best {best/1e12:.2f}")
```

```text
mps torch.float32   n= 1024   2.32 TFLOP/s
mps torch.float32   n= 2048   3.38 TFLOP/s
mps torch.float32   n= 4096   3.22 TFLOP/s
  best 3.38
mps torch.bfloat16  n= 1024   3.52 TFLOP/s
mps torch.bfloat16  n= 2048   3.73 TFLOP/s
mps torch.bfloat16  n= 4096   3.76 TFLOP/s
  best 3.76
```

Two observations. First, bf16 buys only 11% on this GPU. On an NVIDIA GPU with tensor cores the gap between strict fp32 and bf16 is more than 10x (A100: 19.5 vs 312 dense TFLOP/s by spec), so the size of every ladder step depends on the hardware. Second, the timing script above measured 2.87 TFLOP/s for the same 4096 matmul that this one measured at 3.22. That is run-to-run variance of about 10% on a laptop GPU that shares power and thermals with everything else. Always repeat a measurement before you believe a difference smaller than that.

The baseline training run below achieved 1.40 TFLOP/s of model FLOPs. That is 41% of the measured fp32 ceiling, and the final optimized run reaches 57% of the bf16 ceiling.

### Worked example at H100 scale

**[arithmetic only; no H100 run]** GPT-2 small (12 layers, $d_{\text{model}} = 768$) at $\text{seq\_len} = 1024$, counting non-embedding parameters as $12 \cdot L \cdot d^2 = 84.9\text{M}$:

```python
# [ran: arithmetic]
L, d, T = 12, 768, 1024
f = 6 * 12 * L * d * d + 12 * L * T * d
print(f"{f/1e9:.3f} GFLOP/token; 40% MFU on one H100 = {0.40 * 989.4e12 / f:,.0f} tok/s")
```

```text
0.623 GFLOP/token; 40% MFU on one H100 = 635,398 tok/s
```

If your logged throughput for such a run is 150k tok/s on one H100, MFU is about 9.4%, and one of the five bottleneck classes is eating three quarters of the machine. For a small model, the usual culprits are CPU overhead and fp32 matmuls.


## The test subject: a small GPT

All case-study runs use this one file. It trains a byte-level (vocabulary 256) GPT on the bytes of this book. Every optimization is a command-line flag, so each run changes exactly one thing and the diff between two runs is the flag. **[ran]**

```python
# gpt.py - tiny byte-level GPT; every optimization is a flag so each run changes one thing.
import argparse, math, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Windows(torch.utils.data.IterableDataset):
    def __init__(self, path, B, T, ms):
        self.path, self.B, self.T, self.ms = path, B, T, ms
    def __iter__(self):
        data = torch.frombuffer(bytearray(open(self.path, "rb").read()), dtype=torch.uint8)
        g = torch.Generator().manual_seed(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0)
        while True:
            if self.ms: time.sleep(self.ms / 1e3)          # stand-in for tokenization/augmentation
            ix = torch.randint(len(data) - self.T - 1, (self.B,), generator=g)
            xy = torch.stack([data[i:i + self.T + 1] for i in ix]).long()  # (B, T+1)
            yield xy[:, :-1], xy[:, 1:]                     # (B, T), (B, T)

class Block(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.H = H
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.qkv, self.proj = nn.Linear(C, 3 * C), nn.Linear(C, C)
        self.fc, self.out = nn.Linear(C, 4 * C), nn.Linear(4 * C, C)
    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                                # 3 x (B, T, C)
        q, k, v = (t.view(B, T, self.H, C // self.H).transpose(1, 2) for t in (q, k, v))  # (B, H, T, hd)
        if args.attn == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # (B, H, T, hd)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))         # (B, H, T, T)
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            y = att @ v                                                      # (B, H, T, hd)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))                 # (B, T, C)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))

class GPT(nn.Module):
    def __init__(self, V, T, C, L, H):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, C), nn.Embedding(T, C)
        self.blocks = nn.ModuleList(Block(C, H) for _ in range(L))
        self.ln, self.head = nn.LayerNorm(C), nn.Linear(C, V, bias=False)
    def forward(self, idx, tgt):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))  # (B, T, C)
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if args.ckpt and self.training else b(x)
        logits = self.head(self.ln(x))                                       # (B, T, V)
        return F.cross_entropy(logits.flatten(0, 1).float(), tgt.flatten())

if __name__ == "__main__":                                # DataLoader workers re-import this file (spawn)
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--data", default="data.bin")
    p.add_argument("--B", type=int, default=16); p.add_argument("--T", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4); p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--attn", default="manual", choices=["manual", "sdpa"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--opt", default="plain", choices=["plain", "foreach", "fused"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--ckpt", action="store_true")             # activation checkpointing per block
    p.add_argument("--log_every", type=int, default=1)        # 1 = loss.item() every step (a host sync)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--data_ms", type=float, default=0.0)      # simulated per-batch CPU preprocessing cost
    p.add_argument("--profile", action="store_true")
    p.add_argument("--steps", type=int, default=30); p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)

    def sync():
        if dev.type == "cuda": torch.cuda.synchronize()
        elif dev.type == "mps": torch.mps.synchronize()

    V = 256
    model = GPT(V, args.T, args.n_embd, args.n_layer, args.n_head).to(dev)
    N = sum(p.numel() for p in model.parameters())
    N_nonemb = N - model.tok.weight.numel() - model.pos.weight.numel()
    opt_kw = {"plain": dict(foreach=False), "foreach": dict(foreach=True), "fused": dict(fused=True)}[args.opt]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, **opt_kw)
    step_fn = torch.compile(model) if args.compile else model
    loader = iter(torch.utils.data.DataLoader(Windows(args.data, args.B, args.T, args.data_ms), batch_size=None,
                  num_workers=args.workers, pin_memory=(dev.type == "cuda"), persistent_workers=args.workers > 0))
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.dtype == "bf16")

    # 6N for the matmuls + 12*L*T*C for attention scores/values (non-causal-halved convention, see arithmetic.md)
    flop_per_tok = 6 * N_nonemb + 12 * args.n_layer * args.T * args.n_embd
    data_t = 0.0
    prof = None
    if args.profile:
        from torch.profiler import profile, schedule, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        prof = profile(activities=acts, record_shapes=True, profile_memory=True,
                       schedule=schedule(wait=1, warmup=2, active=3, repeat=1),       # skip 1, warm 2, record 3 steps
                       on_trace_ready=lambda p: p.export_chrome_trace("trace.json"))
        prof.start()
    rf = torch.profiler.record_function                  # named range; shows up in the trace (and NVTX via emit_nvtx)
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            sync(); t0 = time.perf_counter(); data_t = 0.0
        td = time.perf_counter()
        with rf("data"):
            x, y = next(loader)                                # CPU-side batch fetch only
        data_t += time.perf_counter() - td
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with rf("forward"), amp:
            loss = step_fn(x, y)
        with rf("backward"):
            loss.backward()
        with rf("optimizer"):
            opt.step(); opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            last = loss.item()                                 # host sync: CPU waits for the GPU here
        if prof: prof.step()
    sync(); dt = time.perf_counter() - t0
    tok_s = args.steps * args.B * args.T / dt
    print(f"N={N/1e6:.2f}M  step={dt/args.steps*1e3:6.1f}ms  tok/s={tok_s:8.0f}  "
          f"achieved={tok_s*flop_per_tok/1e12:.3f} TFLOP/s  data_wait={data_t/dt*100:4.1f}%  loss={last:.3f}")
    if prof:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12, max_name_column_width=28))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=6,
              max_name_column_width=22, max_shapes_column_width=44))
```

```bash
$ python gpt.py                       # naive baseline on the M4 GPU  [ran]
N=3.36M  step=  65.6ms  tok/s=   62486  achieved=1.406 TFLOP/s  data_wait= 0.5%  loss=3.226
```

Two details are worth noticing. The `if __name__ == "__main__":` guard is required: on macOS (and anywhere the multiprocessing start method is `spawn`), DataLoader workers re-import the script, and without the guard every worker would try to train a model. The first `--workers 4` attempt died with `DataLoader worker (pid(s) ...) exited unexpectedly` for exactly that reason (see [multiprocessing start methods](../foundations/concurrency.md#multiprocessing-and-the-three-start-methods)). And `F.cross_entropy(... .float())` keeps the loss in fp32 even under bf16 autocast, which is one of the ops that [must stay fp32](arithmetic.md#what-must-stay-fp32-and-why).


## torch.profiler end to end

`torch.profiler` records every PyTorch operator the CPU dispatches (`aten::mm`, `aten::softmax`, ...), your own named ranges, and, on CUDA, every kernel and memcpy the GPU executes, linked back to the op that launched it. The [debug chapter section](../debug/pytorch.md#torchprofiler) (reference material outside this book's reading order) shows the one-shot `with profile(): ...` form. A training loop needs the scheduled form.

### Schedule, activities, shapes and memory

```python
from torch.profiler import profile, schedule, ProfilerActivity, record_function

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],   # CUDA: kernels + memcpy via CUPTI
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),    # per prof.step(): skip, warm (record+discard), record
    record_shapes=True,       # input shapes per op -> group_by_input_shape, find the one bad shape
    profile_memory=True,      # allocations/frees per op -> "CPU Mem"/"CUDA Mem" columns
    with_stack=False,         # True attributes ops to Python file:line; costly, turn on when you need it
    on_trace_ready=lambda p: p.export_chrome_trace("trace.json"),
    # or on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb")  -> one file per rank, for HTA (Holistic Trace Analysis, below)/TensorBoard
)                             # (TensorBoard: a browser dashboard that reads log/trace files from a directory)
prof.start()
for step, batch in enumerate(loader):
    with record_function("forward"):
        loss = model(batch)
    ...
    prof.step()               # advances the schedule; MUST be called once per iteration
prof.stop()
```

What the arguments are for:

- **`schedule`** exists because the first steps are not representative: they include compilation, cuBLAS autotuning and allocator growth. The profiler also has a warm-up of its own, since CUPTI (the CUDA profiling interface) has startup cost. `wait` steps are ignored entirely, `warmup` steps are traced and then thrown away, and `active` steps are kept. Three active steps are enough to see the steady state without producing a 1 GB trace.
- **`record_function("name")`** adds a named range. Use names that match how you think about the step (`data`, `forward`, `backward`, `optimizer`, `allreduce`). They appear in the trace as `user_annotation` events and, under `emit_nvtx`, as NVTX ranges in Nsight Systems.
- **`record_shapes`** and **`profile_memory`** add overhead (see below), so use them for diagnosis and leave them off for timing.

`ProfilerActivity` in this build of PyTorch has `CPU, CUDA, HPU, MTIA, XPU` and no MPS entry, so on Apple GPUs the profiler sees only CPU-side dispatch. The tables below were therefore taken with `--device cpu`, where CPU time *is* compute time.

### Reading `key_averages()`

```text
$ python gpt.py --device cpu --log_every 10 --steps 8 --warmup 0 --profile        [ran, CPU]
N=3.36M  step= 331.3ms  tok/s=   12363  achieved=0.278 TFLOP/s  data_wait= 0.1%  loss=5.740
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                        Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
               ProfilerStep*         0.02%     223.641us       100.00%     945.741ms     315.247ms      32.25 KB           0 B             3
                    backward         0.53%       5.055ms        53.56%     506.543ms     168.848ms    -959.29 MB         -12 B             3
                     forward         2.19%      20.700ms        39.90%     377.331ms     125.777ms     997.69 MB    -890.25 MB             3
autograd::engine::evaluat...         1.56%      14.711ms        21.77%     205.930ms       4.290ms    -299.89 MB    -672.00 MB            48
                    aten::mm        19.13%     180.951ms        19.14%     180.976ms       1.724ms     396.75 MB     396.75 MB           105
              AddmmBackward0         0.07%     616.864us        18.28%     172.892ms       3.602ms     372.00 MB           0 B            48
                aten::linear         0.07%     634.049us        15.00%     141.893ms       2.782ms     444.00 MB           0 B            51
                 aten::addmm        11.36%     107.431ms        14.19%     134.181ms       2.795ms     432.00 MB     432.00 MB            48
                   aten::bmm        13.69%     129.511ms        13.74%     129.914ms       1.804ms     576.00 MB     576.00 MB            72
                 aten::copy_         9.45%      89.394ms         9.45%      89.394ms      90.571us           0 B           0 B           987
autograd::engine::evaluat...         0.67%       6.293ms         8.98%      84.954ms       3.540ms     -48.00 MB    -384.00 MB            24
                BmmBackward0         0.02%     197.662us         8.32%      78.660ms       3.278ms     336.00 MB           0 B            24
----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 945.741ms
```

How to read it:

- **Self** time is time spent in the op itself, excluding the ops it called. **Total** time includes its children. Sort by `self_cpu_time_total` (or `self_cuda_time_total` on GPU) to find where cycles go. Sort by total to see which high-level region they belong to. Here the backward pass is 54% of the step and the forward 40%, a ratio of 1.34, well below the 2:1 the FLOP count predicts. Wall-time ratios under the profiler are not FLOP ratios (per-op overhead and non-matmul ops weigh differently in the two passes), so use the trace for *where*, not for checking the 2:1.
- The matmul family (`mm`, `addmm`, `bmm`) has about 44% self time. For a model this small, that is a healthy share.
- **`aten::copy_` with 987 calls and 9.5%** is the thing to chase. Those are dtype and layout copies (the `.long()` in the data path, `contiguous()` calls in attention reshapes, the `.float()` before the loss). At 90 us per call they are individually tiny and collectively a tenth of the step.
- The memory columns show allocation minus deallocation per op. `forward` allocates about 1 GB over 3 steps and `backward` frees it: that is the activation memory, about 330 MB per step, which matches the [saved-tensor count below](#measure-what-autograd-saves).

Grouping by input shape finds the one bad call site among many identical op names:

```text
----------------------  ------------  ------------  ...  ------------  --------------------------------------------
                  Name    Self CPU %      Self CPU  ...    # of Calls                                  Input Shapes
----------------------  ------------  ------------  ...  ------------  --------------------------------------------
             aten::bmm         5.97%      56.443ms  ...            24                [[64, 256, 64], [64, 64, 256]]
             aten::bmm         5.77%      54.553ms  ...            36               [[64, 256, 256], [64, 256, 64]]
aten::gelu_backward...         4.80%      45.379ms  ...            12        [[16, 256, 1024], [16, 256, 1024], []]
           aten::addmm         4.08%      38.568ms  ...            12    [[1024], [4096, 256], [256, 1024], [], []]
        aten::_softmax         3.92%      37.106ms  ...            12                   [[16, 4, 256, 256], [], []]
              aten::mm         3.79%      35.839ms  ...            12                    [[4096, 256], [256, 1024]]
```

The top two rows are the manual attention: $QK^\top$ producing $(B \cdot H, T, T) = (64, 256, 256)$ and $PV$. The softmax over the `(16, 4, 256, 256)` scores is fifth. That is the $T^2$ work that SDPA/FlashAttention replaces, and the trace shows it before you have changed anything.

### Exporting and opening a trace

`export_chrome_trace("trace.json")` writes the [Chrome Trace Event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU): a JSON object with a `traceEvents` list. Open it in [Perfetto UI](https://ui.perfetto.dev) (drag and drop; handles large files, runs locally in the browser) or `chrome://tracing`. Large traces load faster gzipped (`trace.json.gz`; Perfetto reads it directly).

In the UI, each process and thread is a row. On CUDA you get a CPU thread row with nested op slices, plus one row per CUDA stream with kernels. Clicking a kernel highlights a flow arrow back to the `cudaLaunchKernel` that issued it. Keys: `W/S` zoom, `A/D` pan, `M` marks a slice's extent, and dragging an area gives the summed durations of the selected slices.

Every slice is one event:

```json
{
 "ph": "X",              // "complete" event: has ts and dur
 "cat": "cpu_op",        // cpu_op | user_annotation | kernel | gpu_memcpy | cuda_runtime | ...
 "name": "aten::mm",
 "pid": 59618, "tid": 3350439,
 "ts": 3057069914434.109,  // microseconds
 "dur": 1326.621,          // microseconds
 "args": { "Input Dims": [[4096, 256], [256, 256]], "Input type": ["float", "float"],
           "Input Strides": [[256, 1], [1, 256]], "Sequence number": 839, ... }
}
```

(That is a real event from the trace above, abbreviated. JSON has no comments; they are added here for explanation.) On CUDA traces, kernel events carry `args` such as grid and block dimensions, and a `correlation` id that links them to the launching runtime call. The exact field set varies with the PyTorch version, so check your own trace before you script against it.

### Reading a trace without a GUI

A trace is data, so query it. This summary prints per-step time split by your `record_function` ranges, plus the top ops:

```python
# trace_summary.py - read a Chrome trace from torch.profiler without a GUI  [ran]
import json, sys, collections
ev = json.load(open(sys.argv[1]))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X"]                       # complete events: have ts (us) and dur (us)
print(f"{len(ev)} events, {len(X)} complete; categories:", dict(collections.Counter(e.get("cat") for e in X)))
steps = sorted((e for e in X if e["name"].startswith("ProfilerStep")), key=lambda e: e["ts"])
for s in steps:
    inside = [e for e in X if e is not s and e.get("cat") == "user_annotation" and s["ts"] <= e["ts"] < s["ts"] + s["dur"]]
    parts = {e["name"]: e["dur"] / 1e3 for e in inside}
    print(f'{s["name"]:16s} {s["dur"]/1e3:7.1f} ms  ' + "  ".join(f"{k}={v:.1f}" for k, v in parts.items()))
ops = collections.defaultdict(float)
for e in X:
    if e.get("cat") == "cpu_op": ops[e["name"]] += e["dur"]
tot = sum(s["dur"] for s in steps)
print("top ops by inclusive time (nested ops double count):")
for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:6]: print(f"  {k:28s} {v/1e3:8.1f} ms  {v/tot*100:5.1f}%")
```

```text
$ python trace_summary.py trace.json
29945 events, 22861 complete; categories: {'user_annotation': 24, 'cpu_op': 22836, 'Trace': 1}
ProfilerStep#3     316.8 ms  data=0.6  enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__=0.5  forward=125.8  backward=171.5  optimizer=18.9  Optimizer.step#AdamW.step=17.0  Optimizer.zero_grad#AdamW.zero_grad=1.8
ProfilerStep#4     312.4 ms  data=0.4  enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__=0.2  forward=123.1  backward=164.6  optimizer=24.3  Optimizer.step#AdamW.step=21.7  Optimizer.zero_grad#AdamW.zero_grad=2.5
ProfilerStep#5     316.5 ms  data=0.2  enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__=0.2  forward=128.5  backward=170.5  optimizer=17.2  Optimizer.step#AdamW.step=14.2  Optimizer.zero_grad#AdamW.zero_grad=3.0
top ops by inclusive time (nested ops double count):
  autograd::engine::evaluate_function: AddmmBackward0    205.9 ms   21.8%
  aten::mm                        181.0 ms   19.1%
  AddmmBackward0                  172.9 ms   18.3%
  aten::linear                    141.9 ms   15.0%
  aten::addmm                     134.2 ms   14.2%
  aten::bmm                       129.9 ms   13.7%
```

Note that PyTorch adds its own annotations (`enumerate(DataLoader)...`, `Optimizer.step#AdamW.step`) next to yours. 23k events over three steps of a 4-layer model is also why you keep `active` small for large models.

For multi-rank CUDA traces, Meta's [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) does this kind of analysis at scale: temporal breakdown (compute / communication / idle per rank), kernel-type breakdown, and communication-computation overlap. **[GPU, untested]**

```python
from hta.trace_analysis import TraceAnalysis          # pip install HolisticTraceAnalysis
analyzer = TraceAnalysis(trace_dir="./tb")            # dir written by tensorboard_trace_handler, one file per rank
print(analyzer.get_temporal_breakdown())              # per-rank idle / compute / non-compute time
print(analyzer.get_comm_comp_overlap())               # fraction of comm time overlapped with compute
```

### Profiler overhead is real

The profiled CPU run above took 315 ms/step against 173-180 ms unprofiled (`python gpt.py --device cpu --attn manual`), so with `record_shapes` and `profile_memory` on, overhead was about 75% here. On a GPU, CPU-side overhead matters only when it delays launches. In a launch-bound loop it does, so the profiler makes launch-bound loops look *more* launch-bound. Use the profiler to find *where* time goes and the plain timer to measure *how much*.

### What a CUDA trace adds

**[GPU, untested]** With `ProfilerActivity.CUDA`, `key_averages()` gains `Self CUDA` / `CUDA total` columns, and the trace gains:

- `cuda_runtime` slices on the CPU thread: `cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSynchronize`. A long `cudaStreamSynchronize` or `cudaMemcpy` (DtoH, device to host) is a host sync.
- one `kernel` row per stream, e.g. `ampere_bf16_s16816gemm...`, `flash_fwd_kernel`, `triton_poi_fused_...` (an Inductor-generated pointwise kernel), and `ncclDevKernel_AllReduce_...` on the NCCL stream.
- `gpu_memcpy` / `gpu_memset` slices for transfers.
- a "GPU utilization" style view in TensorBoard / HTA: kernel-busy time divided by wall time, per stream.

The single most useful thing to look at is the GPU kernel row in steady state. If it is dense, you are kernel-bound and should read the kernel names. If it is full of whitespace, the bottleneck is on the CPU side, and the CPU row above the gap tells you what the CPU was doing instead.


## Nsight Systems with NVTX ranges

Nsight Systems (`nsys`) is NVIDIA's system-wide timeline profiler. Compared with `torch.profiler`, it sees more and costs less: OS runtime calls, CPU sampling, CUDA API, kernels, memcpys, NCCL, NVLink and PCIe counters, and processes outside PyTorch. It knows nothing about PyTorch semantics unless you tell it. NVTX (NVIDIA Tools Extension) is the API for telling it: named push/pop ranges that appear as labelled bars on the timeline. The command-line reference and the "what you see vs what it implies" table are in [execution-model.md](../compute/accelerator/execution-model.md#nsight-systems-where-does-the-time-go). Here is how to instrument a training loop for it. **[GPU, untested]**

```python
# train_nsys.py - only the profiling-relevant lines
import torch
nvtx = torch.cuda.nvtx

for step, (x, y) in enumerate(loader):
    if step == 20: torch.cuda.profiler.start()          # cudaProfilerStart: nsys begins capture here
    nvtx.range_push(f"step {step}")
    nvtx.range_push("data");      x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True); nvtx.range_pop()
    nvtx.range_push("forward");   loss = model(x, y);                                       nvtx.range_pop()
    nvtx.range_push("backward");  loss.backward();                                          nvtx.range_pop()
    nvtx.range_push("optimizer"); opt.step(); opt.zero_grad(set_to_none=True);              nvtx.range_pop()
    nvtx.range_pop()
    if step == 25: torch.cuda.profiler.stop(); break    # 5 steady-state steps, then exit
```

```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o gpt_steps --force-overwrite true python train_nsys.py
nsys stats --report cuda_gpu_kern_sum gpt_steps.nsys-rep     # kernels by total GPU time
nsys stats --report cuda_api_sum      gpt_steps.nsys-rep     # cudaLaunchKernel count, cudaStreamSynchronize time
nsys stats --report nvtx_sum          gpt_steps.nsys-rep     # time per NVTX range (older nsys: nvtx_pushpop_sum)
```

Sketched (not real) shape of the kernel summary:

```text
 Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
 --------  ---------------  ---------  --------       ------------------------------------------
     38.1      ...             ...       ...           sm90_xmma_gemm_bf16bf16_bf16f32_...
     14.7      ...             ...       ...           flash_fwd_kernel / flash_bwd_...
      9.9      ...             ...       ...           ncclDevKernel_AllReduce_Sum_bf16_RING_LL
      ...
```

Three ways to get NVTX ranges without hand-writing them:

- `with torch.autograd.profiler.emit_nvtx():` wraps every autograd op in an NVTX range named after the op (with sequence numbers to match forward and backward ops). It is very detailed and adds overhead, so use it for a few steps.
- `torch.profiler.record_function` ranges become NVTX ranges under `emit_nvtx`, so the `data/forward/backward/optimizer` names from `gpt.py` carry over.
- Recent nsys versions can annotate PyTorch automatically (`--pytorch=autograd-nvtx`) and sample Python call stacks (`--python-sampling=true`). Both flags are version-dependent; check `nsys profile --help` on your install before relying on them.

Choosing between the two profilers: `torch.profiler` when the question is "which PyTorch op or module", nsys when the question involves anything outside PyTorch (dataloader processes, NCCL internals, PCIe/NVLink traffic, CPU scheduling, multiple processes on one node), and `ncu` once you know *which kernel* to look at. Remember that `ncu` replays kernels and locks clocks, so its times are not wall-clock times ([details](../compute/accelerator/execution-model.md#nsight-compute-why-is-this-kernel-slow)).


## Memory: snapshots and finding OOM causes

[Anatomy of model's memory usage](performance/README.md#anatomy-of-models-memory-usage) and [total memory and the caching allocator](arithmetic.md#total-memory-and-the-caching-allocator) explain what lives in GPU memory: weights, gradients, optimizer state, activations, temporaries, the CUDA context, NCCL buffers and allocator fragmentation. This section is about *measuring* which of those is your problem.

### Measure what autograd saves

Activation memory is whatever autograd saves for the backward pass. You can count it exactly, on any device, with a saved-tensor hook (`torch.autograd.graph.saved_tensors_hooks` lets you intercept every tensor the moment autograd saves it):

```python
# mem_probe.py - what autograd saves for backward, per module, with and without checkpointing  [ran, CPU]
import argparse, collections, torch, gpt
from torch.autograd.graph import saved_tensors_hooks
B, T, C, L = 16, 256, 256, 4
for attn in ("manual", "sdpa"):
    for ckpt in (False, True):
        gpt.args = argparse.Namespace(attn=attn, ckpt=ckpt)
        torch.manual_seed(0)
        model = gpt.GPT(256, T, C, L, 4)
        idx = torch.randint(256, (B, T))                                  # (B, T)
        seen, total = set(), 0
        by_shape = collections.Counter()
        def pack(t):
            global total
            key = (t.untyped_storage().data_ptr(), t.dtype)               # count each storage once
            if key not in seen:
                seen.add(key); n = t.untyped_storage().nbytes(); total += n
                by_shape[tuple(t.shape)] += n
            return t
        with saved_tensors_hooks(pack, lambda t: t):
            loss = model(idx, idx)
        print(f"attn={attn:6s} ckpt={ckpt!s:5s} saved for backward: {total/2**20:6.1f} MiB")
        if attn == "manual" and not ckpt:
            for s, n in by_shape.most_common(4): print(f"    {str(s):22s} {n/2**20:6.1f} MiB")
```

```text
attn=manual ckpt=False saved for backward:  344.8 MiB
    (16, 4, 256, 256)        64.0 MiB
    (16, 256, 1024)          64.0 MiB
    (4096, 1024)             64.0 MiB
    (4096, 256)              56.0 MiB
attn=manual ckpt=True  saved for backward:   28.3 MiB
attn=sdpa   ckpt=False saved for backward:  280.8 MiB
attn=sdpa   ckpt=True  saved for backward:   28.3 MiB
```

Check it by hand. One fp32 attention-probability tensor `(16, 4, 256, 256)` is $16 \cdot 4 \cdot 256 \cdot 256 \cdot 4\ \text{B} = 16\ \text{MiB}$, and 4 layers make 64 MiB. Switching to SDPA removes exactly those 64 MiB, because the fused kernel does not save the $T \times T$ probabilities. Checkpointing every block leaves only the block inputs ($16 \cdot 256 \cdot 256 \cdot 4\ \text{B} = 4\ \text{MiB}$ each) plus the head and loss. The count also includes weight tensors that autograd references (they are not extra memory), so treat it as an upper bound on activation bytes. Comparing this number with the [Korthikanti formula](arithmetic.md#korthikanti-et-al-derived) for your model is the fastest way to find an op that saves an unexpected fp32 copy.

On the Apple GPU, `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` give current numbers. There is no peak counter or snapshot API on MPS as of PyTorch 2.15, so the CUDA tools below have no Mac equivalent.

### CUDA memory snapshots

**[GPU, untested]** PyTorch can record every allocation and free with its Python stack, then dump a pickle that the [memory_viz](https://pytorch.org/memory_viz) page renders as a timeline of stacked allocations (drag the file in; it runs locally). This is the tool for "what is at the peak".

```python
import torch
torch.cuda.memory._record_memory_history(max_entries=100_000)   # private API; stable since ~2.1, check yours
try:
    for step in range(5):
        loss = model(*next(loader)); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
except torch.OutOfMemoryError:
    pass                                                         # dump even (especially) when it OOMs
finally:
    torch.cuda.memory._dump_snapshot("snap.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)       # stop recording
print(torch.cuda.memory_summary(abbreviated=True))
print({k: torch.cuda.memory_stats()[k] for k in
       ("allocated_bytes.all.peak", "reserved_bytes.all.peak", "num_alloc_retries", "num_ooms")})
```

How to read the memory_viz "Active Memory Timeline" for a training step: weights, gradients and optimizer state form a flat band at the bottom. Activations stack up during the forward pass in a staircase, one step per layer. The peak usually comes at the start of backward, or *during the first `optimizer.step()`*, when Adam's `exp_avg` and `exp_avg_sq` are allocated for the first time. Click a block to see the line of Python that allocated it. For analysing the snapshot as a dict (fragmentation, owner by frame), see [Reading `torch.cuda.memory._snapshot()`](arithmetic.md#reading-torchcudamemory_snapshot). `torch.profiler` can also produce a categorized memory timeline (`prof.export_memory_timeline("mem.html")`, which needs `record_shapes`, `profile_memory` and `with_stack`).

### OOM triage

Read the OOM message first. It contains the two numbers that decide which branch you are in: "X GiB is allocated by PyTorch" and "Y GiB is reserved by PyTorch but unallocated".

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.10 GiB
of which 1.20 GiB is free. Including non-PyTorch memory, this process has 77.88 GiB memory in use. Of the allocated
memory 70.12 GiB is allocated by PyTorch, and 6.95 GiB is reserved by PyTorch but unallocated. ...
```
(sketched; the format is real, the numbers are not)

| symptom | likely cause | check | fix |
| :-- | :-- | :-- | :-- |
| OOM on step 1, in forward | activations too big for this micro-batch | saved-tensor count vs formula | smaller micro-batch + grad accumulation; SDPA; checkpointing |
| OOM on step 1, at `optimizer.step()` | optimizer state allocated lazily on the first step | snapshot timeline: jump at first step | sharded optimizer ([ZeRO/FSDP](model-parallelism/derivations.md#zero-and-fsdp)), 8-bit or [bf16 states](arithmetic.md#bf16-optimizer-states), fused step |
| OOM after N steps, memory creeps | a reference leak: `losses.append(loss)` keeps graphs alive; growing caches | `memory_allocated()` per step rising | `loss.item()` / `.detach()` before storing |
| OOM only on some batches | variable sequence length; one long batch | log shape at OOM | bucket by length, cap length, `expandable_segments:True` |
| large "reserved but unallocated" | fragmentation | `inactive_split_bytes`, `num_alloc_retries > 0` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `nvidia-smi` shows much more than `max_memory_reserved` | CUDA context, NCCL buffers, another process | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | account for it; fewer NCCL channels; kill stray processes |
| OOM only at eval | eval run without `torch.no_grad()`, or bigger eval batch | snapshot at eval | `torch.inference_mode()` |

The leak row is the most common one in research code, and it is easy to reproduce on a CPU:

```python
# leak.py - a stored loss keeps its whole graph (and every saved activation) alive  [ran, CPU]
import torch, gc
w = torch.randn(1024, 1024, requires_grad=True)
def step(): return ((torch.randn(256, 1024) @ w).relu() @ w).sum()   # saves (256,1024) activations
keep = [step() for _ in range(50)]                                    # BUG: tensors with graphs
print("graphs alive:", sum(1 for o in gc.get_objects() if torch.is_tensor(o) and o.grad_fn is not None))
keep = [step().item() for _ in range(50)]                             # fix: store python floats
gc.collect()
print("graphs alive:", sum(1 for o in gc.get_objects() if torch.is_tensor(o) and o.grad_fn is not None))
```

```text
graphs alive: 100
graphs alive: 0
```

100, not 50: each stored loss also keeps alive the `relu` output that the second matmul saved for backward, along with its own `grad_fn`. Every saved activation of every stored step stays resident until the list goes away.

`debug/pytorch.md` has a longer walk-through of [OOM in forward](../debug/pytorch.md#debugging-cuda-oom-in-forward), [OOM in backward](../debug/pytorch.md#debugging-cuda-oom-in-backward) and [fragmentation](../debug/pytorch.md#overcoming-cuda-oom-due-to-memory-fragmentation).


## Spotting the four classic pathologies

### Dataloader starvation

The signature is a fraction of wall time spent inside `next(loader)` (time only the fetch, per [the attribution trap](#the-attribution-trap)), or, in a trace, a GPU gap at the start of each step. Simulate a slow preprocessing stage of 50 ms per batch:

```text
$ python gpt.py --log_every 10 --data_ms 50                 [ran, MPS]
N=3.36M  step=  74.5ms  tok/s=   54996  achieved=1.237 TFLOP/s  data_wait=73.6%  loss=3.245
$ python gpt.py --log_every 10 --data_ms 50 --workers 4
N=3.36M  step=  65.0ms  tok/s=   63047  achieved=1.418 TFLOP/s  data_wait= 0.5%  loss=3.137
$ python gpt.py --log_every 10 --data_ms 0                  # reference: free data
N=3.36M  step=  64.6ms  tok/s=   63447  achieved=1.427 TFLOP/s  data_wait= 0.3%  loss=3.245
$ python gpt.py --log_every 1  --data_ms 50                 # same slow data, plus a sync every step
N=3.36M  step= 124.7ms  tok/s=   32858  achieved=0.739 TFLOP/s  data_wait=43.9%  loss=3.226
```

Four lessons fit in those four lines:

1. 50 ms of data work on a 65 ms step cost only 10 ms when the loop had no per-step sync. The CPU slept in the dataloader *while* the GPU chewed on the previous step's queue, so the work overlapped. Starvation is the part of data time that is not hidden.
2. With a sync every step, the overlap disappears and the costs add: 65 + 50 + overhead = 125 ms. Syncs and slow data compound.
3. Four workers (separate processes producing batches in parallel; see [how the DataLoader works inside](../foundations/concurrency.md#part-4-how-the-pytorch-dataloader-works-inside)) brought the step back to the free-data number.
4. "data_wait 73.6%" is not "73.6% of the time is lost". It is the share of wall time the main thread spent in `next()`, which includes waiting that overlapped GPU work. The step-time delta (74.5 vs 64.6 ms) is the cost.

The loss changed with workers (3.137 vs 3.245) because each worker seeds its own generator, so the data order differs. That is expected, but it means a dataloader change is not a pure systems change, and you should compare loss curves over more steps than this. The general settings (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`) are in [DataLoader](performance/README.md#dataloader).

### CPU-launch-bound loops

Every eager op costs a few microseconds of Python and dispatcher time before the GPU sees it ([what it costs](../compute/accelerator/execution-model.md#what-it-costs)). Shrink the model until each kernel takes about as long as its launch:

```text
$ python gpt.py --n_embd 64 --B 4 --T 64 --steps 100 --warmup 20                   [ran, MPS]
N=0.24M  step=   6.2ms  tok/s=   41411  achieved=0.062 TFLOP/s  data_wait= 2.1%  loss=3.506
$ python gpt.py --n_embd 64 --B 4 --T 64 --steps 100 --warmup 20 --log_every 10
N=0.24M  step=   5.0ms  tok/s=   51496  achieved=0.077 TFLOP/s  ...
N=0.24M  step=   4.4ms  tok/s=   57850  achieved=0.087 TFLOP/s  ...     (second run)
```

The profiler counts what the CPU has to do per step:

```python
# count ops in one profiled step  [ran, MPS, after: python gpt.py --n_embd 64 --B 4 --T 64 --log_every 10 --steps 8 --warmup 0 --profile]
import json
X = [e for e in json.load(open("trace.json"))["traceEvents"] if e.get("ph") == "X"]
s = [e for e in X if e["name"].startswith("ProfilerStep")][-1]
ops = [e for e in X if e.get("cat") == "cpu_op" and s["ts"] <= e["ts"] < s["ts"] + s["dur"]]
print("aten ops in one step:", len(ops), " step ms:", round(s["dur"] / 1e3, 2), " us/op:", round(s["dur"] / len(ops), 1))
```

```text
aten ops in one step: 3587  step ms: 8.18  us/op: 2.3
```

3,587 dispatched ops (nested ones included) for a 4-layer model, at 0.06-0.09 TFLOP/s, which is 2% of the GPU's measured ceiling. Nothing in the math is slow; the step is all overhead. The signature: step time barely changes when you shrink the batch (so achieved FLOP/s falls with it), step time scales with the *number of ops* rather than their size, and in a CUDA trace you see a GPU row of short kernels separated by gaps, while the CPU row is solid `cudaLaunchKernel`. The fixes, in order: a bigger micro-batch (more work per launch), fewer ops (`torch.compile`, fused optimizer), and CUDA graphs (one launch for the whole step).

### Host-device syncs

A sync forces the CPU to wait until the GPU queue is empty. After it, the GPU sits idle while the CPU launches the next few kernels. Common hidden syncs: `.item()`, `.cpu()`, `.tolist()`, `print(tensor)`, `if tensor.any():`, `torch.nonzero`, boolean-mask indexing `x[mask]` (the output shape depends on the data), `.to("cuda")` from non-pinned memory, and `torch.cuda.empty_cache()`.

The cost scales with how much CPU work the sync exposes. In the launch-bound config above, removing a per-step `loss.item()` cut the step from 6.2 to 4.4-5.0 ms (20-30%). In the big config, the change was within noise (65.6 vs 64.2 ms), because the GPU work per step dwarfed the exposed launch time.

PyTorch can flag syncs for you on CUDA: **[GPU, untested]**

```python
torch.cuda.set_sync_debug_mode("warn")     # "error" raises; 0 / "default" turns it off
loss = model(x, y)
n = (y != -100).sum().item()              # -> UserWarning: called a synchronizing CUDA operation
```

In a trace, look for `cudaStreamSynchronize`, `cudaDeviceSynchronize` or `cudaMemcpyAsync` (DtoH) slices on the CPU thread, with a gap on the GPU row immediately after. Fixes: accumulate metrics on the GPU and call `.item()` every N steps; replace data-dependent indexing with masking (`torch.where`); use `non_blocking=True` with pinned memory for transfers to the device.

### Exposed communication in multi-GPU traces

In data parallelism (DDP), each GPU computes gradients on its own micro-batch, and the gradients are all-reduced (summed across GPUs) before the optimizer step. DDP groups gradients into **buckets** (default 25 MB) and launches each bucket's all-reduce as soon as backward has produced all of that bucket's gradients. The goal is for communication to run *while* the earlier layers are still computing their gradients. Communication that runs while no compute is running is **exposed**: it adds directly to step time. ([Compute-communication overlap](../network/collectives-internals.md#compute-communication-overlap) covers the theory; [DDP bucketing and overlap](model-parallelism/derivations.md#bucketing-and-overlap) covers how DDP builds the buckets, including the smaller first bucket and the rebuild after iteration 1.)

You can watch this happen on a laptop, using the gloo backend (PyTorch's CPU collective library; NCCL's role on CPU):

```python
# ddp_trace.py - torchrun --nproc_per_node 2 ddp_trace.py ; CPU + gloo stand-in for NCCL  [ran, CPU]
import os, torch, torch.distributed as dist, torch.nn as nn
from torch.profiler import profile, schedule, ProfilerActivity
dist.init_process_group("gloo"); rank = dist.get_rank(); torch.manual_seed(0)
torch.set_num_threads(4)
model = nn.Sequential(*[nn.Sequential(nn.Linear(1024, 1024), nn.GELU()) for _ in range(8)])  # 8.4M params
ddp = nn.parallel.DistributedDataParallel(model, bucket_cap_mb=int(os.environ.get("BUCKET_MB", 25)))
opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
x = torch.randn(64, 1024)                                                   # (B, C)
with profile(activities=[ProfilerActivity.CPU], schedule=schedule(wait=1, warmup=1, active=2),
             on_trace_ready=lambda p: p.export_chrome_trace(f"ddp_rank{rank}.json")) as prof:
    for _ in range(4):
        ddp(x).square().mean().backward(); opt.step(); opt.zero_grad(); prof.step()
if rank == 0:
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10, max_name_column_width=34))
dist.destroy_process_group()
```

```text
$ GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29511 ddp_trace.py
                              Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
                     ProfilerStep*        30.18%      25.531ms       100.00%      84.605ms      42.302ms             2
                   gloo:all_reduce         0.00%       0.000us             0      33.774ms       8.444ms             4
         Optimizer.step#AdamW.step         3.18%       2.690ms        32.94%      27.868ms      13.934ms             2
...
```

What the command does: `torchrun` starts 2 copies of the script (`--nproc_per_node 2`), each a separate rank, and gives them `--master_addr`/`--master_port` as the address where they meet to exchange connection details (the `TCPStore` rendezvous, see [networking](../foundations/networking.md#step-1-rendezvous-over-tcp-the-store)). The leading `GLOO_SOCKET_IFNAME=lo0` sets an environment variable for that one command, telling gloo to use `lo0`, the Mac's loopback interface (the network interface that only talks to the same machine).

(Without `GLOO_SOCKET_IFNAME=lo0 --master_addr 127.0.0.1`, the first attempt on this Mac failed in `TCPStore` setup and the next one hung, because the hostname did not resolve to a usable interface. See [which interface NCCL and gloo pick](../foundations/networking.md#which-interface-does-nccl-pick) for why loopback is the safe choice on one machine.)

The `gloo:all_reduce` slices are on a different thread from the compute. To measure what is exposed, take the union of communication intervals and subtract the part that overlaps any compute op on the main thread:

```python
# exposed_comm.py - how much collective time is NOT hidden under compute, from a Chrome trace  [ran]
import json, sys
X = [e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("ph") == "X"]
step = [e for e in X if e["name"].startswith("ProfilerStep")]
main_tid = step[0]["tid"]
comm = sorted({(e["ts"], e["ts"] + e["dur"]) for e in X if e["name"].startswith(("gloo:", "nccl:"))})  # c10d annotations
compute = sorted((e["ts"], e["ts"] + e["dur"]) for e in X if e["tid"] == main_tid and e["name"].startswith("aten::")
                 and "all_reduce" not in e["name"])
def union(iv):                                     # merge overlapping intervals -> list of disjoint ones
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
def length(iv): return sum(b - a for a, b in iv)
C, K = union(comm), union(compute)
overlap = sum(max(0, min(b, d) - max(a, c)) for a, b in C for c, d in K)
step_us = sum(e["dur"] for e in step)
print(f"steps={len(step)} step_total={step_us/1e3:.1f}ms comm={length(C)/1e3:.1f}ms "
      f"hidden={overlap/1e3:.1f}ms exposed={(length(C)-overlap)/1e3:.1f}ms ({(length(C)-overlap)/step_us*100:.0f}% of step)")
```

Varying only the bucket size, two runs each:

```text
bucket_cap_mb=1000  steps=2 step_total=85.2ms comm=25.6ms hidden=0.0ms exposed=25.6ms (30% of step)
bucket_cap_mb=1000  steps=2 step_total=85.6ms comm=27.2ms hidden=0.0ms exposed=27.2ms (32% of step)
bucket_cap_mb=25    steps=2 step_total=78.4ms comm=26.8ms hidden=2.0ms exposed=24.8ms (32% of step)
bucket_cap_mb=25    steps=2 step_total=80.2ms comm=23.1ms hidden=1.7ms exposed=21.4ms (27% of step)
bucket_cap_mb=1     steps=2 step_total=67.3ms comm=28.3ms hidden=23.7ms exposed=4.6ms (7% of step)
bucket_cap_mb=1     steps=2 step_total=62.4ms comm=24.3ms hidden=19.5ms exposed=4.9ms (8% of step)
```

With one giant bucket, nothing can start until backward finishes, so the whole all-reduce is exposed. The model has 33.6 MB of fp32 gradients. With the default setting, after DDP rebuilds its buckets following the first iteration ([how DDP builds buckets](model-parallelism/derivations.md#bucketing-and-overlap)), the first bucket is capped small (1 MiB, so here it holds just the last layer's 4 MB gradient), and the second bucket holds all the rest (the `gloo:all_reduce` count above is 4 calls over 2 profiled steps, so 2 buckets per step), so it only fills once backward is done. Almost all of the traffic therefore waits until late in backward, which gives almost no overlap. With 1 MB buckets (each layer's 4 MB gradient becomes its own bucket), communication starts after the last layer's backward and 80-84% of it hides, and the step gets 21-27% faster. On a real GPU cluster the trade-off has a second side: too-small buckets pay per-collective latency (the alpha term in the [alpha-beta model](../network/collectives-internals.md#the-alpha-beta-model)), which is why the default is tens of MB. On this laptop compute and gloo also share the same CPU cores, so treat the exact split as an anecdote. The method transfers unchanged to an NCCL trace: `nccl:all_reduce` annotations on the CPU side, `ncclDevKernel_*` kernels on the GPU side, and the same interval arithmetic against compute kernels.

In a real multi-GPU trace, also look for:

- **Stragglers.** One rank's compute is slower, so every other rank's all-reduce starts early and *waits*. The collective looks slow on every rank except the slow one. Compare the start times of the same collective across ranks.
- **Serialized FSDP all-gathers.** With FSDP (fully sharded data parallel: each GPU stores a shard of the parameters and all-gathers a layer just before using it; derived in [ZeRO and FSDP](model-parallelism/derivations.md#zero-and-fsdp)), the next layer's all-gather should run during the current layer's compute. A gap of "all-gather, then compute, then all-gather" means prefetching is off or blocked by memory limits.
- **Communication and compute fighting over SMs.** NCCL kernels occupy SMs. Overlapped compute kernels slow down a little; that is normal and much cheaper than exposure.


## The optimization ladder

These are the standard changes, roughly in the order you should try them: cheapest and most reliable first. Each row says which bottleneck class the change attacks, what to expect on an NVIDIA GPU (from the cited sources or from first principles, not measured here), how to verify it, and what it did on the M4 test model.

| step | attacks | expected effect on NVIDIA (not measured here) | verify | M4 GPU result (ran) |
| :-- | :-- | :-- | :-- | :-- |
| bf16 autocast (and TF32 for leftover fp32) | kernels | matmul peak 16x vs strict fp32 on A100 (312 vs 19.5 TF spec); end-to-end usually 2-3x for small models | kernel names contain `bf16`; loss curve matches fp32 | +11% matmul ceiling; within noise alone, ~1.15x on top of compile |
| remove per-step syncs | syncs, CPU overhead | large if launch-bound, ~0 if GPU-bound | no `cudaStreamSynchronize` per step | 20-30% (tiny model), noise (big model) |
| DataLoader workers / pinned memory | dataloader | eliminates data wait when CPU-side prep is the bottleneck | `next()` time ~0; no GPU gap at step start | 74.5 -> 65.0 ms with 50 ms/batch prep |
| fused / foreach optimizer | CPU overhead | fewer kernels in `optimizer.step()` | optimizer range time in trace | within noise |
| SDPA / FlashAttention | kernels, memory | FlashAttention paper: 3x end-to-end on GPT-2 at seq 1K vs the Hugging Face Transformers implementation (arXiv:2205.14135); no $T^2$ activation | `flash`/`efficient_attention` kernel in trace; activation memory drop | memory -64 MiB; speed within noise at $T=256$ |
| `torch.compile` | CPU overhead, kernels (fusion) | PyTorch 2.0 release: ~1.4x mean training speedup over 163 models on A100 (as of Mar 2023) | `triton_*` kernels; fewer kernels; check graph breaks | 1.22x (64.2 -> 52.6-53.9 ms) |
| activation checkpointing | memory (costs time) | +~33% compute for full recompute ([arithmetic](arithmetic.md#what-recomputation-adds)); pays when freed memory buys a bigger batch or removes sharding | activation memory drop, step time up | memory 345 -> 28 MiB; 1.30x slower |
| bigger micro-batch | CPU overhead, small-kernel inefficiency | better until GPU saturates, then flat; changes optimization dynamics | tok/s vs B curve flattens | flat (GPU already saturated at B=16) |
| overlap communication | communication | hides up to min(comm, backward compute) | exposed comm from trace | 30% -> 7% exposed (gloo, CPU) |
| CUDA graphs | CPU overhead | removes launch cost; matters when kernels are ~us | one `cudaGraphLaunch` per step | not available on MPS |

Each step in more detail:

### bf16 mixed precision

Autocast runs matmuls and convolutions in bf16 while keeping reductions, norms and the loss in fp32 ([what must stay fp32](arithmetic.md#what-must-stay-fp32-and-why)). bf16 has the same exponent range as fp32, so unlike fp16 it needs no loss scaling.

```python
torch.backends.cuda.matmul.allow_tf32 = True          # any fp32 matmul left over uses TF32 tensor cores (CUDA)
torch.backends.cudnn.allow_tf32 = True
with torch.autocast("cuda", dtype=torch.bfloat16):     # "mps" / "cpu" also accepted
    loss = model(x, y)
loss.backward()                                        # outside autocast; grads come out in param dtype (fp32)
```

**Verify:** on CUDA the kernel names change from `sgemm` (single-precision fp32 GEMM)/`tf32` to `bf16`, and the loss curve over a few hundred steps stays within seed noise of fp32. On the M4, bf16 alone moved the step from 64.2 to 62.8-66.3 ms, which is within noise: the Apple GPU's bf16 matmul is only 11% faster than fp32 (measured above). On top of `torch.compile` it gave 51-54 -> 44-47 ms, because Inductor fuses the casts that eager autocast leaves as separate kernels.

### Fused optimizers

The default AdamW update loops over parameters in Python and launches several kernels per parameter tensor. `foreach=True` (the CUDA default when available) groups them into multi-tensor kernels. `fused=True` does the whole update in one or a few kernels.

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)   # CUDA/MPS/CPU support varies by version
```

**Verify:** the `Optimizer.step#AdamW.step` range shrinks in the trace. On CPU it was 14-22 ms of a 315 ms profiled step (5-7%), so even a perfect optimizer would gain little here. On the M4 GPU, `plain`, `foreach` and `fused` were all within the 65-75 ms noise band, and one fused run on the tiny config hit a one-off 11.5 ms outlier: repeat measurements. Expect the win to matter for models with many small parameter tensors, and in launch-bound regimes.

### torch.compile

`torch.compile` captures the Python model into a graph (Dynamo), derives the backward graph (AOTAutograd) and generates fused Triton/C++ kernels (Inductor). How it works is in [torch.compile internals](../compute/accelerator/execution-model.md#torchcompile-internals). When it pays off and when it fails are in [torch.compile](performance/README.md#torchcompile).

```python
model = torch.compile(model)                          # default mode: fusion, no CUDA graphs
# model = torch.compile(model, mode="max-autotune")   # + GEMM autotuning and CUDA graphs; longer compile
```

```bash
TORCH_LOGS=graph_breaks python gpt.py --compile       # list every place Dynamo fell back to Python
TORCH_LOGS=recompiles   python gpt.py --compile       # shape/dtype changes that trigger recompilation
```

**Verify:** the step time drop must survive excluding compile time. The `warmup` steps in `gpt.py` absorb compilation; a 1-step run took 2.4 s wall time compiled versus 1.25 s eager (with a warm Inductor cache). Then check for graph breaks and recompiles. On the M4 GPU: 62.2-64.2 ms -> 52.6-53.9 ms (1.2x). On CPU: 179.8 -> 173.4 ms, which is within noise, since the CPU run is dominated by matmuls that compile does not change.

### SDPA and FlashAttention

`F.scaled_dot_product_attention` dispatches to a fused attention kernel (FlashAttention-2, memory-efficient attention, cuDNN attention, or a math fallback) chosen by dtype, shape and hardware. The fused kernels never write the `(B, H, T, T)` score matrix to memory ([FlashAttention](../compute/accelerator/execution-model.md#flashattention)).

```diff
-            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))         # (B, H, T, T)
-            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
-            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
-            y = att @ v                                                      # (B, H, T, hd)
+            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # (B, H, T, hd)
```

To verify which backend ran, force one and see whether it errors: **[GPU, untested]**

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):          # raises if flash is not eligible (dtype, head dim, mask...)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Common reasons it silently falls back to the slow math path: fp32 inputs (flash needs fp16/bf16), an explicit `attn_mask` instead of `is_causal=True`, or an unsupported head dimension. In the trace, look for `flash_fwd`/`flash_bwd` or `efficient_attention` kernels instead of `bmm` + `softmax`. On the M4 at $T = 256$ it saved the 64 MiB of saved probabilities (measured above) but not time. The $T^2$ part is 14% of FLOPs here (3.1M of 22.5M per token), and the payoff grows with sequence length.

### Activation checkpointing

Activation checkpointing (gradient checkpointing) discards a block's intermediate activations in the forward pass and recomputes them in the backward pass. It trades about one extra forward pass of compute for memory ([Gradient Checkpointing](performance/README.md#gradient-checkpointing)).

```python
from torch.utils.checkpoint import checkpoint
for b in self.blocks:
    x = checkpoint(b, x, use_reentrant=False)       # use_reentrant=False is the recommended implementation
```

**Verify:** saved-for-backward bytes fall (345 -> 28 MiB above) and step time rises (64-75 -> 84.6-85.9 ms, 1.3x on the M4, matching the ~4/3 expected for full recompute). Checkpointing is never a speed optimization on its own. It is worth it only when the freed memory buys something faster: a bigger micro-batch, less sharding, or no CPU offload. Selective checkpointing (only some layers, or only the attention core) sits between the two extremes.

### Bigger micro-batches

A bigger micro-batch means more work per kernel launch, which amortizes CPU overhead and fills the GPU better. It stops helping once the GPU is saturated, and it changes the optimization (the effective batch size and the learning rate you should use; see [batch sizes](performance/README.md#batch-sizes) and [critical batch size](arithmetic.md#critical-batch-size)):

```text
$ python gpt.py --log_every 10 --attn sdpa --opt fused --B 64              [ran, MPS]
N=3.36M  step= 260.9ms  tok/s=   62800  achieved=1.413 TFLOP/s  ...
```

4x the batch gave 4x the step time and the same tokens/s: at B=16 this GPU was already saturated, so the batch is not the lever here. On the launch-bound tiny config it would be. Hold the global batch fixed with gradient accumulation when you change the micro-batch for speed reasons.

### Overlapping communication

For DDP: tune `bucket_cap_mb` (measured above), set `gradient_as_bucket_view=True` (saves a gradient copy), and use `static_graph=True` when the graph does not change between steps. For FSDP2 (`torch.distributed.fsdp.fully_shard`), forward and backward prefetching of the next layer's all-gather is on by default; check the trace for gaps between all-gather and compute. **[GPU, untested]**

```python
ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], bucket_cap_mb=25,
                                                gradient_as_bucket_view=True, static_graph=True)
```

**Verify** with the exposed-communication measurement above, not with step time alone. A step can get faster for an unrelated reason. Larger designs (tensor and pipeline parallelism, async TP, DiLoCo-style infrequent synchronization, where each worker takes many local optimizer steps and workers average their weights only every few hundred steps) are covered in [model parallelism](model-parallelism/README.md), [the derivations](model-parallelism/derivations.md) and [decentralized training](decentralized-training.md).

### CUDA graphs

A CUDA graph records a step's kernels once and replays them with one launch. That removes the CPU overhead completely, but it requires static shapes and no syncs inside the step ([constraints and manual capture](../compute/accelerator/execution-model.md#cuda-graphs)). For training, the easy path is through compile: **[GPU, untested]**

```python
model = torch.compile(model, mode="reduce-overhead")   # Inductor + CUDA graphs (cudagraph trees)
# the input batch must keep the same shape every step; pad or bucket variable-length batches
```

**Verify:** the trace shows `cudaGraphLaunch` instead of thousands of `cudaLaunchKernel` calls, and the GPU row has no gaps. `TORCH_LOGS=cudagraphs` (or warnings at runtime) tells you why a graph was skipped, e.g. a CPU tensor input or a mutation. Expect large wins in launch-bound regimes (small models, small batches, decode-style workloads) and near zero for big GPU-bound steps. "Decode" is LLM generation, one token per forward pass reusing the [KV cache](../models/architectures.md#the-kv-cache): each step is many tiny kernels, which is why inference engines lean on CUDA graphs ([serving hands-on](../inference/serving-hands-on.md), a later chapter).


## Case study: naive to optimized, step by step

Starting point: `python gpt.py`, the naive script above, on the M4 GPU. Each row adds **one** change to the previous row. "runs" shows every measurement I took (30 timed steps after 10 warmup steps each), so you can see the noise. The loss is `loss.item()` at step 30 and must not move, except where the data order changes.

| # | change (flag added) | step ms (runs) | tok/s (best) | achieved TFLOP/s | loss@30 |
| :-- | :-- | :-- | --: | --: | --: |
| 0 | naive: manual attention, fp32, plain AdamW, `.item()` every step | 66.9, 65.6, 65.6 | 62,486 | 1.41 | 3.226 |
| 1 | `--log_every 10` (remove per-step sync) | 63.9, 64.2, 64.5 | 64,104 | 1.44 | 3.245 |
| 2 | `--attn sdpa` | 64.0, 68.3, 68.9 / later 62.6, 62.2 | 65,842 | 1.48 | 3.245 |
| 3 | `--opt fused` (on top of 2) | 73.8, 66.3, 75.9 | 61,800 | 1.39 | 3.245 |
| 3' | `--compile` (on top of 2, instead of 3) | 52.6, 53.9 (also 54.8) | 77,933 | 1.75 | 3.245 |
| 4 | `--compile --opt fused` | 51.3, 50.9 | 80,478 | 1.81 | 3.245 |
| 5 | `--compile --dtype bf16` (no fused) | 44.2, 46.8 | 92,708 | 2.09 | 3.245 |
| 6 | `--compile --opt fused --dtype bf16 --workers 2` | 43.4, 43.3 | 94,530 | 2.13 | 3.137, 3.138 |

The loss changes between rows 0 and 1 (3.226 vs 3.245) because row 0 logs at step 39 and row 1 at step 30: `last` is the last logged step, not a numerical difference. The row-6 change comes from the worker data order. Both are artifacts of the harness, and they are the kind of thing to explain before claiming anything.

The result is **65.6 -> 43.3 ms per step, 1.51x**, from 41% to 57% of the measured matmul ceiling. Of the ladder steps, only `torch.compile` (1.2x) and bf16 *on top of* compile (1.15x) were clearly outside the noise on this hardware. Removing syncs, SDPA and the fused optimizer were each within the ±8% run-to-run spread at this size. An honest summary is "compile and compile+bf16 helped; the others did nothing measurable on a 3M-parameter model on an Apple GPU at T=256." On an H100, the same ladder would look different: bf16 would be the largest single step, and SDPA would matter a lot at longer sequence lengths. That is the point of measuring instead of copying someone else's ladder.

The full diff from naive to optimized, as code rather than flags:

```diff
- loss = model(x, y)
+ model = torch.compile(model)                                    # once, before the loop
+ with torch.autocast(dev.type, dtype=torch.bfloat16):
+     loss = model(x, y)
- opt = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
+ opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
- att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)); ...; y = att @ v
+ y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
- last = loss.item()                                              # every step
+ if step % 10 == 0: last = loss.item()
- DataLoader(ds, batch_size=None, num_workers=0)
+ DataLoader(ds, batch_size=None, num_workers=2, pin_memory=True, persistent_workers=True)
```

Reproduce the table:

```bash
P=python
for a in "" "--log_every 10" "--log_every 10 --attn sdpa" "--log_every 10 --attn sdpa --opt fused" \
         "--log_every 10 --attn sdpa --compile" "--log_every 10 --attn sdpa --compile --opt fused" \
         "--log_every 10 --attn sdpa --compile --dtype bf16" \
         "--log_every 10 --attn sdpa --compile --opt fused --dtype bf16 --workers 2"; do
  echo "== $a"; for r in 1 2 3; do $P gpt.py $a | tail -1; done      # bash; in zsh use ${=a}
done
```

The outer loop sets `a` to each flag string in turn; the unquoted `$a` is split on spaces into separate arguments in bash. zsh does not split unquoted variables by default, so there `${=a}` forces the split (macOS's default shell is zsh). `| tail -1` keeps only the last line of output, the summary.

What to do next on a real GPU, in order: rerun this table on CUDA (expect bf16 and TF32 to be large); profile one compiled step and read the kernel row (are the GEMMs `bf16` tensor-core kernels; what fraction of time is non-GEMM); round the vocabulary and hidden sizes to multiples of 64/128 ([why](../compute/accelerator/execution-model.md#why-dims-want-multiples-of-8-16-64-128)); try `mode="max-autotune"` for CUDA graphs once shapes are static; and only then reach for `ncu` on the top kernel.

**Untested on GPU:** every number in this case study is from an Apple M4. No CUDA run was performed for this chapter.


## You can now read

- [GPU execution model](../compute/accelerator/execution-model.md), especially the roofline and `ncu` sections, to go from "this kernel dominates" to "why".
- [Training arithmetic](arithmetic.md) for deriving the expected step time and memory of any configuration before you run it.
- [Model parallelism](model-parallelism/README.md) and [decentralized training](decentralized-training.md), where the exposed-communication analysis becomes the central problem.
- [NCCL performance debugging](../network/debug/README.md) and [collectives internals](../network/collectives-internals.md) for when the communication row dominates.
- [Serving hands-on](../inference/serving-hands-on.md) (later) for the inference-side counterpart (latency budgets, CUDA graphs per batch size).
- [RL infrastructure](../platform/rl-infrastructure.md) (later), where the same methodology applies to rollout processes (the policy generating samples) and learner processes (the policy training on them), which alternate generation and training.


## Further reading

- PyTorch profiler recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- PyTorch `torch.profiler` API: https://docs.pytorch.org/docs/stable/profiler.html
- Understanding CUDA memory usage (snapshots, memory_viz): https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
- PyTorch CUDA semantics: asynchronous execution, CUDA graphs, allocator config: https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch performance tuning guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Holistic Trace Analysis: https://github.com/facebookresearch/HolisticTraceAnalysis
- Nsight Systems user guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Perfetto UI: https://ui.perfetto.dev
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", arXiv:2205.05198
- Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (DDP bucketing and overlap), arXiv:2006.15704
- PyTorch 2.0 announcement (torch.compile benchmark methodology and results): https://pytorch.org/get-started/pytorch-2.0/
- Andrej Karpathy, nanoGPT (https://github.com/karpathy/nanoGPT) and llm.c (https://github.com/karpathy/llm.c): small GPT training code with this ladder applied.
