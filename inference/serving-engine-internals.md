# Inference Internals II: Inside a Serving Engine

## Prerequisites

Read these first if the systems side is unfamiliar; the chapter defines each systems term once inline, but does not teach the background:

- [Computer Architecture](../foundations/computer-architecture.md) - GPU memory hierarchy, SMs, warps and the bandwidth-vs-compute distinction that every kernel discussion here relies on.
- [Linux and the OS](../foundations/linux-and-os.md) - processes, virtual memory and paging, which PagedAttention borrows directly.
- [Concurrency](../foundations/concurrency.md) - threads, the Python GIL, asynchronous execution and synchronization, needed for the engine-loop and async-scheduling sections.
- [Networking](../foundations/networking.md) - bandwidth vs latency, RDMA and NICs, needed for disaggregation and expert-parallel communication.
- [Distributed Systems](../foundations/distributed-systems.md) - collectives (all-reduce, all-to-all), load balancing and routing across replicas.
- [Model Architectures](../models/architectures.md) - GQA, MLA and MoE, whose shapes determine KV cache layout and the attention kernels discussed here.

This chapter is about what happens inside a modern LLM serving engine - vLLM, SGLang, TensorRT-LLM and their relatives - at the level of the data structures, the scheduler loop and the kernels. It assumes you have read the [Inference](README.md) chapter for the vocabulary (prefill, decode, TTFT, ITL, continuous batching, speculative decoding) and [Inference Internals I: The Arithmetic of LLM Inference](arithmetic.md) for rooflines, KV cache sizing, quantization and capacity planning. Neither is repeated here.

A note on accuracy: these engines change monthly. Where this chapter describes a specific engine's implementation it describes the design as publicly documented in its source, docs or design blogs around 2025-2026. Flag names and defaults drift - check the version you run. Where something is a recollection rather than a verified fact, it says so.

The running examples use two reference models so that the numbers are reproducible:

| Model | Layers | KV heads | Head dim | KV bytes/token (bf16) |
|-|-|-|-|-|
| Llama-3-8B-like | 32 | 8 | 128 | `2*32*8*128*2 = 131072` = 128 KiB |
| Llama-3-70B-like | 80 | 8 | 128 | `2*80*8*128*2 = 327680` = 320 KiB |

and one reference GPU: H100 SXM, ~989 dense bf16 TFLOPS, ~3.35 TB/s HBM (High Bandwidth Memory: the GPU's main on-package DRAM, where weights and KV cache live). Throughout, a *kernel* means one function launched by the CPU to run on the GPU (a matmul, an attention op), and the *host* means the CPU side. All arithmetic in this chapter was checked with `python3`.



## Request lifecycle

### The pipeline

Every engine has the same five logical stages, regardless of how they are mapped onto processes:

```
HTTP/gRPC  ->  tokenizer  ->  scheduler  ->  model runner  ->  sampler  ->  detokenizer  ->  stream to client
 (parse,       (chat         (pick which    (build input    (logit        (incremental
  validate)     template,     requests and   tensors, run    processors,   UTF-8 safe
                encode)       how many       forward,        top-k/p,      decode, stop
                              tokens each)   CUDA graphs)    sample)       strings)
```

The original vLLM architecture: a centralized scheduler driving workers that each hold a slice of the paged KV cache:

![vLLM system overview](images/vllm-system-overview.png)

([source](https://arxiv.org/abs/2309.06180))

Only the model runner and the sampler run on the GPU. Everything else is CPU work, and in a naive implementation all of it is on the critical path of every decode step.

What the CPU does per step:

1. **Scheduling.** Walk the running and waiting queues, check KV block availability, decide preemptions, allocate new blocks for sequences crossing a block boundary.
2. **Input preparation.** Build `input_ids`, `positions`, `slot_mapping` (the physical KV slot each new token writes to), the block tables, `seq_lens`, `query_start_loc` for ragged batches, and the attention-backend metadata. For a batch of 256 sequences at 8k context and block size 16 the block table alone is `256 * 512` int32 entries.
3. **Sampling bookkeeping.** Per-request temperature, top-p, penalties, seeds, grammar masks.
4. **Output processing.** Append sampled token ids, incremental detokenization, stop-string matching, streaming the delta to the client.

### Why CPU overhead matters at decode

The GPU side of a decode step is short. For the 8B model at bf16 the step is bounded below by reading the weights once:

```
16e9 bytes / 3.35e12 bytes/s = 4.8 ms
```

If the CPU stages above take even 1-2 ms and run serially with the GPU, the GPU is idle for 20-30% of wall time. This is not hypothetical: the vLLM team's V1 re-architecture blog (January 2025) was motivated largely by CPU overhead becoming the bottleneck once kernels got fast. The exact fraction depends on batch size, Python version, sampling params in use and tokenizer, so measure it on your setup - `nsys` (NVIDIA's timeline profiler, recording when each CPU call and GPU kernel ran) with NVTX ranges (named markers you insert in code so they appear on that timeline) around schedule/prepare/forward/sample shows the gaps directly as idle stretches on the GPU timeline.

Long prompts also have a tokenization cost: fast (Rust) tokenizers are linear in input length but still take on the order of milliseconds for 100k-token prompts, and chat-template rendering in Jinja is pure Python. Python's GIL (Global Interpreter Lock: only one thread of a Python process executes Python code at a time) means that if that runs in the same process as the scheduler loop it stalls every running request.

### The fix: separate processes, persistent state, async scheduling

The engines converged on three structural changes.

**Multi-process split.** A process is an OS-level program instance with its own memory and its own GIL, so separate processes truly run in parallel on different CPU cores. vLLM V1 runs an API-server process (HTTP, tokenization, detokenization, output streaming) and an `EngineCore` process (scheduler + model executor) connected via ZeroMQ (a message-passing library for sending bytes between processes), so tokenization of a new 100k-token prompt no longer stalls the busy loop. With tensor parallelism, each GPU has its own worker process and the scheduler broadcasts a compact "scheduler output" (request ids, new token counts, new block ids) rather than full tensors. SGLang has an analogous split: a tokenizer manager, a scheduler process per TP group and a detokenizer manager process. TensorRT-LLM's runtime has historically been C++ (the executor API with in-flight batching), which removes most Python overhead from the loop by construction.

**Persistent batch.** Rebuilding all input tensors from Python lists every step is O(total tokens in batch). vLLM V1's model runner keeps persistent, pre-allocated GPU/CPU buffers for the batch state (token ids, block tables, sampling params) and applies only diffs: requests added, removed, or that received new blocks. The per-step CPU work becomes O(changes) plus a handful of small host-to-device copies (transfers from CPU RAM to GPU memory over PCIe, the CPU-GPU bus).

**Async / overlapped scheduling.** GPU kernel launches are asynchronous: the CPU enqueues work on a *stream* (an ordered GPU work queue) and returns immediately; it only blocks at a *sync*, e.g. when it reads a GPU tensor's value back. The fundamental serialization is that step `n+1` needs the token sampled at step `n`. But the scheduler does not need the token's *value* to plan step `n+1` - it only needs to know that each running request will have exactly one more token (absent spec decode and stop conditions). So the scheduler can plan step `n+1` while the GPU executes step `n`, and the sampled token ids are fed into step `n+1`'s `input_ids` buffer on the GPU directly. SGLang calls this the overlap scheduler ("zero-overhead batch scheduler", introduced in v0.4, late 2024); vLLM V1 added it as `--async-scheduling`; TensorRT-LLM's PyTorch backend has an overlap scheduler too. The cost is one step of lag in detecting termination: a request that hit EOS at step `n` has already been scheduled for step `n+1`, which wastes one slot and its KV write, and must be discarded on output.

```python
# Overlapped engine loop (sketch). GPU work is asynchronous; .tolist() is the only sync.
prev = None
while True:
    sched = scheduler.schedule()             # CPU: uses token *counts*, not values
    fut = runner.execute_async(sched)        # enqueue kernels; next input ids taken from
                                             # prev step's sampled-token GPU buffer
    if prev is not None:
        tokens = prev.sampled.tolist()       # sync on step n-1 while step n runs
        outputs = scheduler.update(prev.sched, tokens)  # stop checks, free finished blocks
        detok_queue.put(outputs)             # another process detokenizes and streams
    prev = fut
```

### Incremental detokenization

Detokenization is not `tokenizer.decode(new_token)`. BPE tokens can be partial UTF-8 byte sequences, and SentencePiece-style tokenizers attach leading spaces in a context-dependent way. Engines keep two offsets per request (vLLM's names: `prefix_offset`, `read_offset`), decode the window `tokens[prefix_offset:]`, and emit only the suffix beyond what `tokens[prefix_offset:read_offset]` decoded to, holding back output while the tail ends in an incomplete UTF-8 sequence (`�`). Stop strings must be matched on the detokenized text, and the engine must hold back up to `max(len(stop_string)) - 1` characters from the stream so it never emits a partial stop string it later has to retract.



## PagedAttention

The [overview](README.md#paged-attention) says what it is. This section is about how it works. The name comes from OS virtual memory, where a process sees contiguous addresses but the OS maps them in fixed-size *pages* onto scattered physical memory through a page table; here the "pages" are KV blocks and the page table is the block table. Reference: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (arXiv 2309.06180).

### Block tables

The KV cache is one large pre-allocated pool per layer, carved into fixed-size physical blocks of `block_size` tokens. Each sequence owns a block table: a list of physical block ids, one per `block_size` logical tokens. Token at logical position `t` of sequence `s` lives at:

```python
blk  = block_table[s, t // block_size]      # physical block id
slot = blk * block_size + t % block_size    # flat slot in the pool
```

The vLLM launch post animates how logical KV blocks map to non-contiguous physical blocks:

![PagedAttention block table](images/vllm-paged-attention-block-table.gif)

([source](https://blog.vllm.ai/2023/06/20/vllm.html))

The pool is allocated once at startup: the engine runs a profiling forward pass at max batched tokens, measures peak activation memory, and gives everything left under `gpu_memory_utilization` to KV blocks. For the 8B model on an 80 GB GPU at 0.9 utilization:

```
(80*0.9 - 16) GB = 56 GB of KV
56e9 / 131072 bytes/token = ~427k tokens = ~26.7k blocks of 16
```

Writing new KV: the model runner passes `slot_mapping` (one slot per new token, shape `(num_new_tokens,)`) and a `reshape_and_cache` kernel scatters the freshly computed K and V into those slots before or inside attention.

Layout matters for the attention kernel's access pattern. The original vLLM CUDA paged-attention kernel used a key cache of shape `(num_blocks, num_kv_heads, head_size // x, block_size, x)` with `x = 16 bytes / elem_size`, and values `(num_blocks, num_kv_heads, head_size, block_size)` - arranged so that a warp (a group of 32 GPU threads that execute in lockstep) reading one block reads contiguous 16-byte vectors - "coalesced" access, where neighbouring threads touch neighbouring addresses so the hardware can merge them into few wide memory transactions. FlashAttention/FlashInfer-backed paths use `(num_blocks, block_size, num_kv_heads, head_size)` per K and V (often stacked as a leading dimension of 2), which keeps `head_size` contiguous for tensor-core tiles. The exact layouts are backend-specific and have changed over versions; the invariant is that within a block the memory is contiguous, and between blocks there is one indirection through the block table.

### The kernel's memory access pattern

Decode attention for one `(sequence, kv_head)` pair reads `seq_len * head_dim * 2` elements of K and V once and does `O(seq_len * head_dim)` FLOPs per query head - arithmetic intensity of a few FLOPs per byte, so it is purely bandwidth-bound (see [Internals I](arithmetic.md) for why). The paged kernel therefore has one job: stream KV at near-peak bandwidth despite the indirection. It does that by:

- loading the block table entries for its sequence once (into registers/shared memory - the small, fast on-chip storage private to a thread or shared by a thread block),
- issuing loads one whole block at a time, so each physical block is a contiguous `block_size * head_dim * 2` byte region (4 KiB per K or V block for `block_size=16, head_dim=128`, bf16),
- prefetching the next block while computing on the current one (software pipelining via `cp.async` or, on Hopper, TMA - hardware units that copy tiles from HBM to shared memory asynchronously).

Because blocks are big enough to fill a DRAM burst (the minimum contiguous chunk memory delivers efficiently) and a warp's loads, the indirection costs little. The overhead shows up only when blocks get very small (block size 1, as in token-granular radix caches) where every token's address is a separate gather - FlashInfer's page-size-1 path handles this but with lower achieved bandwidth than large pages, by an amount that depends on the GPU and head dim.

### Block size tradeoffs

Internal fragmentation is only in the last, partially filled block. If sequence lengths are uniformly distributed modulo `block_size`, the expected waste is `(B - 1) / 2` tokens per sequence:

| Block size `B` | Mean waste (tokens/seq) | Waste fraction at `L=500` | at `L=2000` |
|-|-|-|-|
| 8 | 3.5 | 0.7% | 0.17% |
| 16 | 7.5 | 1.5% | 0.37% |
| 32 | 15.5 | 3.0% | 0.77% |
| 64 | 31.5 | 5.9% | 1.6% |
| 128 | 63.5 | 11.3% | 3.1% |
| 256 | 127.5 | 20.3% | 6.0% |

Compare with static preallocation to `max_model_len`: a request reserving 8192 slots that ends at 1500 tokens uses `1500/8192 = 18%` of its reservation. That gap - not the few percent above - is what PagedAttention buys you.

Why not block size 1? Costs that grow as blocks shrink:

- **Block table size and CPU bookkeeping.** A 128k-token sequence at `B=16` is 8192 block ids; at `B=1` it is 131072. Allocation, freeing and hashing are per block.
- **Kernel efficiency.** Discussed above.
- **Prefix caching granularity.** Hash-based caching (below) only caches full blocks, so smaller blocks give finer-grained reuse but more hash computations.

Why not block size 256? Fragmentation, and coarser copy-on-write and prefix sharing. In practice vLLM defaults to 16 on CUDA; some kernels require specific sizes (FlashMLA uses 64; some attention backends require multiples of 16 or 32). SGLang's radix cache historically used page size 1 (token-granular) and later made `page_size` configurable for kernels that prefer larger pages.

### Sharing and copy-on-write

Blocks carry a reference count. This gives sharing for free:

- **Parallel sampling (`n > 1`).** All `n` children point at the same prompt blocks (`ref_count = n`). Each child's first generated token goes into the last prompt block if it is partially full; on write to a block with `ref_count > 1`, the allocator allocates a fresh block, copies the existing slots, decrements the original's count and updates that child's table. Only the partially filled last block is ever copied; full shared blocks are immutable.
- **Beam search.** Forking a beam is copying a block table and incrementing ref counts; pruning a beam decrements them. The copy cost per fork is bounded by one block.

Copy-on-write for parallel sampling, from the same post:

![PagedAttention copy-on-write for parallel sampling](images/vllm-paged-attention-cow.gif)

([source](https://blog.vllm.ai/2023/06/20/vllm.html))

vLLM V1 moved away from an explicit CoW path in the core: parallel sampling is fanned out into independent child requests that share prompt blocks through prefix caching (next section), and beam search was moved out of the engine core into a higher-level loop. The effect is the same - shared immutable full blocks - with one fewer mechanism in the core. (This reflects the V1 design as I understand it; verify against your version if you depend on it.)

```python
class BlockAllocator:
    def __init__(self, n): self.free = deque(range(n)); self.ref = [0]*n
    def alloc(self):       b = self.free.popleft(); self.ref[b] = 1; return b
    def fork(self, table): [self._inc(b) for b in table]; return list(table)
    def _inc(self, b):     self.ref[b] += 1
    def release(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0: self.free.append(b)
    def writable_last(self, table, kv_pool):       # call before appending a token
        b = table[-1]
        if self.ref[b] > 1:                        # copy-on-write
            nb = self.alloc(); kv_pool[:, nb] = kv_pool[:, b]   # (layers, block, ...)
            self.release(b); table[-1] = nb
        return table[-1]
```



## Continuous batching and the scheduler

The [overview](README.md#continuous-batching-or-in-flight-batching) describes the idea. The original system is Orca (Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022 - not on arXiv as far as I know), which introduced iteration-level scheduling and *selective batching*: batch the token-wise ops (linear layers, norms) across requests of different lengths by flattening tokens, and run attention per request. That flattening is still how every engine works: the model sees a single ragged batch of shape `(total_tokens, hidden)` with no padding, and only attention needs per-sequence metadata.

### The unified scheduler

Modern schedulers no longer distinguish "prefill steps" from "decode steps". vLLM V1's scheduler, for example, produces a mapping `{request_id: num_tokens_to_compute_this_step}` under two budgets: `max_num_batched_tokens` (token budget per step) and `max_num_seqs`. A decode contributes 1 token (or `1 + k` with speculation), a prefill chunk contributes up to whatever budget is left. A request is just "computed tokens so far" vs "tokens known so far" - prefill, chunked prefill, decode, spec-decode verification and resumption after preemption are all the same thing: catching `num_computed_tokens` up to `num_tokens`.

```python
def schedule(running, waiting, kv, token_budget, max_seqs):
    out = {}
    # 1. running requests first (decodes and in-progress chunked prefills)
    for r in list(running):
        n = min(r.num_tokens - r.num_computed, token_budget)
        while not kv.can_append(r, n):          # need new blocks for these tokens
            victim = running.pop()              # lowest priority / latest arrival
            kv.free(victim); victim.num_computed = 0   # recompute-style preemption
            waiting.appendleft(victim)
            if victim is r: break
        else:
            kv.append_slots(r, n); out[r.id] = n; token_budget -= n
        if token_budget == 0: return out
    # 2. admit waiting requests if KV and budget allow
    while waiting and token_budget > 0 and len(out) < max_seqs:
        r = waiting[0]
        hit = kv.lookup_prefix(r)               # prefix-cache hit, in tokens
        n = min(r.num_tokens - hit, token_budget)
        if not kv.can_allocate(r, hit + n, watermark=0.01):
            break                               # admission control: KV budget
        waiting.popleft(); r.num_computed = hit
        kv.allocate(r, hit + n); running.append(r)
        out[r.id] = n; token_budget -= n
    return out
```

### Policies

- **FCFS** is the default in most engines. Preemption victims are taken from the *back* of the running queue (most recently admitted), so older requests make progress - otherwise you get livelock (the system stays busy but makes no progress) where two long requests repeatedly evict each other.
- **Priority scheduling** (vLLM `--scheduling-policy priority`, per-request `priority`) orders both admission and preemption by priority, then arrival.
- **Cache-aware ordering.** SGLang can reorder the waiting queue by longest prefix match (LPM) so requests sharing a cached prefix run back to back before it is evicted - the SGLang paper shows this approximates a DFS over the radix tree. Pure LPM can starve requests with no cache hit; it degrades to FCFS when the queue is short and has a fallback when it is long.
- **Fairness across tenants** (e.g. virtual token counters as in "Fairness in Serving Large Language Models", Sheng et al., arXiv 2401.00588 (id unverified)) is mostly not in open-source engines' cores; it is done in routers or gateways.

### Admission control by KV budget

The scheduler admits a request only if the pool has enough free blocks for the tokens it will compute this step (plus a small watermark reserve so running decodes do not immediately trigger preemptions). It does *not* reserve blocks for the full eventual output: output length is unknown, and reserving `max_tokens` worth would recreate the static-allocation waste above. This is an explicit bet that not every running sequence grows to its max at once; when the bet loses, the scheduler preempts.

Practical consequence: `kv_cache_usage` near 100% with a non-zero preemption rate means you are oversubscribed; reduce `max_num_seqs` or add capacity. High usage with zero preemption is fine - it is what you want.

### Preemption: swap vs recompute

When a running sequence needs a block and none is free, a victim is preempted. Two options:

- **Swap**: copy the victim's KV blocks to pinned CPU memory (host RAM the OS is told never to page out, which the GPU can DMA - copy directly without CPU involvement) and back later.
- **Recompute**: drop the victim's KV and later re-prefill prompt + generated-so-far.

Cost for the 8B model, a 2000-token victim:

```
KV size         = 2000 * 131072 B = 262 MB
swap one way    = 262 MB / ~25-50 GB/s (PCIe Gen5 x16, effective) = 5-10 ms, x2 for out+in
recompute       = 2 * 8e9 * 2000 FLOPs / (989e12 * 0.5) = ~65 ms of GPU compute
```

Swap looks cheaper per sequence, but recompute won in practice for several reasons: the recompute is a prefill that batches with other work at high arithmetic intensity, so it uses otherwise idle compute; swap needs CPU memory, synchronized copies, and a second block allocator; and with prefix caching, the dropped blocks stay in the hash table until actually evicted, so "recompute" is often a cache hit and costs nearly nothing. vLLM V1 dropped swap preemption and uses recompute only (V0 supported both). The case where swap would still win is very long sequences on compute-saturated GPUs - and the modern answer there is a tiered KV cache offload (CPU/SSD, e.g. LMCache or Mooncake's store) rather than per-request swap.



## Chunked prefill and stall-free batching

### The stall

Without chunking, a new request's prefill runs whole. If that step also contains decodes (or if prefill-prioritizing schedulers pause decodes entirely, as vLLM V0 did by default), every running request's ITL for that step becomes the prefill latency. For the 8B model at 50% MFU (Model FLOPs Utilization: achieved FLOPs as a fraction of the hardware peak):

| Tokens in step | Compute time |
|-|-|
| 256 | 8.3 ms |
| 512 | 16.6 ms |
| 2048 | 66 ms |
| 8192 | 265 ms |

So with a ~5-10 ms decode step, one 8k-token arrival puts a ~265 ms hiccup into every other user's stream. Under steady arrival this is what dominates p99 ITL.

### Sarathi and Sarathi-Serve

Sarathi (Agrawal et al., arXiv 2308.16369) proposed splitting prefills into chunks and piggybacking decodes onto each chunk ("decode-maximal batching"). Sarathi-Serve (Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", arXiv 2403.02310) made it the scheduling policy: pick a **token budget** per step from the ITL SLO, then fill each step with all running decodes first and use the remainder for prefill chunks. No step exceeds the budget, so no decode stalls - "stall-free batching".

The key observation is the roofline: a decode-only step is memory-bound, so adding a few hundred prefill tokens to it costs little extra time because the weights are already being read. Until the step's total tokens reach roughly the ridge point - the arithmetic intensity at which a kernel stops being limited by memory bandwidth and starts being limited by compute (for bf16 GEMMs on H100, `989/3.35 ≈ 295` tokens - see [Internals I](arithmetic.md)), prefill tokens ride almost free. Beyond that, the step becomes compute-bound and its time grows linearly with token count.

Choosing the budget is a direct ITL/TTFT trade:

- small budget (e.g. 256-512): tight ITL, but long prompts take many steps, so TTFT grows; per-chunk overhead (re-reading the prompt's earlier KV in each chunk's attention, kernel launches) adds up;
- large budget (e.g. 4096-8192): better prefill efficiency and TTFT, worse ITL tail.

vLLM V1 enables chunked prefill always; the knob is `max_num_batched_tokens`. SGLang's is `--chunked-prefill-size`. TensorRT-LLM supports chunked context as well.

Chunked attention for the *n*-th chunk attends to all previous chunks' KV, which now lives in the paged cache. So chunk `i` of an `L`-token prompt with chunk size `c` reads `i*c` cached tokens: total KV read over the prompt is `O(L^2 / c)`, versus `O(L)` for a single unchunked pass. With FlashAttention-style kernels this is still far below compute cost for typical `L` and `c`, but it is why very small chunks hurt long prompts.

Kernel-side, mixing prefill and decode in one attention launch is awkward: they want different tilings. Options are (a) one varlen kernel for everything (FlashAttention varlen, FlashInfer with a scheduler that partitions work), (b) launching prefill and decode attention separately within the step, or (c) a fused kernel that co-schedules both on the same SMs (POD-Attention, Kamath et al., arXiv 2410.18038 (id unverified)).



## Prefix caching

When many requests share a prefix (system prompt, few-shot examples, multi-turn history, agent scaffolds), its KV can be reused. For a 4096-token system prompt and 500-token user messages, prefix caching skips `4096/4596 = 89%` of prefill compute. Two designs dominate.

### Hash-based block caching (vLLM APC)

Each *full* block gets a hash that commits to its entire prefix:

```python
h_0 = H(None,  tuple(tokens[0:16]),  extra_keys)
h_i = H(h_{i-1}, tuple(tokens[16*i:16*(i+1)]), extra_keys)
```

Mooncake's KV pool uses the same idea - each block's key hashes its own tokens and its prefix's hash:

![KV blocks keyed by chained hashes](images/mooncake-kv-block-hash-chain.png)

([source](https://arxiv.org/abs/2407.00079))

`extra_keys` carry anything that changes KV for identical token ids: LoRA adapter id, multimodal input hashes (image placeholder tokens are identical across images), and an optional per-tenant cache salt for isolation. A global dict maps `hash -> physical block`. On admission the scheduler walks the request's block hashes until the first miss; the hit blocks get `ref_count += 1` and the request starts with `num_computed_tokens = hits * block_size`.

Eviction: blocks with `ref_count == 0` are not freed but placed in a free queue *while keeping their hash*. Allocation takes from the queue head; a block taken from the queue that still has a hash is evicted (hash removed). Freed blocks are appended so that least recently used goes first; vLLM V1 also appends a finished request's blocks in reverse order so the tail of a prefix (least likely to be shared) is evicted before its head. Everything is O(1) per block with a doubly linked list, which is why V1 enables prefix caching by default - overhead at 0% hit rate is small.

Two details that bite:

- **Hash collisions.** Early versions used Python's builtin `hash`, so a collision silently serves another prompt's KV - a correctness and cross-tenant leak risk. Current vLLM offers SHA-256-based hashing (`--prefix-caching-hash-algo`); use it in multi-tenant deployments.
- **Last token is always recomputed.** Even with a full hit, the engine must compute at least one token to produce logits, so a 100% match still recomputes the final token (and therefore, with block granularity, possibly the last block).

### RadixAttention (SGLang)

SGLang (Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv 2312.07104) keeps a radix tree whose edges are token sequences and whose nodes hold the KV indices for those tokens. Lookup is longest-prefix match on the tree; insertion splits edges. Nodes in use by running requests are locked via ref counts; eviction is LRU over unlocked leaves, recursively, so a shared prefix is evicted only after all its extensions.

Differences from hash blocks:

| | Hash blocks (vLLM) | Radix tree (SGLang) |
|-|-|-|
| Granularity | full blocks only | token (page size 1) or page |
| Lookup | `O(L / B)` dict lookups | `O(L)` tree walk |
| Structure | flat; prefix relation implicit in chained hash | explicit tree; easy to compute "what is cached under this node" |
| Eviction | LRU over free blocks, tail-first | LRU over leaves |
| Scheduling integration | hit length used for admission | LPM ordering of waiting queue, tree used by router |

In practice both get similar hit rates on chat traffic; the radix tree's advantages are partial-block reuse and that the tree is a convenient structure for cache-aware scheduling.

### Cache-aware routing across replicas

Prefix caching is per replica (an independent copy of the full serving engine on its own GPUs). With N data-parallel replicas behind a round-robin load balancer (a front-end that hands requests to replicas in turn), each replica sees 1/N of the conversations, so multi-turn hit rate is roughly that of a single replica serving the same total traffic *only if* each conversation sticks to one replica. Routing options:

- **Session affinity** (hash on conversation id): simple, breaks under load imbalance.
- **Prefix-aware routing**: the router maintains an approximate prefix tree per replica (from the requests it has routed, without asking the replicas) and sends each request to the replica with the longest match, unless that replica's load exceeds the least-loaded one's by a threshold - then it falls back to load balancing. SGLang's router implements this; vLLM's production stack, llm-d and NVIDIA Dynamo implement variants that also use KV events published by the engines. Mooncake goes further and treats KV as a cluster-wide resource in a disaggregated DRAM/SSD pool (arXiv 2407.00079).

The trade is load balance vs hit rate. Hot prefixes (one system prompt for all traffic) are best replicated everywhere; long-tail conversation histories are best sharded with affinity.



## Inference attention kernels

### Why training kernels are wrong for decode

FlashAttention (Dao et al., arXiv 2205.14135; FA2 arXiv 2307.08691; FA3 for Hopper arXiv 2407.08608) parallelizes over `(batch, head, query_block)`. In prefill there are many query blocks. In decode there is one query row per `(sequence, head)`, so the grid is `batch * heads` CTAs (a CTA, or thread block, is the unit of work scheduled onto one SM - Streaming Multiprocessor, one of the GPU's ~100+ independent compute cores). With batch 4 and 8 KV heads that is 32 CTAs on a 132-SM H100 - most SMs idle, and each CTA serially walks the whole KV.

### Split-KV (FlashDecoding)

FlashDecoding (Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference", blog, October 2023) adds a parallel dimension over the KV sequence: split KV into `S` chunks, each CTA computes an unnormalized partial output and its log-sum-exp, then a small reduction kernel merges them:

```python
# partials from S splits for one (seq, head): o_s (S, D), lse_s (S,)
lse = torch.logsumexp(lse_s, dim=0)            # ()
o   = (torch.exp(lse_s - lse)[:, None] * o_s).sum(0)   # (D,)
```

The number of splits is chosen per launch from batch size, heads, KV length and SM count; that choice is a *shape-dependent reduction order*, which matters for determinism (below). FlashDecoding++ (Hong et al., arXiv 2311.01282) is a follow-up with a unified-max trick to avoid synchronized rescaling (id from memory, believed correct).

### GQA packing

With GQA, `group = num_q_heads / num_kv_heads` query heads read the same KV. Decode kernels pack the group into the query tile's M dimension: one CTA loads a KV tile once and computes against `group` query rows (8 for Llama-3), which both cuts KV reads by `group` and makes the QK^T small GEMM tall enough to use tensor cores (the matrix-multiply units that provide nearly all of the GPU's FLOPs, but only on tiles of at least ~16 rows). With speculative decoding the `k+1` verification tokens get packed in the same way, giving `group * (k+1)` rows.

### FlashInfer

FlashInfer (Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving", arXiv 2501.01005) is the kernel library behind SGLang and one of vLLM's backends. Ideas worth knowing:

- **Block-sparse/paged KV as a unified format.** Paged KV, ragged KV and tree masks are all expressed as block-sparse matrices (BSR), so one kernel family handles every layout.
- **Plan/run split.** A CPU-side `plan()` computes work partitioning (how to split KV across CTAs for load balance across ragged sequence lengths) once per batch shape; `run()` is then CUDA-graph-capturable because it only reads the plan from fixed buffers. This is how dynamic, load-balanced split-KV is reconciled with static graphs.
- **JIT templates** (kernels compiled Just-In-Time at runtime for the exact variant requested) for attention variants (custom masks, logits soft-cap, sliding window, sink tokens) instead of hand-written kernels per variant.

### Ragged batching

The model sees a flattened `(total_tokens, hidden)` batch. Attention consumes `query_start_loc` (cumulative query lengths, `(B+1,)`), `seq_lens` (`(B,)` total KV lengths including the new tokens), and block tables (`(B, max_blocks)`). Each sequence's queries attend causally to its own KV only; there is no padding anywhere except in CUDA-graph batch-size padding (below).

### Cascade attention for shared prefixes

When many sequences in a batch share a long prefix (one system prompt for a whole batch, or `n=64` samples of one prompt), decode attention re-reads that prefix's KV once per sequence. Cascade attention splits each sequence's attention into two parts:

1. **Shared prefix**: all `B` queries attend to the shared KV in *one* multi-query pass. This is prefill-shaped (many query rows, one KV), high arithmetic intensity, and reads the prefix once.
2. **Unique suffix**: each sequence attends to its own KV with a normal decode kernel.

Then merge the two partial results with the same log-sum-exp formula as split-KV:

```python
def merge(o1, lse1, o2, lse2):        # o: (B, H, D), lse: (B, H)
    lse = torch.logaddexp(lse1, lse2)
    return o1 * torch.exp(lse1 - lse)[..., None] + o2 * torch.exp(lse2 - lse)[..., None], lse
```

KV bytes read drop from `B * (P + U)` to `P + B * U` tokens. For `B=64`, `P=4096`, `U=256`: from 278,528 to 20,480 token-reads, a 13.6x reduction in attention memory traffic (attention, not the whole step - weights still dominate at moderate context). FlashInfer ships cascade kernels; Hydragen (Juravsky et al., arXiv 2402.05099) is the paper-form treatment; vLLM V1 applies cascade attention automatically when a batch's common prefix is long enough to be worth it, using a heuristic on prefix length and batch size.

### MLA kernels

DeepSeek's Multi-head Latent Attention (DeepSeek-V2, arXiv 2405.04434) caches a compressed latent `c_kv` (dim 512 in V3) plus a decoupled RoPE key (dim 64) per token per layer, shared across all heads:

```
DeepSeek-V3: (512 + 64) * 61 layers * 2 bytes = 70,272 B/token = ~68.6 KiB
```

At decode, the up-projection `W_UK` is absorbed into the query (`q' = q W_UK^T`, per head) and `W_UV` into the output projection, so attention runs directly against the latent: effectively MQA with 128 query heads, one "KV head" of dim 576 for keys and 512 for values. That is a very different kernel shape: 128 query rows per sequence against a single KV stream, so decode attention is *much* more compute-dense than GQA decode. FlashMLA (DeepSeek, open-sourced February 2025) is the reference Hopper kernel (paged, block size 64); FlashInfer and TRT-LLM have their own. For prefill, engines typically do the opposite: decompress K/V and run standard MHA, because absorbed form does more FLOPs when there are many queries. The single latent head also breaks TP head-sharding of the KV cache - see [DP-attention](#dp-attention-and-expert-parallelism-for-moe).



## CUDA graphs

### Why decode needs them

A decode step launches several kernels per layer (norms, QKV GEMM, RoPE, cache write, attention, output proj, MLP GEMMs, activation, residual adds, plus all-reduces under TP - an all-reduce is a collective where every GPU contributes a tensor and every GPU ends up with the elementwise sum). Assuming ~12 launches per layer and ~5 us per launch from Python/PyTorch (order of magnitude; actual eager overhead varies widely), the 8B model's 32 layers give:

```
32 * 12 * 5e-6 s = ~1.9 ms of launch overhead vs ~4.8 ms of GPU work
```

Worse, at small batch the kernels themselves take a few microseconds, so the GPU drains the launch queue and waits on the CPU. A CUDA graph records the whole step's kernel sequence once (with all kernel arguments, including memory addresses) and replays it with one launch.

### Constraints and how engines satisfy them

A captured graph replays fixed kernels with fixed arguments on fixed memory addresses. So:

- **Static input buffers.** `input_ids`, `positions`, `slot_mapping`, block tables, `seq_lens` live in persistent buffers sized for the maximum; each step copies the real values in before replay.
- **Static shapes -> batch-size buckets.** One graph per batch size bucket. vLLM's default capture list is roughly `[1, 2, 4] + list(range(8, max, 8))` up to `max_cudagraph_capture_size` (512 in recent defaults - check yours). A batch of `n` runs in the smallest bucket `>= n`, padded. Padded slots point at a dummy slot / null block so their KV writes go nowhere harmful.
- **No host syncs, no dynamic control flow**, no allocations inside the region (the graph uses a private memory pool - a reserved region the allocator hands out from - and graphs share a pool so total extra memory is the max, not the sum).
- **Attention metadata must be graph-friendly**, which is why FlashInfer has the plan/run split and why decode-only batches are much easier to capture than mixed prefill/decode ones.

Padding waste with that bucket list, averaged over batch sizes 1-512: 2.8%; worst case 7/16 = 44% at `n=9` (padded to 16). It is a small-batch effect only.

Each captured size costs capture time at startup (seconds to tens of seconds total) and some memory; that is the knob behind SGLang's `--cuda-graph-max-bs` and vLLM's capture-size settings. Above the largest captured size, engines fall back to eager - at large batch the kernels are long enough that launch overhead stops mattering.

### Interaction with torch.compile

vLLM V1 compiles the model with `torch.compile` (custom Inductor passes for fusions such as RMSNorm+quant, SiLU-mul+quant, all-reduce+norm) and then captures CUDA graphs. Its default is *piecewise* graphs: the FX graph is split at attention ops, the non-attention pieces are compiled and captured per batch-size bucket, and attention runs outside the graph so it can handle arbitrary mixed prefill/decode metadata. Later versions added a mode that captures *full* graphs for uniform decode-only batches (when the attention backend supports capture) and piecewise graphs for mixed batches (`cudagraph_mode`, e.g. `FULL_AND_PIECEWISE`; names from memory - verify). Compilation is done with dynamic batch dimension once, then specialized per captured size; compiled artifacts are cached on disk so restarts are faster.

Practical gotchas: a model change that introduces a graph break or host sync silently drops you to slow paths; dynamic per-request logit processors or Python-side sampling cannot be in the graph; LoRA, spec decoding and structured output each need explicit graph support (fixed max adapters, fixed draft length, mask applied as a captured kernel on a static bitmask buffer).



## Speculative decoding

The [overview](README.md#speculative-decoding) has the intuition. Here: the math, the variants, and the serving-system consequences.

Each line is one iteration: green tokens are draft proposals the target accepted, red the rejected one, blue the target's correction:

![speculative decoding iterations](images/spec-decoding-leviathan.png)

([source](https://arxiv.org/abs/2211.17192))

### Correctness: rejection sampling

Target distribution `p(.)`, draft distribution `q(.)` at a position. Sample `x ~ q`. Accept with probability `min(1, p(x)/q(x))`. If rejected, sample from the residual `r(x) = max(0, p(x) - q(x)) / Z` where `Z = sum_x max(0, p(x) - q(x))`.

Proof sketch that the output is exactly `p`. Let `beta = sum_x min(p(x), q(x))` be the total acceptance probability. Then

```
P(output = x) = q(x) * min(1, p(x)/q(x))         + (1 - beta) * r(x)
              = min(p(x), q(x))                  + (1 - beta) * max(0, p(x) - q(x)) / Z
```

and `Z = sum_x (p(x) - min(p(x), q(x))) = 1 - beta`, so the second term is `max(0, p(x) - q(x))` and the sum is `min(p, q) + max(0, p - q) = p(x)`. Applied position by position on the draft's tokens - each conditioned on the accepted prefix - this gives exact samples from the target. The first rejection terminates the step; if all `k` drafts are accepted, one more "bonus" token is sampled from the target's distribution at position `k+1`, which the verify pass computed anyway. References: Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (arXiv 2211.17192); Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv 2302.01318).

For greedy decoding this degenerates to "accept while `argmax p == draft token`".

```python
def verify(draft_tok, q, p, gen):
    # draft_tok: (B, k) proposed ids; q: (B, k, V) draft probs; p: (B, k+1, V) target probs
    B, k = draft_tok.shape
    p_d = p[:, :k].gather(-1, draft_tok[..., None]).squeeze(-1)    # (B, k)
    q_d = q.gather(-1, draft_tok[..., None]).squeeze(-1)            # (B, k)
    u = torch.rand(B, k, generator=gen, device=p.device)
    accept = u < (p_d / q_d).clamp(max=1.0)                         # (B, k)
    n_acc = accept.int().cumprod(-1).sum(-1)                        # (B,) accepted prefix length
    # next token: residual at first rejection, or bonus from p[:, k] if all accepted
    idx = n_acc.clamp(max=k - 1)
    resid = (p[torch.arange(B), idx] - q[torch.arange(B), idx]).clamp(min=0)   # (B, V)
    nxt_dist = torch.where((n_acc == k)[:, None], p[:, k], resid)
    nxt = torch.multinomial(nxt_dist / nxt_dist.sum(-1, keepdim=True), 1)      # (B, 1)
    return n_acc, nxt   # emit draft_tok[:, :n_acc] then nxt -> n_acc + 1 tokens
```

(Real implementations are fused GPU kernels that avoid materializing `(B, k, V)` probability tensors where possible; FlashInfer and vLLM have them.)

### Expected tokens per step

Model acceptance at each draft position as i.i.d. with probability `alpha` (a simplification: real acceptance decays with position and depends on content). The number of accepted drafts `N` satisfies `P(N >= i) = alpha^i` for `i <= k`. Tokens emitted per verification step = `N + 1`:

```
E[N + 1] = sum_{i=0}^{k} alpha^i = (1 - alpha^(k+1)) / (1 - alpha)
```

| alpha \ k | 1 | 2 | 3 | 4 | 6 | 8 |
|-|-|-|-|-|-|-|
| 0.5 | 1.50 | 1.75 | 1.88 | 1.94 | 1.98 | 2.00 |
| 0.6 | 1.60 | 1.96 | 2.18 | 2.31 | 2.43 | 2.47 |
| 0.7 | 1.70 | 2.19 | 2.53 | 2.77 | 3.06 | 3.20 |
| 0.8 | 1.80 | 2.44 | 2.95 | 3.36 | 3.95 | 4.33 |
| 0.9 | 1.90 | 2.71 | 3.44 | 4.10 | 5.22 | 6.13 |

The asymptote is `1/(1 - alpha)`; returns diminish fast beyond `k ≈ 1/(1-alpha)`.

Wall-clock speedup (Leviathan et al.) with draft step cost `c` relative to a target step, assuming the verify pass costs the same as one target decode step:

```
speedup = (1 - alpha^(k+1)) / ((1 - alpha) * (k*c + 1))
```

At `alpha = 0.8`:

| k | c = 0.05 | c = 0.1 | c = 0.2 |
|-|-|-|-|
| 1 | 1.71 | 1.64 | 1.50 |
| 2 | 2.22 | 2.03 | 1.74 |
| 4 | 2.80 | 2.40 | 1.87 |
| 6 | 3.04 | 2.47 | 1.80 |
| 8 | 3.09 | 2.40 | 1.66 |

So the optimal `k` shrinks as the draft gets more expensive.

### When speculation hurts at high batch

The "verify costs one decode step" assumption holds only while the step is memory-bound. A verify step processes `B * (k+1)` tokens. Once `B * (k+1)` crosses the GEMM ridge point (~295 tokens for bf16 on H100), the step becomes compute-bound and its time grows roughly linearly with tokens. In the fully compute-bound limit the verify costs `~(k+1)` decode steps:

```
alpha=0.8, k=4, c=0.05, verify cost=5:  3.36 / (5 + 4*0.05) = 0.65x   -> a slowdown
```

Equivalently, the token efficiency is `E[N+1]/(k+1) = 3.36/5 = 67%`: a third of verified tokens are wasted compute, which only matters when compute is the bottleneck. So at `B=64, k=4` you are already at `320` tokens per step - near the ridge. Rules of thumb that follow from the arithmetic:

- Speculation helps latency at low/medium batch; at high batch it can reduce throughput.
- Engines should adapt `k` (or disable speculation) with batch size and observed acceptance. Several papers study this ("Optimizing Speculative Decoding for Serving Large Language Models Using Goodput", Liu et al., arXiv 2406.14066 (id unverified)). For *long-context* high-batch decode, the KV read dominates and the step stays memory-bound longer, which is the regime where MagicDec (Chen et al., arXiv 2408.11049 (id unverified)) argues speculation helps even at large batch.
- MoE targets complicate this: verifying more tokens activates more distinct experts, increasing weight reads, so the "free" verify assumption breaks earlier.

### Draft variants

| Method | Draft source | Cost `c` | Notes |
|-|-|-|-|
| Separate draft model | small LM, same tokenizer | 0.05-0.2 | needs its own KV cache and scheduling; drafts autoregressively |
| n-gram / prompt lookup | match last n tokens in the context, copy what followed | ~0 (CPU) | great for code editing, RAG, summarization; useless for open-ended generation. ([prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding); suffix-tree variants such as SuffixDecoding, arXiv 2411.04975 (id unverified)) |
| Medusa | extra decoding heads on the last hidden state, each predicting position `t+i` | small | heads predict independently, so tree candidates + tree attention are used to recover joint plausibility. Cai et al., arXiv 2401.10774 |
| EAGLE | a one-layer autoregressive head over the target's top-layer *features* plus shifted token embeddings | small | features are more predictable than tokens; EAGLE (Li et al., arXiv 2401.15077), EAGLE-2 dynamic draft trees using draft confidence (arXiv 2406.16858), EAGLE-3 fuses low/mid/high-layer features and trains with on-policy "training-time test" (arXiv 2503.01840) |
| MTP heads | multi-token-prediction modules trained with the model | small | Gloeckle et al., arXiv 2404.19737; DeepSeek-V3 (arXiv 2412.19437) trains one sequential MTP module and reports 85-90% acceptance for the second token, ~1.8x TPS when used for speculation (numbers as reported in the paper) |
| Lookahead decoding | Jacobi iteration n-grams generated by the target itself | none | Fu et al., arXiv 2402.02057 |

EAGLE drafts at the feature level, feeding the target model's top-layer features plus the shifted token embedding into a one-layer draft head:

![EAGLE drafting pipeline](images/eagle-pipeline.png)

([source](https://arxiv.org/abs/2401.15077))

Tree speculation (SpecInfer, Miao et al., arXiv 2305.09781; Medusa; EAGLE-2) verifies a tree of candidates in one pass using a tree attention mask; it raises expected accepted length per step at the cost of more verified tokens per step - which, per the previous subsection, is exactly the wrong trade at high batch.

### Engine mechanics

- **KV for rejected tokens.** Verification writes KV for all `k+1` positions. Rejected positions are rolled back by not advancing `num_computed_tokens`; their slots get overwritten next step. Blocks allocated only for rejected tokens are freed. For hash-based prefix caching, only blocks consisting entirely of accepted tokens may be hashed.
- **Variable query lengths.** Verify is a mini-prefill of `k+1` tokens per sequence - the attention kernel needs multi-token decode support (and tree masks for tree methods). For CUDA graphs, `k` must be fixed per captured graph.
- **Draft KV.** Draft models and EAGLE heads have their own (small) KV cache that also needs paging and must stay in sync on rejection.
- **Sampling parameters** must match: temperature, top-p etc. are applied to both `p` and `q` before acceptance, or the guarantee holds for the wrong distribution. Structured-output masks must be applied to the draft too, or acceptance collapses.



## Serving-time parallelism

Training-side background is in [Model Parallelism](../training/model-parallelism/README.md) and [the overview's notes](README.md#model-parallelism). Inference differs because the decode step is tiny and latency-sensitive.

### Tensor parallelism

Megatron-style TP (Shoeybi et al., arXiv 1909.08053) needs two all-reduces per layer (after attention output projection and after the MLP down-projection). At decode the messages are small: `B * hidden * 2` bytes, e.g. `64 * 8192 * 2 = 1 MiB` per all-reduce, 160 of them per step for 80 layers. Small all-reduces are latency-bound (dominated by fixed per-message startup and synchronization cost), not bandwidth-bound, so NCCL's (NVIDIA's collective-communication library) ring algorithm, which passes data around all GPUs in `2*(N-1)` sequential hops, is the wrong choice. Engines use custom one-shot / two-shot all-reduce kernels over NVLink P2P (NVLink is the direct GPU-to-GPU interconnect inside a server, ~900 GB/s total per H100; P2P means one GPU's kernel reads or writes another GPU's memory directly) (vLLM's custom all-reduce, TRT-LLM's, FlashInfer's comm kernels), and fuse all-reduce with the following residual add + RMSNorm (+ quantization) to save a kernel launch and a pass over memory. Across nodes, TP is generally a bad idea for decode for the same latency reason.

TP shards KV heads too: TP=8 on a model with 8 KV heads puts one KV head per GPU. Beyond `num_kv_heads`, KV heads get replicated and KV memory per GPU stops shrinking.

### Pipeline parallelism

PP at inference has no backward pass, so the bubble is only a fill/drain effect - but only if the engine keeps `pp_size` micro-batches in flight. A synchronous scheduler that waits for the last stage before scheduling the next step leaves `(pp-1)/pp` of the pipeline idle. vLLM runs multiple batches concurrently in PP mode (virtual engines in V0; batch queue in V1). PP's per-stage traffic is just one activation tensor per micro-batch, making it the sane way to span nodes for dense models. Cost: per-token latency is the sum of stage latencies plus hops; it improves throughput, not ITL.

### DP-attention and expert parallelism for MoE

For large MoE models like DeepSeek-V3/R1 (671B total, 37B active, 256 routed experts, top-8 + 1 shared expert, hidden 7168), the serving layout that emerged is:

- **Attention: data parallel.** MLA's single latent KV head cannot be sharded by heads; TP would replicate the KV cache on every rank. So each rank runs attention for its own requests with its own KV cache (SGLang `--enable-dp-attention`; vLLM data-parallel + EP deployment).
- **MoE FFN: expert parallel.** Each rank (one GPU's worker process in the group) holds a subset of experts. After attention, tokens are dispatched (all-to-all: a collective where every GPU sends a different chunk of data to every other GPU) to the ranks holding their top-k experts and the results combined (second all-to-all) back.

DeepSeek's published inference-system overview (open-infra-index, February 2025) describes prefill at EP32 with DP32 attention on 4-node units, and decode at EP144 with DP144 attention on 18-node units, with redundant copies of hot experts. Treat these as their reported configuration, not a recommendation.

Across nodes, GPUs talk via RDMA (Remote Direct Memory Access: a network card reads/writes another machine's memory, including GPU memory, without involving either CPU) over InfiniBand or RoCE NICs (network interface cards), typically one 400 Gb/s NIC per GPU, i.e. ~50 GB/s - roughly 18x less than NVLink. Communication arithmetic for decode dispatch/combine, upper bound assuming every expert lands on a remote GPU, fp8 dispatch and bf16 combine:

```
dispatch per token = 7168 * 1 B * 8 experts  = 57,344 B
combine  per token = 7168 * 2 B * 8 experts  = 114,688 B
128 tokens/GPU, 58 MoE layers: (57344 + 114688) * 128 * 58 = 1.28 GB per step per GPU
at ~50 GB/s RDMA per GPU (one 400 Gb/s NIC): ~25.5 ms per step
```

That is far larger than the compute per step, which is why (a) the all-to-all must be overlapped with compute, (b) dispatch is done in fp8, and (c) "wide EP" spreads experts over many GPUs so each GPU holds few experts and has enough tokens per expert to make the expert GEMMs efficient - EP is what turns a memory-bound small-batch MoE into reasonably sized GEMMs, at the price of all-to-all traffic. Not every routed expert is remote in practice (intra-node destinations go over NVLink), so the real number is lower; the order of magnitude is what drives the design.

### All-to-all kernels: DeepEP

DeepEP (DeepSeek, open-sourced February 2025; GitHub, no paper) provides two kernel families:

- **Normal (high-throughput) kernels** for prefill: hierarchical - tokens go over RDMA to the same-index GPU on the destination node, then are forwarded over NVLink - exploiting NVLink's much higher bandwidth and limiting each token's RDMA fan-out to the number of distinct destination nodes (DeepSeek-V3's routing is node-limited to at most 4 nodes per token).
- **Low-latency kernels** for decode: pure RDMA, CUDA-graph compatible, with a hook-based receive that does not occupy SMs while waiting, so communication can overlap compute.

Overlap is done with **two-batch overlap (TBO)**: split the step's batch into two micro-batches and interleave so that micro-batch A's attention/expert compute runs while B's dispatch/combine is in flight. SGLang and vLLM both implement TBO / dual-batch overlap for DeepEP deployments. **EPLB** (expert-parallel load balancer) periodically replicates hot experts and re-places them using observed routing statistics so no rank becomes the straggler that sets the all-to-all latency.

Other all-to-all implementations exist (pplx-kernels, NCCL's newer device APIs, TRT-LLM's own); the design space is the same.



## Disaggregated prefill and decode



The two phases of a request, from the Splitwise paper:

![prefill and decode phases of one request](images/splitwise-prefill-decode-phases.png)

([source](https://arxiv.org/abs/2311.18677))

### Why

Prefill is compute-bound with large, bursty steps; decode is memory-bound and latency-sensitive. Colocating them forces one scheduler to trade TTFT against ITL (chunked prefill is the colocated compromise). Disaggregation runs them on separate GPU pools: prefill workers compute the prompt's KV, ship it to a decode worker, and the decode worker generates. Each pool can then use its own parallelism (e.g. high TP for prefill to cut TTFT, wide EP / DP for decode to raise batch) and be scaled independently against its own SLO.

- DistServe (Zhong et al., arXiv 2401.09670): optimize *goodput* (requests/s meeting both TTFT and TPOT SLOs) by placing and sizing prefill and decode instances separately.
- Splitwise (Patel et al., arXiv 2311.18677): phase splitting, including running the phases on different hardware generations.
- Mooncake (Qin et al., arXiv 2407.00079): Kimi's production system; KV-cache-centric, with a distributed KV pool over CPU DRAM and SSD, and a scheduler that routes by cache location.

Implementations: vLLM (KV connector API with NIXL, LMCache, Mooncake connectors), SGLang (PD disaggregation with Mooncake/NIXL transfer backends), NVIDIA Dynamo, llm-d. These move fast; the docs of the version you use are authoritative.

### KV transfer cost

The transfer must cost much less than the prefill it replaces, and should be overlapped with it (layer-by-layer streaming: send layer `l`'s KV while computing layer `l+1`).

Per token, 70B model:

```
KV bytes/token      = 327,680 B
transfer @ 50 GB/s  = 6.6 us/token      (one 400 Gb/s NIC)
prefill compute     = 2 * 70e9 FLOPs / (8 GPUs * 989e12 * 0.4) = 44 us/token   (TP=8, 40% MFU)
```

So with a single NIC the transfer is ~15% of prefill time and fully hideable behind it with layer-wise streaming. For an 8k prompt: `2.5 GiB` of KV, `54 ms` on one 400 Gb/s NIC vs `~360 ms` of prefill; with all 8 NICs (one per GPU, each GPU sending its TP shard) it is `~7 ms`. For the 8B model at TP=1: `2.6 us/token` transfer vs `40 us/token` prefill. MLA models are ~5x cheaper per token again. The ratio that matters is

```
transfer/prefill = (KV bytes per token / link bandwidth) / (2 * N_params / (achieved FLOPs))
```

and it is independent of prompt length (ignoring attention FLOPs, which only make prefill longer). Dense GQA models at bf16 over RDMA are comfortably below 1. Where it breaks: slow interconnects (plain TCP - the ordinary kernel-mediated network stack - over 25 Gb/s Ethernet is `~3 GB/s`, making transfer ~2x the prefill for the 70B example: `327680/3e9 = 109 us` vs 44 us), or very small, very fast prefill pools.

### When it wins and when it does not

Wins:

- long prompts with strict ITL SLOs (prefill interference is the dominant ITL tail);
- large scale where pools can be sized independently and the router can keep both saturated;
- MoE models where decode wants wide EP and prefill wants a different layout.

Loses or is neutral:

- short prompts / output-heavy workloads: little prefill to move, but now two hops and a transfer;
- small deployments: you cannot keep two separate pools well utilized, and idle prefill GPUs are pure cost;
- high prefix-cache hit rates: the prefill is cheap anyway, and the cache must now be coordinated across pools.

Chunked prefill in a colocated engine is the baseline to beat. Measure goodput at your SLO, not raw throughput.



## Structured decoding

The [overview](README.md#guided-text-generation) covers what it is for. Internals:

### Grammar to automaton

- **Regex** compiles to a DFA over characters. Outlines (Willard & Louf, "Efficient Guided Generation for Large Language Models", arXiv 2307.09702) lifts the DFA to the token level: for each DFA state, precompute the set of vocabulary tokens whose full character sequence is a valid walk from that state, giving an index `state -> allowed tokens` and `(state, token) -> next state`. Per-step cost is a lookup.
- **JSON Schema** is converted to a grammar; for schemas with bounded nesting this can be a regex/DFA, but general JSON (and any recursive structure) needs a **context-free grammar**, i.e. a pushdown automaton (PDA). The token mask now depends on the stack, not just the state, so it cannot be fully precomputed.

A regex compiled into an FSM that masks invalid tokens at each step (SGLang paper):

![regex compiled to an FSM guiding decoding](images/sglang-regex-fsm.png)

([source](https://arxiv.org/abs/2312.07104))

### Token mask computation

At each step the engine needs a `(B, V)` boolean mask, stored as a bitmask: `V/32` int32 words per row - `4008` words, ~16 KB per request for a 128,256-token vocabulary. It is applied on GPU before sampling (`logits.masked_fill_(~mask, -inf)`, fused into a small kernel). The mask computation is on the CPU and must finish within the GPU step (~5-30 ms) for the whole batch, overlapped with the forward pass.

XGrammar (Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models", arXiv 2411.15100) is the default backend in vLLM and SGLang as of 2025. Its main ideas:

- **Context-independent vs context-dependent tokens.** For each PDA position, most tokens can be classified as valid/invalid by looking only at the current position, without inspecting the stack. Those are precomputed into an *adaptive token mask cache* per grammar position (stored in whichever of accepted-set / rejected-set / bitmask is smallest). Only the small context-dependent remainder is checked at runtime against the actual stack.
- **Persistent stack** (tree-structured, shared across branches) so rollback for speculative decoding and multiple parallel expansions is cheap.
- **Overlap** of mask generation with the GPU forward pass.
- **Grammar compilation caching**: compiling a JSON schema to the PDA plus mask cache costs from milliseconds to seconds; engines cache compiled grammars keyed by schema so the cost is paid once per distinct schema. First-request latency with a new schema is the cost to watch.

Other backends with different tradeoffs: llguidance (Microsoft; lazy, Earley-parser-based, low compile cost), Outlines-core, lm-format-enforcer - all four are pluggable in vLLM (see the overview).

### Interactions

- **Jump-forward decoding.** When the automaton has a single valid continuation (e.g. a fixed JSON key), emit it without decoding (SGLang's compressed FSM). Careful: the forced string must be retokenized consistently with how the model would have tokenized it, otherwise you feed the model a token sequence it never sees in training.
- **Speculative decoding.** Draft tokens must be validated against the grammar (and the grammar state advanced/rolled back per accepted prefix); apply the mask to the draft distribution too.
- **Token boundary artifacts.** Masks that forbid a token which would be the natural tokenization of a valid string push the model off-distribution. This is the mechanism behind reports of structured output hurting quality; whether it matters is task-dependent and should be measured, not assumed.



## Sampling on the GPU

### Avoiding the full sort and the sync

Naive top-p: softmax, sort the `V`-wide row (`V = 128,256` for Llama-3), cumsum, threshold, renormalize, `torch.multinomial`. Sorting `B * V` floats every step is costly at large batch, and `torch.multinomial` plus any `.item()` adds host syncs.

Techniques in use:

- **Exponential-race sampling instead of multinomial.** `argmax(probs / E)` with `E ~ Exp(1)` i.i.d. is an exact sample from `probs` (equivalent to the Gumbel-max trick on logits). It is a single elementwise op plus an argmax with no cumsum and no sync. vLLM's sampler uses this form.
- **Top-k via partial selection** (`torch.topk` / radix select) rather than full sort; for small `k` this is far cheaper.
- **Sorting-free top-p/top-k via rejection sampling.** FlashInfer's sampling kernels sample a candidate, compute the probability mass of tokens with probability greater than the candidate's in one pass, and if the candidate is outside the nucleus raise a pivot and resample - an iterative, sort-free procedure that converges in a few rounds (FlashInfer blog, "Sorting-Free GPU Kernels for LLM Sampling", 2025; details from memory). vLLM and SGLang can use these kernels.
- **Min-p** is cheap: one max reduction and a threshold.

### Logit processors

Penalties (repetition/presence/frequency) need per-request counts of previously generated tokens. Done naively in Python over lists, they are a large CPU cost. Engines keep a `(B, V)` count tensor or apply penalties via `scatter_add` over the output-token-id tensor on GPU. Arbitrary per-request Python logit processors break batching and CUDA graphs; vLLM V1 replaced the per-request callable interface with a batch-level logits-processor interface for this reason. Bad-word lists, logit bias, and allowed-token lists become sparse scatters.

Order matters and is engine-specific: typically penalties, then temperature, then top-k/top-p/min-p, then structured-output mask (or mask first), then sample. If you compare engines' outputs for "the same" sampling params, check the order.

### Determinism and batch invariance

Fixing the seed is not enough for reproducible outputs, even at temperature 0. The dominant cause is not GPU atomics (hardware read-modify-write operations whose ordering across threads is unspecified); it is that the *reduction order of kernels depends on batch shape*: split-K choices in GEMMs, split-KV counts in decode attention, and reduction strategies in RMSNorm are all picked from the batch size and sequence lengths, and floating-point addition is not associative. In a server the batch composition depends on other users' traffic, so the same request gets different numerics on different runs, and greedy decoding eventually diverges at a near-tie. Horace He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (blog, September 2025) is the clear treatment. Additional sources of variation: chunked-prefill chunk boundaries, prefix-cache hit vs miss (cached KV was computed in a different batch shape), speculative decoding on/off, and TP degree.

Fixes are **batch-invariant kernels**: fixed split sizes and reduction orders independent of batch size, for matmul, attention (fixed KV split size, same path for prefill and decode) and normalization. vLLM and SGLang have added batch-invariant / deterministic modes following that work; expect a throughput cost (reported as substantial in the blog's unoptimized version; engines have narrowed it - measure on your workload). For RL, where sampler/trainer logprob mismatch matters, this is worth paying for; for chat serving it usually is not.



## Multi-LoRA serving

Serving many fine-tunes of one base model: keep one copy of base weights, batch requests that use different adapters into the same step. Each linear layer computes `y = x W + (x A_i) B_i` where `i` is the request's adapter.

Size of a rank-16 adapter on all linear layers of the 8B model:

```
per layer: q (4096->4096), k,v (4096->1024 each), o (4096->4096),
           gate, up (4096->14336), down (14336->4096)
params = 32 * 16 * [(4096+4096)*2 + (4096+1024)*2 + (4096+14336)*3] = 41.9M  -> 84 MB bf16
```

So 100 adapters are ~8.4 GB - they compete with the KV cache and cannot all sit on the GPU at higher ranks or larger models.

- **Punica** (Chen et al., arXiv 2310.18547) introduced SGMV (segmented gather matrix-vector multiplication): sort/segment the batch's tokens by adapter, then one kernel launch computes all segments' `x A_i` and `(.) B_i`, gathering adapter weights by index. A BGMV variant handles decode, where each token may have a different adapter.
- **S-LoRA** (Sheng et al., arXiv 2311.03285) adds *unified paging*: adapter weights and KV cache blocks live in one paged pool, adapters are loaded from CPU on demand and evicted LRU, so the number of servable adapters is bounded by host memory rather than GPU memory. It also uses custom kernels for heterogeneous ranks.

Engine knobs (vLLM): `max_loras` (adapters resident per step - a hard cap on adapter diversity within a batch, and a fixed dimension for CUDA graphs), `max_lora_rank` (kernels and buffers are sized for it, so a single rank-256 adapter makes every graph bigger), `max_cpu_loras` (host-side cache). Two interactions that are easy to miss: the adapter id must be in the prefix-cache hash (identical tokens, different KV), and the scheduler must not admit a request whose adapter cannot be loaded without exceeding `max_loras` for the step.

The LoRA FLOPs are small (`~2 * 41.9M` per token vs `~2 * 8e9`, i.e. ~0.5%) but the kernels are memory-bound gathers with poor arithmetic intensity, so multi-LoRA overhead is visibly larger than the FLOP ratio suggests, growing with the number of distinct adapters per batch. Measure it at your expected adapter diversity.



## Observability

The [metrics section of the overview](README.md#key-inference-performance-metrics) defines TTFT, ITL and throughput from the client side. From the engine side, export these (names are vLLM's Prometheus names - Prometheus being the standard metrics scraper that polls an HTTP `/metrics` endpoint; they have been renamed between versions, e.g. `gpu_cache_usage_perc` -> `kv_cache_usage_perc` and `time_per_output_token_seconds` -> `inter_token_latency_seconds` - check `/metrics` on your build; SGLang and TRT-LLM export equivalents):

| Metric | What it tells you | Act when |
|-|-|-|
| `num_requests_running`, `num_requests_waiting` | concurrency and queue depth | waiting > 0 sustained: under-provisioned or KV-bound |
| `kv_cache_usage_perc` | fraction of KV blocks in use | ~1.0 *with* preemptions: oversubscribed |
| `num_preemptions_total` (rate) | KV pressure causing recompute | any sustained rate: lower `max_num_seqs` or add capacity |
| `prefix_cache_queries`, `prefix_cache_hits` | hit rate (in tokens) | low on chat traffic: check routing affinity, cache salt, block size |
| `time_to_first_token_seconds` (histogram) | TTFT = queue + prefill | split it with the next two |
| `request_queue_time_seconds` | time waiting for admission | dominates TTFT: capacity or admission problem, not kernel speed |
| `request_prefill_time_seconds` | prefill duration | dominates TTFT: prefill throughput, chunk budget, disaggregation |
| `inter_token_latency_seconds` (histogram) | decode cadence | p99 >> p50: prefill interference, preemption, or GC/CPU stalls |
| `e2e_request_latency_seconds` | end-to-end | sanity check against the client |
| `iteration_tokens_total` (histogram) | tokens per engine step | small steps at high load: CPU-bound loop or tight `max_num_seqs` |
| `prompt_tokens_total`, `generation_tokens_total` | input/output throughput | divide by GPU count and price for cost per token |
| `spec_decode_num_draft_tokens_total`, `..._accepted_tokens_total` | empirical acceptance rate `alpha` | plug into the table above; disable speculation if speedup < 1 |

What engine metrics cannot tell you, and what to add:

- **GPU idle time between steps** (CPU overhead): from `nsys` or from comparing step wall time to the sum of kernel times; a cheap proxy is `DCGM_FI_PROF_SM_ACTIVE` vs observed tokens/s.
- **Achieved HBM bandwidth during decode** (`DCGM_FI_PROF_DRAM_ACTIVE`; DCGM is NVIDIA's GPU telemetry daemon that exports hardware counters): if decode is far below the roofline from [Internals I](arithmetic.md), the problem is overhead or small-batch kernels, not hardware.
- **Grammar compile time and mask time** per request for structured output, since these show up as unexplained TTFT and ITL.
- **Per-tenant / per-adapter** breakdowns for multi-LoRA and priority scheduling - aggregate p99 hides the tenant who is starving.

Measure TTFT and ITL at the client as well; the engine does not see HTTP queueing, network or detokenization in the API server.



## Further reading

Papers (arXiv ids verified to the best of my knowledge unless marked):

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", arXiv 2309.06180
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022
- Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills", arXiv 2308.16369
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", arXiv 2403.02310
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv 2312.07104
- Dao et al., FlashAttention arXiv 2205.14135; FlashAttention-2 arXiv 2307.08691; Shah et al., FlashAttention-3 arXiv 2407.08608
- Hong et al., "FlashDecoding++", arXiv 2311.01282
- Ye et al., "FlashInfer", arXiv 2501.01005
- Juravsky et al., "Hydragen: High-Throughput LLM Inference with Shared Prefixes", arXiv 2402.05099
- DeepSeek-AI, "DeepSeek-V2", arXiv 2405.04434; "DeepSeek-V3 Technical Report", arXiv 2412.19437
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", arXiv 2211.17192
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", arXiv 2302.01318
- Miao et al., "SpecInfer", arXiv 2305.09781
- Cai et al., "Medusa", arXiv 2401.10774
- Li et al., EAGLE arXiv 2401.15077; EAGLE-2 arXiv 2406.16858; EAGLE-3 arXiv 2503.01840
- Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction", arXiv 2404.19737
- Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding", arXiv 2402.02057
- Shoeybi et al., "Megatron-LM", arXiv 1909.08053
- Zhong et al., "DistServe", arXiv 2401.09670
- Patel et al., "Splitwise", arXiv 2311.18677
- Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", arXiv 2407.00079
- Willard & Louf, "Efficient Guided Generation for Large Language Models", arXiv 2307.09702
- Dong et al., "XGrammar", arXiv 2411.15100
- Chen et al., "Punica: Multi-Tenant LoRA Serving", arXiv 2310.18547
- Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters", arXiv 2311.03285
- Unverified ids: Sheng et al., "Fairness in Serving Large Language Models", arXiv 2401.00588; Kamath et al., "POD-Attention", arXiv 2410.18038; Liu et al., "Optimizing Speculative Decoding for Serving Large Language Models Using Goodput", arXiv 2406.14066; Chen et al., "MagicDec", arXiv 2408.11049; Oliaro et al., "SuffixDecoding", arXiv 2411.04975

Blogs, docs and code:

- vLLM V1 alpha announcement blog (January 2025) and the vLLM design docs (`docs/design/`) on prefix caching, the V1 scheduler, torch.compile integration and CUDA graphs - https://docs.vllm.ai
- SGLang v0.4 blog (overlap scheduler, cache-aware load balancer) and the DeepSeek large-scale EP deployment blog - https://lmsys.org/blog
- Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference" (October 2023)
- FlashInfer blogs on cascade inference and sorting-free sampling - https://flashinfer.ai
- DeepSeek open-infra-index: inference system overview, FlashMLA, DeepEP, EPLB, two-batch overlap profiles - https://github.com/deepseek-ai/open-infra-index
- He / Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (September 2025)
- Source to read top to bottom: `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/worker/gpu_model_runner.py`; SGLang `python/sglang/srt/managers/scheduler.py` and `mem_cache/radix_cache.py` (paths as of 2025; may have moved)
