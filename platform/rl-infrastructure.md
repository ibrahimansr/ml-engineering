# Infrastructure for Large-Scale RL on LLMs

## Prerequisites

This chapter assumes you know the RL algorithms (PPO, GRPO, REINFORCE-style estimators, KL penalties, importance sampling). It does not assume any systems background. It does use the following chapters heavily, so read them first if any of these ideas is new:

- [Concurrency and Parallelism](../foundations/concurrency.md): processes vs threads, `asyncio`, bounded queues and backpressure. An RL system is mostly a producer/consumer pipeline, and the [backpressure section](../foundations/concurrency.md#producerconsumer-and-backpressure) is the core idea of this chapter at small scale.
- [Distributed Systems](../foundations/distributed-systems.md): ranks, process groups, failures and timeouts.
- [Networking](../foundations/networking.md): bandwidth vs latency, NVLink vs InfiniBand vs ordinary Ethernet. Every weight-sync number below is a networking number.
- [Inference Arithmetic](../inference/arithmetic.md) and [Inside a Serving Engine](../inference/serving-engine-internals.md): decode is memory-bound, KV cache sizing, continuous batching, prefix caching, and [determinism and batch invariance](../inference/serving-engine-internals.md#determinism-and-batch-invariance). The generation half of RL *is* an inference server.
- [Parallelism derivations](../training/model-parallelism/derivations.md): [FSDP](../training/model-parallelism/derivations.md#zero-and-fsdp), [tensor parallelism](../training/model-parallelism/derivations.md#tensor-parallelism-megatron) and the [collective cost model](../training/model-parallelism/derivations.md#collective-cost-models-alpha-beta-ring). The training half of RL is an ordinary sharded training job, and weight synchronization is a resharding problem between those two layouts.
- [Containers and Virtualization](../foundations/containers-and-virtualization.md): needed for the sandboxed code-execution section.
- [Training Arithmetic](../training/arithmetic.md): FLOPs per token (`6N`) and MFU.

Online RL for LLMs differs from pretraining in one important way: the training data does not exist until the current model creates it. Every step therefore runs an inference workload (generate rollouts), an evaluation workload (score them, possibly by executing code), and a training workload (compute gradients), and then moves the new weights back to the inference side. Each of these is a distributed system on its own. The RL algorithm determines which of them may overlap and by how much. This chapter covers the infrastructure that connects them: placement, staleness, weight transfer, rollout serving, environments, queues, failure handling, and how the open frameworks are organized.

**What was run.** Every Python block marked with a file name below was run on an Apple M4 laptop (10 cores, macOS, Python 3.14, PyTorch 2.15 CPU, NumPy 2.5, `transformers` 5.22 for the tokenizer check). Their outputs are pasted verbatim. Timings from the toy pipelines depend on the machine and change between runs. Anything that needs GPUs, a cluster, Ray, Slurm or Docker is marked **untested**. All throughput and time estimates are first-order arithmetic from stated assumptions, not benchmarks, unless a source is cited.

## The shape of an RL step

### Components and vocabulary

These terms are used throughout the chapter:

- **Policy**: the model being trained. Its weights have a **version**, an integer that increments with every optimizer step.
- **Rollout** (also *trajectory* or *sample*): one generated response, or for agents one whole multi-turn episode, together with the per-token logprobs the sampler assigned to it.
- **Generator / rollout worker / inference engine**: a process (normally a vLLM or SGLang server) that holds a copy of the policy and generates rollouts.
- **Trainer**: the sharded training job (FSDP, Megatron) that turns rollouts into gradient steps.
- **Environment**: code that supplies prompts, and for agents also responds to the model's actions with tool outputs.
- **Verifier / reward function**: code that turns a finished rollout into a scalar reward. It ranges from a regex to a test suite running in a sandbox to another LLM acting as a judge.
- **Orchestrator / controller**: the process that drives the loop. It hands prompts to generators, collects rollouts, calls verifiers, forms batches, tells the trainer to step, and triggers weight sync.
- **Weight sync**: copying the trainer's new weights into every inference engine.
- **Staleness**: for a token used in a gradient step, the trainer's current version minus the version that sampled it. Zero means on-policy.

```
             prompts                     rollouts + sampler logprobs
 [dataset/env] ──────► [ generators (vLLM/SGLang) ] ──────► [ verifiers / reward ]
                              ▲                                     │ rewards
                              │ weights (version v+1)               ▼
                       [ weight sync ] ◄────────── [ trainer (FSDP/Megatron) ] ◄── [ batcher / queue ]
```

The synchronous loop, written as the simplest possible code:

```python
# The synchronous RL loop. Every arrow above is one line here, and each line hides a distributed system.
for step in range(num_steps):
    prompts  = dataset.next_batch()                        # (P,)
    rollouts = engines.generate(prompts, n=G)              # (P*G,) sequences + sampler logprobs
    rewards  = verifiers.score(rollouts)                   # (P*G,)  may execute code
    batch    = make_batch(rollouts, rewards)               # advantages, masks, packing
    trainer.step(batch)                                    # recompute logprobs, loss, backward, optimizer
    engines.load_weights(trainer.state_dict())             # weight sync -> version step+1
```

Each line in that loop blocks until the previous one finishes. The rest of the chapter is mostly about removing those blocking waits without breaking the algorithm.

### Where the time goes: arithmetic for a realistic config

Configuration: a Qwen2.5-32B-like dense model (64 layers, 8 KV heads of dim 128, about 32.5B parameters) on 64 H100s. One GRPO step uses 256 prompts × 16 samples, 1024-token prompts, and long-tailed reasoning responses (lognormal, median about 6k tokens, capped at 32k). Generation runs on 16 TP=4 engines. Training recomputes the old-policy logprobs (`2N` FLOPs per token), runs a reference-model forward (`2N`), and a forward+backward (`6N`), for `10N` FLOPs per token.

The script below simulates one engine with continuous batching. Each decode step costs one read of the weights plus one read of every active sequence's KV cache from HBM. That is the decode roofline from [inference arithmetic](../inference/arithmetic.md#decode-latency-lower-bound), assuming 60% of peak HBM bandwidth is achieved.

```python
# step_budget.py - where the time goes in one synchronous GRPO-style step.
# Qwen2.5-32B-like dense model, 64x H100, colocated. First-order model, not a benchmark.
import numpy as np

# model
N      = 32.5e9                 # params
L, kvh, hd = 64, 8, 128
kv_tok = 2 * L * kvh * hd * 2   # bytes/token, bf16 K and V
W      = 2 * N                  # bf16 weight bytes

# hardware (H100 SXM)
HBM, PEAK = 3.35e12, 989e12
MBU, MFU  = 0.6, 0.4            # assumed achieved fraction of HBM bw (decode) / FLOPs (train)

# workload
prompts, G = 256, 16            # 256 prompts x 16 samples = 4096 rollouts
P_len      = 1024
rng = np.random.default_rng(0)
resp = np.clip(rng.lognormal(np.log(6000), 0.8, prompts * G), 200, 32768).astype(int)

# inference layout: TP=4 engines on 64 GPUs
gpus, tp = 64, 4
engines  = gpus // tp
kv_budget = 0.9 * tp * 80e9 - W            # bytes left for KV per engine
max_tokens_in_kv = kv_budget / kv_tok

def simulate_engine(lengths, max_seqs=256):
    """Continuous batching (oracle admission: a seq is admitted only if its final length fits;
    real engines admit optimistically and preempt), one decode token per active seq per step.
    Step time = (weights + KV of all active seqs) / (tp * HBM * MBU). Prefill ignored."""
    waiting = list(lengths); active = []  # active: [generated, target]
    t, steps, util = 0.0, 0, []
    while waiting or active:
        ctx = sum(P_len + g for g, _ in active)
        while waiting and len(active) < max_seqs and ctx + P_len + waiting[0] <= max_tokens_in_kv:
            active.append([0, waiting.pop(0)]); ctx += P_len
        B = len(active)
        kv_read = sum(P_len + g for g, _ in active) * kv_tok
        t += (W + kv_read) / (tp * HBM * MBU)
        util.append(B)
        for a in active: a[0] += 1
        active = [a for a in active if a[0] < a[1]]
        steps += 1
    return t, steps, np.array(util)

per_engine = np.array_split(rng.permutation(resp), engines)
t_gen, steps, util = simulate_engine(list(per_engine[0]))
gen_tokens = resp.sum()

# training: old-logprob fwd (2N) + ref fwd (2N) + fwd/bwd (6N) per token, prompt+response
train_tokens = (P_len + resp).sum()
flops = 10 * N * train_tokens
t_train = flops / (gpus * PEAK * MFU)

print(f"KV bytes/token          {kv_tok/1024:.0f} KiB; KV budget/engine {kv_budget/1e9:.0f} GB = {max_tokens_in_kv/1e3:.0f}k tokens")
print(f"rollouts                {resp.size}, response tokens mean {resp.mean():.0f}, p50 {np.median(resp):.0f}, p99 {np.percentile(resp,99):.0f}, max {resp.max()}")
print(f"generated tokens        {gen_tokens/1e6:.1f} M")
print(f"generation (1 engine)   {t_gen:.0f} s over {steps} decode steps")
print(f"  batch size: mean {util.mean():.0f}, steps with batch<=16: {(util<=16).mean():.0%}")
print(f"  time-weighted throughput {util[0:].sum()/t_gen:.0f} tok/s/engine")
print(f"training (10N/token)    {t_train:.0f} s  ({flops:.2e} FLOPs at {MFU:.0%} MFU)")
print(f"generation share        {t_gen/(t_gen+t_train):.0%} of a synchronous step")
same = [int(per_engine[0].mean())] * len(per_engine[0])   # same total tokens, no tail
t_flat, _, _ = simulate_engine(same)
print(f"same tokens, no tail    {t_flat:.0f} s  -> the tail costs {t_gen/t_flat:.1f}x")
```

```
$ python step_budget.py
KV bytes/token          256 KiB; KV budget/engine 223 GB = 851k tokens
rollouts                4096, response tokens mean 7978, p50 5830, p99 32768, max 32768
generated tokens        32.7 M
generation (1 engine)   837 s over 32768 decode steps
  batch size: mean 65, steps with batch<=16: 35%
  time-weighted throughput 2560 tok/s/engine
training (10N/token)    473 s  (1.20e+19 FLOPs at 40% MFU)
generation share        64% of a synchronous step
same tokens, no tail    432 s  -> the tail costs 1.9x
```

How to read the output:

1. **Generation is about two thirds of a synchronous step, even with oracle scheduling.** It also understates the problem. The step waits for the slowest of 16 engines, and this simulation measures only one of them.
2. **The tail costs 1.9×.** The same total number of tokens, spread evenly across sequences, would take 432 s instead of 837 s. For 35% of decode steps the engine has 16 or fewer sequences running, so it reads 65 GB of weights per step to produce at most 16 tokens. A 32k-token response needs 32,768 *sequential* decode steps no matter how many GPUs you have. Generation time has a floor set by the longest sequence, not by total work.
3. **Training is regular and predictable.** 473 s comes from `10N × tokens / (64 × 989 TFLOP/s × 0.4)`. It scales almost linearly with GPU count and does not depend on the length distribution, only on the total number of tokens.

The same reasoning applies to any long-chain-of-thought setup. Reports from frameworks and labs put generation at roughly half to 80% of wall time. That share depends on the length distribution, which is why almost every infrastructure idea in this chapter targets the generation tail.

### Why the tail is a systems problem

Decode on a memory-bound GPU costs about the same per step whether the batch holds 16 sequences or 200 (see the [roofline](../inference/arithmetic.md#arithmetic-intensity-and-the-roofline)). Throughput per GPU is therefore roughly proportional to batch size, and batch size collapses at the end of every synchronous generation phase. There are four ways out, each covered later:

| Fix | Idea | Where |
|---|---|---|
| Async / pipelined RL | start the next batch's rollouts while the tail finishes | [Synchronous vs asynchronous RL](#synchronous-vs-asynchronous-rl) |
| Partial rollouts | cap tokens per iteration and carry unfinished rollouts over | [Partial rollouts](#partial-rollouts) |
| Over-sampling | start more rollouts than needed, keep the first that finish | [Continuous batching and the long tail](#continuous-batching-and-the-long-tail) |
| Length limits / penalties | an algorithm change, not an infrastructure one | out of scope |

Over-sampling and keeping the first finishers biases the batch toward short responses, and partial rollouts mix policy versions inside one response. Each of these fixes has an algorithmic cost, and the importance-ratio corrections [below](#importance-ratio-corrections-systems-use) are how systems pay for it.

## Placement: colocated vs disaggregated

*Placement* is the decision about which GPUs run which component. There are two basic designs.

### Colocated (the "hybrid engine")

In a **colocated** design every GPU alternates between roles. It holds the inference engine during generation and the trainer shards during training. verl's HybridFlow (Sheng et al., arXiv 2409.19256) made this the default, and OpenRLHF's `--train.colocate_all` and TRL's `vllm_mode="colocate"` do the same. Its appeal is that no GPU is ever idle because it is waiting on another role. Its cost is memory juggling: both roles must fit in, or be swapped through, the same 80 GB.

Memory per GPU for the 32B example on 64 GPUs:

```python
# colocated_memory.py - what has to share one 80 GB H100 in the colocated 32B setup
N, gpus, tp = 32.5e9, 64, 4
train_state = 16 * N / gpus / 1e9          # FSDP: bf16 params+grads, fp32 master+Adam m,v, sharded 64 ways
infer_weights = 2 * N / tp / 1e9           # bf16 weights, TP=4 shard
kv = 0.9 * 80 - infer_weights              # what vLLM would grab for KV at gpu_memory_utilization=0.9
print(f"trainer sharded state {train_state:.1f} GB + activations (several GB to tens of GB)")
print(f"inference weights     {infer_weights:.1f} GB + KV cache {kv:.1f} GB")
```

```
$ python colocated_memory.py
trainer sharded state 8.1 GB + activations (several GB to tens of GB)
inference weights     16.2 GB + KV cache 55.8 GB
```

Both sets of state cannot sit in memory together at full size, so colocated systems **sleep** the engine while training. vLLM's sleep mode (as of vLLM's docs, 2026) has two levels. Level 1 offloads weights to CPU RAM and discards the KV cache. Level 2 discards both, which is what you want when the trainer is about to overwrite the weights anyway. `wake_up(tags=[...])` lets you bring back only the weights first, load the new ones, and only then allocate the KV cache. That ordering keeps peak memory low. SGLang has the equivalent `release_memory_occupation` / `resume_memory_occupation` engine methods. The trainer offloads its optimizer state (and sometimes its parameters) to CPU in the opposite direction. **Untested** (needs a GPU):

```python
# colocated step, vLLM offline engine (API per vLLM docs/features/sleep_mode.md; untested here)
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=4, enable_sleep_mode=True)

outs = llm.generate(prompts, sampling_params)          # 1. generate (KV cache allocated)
llm.sleep(level=2)                                      # 2. drop weights and KV: trainer gets the HBM
trainer.load_to_gpu(); trainer.step(batch); trainer.offload_optimizer_to_cpu()   # 3. train
llm.wake_up(tags=["weights"])                           # 4. allocate weight buffers only
push_weights(trainer, llm)                              # 5. sync (CUDA IPC / in-process copy)
llm.wake_up(tags=["kv_cache"])                          # 6. now allocate KV and generate again
```

Phase switches are not free. Moving 8 GB of optimizer state over PCIe at about 25-50 GB/s takes a fraction of a second. Re-creating CUDA graphs, re-allocating the KV pool and running Python-side weight loading take seconds to tens of seconds. That overhead is the `t_switch` in the next script, and I assume it rather than measure it.

### Disaggregated

In a **disaggregated** design inference and training run on separate GPU pools, usually on separate nodes, and weights travel over the network after every step (or every few steps). This only makes sense with asynchronous RL. With a synchronous loop, the training pool sits idle during generation and vice versa. With async, both pools run all the time, and the question becomes how to split the GPUs.

```python
# placement.py - colocated vs disaggregated, using step_budget.py's numbers for 64 GPUs.
t_gen_tail, t_gen_flat, t_train, G = 837, 432, 473, 64   # seconds on all 64 GPUs
t_switch = 30          # assumed: offload/onload weights+optimizer, KV realloc, weight sync

print(f"colocated, sync               {t_gen_tail + t_train + t_switch:5.0f} s/step")
print(f"colocated, partial rollouts   {t_gen_flat + t_train + t_switch:5.0f} s/step  (tail carried to next step)")
for g in (16, 24, 30, 32, 40, 48):
    gen   = t_gen_flat * G / g          # assume perfect scaling with GPU count
    train = t_train * G / (G - g)
    step  = max(gen, train)             # async pipeline: slower side sets the pace
    idle  = 1 - min(gen, train) / step
    side  = "trainer" if gen > train else "generator"
    print(f"disagg {g:2d} gen / {G-g:2d} train   {step:5.0f} s/step  ({side} idle {idle:.0%})")
```

```
$ python placement.py
colocated, sync                1340 s/step
colocated, partial rollouts     935 s/step  (tail carried to next step)
disagg 16 gen / 48 train    1728 s/step  (trainer idle 64%)
disagg 24 gen / 40 train    1152 s/step  (trainer idle 34%)
disagg 30 gen / 34 train     922 s/step  (trainer idle 3%)
disagg 32 gen / 32 train     946 s/step  (generator idle 9%)
disagg 40 gen / 24 train    1261 s/step  (generator idle 45%)
disagg 48 gen / 16 train    1892 s/step  (generator idle 70%)
```

What this shows:

- **With perfect linear scaling, overlap by itself is zero-sum.** The best disaggregated split (30/34) gives 922 s, which is just `432 + 473`: generation work plus training work. That is the same as colocated with partial rollouts (935 s, which differs only by the assumed switch cost). The 1340 → ~930 s improvement comes from **removing the tail**, not from overlapping the two phases.
- **Disaggregation is unforgiving about the split.** Misjudge the ratio by 25% and one side idles 34-45% of the time. The right split also moves during training, because response lengths change as the policy learns. (Response length often grows during reasoning RL.)
- Disaggregation wins when the linear-scaling assumption fails in its favour. Inference and training can use different parallelism, different hardware (cheaper or older GPUs for inference, even consumer GPUs over the internet in decentralized setups), and different failure domains. Inference capacity becomes elastic: engines join and leave without stopping the trainer. You also skip the memory juggling entirely. Colocation wins when the cluster is small, the network between pools is slow, or the run must be strictly on-policy.

### Choosing

| Question | Points toward colocated | Points toward disaggregated |
|---|---|---|
| Must training be strictly on-policy? | yes | no, bounded staleness is acceptable |
| Cluster size | one to a few nodes | many nodes, or heterogeneous hardware |
| Network between would-be pools | slow | InfiniBand/RoCE, or acceptable WAN with compression |
| Response length distribution | short, regular | long-tailed (async fills the tail) |
| Need elastic or preemptible inference capacity | no | yes |
| Engineering budget | less moving parts at runtime, but tricky memory management | more services, queues, failure modes |

Most large-scale frameworks now support both. Open systems built for long agentic rollouts have moved toward disaggregated + asynchronous designs (prime-rl, AReaL, slime's async mode, verl's `fully_async_policy`).

## Synchronous vs asynchronous RL

### The staleness spectrum

Systems use the following modes, ordered from fully on-policy to most off-policy:

| Mode | Behaviour policy for a batch trained at version `v` | Who uses it (as of late 2026; verify) |
|---|---|---|
| Synchronous | `v` exactly | TRL default, verl/OpenRLHF default colocated mode |
| One-step off-policy | `v-1`: generate batch `n+1` while training on batch `n` | verl `one_step_off_policy`, prime-rl's basic overlap, OpenRLHF `--train.async_enable` with queue size 1 |
| Bounded async | any version `≥ v-k` | AReaL (arXiv 2505.24298), prime-rl `orchestrator.max_staleness` |
| In-flight weight updates | a *single* rollout spans several versions; weights change mid-generation | PipelineRL (arXiv 2509.19128), OpenRLHF `--train.partial_rollout_enable`, AReaL interruptible rollouts |

In-flight updates exist because of the tail. Without them, a 32k-token rollout started at version `v` will finish at version `v+5` and is entirely 5 steps stale. With them, its last tokens were sampled by a recent policy. The cost is that the KV cache for the first part of the response was computed under the old weights. Systems generally keep that stale KV rather than recompute it, and record per-token versions so the loss can account for them.

### A toy async RL pipeline

The following toy contains every moving part of an async system, scaled down until it runs in two seconds. The **trainer** is a separate OS process (`multiprocessing`), as a real trainer would be. The **generator** is one `asyncio` event loop that keeps `SLOTS` rollouts in flight at once, as an inference engine with continuous batching does. The two are connected by a **bounded queue**, which provides backpressure. Weights are a shared-memory array published together with a **version counter** under a lock, which stands in for weight sync. The policy is a 4-way softmax, reward is 1 for action 3, and rollout length is lognormal so the tail is real. `k` is the maximum lag: a rollout may *start* only if its batch index is at most `k` ahead of the trainer's version. `k=0` is the synchronous loop.

```python
# toy_async_rl.py - generator (asyncio, many concurrent rollouts) -> bounded queue -> trainer (process).
# The "policy" is softmax over 4 actions; reward = 1 for action 3. Rollout length is long-tailed.
# max_lag k: a rollout may start only while (its batch index - trainer version) <= k.  k=0 is synchronous.
import asyncio, multiprocessing as mp, time
import numpy as np

A, B, STEPS, TRAIN_S, TOKEN_S = 4, 16, 12, 0.15, 0.002

def logp(theta): z = theta - theta.max(); return z - np.log(np.exp(z).sum())

def trainer(q, weights, version, lock, log, produced):
    theta = np.zeros(A); busy = 0.0; t_start = None
    for step in range(STEPS):
        batch = q.get()                                   # blocks when the generator is behind
        t0 = time.perf_counter(); t_start = t_start or t0
        acts = np.array([r["a"] for r in batch]); rew = np.array([r["r"] for r in batch])
        lp_now = logp(theta)[acts]                        # recompute under the *current* policy
        lp_beh = np.array([r["lp"] for r in batch])       # what the sampler reported
        ratio = np.exp(lp_now - lp_beh)                   # importance ratio pi_theta / pi_behaviour
        w = np.minimum(ratio, 2.0)                        # truncated IS (cap C=2)
        adv = rew - rew.mean()                            # group-mean baseline
        grad = np.zeros(A)
        for a, wi, ad in zip(acts, w, adv):               # REINFORCE: grad of log softmax = onehot - p
            grad += wi * ad * (np.eye(A)[a] - np.exp(logp(theta)))
        theta += 0.5 * grad / len(batch)
        time.sleep(TRAIN_S)                               # pretend fwd/bwd/optimizer
        with lock:                                        # "weight sync": publish theta + version together
            weights[:] = theta; version.value = step + 1
        busy += time.perf_counter() - t0
        stale = [step - r["v"] for r in batch]
        log.put(dict(step=step, stale=stale, ratio_dev=float(np.abs(ratio - 1).max()),
                     reward=float(rew.mean()), qdepth=produced.value - step - 1))   # mp.Queue.qsize() is unimplemented on macOS
    log.put(dict(done=True, busy=busy, span=time.perf_counter() - t_start))

async def generator(q, weights, version, lock, k, seed, produced, SLOTS):
    rng = np.random.default_rng(seed)
    started, slots, done = 0, asyncio.Semaphore(SLOTS), []
    loop = asyncio.get_running_loop()
    flush = asyncio.Lock()

    async def rollout(idx):
        while idx // B - version.value > k:               # staleness gate
            await asyncio.sleep(0.001)
        with lock: theta, v = np.array(weights[:]), version.value   # snapshot weights at start
        p = np.exp(logp(theta))
        a = rng.choice(A, p=p)
        n_tok = int(min(rng.lognormal(np.log(30), 0.9), 400))      # long-tailed response length
        await asyncio.sleep(n_tok * TOKEN_S)                         # "decode"
        done.append(dict(a=int(a), lp=float(np.log(p[a])), r=float(a == 3), v=v))
        slots.release()
        async with flush:                                 # first B finished rollouts form a batch
            if len(done) >= B:
                batch = done[:B]; del done[:B]
                await loop.run_in_executor(None, q.put, batch)   # blocks if the queue is full
                produced.value += 1

    tasks = []
    while started < STEPS * B:
        await slots.acquire()                             # continuous batching: refill a free slot
        tasks.append(asyncio.create_task(rollout(started))); started += 1
    await asyncio.gather(*tasks)

def run(k, slots=16, qmax=2, verbose=False):
    q, log = mp.Queue(maxsize=qmax), mp.Queue()
    weights, version, lock = mp.Array("d", A, lock=False), mp.Value("i", 0, lock=False), mp.Lock()
    produced = mp.Value("i", 0, lock=False)
    tp = mp.Process(target=trainer, args=(q, weights, version, lock, log, produced)); tp.start()
    t0 = time.perf_counter()
    asyncio.run(generator(q, weights, version, lock, k, 0, produced, slots))
    rows = []
    while "done" not in (end := log.get()): rows.append(end)
    tp.join(); wall = time.perf_counter() - t0
    if verbose:
        for r in rows:
            h = np.bincount(r["stale"], minlength=3)
            print(f"  step {r['step']:2d}  staleness hist {dict(enumerate(h.tolist()))}  "
                  f"max|ratio-1| {r['ratio_dev']:.3f}  reward {r['reward']:.2f}  queue {r['qdepth']}")
    st = np.concatenate([r["stale"] for r in rows])
    return wall, end["busy"] / wall, st.mean(), st.max(), np.mean([x["ratio_dev"] for x in rows]), rows[-1]["reward"], max(x["qdepth"] for x in rows)

if __name__ == "__main__":
    print("k=1, step by step:")
    run(1, verbose=True)
    print(f"\n{'k':>3} {'slots':>5} {'wall s':>7} {'trainer busy':>12} {'mean stale':>10} {'max stale':>9} {'mean max|r-1|':>13} {'final reward':>12} {'max queue':>9}")
    for k, slots in ((0, 16), (1, 16), (2, 16), (4, 16), (4, 64)):
        wall, util, ms, mx, rd, rew, qd = run(k, slots)
        print(f"{k:>3} {slots:>5} {wall:7.2f} {util:12.0%} {ms:10.2f} {mx:9d} {rd:13.3f} {rew:12.2f} {qd:9d}")
```

```
$ python toy_async_rl.py
k=1, step by step:
  step  0  staleness hist {0: 16, 1: 0, 2: 0}  max|ratio-1| 0.000  reward 0.38  queue 0
  step  1  staleness hist {0: 2, 1: 14, 2: 0}  max|ratio-1| 0.121  reward 0.12  queue 0
  step  2  staleness hist {0: 3, 1: 11, 2: 2}  max|ratio-1| 0.099  reward 0.12  queue 0
  step  3  staleness hist {0: 2, 1: 12, 2: 2}  max|ratio-1| 0.114  reward 0.56  queue 0
  step  4  staleness hist {0: 1, 1: 13, 2: 1, 3: 1}  max|ratio-1| 0.249  reward 0.44  queue 0
  step  5  staleness hist {0: 2, 1: 13, 2: 1}  max|ratio-1| 0.250  reward 0.44  queue 0
  step  6  staleness hist {0: 1, 1: 13, 2: 2}  max|ratio-1| 0.138  reward 0.50  queue 0
  step  7  staleness hist {0: 3, 1: 13, 2: 0}  max|ratio-1| 0.106  reward 0.44  queue 0
  step  8  staleness hist {0: 2, 1: 12, 2: 2}  max|ratio-1| 0.193  reward 0.69  queue 0
  step  9  staleness hist {0: 3, 1: 13, 2: 0}  max|ratio-1| 0.097  reward 0.56  queue 0
  step 10  staleness hist {0: 2, 1: 12, 2: 1, 3: 0, 4: 0, 5: 1}  max|ratio-1| 0.358  reward 0.50  queue 0
  step 11  staleness hist {0: 0, 1: 14, 2: 1, 3: 0, 4: 1}  max|ratio-1| 0.325  reward 0.62  queue 0

  k slots  wall s trainer busy mean stale max stale mean max|r-1| final reward max queue
  0    16    5.99          31%       0.00         0         0.000         0.62         0
  1    16    2.32          79%       0.92         5         0.163         0.62         0
  2    16    2.01          91%       1.75         6         0.252         0.62         1
  4    16    1.95          94%       2.96         9         0.422         0.44         2
  4    64    1.95          95%       3.17         8         0.439         0.50         2
```

What this output shows (your timings will differ):

- **Synchronous (`k=0`) keeps the trainer only 31% busy.** It waits for the slowest rollout of every batch. Allowing one step of lag cuts wall time 2.6× (5.99 → 2.32 s), and the trainer reaches about 80-95% busy. This is the async win: the next batch's rollouts fill the slots the tail leaves empty.
- **Staleness at training time is not bounded by `k`.** With `k=1` the step-by-step log shows rollouts 4 and 5 versions stale. The gate applies when a rollout *starts*, and a long rollout started at version 5 can finish after the trainer reaches version 10. Every system that gates only at submission has this property. To bound staleness at training time you must also act at consumption: drop or mask stale tokens (prime-rl masks tokens beyond `max_staleness` and never cancels rollouts, per its `docs/algorithms.md`), or interrupt and resume long rollouts with new weights.
- **The importance ratio drifts away from 1 as staleness grows** (mean max `|r-1|` goes 0 → 0.16 → 0.25 → 0.42). This toy has 4 actions and a large learning rate. With real vocabularies and long sequences the effect compounds per token, which is the next section.
- **Backpressure works.** With `k=4` the generator outruns the trainer, the queue fills to its bound of 2, and `q.put` blocks the generator instead of letting memory grow and staleness explode. Adding 4× more slots (`slots=64`) changes nothing, because the trainer is now the bottleneck. The queue depth tells you which side to scale, exactly as in [the concurrency chapter](../foundations/concurrency.md#producerconsumer-and-backpressure).
- Two macOS details came up while writing this. `multiprocessing.Queue.qsize()` raises `NotImplementedError` on macOS, so the queue depth is computed from counters. And the lock around the weights+version write matters: without it a generator could read new weights with an old version number, or the reverse, and every staleness metric would be wrong. Real systems have the same requirement. The version must travel with the weights, atomically.

### Importance-ratio corrections systems use

Let `π` be the trainer's current policy and `μ` the behaviour policy that produced the tokens (stale weights, a different numeric stack, or both). Then the per-token ratio is `r_t = π(y_t | x, y_<t) / μ(y_t | x, y_<t)`. PPO and GRPO already use a ratio, but against `π_old`. Infrastructure mismatch adds a *second* ratio, and it is easy to get the two confused. The standard corrections:

- **Recompute `π_old` on the trainer** instead of trusting the sampler's logprobs (the `2N` forward in the step budget). PPO clipping then operates on trainer-vs-trainer ratios, and the sampler/trainer gap is handled separately.
- **Truncated importance sampling (TIS)**: weight each token by `min(r_t, C)`, with `C` typically 2. This was popularised for exactly this sampler/trainer mismatch by Yao et al., "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (blog, August 2025). It has V-trace lineage (IMPALA, arXiv 1802.01561).
- **Masking instead of weighting**: drop tokens, or whole sequences, whose ratio leaves a band. IcePop (from Ant's Ling team, 2025) masks tokens outside `[ratio_low, ratio_high]`. prime-rl ships it as an opt-in loss with defaults 0.2 and 5.0, and its default loss masks tokens whose *probability* change `|π − μ|` exceeds `eps = 0.3` (prime-rl `docs/algorithms.md`, as of November 2026). TRL's GRPO has `vllm_importance_sampling_correction` (default `True`), `vllm_importance_sampling_mode` (default `"sequence_mask"`) and `vllm_importance_sampling_clip_max` (default `3.0`) in `grpo_config.py` as of November 2026.
- **Decoupled PPO objective**: separate the *proximal* policy (for the trust region) from the *behaviour* policy (for the IS weight). Hilton et al., "Batch size-invariance for policy optimization" (arXiv 2110.00641). AReaL uses it for stale data.
- **Staleness bounds plus diagnostics**: cap `k`, and log the mismatch KL *per staleness bin*. Bin 0 then isolates pure numerics, and higher bins show drift from staleness (prime-rl logs `mismatch_kl_by_staleness/{bin}`).

How large is the ratio spread from staleness alone? Here is a synthetic experiment where each optimizer step moves the logits by a fixed amount of noise:

```python
# ratios.py - why the sampler's logprobs are not the trainer's, and what TIS does about it.
import torch
torch.manual_seed(0)
T, V = 8192, 32000                                   # one 8k-token response, 32k vocab
logits = torch.randn(T, V) * 3                       # (T, V) "trainer" logits, fp32

# sampler: same weights, but bf16 logits (different kernels / precision / batch shape)
samp_logits = logits.to(torch.bfloat16).float()      # (T, V)
samp_logp = torch.log_softmax(samp_logits, -1)
tok = torch.multinomial(samp_logp.exp(), 1).squeeze(1)            # (T,) sampled by the "sampler"
lp_samp = samp_logp.gather(1, tok[:, None]).squeeze(1)            # (T,) logprobs the sampler reports
lp_train = torch.log_softmax(logits, -1).gather(1, tok[:, None]).squeeze(1)  # (T,) trainer recompute

log_r = lp_train - lp_samp                                        # (T,) per-token log ratio
print(f"per-token |log ratio|: mean {log_r.abs().mean():.2e}  max {log_r.abs().max():.2e}")
print(f"sequence log ratio (sum over {T} tokens): {log_r.sum():+.3f}  -> ratio {log_r.sum().exp():.3f}")
mis_kl = (lp_samp - lp_train).mean()                              # k1 estimate of KL(sampler || trainer)
print(f"mismatch KL estimate (k1, per token): {mis_kl:.2e}")

# staleness: the behaviour policy is k optimizer steps old; each step moves logits by ~eps
for k in (0, 1, 4, 16):
    old = logits - 0.02 * k * torch.randn(T, V)                   # (T, V) policy k steps ago
    old_logp = torch.log_softmax(old, -1)
    a = torch.multinomial(old_logp.exp(), 1).squeeze(1)
    r = (torch.log_softmax(logits, -1).gather(1, a[:, None]) - old_logp.gather(1, a[:, None])).exp().squeeze(1)
    tis = r.clamp(max=2.0)                                        # truncated IS weight, cap C=2
    print(f"staleness k={k:2d}: ratio p1 {r.quantile(.01):.3f} p99 {r.quantile(.99):.3f} max {r.max():6.2f}"
          f"  | frac clipped by TIS(C=2) {(r > 2).float().mean():.4f}  | ESS/T {(r.sum()**2 / (r**2).sum() / T):.3f}")
```

```
$ python ratios.py
per-token |log ratio|: mean 1.21e-02  max 3.95e-02
sequence log ratio (sum over 8192 tokens): +0.014  -> ratio 1.014
mismatch KL estimate (k1, per token): -1.71e-06
staleness k= 0: ratio p1 1.000 p99 1.000 max   1.00  | frac clipped by TIS(C=2) 0.0000  | ESS/T 1.000
staleness k= 1: ratio p1 0.955 p99 1.048 max   1.08  | frac clipped by TIS(C=2) 0.0000  | ESS/T 1.000
staleness k= 4: ratio p1 0.824 p99 1.205 max   1.34  | frac clipped by TIS(C=2) 0.0000  | ESS/T 0.994
staleness k=16: ratio p1 0.455 p99 1.986 max   3.08  | frac clipped by TIS(C=2) 0.0095  | ESS/T 0.908
```

The bottom four lines are the staleness part. The p1-p99 ratio band widens from `[0.96, 1.05]` at one step of staleness to `[0.46, 1.99]` at 16, and TIS clipping only starts to bite at large `k`. The effective sample size (`ESS = (Σr)² / Σr²`) falls slowly, so in this toy moderate staleness wastes little data. Real policy drift per step depends on learning rate and data and is not Gaussian. Measure it on your own run with the per-bin mismatch KL. The top three lines are the zero-staleness part, covered next.

### Sampler-trainer mismatch at zero staleness

Even with identical weights and version, the inference engine and the trainer compute different logprobs. The reasons:

- **Precision and kernels.** The engine uses fused attention and decode kernels, sometimes FP8 weights or KV, and its reduction order depends on batch shape. The trainer uses training kernels in bf16 with fp32 accumulation. See [determinism and batch invariance](../inference/serving-engine-internals.md#determinism-and-batch-invariance).
- **MoE routing flips**: a near-tie in the router sends the token to a different expert, so the output is different, not just rounded differently.
- **Sampling post-processing**: temperature, top-p/top-k and repetition penalty change `μ`. If the engine returns logprobs of the *raw* distribution but sampled from the *processed* one (or the reverse), the ratio is wrong for every token. TRL's config notes that `repetition_penalty != 1.0` requires turning its IS correction off for this reason.
- **Tokenization and templates**: the trainer may score a *different token sequence* than the one sampled. See [chat templates and tokenization](#chat-templates-and-tokenization-consistency).

In `ratios.py` above, only bf16 rounding of the logits is simulated. The per-token `|log r|` is about 1.2e-2, but it is unbiased noise, so the sequence-level sum over 8192 tokens nearly cancels (ratio 1.014) and the KL estimate is around 0. Real mismatch is **systematic**: the same kernel errs the same way, and MoE flips are discrete. Systematic error accumulates over long sequences instead of cancelling. That is why sequence-level masks exist, and why batch-invariant inference kernels and an fp32 LM head (TRL's `cast_lm_head_to_fp32`) are things RL teams actually pay for. The practical rule: **log the mismatch KL from the very first step**. In a healthy setup it is small (around 1e-3 or less per token is a commonly cited range; this is not a universal threshold) and flat. A jump means something in the infrastructure changed.

## Weight synchronization

### What has to move, and how long it takes

After every trainer step, every inference engine needs the new weights. The lower bound on transfer time is bytes per receiver divided by the receiver's bandwidth:

```python
# weight_sync.py - time to move a fresh policy from trainer to inference engines.
GB = 1e9
cases = {"32B dense bf16": 32.5e9 * 2, "235B MoE bf16": 235e9 * 2, "1T MoE fp8": 1.04e12 * 1}
paths = [  # (name, bytes/s per receiving GPU or per receiving node, what limits it)
    ("CUDA IPC, same GPU (colocated)",           1.5e12, "HBM copy, ~half of 3.35 TB/s r+w"),
    ("NVLink, per GPU",                          450e9,  "1 direction"),
    ("IB 400G per GPU, sharded broadcast",       50e9,   "each GPU receives only its TP shard"),
    ("one 400G NIC, unsharded (naive rank0)",    50e9,   "whole model through one NIC"),
    ("shared FS, 10 GB/s per node read",         10e9,   "per node"),
    ("WAN 10 Gb/s",                              1.25e9, "per node, decentralized"),
    ("WAN 1 Gb/s",                               0.125e9,"per node, decentralized"),
]
import math
for name, M in cases.items():
    tp = max(8, 8 * math.ceil(M / (8 * 60 * GB)))   # GPUs per replica: >=8, <=60 GB weights each
    print(f"\n{name}: {M/GB:.0f} GB, one replica = {tp} GPUs, {M/tp/GB:.1f} GB/GPU")
    for p, bw, why in paths:
        per_gpu = "per GPU" in p or "same GPU" in p or "sharded" in p
        bytes_ = M / tp if per_gpu and "unsharded" not in p else M
        print(f"  {p:40s} {bytes_/bw:9.2f} s   ({why})")
```

```
$ python weight_sync.py

32B dense bf16: 65 GB, one replica = 8 GPUs, 8.1 GB/GPU
  CUDA IPC, same GPU (colocated)                0.01 s   (HBM copy, ~half of 3.35 TB/s r+w)
  NVLink, per GPU                               0.02 s   (1 direction)
  IB 400G per GPU, sharded broadcast            0.16 s   (each GPU receives only its TP shard)
  one 400G NIC, unsharded (naive rank0)         1.30 s   (whole model through one NIC)
  shared FS, 10 GB/s per node read              6.50 s   (per node)
  WAN 10 Gb/s                                  52.00 s   (per node, decentralized)
  WAN 1 Gb/s                                  520.00 s   (per node, decentralized)

235B MoE bf16: 470 GB, one replica = 8 GPUs, 58.8 GB/GPU
  CUDA IPC, same GPU (colocated)                0.04 s   (HBM copy, ~half of 3.35 TB/s r+w)
  NVLink, per GPU                               0.13 s   (1 direction)
  IB 400G per GPU, sharded broadcast            1.18 s   (each GPU receives only its TP shard)
  one 400G NIC, unsharded (naive rank0)         9.40 s   (whole model through one NIC)
  shared FS, 10 GB/s per node read             47.00 s   (per node)
  WAN 10 Gb/s                                 376.00 s   (per node, decentralized)
  WAN 1 Gb/s                                 3760.00 s   (per node, decentralized)

1T MoE fp8: 1040 GB, one replica = 24 GPUs, 43.3 GB/GPU
  CUDA IPC, same GPU (colocated)                0.03 s   (HBM copy, ~half of 3.35 TB/s r+w)
  NVLink, per GPU                               0.10 s   (1 direction)
  IB 400G per GPU, sharded broadcast            0.87 s   (each GPU receives only its TP shard)
  one 400G NIC, unsharded (naive rank0)        20.80 s   (whole model through one NIC)
  shared FS, 10 GB/s per node read            104.00 s   (per node)
  WAN 10 Gb/s                                 832.00 s   (per node, decentralized)
  WAN 1 Gb/s                                 8320.00 s   (per node, decentralized)
```

The bandwidths are rounded hardware numbers: H100 HBM 3.35 TB/s, NVLink 450 GB/s per direction, one 400 Gb/s NIC per GPU = 50 GB/s. Conclusions:

- **Inside a datacenter, the bandwidth term is small** if every GPU receives only its own shard and all GPUs receive in parallel: 0.16 s for 32B, about 1 s for the 1T MoE. Sending everything through one NIC (rank 0 broadcasts the whole model) is 8-24× slower. That is the most common first implementation.
- **Real systems are well above these bounds.** verl's checkpoint-engine README (as of November 2026) reports about 7 s to sync Qwen3-30B-A3B (about 61 GB in bf16) across 4×8 H100s with 400 Gb/s ConnectX-7 over NCCL, quoted as 8.25 GB/s, and 5.93 s over Mooncake on 2×8 H100. That is roughly 6× below one NIC's line rate. The time goes to gathering shards from the trainer layout, dtype casts or quantization, per-tensor bookkeeping, and the engine's weight-loading code. The section below on per-message overhead shows where some of it goes.
- **Over a WAN** (decentralized RL across datacenters or volunteer machines), bandwidth dominates. At 1 Gb/s a 32B model takes nearly 9 minutes per sync, longer than a training step. This is why decentralized RL systems lean on async (generation keeps going on old weights while new ones download), on checkpoint relays or CDNs (prime-rl's INTELLECT-2 report, arXiv 2505.07291, describes an HTTP-based relay for broadcasting checkpoints to untrusted inference workers), and on compressing or sparsifying the update. See [decentralized training](../training/decentralized-training.md).

### Resharding between FSDP and TP layouts

The trainer stores parameters in its own layout, and the engine expects a different one. FSDP2 keeps each parameter as a dim-0 shard per rank. An engine with TP=`t` expects column-parallel weights (q/k/v, gate/up) split by output rows and row-parallel weights (o, down) split by input columns ([TP derivation](../training/model-parallelism/derivations.md#column-and-row-splits)). Engines also **fuse** weights: vLLM loads q, k and v into a single `qkv_proj` and gate and up into `gate_up_proj`. Resharding means gathering each parameter, cutting it the way the engine wants, and fusing per TP rank. Fusing in the wrong order is the classic silent bug:

```python
# reshard.py - FSDP (dim-0 shards over 4 trainer ranks) -> TP=2 inference layout with a fused QKV.
# Checks that the inference-side matmul reproduces the trainer-side one.
import numpy as np
rng = np.random.default_rng(0)
h, nq, nkv, hd = 64, 8, 2, 8            # hidden, q heads, kv heads, head dim
FSDP, TP = 4, 2

# trainer's logical weights, torch Linear layout (out_features, in_features)
Wq = rng.standard_normal((nq * hd, h))  # (64, 64)
Wk = rng.standard_normal((nkv * hd, h)) # (16, 64)
Wv = rng.standard_normal((nkv * hd, h)) # (16, 64)

# FSDP2 stores each param as a dim-0 shard per rank
fsdp = {n: np.array_split(W, FSDP, axis=0) for n, W in dict(q=Wq, k=Wk, v=Wv).items()}

# step 1: gather (on a real system: all-gather over the trainer mesh, or per-rank sends)
full = {n: np.concatenate(shards, axis=0) for n, shards in fsdp.items()}

# step 2a (WRONG): fuse first, then split the fused matrix into TP pieces
fused = np.concatenate([full["q"], full["k"], full["v"]], axis=0)       # (96, 64)
wrong = np.array_split(fused, TP, axis=0)

# step 2b (RIGHT): split each of q/k/v by heads for TP rank r, then fuse per rank
right = [np.concatenate([np.array_split(full[n], TP, axis=0)[r] for n in "qkv"], axis=0)
         for r in range(TP)]

x = rng.standard_normal((3, h))                                          # (B, h)
ref_q = x @ Wq.T                                                         # (B, nq*hd)
for name, shards in (("wrong", wrong), ("right", right)):
    q_dim = nq * hd // TP                                                # q columns per TP rank
    q = np.concatenate([x @ s.T[:, :q_dim] for s in shards], axis=1)     # each rank's q slice
    print(f"{name}: per-rank fused shape {shards[0].shape}, max |q - ref| = {np.abs(q - ref_q).max():.3g}")
```

```
$ python reshard.py
wrong: per-rank fused shape (48, 64), max |q - ref| = 25.4
right: per-rank fused shape (48, 64), max |q - ref| = 1.42e-14
```

Both versions produce tensors of the same shape. Nothing crashes and the model still generates text, just *worse* text. Rollouts get worse, rewards drop, and it looks like an RL instability. Defences: after the first weight sync of every run, check that engine logprobs on a fixed prompt match the trainer's (the mismatch KL check), and unit-test the resharding function against a reference forward, as the script does.

Real implementations differ mainly in *where* the gather happens:

| Approach | How | Cost |
|---|---|---|
| Gather on trainer, send full tensors | trainer all-gathers each param (as FSDP does every forward), rank 0 or a few ranks send | simple. One sender's NIC bounds it, unless you shard the sending |
| Gather per engine shard | each trainer rank sends its slice directly to whichever engine ranks need it (an `M × N` transfer plan) | optimal bandwidth, complex planning. Several frameworks precompute such a plan |
| Same layout both sides | inference uses the trainer's own sharding (e.g. Megatron TP=TP) | no resharding, but constrains both sides |
| Engine reads a checkpoint | trainer writes safetensors to shared storage or serves it over HTTP, engines load | most decoupled and elastic. Storage bandwidth bounds it (see [storage](../foundations/storage-and-data.md)) |

### NCCL broadcast between trainer and engines

To broadcast over NCCL you need a **process group** that contains both trainer and engine ranks. They were not launched together, so they are not in the same `torch.distributed` world. Engines therefore expose an RPC that creates an extra, separate communicator. As of November 2026, SGLang's engine has `init_weights_update_group(...)` plus `update_weights_from_distributed(...)` (and `update_weights_from_tensor`, `update_weights_from_disk`, `update_weights_from_ipc`). vLLM ships worked examples under `examples/rl/` (e.g. `rlhf_http_nccl.py`, `rlhf_http_ipc.py`, `rlhf_nccl_fsdp_ep.py`). OpenRLHF selects the path with `--vllm.sync_backend nccl`. The protocol is the same in all of them. **Untested sketch:**

```python
# trainer side (sketch; real API names differ per engine - read the version you run)
group = init_custom_group(master=trainer0_addr, world_size=1 + n_engine_ranks, rank=0)  # stateless group
for engine in engines:
    engine.rpc("init_weights_update_group", master=trainer0_addr, rank_offset=..., world_size=...)

engines.rpc("pause_generation")             # stop scheduling new tokens (or keep going: in-flight update)
for name, tensor in gathered_params_in_buckets():           # gather FSDP shards, cast to bf16, fuse
    engines.rpc_async("update_weights_from_distributed", names=..., dtypes=..., shapes=...)
    group.broadcast(tensor, src=0)          # engine ranks post matching recvs inside the RPC
engines.rpc("reset_prefix_cache")           # cached KV was computed with the OLD weights
engines.rpc("resume_generation", version=v + 1)
```

Three details in that sketch are easy to miss:

- **Bucketing.** Every collective pays a fixed latency `α` (see the [cost model](../training/model-parallelism/derivations.md#collective-cost-models-alpha-beta-ring)). A large model has thousands of tensors (norm weights, biases, per-expert weights), and broadcasting them one by one is latency-bound. Flatten them into buckets of tens to hundreds of MB. The effect is visible even on CPU with gloo:

```python
# bcast.py - trainer rank 0 broadcasts a "state dict" to 3 inference ranks (gloo, CPU).
# Compares one broadcast per tensor vs flattening into 64 MB buckets.
import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp

def run(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    # 4000 tensors of 4096 fp32 (16 KB) = 64 MB, like many small norm/bias + medium matrices
    params = [torch.full((4096,), float(rank == 0)) for _ in range(4000)]
    for mode in ("per-tensor", "bucketed"):
        dist.barrier(); t0 = time.perf_counter()
        if mode == "per-tensor":
            for p in params: dist.broadcast(p, src=0)
        else:
            per_bucket = 64 * 2**20 // (4096 * 4)          # tensors per 64 MB bucket
            for i in range(0, len(params), per_bucket):
                chunk = params[i:i + per_bucket]
                flat = torch.cat(chunk) if rank == 0 else torch.empty(sum(c.numel() for c in chunk))
                dist.broadcast(flat, src=0)
                if rank:
                    for c, f in zip(chunk, flat.split(4096)): c.copy_(f)
        dist.barrier(); dt = time.perf_counter() - t0
        ok = all(bool((p == 1).all()) for p in params)
        if rank == 1: print(f"{mode:10s}: {dt*1e3:7.1f} ms  {64/dt:6.0f} MB/s  weights correct on rank 1: {ok}")
        for p in params: p.fill_(float(rank == 0))
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, args=(4,), nprocs=4)
```

```
$ GLOO_SOCKET_IFNAME=lo0 python bcast.py
per-tensor:   647.4 ms      99 MB/s  weights correct on rank 1: True
bucketed  :    37.9 ms    1688 MB/s  weights correct on rank 1: True
```

On this run, 4000 small broadcasts were 17× slower than one 64 MB broadcast. (Other runs gave between 17× and 38×. The absolute numbers are loopback TCP and mean nothing for NCCL, but the ratio is the α term.) The first version of this script aborted with a gloo `Unexpected opcode` error on macOS. It ran after I pinned the socket interface with `GLOO_SOCKET_IFNAME=lo0` and fixed a copy bug in the same edit, so I did not isolate which change fixed it.

- **Prefix-cache invalidation.** Prefix caching ([serving internals](../inference/serving-engine-internals.md#prefix-caching)) reuses KV blocks keyed by token hashes. After a weight update those blocks hold KV computed with the old weights, so the engine must flush them (vLLM's `reset_prefix_cache`). If it doesn't, the next rollouts quietly mix versions. GRPO makes heavy use of prefix caching, since `G` samples share a prompt, so this matters.
- **Version tagging.** Every rollout must record the version of the weights that produced each token. With in-flight updates that means a per-token version.

### Colocated: CUDA IPC

When trainer and engine share a GPU, there is no need to copy bytes over a network. CUDA IPC (inter-process communication) lets one process export a handle to a GPU allocation, and another process on the same GPU maps the same memory. The trainer gathers a parameter into a GPU buffer, sends the IPC handle (a few bytes) to the engine process over a local socket, and the engine copies from the mapped buffer into its own weight tensor at HBM speed. PyTorch exposes this through `torch.multiprocessing` tensor sharing (`reduce_tensor` / `rebuild_cuda_tensor` under the hood). OpenRLHF, verl and SGLang (`update_weights_from_ipc`) all use variants of it. **Untested** (needs CUDA):

```python
# CUDA IPC handoff between two processes on the same GPU (sketch; needs CUDA)
import torch, torch.multiprocessing as mp
from torch.multiprocessing.reductions import reduce_tensor

def engine(conn):
    fn, args = conn.recv()                    # the "handle": a rebuild function + small args
    w = fn(*args)                             # maps the trainer's allocation; no copy yet
    engine_weight.copy_(w)                    # device-to-device copy at HBM bandwidth
    conn.send("done")                         # trainer must not free the buffer before this

if __name__ == "__main__":
    mp.set_start_method("spawn")
    a, b = mp.Pipe(); p = mp.Process(target=engine, args=(b,)); p.start()
    full = gather_full_param("model.layers.0.self_attn.q_proj.weight").cuda()   # (out, in)
    a.send(reduce_tensor(full)); assert a.recv() == "done"
```

The protocol detail that matters is **lifetime**. The trainer must keep `full` alive until the engine confirms the copy. If the trainer frees it and the caching allocator reuses the memory, the engine reads garbage. Nothing errors, and logprobs are simply wrong.

### Over the network: checkpoint streaming

When engines are elastic (they join and leave), live in another datacenter, or are not trusted, a shared NCCL group is the wrong tool. A single failed member hangs a collective (see [distributed systems](../foundations/distributed-systems.md)). The alternative is the same method you would use to distribute any large file: the trainer writes a versioned checkpoint, and engines pull it. A minimal HTTP version, runnable with the stdlib:

```python
# ckpt_stream.py - trainer publishes versioned shards over HTTP; an engine polls and pulls the newest.
import hashlib, http.server, json, os, tempfile, threading, urllib.request

root = tempfile.mkdtemp()
def publish(version, shards):                        # trainer side, after each step
    d = os.path.join(root, f"step_{version}"); os.makedirs(d)
    manifest = {}
    for name, blob in shards.items():
        open(os.path.join(d, name), "wb").write(blob)
        manifest[name] = hashlib.sha256(blob).hexdigest()
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"))
    open(os.path.join(root, "LATEST.tmp"), "w").write(str(version))
    os.replace(os.path.join(root, "LATEST.tmp"), os.path.join(root, "LATEST"))   # atomic: last thing written

handler = lambda *a: http.server.SimpleHTTPRequestHandler(*a, directory=root)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}"

def pull(have):                                      # engine side
    v = int(urllib.request.urlopen(f"{url}/LATEST").read())
    if v <= have: return have, None
    man = json.loads(urllib.request.urlopen(f"{url}/step_{v}/manifest.json").read())
    shards = {n: urllib.request.urlopen(f"{url}/step_{v}/{n}").read() for n in man}
    assert all(hashlib.sha256(b).hexdigest() == man[n] for n, b in shards.items()), "corrupt shard"
    return v, shards

publish(1, {"shard0.bin": os.urandom(8 << 20), "shard1.bin": os.urandom(8 << 20)})
v, s = pull(0); print("pulled version", v, {k: len(b) for k, b in s.items()})
print("second poll:", pull(v)[0], "(nothing new)")
```

```
$ python ckpt_stream.py
pulled version 1 {'shard0.bin': 8388608, 'shard1.bin': 8388608}
second poll: 1 (nothing new)
```

The design points carry over to real systems. The `LATEST` pointer is written last and atomically (`os.replace`), so readers never see a half-written version. Every shard has a checksum, because WAN transfers do corrupt data and a corrupt shard gives a broken policy with no error. Old versions are kept until no engine is reading them. To scale out, fan out through relays or peer-to-peer (each engine re-serves what it downloaded) instead of having a thousand engines pull from one trainer NIC. This is the same reasoning as a pipelined broadcast tree.

## Rollout workers

### Engines as rollout workers

RL uses an inference engine differently from chat serving:

- **Token-in, token-out.** The orchestrator should send token ids and receive token ids plus per-token logprobs, not text. Text round-trips break things (see [tokenization](#chat-templates-and-tokenization-consistency)). vLLM and SGLang both accept token-id prompts and return sampled-token logprobs, and prime-rl drives a token-in `/generate` route on its vLLM server (as of November 2026).
- **Throughput, not latency.** There is no human waiting on time-to-first-token. Batch as large as KV memory allows and accept long per-token latency.
- **Sampling must be exactly what the loss assumes.** Record the temperature, top-p, top-k and the logprob definition (raw vs processed) together with the rollout.

A request to an OpenAI-compatible vLLM server. The flags follow vLLM's API as I know it; check against the version you run. **Untested:**

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "policy", "prompt": [151644, 872, 198, 3838, 374, 220, 17, 10, 18, 30],
  "max_tokens": 4096, "temperature": 1.0, "top_p": 1.0, "n": 16,
  "logprobs": 0, "return_tokens_as_token_ids": true
}' | jq '.choices[0] | {finish_reason, ntok: (.logprobs.tokens | length)}'
# expected shape: {"finish_reason": "stop" | "length", "ntok": 1834}
```

`n=16` asks the engine for the whole GRPO group in one request. The engine prefills the prompt once and the 16 samples share its KV blocks through prefix caching.

### Continuous batching and the long tail

Continuous batching ([serving internals](../inference/serving-engine-internals.md#continuous-batching-and-the-scheduler)) keeps one engine's batch full by admitting new sequences as old ones finish. In synchronous RL there are no new sequences to admit at the end of a step, and that is exactly the tail in `step_budget.py`. The systems responses:

- **Feed the engine continuously** (async RL): the orchestrator always has more prompts queued, so the engine never drains.
- **Over-sample**: request, say, 1.25× the prompts the batch needs and stop when enough groups have finished. Abort the rest, or carry them over. DAPO's dynamic sampling (arXiv 2503.14476) over-samples for a different reason, to discard groups whose rewards have zero variance, but the infrastructure is the same: the batch is formed from *completed* groups and the rest is surplus. OpenRLHF's `--rollout.vllm_generate_batch_size` is this knob.
- **Balance work across engines by tokens, not requests.** Assign whole groups to one engine (for prefix sharing) and route to the engine with the fewest outstanding *tokens*.

```python
# least-outstanding-tokens routing for GRPO groups (runnable)
import heapq, random
random.seed(0)
engines = [(0, i) for i in range(4)]                  # (outstanding_tokens, engine_id) min-heap
heapq.heapify(engines); load = [0] * 4
for g in range(64):                                   # 64 groups, each G=16 samples of one prompt
    expected = 16 * random.choice([500, 2000, 8000])  # e.g. from the prompt's past response lengths
    tok, eid = heapq.heappop(engines)
    load[eid] += expected; heapq.heappush(engines, (tok + expected, eid))
print("tokens per engine:", load, f"imbalance {max(load)/min(load):.2f}x")
```

```
$ python route.py
tokens per engine: [896000, 976000, 1008000, 944000] imbalance 1.12x
```

Greedy assignment leaves 1.12x imbalance here, because groups are large and few. It also balances only *expected* tokens. Actual lengths are only revealed during generation, so a balancer that assigns everything up front cannot be perfect. Continuous refill, where engines pull work as they free up, absorbs the rest of the variance.

### Partial rollouts

Kimi k1.5 (arXiv 2501.12599) describes **partial rollouts**: each iteration generates at most a fixed token budget per trajectory. Unfinished trajectories are saved (prompt + tokens so far) and continued in the next iteration, possibly under newer weights. Long responses then stop dictating iteration time, at the cost of multi-version trajectories. A runnable sketch of the bookkeeping:

```python
# partial_rollouts.py - cap tokens per iteration, carry the rest over (bookkeeping only)
import random
random.seed(0)
BUDGET, NEW_PER_ITER = 2048, 32
carry = []                                     # [prompt_id, target_len, done_tokens, versions]
next_prompt = 0
for it in range(6):
    while len(carry) < NEW_PER_ITER + len([c for c in carry if c[2] > 0]):   # top up with new prompts
        carry.append([next_prompt, min(int(random.lognormvariate(7.5, 1.0)), 32768), 0, []]); next_prompt += 1
    finished, still = [], []
    for c in carry:
        n = min(BUDGET, c[1] - c[2]); c[2] += n; c[3].append(it)   # tokens generated this iter, at version it
        (finished if c[2] == c[1] else still).append(c)
    carry = still
    multi = sum(len(set(c[3])) > 1 for c in finished)
    print(f"iter {it}: finished {len(finished):2d}, carried {len(carry):2d}, "
          f"finished-with-multiple-versions {multi}, max tokens/iter/traj {BUDGET}")
```

```
$ python partial_rollouts.py
iter 0: finished 17, carried 15, finished-with-multiple-versions 0, max tokens/iter/traj 2048
iter 1: finished 27, carried 20, finished-with-multiple-versions 10, max tokens/iter/traj 2048
iter 2: finished 28, carried 24, finished-with-multiple-versions 12, max tokens/iter/traj 2048
iter 3: finished 32, carried 24, finished-with-multiple-versions 11, max tokens/iter/traj 2048
iter 4: finished 26, carried 30, finished-with-multiple-versions 13, max tokens/iter/traj 2048
iter 5: finished 27, carried 35, finished-with-multiple-versions 13, max tokens/iter/traj 2048
```

Training batches would be formed from the `finished` trajectories. Watch the `carried` column: it grows from 15 to 35 because each iteration adds 32 new prompts while long trajectories pile up, and from iteration 1 onward 10-13 of the finished trajectories were sampled by more than one policy version. Real systems cap the carry-over pool or prioritise finishing old trajectories. If the carry-over pool grows without bound, that is a queue without backpressure, and the oldest entries become arbitrarily stale. Implementation-wise, continuing a trajectory means re-prefilling `prompt + tokens_so_far` on whichever engine picks it up. That is cheap with prefix caching if the same engine is reused and weights have not changed, and otherwise a full prefill.

## Agentic and multi-turn rollouts

### The environment as code

For agentic RL a rollout is a **program**, not a single generate call. The environment interleaves model turns with tool calls (run code, search, edit a file), may span tens of turns and hundreds of thousands of tokens, and each tool call may take seconds to minutes. The rollout worker is now an `async` function that drives one episode. Thousands of such coroutines share a few inference engines and a sandbox fleet. The budgets that keep this from melting down:

- **Max turns** and a **context budget** per episode (the engine's max context, minus headroom).
- **Per-tool-call timeout**: a hung tool returns an error observation instead of hanging the episode.
- **Per-episode wall-clock timeout**: a final guard against anything else that hangs.
- **Per-turn token accounting**: which tokens were sampled (and are trained on) and which were injected by the environment.

```python
# agent_loop.py - one multi-turn episode: model turn -> tool call -> env turn -> ..., with budgets.
# The "model" and "tool" are fakes; the accounting (ids, loss mask, budgets, timeouts) is the point.
import asyncio, random, json

MAX_TURNS, MAX_CTX, TOOL_TIMEOUT_S, EPISODE_TIMEOUT_S = 6, 600, 0.2, 2.0

async def fake_generate(ctx_len, rng):                 # stands in for POST /generate on the engine
    await asyncio.sleep(0.01)
    n = rng.randint(20, 120)
    call = rng.random() < 0.7                          # 70% of turns end in a tool call
    return list(range(n)), ({"tool": "python", "code": "..."} if call else None)

async def fake_tool(call, rng):                        # stands in for a sandbox exec
    await asyncio.sleep(rng.choice([0.01, 0.01, 0.01, 0.5]))   # 1 in 4 calls hangs
    return list(range(rng.randint(10, 200)))           # tool output tokens

async def episode(seed):
    rng = random.Random(seed)
    ids, mask, turns, log = list(range(50)), [0] * 50, 0, []   # 50-token prompt, not trained
    stop = "max_turns"
    while turns < MAX_TURNS:
        out, call = await fake_generate(len(ids), rng)
        out = out[: MAX_CTX - len(ids)]                        # never exceed the context budget
        ids += out; mask += [1] * len(out); turns += 1
        log.append(("model", len(out)))
        if len(ids) >= MAX_CTX: stop = "ctx_budget"; break
        if call is None: stop = "answered"; break
        try:
            obs = await asyncio.wait_for(fake_tool(call, rng), TOOL_TIMEOUT_S)
        except asyncio.TimeoutError:
            obs = list(range(8))                               # "<tool timed out>" as a short observation
            log.append(("tool_timeout", len(obs)))
        else:
            log.append(("tool", len(obs)))
        obs = obs[: MAX_CTX - len(ids)]
        ids += obs; mask += [0] * len(obs)                     # env tokens: in context, not in loss
        if len(ids) >= MAX_CTX: stop = "ctx_budget"; break
    return dict(seed=seed, stop=stop, turns=turns, total=len(ids), trained=sum(mask), log=log)

async def main():
    eps = await asyncio.gather(*(asyncio.wait_for(episode(s), EPISODE_TIMEOUT_S) for s in range(200)))
    for e in eps[:3]: print(json.dumps(e))
    from collections import Counter
    print("stop reasons:", dict(Counter(e["stop"] for e in eps)))
    print("timeouts per episode:", sum(t == "tool_timeout" for e in eps for t, _ in e["log"]) / len(eps))
    print(f"trained-token fraction: {sum(e['trained'] for e in eps) / sum(e['total'] for e in eps):.2f}")
    assert all(e["total"] <= MAX_CTX for e in eps)
asyncio.run(main())
```

```
$ python agent_loop.py
{"seed": 0, "stop": "answered", "turns": 1, "total": 119, "trained": 69, "log": [["model", 69]]}
{"seed": 1, "stop": "ctx_budget", "turns": 5, "total": 600, "trained": 309, "log": [["model", 37], ["tool", 75], ["model", 35], ["tool_timeout", 8], ["model", 80], ["tool", 34], ["model", 82], ["tool_timeout", 8], ["model", 75], ["tool", 188]]}
{"seed": 2, "stop": "answered", "turns": 2, "total": 244, "trained": 141, "log": [["model", 27], ["tool", 53], ["model", 114]]}
stop reasons: {'answered': 145, 'ctx_budget': 55}
timeouts per episode: 0.435
trained-token fraction: 0.49
```

Observations:

- **27.5% of episodes (55/200) ended on the context budget, not by answering.** In real agentic RL, the truncation rate is a first-class metric. Truncated episodes usually get zero or undefined reward, and a rising truncation rate looks like the policy getting worse when it is really an infrastructure limit.
- **Half the tokens in context are not trained on** (trained fraction 0.49): prompt plus tool outputs. They still cost prefill compute and KV memory on the engine, and forward/backward compute on the trainer, because the model must attend to them. Size the trainer for total tokens, not trained tokens.
- **Tool timeouts are frequent** (0.44 per episode here, by construction). Whether a timeout is shown to the model as an observation, ends the episode, or invalidates the sample is a design decision with reward consequences. Make it explicit and log it.

Libraries like `verifiers` (Prime Intellect), verl's `agent_loop`, OpenRLHF's `AgentInstanceBase` (`reset` / `step`) and SkyRL's environment interface package this loop so that environment authors write only the `step` logic. The [survey](#open-frameworks-survey) below lists them.

### Token accounting and loss masking

The trainer needs three aligned arrays per episode: `input_ids` (everything in context), `loss_mask` (1 for tokens the policy sampled), and `sampler_logprobs` (for those tokens). The robust way to build them is **token-in, token-out** throughout. Keep the exact ids the engine returned. Tokenize environment text yourself, append it, and never re-tokenize the concatenated text. Re-tokenizing is the source of most of the bugs in the next section.

```python
# masked policy-gradient loss over a packed multi-turn episode (torch; ran on CPU)
import torch
T = 12
logp_new = torch.randn(T, requires_grad=True)              # (T,) trainer logprobs of input_ids[1:]
logp_old = logp_new.detach() + 0.01 * torch.randn(T)       # (T,) sampler logprobs (only valid where mask=1)
mask = torch.tensor([0,0,0,1,1,1,0,0,1,1,1,1.])            # (T,) prompt/tool tokens = 0
adv = torch.full((T,), 0.7)                                # (T,) sequence advantage broadcast to tokens
ratio = torch.exp(logp_new - logp_old)                     # (T,)
loss = -(torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv) * mask).sum() / mask.sum()
loss.backward()
print("grad nonzero only where mask=1:", torch.equal(logp_new.grad != 0, mask.bool()))
```

```
$ python masked_loss.py
grad nonzero only where mask=1: True
```

The normalisation (`/ mask.sum()` per sequence, per batch, or a constant) is an algorithm choice, but one with infrastructure consequences. With sequence packing, "per sequence" requires the packer to keep sequence boundaries alongside the mask.

### Chat templates and tokenization consistency

The importance ratio is only meaningful if the trainer scores **the exact token ids the sampler produced**, in **the exact context it produced them in**. Three ways to break that, demonstrated with a real tokenizer (Qwen3, loaded offline from the local HF cache):

```python
# tokcheck.py - three ways trainer and sampler disagree about "the same" text. Qwen3 tokenizer, offline.
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

# 1. Round-trip drift: the sampler emitted a non-canonical split; decode -> re-encode changes the ids.
canon = tok.encode(" unbelievable", add_special_tokens=False)
sampled = tok.encode(" unbel", add_special_tokens=False) + tok.encode("ievable", add_special_tokens=False)
text = tok.decode(sampled)
print("1. canonical ids", canon, "| sampled ids", sampled)
print("   decode(sampled) == decode(canon):", text == tok.decode(canon),
      "| re-encode(decode(sampled)) == sampled:", tok.encode(text, add_special_tokens=False) == sampled)

# 2. Multi-turn template: is turn 2's prompt an extension of what was actually sampled in turn 1?
msgs = [{"role": "user", "content": "What is 2+3?"}]
p1 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
completion = "<think>\n2+3=5\n</think>\n\nThe answer is 5.<|im_end|>"
msgs += [{"role": "assistant", "content": "<think>\n2+3=5\n</think>\n\nThe answer is 5."},
         {"role": "user", "content": "Times 4?"}]
p2 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
seen = p1 + completion
print("2. turn-2 prompt starts with turn-1 prompt+completion:", p2.startswith(seen))
i = next(i for i, (a, b) in enumerate(zip(seen, p2)) if a != b)
print(f"   first difference at char {i}: sampled {seen[i:i+30]!r} vs re-rendered {p2[i:i+30]!r}")

# 3. Token-in/token-out: build the episode from ids, and mask env/tool tokens out of the loss.
ids, mask = [], []
def add(t, train):  # t: str for env/template text, list[int] for sampled tokens (kept verbatim)
    new = t if isinstance(t, list) else tok.encode(t, add_special_tokens=False)
    ids.extend(new); mask.extend([int(train)] * len(new))
add(p1, False)
add(tok.encode(completion, add_special_tokens=False), True)        # stands in for the sampler's ids
add("\n<|im_start|>user\n<tool_response>\n5 * 4 = 20\n</tool_response><|im_end|>\n<|im_start|>assistant\n", False)
add(tok.encode("20.<|im_end|>", add_special_tokens=False), True)
print(f"3. episode {len(ids)} tokens, {sum(mask)} trained, {len(ids)-sum(mask)} masked (prompt + tool output)")
print("   trained text:", repr(tok.decode([t for t, m in zip(ids, mask) if m])))
```

```
$ python tokcheck.py
1. canonical ids [51129] | sampled ids [650, 9779, 645, 23760]
   decode(sampled) == decode(canon): True | re-encode(decode(sampled)) == sampled: False
2. turn-2 prompt starts with turn-1 prompt+completion: False
   first difference at char 62: sampled '<think>\n2+3=5\n</think>\n\nThe an' vs re-rendered 'The answer is 5.<|im_end|>\n<|i'
3. episode 57 tokens, 21 trained, 36 masked (prompt + tool output)
   trained text: '<think>\n2+3=5\n</think>\n\nThe answer is 5.<|im_end|>20.<|im_end|>'
```

1. **Round-trip drift.** The sampler is free to emit a non-canonical tokenization (` unbel` + `iev` + `able`, 4 tokens) of text that the tokenizer would encode as 1 token. Decoding gives identical text, but re-encoding gives different ids. If your pipeline passes *text* from engine to trainer, the trainer computes logprobs for tokens that were never sampled. The ratio for that span is meaningless, and nothing raises an error.
2. **Template non-prefix property.** Qwen3's chat template strips the `<think>` block from earlier assistant turns when it renders a conversation. Turn 2's prompt, re-rendered from messages, is *not* turn 1's prompt plus what the model actually generated: the reasoning has disappeared from the context. If the trainer re-renders the conversation from messages, it trains on a context the sampler never saw. If the engine re-renders it for turn 2, the model's own reasoning is hidden from it on later turns. Either can be intended, but it has to be the *same* choice on both sides. Other templates differ in their own ways (system-prompt injection, tool-call formatting, trailing newlines).
3. **The fix** is the token-in/token-out construction in part 3: build the episode from ids, mark which ones were sampled, and keep env/template text masked.

The check worth adding to CI and to the first step of every run: take a few real rollouts, rebuild the trainer's `input_ids` the way the trainer will, and assert they are equal to the engine's `prompt_ids + output_ids`. Then compare trainer and sampler logprobs on those ids and assert the mismatch KL is small. Most "RL is unstable on model X" reports that turn out to be infrastructure bugs would have been caught by this.

## Environments, verifiers and rewards as services

### Reward types and what they cost

| Reward | Example | Latency per rollout | Scaling problem |
|---|---|---|---|
| Rule-based | exact match on a boxed math answer, regex, JSON schema check | µs-ms, in-process | none. Run it inline |
| Programmatic, with execution | run unit tests on generated code, compile, run a SQL query | 10 ms - minutes | needs **sandboxes**: isolation, a fleet, cold starts |
| Model-based | LLM-as-judge, reward model, rubric grading | 100 ms - 10 s | it is **another inference fleet**: GPUs, batching, its own weights |
| Environment outcome | did the agent's edits make the repo's tests pass | minutes | per-task container images, long episodes |

The infrastructure rule: **reward computation must not be on the trainer's critical path.** Score each rollout as soon as it finishes, concurrently with ongoing generation, so that by the time a batch is complete its rewards are too. Treat the verifier as a service with a bounded queue, a concurrency limit and a timeout, like any other.

### Sandboxed code execution

Model-written code is **untrusted input**. It will, sometimes by accident and sometimes because RL rewards it, loop forever, fork-bomb, fill the disk, flood stdout, read the test file to find expected answers, open network connections, or kill its parent. The minimum single-machine harness: a fresh working directory, a child process in its own process group, CPU/file/fd limits, a wall-clock timeout, bounded output capture, and killing the *whole process group* on timeout:

```python
# verify.py - a reward service's inner loop: run untrusted model-written code with limits, many at once.
import asyncio, os, resource, sys, tempfile, time, collections

GOOD  = "def add(a, b): return a + b"
WRONG = "def add(a, b): return a - b"
HANG  = "def add(a, b):\n    while True: pass"
BOMB  = "def add(a, b):\n    print('x' * 10**8)"            # floods stdout
CRASH = "import os; os._exit(3)"
TESTS = "\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('PASS')"

def limits():                                            # runs in the child before exec
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))      # 2 s of CPU, then SIGXCPU
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))  # 1 MB max file write
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.setsid()                                          # own process group so we can kill all of it

async def verify(code, sem, wall=3.0, max_out=1 << 16):
    async with sem:                                      # bounded concurrency = backpressure
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sol.py")
            with open(path, "w") as f: f.write(code + TESTS)
            p = await asyncio.create_subprocess_exec(sys.executable, "-I", path, cwd=d,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    stdin=asyncio.subprocess.DEVNULL, preexec_fn=limits)
            async def drain():                           # read a bounded prefix, discard the rest:
                out = await p.stdout.read(max_out)       # asyncio's wait() only returns once the
                while await p.stdout.read(1 << 16): pass # pipe is closed, so it must be drained
                return out, await p.wait()
            try:
                out, rc = await asyncio.wait_for(drain(), wall)
            except asyncio.TimeoutError:
                os.killpg(p.pid, 9)                      # whole process group, not just the child
                while await p.stdout.read(1 << 16): pass
                await p.wait()
                return "timeout", 0.0
            if rc == 0 and out.strip().endswith(b"PASS"): return "pass", 1.0
            return ("killed" if rc < 0 else f"exit{rc}"), 0.0

async def main(n=200, conc=8):
    pool = [GOOD] * 6 + [WRONG] * 2 + [HANG, BOMB, CRASH]
    jobs = [pool[i % len(pool)] for i in range(n)]
    sem = asyncio.Semaphore(conc)
    t0 = time.perf_counter()
    res = await asyncio.gather(*(verify(c, sem) for c in jobs))
    dt = time.perf_counter() - t0
    print(f"{n} verifications, concurrency {conc}: {dt:.1f} s, {n/dt:.1f} verifications/s")
    print(collections.Counter(s for s, _ in res))
    print(f"mean reward {sum(r for _, r in res)/n:.3f} (expected {6/11:.3f})")

if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 8))
```

```
$ python verify.py 110 8 ; python verify.py 110 32
110 verifications, concurrency 8: 4.9 s, 22.6 verifications/s
Counter({'pass': 60, 'exit1': 30, 'killed': 10, 'exit3': 10})
mean reward 0.545 (expected 0.545)
110 verifications, concurrency 32: 3.5 s, 31.8 verifications/s
Counter({'pass': 60, 'exit1': 30, 'exit3': 10, 'killed': 7, 'timeout': 3})
mean reward 0.545 (expected 0.545)
```

Points from this output and from writing the code:

- **Every failure mode ends with reward 0, and the harness itself survives.** The infinite loop is killed by `RLIMIT_CPU` (`killed`). The output flood is truncated at 64 KB. The crash returns its exit code. The mean reward comes out exactly at the expected 6/11.
- **Wall-clock timeouts become flaky under load.** At concurrency 32 on 10 cores, three of the ten hung programs hit the 3 s *wall* timeout before reaching 2 s of *CPU* time, because they were time-sliced with other jobs. (In an earlier run of the same command it was 9 of 10.) Here both outcomes give reward 0, so nothing changes. For a *correct but slow* solution, though, the outcome depends on how loaded the sandbox host is, and that is **reward noise produced by the infrastructure**. Use CPU-time limits or fixed per-job CPU allocations (cgroup quotas, see [containers](../foundations/containers-and-virtualization.md)) for anything that decides correctness, and keep the wall-clock limit as a last-resort backstop.
- **asyncio subprocess gotcha (found while writing this).** Since Python 3.12, `Process.wait()` does not return until the child's pipes are closed. The first version of this script read a bounded prefix of stdout, killed the child, then called `wait()`, and hung forever because unread data stayed in the pipe. The fix is the `drain()` loop: keep reading and discarding until EOF. In a production reward service this bug shows up as a slow leak of stuck verification tasks.
- **This is not a security boundary.** `setrlimit` plus a temp directory stops accidents, not adversaries. It does not stop network access, reading other files on the host, or kernel exploits. `RLIMIT_AS` (memory) is not enforced on macOS at all. On Linux, use real isolation layers. From weakest to strongest: namespaces + cgroups + seccomp (containers, nsjail), a user-space kernel (gVisor), a microVM (Firecracker, Kata). See [containers and virtualization](../foundations/containers-and-virtualization.md) for what each one actually isolates.

A container-based version of the same harness would look like this. **Untested**: Docker was not used for this chapter. The flags are standard `docker run` options.

```bash
docker run --rm --network none --read-only --tmpfs /work:rw,size=64m \
  --memory 512m --memory-swap 512m --cpus 1 --pids-limit 64 \
  --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 \
  -v "$PWD/sol.py:/work/sol.py:ro" -w /work python:3.12-slim \
  timeout -s KILL 10 python -I sol.py
# expected: "PASS" on stdout and exit 0 for a correct solution; 137 (SIGKILL) on timeout
```

### Sandbox fleets: throughput and cold starts

The fleet size follows from [Little's law](../inference/arithmetic.md#littles-law), `concurrency = arrival rate × time in system`:

```python
# sandbox_fleet.py - how many sandboxes does the reward side need?
rollouts_per_step, step_s = 4096, 935            # from the 32B example (partial-rollout step time)
tests_per_rollout = 1                            # one sandbox session runs the whole test file
for exec_s, cold_s, label in ((2, 0.0, "warm pool, 2 s tests"), (2, 1.0, "cold container start 1 s"),
                              (30, 0.0, "heavier repo tests, 30 s"), (30, 15.0, "per-task image pull/start 15 s")):
    rate = rollouts_per_step * tests_per_rollout / step_s     # sandbox sessions per second (average)
    conc = rate * (exec_s + cold_s)
    print(f"{label:32s} {rate:5.1f}/s x {exec_s + cold_s:5.1f} s = {conc:6.1f} concurrent sandboxes (avg)")
```

```
$ python sandbox_fleet.py
warm pool, 2 s tests               4.4/s x   2.0 s =    8.8 concurrent sandboxes (avg)
cold container start 1 s           4.4/s x   3.0 s =   13.1 concurrent sandboxes (avg)
heavier repo tests, 30 s           4.4/s x  30.0 s =  131.4 concurrent sandboxes (avg)
per-task image pull/start 15 s     4.4/s x  45.0 s =  197.1 concurrent sandboxes (avg)
```

These are *averages*. Completions arrive in bursts (a whole group finishes together, a whole batch at the end of a synchronous step), so provision for the peak, or put a queue in front and accept the extra latency. The levers:

- **Warm pools**: keep pre-started sandboxes and hand one out per job, so per-job cost is reset time rather than boot time.
- **Snapshot/restore**: microVM snapshots restore in tens to hundreds of ms (Firecracker's published goal is fast boot; measure your own setup).
- **Image locality**: for SWE-style tasks every task has its own multi-GB image (for example, `swebench/sweb.eval.*` images are around 4 GB each). Pre-pull to nodes and schedule jobs where the image already is. Pulling on demand is the cold-start cost.
- **Multi-turn agents hold a sandbox for the whole episode**, not per call, so `time in system` is the episode duration. This is where hosted sandbox services and Kubernetes-based fleets ([Kubernetes](../orchestration/kubernetes-from-zero.md)) come in.

### Reward-computation throughput

Reward work should be overlapped with generation and measured like everything else. The key per-step metrics are: reward latency p50/p99, verifier queue depth, timeouts per 1k rollouts, and the fraction of rollouts whose reward is missing or non-finite. A model-based judge is an inference deployment with its own capacity plan. Use [inference arithmetic](../inference/arithmetic.md#worked-sizing-example) for it, and remember that it competes with the policy for GPUs if you colocate it.

```python
# reward service with overlap: score as rollouts finish, not after the batch (runnable)
import asyncio, random, time
async def generate(i): await asyncio.sleep(random.lognormvariate(-2.5, 0.8)); return i
async def score(i):    await asyncio.sleep(0.05); return float(i % 3 == 0)
async def main(overlap):
    random.seed(0); t0 = time.perf_counter(); sem = asyncio.Semaphore(16)
    async def one(i):
        r = await generate(i)
        if overlap:
            async with sem: return await score(r)
        return r
    out = await asyncio.gather(*(one(i) for i in range(256)))
    if not overlap:                                         # score after the whole batch is done
        async def s(r):
            async with sem: return await score(r)
        out = await asyncio.gather(*(s(r) for r in out))
    print(f"overlap={overlap}: {time.perf_counter()-t0:.2f} s, mean reward {sum(out)/len(out):.3f}")
asyncio.run(main(False)); asyncio.run(main(True))
```

```
$ python reward_overlap.py
overlap=False: 1.40 s, mean reward 0.336
overlap=True: 0.83 s, mean reward 0.336
```

## Data flow, queues and backpressure

### The pipeline as queues

An async RL system is a chain of stages connected by queues:

```
prompt source -> [Q1] -> generators -> [Q2] -> verifiers -> [Q3] -> batcher -> [Q4] -> trainer
                                                                                         |
                       generators <---------------- weights (version v) ----------------+
```

Every queue must be **bounded**, and every bound is a policy decision. A bound on Q4 is a bound on in-queue staleness: a batch can be at most `|Q4|` steps old when the trainer pulls it. A bound on Q1 together with the number of generator slots bounds in-flight work, and with it in-flight staleness and KV memory. A bound on Q2 protects the verifier fleet. Unbounded queues turn a slow consumer into an OOM, or worse, into silently growing staleness. The monitoring rule from [concurrency](../foundations/concurrency.md#producerconsumer-and-backpressure) applies: queues near full mean the downstream stage is the bottleneck, queues near empty mean the upstream stage is.

```python
# queue_depth.py - which stage is the bottleneck? Read the queue depths. (runnable)
import asyncio
async def stage(name, inq, outq, sec, n):
    for _ in range(n):
        x = await inq.get(); await asyncio.sleep(sec)
        if outq: await outq.put(x)
async def main():
    q1, q2, q3 = (asyncio.Queue(maxsize=8) for _ in range(3))
    async def src():
        for i in range(60): await q1.put(i)
    async def monitor():
        for _ in range(5):
            await asyncio.sleep(0.2); print(f"depth  gen-in {q1.qsize()}  verify-in {q2.qsize()}  train-in {q3.qsize()}")
    await asyncio.gather(src(), stage("gen", q1, q2, 0.005, 60), stage("verify", q2, q3, 0.02, 60),
                         stage("train", q3, None, 0.005, 60), monitor())
asyncio.run(main())
```

```
$ python queue_depth.py
depth  gen-in 8  verify-in 8  train-in 0
depth  gen-in 8  verify-in 8  train-in 0
depth  gen-in 8  verify-in 8  train-in 0
depth  gen-in 8  verify-in 8  train-in 0
depth  gen-in 3  verify-in 8  train-in 0
```

The verifier is the slow stage (20 ms vs 5 ms): the queue in front of it is full, the queue after it is empty, and backpressure holds the generator at the verifier's pace. Scaling the generators would change nothing.

### Batch formation

The batcher between the verifiers and the trainer has several jobs:

- **Group completeness.** GRPO-style advantages need all `G` samples of a prompt. Batch by completed *groups*, not by individual rollouts. A group with one stuck sample blocks the whole group, so put a timeout on groups too: drop the group, or accept `G-1` samples if your estimator allows it.
- **Filtering.** Drop zero-variance groups (all rewards identical, so every advantage is 0), groups with NaN rewards, and samples beyond the staleness limit. Filtering shrinks the batch, so either over-sample upstream or allow variable batch sizes. Each choice has different loss-normalisation consequences.
- **Packing.** Concatenate variable-length sequences into fixed token budgets per micro-batch so the trainer doesn't waste compute on padding. Packing must keep sequence boundaries (for the attention mask, or `cu_seqlens` for varlen FlashAttention) and per-sequence metadata (advantage, version, mask).

```python
# pack.py - first-fit-decreasing packing of rollouts into 16k-token micro-batches (runnable)
import random
random.seed(0)
lens = [min(int(random.lognormvariate(8.3, 0.8)), 16384) for _ in range(256)]   # tokens per rollout
CAP = 16384
bins = []                                              # each bin: [used, [seq lens]]
for L in sorted(lens, reverse=True):
    for b in bins:
        if b[0] + L <= CAP: b[0] += L; b[1].append(L); break
    else: bins.append([L, [L]])
pad_mb = len(lens) * max(lens) / CAP                   # micro-batches if every sequence were padded to the max
print(f"{len(lens)} seqs, {sum(lens)/1e6:.2f}M tokens -> {len(bins)} packed micro-batches "
      f"(fill {sum(lens)/(len(bins)*CAP):.1%}); padded-to-max would need {pad_mb:.0f}")
```

```
$ python pack.py
256 seqs, 1.36M tokens -> 84 packed micro-batches (fill 98.9%); padded-to-max would need 256
```

### Transports

How the stages actually talk, as of the frameworks surveyed:

| Transport | Used for | Notes |
|---|---|---|
| Ray actors + object store | verl, OpenRLHF, slime, NeMo-RL control and data | one Python "driver" calls methods on remote actors. Large objects go through shared memory. Single-controller programming model |
| ZMQ sockets | prime-rl orchestrator ↔ trainer (default `rollout_transport`) | point-to-point, the trainer pulls one step at a time |
| Shared filesystem | rollout dumps, checkpoints, prime-rl's `rollout_transport.type = "filesystem"` | simple and persistent. Bounded by storage throughput |
| HTTP | engines (OpenAI-style + `/generate`), remote reward servers (OpenRLHF `--reward.remote_url`), weight relays | crosses trust and network boundaries, easy to load-balance |
| NCCL / NIXL / Mooncake | weights only | high bandwidth, fragile under membership changes |

A Ray single-controller sketch, to show the programming model. **Untested** (Ray is not installed here):

```python
import ray
ray.init()

@ray.remote(num_gpus=4)
class Engine:
    def __init__(self, model): ...                     # start vLLM with TP=4 inside this actor
    def generate(self, prompt_ids, n): ...             # -> list of (ids, logprobs, version)
    def update_weights(self, version): ...             # join the NCCL group and receive

@ray.remote(num_gpus=1)
class TrainerRank:
    def step(self, batch_ref): ...                     # one FSDP rank; all ranks called together

engines  = [Engine.remote("policy") for _ in range(8)]
trainers = [TrainerRank.remote() for _ in range(32)]
futs = [e.generate.remote(p, 16) for e, p in zip(engines, prompt_shards)]   # non-blocking
rollouts = ray.get(futs)                               # the driver is the orchestrator
```

The driver process sees the whole dataflow as ordinary Python method calls. Ray handles placement (`num_gpus`), serialization and the object store. That makes RL algorithms easy to change, which is HybridFlow's argument. The costs are that the driver is a single point of failure, and that large tensors going through the object store can become the bottleneck if you are not careful about what flows through the driver.

## Evals inside the RL loop

Training reward is measured on the training distribution, with the training sampling settings, using a policy that may be stale. It is not an evaluation. Evals should run *in the loop*, periodically, on the same inference fleet, under conditions you control:

- **Pin the version.** Evaluate a specific checkpoint version and record it. In async systems, "the model at step 400" has to be defined: the trainer's weights after step 400, synced to an engine.
- **Fixed sampling settings and seeds**, distinct from training settings (e.g. temperature 0.6, top-p 0.95, `n=16` for pass@k). With batch-dependent numerics, even greedy decoding is not reproducible across fleet states. Report averages over enough samples.
- **Separate priority or a separate engine slice**, so that evals do not steal the tail-filling slots and change training dynamics. prime-rl's orchestrator has explicit eval source and sink modules for this. In verl it is a periodic validation step.
- **Uncertainty.** On a 500-problem benchmark, a pass@1 difference of 1-2 points is often within noise. Use pass@k with the unbiased estimator and report a confidence interval.

```python
# passk.py - unbiased pass@k from n samples with c correct (Chen et al. 2021, arXiv 2107.03374)
from math import comb
def pass_at_k(n, c, k): return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
results = {"p0": (16, 0), "p1": (16, 3), "p2": (16, 16), "p3": (16, 8)}   # prompt -> (n, c)
for k in (1, 4, 16):
    print(f"pass@{k:<2} = {sum(pass_at_k(n, c, k) for n, c in results.values()) / len(results):.3f}")
```

```
$ python passk.py
pass@1  = 0.422
pass@4  = 0.642
pass@16 = 0.750
```

Treat a sudden eval drop together with *flat* training reward as an infrastructure alarm first. Common causes: a template change, a stale or broken weight sync to the eval engine, or eval contamination being fixed.

## Fault tolerance and debugging

### Failure modes, and what they look like

RL adds failure modes that pretraining doesn't have, and several of them look like algorithm problems:

| Symptom in metrics | Common infrastructure cause | First check |
|---|---|---|
| Step time grows, GPU utilization falls | a stuck rollout or group, a hung sandbox, an engine stuck in preemption loops | per-rollout age histogram, engine `num_running` / `num_waiting`, preemption counters |
| Reward suddenly 0 or NaN | verifier crashed or timed out (treated as 0), a NaN from a judge model, a dataset field renamed | fraction of non-finite or missing rewards, verifier error logs |
| Mismatch KL jumps at a specific step | weight sync bug (partial update, wrong resharding, version mix), engine restarted with a different dtype or quantization, prefix cache not reset | mismatch KL at staleness bin 0 |
| KL to reference or policy entropy blows up | can be algorithmic, but rule out stale or mixed-version rollouts and mis-masked env tokens first | mismatch KL by staleness bin, trained-token fraction |
| Response length collapses or explodes | max-tokens or truncation change, stop-token or template mismatch (model never emits EOS as the engine expects) | finish-reason histogram (`stop` vs `length`) |
| Trainer idle, queues empty | generator-bound: engines down, KV oversubscribed, sandbox fleet saturated | queue depths per stage |
| Run hangs completely | NCCL weight-sync collective with a dead member, a deadlocked orchestrator | stack dumps (`py-spy dump`), [hanging guide](../debug/torch-distributed-hanging-solutions.md) |

### Stuck rollouts and timeouts

Every remote call in the loop gets a deadline, and every deadline expiry has a defined outcome. A pattern for generation with a deadline and abort:

```python
# deadline.py - generation with a deadline: abort the request, return a marked-incomplete rollout (runnable)
import asyncio, random
random.seed(1)
async def engine_generate(req_id):                      # stand-in for an HTTP call to vLLM/SGLang
    await asyncio.sleep(random.choice([0.05, 0.05, 0.05, 5.0]))     # 1 in 4 never comes back in time
    return {"id": req_id, "finish": "stop"}
async def engine_abort(req_id): await asyncio.sleep(0)  # real engines expose abort by request id
async def rollout(req_id, deadline_s=0.5):
    try:
        return await asyncio.wait_for(engine_generate(req_id), deadline_s)
    except asyncio.TimeoutError:
        await engine_abort(req_id)                      # free its KV on the engine, or it keeps generating
        return {"id": req_id, "finish": "abort"}
async def main():
    out = await asyncio.gather(*(rollout(i) for i in range(20)))
    print({k: sum(o["finish"] == k for o in out) for k in ("stop", "abort")})
asyncio.run(main())
```

```
$ python deadline.py
{'stop': 12, 'abort': 8}
```

Cancelling the client-side coroutine does not stop the engine. Without the explicit abort, the engine keeps generating the orphaned request and holding its KV cache. That is a slow capacity leak, and it shows up as `num_running` staying high while throughput drops.

### Infrastructure health signals

Run cheap checks on every batch before the optimizer step. These signals are inexpensive to compute and catch most infrastructure bugs before they become a "training instability":

```python
# health.py - per-batch infra health checks to run before every optimizer step.
import numpy as np
rng = np.random.default_rng(0)
n = 512
batch = dict(
    reward=rng.binomial(1, 0.4, n).astype(float),
    lp_sampler=rng.normal(-0.6, 0.3, (n, 64)),          # (n, T) per-token logprobs from the engine
    finish=rng.choice(["stop", "length", "abort"], n, p=[0.9, 0.08, 0.02]),
    staleness=rng.integers(0, 3, n),
)
batch["lp_trainer"] = batch["lp_sampler"] + rng.normal(0, 0.01, (n, 64))   # (n, T) recomputed
batch["reward"][7] = np.nan                                                 # a broken verifier
batch["lp_trainer"][13] += 2.0                                              # e.g. wrong template / tokens

def check(b, max_mismatch_kl=1e-3, max_trunc=0.2):
    alerts = []
    bad = ~np.isfinite(b["reward"])
    if bad.any(): alerts.append(f"non-finite reward rows {np.flatnonzero(bad).tolist()} -> drop + page verifier owner")
    d = b["lp_sampler"] - b["lp_trainer"]                                  # (n, T)
    k3 = (np.exp(-d) - 1 + d).mean(1)                                      # k3 estimator of KL(sampler||trainer), per seq
    for s in np.unique(b["staleness"]):
        sel = b["staleness"] == s
        print(f"  mismatch KL staleness={s}: mean {k3[sel].mean():.2e}  max {k3[sel].max():.2e}")
    worst = np.flatnonzero(k3 > max_mismatch_kl)
    if k3.max() > max_mismatch_kl: alerts.append(f"mismatch KL outliers rows {worst.tolist()} (max {k3.max():.2f})")
    trunc = (b["finish"] == "length").mean()
    if trunc > max_trunc: alerts.append(f"truncation rate {trunc:.0%}")
    if (b["finish"] == "abort").any(): alerts.append(f"{(b['finish']=='abort').sum()} aborted rollouts (engine restarts?)")
    r = b["reward"][~bad]
    if r.std() == 0: alerts.append("zero reward variance: every advantage is 0")
    return alerts

for a in check(batch): print("ALERT:", a)
```

```
$ python health.py
  mismatch KL staleness=0: mean 5.02e-05  max 7.34e-05
  mismatch KL staleness=1: mean 5.02e-05  max 7.68e-05
  mismatch KL staleness=2: mean 2.61e-02  max 4.40e+00
ALERT: non-finite reward rows [7] -> drop + page verifier owner
ALERT: mismatch KL outliers rows [13] (max 4.40)
ALERT: 6 aborted rollouts (engine restarts?)
```

The mismatch KL here is the `k3` estimator (`exp(-d) - 1 + d` with `d = log μ - log π`), which is non-negative per sample, so it can be used for alerting. The planted bug (row 13, the trainer scoring different tokens) stands out by about five orders of magnitude, and so does the NaN reward. Per-staleness-bin means separate it from drift: the bin-2 mean is inflated only by the bad row. In production these become per-step dashboard series with alert thresholds, and a bad batch is quarantined (saved to disk for inspection) instead of trained on.

### Checkpointing and resuming an RL run

Resuming RL means restoring more than model weights:

| State | Why it matters |
|---|---|
| model + optimizer (sharded) | as in any [training checkpoint](../training/checkpoints/README.md) |
| policy version counter | staleness accounting and logged step numbers continue correctly |
| prompt/data sampler position + RNG | so the resumed run doesn't repeat or skip prompts |
| reference model / reward model identity | KL target and rewards must be the same models as before |
| carry-over pool (partial rollouts), replay buffers | or decide to drop them on resume, and record that you did |
| engine weights | engines restart from the checkpoint, *then* sync. Never assume the engines kept their weights |

In-flight rollouts at the time of a crash are usually cheaper to drop than to persist. Test resumption once, early: kill the run at step `N`, resume, and check that step `N+1` metrics are continuous. See [fault tolerance](../training/fault-tolerance/README.md) for the general machinery.

## Open frameworks survey

### Architectures at a glance

Architecture only, from READMEs, docs and source trees. These projects change monthly. Module layouts were checked on GitHub on 2026-11-05; the other descriptions are from documentation as I know it and should be verified against the current versions.

| Framework | Trainer | Inference | Control plane | Placement | Async |
|---|---|---|---|---|---|
| **verl** (ByteDance Seed; HybridFlow, arXiv 2409.19256) | FSDP / FSDP2 / Megatron-LM (plus other "engine workers") | vLLM, SGLang, HF | Ray, single controller | colocated by default, disaggregated supported | `verl/experimental/{one_step_off_policy,fully_async_policy}`. Pluggable `checkpoint_engine` backends for weight sync (naive, nccl, nixl, mooncake, kimi_ckpt_engine, hccl) |
| **OpenRLHF** (arXiv 2405.11143) | DeepSpeed (ZeRO-3, AutoTP) | vLLM | Ray | separate placement of actor/critic/reward/ref, or `--train.colocate_all` | `--train.async_enable`, `--train.async_queue_size`, `--train.partial_rollout_enable` |
| **prime-rl** (Prime Intellect; INTELLECT-2, arXiv 2505.07291) | FSDP2 | vLLM | separate processes: trainer, orchestrator (CPU), inference server. ZMQ or filesystem for rollouts, NCCL for weights. Launched via `uv run rl @ config.toml`, Slurm for multi-node | disaggregated | fully async, `orchestrator.max_staleness` (default 8) with token masking |
| **TRL** (Hugging Face) | Accelerate/DeepSpeed/FSDP | vLLM via `vllm_mode="colocate"` (default) or `"server"` | single program | either | primarily synchronous; vLLM IS correction flags |
| **slime** (THUDM / Z.ai; used for GLM-4.5-era RL) | Megatron-LM | SGLang | Ray | colocated or disaggregated | supports async (verify current docs) |
| **AReaL** (Ant / Tsinghua; arXiv 2505.24298) | own / FSDP-style | SGLang / vLLM | own | disaggregated | fully async, interruptible rollouts, staleness-bounded, decoupled PPO |
| **NeMo-RL** (NVIDIA) | Megatron / DTensor | vLLM | Ray | both | async support added over 2025 (verify) |
| **PipelineRL** (ServiceNow; arXiv 2509.19128) | own | vLLM | own | disaggregated | in-flight weight updates |
| **SkyRL** (Berkeley), **ROLL** (Alibaba) | various | vLLM / SGLang | Ray | both | agentic / multi-turn focus (verify) |

### How to read an open RL codebase

Each of these frameworks implements the same components from this chapter. When you open a new one, look for these, in this order:

1. **The entrypoint and config.** What processes start, and on which GPUs (placement).
2. **The control loop.** Who calls generate, score, train and sync, and is it one driver or several services?
3. **The rollout function.** Single-turn generate, or a multi-turn environment loop. Where do the masks and sampler logprobs come from?
4. **The batch / data structure** that crosses from rollout to trainer.
5. **The loss.** Where the IS correction, masking and staleness handling live.
6. **Weight sync.** Gather → reshard → transport → engine load → cache reset.
7. **The engine adapter.** Patches to vLLM/SGLang, weight-loading hooks, sleep/wake.

Mapping onto three codebases (paths from the GitHub trees on 2026-11-05; they will move):

| Component | prime-rl (`src/prime_rl/`) | verl (`verl/`) | OpenRLHF (`openrlhf/`) |
|---|---|---|---|
| Entrypoint / launch | `entrypoints/`, `uv run rl @ examples/.../rl.toml` | `trainer/main_ppo.py` | `cli/train_ppo_ray.py` via `ray job submit` |
| Control loop / orchestrator | `orchestrator/orchestrator.py` (+ `dispatcher.py`, `concurrency.py`) | `trainer/ppo/` (Ray trainer driver), `single_controller/ray/` | `trainer/ppo_trainer.py`, `trainer/ppo_trainer_async.py`, `trainer/ray/launcher.py` |
| Environments / rollout logic | `orchestrator/envs.py`, `orchestrator/trajectories.py`, plus the separate `verifiers` library | `experimental/agent_loop/`, `workers/rollout/` | agent executors (`AgentInstanceBase`, `MultiTurnAgentExecutor`, per README) |
| Rewards | via `verifiers` environments | `workers/reward_manager/`, `experimental/reward_loop/` | `--reward.remote_url` reward servers, `reward_func()` |
| Batch to trainer | `orchestrator/batch.py`, `orchestrator/train_sink.py`, `transports/batch/` | `protocol.py` (`DataProto`) | `trainer/ppo_utils/` |
| Trainer + loss | `trainer/rl/`, `trainer/train.py`, `trainer/models/` | `workers/engine_workers.py`, `workers/engine/` | `trainer/ray/ppo_actor.py`, `trainer/ray/ppo_critic.py` |
| Weight sync | `transports/weights/` | `checkpoint_engine/` | `trainer/ray/vllm_worker_wrap.py` (engine side) |
| Engine adapter | `inference/server.py`, `inference/vllm/` | `workers/rollout/vllm_rollout/`, `sglang_rollout/` | `trainer/ray/vllm_engine.py` |
| Async / staleness | `orchestrator` + `max_staleness` (docs/algorithms.md) | `experimental/fully_async_policy/`, `one_step_off_policy/` | `ppo_trainer_async.py`, `--train.async_queue_size` |
| Evals | `orchestrator/eval_source.py`, `eval_sink.py`, `eval/` | `trainer/main_eval.py`, periodic validation | eval options in the trainer |

A practical exercise that teaches a codebase faster than reading: run its smallest example end to end (for prime-rl, the README's 2-GPU `reverse-text` example; for verl and OpenRLHF, their GSM8K quickstarts), then put a breakpoint or a log line at each of the seven points above and watch one rollout travel through the system.

A Slurm launch for a disaggregated layout, as a sketch of what "launched via Slurm" means. **Untested**. The heterogeneous-job syntax is standard Slurm, but the program names are placeholders:

```bash
#!/bin/bash
#SBATCH --job-name=rl-32b-async
#SBATCH --nodes=4 --gpus-per-node=8 --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --nodes=4 --gpus-per-node=8 --ntasks-per-node=1
# component 0: inference (4 nodes x 2 TP=4 engines); component 1: trainer (FSDP over 32 GPUs)
srun --het-group=0 bash -c 'serve_policy --tp 4 --port 8000 --replicas-per-node 2' &
srun --het-group=0 --nodes=1 --ntasks=1 bash -c 'orchestrator --engines-from-slurm --max-staleness 2' &
srun --het-group=1 torchrun --nnodes 4 --nproc-per-node 8 --rdzv-backend c10d \
     --rdzv-endpoint "$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1 | head -1):29500" train_rl.py
wait
```

## You can now read

- [Decentralized Training](../training/decentralized-training.md): RL over the WAN, where weight sync and verification of untrusted rollouts dominate.
- [Services and Control Planes](control-planes.md): how to run the orchestrator, reward services and sandbox fleets as production services.
- [Serving Hands-On](../inference/serving-hands-on.md): running the vLLM/SGLang engines that serve as rollout workers.
- [Profiling and Optimization](../training/profiling-and-optimization.md): measuring where an RL step's time actually goes, rather than estimating it as this chapter does.
- [Kubernetes from Zero](../orchestration/kubernetes-from-zero.md): the substrate many sandbox fleets and elastic inference pools run on.
- The HybridFlow, AReaL, PipelineRL and Kimi k1.5 papers below, and the `docs/` folders of prime-rl and verl.

## Further reading

- Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (verl), arXiv 2409.19256.
- Hu et al., "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework", arXiv 2405.11143.
- Prime Intellect, "INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning", arXiv 2505.07291.
- Fu et al., "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning", arXiv 2505.24298.
- Piché et al., "PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation", arXiv 2509.19128.
- Kimi Team, "Kimi k1.5: Scaling Reinforcement Learning with LLMs" (partial rollouts), arXiv 2501.12599.
- Noukhovitch et al., "Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models", arXiv 2410.18252.
- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (dynamic sampling), arXiv 2503.14476.
- Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" (V-trace), arXiv 1802.01561.
- Hilton, Cobbe, Schulman, "Batch size-invariance for policy optimization" (decoupled PPO), arXiv 2110.00641.
- Yao et al., "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training" (blog, August 2025): truncated importance sampling for sampler/trainer mismatch.
- He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (blog, September 2025).
- Chen et al., "Evaluating Large Language Models Trained on Code" (unbiased pass@k), arXiv 2107.03374.
- vLLM sleep mode: https://docs.vllm.ai/en/latest/features/sleep_mode.html ; RL weight-sync examples: https://github.com/vllm-project/vllm/tree/main/examples/rl
- verl checkpoint engine: https://github.com/volcengine/verl/tree/main/verl/checkpoint_engine
- prime-rl docs (overview, algorithms): https://github.com/PrimeIntellect-ai/prime-rl/tree/main/docs
- OpenRLHF README (Ray + vLLM architecture, async and agent flags): https://github.com/OpenRLHF/OpenRLHF
- TRL GRPO config: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py
