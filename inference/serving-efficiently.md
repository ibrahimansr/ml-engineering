# Serving Efficiently on a Fixed Fleet

## Prerequisites

This is the last chapter of the inference part. It assumes the three before it and does not re-teach them. Every mechanism it builds on is linked where it is used.

- [Inference Internals I: The Arithmetic of LLM Inference](arithmetic.md): the [KV cache formulas](arithmetic.md#formulas), [max concurrent sequences](arithmetic.md#max-concurrent-sequences), [the KV cache calculator](arithmetic.md#a-kv-cache-calculator), [the decode latency bound](arithmetic.md#decode-latency-lower-bound), [tensor-parallel communication in decode](arithmetic.md#tensor-parallel-communication-in-decode), [KV cache quantization](arithmetic.md#kv-cache-quantization) and [Little's law](arithmetic.md#littles-law). The calculators below reuse those formulas and the same model configs.
- [Inference Internals II: Inside a Serving Engine](serving-engine-internals.md): [admission control by KV budget](serving-engine-internals.md#admission-control-by-kv-budget), [chunked prefill](serving-engine-internals.md#chunked-prefill-and-stall-free-batching), [prefix caching](serving-engine-internals.md#prefix-caching), [cache-aware routing](serving-engine-internals.md#cache-aware-routing-across-replicas), [CUDA graphs](serving-engine-internals.md#cuda-graphs), [speculative decoding](serving-engine-internals.md#speculative-decoding) and [when it hurts](serving-engine-internals.md#when-speculation-hurts-at-high-batch), [disaggregation](serving-engine-internals.md#disaggregated-prefill-and-decode) and [multi-LoRA serving](serving-engine-internals.md#multi-lora-serving).
- [Serving LLMs in Practice](serving-hands-on.md): [the flags that matter](serving-hands-on.md#the-flags-that-matter), [open vs closed loop](serving-hands-on.md#closed-loop-vs-open-loop), [seeds](serving-hands-on.md#seeds-and-why-one-run-is-an-anecdote), [the chunked-prefill experiment](serving-hands-on.md#chunked-prefill-trades-ttft-for-itl-tail), [the gateway and its routing experiment](serving-hands-on.md#routing-experiment), [engine metrics](serving-hands-on.md#monitoring-engine-metrics) and [capacity planning from measured numbers](serving-hands-on.md#capacity-planning-from-measured-numbers).
- Forward links, to chapters that come later in the book: [Infrastructure for Large-Scale RL](../platform/rl-infrastructure.md) (rollout serving) and [Control Planes](../platform/control-planes.md) (bounded queues, admission and per-tenant limits).

**What was run.** Every Python block with a file name in its first line ran on an Apple M4 laptop (macOS 15.7, Python 3.14.5, standard library only). Their outputs are pasted verbatim. Keep them in one directory: `trace.py`, `router_sim.py`, `mean_vs_full.py`, `sweep.py` and `backfill.py` import `sim.py`. Most of them are **simulations**. The engine model in `sim.py` is a cost model shaped like the roofline from [Internals I](arithmetic.md#decode-latency-lower-bound) for an 8B model on one H100. It reproduces the qualitative effects this chapter cares about: queueing, KV capacity, prefix-cache eviction and routing. Treat its absolute numbers as illustrations, not as predictions for your hardware. The calculators (`kv_long.py`, `layout.py`, `pd_ratio.py`, `spec.py`, `mm_tokens.py`) are arithmetic from stated inputs. **Every `vllm` and `sglang` command is untested**: no GPU was available. Their flag names were checked against the vLLM `latest` docs, the vLLM source on GitHub and the SGLang docs on 12 November 2026. Both projects rename flags often, so `--help` on your installed version is the authority. Numbers attributed to companies or papers link to the page they came from, and each was re-read on that page for this chapter. Vendor numbers are vendor-reported.

The setting is a startup that leases a fixed pool of GPUs, for example 16 or 64 H100s on a monthly contract. The lease is paid whether the GPUs are busy or idle. Two consequences follow, and they shape the whole chapter:

1. **The objective is goodput per leased GPU-hour.** [Goodput](serving-hands-on.md#disaggregated-prefill-and-decode) is the rate of requests (or tokens) served while meeting the latency SLO. A token served outside the SLO counts for nothing, and an idle GPU-second is lost for good.
2. **Autoscaling and cold starts mostly do not apply.** Much of what serverless vendors publish is about starting replicas quickly. On a warm fixed fleet the replicas are already up. What you can change is how each replica is configured, how the fleet is divided into replicas, where each request goes, and what fills the idle time.

Each section below gives the knob, the measured effect where someone has published one, and a small runnable model of why it works. Whenever a technique comes from a vendor (Modal, Baseten, Fireworks, Together, Prime Intellect, Applied Compute, llm-d or NVIDIA Dynamo), a short **On a leased fleet** note says whether it carries over and, if not, what the cheaper equivalent is.

Terms used throughout:

- A **session** is one conversation or one agent run: a sequence of requests whose prompts each extend the previous one. Agents make many requests per session, separated by **tool waits** (time spent running a tool before the next model call).
- A **page** or **block** is the unit of the paged KV cache (16 tokens in vLLM by default). A **prefix-cache hit** means a request's leading pages are already in the KV cache, so prefill can skip them.
- **Queue wait** is the time a request spends in the engine's waiting queue before its first prefill step. It is part of TTFT.
- **Session concurrency** is the number of sessions in progress at once, including those in a tool wait.



## Where the efficiency actually comes from

The levers below are ranked by the size of the published, measured effect on throughput or latency at fixed hardware. The ranking is rough: each number comes from a different workload, and it applies only when the workload has the property in the last column.

| Rank | Lever | Measured effect | Source | Only when |
|-:|-|-|-|-|
| 1 | Prefix/KV-aware routing across replicas | output throughput 8,730 vs 4,429 tok/s (1.97x) and P90 TTFT 0.54 s vs 92.6 s, precise vs random | [llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see) | many requests share long prefixes |
| 1 | same | 89% cache hit, mean TTFT -50%, TPOT -34%, +61% RPS vs random | [Baseten](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/) | same |
| 2 | Layout of a fixed node: replicas vs one sharded copy | Llama 70B on 8 GPUs "4x cheaper per token" in a volume-optimized layout than in a single-request-speed layout | [Fireworks](https://fireworks.ai/blog/fireoptimizer) | per-user speed target is loose enough |
| 2 | Independent replicas vs one multi-node engine | throughput "scales linearly with the number of nodes"; vLLM's built-in multi-node DP "quickly plateaued" | [INTELLECT-3 report](https://arxiv.org/html/2512.16144v1) | dense or modest-size models |
| 3 | Speculative decoding with a draft trained on your traffic | 2x faster at 76% hit rate; a generic draft at 29% made latency 1.5x *worse* | [Fireworks](https://fireworks.ai/blog/fireoptimizer) | small batch sizes |
| 4 | Scheduler token budget | halving the prefill budget from 8K to 4K per GPU cut median queue wait 550 ms to 110 ms and median TTFT about 20% | [Prime Intellect](https://www.primeintellect.ai/blog/prime-inference) | prefill-heavy traffic |
| 4 | KV capacity (KV dtype, offload) | NVFP4 KV: 1.09M to 1.63M cached tokens per decoder at the same memory | [Prime Intellect](https://www.primeintellect.ai/blog/prime-inference) | KV-bound workloads (agents, long context) |
| 5 | Cache-aware disaggregation | about 40% more sustainable QPS/GPU than plain 2P1D | [Together](https://www.together.ai/blog/cache-aware-disaggregated-inference) | large fleets, long reused contexts |
| 6 | Backfilling idle capacity with batch work | no throughput gain on the interactive path; turns idle lease-hours into useful tokens | this chapter's `backfill.py` (simulated) | the interactive load has troughs |

Three things are missing from the table on purpose. Quantization is not in it because its effect is mostly a capacity effect, which the KV row covers and [Internals I](arithmetic.md#quantization-internals) works out. Kernel work is not in it because a startup on vLLM or SGLang inherits the engine's kernels and does not write its own. Autoscaling is not in it because the fleet is fixed.

The common thread is that most of the large wins come from **not doing work twice**: not recomputing a prefix another replica already holds, not reading weights on more GPUs than the speed target needs, not running the target model once per token when a draft could have guessed it. The second thread is **KV capacity**. On agentic and long-context traffic, the number of tokens of KV the fleet can hold decides both the batch size and the prefix-cache hit rate. Applied Compute benchmarked vLLM and SGLang on DeepSeek R1 with traces from more than 100 of their own post-training runs and concluded that "[our primary bottleneck is KV capacity](https://www.appliedcompute.com/research/inference-benchmark)".



## Measuring your own traffic first

Every lever in the table depends on a property of the traffic: prompt and output lengths, how much of each prompt is shared, how bursty arrivals are, how long sessions hold context. These have to be measured on your traffic before tuning. A benchmark with uniform random prompts measures a workload nobody has.

### What to log at the gateway

The [gateway](serving-hands-on.md#a-gateway-in-front-of-several-replicas) sees every request, so it is the natural place to log. Per request you need arrival time, prompt and output token counts, TTFT and end-to-end latency, a session id and a tenant id. To measure prefix sharing without storing prompt text, log **chained hashes of 16-token blocks** of the token ids. Each block's hash includes the previous block's hash, so two requests share their first $k$ hashes exactly when they share their first $16k$ tokens. This is the same scheme vLLM's prefix cache uses internally ([Internals II](serving-engine-internals.md#hash-based-block-caching-vllm-apc)). A secret salt that you rotate stops anyone from confirming a guessed prompt by hashing it.

The field names below are chosen so that the log replays directly with `vllm bench serve --dataset-name timed_trace` (see [below](#a-replayable-benchmark)):

```python
# trace_log.py - what the gateway should log per request: lengths, timing, and prefix-block hashes
# (chained hashes of 16-token blocks, the same scheme as vLLM's prefix cache) - no prompt text.
# Field names match `vllm bench serve --dataset-name timed_trace` defaults, so the log replays as-is.
import hashlib, json

BLOCK = 16
def block_hashes(token_ids, salt=b"rotate-me-monthly"):
    out, h = [], salt
    for i in range(0, len(token_ids) - len(token_ids) % BLOCK, BLOCK):     # full blocks only
        h = hashlib.blake2b(h + repr(token_ids[i:i + BLOCK]).encode(), digest_size=8).digest()
        out.append(int.from_bytes(h, "big") >> 1)                          # depends on the whole prefix
    return out

def record(session_id, token_ids, out_tokens, t_arrival, t_first, t_done, tenant):
    return dict(timestamp=round(t_arrival * 1000), input_length=len(token_ids), output_length=out_tokens,
                hash_ids=block_hashes(token_ids), session_id=session_id, tenant=tenant,
                ttft_ms=round((t_first - t_arrival) * 1000), e2e_ms=round((t_done - t_arrival) * 1000))

if __name__ == "__main__":
    system = list(range(1000, 1000 + 64))                  # 64-token "system prompt"
    turn1 = system + list(range(5000, 5040))               # 104 tokens
    turn2 = turn1 + list(range(7000, 7030))                # history grows append-only
    other = list(range(1000, 1000 + 32)) + [9] * 72        # shares only the first 32 tokens
    now = 1_700_000_000.0                                  # fixed clock so the output is reproducible
    a, b, c = (record("s1", turn1, 50, now, now + .1, now + 1, "acme"),
               record("s1", turn2, 50, now + 3, now + 3.05, now + 4, "acme"),
               record("s2", other, 50, now + 4, now + 4.2, now + 5, "acme"))
    def shared(x, y):
        n = 0
        for p, q in zip(x["hash_ids"], y["hash_ids"]):
            if p != q: break
            n += 1
        return n * BLOCK
    print(json.dumps(dict(a, hash_ids=a["hash_ids"][:2] + ["..."])))
    print("turn2 reuses", shared(a, b), "of", b["input_length"], "tokens;  other session reuses", shared(a, c))
```

Output of `python3 trace_log.py`:

```text
{"timestamp": 1700000000000, "input_length": 104, "output_length": 50, "hash_ids": [8538110349002366337, 1579529013749511410, "..."], "session_id": "s1", "tenant": "acme", "ttft_ms": 100, "e2e_ms": 1000}
turn2 reuses 96 of 134 tokens;  other session reuses 32
```

Turn 2 reuses 96 of its 134 tokens: the six full blocks of turn 1. Its last partial block is not cacheable. The other session shares the 32-token start of the system prompt and nothing else. Note that the hash only says *that* prefixes are equal, not what they contain. That is all routing analysis needs.

### What to compute from a trace

Real logs cannot be shipped with a book, so the rest of the chapter uses a synthetic trace shaped after the agentic workload that Applied Compute describes from more than 100 production post-training runs: initial prompts "centered around 10k" tokens (mostly system prompt and tool descriptions), "a mean of about twenty tool turns per trace" with a long tail, assistant responses of 200-300 tokens per turn, tool outputs around 500 tokens, and tool latency "around one second, but can extend past hundreds of seconds in the tail" ([source](https://www.appliedcompute.com/research/inference-benchmark)). Eight tenants each have their own system prompt, and a few tenants dominate traffic. Session arrivals are a gamma process with coefficient of variation 2, burstier than Poisson.

`sim.py` is the engine model the whole chapter uses. A `Replica` holds a paged prefix cache with LRU eviction and pinning, admits requests by KV budget, and runs continuous-batching steps with decodes first, then prefill chunks up to a token budget, the vLLM V1 policy described in [Internals II](serving-engine-internals.md#the-unified-scheduler). Step time is the roofline: one weight read, plus the live KV read, plus prefill FLOPs at 50% MFU. The simulator uses 256-token pages rather than 16 to run 16x faster, which slightly understates hit rates.

```python
# sim.py - a continuous-batching replica with a paged prefix cache, plus a tiny event loop
# Step time is a roofline-shaped cost model of an 8B bf16 model on one H100 (Internals I numbers).
import heapq, math
from collections import OrderedDict

BLOCK = 256   # tokens per cache page. ponytail: engines use 16; 256 keeps the sim 16x faster (LMCache chunk size)
W_READ = 16.06e9 / 3.35e12         # s: read all 8B weights once per step (bf16)
KV_READ = 131072 / 3.35e12         # s per cached token read by decode attention (128 KiB/token)
PREFILL = 2 * 8.03e9 / (989e12 * 0.5)   # s per prefill token at 50% MFU
T0 = 1.0e-3                        # s fixed per-step overhead (scheduler, launches)

class Req:
    def __init__(self, sid, turn, blocks, n_prompt, n_out, prio=0):
        self.sid, self.turn, self.blocks = sid, turn, blocks     # blocks: prompt block hashes
        self.n_prompt, self.n_out, self.prio = n_prompt, n_out, prio
        self.t_arr = self.t_first = self.t_done = None
        self.cached = 0                                          # prompt tokens served from cache
        self.prefilled = 0; self.generated = 0; self.held = []   # blocks pinned while running

class Replica:
    def __init__(self, kv_tokens=427_000, max_num_seqs=256, max_batched=8192, priority=False):
        self.cap = kv_tokens // BLOCK                 # KV capacity in blocks
        self.ref = {}                                 # block hash -> refcount (resident blocks)
        self.lru = OrderedDict()                      # resident blocks with refcount 0 (evictable)
        self.private = 0                              # blocks held for not-yet-shareable output
        self.waiting, self.running = [], []
        self.max_seqs, self.budget, self.priority = max_num_seqs, max_batched, priority
        self.busy = False; self.hit_tok = self.prompt_tok = 0; self.steps = 0

    def used(self): return len(self.ref) + self.private
    def kv_util(self): return (self.used() - len(self.lru)) / self.cap
    def load(self): return len(self.waiting) + len(self.running)
    def match(self, blocks):                          # longest cached prefix, in blocks
        n = 0
        for b in blocks:
            if b not in self.ref: break
            n += 1
        return n

    def _admit(self, r):
        hit = self.match(r.blocks)
        need = len(r.blocks) - hit + math.ceil(r.n_out / BLOCK)
        while self.used() + need > self.cap and self.lru:
            del self.ref[self.lru.popitem(last=False)[0]]            # evict LRU, unpinned only
        if self.used() + need > self.cap: return False
        hit = self.match(r.blocks)                                   # eviction may have shortened it
        for b in r.blocks:                                           # pin the whole prompt
            if self.ref.get(b, 0) == 0: self.lru.pop(b, None)
            self.ref[b] = self.ref.get(b, 0) + 1
        r.extra = math.ceil(r.n_out / BLOCK); self.private += r.extra
        r.cached = min(hit * BLOCK, r.n_prompt - 1)                  # last token always recomputed
        r.prefilled = r.cached
        self.hit_tok += r.cached; self.prompt_tok += r.n_prompt
        return True

    def _release(self, r, out_blocks):
        self.private -= r.extra
        for b in r.blocks:                                           # unpin the prompt
            self.ref[b] -= 1
            if self.ref[b] == 0: self.lru[b] = True; self.lru.move_to_end(b)
        for b in out_blocks:                                         # output becomes cacheable history
            if b not in self.ref: self.ref[b] = 0; self.lru[b] = True

    def step(self, now):
        """Schedule one step: decodes first, then prefill chunks within the token budget."""
        if self.priority: self.waiting.sort(key=lambda r: (r.prio, r.t_arr))
        while self.waiting and len(self.running) < self.max_seqs and self._admit(self.waiting[0]):
            self.running.append(self.waiting.pop(0))
        if not self.running: return None
        decode = [r for r in self.running if r.prefilled >= r.n_prompt]
        budget, chunks = self.budget - len(decode), []
        for r in self.running:
            if r.prefilled < r.n_prompt and budget > 0:
                c = min(budget, r.n_prompt - r.prefilled); chunks.append((r, c)); budget -= c
        kv = sum(r.prefilled + r.generated for r in decode)
        dt = T0 + W_READ + kv * KV_READ + sum(c for _, c in chunks) * PREFILL
        self.steps += 1
        return dt, decode, chunks

def run(replicas, router, arrivals, on_done, horizon=1e9):
    """Event loop. arrivals: list of (t, Req). router(req, replicas) -> replica index.
    on_done(req, t) may return new (t, Req) arrivals (next turn of a session)."""
    ev = [(t, 0, i, r) for i, (t, r) in enumerate(arrivals)]; heapq.heapify(ev)
    seq, done = len(ev), []
    def kick(i, now):
        nonlocal seq
        rep = replicas[i]
        if rep.busy: return
        s = rep.step(now)
        if s is None: return
        rep.busy = True; seq += 1
        heapq.heappush(ev, (now + s[0], 1, seq, (i, s)))
    while ev:
        t, kind, _, x = heapq.heappop(ev)
        if t > horizon: break
        if kind == 0:                                   # arrival
            x.t_arr = t; i = router(x, replicas); replicas[i].waiting.append(x); kick(i, t)
        else:                                           # step finished on replica i
            i, (dt, decode, chunks) = x; rep = replicas[i]; rep.busy = False
            for r, c in chunks:
                r.prefilled += c
                if r.prefilled >= r.n_prompt: r.t_first = t; r.generated = 1
            for r in decode: r.generated += 1
            for r in [r for r in rep.running if r.generated >= r.n_out and r.t_first is not None]:
                rep.running.remove(r); r.t_done = t; done.append(r)
                rep._release(r, r.out_blocks)
                for nt, nr in on_done(r, t):
                    seq += 1; heapq.heappush(ev, (nt, 0, seq, nr))
            kick(i, t)
    return done
```

`trace.py` generates sessions and turns them into requests. The `Driver` is closed-loop per session, so turn $k+1$ arrives one tool wait after turn $k$ finishes. A slow server therefore delays the session's later turns, as it does in production:

```python
# trace.py - synthetic multi-turn agent trace, shaped after Applied Compute's description
# (~10k-token initial prompt that is mostly system prompt + tool schemas, ~20 turns with a long
# tail, 200-300-token assistant turns, ~500-token tool outputs, ~1 s tool latency with a long tail)
import json, math, random
from sim import Req, BLOCK

def rnd16(x): return max(16, int(round(x / 16)) * 16)

def gen_sessions(n, rate, seed=0, tenants=8, cv=1.0):
    """n sessions arriving at `rate`/s. cv>1 = burstier than Poisson (gamma inter-arrivals)."""
    rng = random.Random(seed)
    sys_len = {t: rnd16(rng.uniform(6000, 9000)) for t in range(tenants)}   # per-tenant system prompt
    shape = 1 / cv**2                                    # gamma shape; mean gap fixed at 1/rate
    t, out = 0.0, []
    for sid in range(n):
        t += rng.gammavariate(shape, 1 / (rate * shape))
        turns = min(200, max(1, int(rng.lognormvariate(math.log(14), 0.8))))  # mean ~19, tail to 200
        tenant = min(tenants - 1, int(rng.paretovariate(1.2)) - 1)          # a few tenants dominate
        user = rnd16(rng.uniform(500, 2500))
        steps = [dict(inp=user if k == 0 else rnd16(rng.lognormvariate(math.log(450), 0.6)),
                      out=rnd16(rng.uniform(200, 300)),
                      wait=min(300.0, rng.lognormvariate(0.0, 1.2)))         # ~1 s median, long tail
                 for k in range(turns)]
        out.append(dict(sid=sid, t0=round(t, 3), tenant=tenant, sys=sys_len[tenant], turns=steps))
    return out

class Driver:
    """Turns sessions into engine requests. Each session is closed-loop: turn k+1 arrives
    `wait` seconds after turn k finishes, so a session holds its slot through tool waits."""
    def __init__(self, sessions):
        self.s = {x["sid"]: x for x in sessions}
    def req(self, sid, k):
        x = self.s[sid]; T = x["turns"]
        hist = sum(T[j]["inp"] + T[j]["out"] for j in range(k)) + T[k]["inp"]
        sysb = [("sys", x["tenant"], i) for i in range(x["sys"] // BLOCK)]
        h0 = hist // BLOCK                               # only full pages are cacheable
        r = Req(sid, k, sysb + [("h", sid, j) for j in range(h0)], x["sys"] + hist, T[k]["out"])
        r.out_blocks = [("h", sid, j) for j in range(h0, (hist + T[k]["out"]) // BLOCK)]
        return r
    def first(self): return [(x["t0"], self.req(x["sid"], 0)) for x in self.s.values()]
    def on_done(self, r, t):
        T = self.s[r.sid]["turns"]
        return [] if r.turn + 1 >= len(T) else [(t + T[r.turn]["wait"], self.req(r.sid, r.turn + 1))]

if __name__ == "__main__":
    S = gen_sessions(400, rate=0.5, seed=0, cv=2.0)
    with open("trace.jsonl", "w") as f:
        for x in S: f.write(json.dumps(x) + "\n")
    print(json.dumps({k: v for k, v in S[0].items() if k != "turns"}), "turns:", len(S[0]["turns"]))
    print("first turn:", S[0]["turns"][0])
```

Output of `python3 trace.py` (writes `trace.jsonl`):

```text
{"sid": 0, "t0": 0.587, "tenant": 0, "sys": 8528} turns: 43
first turn: {'inp': 2016, 'out': 256, 'wait': 3.3205467834992604}
```

The analysis prints the four things to look at first:

```python
# trace_stats.py - the numbers to pull out of a captured trace before tuning anything
import json, statistics as st, sys
from collections import Counter

S = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl")]

def pct(xs, ps=(50, 90, 99)):
    xs = sorted(xs); return [xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in ps]

# 1. per-request lengths. A request's prompt = system prompt + the whole history so far.
inp, out, new = [], [], []
for s in S:
    hist = 0
    for k, t in enumerate(s["turns"]):
        hist += t["inp"]
        inp.append(s["sys"] + hist); out.append(t["out"]); new.append(t["inp"])
        hist += t["out"]
print(f"sessions {len(S)}  requests {len(inp)}  turns/session p50/p90/p99 "
      f"{pct([len(s['turns']) for s in S])}  max {max(len(s['turns']) for s in S)}")
for name, xs in (("prompt tokens", inp), ("new tokens/turn", new), ("output tokens", out)):
    print(f"{name:16s} mean {st.mean(xs):8.0f}  p50/p90/p99 {pct(xs)}")
print(f"prompt:output token ratio {sum(inp)/sum(out):.0f}:1   "
      f"new(uncached-if-perfect):output {sum(new)/sum(out):.1f}:1")

# 2. prefix structure: with an infinite cache, which prompt tokens were seen before?
sys_tok = sum(s["sys"] * len(s["turns"]) for s in S)
first_sys = sum(dict((s["tenant"], s["sys"]) for s in S).values())
reuse_sys = sys_tok - first_sys                     # every system prompt after its first use
reuse_hist = sum(i - s - n for i, s, n in
                 zip(inp, [s["sys"] for s in S for _ in s["turns"]], new))  # history from earlier turns
print(f"ideal (infinite-cache) hit rate {(reuse_sys + reuse_hist)/sum(inp):.1%}: "
      f"system prompts {reuse_sys/sum(inp):.1%}, own history {reuse_hist/sum(inp):.1%}")
ten = Counter(s["tenant"] for s in S)
print("sessions per system prompt:", dict(sorted(ten.items())))

# 3. arrival burstiness: coefficient of variation of gaps, and peak-to-mean in 60 s windows
t0 = [s["t0"] for s in S]; gaps = [b - a for a, b in zip(t0, t0[1:])]
cv = st.pstdev(gaps) / st.mean(gaps)
win = Counter(int(t // 60) for t in t0); counts = [win.get(i, 0) for i in range(int(t0[-1] // 60) + 1)]
print(f"session arrivals: mean gap {st.mean(gaps):.2f}s  CV {cv:.2f} (Poisson = 1.0)  "
      f"60s-window peak/mean {max(counts)/st.mean(counts):.1f}")

# 4. session lifetime footprint: tokens held at the end of a session, which is what KV must hold
final = [s["sys"] + sum(t["inp"] + t["out"] for t in s["turns"]) for s in S]
waits = [t["wait"] for s in S for t in s["turns"]]
print(f"final context per session p50/p90/p99 {pct(final)}   tool wait p50/p99 "
      f"{pct(waits, (50,))[0]:.1f}s/{pct(waits, (99,))[0]:.1f}s")
```

Output of `python3 trace_stats.py trace.jsonl`:

```text
sessions 400  requests 7118  turns/session p50/p90/p99 [14, 36, 93]  max 144
prompt tokens    mean    22467  p50/p90/p99 [17328, 40256, 90896]
new tokens/turn  mean      593  p50/p90/p99 [480, 1120, 2208]
output tokens    mean      250  p50/p90/p99 [256, 288, 304]
prompt:output token ratio 90:1   new(uncached-if-perfect):output 2.4:1
ideal (infinite-cache) hit rate 97.3%: system prompts 36.0%, own history 61.3%
sessions per system prompt: {0: 225, 1: 73, 2: 31, 3: 15, 4: 12, 5: 5, 6: 10, 7: 29}
session arrivals: mean gap 2.19s  CV 1.81 (Poisson = 1.0)  60s-window peak/mean 1.4
final context per session p50/p90/p99 [19680, 37536, 76816]   tool wait p50/p99 1.0s/17.3s
```

How to read it, and which later section each line feeds:

- **Prompt tokens per request average 22k, with p99 at 91k, while new tokens per turn average about 600.** The prompt is 90 times the output, but only 2.4 times the output if every repeated prefix hits the cache. The whole economics of this workload rests on that cache hit. It feeds [routing](#routing) and [KV capacity](#kv-capacity-is-the-bottleneck-for-agentic-workloads).
- **The ideal hit rate is 97%**, split into 36% from shared system prompts and 61% from each session's own history. History reuse only works if a session's turns land where its KV is. Shared system prompts benefit from any replica that has served that tenant. That split decides between session affinity and prefix-aware routing.
- **Session arrivals have CV 1.8.** A Poisson benchmark (CV 1) would understate queueing at the same mean rate.
- **Final context per session has p50 20k and p99 77k tokens.** These sizes set how many sessions fit in one replica's KV. An 8B bf16 model on one H100 has room for about 427k tokens ([Internals I](arithmetic.md#max-concurrent-sequences)), which is about 21 median sessions or 5 p99 sessions.

Two more things are worth computing on real logs, which a synthetic trace cannot show. The first is **drift**. Together reports that prompt length rose "80%" two weeks after a model release as usage shifted to agents, and that one country's working hours had "1.4x longer prompts than another country (97k vs 70k)" ([source](https://together.ai/blog/together-compiler-continual-compilation-for-ai-inference)). Recompute the statistics weekly, and re-tune when they move. The second is **truncation**. Agent harnesses that drop old turns to fit the context window break the append-only prefix structure. LMCache's authors found in production traces that context truncation made "prefix cache hit ratios drop from roughly 85% to 45%" ([source](https://arxiv.org/html/2510.09665)). If your harness truncates, truncate in large steps so that the new prefix stays stable for many turns.

### Why averaged traces mislead

A common shortcut is to build a benchmark from the means: average prompt length, average turns, average output. Applied Compute measured what this costs: replaying a mean trace instead of the full captured traces overestimated throughput by 10-20% ([source](https://www.appliedcompute.com/research/inference-benchmark)). The reason is the tail. Long sessions hold a lot of KV for a long time, and once the cache is full they evict other sessions' prefixes, which then have to be recomputed. A mean session never gets long enough to do that.

The simulation below reproduces their setup: a fixed number of session slots on one replica, a new session starting the moment one finishes, each session holding its slot through tool waits, and the first 20% of wall-clock time discarded as warmup. It compares the real (synthetic) sessions with the same number of "mean sessions":

```python
# mean_vs_full.py - replay the captured sessions vs a "mean session" at fixed session concurrency
import statistics as st
from sim import Replica, run
from trace import gen_sessions, Driver, rnd16

S = gen_sessions(400, rate=1.0, seed=0, cv=2.0)
T = [t for s in S for t in s["turns"]]
mean_turns = round(st.mean(len(s["turns"]) for s in S))
mean_sys = rnd16(st.mean(s["sys"] for s in S))
mean_turn = dict(inp=rnd16(st.mean(t["inp"] for t in T[1:])), out=rnd16(st.mean(t["out"] for t in T)),
                 wait=st.mean(t["wait"] for t in T))
M = [dict(sid=s["sid"], t0=0.0, tenant=s["tenant"], sys=mean_sys,
          turns=[dict(s["turns"][0], out=mean_turn["out"], wait=mean_turn["wait"])] +
                [dict(mean_turn)] * (mean_turns - 1)) for s in S]

def replay(sessions, C):
    """C session slots; a new session starts the moment one finishes (Applied Compute's setup)."""
    d = Driver(sessions); queue = list(d.s)[C:]
    def on_done(r, t):
        nxt = d.on_done(r, t)
        if not nxt and queue: nxt = [(t, d.req(queue.pop(0), 0))]
        return nxt
    rep = Replica()
    done = run([rep], lambda r, reps: 0, [(0.0, d.req(sid, 0)) for sid in list(d.s)[:C]], on_done)
    end = max(r.t_done for r in done); warm = 0.2 * end                 # drop first 20% of wall-clock
    toks = sum(r.n_out for r in done if r.t_done > warm)
    return toks / (end - warm), rep.hit_tok / rep.prompt_tok

CS = (8, 16, 24, 32, 48)
print(f"mean session: {mean_turns} turns, system {mean_sys}, per turn {mean_turn['inp']} in / "
      f"{mean_turn['out']} out, wait {mean_turn['wait']:.2f}s")
print(f"{'C':>3} {'full tok/s':>10} {'mean tok/s':>10} {'overstate':>9} {'hit full':>8} {'hit mean':>8}")
for C in CS:
    f, hf = replay(S, C); m, hm = replay(M, C)
    print(f"{C:3d} {f:10.0f} {m:10.0f} {m/f-1:+9.1%} {hf:8.1%} {hm:8.1%}")
```

Output of `python3 mean_vs_full.py`:

```text
mean session: 18 turns, system 8144, per turn 592 in / 256 out, wait 2.13s
  C full tok/s mean tok/s overstate hit full hit mean
  8        413        402     -2.6%    96.4%    94.1%
 16        609        613     +0.6%    96.3%    94.1%
 24        535        729    +36.3%    85.1%    92.8%
 32        358        615    +72.0%    56.9%    72.6%
 48        300        546    +81.9%    40.9%    57.6%
```

At 8 and 16 concurrent sessions all the KV fits, and the two traces agree to within 3%. From 24 sessions on, the real trace's long sessions overflow the 427k-token cache, its hit rate drops from 96% to 41%, and its throughput falls *as concurrency rises*. The mean trace degrades later and less. At 24 sessions it overstates throughput by 36%, and at 48 by 82%. This synthetic trace has a heavier tail than Applied Compute's, which is why the overstatement here is larger than their 10-20%. The direction and the cause are the same. The table also shows the second lesson of their benchmark: **past the point where KV fills, adding concurrency lowers throughput**, because every extra session evicts someone's prefix. On this replica the best session concurrency is about 16.

### A replayable benchmark

A benchmark you can rerun after every change needs the captured trace, a replayer, and a fixed seed. Two replay modes exist, and they answer different questions:

- **Timestamp replay (open loop).** Requests are sent at their logged arrival times, whatever the server does. vLLM's benchmark client has this built in as the `timed_trace` dataset. It reads JSONL lines with `timestamp`, `input_length`, `output_length` and `hash_ids` fields (the names are configurable), and expands each hash id into a deterministic run of 16 random tokens, so requests with equal hash prefixes get equal token prefixes. That reproduces the prefix-sharing structure without the text. The format matches the published Mooncake traces. The field names and the 16-token default were read from `vllm/benchmarks/datasets/datasets.py` on GitHub.
- **Session replay (closed loop per session).** Each session's next turn is sent one tool wait after the previous turn *finishes*, which is what `trace.py`'s `Driver` does. It is the right model for agents: a slow server makes sessions last longer and hold KV longer. Timestamp replay cannot show that feedback. Applied Compute's harness works this way. Each trace occupies "one concurrency slot for its full lifetime, including tool wait time", and the harness is open source at [github.com/Applied-Compute/trie](https://github.com/Applied-Compute/trie), in under 1k lines of Python ([source](https://www.appliedcompute.com/research/inference-benchmark)).

The timestamp replay against a running vLLM server (untested):

```bash
# untested. requests.jsonl: one record per line from trace_log.py (timestamps in ms)
vllm bench serve --model $MODEL --base-url http://localhost:8000 \
  --dataset-name timed_trace --dataset-path requests.jsonl \
  --timed-trace-sec-multiplier 0.001 --timed-trace-chunk-hash-size 16 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --goodput ttft:2000 tpot:50 \
  --seed 0 --save-result --result-filename replay-seed0.json
```

`--goodput` takes SLO pairs and reports the rate of requests meeting all of them. `--burstiness` (1.0 = Poisson, below 1 burstier) and `--ramp-up-strategy` exist for synthetic datasets, where arrivals are generated rather than replayed. All of these flags appear in the [`vllm bench serve` docs](https://docs.vllm.ai/en/latest/cli/bench/serve.html). Run each configuration on three seeds, as the [hands-on chapter](serving-hands-on.md#seeds-and-why-one-run-is-an-anecdote) does, and compare the worst seed.

**On a leased fleet.** Applied Compute's harness and the `timed_trace` replayer both run anywhere and cost one replica's time per benchmark. The cheapest setup is to take one replica out of the router for an hour, replay yesterday's peak hour against it, and put it back.



## Tuning the engine for a workload

With a trace and a replayer you can tune. This section covers the knobs that matter most on one replica, in the order to tune them: the parallel layout, the batch and token budgets, KV capacity, number formats and compilation. It ends with a procedure that changes one knob at a time.

### TP vs DP vs EP: the layout per model size

**Tensor parallelism** (TP) shards every layer across GPUs. **Data parallelism** (DP) runs independent copies. For mixture-of-experts models, **expert parallelism** (EP) spreads the experts across GPUs while attention runs data-parallel. The theory is in [Internals II](serving-engine-internals.md#serving-time-parallelism), and the [EP case](serving-engine-internals.md#dp-attention-and-expert-parallelism-for-moe) is not repeated here. For dense models on a fixed fleet the question is concrete: you have 16 GPUs, so do you run 16 copies at TP1, 8 at TP2, 4 at TP4, or 2 at TP8?

Three effects decide it:

- **Weights take room from KV.** Each replica stores a full copy of the weights, split over its $P$ GPUs. A big model at small TP leaves almost no HBM for KV, so the batch stays small.
- **TP has a fixed communication floor.** Each decode step does $2L$ small all-reduces at about 15 µs each, whatever the model's precision ([Internals I](arithmetic.md#tensor-parallel-communication-in-decode)).
- **TP buys per-user speed.** At a fixed batch size, larger $P$ divides the weight read, so each user's tokens come faster.

`layout.py` puts the three together for a fleet of 16 H100s with FP8 weights and FP8 KV. For each layout and each per-user speed target, it finds the largest batch that still meets the target and reports fleet output throughput there. The step time is the decode bound: the maximum of HBM time and GEMM time, plus the all-reduces. The all-reduce bandwidth of 360 GB/s is an assumption taken from Internals I.

```python
# layout.py - 16 H100s, FP8 weights + FP8 KV: how to cut the fleet into replicas
import sys
# Decode bound per step = max(HBM reads, GEMM FLOPs) + 2L all-reduces (alpha-beta, Internals I).
HBM, BW, UTIL = 80e9, 3.35e12, 0.90
MODEL = sys.argv[2] if len(sys.argv) > 2 else "70B"
N, L, H = {"70B": (70.55e9, 80, 8192), "8B": (8.03e9, 32, 4096)}[MODEL]   # params, layers, hidden
KV_TOK = 2 * L * 8 * 128 * 1                          # FP8 KV bytes/token; FP8 weights = 1 B/param
ALPHA, BUSBW = 15e-6, 360e9    # all-reduce latency inside a node; busbw is an assumption (Internals I)
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 8192    # average live context per sequence
FP8_PEAK = 1979e12                                     # dense FP8 FLOP/s per H100
FLEET = 16

def replica(P, B=None):
    kv_budget = HBM * UTIL - N / P                     # per GPU
    max_tok = kv_budget * P / KV_TOK                   # per replica (KV heads split over P <= 8)
    b_max = int(max_tok // CTX)
    B = min(B or b_max, b_max)
    t_mem = (N / P + B * CTX * KV_TOK / P) / BW
    t_cmp = 2 * N * B / P / FP8_PEAK                   # GEMM FLOPs, ignoring attention math
    t_ar = 2 * L * (ALPHA + B * H * 2 * 2 * (P - 1) / P / BUSBW) if P > 1 else 0   # bf16 activations
    t = max(t_mem, t_cmp) + t_ar                       # no overlap of comm with compute
    return dict(b_max=b_max, B=B, step_ms=t * 1e3, user_tok_s=1 / t, gpu_tok_s=B / t / P)

def best_at(P, target):
    """Largest batch whose per-user speed still meets `target` tok/s; fleet throughput there."""
    ok = [replica(P, B) for B in range(1, replica(P)["b_max"] + 1)]
    ok = [r for r in ok if r["user_tok_s"] >= target]
    return max(ok, key=lambda r: r["gpu_tok_s"]) if ok else None

TPS = (1, 2, 4) if MODEL == "8B" else (2, 4, 8)
for P in (1, 2, 4, 8):
    r = replica(P)
    print(f"TP{P}: weights/GPU {N/P/1e9:5.1f} GB, KV room for {r['b_max']} seqs of {CTX} tokens")
print(f"Llama-3-{MODEL}, live context {CTX} tokens")
print(f"{'per-user target':>15s} " + " ".join(f"{'TP%d x %d' % (P, FLEET // P):>16s}" for P in TPS))
for target in (20, 40, 60, 100, 150):
    cells = []
    for P in TPS:
        r = best_at(P, target)
        cells.append(f"{'-':>16s}" if r is None else f"{r['gpu_tok_s']*FLEET:7.0f} (B={r['B']:3d})")
    print(f"{target:>9d} tok/s " + " ".join(f"{c:>16s}" for c in cells))
```

Output of `python3 layout.py 8192 70B; python3 layout.py 8192 8B; python3 layout.py 2048 8B`:

```text
TP1: weights/GPU  70.5 GB, KV room for 1 seqs of 8192 tokens
TP2: weights/GPU  35.3 GB, KV room for 54 seqs of 8192 tokens
TP4: weights/GPU  17.6 GB, KV room for 162 seqs of 8192 tokens
TP8: weights/GPU   8.8 GB, KV room for 376 seqs of 8192 tokens
Llama-3-70B, live context 8192 tokens
per-user target          TP2 x 8          TP4 x 4          TP8 x 2
       20 tok/s    17895 (B= 54)    25253 (B=162)    26244 (B=376)
       40 tok/s    17895 (B= 54)    24966 (B=156)    25413 (B=317)
       60 tok/s     8263 (B= 17)    19444 (B= 81)    22216 (B=185)
      100 tok/s                -     8402 (B= 21)    15807 (B= 79)
      150 tok/s                -                -     7801 (B= 26)
TP1: weights/GPU   8.0 GB, KV room for 119 seqs of 8192 tokens
TP2: weights/GPU   4.0 GB, KV room for 253 seqs of 8192 tokens
TP4: weights/GPU   2.0 GB, KV room for 521 seqs of 8192 tokens
TP8: weights/GPU   1.0 GB, KV room for 1057 seqs of 8192 tokens
Llama-3-8B, live context 8192 tokens
per-user target         TP1 x 16          TP2 x 8          TP4 x 4
       20 tok/s    88690 (B=119)    88773 (B=253)    88413 (B=521)
       40 tok/s    88690 (B=119)    88773 (B=253)    88413 (B=521)
       60 tok/s    85473 (B= 89)    85305 (B=177)    85805 (B=357)
      100 tok/s    75736 (B= 47)    76871 (B= 96)    79864 (B=199)
      150 tok/s    63378 (B= 26)    66208 (B= 55)    72407 (B=120)
TP1: weights/GPU   8.0 GB, KV room for 476 seqs of 2048 tokens
TP2: weights/GPU   4.0 GB, KV room for 1013 seqs of 2048 tokens
TP4: weights/GPU   2.0 GB, KV room for 2085 seqs of 2048 tokens
TP8: weights/GPU   1.0 GB, KV room for 4231 seqs of 2048 tokens
Llama-3-8B, live context 2048 tokens
per-user target         TP1 x 16          TP2 x 8          TP4 x 4
       20 tok/s   354761 (B=476)  338701 (B=1013)  308913 (B=2085)
       40 tok/s   354761 (B=476)  338701 (B=1013)  307397 (B=1921)
       60 tok/s   341894 (B=356)   324062 (B=675)  297171 (B=1238)
      100 tok/s   303331 (B=189)   291770 (B=364)   276677 (B=691)
      150 tok/s   255272 (B=106)   251441 (B=209)   251082 (B=418)
```

What the bound says:

- **For a model that nearly fills a GPU, more TP wins** until the weights are a small share of HBM. Llama-3-70B in FP8 at TP2 uses 35 GB of each 80 GB GPU for weights, leaving room for only 54 sequences at 8k. At every per-user target, TP4 or TP8 serves more tokens per GPU, and only TP8 reaches 150 tok/s per user.
- **For a small model, TP buys per-user speed and nothing else.** Llama-3-8B at 8k context is within 5% across TP1, TP2 and TP4 at relaxed targets. At short contexts TP1 replicas lead by 5-15% because they skip the all-reduces.
- **The bound flatters TP.** It assumes perfect scaling of every other part of the step, and real TP does not deliver that. Modal measured "a speedup of about 30% moving from one to two H100s ... rather than 100%" on a 35B-A3B MoE in SGLang ([source](https://modal.com/docs/examples/sglang_low_latency)). Fireworks found that the same eight GPUs serving Llama 70B "can be 4x cheaper per token" when configured for volume rather than for single-request speed ([source](https://fireworks.ai/blog/fireoptimizer)).

The rule that follows: **pick the smallest TP at which weights take at most about a quarter of each GPU's HBM and your per-user speed target is met, then fill the fleet with replicas of that size.** Modal reached the same answer for a 1T-parameter MoE (Kimi K2.6 in NVFP4 on B200) by measurement. TP4 was the choice, because "TP8 running only prefill achieved roughly the same throughput per GPU as TP4 running both prefill and decode". DP-attention gave "higher concurrency by sharding KV cache, but latency was worse", and per-GPU throughput dropped ([source](https://modal.com/blog/trillion-tokens-trillion-parameters)).

For large MoE models the rule changes, because expert weights dominate and EP with DP-attention spreads them without replicating KV. The [multi-node DP+EP launch](serving-hands-on.md#multi-node-data-parallel-moe) and the [arithmetic](arithmetic.md#moe-inference-arithmetic) cover that case. The flags, for completeness (untested):

```bash
# untested. vLLM: dense model, 4 replicas of TP2 on one 8-GPU node, one process per replica
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((2*i)),$((2*i+1)) vllm serve $MODEL -tp 2 --port $((8000+i)) &
done
# vLLM: MoE with expert parallelism across 8 GPUs (attention data-parallel)
vllm serve $MOE_MODEL -dp 8 --enable-expert-parallel
# SGLang: the same MoE layout. --enable-dp-attention is deprecated in favour of --attn-dp-size
python -m sglang.launch_server --model-path $MOE_MODEL --tp-size 8 --ep-size 8 --attn-dp-size 8
```

Separate processes per replica, behind your own router, are preferable to vLLM's `-dp` for dense models. Each one fails, restarts and upgrades on its own, and the router (not an internal load balancer) decides where each session goes.

**On a leased fleet.** Fireworks' and Modal's layout findings are measurements on rented GPUs like yours and carry over directly. What does not carry over is the absolute numbers: rerun `layout.py` with your model's config, then measure the two best layouts with the replay benchmark.

### The batch knobs: `max_num_seqs`, `max_num_batched_tokens` and the chunked-prefill budget

Two numbers bound each scheduler step ([Internals II](serving-engine-internals.md#the-unified-scheduler)):

- `--max-num-seqs` (vLLM) / `--max-running-requests` (SGLang) caps how many sequences run at once. It sets the decode batch size, so it trades ITL against throughput. vLLM also has `--max-num-active-seqs`, which caps the requests in the RUNNING state without shrinking the CUDA-graph capacity.
- `--max-num-batched-tokens` (vLLM) / `--chunked-prefill-size` (SGLang) caps the tokens processed per step. Decodes go in first and prefill chunks fill the rest, so this is the chunked-prefill budget. vLLM's [optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization.html) states the trade: "Smaller values (e.g., 2048) achieve better ITL", "Higher values achieve better time to first token (TTFT)", and "For optimal throughput, we recommend setting `max_num_batched_tokens > 8192` especially for smaller models on large GPUs." `--long-prefill-token-threshold` (default 0, meaning no cap) limits how many tokens one long prompt may take from a step, so a single 100k-token prompt does not take every chunk.

The [hands-on chapter measured](serving-hands-on.md#chunked-prefill-trades-ttft-for-itl-tail) this trade on a mock. The best published production number goes the other way from what intuition suggests. Prime Intellect, serving GLM-5.3 on GB200 NVL72 with vLLM and disaggregated prefill, *halved* the prefill token budget "from 8K to 4K per GPU", and "median queue wait fell from 550ms to 110ms, reducing median time to first token by roughly 20%" ([source](https://www.primeintellect.ai/blog/prime-inference)). A smaller budget lowered TTFT. The likely reason is that with large budgets, a few long prompts monopolized each step while the queue behind them grew. Smaller chunks let the scheduler interleave more requests' prefills, and the waiting requests started sooner. Which effect wins depends on the mix of prompt lengths, so this knob must be swept on the replayed trace, not set from a rule.

Applied Compute describes a trick in the opposite direction for short-output traffic (a routing model whose answers are a few tokens): "a short, bounded delay before prefill", so that fixed per-step costs are spread over larger prefill batches ([source](https://www.appliedcompute.com/case-studies/clay)). There is no engine flag for this. It lives in the gateway as a few milliseconds of batching. It only pays when steps are dominated by fixed costs, that is, when prompts are short and outputs are a handful of tokens.

```bash
# untested. Starting points to sweep from, not answers
vllm serve $MODEL --max-num-seqs 128 --max-num-batched-tokens 8192 --long-prefill-token-threshold 4096
python -m sglang.launch_server --model-path $MODEL --max-running-requests 128 --chunked-prefill-size 4096 \
  --schedule-policy lpm                       # longest-prefix-match first: favours cache hits
```

SGLang's `--schedule-policy` accepts `lpm`, `random`, `fcfs`, `dfs-weight`, `lof`, `priority` and `routing-key` (default `fcfs`). `lpm` admits the waiting request with the longest cached prefix first, which raises hit rates on shared-prefix traffic at some cost in fairness.

**On a leased fleet.** Prime Intellect's result is a knob any vLLM user can turn on any hardware. Its size depends on the prompt-length mix. Applied Compute's bounded delay is a gateway feature you write yourself in a few lines.

### `gpu_memory_utilization` and KV capacity

KV capacity is set by what is left after weights, activations and CUDA graphs:

- `--gpu-memory-utilization` (default 0.92 in the current docs) is the fraction of GPU memory vLLM may use. `--kv-cache-memory-bytes` sets the KV size per GPU directly and overrides it, which is the reproducible choice: the KV pool stays the same size when a new engine version changes activation memory. SGLang's equivalent is `--mem-fraction-static`.
- `--max-model-len` caps prompt plus output. A smaller cap does not shrink the pool, but it stops one request from claiming a huge share of it, and it changes the startup check that at least one full-length sequence fits.

The startup log prints the resulting KV size in tokens and the "maximum concurrency" at `max_model_len`. Read it and compare it with the session sizes from your trace. In the simulation above, about 16 sessions of this trace fit in 427k tokens, and every session beyond that cost throughput.

### KV capacity is the bottleneck for agentic workloads

Agentic traffic has long prompts, short outputs and long pauses. While a session waits for a tool, its KV either stays resident or gets evicted and recomputed. That makes KV capacity, not FLOPs, the binding constraint, as the `mean_vs_full.py` output showed. Applied Compute saw the same in its engine benchmark: both vLLM and SGLang degrade at high concurrency because KV evictions lower the cache hit rate, and "our primary bottleneck is KV capacity" ([source](https://www.appliedcompute.com/research/inference-benchmark)). Prime Intellect, running agentic RL at 1T scale, writes that "With more KV cache space, we can increase concurrency" ([source](https://www.primeintellect.ai/blog/rl-at-1t-scale)).

The levers, in order of cost:

1. **Do not waste KV on idle sessions you could cap.** Limit session concurrency per replica to the measured knee (16 in the simulation) and queue new sessions at the gateway, rather than letting the engine thrash.
2. **FP8 KV** halves bytes per token, which roughly doubles the sessions that fit ([Internals I](arithmetic.md#kv-cache-quantization) covers the accuracy side). In vLLM: `--kv-cache-dtype fp8` (also `fp8_e4m3`, `fp8_e5m2` and others). In SGLang: `--kv-cache-dtype fp8_e4m3` (SGLang has no plain `fp8` value). On Blackwell, NVFP4 KV goes further: Prime Intellect's NVFP4 KV raised capacity "from 1.09 million to 1.63 million cached tokens per decoder at the same memory budget" ([source](https://www.primeintellect.ai/blog/prime-inference)). vLLM lists `nvfp4` among its `--kv-cache-dtype` values.
3. **Offload KV to host memory.** A session in a tool wait does not need its KV in HBM, only somewhere faster to reload than to recompute. [Long-context serving](#kv-offloading-to-cpu-or-disk) below works out the arithmetic and the flags.
4. **Route so that sessions don't duplicate KV across replicas**, covered in [Routing](#routing).

### FP8 weights and FP8 KV cache

FP8 weights halve the weight bytes, which helps twice on a fixed fleet: decode reads fewer bytes per step, and the freed HBM becomes KV. For Llama-3-70B on 8 H100s, FP8 weights plus FP8 KV take capacity from 1.33M to 3.09M tokens ([Internals I table](arithmetic.md#max-concurrent-sequences)). Use a checkpoint calibrated offline when one exists. Otherwise `--quantization fp8` quantizes a bf16 checkpoint at load time ([hands-on](serving-hands-on.md#quantized-weights)). Check quality on your eval, not on a public benchmark. Applied Compute found that keeping the attention projections (about 17% of the parameters) in BF16 "reduced KL against the original model by 49-54% across all four recipes" they tried, from W8A16 to NVFP4 W4A4, on a 27B model ([source](https://www.appliedcompute.com/research/quantized-models)). Their starting point, "preserve attention and quantize the MLPs", is a good first variant when full FP8 loses quality.

### CUDA graphs and compile settings

[CUDA graphs](serving-engine-internals.md#cuda-graphs) remove per-kernel launch overhead from decode, and torch.compile fuses small ops. Both are on by default in vLLM, and both cost startup time. On a fixed fleet the replicas start rarely, so the trade is almost always to keep them on and make startup cheap with caches:

- `--optimization-level` (default 2): `-O0` gives the fastest startup and `-O3` the best steady-state performance. `--enforce-eager` disables both compilation and CUDA graphs, which is only for debugging. `--performance-mode` (`balanced`, `interactivity`, `throughput`) selects a preset.
- `--compilation-config` / `-cc` takes JSON or dotted overrides. The keys that matter are `cudagraph_capture_sizes` (the batch sizes captured) and `max_cudagraph_capture_size`. Batches larger than the largest captured size run without graphs. If your `--max-num-seqs` is 256 but the capture stops lower, large batches lose the benefit. If it is 512 and you never exceed 64, you pay for capture and graph memory you don't use. Set the capture list to cover the batch sizes your metrics actually show.
- SGLang: `--cuda-graph-max-bs-decode` and `--cuda-graph-max-bs-prefill` (the old `--cuda-graph-max-bs` is split in two), and `--enable-torch-compile` (marked experimental).
- Keep the compile cache. vLLM caches compiled artifacts under `~/.cache/vllm`, and Modal's vLLM example mounts that directory on a volume because compilation takes "typically tens of seconds to a few minutes, reduced to about ten seconds when loaded from the cache" ([source](https://modal.com/docs/examples/vllm_inference)). On a leased fleet, put it on local NVMe and copy it between identical nodes.

```bash
# untested. Capture graphs only for the batch sizes you serve
vllm serve $MODEL --max-num-seqs 64 \
  -cc '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]}'
```

### A methodical sweep, one variable at a time

Tuning by intuition fails because the knobs interact and the noise between seeds is often larger than the effect. The procedure:

1. Fix the SLO as percentiles (here TTFT p99 ≤ 5 s and per-request mean ITL p99 ≤ 50 ms) and fix the replay trace.
2. For each configuration, find its **goodput**: the highest offered rate at which the *worst of three seeds* still meets the SLO. Throughput at a fixed rate is the wrong metric because below saturation every configuration shows the offered rate.
3. Change one knob at a time, holding the others at the current best. Keep a value only if its goodput beats the current best. When goodput ties, prefer the better tail. This is coordinate descent: it can miss interactions, so after one pass, re-check the first knob at the final setting of the others.
4. Write each configuration, its seeds and the results to a file next to the deployment config, with the engine version and git SHA.

`sweep.py` implements it. The `sim` backend runs here. The `vllm` backend shows the same loop against a real server and is untested. The result JSON keys `output_throughput` and `p99_ttft_ms` are the ones `vllm/benchmarks/serve.py` writes, read from the source on GitHub.

```python
# sweep.py - one-knob-at-a-time tuning harness. Backend "sim" runs here; backend "vllm" is untested.
import json, subprocess, sys
from sim import Replica, run
from trace import gen_sessions, Driver

SLO = dict(ttft_p99=5.0, itl_p99=0.050)              # seconds; set from the product, not the engine
BASE = dict(max_num_batched_tokens=8192, max_num_seqs=256)
SWEEP = [("max_num_batched_tokens", [1024, 2048, 4096, 8192, 16384]),
         ("max_num_seqs", [8, 16, 32, 64, 256])]
SEEDS = (0, 1, 2)
RATES = (0.06, 0.09, 0.12, 0.15)                  # new sessions/s offered to one replica

def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

def measure_sim(cfg, seed, rate):
    S = gen_sessions(150, rate=rate, seed=seed, cv=2.0)
    d, rep = Driver(S), Replica(max_num_seqs=cfg["max_num_seqs"], max_batched=cfg["max_num_batched_tokens"])
    done = run([rep], lambda r, reps: 0, d.first(), d.on_done)
    t_end = max(r.t_done for r in done)
    itl = [(r.t_done - r.t_first) / (r.n_out - 1) for r in done]
    return dict(tok_s=sum(r.n_out for r in done) / t_end, ttft_p50=pct([r.t_first - r.t_arr for r in done], 50),
                ttft_p99=pct([r.t_first - r.t_arr for r in done], 99), itl_p99=pct(itl, 99))

def measure_vllm(cfg, seed, rate):                          # untested: needs a GPU box and a replay file
    serve = (f"vllm serve $MODEL --max-num-batched-tokens {cfg['max_num_batched_tokens']} "
             f"--max-num-seqs {cfg['max_num_seqs']} --port 8000")
    bench = (f"vllm bench serve --model $MODEL --base-url http://localhost:8000 --dataset-name custom "
             f"--dataset-path replay.jsonl --request-rate {rate} --burstiness 0.25 --seed {seed} "
             f"--save-result --result-filename r.json --percentile-metrics ttft,itl --metric-percentiles 50,99")
    srv = subprocess.Popen(serve, shell=True)       # wait for /health before benching in real use
    try: subprocess.run(bench, shell=True, check=True)
    finally: srv.terminate(); srv.wait()
    r = json.load(open("r.json"))
    return dict(tok_s=r["output_throughput"], ttft_p50=r["p50_ttft_ms"] / 1e3,
                ttft_p99=r["p99_ttft_ms"] / 1e3, itl_p99=r["p99_itl_ms"] / 1e3)

def evaluate(cfg, measure):
    """Goodput: the highest offered rate at which the worst of 3 seeds still meets the SLO."""
    best = None
    for rate in RATES:
        runs = [measure(cfg, s, rate) for s in SEEDS]
        worst = dict(rate=rate, tok_s=min(r["tok_s"] for r in runs),
                     ttft_p99=max(r["ttft_p99"] for r in runs), itl_p99=max(r["itl_p99"] for r in runs))
        if worst["ttft_p99"] > SLO["ttft_p99"] or worst["itl_p99"] > SLO["itl_p99"]: break
        best = worst
    return best

if __name__ == "__main__":
    measure = measure_vllm if "vllm" in sys.argv else measure_sim
    best = dict(BASE)
    for knob, values in SWEEP:                        # coordinate descent: one knob, others fixed
        print(f"-- {knob} (others: {json.dumps({k: v for k, v in best.items() if k != knob})})")
        results = {}
        for v in values:
            m = evaluate(dict(best, **{knob: v}), measure); results[v] = m
            print(f"  {v:>6}: " + ("fails SLO at every rate" if m is None else
                  f"goodput {m['tok_s']:5.0f} tok/s at {m['rate']:.2f} sessions/s   "
                  f"TTFT p99 {m['ttft_p99']:5.2f}s  ITL p99 {m['itl_p99']*1e3:5.1f}ms"))
        ok = [v for v in values if results[v]]
        if ok: best[knob] = max(ok, key=lambda v: (round(results[v]["tok_s"], -1), -results[v]["ttft_p99"]))  # ties: better tail
        print(f"  -> keep {knob}={best[knob]}")
    print("final:", json.dumps(best))
```

Output of `python3 sweep.py` (sim backend, 3 seeds per point):

```text
-- max_num_batched_tokens (others: {"max_num_seqs": 256})
    1024: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.89s  ITL p99  29.3ms
    2048: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.46s  ITL p99  30.0ms
    4096: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.37s  ITL p99  31.7ms
    8192: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.38s  ITL p99  32.5ms
   16384: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.33s  ITL p99  31.9ms
  -> keep max_num_batched_tokens=16384
-- max_num_seqs (others: {"max_num_batched_tokens": 16384})
       8: goodput   228 tok/s at 0.06 sessions/s   TTFT p99  2.17s  ITL p99  17.0ms
      16: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  4.92s  ITL p99  34.5ms
      32: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.33s  ITL p99  31.9ms
      64: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.33s  ITL p99  31.9ms
     256: goodput   386 tok/s at 0.12 sessions/s   TTFT p99  0.33s  ITL p99  31.9ms
  -> keep max_num_seqs=32
final: {"max_num_batched_tokens": 16384, "max_num_seqs": 32}
```

Reading it:

- **`max_num_batched_tokens` hardly matters on this trace.** Goodput is the same at every budget, and only the TTFT tail moves: 0.89 s at 1,024 against 0.33 s at 16,384. The reason is that with a 97% ideal hit rate, a turn's uncached prefill is about 600 tokens, smaller than every budget tried. The budget matters for traffic whose uncached prefills are large, such as Prime Intellect's 6K-token turns or cache-miss traffic. Sweep it anyway: the point of measuring is to find out which knobs matter for *your* trace.
- **`max_num_seqs` matters a lot below about 32**, because it caps how many of the active sessions can decode at once. At 8, goodput falls 41%. At 16 the run still passes at the same rate, but TTFT p99 is 4.9 s, close to the limit. At 32 and above the cap no longer binds, because the KV pool fills first.
- **The resolution is the rate grid.** Configurations that differ by less than one grid step tie. Refine the grid around the winner before trusting a small difference.



## Fleet shape

### Many independent replicas vs one multi-node engine

Once a model fits on one node, there are two ways to use more nodes: one engine spanning them (vLLM's multi-node data parallel, or TP/PP across nodes), or N independent engines behind a router. The published evidence favours independent replicas for anything that fits on one node:

- Prime Intellect's INTELLECT-3 report: with vLLM's built-in multi-node data parallel, "As the number of inference nodes increased, overall performance quickly plateaued." They switched to independent servers: "Each inference node is deployed as an entirely independent server, and the orchestrator maintains one client per node", with "a simple round-robin mechanism", and "inference throughput scales linearly with the number of nodes used, as desired" ([source](https://arxiv.org/html/2512.16144v1)). Their RL framework's docs describe the same layout: each node runs an independent vLLM replica behind one router ([prime-rl scaling docs](https://docs.primeintellect.ai/prime-rl/scaling)).
- Applied Compute, for a high-volume router model: "favor more independent replicas" ([source](https://www.appliedcompute.com/case-studies/clay)).

The reasons are structural. A multi-node engine synchronizes its ranks every step, so one slow rank slows everyone, and a failure takes down the whole engine. Cross-node TP all-reduces cost about 4x the intra-node latency ([Internals I](arithmetic.md#tensor-parallel-communication-in-decode)). Independent replicas fail alone, can be upgraded one at a time, and let the router, which knows about sessions and prefixes, decide placement.

Use one engine across nodes only when the model does not fit in one node, or for MoE models large enough that wide EP is the only efficient layout ([hands-on](serving-hands-on.md#multi-node-data-parallel-moe)).

**On a leased fleet.** This applies directly and costs nothing: run one engine per node, or per TP group within a node, behind the router built [below](#a-prefix-aware-router-and-a-simulator).

### When prefill/decode disaggregation pays on a fixed fleet

[Disaggregation](serving-engine-internals.md#disaggregated-prefill-and-decode) runs prefill and decode on separate GPU pools and moves the KV between them. The vLLM docs are blunt about what it is for: "Disaggregated prefill DOES NOT improve throughput". Its purpose is "Controlling tail ITL" and tuning TTFT and ITL separately ([source](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)). On a fixed fleet it has an extra cost: the split is a static partition of GPUs, and whichever pool is under-loaded at the moment sits idle.

The split ratio is set by the workload. `pd_ratio.py` computes the node-seconds of prefill and decode per request on 70B FP8 TP8 nodes. It assumes 40% MFU in prefill and decode at half its roofline bound, both stated assumptions. The ratio of decode to prefill node-seconds is the number of decode nodes per prefill node:

```python
# pd_ratio.py - how many prefill vs decode GPUs a workload needs (bounds; 70B FP8 on TP8 H100 nodes)
N, L, DM, KV_TOK = 70.55e9, 80, 8192, 2 * 80 * 8 * 128     # FP8 KV: 1 byte/elem
P, BW, PEAK, MFU = 8, 3.35e12, 1979e12, 0.4                # one node; FP8 dense peak; prefill MFU
DEC_EFF, MAX_SEQS = 0.5, 256          # decode reaches half its roofline bound; engine batch cap
HBM_KV = (80e9 * 0.9 - N / P) * P                          # KV bytes per node

def prefill_s(new, ctx):        # node-seconds to prefill `new` tokens on top of `ctx` cached ones
    flops = 2 * N * new + 2 * L * DM * (2 * ctx * new + new * new)   # linear + causal attention
    return flops / (P * PEAK * MFU)

def decode_s(out, ctx, user_tok_s):   # node-seconds per request at the largest batch meeting speed
    best = None
    for B in range(1, min(MAX_SEQS, int(HBM_KV / (ctx * KV_TOK))) + 1):
        t = (N / P + B * ctx * KV_TOK / P) / BW / DEC_EFF + 2 * L * 15e-6
        if 1 / t >= user_tok_s: best = (B, t)
    if best is None: return float('nan'), 0
    B, t = best
    return out * t / B, B

WORK = {  # name: (new prompt tokens per request, cached context, output tokens)
    "chat, no cache":         (1500, 0, 400),
    "RAG, 8k prompt":         (8000, 0, 300),
    "agent turn, 140k ctx":   (6000, 140000, 300),
    "agent, cache miss":      (146000, 0, 300),
}
for speed in (30, 60):
    print(f"per-user decode target {speed} tok/s")
    print(f"  {'workload':22s} {'prefill node-s':>14s} {'decode node-s':>13s} {'decode B':>8s} {'P:D nodes':>9s}")
    for name, (new, ctx, out) in WORK.items():
        p = prefill_s(new, ctx); d, B = decode_s(out, ctx + new + out // 2, user_tok_s=speed)
        print(f"  {name:22s} {p:14.3f} {d:13.3f} {B:8d} {'1:%.1f' % (d / p):>9s}")
```

Output of `python3 pd_ratio.py`:

```text
per-user decode target 30 tok/s
  workload               prefill node-s decode node-s decode B P:D nodes
  chat, no cache                  0.034         0.020      256     1:0.6
  RAG, 8k prompt                  0.191         0.039      256     1:0.2
  agent turn, 140k ctx            0.489         0.700       14     1:1.4
  agent, cache miss               7.665         0.700       14     1:0.1
per-user decode target 60 tok/s
  workload               prefill node-s decode node-s decode B P:D nodes
  chat, no cache                  0.034         0.020      256     1:0.6
  RAG, 8k prompt                  0.191         0.055       90     1:0.3
  agent turn, 140k ctx            0.489         0.996        5     1:2.0
  agent, cache miss               7.665         0.996        5     1:0.1
```

- **The ratio swings by more than 10x with the cache hit rate.** The same agent turn needs 1 prefill node per 2 decode nodes when its 140k-token history is cached, and 10 prefill nodes per decode node when it is not. A static split on a fixed fleet is only right if the hit rate is stable. That is one more reason to fix routing first.
- **The per-user speed target moves it too.** A faster target means smaller decode batches, so more decode nodes per prefill node.
- **Prime Intellect's production ratio is decode-heavy.** On GLM-5.3 with a target of "100 end-to-end tokens per second per user", "a 1:4 P/D ratio serves the most", reaching "66 sessions per prefill group at 101 tok/s/user and 100 output tok/s per GPU" ([source](https://www.primeintellect.ai/blog/prime-inference)). Their turns add "about 6K tokens to a 140K-token prompt", which is the cached agent row above, at a higher speed target. They also report that Dynamo's routing over the split pools "reduced p90 inter-token latency by nearly 40% in our tests". In their RL work they note that some "model↔env pairs produce as much as a 4:1 prefill:decode token ratio", and they disaggregate so that "long prefill requests (lengthy tool outputs, etc.) do not throttle the decode workers" ([source](https://www.primeintellect.ai/blog/rl-at-1t-scale)).
- **Cache-aware disaggregation** adds a third pool. Together sends low-reuse prompts to "pre-prefill" nodes that write KV to a distributed cache, and high-reuse prompts to prefill nodes that read from it. On their coding-agent workload it reached about 1.1-1.15 QPS/GPU against about 0.75-0.8 for plain 2P1D (two prefill, one decode), a "~40% increase in sustainable throughput" ([source](https://www.together.ai/blog/cache-aware-disaggregated-inference)).

When it pays on a fixed fleet: you have enough GPUs that each pool has several nodes (so rounding to whole nodes wastes little), the ITL tail is your binding SLO, the hit rate is stable, and you have RDMA between pools ([KV transfer cost](serving-engine-internals.md#kv-transfer-cost)). Below about two nodes per pool, a colocated engine with a tuned chunk budget is the baseline to beat, and usually wins ([sizing sketch](serving-engine-internals.md#when-it-wins-and-when-it-does-not)). The launch commands are in the [hands-on chapter](serving-hands-on.md#disaggregated-prefill-and-decode).

**On a leased fleet.** Prime Intellect's and Together's setups run on dozens to hundreds of GPUs with RDMA. On 16 GPUs, a 1:4 split means one prefill node and four decode nodes of 2-4 GPUs each, which is workable only if the interconnect between them is fast. The cheaper equivalent at small scale is colocated serving with a small `--max-num-batched-tokens` and `--long-prefill-token-threshold`, which bounds the ITL tail without a second pool.



## Routing

On a fleet of N replicas, each replica has its own prefix cache. A router that ignores this spreads every session over all N caches, so each cache holds a fraction of each session, and the fleet's effective cache size divides by N ([hands-on routing experiment](serving-hands-on.md#routing-experiment)). Routing is the cheapest large lever in this chapter: it needs no new hardware and no engine change.

### What the measurements say

**llm-d** (the Kubernetes-native serving stack from Red Hat, Google, IBM and others) published the cleanest comparison. They ran 8 vLLM pods of 2 H100s each on Qwen3-32B, each holding a 307,328-token KV cache. The workload was 150 customers, each with a 6,000-token shared prefix, 5 users per customer and 1,200-token queries, under Poisson arrivals ramping from 3 to 60 QPS. KV demand was about 73% of cluster capacity. Results ([source](https://llm-d.ai/blog/kvcache-wins-you-can-see)):

| Scheduler | Output tok/s | TTFT p90 (s) | TTFT mean (s) | vLLM wait queue (mean) |
|-|-:|-:|-:|-:|
| precise prefix-aware | 8,730.0 | 0.542 | 0.298 | 0.1 |
| approximate prefix-aware | 6,944.4 | 31.083 | 13.316 | 8.1 |
| load-only | 4,428.7 | 94.865 | 46.987 | 28.9 |
| random | 4,428.7 | 92.551 | 45.281 | 27.3 |

**Precise** routing knows what each pod holds because every vLLM pod publishes *KV events* (block stored, block evicted) and the router keeps a global index of block hash to pods. **Approximate** routing predicts what each pod holds from its own routing history, without feedback. The precise router doubled throughput over load-only and random routing, and cut P90 TTFT 57x relative to approximate routing. The gap between precise and approximate appears because KV demand was close to capacity: pods evicted blocks the approximate router still believed were cached, so its guesses went stale.

**Baseten**, using NVIDIA Dynamo's KV router (it hashes requests into a radix tree of cached blocks and weighs overlap against GPU load), reports a stress test on Qwen3-Coder-480B-A35B with 4 replicas, about 50k-token inputs and about 1k-token outputs, against random routing: an 89% cache hit rate, "34% reduction in TPOT and 50% reduction in TTFT on average", 61% more requests per second and 62% more output tokens per second. On shadowed production traffic, P95 TTFT fell 48% and P99 fell 49%. They "often see 2x faster TTFT in production" ([source](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/)).

**SGLang**'s cache-aware router keeps an approximate radix tree per worker. On a workload of long-prefix groups it reported "up to 1.9x throughput increase and 3.8x hit rate improvement": 82,665 to 158,596 tok/s, with the hit rate going from 20% to 75% ([source](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).

**On a leased fleet.** All three run on any fleet. llm-d targets Kubernetes. Dynamo's router and SGLang's router run as plain processes. The precise mode needs KV events from the engines, which vLLM, SGLang and TensorRT-LLM publish over ZMQ ([llm-d docs](https://llm-d.ai/docs/0.7/well-lit-paths/precise-prefix-cache-aware)). The cheapest equivalent is SGLang's router in `cache_aware` mode in front of your replicas, or the approximate router below, which is about 30 lines. Move to precise routing when KV demand is close to capacity, the regime where llm-d saw approximate routing fall behind.

### Session stickiness for multi-turn agents

**Session affinity** sends every request of a session to the same replica, usually by hashing a session id. For agents, where 61% of prompt tokens in our trace are the session's own history, it captures most of the benefit with no index at all. Hosted APIs work this way. Fireworks says "Prompt caching only works within 1 replica", and asks clients to send a session identifier "either in the `user` field of the request body or in the `x-session-affinity` header" for TTFT reductions "by as much as 80%" ([source](https://docs.fireworks.ai/guides/prompt-caching)).

Pure hashing has three known failure modes, all of which Modal hit when serving Kimi K2.6 to coding agents ([source](https://modal.com/blog/trillion-tokens-trillion-parameters)):

- **Unlucky hashing.** With few replicas, the hash puts several heavy sessions on one replica.
- **Hot sessions.** "The work per session was not uniform." One very active session can saturate its replica. Modal's fix splits heavy sessions above a load threshold across replicas.
- **Scale-up moved too many sessions.** A new replica remapped many existing sessions and threw away their caches. The fix was to send *new* sessions to new replicas and leave existing ones in place.

Their router also places sessions by load (running requests plus KV utilization) and rebalances, which brought load variance below half the mean. Modal found that "only around 4 users' conversations could be scheduled onto a single replica without tanking interactivity" at their context lengths (about 100k input tokens per request). That is a hard cap on sessions per replica, which plain hashing ignores.

Stickiness has a reliability cost too. In Anthropic's August 2025 incident, some requests were "misrouted to servers configured for the upcoming 1M token context window". Because "our routing is 'sticky'", "subsequent follow-ups were likely to be served by the same incorrect server", and at the worst hour "16% of Sonnet 4 requests were affected" ([postmortem](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)). Stickiness concentrates the damage from a bad replica on the same users. A sticky router needs health-based ejection that also re-homes the sessions of an ejected replica.

**On a leased fleet.** Session headers and load-aware placement are router logic, with no infrastructure attached. The "new sessions to new replicas" rule also covers rolling restarts on a fixed fleet: when a replica comes back after an upgrade, give it new sessions rather than migrating old ones.

### A prefix-aware router and a simulator

`router_sim.py` puts five policies in front of 4 replicas of the `sim.py` engine and replays the multi-turn trace, with sessions arriving at 1 per second and CV 2:

- **random**: uniform choice.
- **least-loaded**: fewest waiting plus running requests.
- **session-hash**: a CRC of the session id modulo N, which is pure stickiness.
- **prefix-approx**: the router remembers which pages it sent to each replica, as an LRU sized to the replica's capacity. It picks the replica with the longest remembered prefix, unless that replica has more than `slack` requests above the least-loaded one, in which case it takes the least-loaded. This is the design of SGLang's `cache_aware` policy and of the hands-on gateway's bounded-load prefix mode, using page hashes instead of the first N characters.
- **prefix-precise**: the same rule, but it reads each replica's actual cache, as KV events allow.

```python
# router_sim.py - random vs least-loaded vs prefix-aware routing on a multi-turn agent trace
import random, statistics as st, sys, zlib
from collections import OrderedDict
from sim import Replica, run
from trace import gen_sessions, Driver

class PrefixRouter:
    """Approximate prefix-aware router: remembers which pages it sent where (no engine feedback),
    picks the replica with the longest remembered prefix unless it is `slack` requests busier
    than the least-loaded replica. Same idea as SGLang's cache_aware policy."""
    def __init__(self, n, pages_per_replica, slack=4):
        self.idx = [OrderedDict() for _ in range(n)]          # per replica: page -> None, LRU order
        self.cap, self.slack = pages_per_replica, slack
    def match(self, i, blocks):
        n = 0
        for b in blocks:
            if b not in self.idx[i]: break
            n += 1
        return n
    def __call__(self, r, reps):
        scores = [self.match(i, r.blocks) for i in range(len(reps))]
        best = max(range(len(reps)), key=lambda i: (scores[i], -reps[i].load()))
        least = min(range(len(reps)), key=lambda i: reps[i].load())
        i = best if reps[best].load() - reps[least].load() <= self.slack else least
        idx = self.idx[i]
        for b in r.blocks + r.out_blocks:                      # remember what replica i now holds
            idx[b] = None; idx.move_to_end(b)
        while len(idx) > self.cap: idx.popitem(last=False)
        return i

def precise(slack=4):
    """Precise variant: reads each replica's real cache (what KV events give llm-d / Dynamo)."""
    def route(r, reps):
        best = max(range(len(reps)), key=lambda i: (reps[i].match(r.blocks), -reps[i].load()))
        least = min(range(len(reps)), key=lambda i: reps[i].load())
        return best if reps[best].load() - reps[least].load() <= slack else least
    return route

def policies(n, pages, seed):
    rng = random.Random(seed)
    return {
        "random":       lambda r, reps: rng.randrange(n),
        "least-loaded": lambda r, reps: min(range(n), key=lambda i: reps[i].load()),
        "session-hash": lambda r, reps: zlib.crc32(str(r.sid).encode()) % n,
        "prefix-approx": PrefixRouter(n, pages),
        "prefix-precise": precise(),
    }

def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

if __name__ == "__main__":
    N, rate = 4, float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    S = gen_sessions(600, rate=rate, seed=seed, cv=2.0)
    print(f"{N} replicas, {len(S)} sessions at {rate}/s (gamma CV 2), seed {seed}")
    print(f"{'policy':15s} {'out tok/s':>9s} {'hit':>6s} {'TTFT p50':>8s} {'TTFT p99':>8s} {'max/mean steps':>14s}")
    for name, pol in policies(N, Replica().cap, seed).items():
        d, reps = Driver(S), [Replica() for _ in range(N)]
        done = run(reps, pol, d.first(), d.on_done)
        t_end = max(r.t_done for r in done)
        ttft = [r.t_first - r.t_arr for r in done]
        hit = sum(x.hit_tok for x in reps) / sum(x.prompt_tok for x in reps)
        steps = [x.steps for x in reps]
        print(f"{name:15s} {sum(r.n_out for r in done)/t_end:9.0f} {hit:6.1%} {pct(ttft,50):8.2f} "
              f"{pct(ttft,99):8.2f} {max(steps)/st.mean(steps):14.2f}")
```

Output of `python3 router_sim.py 1.0 0; python3 router_sim.py 1.0 1; python3 router_sim.py 1.0 2; python3 router_sim.py 0.45 0` (out tok/s is total output tokens over the time until the last request finishes; max/mean steps is load imbalance across replicas):

```text
4 replicas, 600 sessions at 1.0/s (gamma CV 2), seed 0
policy          out tok/s    hit TTFT p50 TTFT p99 max/mean steps
random               1083  54.8%     2.10    34.38           1.19
least-loaded         1184  55.9%     1.75    17.96           1.25
session-hash         1147  77.4%     0.07    21.70           1.27
prefix-approx        1492  82.2%     0.07     8.08           1.17
prefix-precise       1486  81.9%     0.07     8.22           1.16
4 replicas, 600 sessions at 1.0/s (gamma CV 2), seed 1
policy          out tok/s    hit TTFT p50 TTFT p99 max/mean steps
random               1210  47.3%     2.73    55.50           1.28
least-loaded         1316  45.3%     9.49    24.10           1.16
session-hash         1403  63.0%     2.43    25.35           1.11
prefix-approx        1554  62.6%     3.87    20.15           1.12
prefix-precise       1545  61.2%     3.60    19.59           1.18
4 replicas, 600 sessions at 1.0/s (gamma CV 2), seed 2
policy          out tok/s    hit TTFT p50 TTFT p99 max/mean steps
random                960  47.6%     2.04    50.33           1.22
least-loaded         1111  48.4%     4.21    21.95           1.16
session-hash          866  60.1%     1.52    17.49           1.28
prefix-approx        1335  69.6%     0.47    18.36           1.17
prefix-precise       1339  69.2%     0.56    18.10           1.15
4 replicas, 600 sessions at 0.45/s (gamma CV 2), seed 0
policy          out tok/s    hit TTFT p50 TTFT p99 max/mean steps
random               1005  63.6%     0.27    23.82           1.07
least-loaded         1123  69.1%     0.15     3.53           1.17
session-hash         1254  96.4%     0.04     0.13           1.11
prefix-approx        1229  95.2%     0.04     0.78           1.15
prefix-precise       1229  95.2%     0.04     0.78           1.15
```

At 1 session per second, which saturates the 4 replicas:

- **Prefix-aware routing beats random by 28-39% in throughput across the three seeds** (1,492 vs 1,083, 1,554 vs 1,210, 1,335 vs 960) and cuts TTFT p99 by 2.7-4.3x. Least-loaded routing recovers only part of that, because it balances load but not cache.
- **Session hashing is erratic.** It is the best policy at low load (0.45/s: TTFT p99 0.13 s, the best hit rate), but at saturation it comes third, fourth and last on the three seeds (866 tok/s on seed 2, below random), depending on how the hash happens to place the heavy sessions. This is Modal's "unlucky hashing" and "hot sessions" in a 30-line model.
- **Precise and approximate routing tie here.** In this simulator the router's memory of what it sent closely tracks each replica's cache, because there is only one router and nothing else evicts. In llm-d's benchmark the gap was large because many requests share prefixes across customers and KV demand was close to capacity. The simulation cannot settle that question. It only shows that approximate routing captures most of the gain when the router is the only writer.
- **One seed is an anecdote.** The ranking of prefix-aware routing first is stable across seeds. The ranking of the others is not.

The router to deploy, using SGLang's (untested; the gateway was renamed "SGLang Model Gateway", and the flags are from its [current docs](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)):

```bash
# untested. cache_aware is the default policy; the numbers shown are the defaults
python -m sglang_router.launch_router \
  --worker-urls http://10.0.0.1:8000 http://10.0.0.2:8000 http://10.0.0.3:8000 http://10.0.0.4:8000 \
  --policy cache_aware --cache-threshold 0.3 \
  --balance-abs-threshold 64 --balance-rel-threshold 1.5
```

`--cache-threshold` is the minimum prefix-match ratio for a cache-directed choice. The two balance thresholds play the role of `slack`: when the load gap exceeds both, the router falls back to the least-loaded worker. The router works in front of vLLM workers as well as SGLang ones, since it routes on the request text. The other policies are `random`, `round_robin`, `power_of_two` and `bucket`.

### Isolation between tenants

A shared prefix cache leaks timing information: a fast TTFT tells a client that someone else recently sent the same prefix. Fireworks notes this "minor timing side-channel" on dedicated deployments and offers an `x-prompt-cache-isolation-key` header ([source](https://docs.fireworks.ai/guides/prompt-caching)). vLLM accepts a per-request `cache_salt`. It is "injected into the hash of the first block, ensuring that only requests with the same salt can reuse cached KV blocks" ([vLLM prefix-caching design doc](https://docs.vllm.ai/en/latest/design/prefix_caching.html)). Set it per tenant in the gateway when tenants must not observe each other. Tenants then stop sharing system-prompt pages with each other, and the router should treat the salt as part of the prefix.



## Speculative decoding in production

The mechanics and the [high-batch arithmetic](serving-engine-internals.md#when-speculation-hurts-at-high-batch) are in Internals II. What production adds is that **the draft's acceptance rate on your traffic decides everything**, and that it is easy to ship a draft that makes serving slower.

The arithmetic at low batch: with per-token acceptance $\alpha$, $k$ drafted tokens and a draft step costing $c$ target steps, one verify step yields $(1-\alpha^{k+1})/(1-\alpha)$ tokens and costs $1 + kc$ target steps:

```python
# spec.py - when does a draft pay? Speedup at low batch (verify costs ~one target step)
# alpha: per-token acceptance; k: drafted tokens; c: one draft step as a fraction of a target step
def tokens_per_verify(a, k): return (1 - a ** (k + 1)) / (1 - a)      # expected accepted + bonus
def speedup(a, k, c): return tokens_per_verify(a, k) / (1 + k * c)
def best_k(a, c): return max(range(1, 9), key=lambda k: speedup(a, k, c))

drafts = {"EAGLE-style head (c=0.03)": 0.03, "small draft model (c=0.10)": 0.10,
          "big draft, e.g. 8B for 70B (c=0.20)": 0.20}
print(f"{'alpha':>5s} " + " ".join(f"{d:>36s}" for d in drafts))
for a in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    cells = []
    for c in drafts.values():
        k = best_k(a, c)
        cells.append(f"k=4: {speedup(a, 4, c):4.2f}x  best k={k}: {speedup(a, k, c):4.2f}x")
    print(f"{a:5.2f} " + " ".join(f"{x:>36s}" for x in cells))
for c in drafts.values():   # acceptance below which speculation (at k=4) is a net loss
    lo = next(a / 100 for a in range(1, 100) if speedup(a / 100, 4, c) >= 1)
    print(f"break-even acceptance at k=4, c={c:.2f}: {lo:.2f}")
```

Output of `python3 spec.py`:

```text
alpha            EAGLE-style head (c=0.03)           small draft model (c=0.10)  big draft, e.g. 8B for 70B (c=0.20)
 0.30          k=4: 1.27x  best k=2: 1.31x          k=4: 1.02x  best k=1: 1.18x          k=4: 0.79x  best k=1: 1.08x
 0.50          k=4: 1.73x  best k=4: 1.73x          k=4: 1.38x  best k=2: 1.46x          k=4: 1.08x  best k=1: 1.25x
 0.60          k=4: 2.06x  best k=5: 2.07x          k=4: 1.65x  best k=3: 1.67x          k=4: 1.28x  best k=2: 1.40x
 0.70          k=4: 2.48x  best k=7: 2.60x          k=4: 1.98x  best k=4: 1.98x          k=4: 1.54x  best k=3: 1.58x
 0.80          k=4: 3.00x  best k=8: 3.49x          k=4: 2.40x  best k=6: 2.47x          k=4: 1.87x  best k=4: 1.87x
 0.90          k=4: 3.66x  best k=8: 4.94x          k=4: 2.93x  best k=8: 3.40x          k=4: 2.28x  best k=7: 2.37x
break-even acceptance at k=4, c=0.03: 0.11
break-even acceptance at k=4, c=0.10: 0.29
break-even acceptance at k=4, c=0.20: 0.46
```

- **A cheap draft head breaks even at very low acceptance (0.11 at $k=4$). A big draft model needs 0.46.** That is the case for EAGLE-style heads, which read the target's hidden states, and against "use the 8B as a draft for the 70B".
- **The best $k$ grows with acceptance.** At $\alpha=0.9$ an EAGLE-style head gains from 3.66x to 4.94x by drafting 8 tokens instead of 4. At $\alpha=0.3$, a big draft should draft 1 token or none.
- **This model is the optimistic case.** It ignores scheduling overhead, the draft competing for the GPU, and the batch effect. Real systems do worse.

Fireworks published what "worse" looks like. A generic draft model with a 29% hit rate "actually increased latency by 1.5x", while a draft trained on the customer's traffic reached a "76% hit rate with a 2x speed increase" ([source](https://fireworks.ai/blog/fireoptimizer)). The idealized model above puts 29% at break-even for a small draft model. The real system lost 50%. Baseten found two reasons why real systems fall short of the model. Draft and target on the same GPU "had a tendency to fight for resources", and the gains held "only at limited batch sizes". Their best results were Qwen2.5-Coder-14B with a 0.5B draft on one H100 (mean latency 444.39 vs 818.43 ms) and Llama-3.1-70B FP8 with an 8B draft on 4 H100s (627.96 vs 1206.25 ms). TTFT got *worse* with speculation, 56.05 vs 42.69 ms ([source](https://baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm)).

The current best practice is a small head trained for the target, then fine-tuned on your traffic:

- **EAGLE-3** took GPT-OSS-120B "from ~400 to ~650 tokens per second" at Baseten, with "EAGLE models ... ~1B parameters", running TP8 EP1 on 8 B200s ([source](https://www.baseten.co/blog/how-we-made-the-fastest-gpt-oss-on-nvidia-gpus-60-percent-faster/)).
- Together, serving a coding-agent workload with prompts "from ~45k to 200k tokens", uses EAGLE with 3 draft tokens at about 70% acceptance ([source](https://www.together.ai/blog/coding-agent-benchmarks)).
- Modal fine-tuned a DFlash speculator on coding traces and raised the mean accepted length from 5.00 to 5.84 tokens, "about 20%" faster ([source](https://modal.com/blog/trillion-tokens-trillion-parameters)).
- Together's ATLAS adapts a speculator online to live traffic. On a FP8 DeepSeek-V3.1 baseline of 105 tok/s, it reached up to about 500 tok/s at batch size 1 on 4 B200s. On RL rollouts of Qwen2.5-7B, acceptance rose "from below 10% to above 80% over 1.4k training steps" ([source](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)).

**When to turn it off: high batch sizes.** Speculation trades spare compute for latency. At large batch the verify step becomes compute-bound and the drafted tokens cost real FLOPs ([Internals II](serving-engine-internals.md#when-speculation-hurts-at-high-batch)). Modal's SGLang guide puts it plainly: "speculative decoding is not good for compute-bound workloads", meaning batches "on the scale of dozens to hundreds" ([source](https://modal.com/docs/examples/sglang_low_latency)). On a fixed fleet run for throughput, that is often your operating point. vLLM can vary $k$ with batch size through `num_speculative_tokens_per_batch_size`, a list of `[start_bs, end_bs, k]` ranges in which `k=0` disables speculation for that range. The docs note it is not compatible with DP > 1.

```bash
# untested. EAGLE-3 with 3 draft tokens up to batch 32, 1 up to 128, off above
vllm serve $MODEL --speculative-config '{"method": "eagle3", "model": "'$EAGLE_HEAD'",
  "num_speculative_tokens": 3,
  "num_speculative_tokens_per_batch_size": [[1, 32, 3], [33, 128, 1], [129, 512, 0]]}'
# untested. SGLang EAGLE-3
python -m sglang.launch_server --model-path $MODEL --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path $EAGLE_HEAD --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

vLLM's `--speculative-config` `method` values include `eagle3`, `draft_model`, `ngram`, `suffix`, `mtp` (the model's own multi-token-prediction head, where one exists) and `dflash`. Prompt-lookup methods (`ngram`, `suffix`) need no draft model and do well on edits and code that copy from the prompt.

The procedure:

1. Measure acceptance on your replayed trace, not on a public benchmark. vLLM exports `spec_decode_*` counters, and [the hands-on chapter](serving-hands-on.md#speculative-decoding) shows where to read them.
2. Keep it only if goodput at your operating batch size improves, with the same sweep as above.
3. If acceptance is under about 50%, train or fine-tune a head on a few days of your outputs before giving up.
4. Watch acceptance in production. It drifts when traffic drifts, which is why Baseten's metrics page offers "an acceptance rate graph" next to "KV cache hit rate" for LLM deployments ([source](https://docs.baseten.co/observability/metrics)).

**On a leased fleet.** Training a draft head is a fine-tuning job of hours on one node, which a leased fleet can run in a trough. Fireworks' and Together's adaptive systems are proprietary. The cheaper equivalent is periodic offline fine-tuning of an EAGLE-3 head on logged outputs, redeployed like any model update.



## Packing many models onto a fixed fleet

A startup rarely serves one model. There are fine-tunes per customer, a small model for classification, a big one for hard requests, and an embedding model. Giving each its own GPUs fragments the fleet: every model is sized for its own peak, and the peaks don't line up. The techniques below let models share GPUs.

### Multi-LoRA: many fine-tunes on one base

If the fine-tunes are LoRA adapters on one base model, one engine can serve all of them in the same batch ([mechanism and kernels](serving-engine-internals.md#multi-lora-serving)). The measured cost is small. Fireworks reports multi-LoRA throughput that can "approach 90% of base model performance", with "hundreds of fine-tuned LoRAs on a single base model" ([source](https://fireworks.ai/blog/multi-lora)). Baseten describes a cache hierarchy for adapters: GPU memory is near-instant, host memory 1-2 ms, disk 10-100 ms and network about 100 ms, with "A LoRA for Llama 3 can be as small as 16 megabytes" ([source](https://www.baseten.co/blog/how-to-serve-10-000-fine-tuned-llms-from-a-single-gpu/)).

```bash
# untested. vLLM: up to 8 adapters per batch, rank <= 64, 64 more cached in host RAM
vllm serve $BASE --enable-lora --max-loras 8 --max-lora-rank 64 --max-cpu-loras 64 \
  --lora-modules acme=/adapters/acme globex=/adapters/globex
# load or unload adapters at runtime without a restart
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve $BASE --enable-lora ...
curl -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "initech", "lora_path": "/adapters/initech"}'
# SGLang: base-model requests count toward --max-loras-per-batch (default 8)
python -m sglang.launch_server --model-path $BASE --lora-paths acme=/adapters/acme --max-loras-per-batch 8
```

Clients select an adapter by passing its name as `model`. Three settings deserve care:

- `--max-lora-rank` (default 16 in vLLM) sizes the kernels' buffers for *every* adapter, so one rank-256 adapter makes all of them more expensive. Group adapters by rank across replicas.
- `--max-loras` (default 1 in vLLM) caps the number of distinct adapters in one step. If a batch needs more, requests wait. Set it from the number of distinct adapters active in a typical second of your trace.
- Adapter affinity is a routing problem, like prefix affinity. The adapter id is part of the prefix-cache hash, so two tenants' requests with the same system prompt don't share KV across adapters.

**On a leased fleet.** This is the most efficient way to host per-customer models on fixed GPUs, and it uses only engine features. It needs fine-tunes to be LoRAs. Full fine-tunes don't share weights, which leads to the next technique.

### Multiplexing and swapping whole models

For different base models, there are three options, in order of cost:

1. **Co-locate small models on one GPU.** Two engines on one GPU, each with `--gpu-memory-utilization 0.45`. Each gets less KV, and they compete for bandwidth. This works for a small classifier or embedding model next to a mid-size LLM.
2. **Sleep and wake.** vLLM's sleep mode frees GPU memory without killing the process. "Level 1 sleep will offload the model weights and discard the KV cache", so waking copies weights back from host RAM (the host needs enough RAM to hold them). "Level 2 sleep will discard the model parameters and the KV cache", so weights must be reloaded on wake ([vLLM docs](https://docs.vllm.ai/en/latest/features/sleep_mode.html)). The process keeps its CUDA context, compiled graphs and Python state, which is most of what makes a cold start slow. The docs give no timings. A rough bound for level 1 is weight bytes over host-to-device bandwidth: 16 GB at an assumed 25-50 GB/s of PCIe Gen5 is about 0.3-0.6 s, plus re-allocation and warmup. Measure it before relying on it.
3. **Restart from a snapshot.** Checkpoint the whole process, GPU state included, and restore it. NVIDIA's `cuda-checkpoint` "can toggle the CUDA state of a process ... between suspended and running" and "can be used with CRIU to checkpoint and restore a CUDA application". As of driver 550 it acts on "a single process, not a process tree" and "Doesn't support UVM or IPC memory" ([NVIDIA](https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/)). Multi-GPU engines are therefore the hard case.

```bash
# untested. Sleep-mode endpoints exist only with VLLM_SERVER_DEV_MODE=1: keep them off any public port
VLLM_SERVER_DEV_MODE=1 vllm serve $MODEL_A --enable-sleep-mode --port 8001 &
curl -X POST 'localhost:8001/sleep?level=1'     # weights -> host RAM, KV dropped, GPU memory freed
curl -X POST 'localhost:8001/wake_up'           # weights back; the KV cache starts empty
curl localhost:8001/is_sleeping
```

The pattern on a fixed fleet: keep the top few models resident on dedicated replicas, and let one or two "swap" replicas hold long-tail models, waking the one that has queued requests. The gateway queues requests for a sleeping model and sends the wake call. That is a few dozen lines of code, plus a policy for how long a model stays awake after its last request.

**On a leased fleet.** Modal's serverless numbers come from the third option at scale. Inference servers "that take upwards of 2 kiloseconds to boot naïvely boot in ~50 seconds on Modal" through a stack of a GPU buffer, lazy image loading, CPU memory snapshots taken with gVisor's `runsc checkpoint`, and GPU memory snapshots via the NVIDIA driver. They note that snapshots don't help with loading weights from storage, and that "snapshotting vLLM or SGLang works best with weight offloading" ([source](https://modal.com/blog/truly-serverless-gpus)). gVisor is only the container runtime that makes CPU checkpointing possible in Modal's setup. CRIU plus `cuda-checkpoint` does the same on a plain Linux host. On a warm fixed fleet, most of the gain comes for less: sleep mode keeps the process alive, weights sit on local NVMe or in host RAM, and the compile cache is kept on disk. Snapshots matter when you start replicas often, which a fixed fleet rarely does.

### Priority: filling idle capacity with batch work

A fleet sized for the interactive peak is under-used the rest of the day. Modal cites a 2024 industry survey in which "the majority of organizations achieve less than 70% GPU Allocation Utilization when running at peak demand" ([source](https://modal.com/blog/gpu-utilization-guide)). On a lease, those idle hours are already paid for. The fix is to keep a queue of offline work (evals, synthetic data, embedding backfills, document processing, RL rollouts) and run it in the gaps, without hurting the interactive SLO.

The naive version fails. `backfill.py` runs the agent trace on one replica at a light load, then adds a backlog of 20,000 offline requests (2k-token unique prompts, 100-900 output tokens) in four ways: plain FCFS, priority ordering, and priority with the gateway keeping only 16 or 48 batch requests in flight:

```python
# backfill.py - fill an under-used replica with offline batch work without hurting interactive TTFT
import random
from sim import Req, Replica, run
from trace import gen_sessions, Driver

HORIZON = 1200.0                                     # s of simulated wall clock

def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

def batch_req(i, rng):                               # offline job: unique 2k-token prompt, 100-900 out
    n = 2048; r = Req(("b", i), 0, [("b", i, j) for j in range(n // 256)], n, rng.randint(100, 900), prio=1)
    r.out_blocks = []; return r

def experiment(mode, cap=None, seed=0):
    rng = random.Random(seed)
    d = Driver(gen_sessions(200, rate=0.06, seed=seed, cv=2.0))
    rep = Replica(priority=(mode != "fcfs"))
    arrivals, nb = d.first(), 0
    if mode != "none":                               # whole backlog queued at t=0, or `cap` in flight
        for _ in range(cap or 20000): arrivals.append((0.0, batch_req(nb, rng))); nb += 1
    def on_done(r, t):
        nonlocal nb
        if r.prio == 1:
            if cap: nb += 1; return [(t, batch_req(nb, rng))]   # feeder keeps `cap` jobs in flight
            return []
        return d.on_done(r, t)
    done = run([rep], lambda r, reps: 0, arrivals, on_done, horizon=HORIZON)
    inter = [r for r in done if r.prio == 0]
    stuck = sum(r.prio == 0 for r in rep.waiting + rep.running)    # arrived, never finished
    batch_tok = sum(r.n_out for r in done if r.prio == 1)
    name = mode + (f" cap={cap}" if cap else "")
    if not inter: return f"{name:18s} interactive starved: 0 served, {stuck} unfinished"
    ttft = [r.t_first - r.t_arr for r in inter]
    return (f"{name:18s} interactive served {len(inter):5d} unfinished {stuck:3d}  TTFT p50 {pct(ttft, 50):5.2f}s "
            f"p99 {pct(ttft, 99):6.2f}s   tok/s interactive {sum(r.n_out for r in inter)/HORIZON:4.0f} "
            f"batch {batch_tok/HORIZON:5.0f}   interactive prefix hit "
            f"{sum(r.cached for r in inter)/sum(r.n_prompt for r in inter):.0%}")

for args in (("none",), ("fcfs",), ("priority",), ("priority", 16), ("priority", 48)):
    print(experiment(*args))
```

Output of `python3 backfill.py`:

```text
none               interactive served  1027 unfinished   0  TTFT p50  0.03s p99   0.29s   tok/s interactive  215 batch     0   interactive prefix hit 96%
fcfs               interactive starved: 0 served, 61 unfinished
priority           interactive served   856 unfinished  14  TTFT p50  1.50s p99   5.97s   tok/s interactive  179 batch  1928   interactive prefix hit 32%
priority cap=16    interactive served  1002 unfinished   6  TTFT p50  0.04s p99   0.36s   tok/s interactive  209 batch  1361   interactive prefix hit 95%
priority cap=48    interactive served   967 unfinished   5  TTFT p50  0.05s p99   0.45s   tok/s interactive  202 batch  2324   interactive prefix hit 95%
```

- **FCFS starves interactive traffic completely.** Every interactive request queues behind the backlog.
- **Priority ordering alone is not enough.** Interactive requests go first in the queue, but the backlog fills the KV cache. The interactive sessions' cached history, which sits idle in the cache during tool waits, gets evicted. The interactive prefix hit rate falls from 96% to 32%, TTFT p99 rises from 0.29 s to 6 s, and the extra prefill makes the replica slower for everyone.
- **Priority plus a cap on batch requests in flight keeps the cache.** With 48 batch requests in flight, interactive TTFT p99 is 0.45 s and the hit rate is 95%, while the replica produces 2,324 tok/s of batch output on top of 202 tok/s of interactive output. That is about 12x the useful output of the interactive-only run, from the same lease. The uncapped version did less batch work (1,928 tok/s) because of the extra recomputation.

The simulator does not preempt running requests. vLLM's priority policy does preempt: when it runs out of KV blocks, the V1 scheduler evicts "the lowest-priority request" (`vllm/v1/core/sched/scheduler.py`). That would shorten the uncapped tail. But preemption does not protect the cached prefixes of sessions in a tool wait, because those have no running request to protect. The in-flight cap is what protects them. The knobs:

```bash
# untested. vLLM: lower "priority" value = earlier; default 0
vllm serve $MODEL --scheduling-policy priority
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "m", "messages": [{"role": "user", "content": "..."}], "priority": 10}'   # batch job
# untested. SGLang: by default HIGHER values run first; flip it to match vLLM's convention
python -m sglang.launch_server --model-path $MODEL --enable-priority-scheduling \
  --schedule-low-priority-values-first --priority-scheduling-preemption-threshold 10
```

The `priority` request field and its "lower means earlier handling" semantics were read from vLLM's OpenAI request schema on GitHub. The two engines use opposite conventions by default, so set the SGLang flag or translate in the gateway. The in-flight cap lives in the gateway: a semaphore per replica for batch traffic, with its size raised while TTFT p99 stays under target and cut when it rises. That is the same [bounded-queue and admission](../platform/control-planes.md#bounded-queues-load-shedding-and-per-tenant-limits) pattern a later chapter builds in general.

**On a leased fleet.** This is the main lever unique to a fixed fleet. Serverless vendors fill their idle capacity by selling it to other customers. You fill yours with your own backlog.



## Long-context serving

Contexts of 128k to 1M tokens change the economics because KV grows linearly with context and prefill attention grows quadratically. `kv_long.py` uses the [formulas from Internals I](arithmetic.md#formulas) and the same model configs:

```python
# kv_long.py - what 128k-1M-token contexts cost in KV memory, prefill compute and reload time
GiB = 2**30
MODELS = {  # L layers, n_kv heads, d head dim, N params, d_model; same configs as Internals I kvcalc.py
    "Llama-3-8B":   dict(L=32, n_kv=8, d=128, N=8.03e9, dm=4096),
    "Llama-3-70B":  dict(L=80, n_kv=8, d=128, N=70.55e9, dm=8192),
    "Llama-3-405B": dict(L=126, n_kv=8, d=128, N=405.9e9, dm=16384),
}
def kv_tok(m, b=2): return 2 * m["L"] * m["n_kv"] * m["d"] * b       # bytes per token, K and V

print("1) KV per sequence (bf16 / fp8 KV)")
for name, m in MODELS.items():
    cells = "  ".join(f"{T//1024:>5}k {kv_tok(m)*T/GiB:6.1f}/{kv_tok(m,1)*T/GiB:5.1f} GiB"
                      for T in (131072, 524288, 1048576))
    print(f"  {name:12s} {kv_tok(m)//1024:4d} KiB/tok  {cells}")

print("2) Displacement: one long request vs a batch of 8k-token requests (70B, bf16 KV)")
m = MODELS["Llama-3-70B"]
for T in (131072, 1048576):
    print(f"  one {T//1024}k request holds the KV of {T // 8192} requests at 8k context, "
          f"for its whole lifetime")

print("3) Prefill FLOPs: linear part 2*N*T vs causal attention 2*L*T^2*d_model")
for name in ("Llama-3-70B", "Llama-3-405B"):
    m = MODELS[name]
    for T in (131072, 1048576):
        lin, att = 2 * m["N"] * T, 2 * m["L"] * T**2 * m["dm"]
        print(f"  {name:12s} T={T//1024:>5}k  linear {lin:.2e}  attention {att:.2e}  "
              f"attention share {att/(lin+att):.0%}")
m = MODELS["Llama-3-405B"]; T = 1048576
fl = 2 * m["N"] * T + 2 * m["L"] * T**2 * m["dm"]
print(f"  405B @1M: {fl:.2e} FLOPs in 77 s on 128 H100 = {fl/77/128/1e12:.0f} TFLOP/s per GPU "
      f"= {fl/77/128/989e12:.0%} of 989 TFLOP/s dense bf16")
fl128 = 2 * m["N"] * 131072 + 2 * m["L"] * 131072**2 * m["dm"]
print(f"  1M/128k FLOP ratio {fl/fl128:.1f}x; reported time ratio 77/3.8 = {77/3.8:.1f}x")

print("4) Reload a whole context's KV from another tier vs recompute it (recompute at 50% MFU)")
LINKS = {"PCIe5 x16": 50e9, "100Gb/s": 12.5e9, "32Gb/s": 4e9}   # bytes/s; a 400 Gb/s NIC = PCIe5 here
for name in ("Llama-3-8B", "Llama-3-70B"):
    m = MODELS[name]; gpus = 1 if name == "Llama-3-8B" else 8       # 70B on a TP8 node
    for T in (8192, 131072):
        comp = (2 * m["N"] * T + 2 * m["L"] * T**2 * m["dm"]) / (gpus * 989e12 * 0.5)
        loads = "  ".join(f"{k} {kv_tok(m)*T/bw*1e3:6.0f} ms" for k, bw in LINKS.items())
        print(f"  {name:11s} T={T//1024:>4}k  recompute {comp*1e3:6.0f} ms   load: {loads}")
```

Output of `python3 kv_long.py`:

```text
1) KV per sequence (bf16 / fp8 KV)
  Llama-3-8B    128 KiB/tok    128k   16.0/  8.0 GiB    512k   64.0/ 32.0 GiB   1024k  128.0/ 64.0 GiB
  Llama-3-70B   320 KiB/tok    128k   40.0/ 20.0 GiB    512k  160.0/ 80.0 GiB   1024k  320.0/160.0 GiB
  Llama-3-405B  504 KiB/tok    128k   63.0/ 31.5 GiB    512k  252.0/126.0 GiB   1024k  504.0/252.0 GiB
2) Displacement: one long request vs a batch of 8k-token requests (70B, bf16 KV)
  one 128k request holds the KV of 16 requests at 8k context, for its whole lifetime
  one 1024k request holds the KV of 128 requests at 8k context, for its whole lifetime
3) Prefill FLOPs: linear part 2*N*T vs causal attention 2*L*T^2*d_model
  Llama-3-70B  T=  128k  linear 1.85e+16  attention 2.25e+16  attention share 55%
  Llama-3-70B  T= 1024k  linear 1.48e+17  attention 1.44e+18  attention share 91%
  Llama-3-405B T=  128k  linear 1.06e+17  attention 7.09e+16  attention share 40%
  Llama-3-405B T= 1024k  linear 8.51e+17  attention 4.54e+18  attention share 84%
  405B @1M: 5.39e+18 FLOPs in 77 s on 128 H100 = 547 TFLOP/s per GPU = 55% of 989 TFLOP/s dense bf16
  1M/128k FLOP ratio 30.4x; reported time ratio 77/3.8 = 20.3x
4) Reload a whole context's KV from another tier vs recompute it (recompute at 50% MFU)
  Llama-3-8B  T=   8k  recompute    302 ms   load: PCIe5 x16     21 ms  100Gb/s     86 ms  32Gb/s    268 ms
  Llama-3-8B  T= 128k  recompute  13364 ms   load: PCIe5 x16    344 ms  100Gb/s   1374 ms  32Gb/s   4295 ms
  Llama-3-70B T=   8k  recompute    314 ms   load: PCIe5 x16     54 ms  100Gb/s    215 ms  32Gb/s    671 ms
  Llama-3-70B T= 128k  recompute  10367 ms   load: PCIe5 x16    859 ms  100Gb/s   3436 ms  32Gb/s  10737 ms
```

What it says, section by section:

1. **One 1M-token Llama-3-70B sequence needs 320 GiB of bf16 KV**, more than four 80 GB GPUs can hold after the weights. FP8 KV halves that. Long-context serving needs KV quantization, offloading, or a model whose architecture caches less.
2. **A long request displaces many short ones for its whole lifetime.** One 128k request holds the KV of 16 requests at 8k. If you sell long context at the same per-token price as short context, the long requests are subsidised by the short ones. Track KV-token-seconds per tenant to see the real cost.
3. **At 1M tokens, attention is 84-91% of prefill FLOPs.** The FLOP ratio between 1M and 128k is 30x for 405B. Meta measured 77 s against 3.8 s, a 20x ratio, which suggests the 128k run used the 128 GPUs less efficiently. My count implies 55% of the H100's dense bf16 peak for the 1M run, while Meta reports "63% FLOPS utilization". The gap is probably a different FLOP-counting convention or peak value. I could not reconcile it exactly.
4. **Reloading KV from host memory is 6-39x faster than recomputing it**, and still faster over a 100 Gb/s network. At 32 Gb/s the two are about equal for a 128k-token 70B context. That matches LMCache's measurement that at 32 Gbps, loading KV "outperforms naive prefilling only when the input context length exceeds 256K tokens" ([source](https://arxiv.org/html/2510.09665)). The 50 GB/s for PCIe Gen5 x16 is an assumed achievable rate, not a measurement.

### Chunked prefill for long prompts

A 128k-token prompt prefilled in one step stalls every decode on the replica for seconds. Chunked prefill splits it into budget-sized pieces interleaved with decodes ([Internals II](serving-engine-internals.md#chunked-prefill-and-stall-free-batching)). Sarathi-Serve, the paper that introduced stall-free batching, measured serving capacity under latency SLOs up to 2.6x higher for Mistral-7B on one A100, 3.7x for Yi-34B on two A100s and 5.6x for Falcon-180B with pipeline parallelism, against vLLM at the time ([source](https://arxiv.org/abs/2403.02310)). In vLLM V1, "chunked prefill is enabled by default whenever possible" ([source](https://docs.vllm.ai/en/latest/configuration/optimization.html)). What is left to tune is the budget and `--long-prefill-token-threshold`, which caps the share of a step one long prompt can take, so that several long prompts make progress together and short prompts are not stuck behind them.

### KV offloading to CPU or disk

Host DRAM is several times larger than HBM, and the table above shows that reloading from it beats recomputing. vLLM can offload natively or through LMCache:

```bash
# untested. Native CPU offload: 200 GiB of host RAM for KV, summed over TP ranks
vllm serve $MODEL --kv-offloading-size 200 --kv-offloading-backend native
# the same through LMCache
vllm serve $MODEL --kv-offloading-size 200 --kv-offloading-backend lmcache
# or the connector form, with explicit block size and bytes
vllm serve $MODEL --kv-transfer-config \
  '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":200000000000}}'
# SGLang HiCache: host KV pool twice the size of the GPU pool, or --hicache-size in GB
python -m sglang.launch_server --model-path $MODEL --enable-hierarchical-cache --hicache-ratio 2
```

vLLM's [KV connector list](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) also includes `LMCacheConnectorV1`, `MooncakeConnector`, `NixlConnector` and `MultiConnector` (which chains several), for offloading to remote tiers and for disaggregation.

What the published systems report:

- **LMCache** moves KV in chunks of "a default size of 256 tokens per chunk", because vLLM's pages are too small to saturate a link. The authors report "up to 15× higher throughput" on their workloads ([source](https://arxiv.org/html/2510.09665)).
- **Mooncake** (Moonshot AI's serving system for Kimi) builds a disaggregated KV store that "leverages the underutilized CPU, DRAM, and SSD resources" of the GPU cluster. On a trace with an average input of 8,590 tokens and an average output of 182, they found that "up to only 55% of the KVCache can be reused" even with infinite storage. Their production SLOs were a 30 s TTFT and a time between tokens "capped at 0.1 seconds per token", and Mooncake "enables Kimi to handle 75% more requests" under them ([source](https://arxiv.org/html/2407.00079)).
- **Prime Intellect** puts Mooncake behind vLLM as "a second cache tier in host DRAM" for GLM-5.3 agent traffic, where each turn "adds about 6K tokens to a 140K-token prompt" ([source](https://www.primeintellect.ai/blog/prime-inference)), and offloads to "both CPU and disk" in its RL rollouts ([source](https://www.primeintellect.ai/blog/rl-at-1t-scale)).

**On a leased fleet.** Native CPU offload uses host RAM you are already paying for, and which mostly sits idle while the GPUs serve (check `free -g` on your nodes). That makes it the first long-context lever to turn on. Mooncake's pooled store pays off when sessions move between replicas (after rebalancing or with disaggregation) and needs RDMA between nodes. On 2-4 nodes, per-node offload plus session-affine routing gets most of the benefit.

### Context-parallel prefill

**Context parallelism** (CP) splits one long sequence across GPUs along the sequence dimension and passes KV (or queries) around a ring, so prefill of a 1M-token prompt uses many nodes at once. Meta's measurements on Llama-3 405B: "1M context prefill ... in 77s (93% parallelization efficiency, 63% FLOPS utilization)" on up to 128 H100s across 16 nodes, and "128K context prefill in 3.8s" ([source](https://arxiv.org/html/2411.01783v3)). TTFT at 128k by layout:

| Layout | CP1+TP8 | CP2+TP8 | TP16 | TP32 | CP4+TP8 |
|-|-:|-:|-:|-:|-:|
| 128k TTFT (ms) | 42,010 | 21,042 | 29,917 | 19,841 | 10,950 |

CP across nodes beats TP across nodes for prefill, because CP sends KV, which is small under GQA, while TP all-reduces activations every layer. **Decode gets slower with CP.** At 128k and batch 1, time per output token was 46.26 ms on TP8, 60.23 ms on CP2+TP8 and 71.31 ms on CP4+TP8 in the paper's Table 7 (its Table 6 gives 66.63 ms for CP2+TP8). CP is a prefill tool, and it argues for disaggregation: CP on the prefill pool, plain TP on decode.

vLLM has `--prefill-context-parallel-size` (`-pcp`) and `--decode-context-parallel-size` (`-dcp`), both defaulting to 1 (untested here; check which attention backends support them in your version).

**On a leased fleet.** CP matters only if you serve single prompts long enough that one node's prefill takes longer than your TTFT target. A 128k prompt to a 405B model takes 42 s on one TP8 node by Meta's numbers. For most startups the cheaper path is to cache and reuse long contexts (offload plus routing), which avoids the prefill altogether.

### Hybrid and sliding-window models

The cheapest long-context KV is the KV a model never stores. Models that mix a few full-attention layers with many sliding-window or linear-attention (Mamba-style) layers cache far less. Gemma-3-27B has 10 global and 52 local layers with a 1,024-token window, so its KV at 128k is 6x smaller than it would be with all layers global ([Internals I](arithmetic.md#sliding-window)). vLLM's hybrid KV cache manager stores both kinds in one paged pool and frees "blocks that are outside the sliding window". For Mamba hybrids, the per-sequence state is much larger than an attention page, so vLLM raises the attention block size to match, which "can lead to more than 400 `block_size` for attention layers, which is too large" ([source](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html)). Prefix caching is more restricted on these models than on dense ones, so check how your engine version caches them before counting on shared-prefix hit rates. When choosing a model for a long-context product on a fixed fleet, KV bytes per token at your context length belongs next to quality and price in the comparison.



## Multimodal serving

Images and video add a third stage before prefill. A **vision encoder** (a ViT) turns pixels into embeddings, which then enter the LLM as ordinary prompt tokens. Three things change: how many tokens an image becomes, the encoder's own compute, and where that compute runs.

### Image token counts and encoder cost

Qwen2.5-VL is a representative example. Its ViT cuts images into 14-pixel patches, and a merger groups "spatially adjacent sets of four patch features" into one LLM token, so each LLM token covers 28x28 pixels (derived from the 14-pixel stride and the 2x2 merge). The ViT uses windowed attention, and "only four layers employ full self-attention" ([source](https://arxiv.org/html/2502.13923)). Images are resized at *dynamic resolution*: their sides are rounded to multiples of 28, then the image is scaled down if it exceeds a pixel budget. The model card gives the default range as "4-16384" visual tokens per image and suggests `min_pixels = 256*28*28` and `max_pixels = 1280*28*28` to keep each image between 256 and 1,280 tokens ([source](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)).

`mm_tokens.py` reproduces the resize rule from `qwen-vl-utils` 0.0.10 and estimates the cost of one image with the vision and LLM dimensions from the model's `config.json`:

```python
# mm_tokens.py - what one image costs Qwen2.5-VL-7B: tokens, encoder FLOPs, LLM prefill FLOPs, KV
import math
# smart_resize as in qwen-vl-utils 0.0.10: sides become multiples of 28 (14-px patches, 2x2 merge)
def smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb

# vision tower from config.json: depth 32, hidden 1280, MLP 3420 (gated), 4 full-attention blocks
V_L, V_H, V_I, V_FULL, WIN = 32, 1280, 3420, 4, 64       # window 112 px = 8x8 = 64 patches
v_params = V_L * (4 * V_H * V_H + 3 * V_H * V_I)
# LLM from config.json: 28 layers, hidden 3584, 4 KV heads x 128, MLP 18944, vocab 152064
L, H, KVH, I, VOCAB = 28, 3584, 4, 18944, 152064
llm_params = L * (2 * H * H + 2 * H * KVH * 128 + 3 * H * I) + H * VOCAB   # + lm_head
kv_tok = 2 * L * KVH * 128 * 2                                             # bf16 bytes per token

def cost(h, w, max_pixels=16384 * 28 * 28):
    hb, wb = smart_resize(h, w, max_pixels=max_pixels)
    tokens, patches = hb * wb // 28**2, hb * wb // 14**2
    enc = 2 * v_params * patches                                   # linear layers
    enc += (V_L - V_FULL) * 4 * patches * WIN * V_H + V_FULL * 4 * patches**2 * V_H   # attention
    llm = 2 * llm_params * tokens + 2 * L * tokens**2 * H          # prefill, causal attention
    return hb, wb, tokens, enc, llm

print(f"vision tower {v_params/1e6:.0f}M params, LLM {llm_params/1e9:.2f}B params, KV {kv_tok//1024} KiB/token")
print(f"{'image':>11s} {'max_pixels':>10s} {'resized':>11s} {'tokens':>6s} {'enc GFLOP':>9s} "
      f"{'LLM GFLOP':>9s} {'enc/LLM':>7s} {'KV MiB':>6s}")
for (h, w) in ((512, 512), (1080, 1920), (2160, 3840)):
    for mp in (16384 * 784, 1280 * 784):
        hb, wb, t, e, l = cost(h, w, mp)
        print(f"{w:>5d}x{h:<5d} {mp//784:>7d}*784 {wb:>5d}x{hb:<5d} {t:6d} {e/1e9:9.0f} {l/1e9:9.0f} "
              f"{e/l:7.2f} {t*kv_tok/2**20:6.0f}")
```

Output of `python3 mm_tokens.py`:

```text
vision tower 630M params, LLM 7.07B params, KV 56 KiB/token
      image max_pixels     resized tokens enc GFLOP LLM GFLOP enc/LLM KV MiB
  512x512     16384*784   504x504      324      1679      4603    0.36     18
  512x512      1280*784   504x504      324      1679      4603    0.36     18
 1920x1080    16384*784  1932x1092    2691     16034     39506    0.41    147
 1920x1080     1280*784  1316x728     1222      6693     17579    0.38     67
 3840x2160    16384*784  3836x2156   10549     90016    171504    0.52    577
 3840x2160     1280*784  1316x728     1222      6693     17579    0.38     67
```

- **One 4K screenshot is 10,549 tokens at the default budget**, about as much prefill as a 10k-token document, plus 577 MiB of KV. With `max_pixels` at 1280*784, the same image is 1,222 tokens, 8.6x fewer. **Setting `max_pixels` on the server is the biggest multimodal cost control there is.** Clients will send whatever resolution they have.
- **The encoder adds 36-52% to the LLM's prefill FLOPs.** The share grows with resolution because the four full-attention ViT layers are quadratic in patch count (42k patches for the 4K image).
- **Count tokens, not images,** in rate limits and capacity plans. One image can cost anywhere from 4 to 16,384 tokens.

### Encoder caching and preprocessing

The same image is often sent many times: every turn of a conversation about a screenshot resends it. Two caches avoid redoing the work:

- vLLM's **processor cache** keeps preprocessed inputs (decoded, resized, patched). `--mm-processor-cache-gb` defaults to 4 GiB, and a copy lives in each API-server and engine-core process. The **encoder cache** keeps vision embeddings per item, keyed by content hash, so chunked prefill over a long image does not rerun the encoder.
- Since the image tokens are part of the prompt, the **prefix cache** covers repeated images too. The multimodal input hash is part of each block's hash ([vLLM prefix-caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html)). Sessions about the same image hit the cache if the router keeps them together, so session affinity matters even more for multimodal agents.

Image decode and resize run on the CPU and can become the bottleneck before the GPU does. Together moved all of it out of the GPU workers: their gateway (SMG) "performs all vision preprocessing on the CPU", including "downloading, decoding, frame sampling, resizing, and converting into patch tensors", and ships "a flat patch tensor and a small grid metadata tensor, packed into a gRPC message", so "The worker just runs the vision encoder directly" ([source](https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets)).

```bash
# untested. Cap images per prompt and their resolution, shard the encoder by data instead of TP
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 1003520}' \
  --mm-processor-cache-gb 8
vllm serve $LARGE_VLM -tp 8 --mm-encoder-tp-mode data
```

`--mm-encoder-tp-mode data` runs the vision encoder data-parallel across the TP ranks, with replicated encoder weights, instead of tensor-parallel. The vLLM docs report "around 10%" better throughput and TTFT at TP=8, and "another 40% improvement compared to regular TP" for encoders built on Conv3D ([source](https://docs.vllm.ai/en/latest/configuration/optimization.html)). Which processor kwargs a model accepts (here `max_pixels`) depends on its Hugging Face processor.

### Encode, prefill, decode disaggregation

Encoding can be split out like prefill: separate encoder workers compute embeddings and ship them to the LLM workers. The EPD paper reports, against non-disaggregated serving, up to 15x lower peak memory, up to 22x larger batches, 10x more images per request, 2.2x larger KV caches and up to 71% lower TTFT ([source](https://arxiv.org/abs/2501.05460)). vLLM has `--ec-transfer-config` for moving encoder outputs between instances, and its connector list includes an `ECMooncakeConnector` for it ([source](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)). The same fixed-fleet caution as for P/D applies: it adds a pool that can sit idle, so it pays at scale or when encoding is so heavy (video, many high-resolution images) that it disrupts decode.

### How capacity planning differs

Text capacity planning ([hands-on](serving-hands-on.md#capacity-planning-from-measured-numbers)) assumes the cost of a request is predictable from its token counts. With images, three things break that:

- **The token count is decided by the server's resize policy, not the client.** Fix `max_pixels` before measuring anything, or the measurements will not transfer.
- **Throughput can fall as concurrency rises.** Together's dedicated-endpoint benchmark showed Qwen3.5-9B (text) scaling from 362 to 789 to 1,464 tok/s at 4, 8 and 16 concurrent requests, while Qwen3-VL-8B went from 495 to 243 to 240 tok/s ([source](https://www.together.ai/blog/configuring-dedicated-model-inference)). Their lesson: rank configurations at your concurrency, not at concurrency 1. For a VLM, sweep concurrency on a replay that includes real images.
- **CPU per request is no longer negligible.** Size API-server CPU (or move preprocessing to the gateway, as Together does) from a measured per-image preprocessing time.



## Serving for RL rollouts

RL training for LLMs needs an inference fleet to generate rollouts, and that fleet behaves differently from product traffic. Requests arrive in large bursts (a whole batch of prompts at once), many requests share a prompt (a group of samples per prompt), outputs have a long tail, and the weights change every step or every few steps. The [RL infrastructure chapter](../platform/rl-infrastructure.md) (later in the book) covers [rollout workers](../platform/rl-infrastructure.md#rollout-workers), [the long tail](../platform/rl-infrastructure.md#continuous-batching-and-the-long-tail) and [weight synchronization](../platform/rl-infrastructure.md#weight-synchronization) in detail. Three points from this chapter carry over:

- **Independent replicas plus a simple router** is what Prime Intellect runs for rollouts, and it is where the linear-scaling quote above comes from ([INTELLECT-3](https://arxiv.org/html/2512.16144v1)). The same report finds that "Without in-flight weight updating, we observe an increase in step time of more than 2×".
- **Agentic rollouts are KV-bound** like agentic product traffic: Prime Intellect offloads rollout KV to CPU and disk to raise concurrency ([source](https://www.primeintellect.ai/blog/rl-at-1t-scale)).
- **Speculation can be trained on the rollouts themselves.** Together's ATLAS acceptance rose from below 10% to above 80% during an RL run ([source](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)).

On a fixed fleet, rollouts are also the natural [backfill](#priority-filling-idle-capacity-with-batch-work): a research team's RL job can soak up the idle hours of the product fleet, at low priority and with an in-flight cap.



## Tuning checklist for a leased fleet

In order. Each step assumes the previous ones are done, and each gain was measured on someone else's workload. Remeasure each step on your replayed trace, on three seeds.

| # | Step | Knobs | Expected gain | Source |
|-:|-|-|-|-|
| 1 | Log traffic with block hashes; build a session replay; never benchmark on a mean trace | gateway logging, `vllm bench serve --dataset-name timed_trace`, the [trie harness](https://github.com/Applied-Compute/trie) | avoids a 10-20% throughput overestimate (36-82% in this chapter's sim once KV fills) | [Applied Compute](https://www.appliedcompute.com/research/inference-benchmark) |
| 2 | Pick the smallest TP that leaves most of HBM for KV and meets per-user speed; fill the fleet with independent replicas | `-tp`, one process per replica | linear scaling with replicas vs a plateau for multi-node DP; up to "4x cheaper per token" vs a latency layout | [INTELLECT-3](https://arxiv.org/html/2512.16144v1), [Fireworks](https://fireworks.ai/blog/fireoptimizer) |
| 3 | Put a prefix-aware router with bounded load in front; add session ids | `sglang_router --policy cache_aware`, `x-session-affinity`-style headers | 1.97x throughput (precise vs random); +61% RPS, TTFT -50% | [llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see), [Baseten](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/) |
| 4 | FP8 weights (calibrated) and FP8 KV; NVFP4 KV on Blackwell | `--kv-cache-dtype fp8`, `--quantization fp8` | 2.3x KV tokens for 70B on 8 H100s (bound); 1.09M to 1.63M tokens with NVFP4 KV | [Internals I](arithmetic.md#max-concurrent-sequences), [Prime Intellect](https://www.primeintellect.ai/blog/prime-inference) |
| 5 | Cap session concurrency per replica at the measured knee; queue the rest at the gateway | gateway semaphore, `--max-num-seqs` | avoids the throughput fall past the KV knee | this chapter's `mean_vs_full.py`; [Applied Compute](https://www.appliedcompute.com/research/inference-benchmark) |
| 6 | Sweep the token budget and the long-prompt cap | `--max-num-batched-tokens`, `--long-prefill-token-threshold` | median queue wait 550 to 110 ms, TTFT -20% (8K to 4K per GPU) | [Prime Intellect](https://www.primeintellect.ai/blog/prime-inference) |
| 7 | Turn on CPU KV offload for agentic and long-context traffic | `--kv-offloading-size`, `--enable-hierarchical-cache` | up to 15x throughput (authors' workloads); host RAM is already paid for | [LMCache](https://arxiv.org/html/2510.09665) |
| 8 | Speculative decoding with a head trained on your traffic; off at high batch | `--speculative-config` with `num_speculative_tokens_per_batch_size` | 2x at 76% acceptance; a 29% generic draft was 1.5x slower | [Fireworks](https://fireworks.ai/blog/fireoptimizer) |
| 9 | Keep compile caches and capture only the batch sizes you serve | `-cc '{"cudagraph_capture_sizes": [...]}'`, cache `~/.cache/vllm` | compile "tens of seconds to a few minutes" down to "about ten seconds" from cache | [Modal](https://modal.com/docs/examples/vllm_inference) |
| 10 | Serve fine-tunes as LoRAs on shared bases; sleep long-tail models | `--enable-lora --max-loras`, `--enable-sleep-mode` | multi-LoRA near "90% of base model performance" | [Fireworks](https://fireworks.ai/blog/multi-lora) |
| 11 | Backfill troughs with batch work at low priority and capped in flight | `--scheduling-policy priority`, gateway cap | interactive p99 kept (0.29 to 0.45 s) while adding about 12x the useful output in the sim | this chapter's `backfill.py` |
| 12 | Multimodal: cap `max_pixels` and images per prompt; data-parallel encoder | `--mm-processor-kwargs`, `--limit-mm-per-prompt`, `--mm-encoder-tp-mode data` | 8.6x fewer tokens for a 4K image; about 10% at TP=8 | `mm_tokens.py`, [vLLM docs](https://docs.vllm.ai/en/latest/configuration/optimization.html) |
| 13 | Disaggregate prefill/decode only at scale, with RDMA, when ITL tail binds | `--kv-transfer-config` (NIXL), Dynamo or llm-d | does not raise throughput; p90 ITL nearly -40% with Dynamo over a 1:4 split; CPD +40% QPS/GPU | [vLLM docs](https://docs.vllm.ai/en/latest/features/disagg_prefill.html), [Prime Intellect](https://www.primeintellect.ai/blog/prime-inference), [Together](https://www.together.ai/blog/cache-aware-disaggregated-inference) |
| 14 | Re-measure weekly; re-tune when prompt lengths or hit rates drift | `trace_stats.py` on the latest logs | prompt length +80% two weeks after a release | [Together](https://together.ai/blog/together-compiler-continual-compilation-for-ai-inference) |



## You can now read

- The vLLM [optimization and tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization.html) and [engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html), knowing which of the knobs above each one moves, and how to measure it.
- Vendor engineering blogs (Baseten, Fireworks, Together, Modal) critically: what the baseline was, which workload property the gain depends on, and whether the technique needs elastic capacity or works on a fixed fleet.
- The [llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see) and [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) router designs, as production versions of the 30-line router in `router_sim.py`.
- Applied Compute's [engine benchmark](https://www.appliedcompute.com/research/inference-benchmark) and its harness, as a template for benchmarking your own agent traffic.
- [Infrastructure for Large-Scale RL](../platform/rl-infrastructure.md), where the same fleet becomes a rollout generator.



## Further reading

- Applied Compute, "Benchmarking Inference Engines on Agentic Workloads", April 2026: https://www.appliedcompute.com/research/inference-benchmark
- Prime Intellect, "Prime Inference: Fast, Reliable Serving for Frontier Open Models", October 2026: https://www.primeintellect.ai/blog/prime-inference
- Prime Intellect, INTELLECT-3 technical report, arXiv 2512.16144: https://arxiv.org/html/2512.16144v1
- llm-d, "KV-Cache Wins You Can See", September 2025: https://llm-d.ai/blog/kvcache-wins-you-can-see
- Baseten, "How Baseten achieved 2x faster inference with NVIDIA Dynamo", October 2025; "How we built production-ready speculative decoding with TensorRT-LLM", December 2024: https://www.baseten.co/blog/
- Fireworks, "FireOptimizer" and "Multi-LoRA", 2024: https://fireworks.ai/blog/fireoptimizer, https://fireworks.ai/blog/multi-lora
- Together, "Cache-aware prefill-decode disaggregation", March 2026; "Configuring Dedicated Model Inference", July 2026; ATLAS, October 2025: https://www.together.ai/blog
- Modal, "How to serve trillions of tokens for trillion-parameter coding agents", September 2026: https://modal.com/blog/trillion-tokens-trillion-parameters
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", arXiv 2403.02310
- Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", arXiv 2407.00079
- "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference", arXiv 2510.09665
- Meta, "Context Parallelism for Scalable Million-Token Inference", arXiv 2411.01783
- "Efficiently Serving Large Multimodal Models Using EPD Disaggregation", arXiv 2501.05460
- Qwen Team, "Qwen2.5-VL Technical Report", arXiv 2502.13923
- Li et al., "EAGLE-3", arXiv 2503.01840
- vLLM design docs on prefix caching and the hybrid KV cache manager: https://docs.vllm.ai/en/latest/design/
- In this repo: [Inference Internals I](arithmetic.md), [Inference Internals II](serving-engine-internals.md), [Serving LLMs in Practice](serving-hands-on.md), [Control Planes](../platform/control-planes.md), [RL Infrastructure](../platform/rl-infrastructure.md).
