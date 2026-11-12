# Model Deploys, Recovery and Autoscaling

Your company leases a fixed GPU fleet: a few hundred GPUs on a one- or three-year contract, already racked, already paid for. Nobody on that fleet ever waits for a cloud VM to boot before their chat request is answered. So in the usual sense of "cold start" (a user's request lands on a function scaled to zero and waits for a container to boot) there is no cold-start problem.

Every minute a replica spends starting still costs you, though. It shows up in four places:

- **Deploy time.** You ship a new model version, a new quantization or an engine upgrade, and it has to roll across N replicas. If one replica takes 8 minutes to become ready and you can only take one down at a time, a 16-replica deploy takes over two hours. A rollback takes as long again.
- **Recovery time.** A GPU throws an uncorrectable ECC error at 3 a.m. The replica on that node is gone, and its capacity stays gone until a replacement replica on some other node is serving. That gap is mostly startup time.
- **Model-swap time.** You serve more models than you have GPUs for at once: a chat model that peaks mid-day, a code model that peaks in the evening, a dozen fine-tunes with occasional traffic. Moving a node from one model to another costs one startup each time, and if that is slow you stop moving nodes and leave GPUs idle instead.
- **Burst capacity.** When demand exceeds the fleet you rent extra GPUs on a cloud or a serverless platform, and there the cold start is the classic one, paid by the user who triggered it.

This chapter teaches what a replica start consists of and how to make each stage shorter, then the mechanics of deploys and recovery that turn startup time into risk or downtime, then autoscaling. On a fixed fleet, autoscaling means moving GPUs between models and filling idle ones with batch work, not buying and releasing machines. Serverless scale-to-zero comes last, as the tool for bursts and for small teams. The chapter ends with a decision table for what infrastructure to run all this on, and an ordered startup-time checklist.

The vendor posts quoted here (Modal, Baseten, Fireworks, Together, fal, Replicate) describe multi-tenant clouds that buy capacity by the minute. Some of their techniques carry over to a leased fleet unchanged; some solve a problem you do not have; some have a cheaper equivalent. After each one there is a short **On a leased fleet** note saying which.

Throughout, $T_{\text{start}}$ means the time from "a replica was asked for" to "the replica is in the router and serving at full speed". Most of the arithmetic in this chapter is about how $T_{\text{start}}$ turns into GPU-hours, minutes of reduced capacity, or user-visible queueing.

**What was run.** Every Python block marked with a file name was run on an Apple M4 laptop (10 cores, macOS, CPython 3.14.5). The pure-Python scripts (`loadmath.py`, `rollout.py`, `drain.py`, `recovery.py`, `autoscale_sim.py`, `realloc.py`, `burst.py`) need only `python3`. The two model scripts (`startup_stages.py`, `probe.py`) ran in a virtualenv with PyTorch 2.15 (CPU), `transformers` 5.22.2, `safetensors` 0.8.0 and `huggingface_hub` 2.4.0, using `Qwen/Qwen2.5-0.5B-Instruct` from the local Hugging Face cache. The runs used an existing virtualenv with those versions; to reproduce it:

```bash
python3 -m venv /tmp/drvenv
/tmp/drvenv/bin/pip install torch transformers safetensors huggingface_hub
/tmp/drvenv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
```

Outputs are pasted verbatim. Simulations are seeded, so you get the same numbers. Kubernetes manifests and `vllm serve` commands are **untested** here (no GPUs, no cluster); their field and flag names were checked against the current Kubernetes, vLLM and SGLang docs in November 2026. Vendor numbers carry an inline link to the page they come from, and every one was re-read on that page; they are the vendor's own measurements, on the vendor's hardware.

## Prerequisites

This chapter builds on the following and links to them instead of repeating them:

- [Control Planes and Fleet Management](control-planes.md): [reconciliation](control-planes.md#desired-state-and-reconciliation), [placement and bin-packing](control-planes.md#3-placement-and-bin-packing), [heartbeats and leases](control-planes.md#heartbeats-and-leases), [health reporting](control-planes.md#health-reporting) and [draining](control-planes.md#draining). A deploy is a reconciler changing desired state; recovery is the reconciler reacting to a lease expiring; an autoscaler is a reconciler whose desired state comes from a metric.
- [Serving LLMs in Practice](../inference/serving-hands-on.md): [speeding up model loading](../inference/serving-hands-on.md#speeding-up-model-loading-time), [deploying on Kubernetes](../inference/serving-hands-on.md#deploying-on-kubernetes), [the gateway in front of several replicas](../inference/serving-hands-on.md#a-gateway-in-front-of-several-replicas) with its [timeouts, retries and ejection](../inference/serving-hands-on.md#timeouts-retries-and-ejection), and [common failure modes](../inference/serving-hands-on.md#common-failure-modes).
- [Orchestration internals](../orchestration/internals.md): [autoscaling on queue depth and KV-cache use](../orchestration/internals.md#autoscale-on-queue-depth-and-kv-cache-use-not-cpu), [cold-start budget arithmetic](../orchestration/internals.md#cold-start-budget-arithmetic), [model weight distribution](../orchestration/internals.md#model-weight-distribution), [failure-rate arithmetic](../orchestration/internals.md#failure-rate-arithmetic) and [hot spares](../orchestration/internals.md#hot-spares). This chapter extends those sections; it does not repeat them.
- [Kubernetes from zero](../orchestration/kubernetes-from-zero.md): [Deployments and rolling updates](../orchestration/kubernetes-from-zero.md#deployment-and-replicaset), [probes](../orchestration/kubernetes-from-zero.md#probes), [a vLLM Deployment](../orchestration/kubernetes-from-zero.md#a-vllm-deployment).
- [Containers and Virtualization](../foundations/containers-and-virtualization.md): [image layers and registries](../foundations/containers-and-virtualization.md#registries-and-pull-time), [containerd snapshotters](../foundations/containers-and-virtualization.md#containerd-shims-and-snapshotters), [VM snapshots](../foundations/containers-and-virtualization.md#snapshots-starting-from-a-warm-vm), [gVisor](../foundations/containers-and-virtualization.md#gvisor-a-kernel-in-user-space) and the [runtime comparison](../foundations/containers-and-virtualization.md#comparison).
- [The GPU node software stack](../compute/gpu-software-stack.md): [Xid errors](../compute/gpu-software-stack.md#xid-errors-reading-finding-and-looking-them-up), [ECC and row remapping](../compute/gpu-software-stack.md#ecc-errors-and-row-remapping-in-nvidia-smi--q) and [`dcgmi diag`](../compute/gpu-software-stack.md#running-diagnostics-with-dcgmi-diag).
- [Storage and Data](../foundations/storage-and-data.md): [safetensors and mmap](../foundations/storage-and-data.md#safetensors-a-header-plus-raw-bytes), [the page cache in measurements](../foundations/storage-and-data.md#the-page-cache-in-your-own-measurements), [local NVMe](../foundations/storage-and-data.md#2-local-nvme), [getting bandwidth out of an object store](../foundations/storage-and-data.md#getting-bandwidth-out-of-an-object-store).
- [Inference arithmetic](../inference/arithmetic.md): [Little's law](../inference/arithmetic.md#littles-law), used to turn request rates into replica counts.
- [Linux and the OS](../foundations/linux-and-os.md): [signals](../foundations/linux-and-os.md#signals) (SIGTERM drives draining).
- Forward link: [Operating inference](operating-inference.md) covers rate limits, tenancy, SLOs, reliability targets and cost, which this chapter leaves out.


## 1. Where startup time goes on a fixed fleet

### Vocabulary

- A **replica** is one running copy of the model that can serve requests on its own: an engine process (vLLM, SGLang, TensorRT-LLM) plus the GPUs it owns. A 70B model at tensor parallelism 8 is one replica per 8-GPU node; an 8B model might be eight replicas per node.
- **Ready** means the replica passes its readiness check and the router may send it traffic. **Warm** means it is also serving at full speed: compiled kernels loaded, CUDA graphs captured, caches allocated. A replica can be ready and not yet warm, if you let it into the router too early.
- **Cold start** is the whole path from "a replica was requested" to "ready and warm". **$T_{\text{start}}$** is its duration.
- **Rolling update**, **blue-green**, **canary**, **drain** and **surge** are defined where they are used in [section 4](#4-deploying-a-new-model-version-on-a-fixed-fleet).

### Four ways $T_{\text{start}}$ costs you

A rough picture for a 16-replica deployment with $T_{\text{start}} = 8$ min, before any of the details:

| Event | How $T_{\text{start}}$ enters | Rough cost |
|---|---|---|
| Deploy, one replica at a time | $N \times (T_{\text{drain}} + T_{\text{start}})$ | 16 × 10 min ≈ 2.7 h of rollout, during which a rollback is also slow |
| GPU dies, no warm spare | lost capacity until replacement is ready | 1 replica × (detect + 8 min) per failure |
| Swap a node to another model | node serves nobody during the swap | 8 min of one node, every move |
| Burst to the cloud | user requests queue behind the start | p95 queue wait ≈ $T_{\text{start}}$ for the burst |

A 10x shorter $T_{\text{start}}$ shrinks every row about 10x. The [autoscaling simulator](#a-discrete-time-autoscaler-simulator) at the end of section 6 shows the less obvious consequence: the headroom you have to keep idle to absorb bursts grows with $T_{\text{start}}$, so startup time ends up setting your GPU bill too.


## 2. The anatomy of a replica start

### The stages

A replica start is a chain of stages, mostly serial. Each one has its own cause and its own fix:

| # | Stage | What happens | Bounded by |
|---|---|---|---|
| 1 | Node provisioning | get a machine with free GPUs: a cloud VM boots, or the scheduler finds a free node in your fleet | cloud capacity and boot time; on a leased fleet, free nodes |
| 2 | Image pull | download and unpack the container image (CUDA, PyTorch, the engine: often 10-20 GB) | registry bandwidth, **single-threaded gzip decompression** |
| 3 | Container start | namespaces, cgroups, mounts, the GPU device nodes | small (a second or less) |
| 4 | Python and CUDA init | `import torch`, the engine's imports, `cuInit`, creating CUDA contexts, NCCL init across TP ranks | file-system metadata, CPU |
| 5 | Weight fetch | copy weights from wherever they live to local disk or host memory | storage and network bandwidth |
| 6 | Weight load | move weights into GPU memory, sharded per TP rank | disk, host memory, PCIe |
| 7 | `torch.compile` | trace the model and generate fused kernels | CPU; cacheable |
| 8 | CUDA graph capture | record the decode step once per batch size so it can be replayed with one launch | number of captured sizes; cacheable in part |
| 9 | KV allocation | profile memory, then preallocate the KV-cache pool | small, unless the engine profiles memory with a dummy run |
| 10 | Warmup | first real requests: lazy initialization, autotuning, allocator growth | the code paths your traffic hits |
| 11 | Health check | the readiness check passes | probe period |
| 12 | Join the router | the gateway or Service starts sending traffic | endpoint propagation, the router's refresh interval |

Stages 7 and 8 exist because of the [CPU overhead of decode](../inference/serving-engine-internals.md#why-cpu-overhead-matters-at-decode): an engine trades startup time for per-token speed. Stage 4 is slow because `import torch` touches thousands of files: Modal counted that ["even just importing `torch` in Python executes 26,000 syscalls"](https://modal.com/blog/mem-snapshots).

### What published breakdowns say

Five measured breakdowns, from five different setups. None is directly comparable to the others, but together they say where the time goes.

**fal, one model on an H100.** fal's [cold-start example](https://fal.ai/docs/examples/serverless/cut-cold-starts) times the stages inside its `setup()` on an H100, measured on 2026-08-12, in seconds:

| Stage | Baseline | First optimized boot | Optimized steady state |
|---|---|---|---|
| Imports | 26.26 | 54.61 | 11.0 |
| Weights | 46.09 | 214.49 | 7.66 |
| Compile + warmup | 202.9 | 198.75 | 77.79 |
| **setup() total** | **275.25** | **467.95** | **96.46** |

Compile and warmup are three quarters of the baseline boot, more than the weight download. The "first optimized boot" is *slower* than the baseline: it downloads, repacks and publishes the weights in a faster format and builds the compile cache, a one-time cost that every later boot avoids. The same page reports spawn-to-healthy times of 114.7-267.3 s for the steady-state runs, which it puts down to GPU-allocation contention on a shared account. Waiting for a GPU varied by more than the whole optimized setup.

**Microsoft, vLLM on an A100.** A [Microsoft study](https://techcommunity.microsoft.com/blog/linuxandopensourceblog/dissecting-llm-container-cold-start-where-the-time-actually-goes/4508831) ran vLLM with Llama-3.1-8B from a ~14 GB image on an A100 80 GB, with the containerd store wiped and the page cache dropped before every run. With gzip layers on overlayfs, the image pull took 140.8 s, the engine startup took 176.0 s, and time to first token was 317.2 s. Uncompressed layers cut the pull to 129.9 s. The startup part, in the post's words, "takes ~176 seconds regardless of how fast the image arrived". It covers weights, compilation, CUDA graphs, KV allocation and warmup, none of which a faster image fixes.

**Together, a dedicated endpoint on one H100.** Together's [autoscaling post](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference) measured, on warm clusters: 86 s from creating a Qwen3.5-9B catalog deployment to READY (about 30 s of it image pull, about 30 s start), 145 s for an 18 GB custom fine-tune, then another 26-40 s from READY until the first token came back through routing, and about 2.5 minutes for a scale-up from 1 to 2 replicas. That last line is the number an autoscaler lives with.

**Baseten, as a product.** Baseten's [cold-start docs](https://docs.baseten.co/deployment/autoscaling/cold-starts) split startup into four phases: acquiring resources, pulling the image, pulling weights, and loading the model (for vLLM and SGLang that includes the GPU load, CUDA graph capture, kernel compilation and KV-cache profiling). Its dashboard reports p50 to p99 per phase, and marks a startup as stalled if the replica did not become ready within 30 minutes.

**Modal, the whole stack.** Modal's [serverless GPU post](https://modal.com/blog/truly-serverless-gpus) puts a naively booted inference server at "upwards of 2 kiloseconds" and its own optimized path at about 50 s. The 2,000 s includes getting a fresh, health-checked cloud GPU, which a leased fleet already has. Over more than ten thousand cold starts of a ~1 GiB Qwen3-0.6B, mean boot latency was 95,679 ms without snapshots and 13,797 ms with them for vLLM, and 83,713 ms vs 17,486 ms for SGLang. Even a tiny model, with weights that load in a second, took 95 s to boot in vLLM: almost all of that is engine initialization, not bytes.

The pattern across all five: **for small and medium models, engine initialization (compile, graphs, profiling) dominates; for big models, weight bytes catch up; on a cloud, getting the GPU can dominate both.** The "Breaking the Ice" study of vLLM cold starts ([Kabakibo et al., MLSys 2026](https://research.ibm.com/publications/breaking-the-ice-analyzing-cold-start-latency-in-vllm)) adds one more finding worth knowing: it splits vLLM startup into six steps and finds the process "predominantly CPU-bound". The host CPU you rent with the GPUs matters for $T_{\text{start}}$.

### Measure it yourself

You cannot fix a stage you have not timed. `startup_stages.py` brings up a small model on CPU stage by stage, the way an engine does, and prints where the time went. The model is `Qwen/Qwen2.5-0.5B-Instruct`, about 1 GB of bf16 safetensors and the same model Modal used for one of its snapshot numbers. The stages it can't reproduce on a laptop (image pull, CUDA init, graph capture) are absent; the ones it can (imports, tokenizer, model construction, weight read, KV allocation, compile, warmup, health) are timed separately.

```python
# startup_stages.py - time each stage of bringing up a model "replica" on CPU and print a breakdown.
#   /tmp/absorb-venv/bin/python startup_stages.py [--compile]
# Model: Qwen2.5-0.5B-Instruct from the local Hugging Face cache (about 1 GB of bf16 safetensors).
import time
T0 = time.perf_counter()
stages = []                                   # (name, seconds)
def mark(name, t_start):
    stages.append((name, time.perf_counter() - t_start))

t = time.perf_counter(); import torch;                          mark("import torch", t)
t = time.perf_counter(); import transformers;                   mark("import transformers", t)
t = time.perf_counter(); from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import http.server, os, sys, threading, urllib.request;          mark("other imports", t)

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
t = time.perf_counter(); path = snapshot_download(REPO, local_files_only=True); mark("resolve weights (cache lookup)", t)
t = time.perf_counter(); tok = AutoTokenizer.from_pretrained(path);              mark("tokenizer load", t)

t = time.perf_counter()
cfg = AutoConfig.from_pretrained(path)
with torch.device("meta"):                    # build the module tree without allocating weights
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
mark("build model skeleton (meta)", t)

t = time.perf_counter()
sd = load_file(os.path.join(path, "model.safetensors"))   # mmap: returns views, pages not read yet
mark("open safetensors (mmap)", t)
nbytes = sum(v.numel() * v.element_size() for v in sd.values())
t = time.perf_counter()
for v in sd.values():                         # touch every page: this is the real read
    v.sum()
mark(f"read weights ({nbytes/1e9:.2f} GB)", t)
t = time.perf_counter()
model.load_state_dict(sd, strict=False, assign=True)       # assign: adopt the tensors, no copy
model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg)  # non-persistent buffer: rebuild on CPU
model.tie_weights(); model.eval()
mark("attach weights to model", t)

# "KV allocation": vLLM preallocates the KV pool up front; here, 8 sequences x 4096 tokens
L, kvh, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
t = time.perf_counter()
kv = torch.zeros(2, L, 8 * 4096, kvh, hd, dtype=torch.bfloat16)  # (K/V, layers, slots, kv_heads, head_dim)
kv.fill_(0)                                   # touch it, as a real allocator's first use would
mark(f"allocate KV pool ({kv.numel()*2/1e9:.2f} GB)", t)

ids = tok("The capital of France is", return_tensors="pt").input_ids   # (1, T)
if "--compile" in sys.argv:
    t = time.perf_counter()
    fwd = torch.compile(model)
    with torch.no_grad(): fwd(ids)            # first call traces and compiles
    mark("torch.compile (first call)", t)
    model_call = fwd
else:
    model_call = model

t = time.perf_counter()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=8, do_sample=False)
mark("warmup generate (8 tokens)", t)

t = time.perf_counter()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=8, do_sample=False)
mark("second generate (8 tokens), for comparison", t)

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers()
    def log_message(self, *a): pass
t = time.perf_counter()
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
code = urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/health").status
mark(f"start HTTP server + first /health ({code})", t)

total = time.perf_counter() - T0
print(f"output: {tok.decode(out[0][ids.shape[1]:])!r}")
print(f"{'stage':48s} {'seconds':>8s} {'share':>6s}")
for name, s in stages:
    if not name.startswith("second"):
        print(f"{name:48s} {s:8.3f} {100*s/total:5.1f}%")
print(f"{'total (process start excluded)':48s} {total:8.3f}")
print(f"{'(second generate, for comparison)':48s} {stages[-2][1]:8.3f}")
```

Plain eager run, no compile:

```console
$ /tmp/drvenv/bin/python startup_stages.py
output: ' Paris. It is the largest city in'
stage                                             seconds  share
import torch                                        0.890  17.9%
import transformers                                 0.704  14.1%
other imports                                       0.210   4.2%
resolve weights (cache lookup)                      0.000   0.0%
tokenizer load                                      0.144   2.9%
build model skeleton (meta)                         0.795  16.0%
open safetensors (mmap)                             0.029   0.6%
read weights (0.99 GB)                              1.638  32.9%
attach weights to model                             0.005   0.1%
allocate KV pool (0.40 GB)                          0.076   1.5%
warmup generate (8 tokens)                          0.265   5.3%
start HTTP server + first /health (200)             0.012   0.2%
total (process start excluded)                      4.980
(second generate, for comparison)                   0.207
```

With `torch.compile`, and an empty Inductor cache (`TORCHINDUCTOR_CACHE_DIR` points it at a directory we control, which we deleted first):

```console
$ export TORCHINDUCTOR_CACHE_DIR=/tmp/drs/inductor-cache; rm -rf $TORCHINDUCTOR_CACHE_DIR
$ /tmp/drvenv/bin/python startup_stages.py --compile
output: ' Paris. It is the largest city in'
stage                                             seconds  share
import torch                                        0.448   3.6%
import transformers                                 0.611   5.0%
other imports                                       0.197   1.6%
resolve weights (cache lookup)                      0.000   0.0%
tokenizer load                                      0.113   0.9%
build model skeleton (meta)                         0.543   4.4%
open safetensors (mmap)                             0.032   0.3%
read weights (0.99 GB)                              0.517   4.2%
attach weights to model                             0.003   0.0%
allocate KV pool (0.40 GB)                          0.092   0.8%
torch.compile (first call)                          9.422  76.4%
warmup generate (8 tokens)                          0.166   1.3%
start HTTP server + first /health (200)             0.012   0.1%
total (process start excluded)                     12.332
(second generate, for comparison)                   0.165
```

The same command again, with the 4.3 MB cache the first run left behind:

```console
$ /tmp/drvenv/bin/python startup_stages.py --compile
output: ' Paris. It is the largest city in'
stage                                             seconds  share
import torch                                        0.422   7.7%
import transformers                                 0.595  10.8%
other imports                                       0.198   3.6%
resolve weights (cache lookup)                      0.000   0.0%
tokenizer load                                      0.118   2.1%
build model skeleton (meta)                         0.677  12.3%
open safetensors (mmap)                             0.007   0.1%
read weights (0.99 GB)                              0.040   0.7%
attach weights to model                             0.003   0.1%
allocate KV pool (0.40 GB)                          0.017   0.3%
torch.compile (first call)                          3.098  56.1%
warmup generate (8 tokens)                          0.168   3.1%
start HTTP server + first /health (200)             0.008   0.2%
total (process start excluded)                      5.517
(second generate, for comparison)                   0.165
```

### Reading the output

- **Compile dominates once it is on.** A cold `torch.compile` was 9.4 s of a 12.3 s start, three quarters of it, the same share fal measured on an H100. The warm cache cut it to 3.1 s. What remains is Dynamo tracing the Python code, which the FX-graph cache does not skip. Real engines cache more aggressively (vLLM caches the traced graph too), but the lesson is the same: compile is the biggest single stage, and it is the most cacheable one.
- **Weight bytes depend on where they are, not on how big they are.** The eager run read 0.99 GB in 1.64 s (about 0.6 GB/s) because those pages had been evicted. In the compile runs the same read took 0.52 s and then 0.04 s, from the page cache at 25 GB/s. I could not drop the page cache on macOS without root, so these are whatever state the cache happened to be in. On Linux, run `sync; echo 3 > /proc/sys/vm/drop_caches` before the first run ([page cache in measurements](../foundations/storage-and-data.md#the-page-cache-in-your-own-measurements)) to time the disk instead of RAM.
- **Imports and model construction are a fixed second or two.** `import torch` + `import transformers` + building the module tree cost 1.8-2.6 s on a laptop. vLLM's own imports, its worker processes and NCCL setup add to this on a server. Modal's snapshot post measured an `import torch` cold start at around 5 s in their containers.
- **Opening a safetensors file is free; reading it is not.** `load_file` returned in 7 ms because it maps the file and returns views ([safetensors and mmap](../foundations/storage-and-data.md#safetensors-a-header-plus-raw-bytes)). The real I/O happens on first touch. A loader that "loads" in milliseconds and then makes the first forward pass slow has just moved the cost.
- **Warmup is real.** In the eager run the first `generate` took 0.265 s and the second 0.207 s, 28% slower; in the compile runs the compiled forward pass had already exercised the model and the two were equal. On a GPU, first calls also pay for lazy cuBLAS initialization and autotuning. Warm up before you admit traffic, not with it.

On a GPU server, add the stages this laptop doesn't have, and log each one with a timestamp the way this script does. vLLM's startup log already prints separate lines for weight loading, compilation and graph capture; ship those lines to your log store and chart their p50 and p99 per deploy, as Baseten's dashboard does per phase.


## 3. Shrinking each stage

The fixes map one-to-one onto the stages. Work through them in the order your own measurements say, not in the order below. The [cold-start budget arithmetic](../orchestration/internals.md#cold-start-budget-arithmetic) in the orchestration chapter shows the serial-sum model; this section is about making each term smaller.

### Node provisioning

On a cloud, getting a GPU machine is the longest stage and the least predictable. Modal reports that on one provider ("Cloud A") a launch request that is accepted [boots 99.6% of the time, "relatively quickly (2-3 minutes)"](https://modal.com/blog/gpu-health), and then still has to pass health checks. Modal's answer is a buffer of idle, already-checked GPUs that containers are placed onto, which the [serverless GPU post](https://modal.com/blog/truly-serverless-gpus) credits with taking tens of minutes off the worst case.

**On a leased fleet:** this stage is zero if a free, healthy node exists, and infinite if none does. The equivalent of Modal's buffer is a small pool of nodes you keep healthy and unassigned (or running preemptible batch work you can evict in seconds); [section 5](#5-recovery) sizes it from failure rates. Run the slow GPU diagnostics on spares while they are idle, not on the critical path.

### Images

A container image is a stack of gzip-compressed tar layers ([images and layers](../foundations/containers-and-virtualization.md#registries-and-pull-time)). The default pull downloads every layer, decompresses it and unpacks it before the container can start. The bottleneck is usually not the network. Modal measured layers downloading [concurrently at about 2 GiB/s while "decompression of the gzipped tarballs happens single-threaded at around 80 MiB/s"](https://modal.com/blog/jono-containers-talk), so an 8 GB image spends about a minute just in gzip. DEFLATE, the algorithm inside gzip, is inherently serial within one stream, which is why Modal's own image format dropped it: [single-threaded DEFLATE "limits you to ~100 MB/s"](https://modal.com/blog/truly-serverless-gpus).

The fixes, cheapest first:

1. **Don't pull at start time at all.** Pre-pull the serving image onto every node that might run it, and set `imagePullPolicy: IfNotPresent`. Kubernetes documents this as ["Pre-pulled images"](https://kubernetes.io/docs/concepts/containers/images/#pre-pulled-images), with the caveat that it "will not work reliably if your cloud provider manages nodes and replaces them automatically", which on a leased fleet you control. A DaemonSet that runs the new image with a no-op command on every node, a day before the deploy, does it.
2. **Keep the image small and stable.** CUDA, PyTorch and the engine in a base layer that changes monthly; your code in a thin top layer. Never put weights in the image (the [weight distribution](../orchestration/internals.md#model-weight-distribution) section explains why).
3. **Skip gzip.** The Microsoft study above measured uncompressed layers pulling in 129.9 s instead of 140.8 s for the 14 GB vLLM image. zstd layers decompress much faster than gzip and are supported by containerd.
4. **Lazy pulling.** A lazy-pulling snapshotter starts the container before the image has arrived and fetches each file's bytes on first read:
   - **SOCI** (AWS) builds a separate index that maps files to byte ranges inside unmodified gzip layers. In the [SOCI paper](https://cdn.amazon.science/84/96/51ea2e0d40fca8cfaee9328278b9/soci-paper.pdf), a 1.3 GB Python image whose application reads about 80 MB is ready to run in about 2.6 s instead of about 20 s (2,630 ms vs 19,551 ms in its Table I), because "94% of the download is wasted" in a full pull. The paper's abstract reports a crossover at 80% access density: if the container reads most of its image anyway, a parallel full pull wins. (The paper is not fully consistent on this: one section says no crossover was observed on its synthetic 1 GB image while a table shows 0.8x at 100% density. Treat 80% as a rough line and measure.)
   - **eStargz** [compresses each file separately](https://github.com/containerd/stargz-snapshotter/blob/main/docs/estargz.md) so the layer stays valid gzip, appends a table of contents, and places files the workload is known to need before a `.prefetch.landmark` entry so they can be fetched in one range request. It requires converting the image.
   - **Nydus** uses its own format and a FUSE daemon. For vLLM it went badly: the Microsoft study "excluded Nydus because FUSE-streaming the 14 GB Python/CUDA stack caused startup to exceed 900s". For llama.cpp, Nydus reached a 77.8 s time to first token with prefetching and 223.4 s without.
   
   An inference image is the worst case for lazy pulling: `import torch` and the engine's imports touch most of the large libraries, so access density is high and every first read becomes a network round trip. Lazy pulling helps a fleet that churns through many different images. A serving fleet that runs three images benefits more from pre-pulling them.
5. **Modal's lazy filesystem.** Modal's ImageFS is a FUSE filesystem over a content-addressed cache: [the image index is "around five megabytes and can be loaded in one to 100 milliseconds"](https://modal.com/blog/jono-containers-talk), files shared between images are stored once, and file contents come from a tiered cache (page cache, local SSD, an in-zone cache server, blob storage). The same post has a tuning detail worth copying for any FUSE-backed model store: read-ahead raised from Linux's 128 KB default to 32 MB "worked wonders", and a 1 GB setting "resulted in disastrous latency issues in production".

**On a leased fleet:** you run a handful of images on nodes you own, so pre-pulling (item 1) captures nearly all of the benefit and costs nothing but disk. Lazy filesystems like Modal's solve a problem you don't have: thousands of customer images that can't all be cached on every host.

### Weights

Weight time is $\text{bytes} / \text{bandwidth}$, and the bandwidth is set by the slowest of: the storage, the network path to it, the loader's parallelism, and host-to-GPU copies. The loader is often the problem. NVIDIA's [Run:ai Model Streamer post](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/) loaded Llama-3-8B (15 GB, one safetensors file) on an AWS instance with vLLM:

| Storage | Default safetensors loader | Run:ai Model Streamer |
|---|---|---|
| EBS gp3 SSD | 47.99 s | 14.34 s (16 threads) |
| EBS io2 SSD | 47 s | 7.53 s (8 threads) |
| S3 | not supported | 4.88 s (32 clients) |

The default loader got about 0.32 GB/s from either volume, regardless of its provisioned speed. The streamer reads many tensors concurrently into a CPU buffer while earlier ones are copied to the GPU, and it reads straight from S3. Total vLLM time-to-ready with the streamer on S3 was 23.18 s: with weights down to 5 s, about 18 s of the 23 s is everything else, the same lesson as fal's table.

In vLLM the streamer is a load format ([docs](https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html)); untested here:

```bash
# untested. S3 path, 16 concurrent readers; runai_streamer_sharded reads per-rank shard files for TP>1
vllm serve s3://models/llama-3.1-8b-instruct/ --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency": 16}'
```

**Local caches.** The other approach is to keep a copy of the weights next to the GPUs. fal's [cold-start explainer](https://fal.ai/learn/serverless/what-is-a-serverless-gpu-cold-start) lists its tiers: 10-15 GB/s from local RAID NVMe on the node, 6-8 GB/s from a distributed cache, and 1.5-8 GB/s from object storage. Its FlashPack loader (one contiguous file, memory-mapped, read in buffers on separate CUDA streams so disk reads and host-to-device copies overlap) lands "a 20 GB model's weights in about six to seven seconds" at about 3.1 GB/s. Modal's [serverless GPU post](https://modal.com/blog/truly-serverless-gpus) gives similar numbers: model weights load "at a few gigabytes per second", for a total of "a few seconds to a few hundred seconds".

Put those bandwidths against 8B, 70B and 405B models in bf16 and FP8 (one byte per parameter). The "network FS" row is an assumption, a typical NFS or parallel-filesystem client; measure yours:

```python
# loadmath.py - weight-load time = bytes / bandwidth, per replica and per fleet-wide deploy.
#   python3 loadmath.py
# Bandwidths are per node, read into host RAM / GPU. Sources for each are in the text; the
# "network FS" figure is an assumption, measure yours.
models = {"8B": 8e9, "70B": 70e9, "405B": 405e9}
dtypes = {"bf16": 2, "fp8": 1}
sources = {                                  # GB/s per node
    "network disk, default loader": 0.32,         # 15 GB in ~48 s (Run:ai post: EBS gp3, plain safetensors)
    "network FS (assumed)": 2.0,
    "S3, parallel streamer": 3.1,           # 15 GB in 4.88 s (Run:ai Model Streamer on S3)
    "local NVMe RAID": 12.0,                # fal: 10-15 GB/s
    "peer over 200 Gb/s NIC": 25.0,         # line rate; real transfers get less
}
print(f"{'source':36s} " + " ".join(f"{m+' '+d:>10s}" for m in models for d in dtypes))
for src, bw in sources.items():
    row = []
    for m, p in models.items():
        for d, b in dtypes.items():
            row.append(f"{p * b / 1e9 / bw:9.0f}s")
    print(f"{src:36s} " + " ".join(row))

# A deploy: every node needs the new weights. Origin egress vs peer fan-out (BDN-style chunking:
# each of N nodes pulls 1/N of the file from origin, then the rest from peers).
print()
size_gb, nodes = 140, 20                     # 70B bf16, 20 cold nodes (Baseten's BDN example)
origin_gbps, nic_gbps = 25.0, 25.0           # origin egress for the whole cluster; NIC per node
naive = nodes * size_gb / origin_gbps
chunked_origin = size_gb / origin_gbps       # origin sends each byte once
chunked_peer = size_gb * (nodes - 1) / nodes / nic_gbps   # each node receives the rest from peers
print(f"{nodes} nodes x {size_gb} GB, origin {origin_gbps} GB/s:")
print(f"  everyone pulls from origin: {nodes*size_gb/1e3:.1f} TB egress, {naive:6.0f} s")
print(f"  chunked peer cache:         {size_gb/1e3:.2f} TB egress, {chunked_origin + chunked_peer:6.0f} s "
      f"(each node pulls {size_gb/nodes:.0f} GB from origin, then {size_gb*(nodes-1)/nodes:.0f} GB from peers; no overlap assumed)")
```

```console
$ python3 loadmath.py
source                                  8B bf16     8B fp8   70B bf16    70B fp8  405B bf16   405B fp8
network disk, default loader                50s        25s       438s       219s      2531s      1266s
network FS (assumed)                         8s         4s        70s        35s       405s       202s
S3, parallel streamer                        5s         3s        45s        23s       261s       131s
local NVMe RAID                              1s         1s        12s         6s        68s        34s
peer over 200 Gb/s NIC                       1s         0s         6s         3s        32s        16s

20 nodes x 140 GB, origin 25.0 GB/s:
  everyone pulls from origin: 2.8 TB egress,    112 s
  chunked peer cache:         0.14 TB egress,     11 s (each node pulls 7 GB from origin, then 133 GB from peers; no overlap assumed)
```

Three things to read off it:

- **The loader matters more than the medium.** The same 70B checkpoint takes 438 s through the default single-stream loader and 45 s through a parallel reader from object storage. Switching loaders is a flag; moving storage is a project.
- **For 405B, only node-local or peer copies are fast.** At 2-3 GB/s, a 405 GB checkpoint takes three to four minutes per replica. From local NVMe it takes about a minute. For multi-node replicas, each node reads only its own shards (vLLM's `runai_streamer_sharded` and `sharded_state` formats, [pre-sharding](../inference/serving-hands-on.md#speeding-up-model-loading-time)).
- **A fleet-wide deploy is an egress problem.** Twenty cold nodes each pulling 140 GB from origin is 2.8 TB of egress. If every node pulls a different 1/20 of the file from origin and the rest from its peers, origin sends the file once.

The last line is the design of Baseten's delivery network. Its [BDN post](https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/) works the same example: "50 replicas of a 140 GB model across 20 nodes in a fully cold cluster" pull about 7 GB each from origin, so "origin bandwidth is 1× the model size (not 50×)". Files are split into fixed-size chunks, a consistent hash ring assigns each chunk an owner node, nodes fetch the rest from peers, and "at most one fetch per (node, content-hash) pair is in progress at a time", so 50 replicas on one node don't download the same file 50 times. Weights are mirrored from Hugging Face at deploy time (at 1-5 GB/s), so serving never depends on the Hub: the post notes "a Hugging Face download that typically takes 90 seconds can take 10 minutes during a popular model launch". Baseten credits the system with cold starts "2-3x faster".

**On a leased fleet:** you have the same thundering-herd problem every time you deploy, so the peer-chunking idea applies directly. The cheaper equivalent is to **pre-stage** the weights on every node's NVMe before the deploy starts: a DaemonSet or a cron job copies the new version to `/mnt/nvme/models/<version>/`, writes a `.complete` marker last (the [init-container pattern](../orchestration/internals.md#model-weight-distribution)), and the rollout waits until every target node has the marker. The copy runs at whatever pace origin and network allow, an hour before anyone needs it, so it is off the critical path entirely. Dragonfly or a small rsync tree gives you peer fan-out if origin egress is the limit. And quantize: an FP8 checkpoint is half the bytes at every row of the table.

Two Hugging Face settings help the first download onto a node: `HF_XET_HIGH_PERFORMANCE=1` makes the client ["try to saturate the network bandwidth of this machine and utilize all CPU cores"](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) (it replaces the deprecated `HF_HUB_ENABLE_HF_TRANSFER`), and `HF_HUB_OFFLINE=1` on serving nodes makes no HTTP calls at all, so a Hub outage or a revision check can't stall a start. Pin the revision so caches stay valid.

### Compile and CUDA graph caches

`torch.compile` and CUDA graph capture are pure functions of (model code, engine version, PyTorch version, GPU type, configuration). Running them on every start recomputes the same answer. Every serving platform that measured this caches the result:

- **Replicate** stores compile artifacts per model version: ["Cache files are keyed on model version and stored close to GPU nodes"](https://replicate.com/blog/torch-compile-caching). Boot times fell from about 120 s to about 60 s for flux-kontext-dev and from about 400 s to about 150 s for flux.1-dev-lora; models "now start 2-3x faster". They keep compiling, because the compiled model runs faster.
- **fal** caches the Inductor output, which is most of the 202.9 s to 77.79 s drop in compile + warmup in its table. Two of its rules carry over: ["Kernel caches are organized per GPU type"](https://fal.ai/docs/examples/serverless/cut-cold-starts) ("A cache built on an H100 gives an L40 boot nothing"), and warm up at the extremes of the input shapes you serve so the cached graph covers all of them.
- **vLLM** keeps its compile cache under `VLLM_CACHE_ROOT` (default `~/.cache/vllm`), in `torch_compile_cache/<hash>/rank_<i>_<j>`. The [design doc](https://docs.vllm.ai/en/latest/design/torch_compile.html) says to "directly copy the whole ~/.cache/vllm/torch_compile_cache directory" to save startup compilation, along with `TRITON_CACHE_DIR` if you set one. The hash covers the configuration and the model's source, so a cache from a different config is ignored, not misused.

The knobs, with what each does to startup ([vLLM optimization levels](https://docs.vllm.ai/en/latest/design/optimization_levels/), [`vllm serve` reference](https://docs.vllm.ai/en/latest/cli/serve.html), [SGLang server arguments](https://docs.sglang.io/advanced_features/server_arguments.html)); all **untested** here:

| Engine | Knob | Effect on startup | Cost |
|---|---|---|---|
| vLLM | `VLLM_CACHE_ROOT=/mnt/nvme/vllm-cache` | compile cache on local disk that survives the container; copy it between nodes of the same GPU type | none |
| vLLM | `-O0` or `--enforce-eager` | no `torch.compile`, no CUDA graphs: the fastest start | slower decode, especially at small batch |
| vLLM | `--cudagraph-capture-sizes 1 2 4 8 16 32` or `--max-cudagraph-capture-size 64` | fewer graphs to capture | batch sizes above the largest captured size run without graphs. (Marked deprecated under the new V2 model runner, which has separate prefill and decode size lists; check `vllm serve --help` on your version) |
| vLLM | `VLLM_ENABLE_STARTUP_PLAN=1` | saves the measured memory profile under `$VLLM_CACHE_ROOT/startup_plan/` and reuses it on later boots, skipping the profiling pass; falls back to full profiling on any mismatch | none |
| SGLang | `--cuda-graph-max-bs-decode N`, `--cuda-graph-bs-decode ...` | fewer decode graphs | same as above |
| SGLang | `TORCHINDUCTOR_CACHE_DIR=/mnt/nvme/inductor` | persistent Inductor cache (default `/tmp/torchinductor_root`) | none |
| SGLang | `--skip-server-warmup` | skips the built-in warmup request | first real request pays for it; prefer your own warmup |

The trade-off between compiled and eager is real: compiled graphs make decode faster, and an engine that starts in half the time but decodes 20% slower needs 20% more replicas for the same traffic. The right default on a fixed fleet is **compile, and cache it**; `-O0` is for development and for the rare case of a model that must start in seconds and runs nearly idle.

**On a leased fleet:** this is the highest-return item on the list. Build the cache once per (model version, engine version, GPU type) in CI, on a node of the target GPU type, and ship it to every node's NVMe with the weights. A homogeneous fleet needs one cache per model; a mixed H100/H200 fleet needs two.

### Process and GPU snapshots

Everything after the image pull (imports, CUDA contexts, weights in GPU memory, compiled kernels, captured graphs) is process state. If you could save that state once and restore it, you would skip all of it. That is checkpoint/restore.

**CRIU and cuda-checkpoint.** CRIU (Checkpoint/Restore In Userspace) dumps a Linux process tree, its memory, file descriptors and registers, to disk and restores it later, possibly on another machine. It can't see GPU memory. NVIDIA's [`cuda-checkpoint`](https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/) fills the gap: `cuda-checkpoint --toggle --pid $PID` locks the CUDA driver API in that process, waits for submitted work to finish, copies device memory into host memory and releases the GPU. The process now holds no GPU state, so CRIU can dump it like any CPU process. On restore the order reverses, and GPU memory mappings come back at their original virtual addresses, so pointers inside the process stay valid.

**Modal's snapshots, and what gVisor does and doesn't do.** Modal runs every container under gVisor's `runsc` ([gVisor](../foundations/containers-and-virtualization.md#gvisor-a-kernel-in-user-space) is a kernel implemented in user space). Because that user-space kernel owns all of a container's process state, it can save and restore it: Modal's [memory snapshot post](https://modal.com/blog/mem-snapshots) notes that "at least eighteen system components implement checkpoint/restore functionality" inside gVisor. Modal added GPU state through the CUDA checkpoint/restore API ([GPU snapshot post](https://modal.com/blog/gpu-mem-snapshots)). The results, at p0:

| Workload | Before | With GPU snapshot |
|---|---|---|
| Parakeet speech recognition | about 20 s | as low as 2 s |
| vLLM, Qwen2.5-0.5B-Instruct | 45 s | 5 s |
| ViT with `torch.compile` | 8.5 s (already with a CPU-only snapshot) | 2.25 s |

and, from the same mechanism, Ministral 3 3B on vLLM went [from about 118 s to about 12 s](https://modal.com/blog/mistral-3). The posts credit these speedups to **snapshot and restore**, not to the sandbox: gVisor is what lets Modal implement checkpoint/restore for arbitrary customer containers, and it is there for multi-tenant isolation. A process restored with CRIU under ordinary `runc` gets the same kind of speedup, which is exactly what NVIDIA's Kubernetes implementation does.

**Dynamo Snapshot.** NVIDIA's [Dynamo Snapshot](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/) checkpoints vLLM and SGLang workers with CRIU on runc-managed containers. Two details from it matter for anyone building this:

- **Free the KV cache first.** An engine preallocates almost all free GPU memory as the KV pool. Snapshot that and you save gigabytes of zeros. "Unmap and release of the KV cache reduces the total artifact size of Qwen3-0.6B on a B200 from ~190 GiB to ~6 GiB." Modal's [snapshot guide](https://modal.com/docs/guide/memory-snapshot) gives the same rule: "it is often better to discard the initial, unfilled KV cache before the snapshot is taken, then recreate it on restore."
- **Restore is a storage problem.** Upstream CRIU restores memory with one read at a time. With Linux AIO and parallel memory files the restore times became:

  | Model | Snapshot | Upstream CRIU | AIO | AIO + parallel memfd |
  |---|---|---|---|---|
  | Qwen3-0.6B | 6.2 GiB | 6.8 s | 2.9 s | 2.4 s |
  | Qwen3-8B | 26 GiB | 24 s | 11 s | 4.7 s |
  | gpt-oss-120b | 129 GiB | 119 s | 54 s | 15 s |

  A proof of concept that keeps weights out of the CRIU image and stripes them across 8 local NVMe drives restored gpt-oss-120b in under 5 s. Status in the post: single-GPU vLLM and SGLang only; multi-GPU is in progress, and some of the optimizations are not yet shipped.

**The limits.** Modal's guide is explicit that GPU snapshots "do not speed up model loading from storage" and that they "are generally incompatible with multi-GPU code". Both matter for large models. Parasail's [vLLM snapshot measurements](https://www.parasail.io/blogs/cutting-cold-start-latency-snapshotting) on H200 show the gain shrinking as the model grows: Qwen2.5-7B at TP1 went from 63.56 s to 15.22 s, while a quantized Llama-3.3-70B at TP1 went from 138.19 s to 119.78 s, because a snapshot of a big model is mostly weights and restoring weights is just loading weights. Snapshots also capture state that should not be shared: Modal warns that anything established before the snapshot "will be reused over and over again" and that "expectations of entropy may be violated", the same [clone problem](../foundations/containers-and-virtualization.md#snapshots-starting-from-a-warm-vm) as VM snapshots. And restores can fail; Modal "automatically fallback[s] to a standard container startup", which you need too.

**The cheaper equivalent: don't let the process die.** On a fixed fleet the replica you want to start is often the one you just stopped. vLLM's **sleep mode** keeps the process, its CUDA context and its compiled graphs alive and only moves memory ([docs](https://docs.vllm.ai/en/latest/features/sleep_mode.html)): level 1 "will offload the model weights and discard the KV cache" (weights go to CPU RAM), level 2 discards both. Waking reverses it. vLLM's [sleep-mode post](https://vllm.ai/blog/sleep-mode) measured, on an A100 at TP1:

| Model | Wake from level 1 | Wake from level 2 | Cold start |
|---|---|---|---|
| Qwen3-0.6B | 0.26 s | 0.85 s | 37.6 s |
| Phi-3-vision | 0.82 s | 2.58 s | 58.1 s |

and quotes "~3-6s for large models" from level 1. It needs `--enable-sleep-mode`, and the HTTP endpoints exist only with `VLLM_SERVER_DEV_MODE=1`, so they must not be exposed outside the node (untested here):

```bash
# untested. Two models time-sharing one GPU; only the awake one serves.
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-8B --port 8001 --enable-sleep-mode --gpu-memory-utilization 0.45 &
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B --port 8002 --enable-sleep-mode --gpu-memory-utilization 0.45 &
curl -X POST 'localhost:8002/sleep?level=1'      # weights to host RAM, KV dropped
curl -X POST  localhost:8001/sleep?level=1 && curl -X POST localhost:8002/wake_up   # swap
curl localhost:8002/is_sleeping
```

SGLang has the same primitive under `--enable-memory-saver`, with `POST /release_memory_occupation` and `POST /resume_memory_occupation` taking `{"tags": ["kv_cache", "weights"]}` ([SGLang for RL](https://docs.sglang.io/docs/advanced_features/sglang_for_rl)); RL frameworks use it to hand the GPU to the trainer between rollouts ([RL infrastructure](rl-infrastructure.md)).

Level 1 needs host RAM for the sleeping weights: an 8-GPU node with 2 TB of RAM can hold several sleeping 70B models. That turns model swaps on a node from minutes into seconds without any checkpoint machinery. For many fine-tunes of one base model, go further and never swap at all: load adapters at runtime into one warm base engine (vLLM: `--enable-lora --max-loras N` with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and `POST /v1/load_lora_adapter`, per the [LoRA docs](https://docs.vllm.ai/en/latest/features/lora.html), which warn to use runtime updating only in a trusted environment).

**On a leased fleet:** you don't need gVisor to get snapshot speedups, and you may not need snapshots at all. In order of cost: keep weights resident (sleep mode, or simply don't stop replicas you will need again); serve fine-tunes as runtime LoRAs on a shared base; and only then build CRIU + cuda-checkpoint restore, for single-GPU replicas that start often, with the KV cache freed before the dump and a fallback to a normal start.

### Python and CUDA init

What's left after the above is a few seconds of imports and CUDA initialization per process. Two things are worth doing anyway:

- **Retry CUDA init.** On one of Modal's clouds, ["L4s flake at CUDA initialization in 0.1% of cases"](https://modal.com/blog/gpu-health), and the post recommends retrying `cuInit` in application code. A replica that dies on a transient init error and goes into a 5-minute crash-loop backoff has turned a 0.1% flake into a 5-minute outage of that replica.
- **Put the Python environment on local disk.** A virtualenv on NFS turns 26,000 syscalls into 26,000 network round trips.

### Warmup, readiness and joining the router

The last three stages are where replicas are let in too early. A replica that is "up" (the HTTP port answers) but has not run a real forward pass will be slow on its first requests: lazy kernel selection, autotuning, the first allocations. vLLM's `/health` endpoint is not a readiness signal for this. Per the [current docs](https://docs.vllm.ai/en/latest/serving/online_serving/), it returns 503 only when the engine is dead and 200 otherwise, including while the engine is asleep, and what it returns while the model is still loading is not documented. Test it on your version before you build a probe on it.

The pattern that works: the server's own startup script, after the engine reports healthy, sends a few warmup requests that cover your shapes (short and long prompts, the maximum batch you expect), and only then creates a ready file or flips a `/ready` endpoint that the readiness probe checks. Modal's SGLang example does exactly this: [three short completions after `/health` passes](https://modal.com/docs/examples/sglang_low_latency). Together's measurement shows why the router step needs measuring too: [READY to first token through routing took another 26-40 s](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference).


## 4. Deploying a new model version on a fixed fleet

A deploy changes the desired state ("16 replicas of version B" instead of "16 replicas of version A") and lets a reconciler move the fleet there ([desired state and reconciliation](control-planes.md#desired-state-and-reconciliation)). The interesting part is what happens in between, on a fleet with no empty GPUs.

### Strategies

- **Rolling update.** Replace replicas a few at a time: start some B replicas, wait until they are ready, retire the same number of A replicas, repeat. Two numbers control it. **maxSurge** is how many replicas *above* the desired count may exist during the update (extra B replicas started before any A replica is removed). **maxUnavailable** is how many replicas *below* the desired count may be not-ready at once. Kubernetes Deployments default both to 25%, with [maxSurge rounded up and maxUnavailable rounded down](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/).
- **Blue-green.** Start a complete second fleet of B ("green") next to A ("blue"), test it, and switch all traffic at once. Rollback is switching back, which is instant while blue is still running.
- **Canary.** Send a small fraction of traffic to B, compare its metrics with A, and widen the fraction step by step if B is no worse. A canary is a traffic-shifting policy; underneath, it still needs B replicas to exist, so it is combined with one of the two above.
- **Shadow.** Copy live requests to B, discard its answers, and compare them offline. It costs B's GPUs for whatever fraction you mirror and puts zero user traffic at risk.

### Surge and unavailability on a full fleet

On a fleet with free GPUs, surge is free: start B replicas on spare nodes, then remove A. On a full fleet, a surge replica needs a node that doesn't exist. The [Kubernetes chapter](../orchestration/kubernetes-from-zero.md#deployment-and-replicaset) points out the result: `maxSurge: 1, maxUnavailable: 0` on a full GPU cluster leaves the new pod Pending forever. The choice is between capacity (spare nodes) and time (removing old replicas first).

`rollout.py` simulates the RollingUpdate rules second by second, with the fleet constraint that a new replica needs a free node and an old replica gives its node back only after it has [drained](#draining-in-flight-streams) (finished its in-flight requests):

```python
# rollout.py - a Deployment-style rolling update on a fixed GPU fleet, simulated second by second.
#   python3 rollout.py
# Rules (the RollingUpdate contract): pods that exist <= N + maxSurge; Ready pods >= N - maxUnavailable.
# Fleet constraint: a new pod needs a free node; an old pod frees its node only after it has drained.
import math

def roll(N, surge, unavail, spare_nodes, start_s, drain_s, horizon=36000):
    nodes_free = spare_nodes
    old_ready, old_draining = N, []          # draining: list of finish times
    new_starting, new_ready = [], 0          # starting: list of ready times
    min_ready, t = N, 0
    while new_ready < N and t < horizon:
        # finished drains free their nodes; finished starts become Ready
        nodes_free += sum(1 for f in old_draining if f <= t); old_draining = [f for f in old_draining if f > t]
        new_ready += sum(1 for r in new_starting if r <= t);  new_starting = [r for r in new_starting if r > t]
        exist = old_ready + len(old_draining) + len(new_starting) + new_ready
        # scale up the new version: within surge budget, needs a free node
        can_add = min(N + surge - exist, nodes_free, N - new_ready - len(new_starting))
        for _ in range(max(0, can_add)):
            new_starting.append(t + start_s); nodes_free -= 1
        # scale down the old version: keep Ready >= N - unavail (starting pods don't count)
        ready = old_ready + new_ready
        can_remove = min(ready - (N - unavail), old_ready)
        for _ in range(max(0, can_remove)):
            old_ready -= 1; old_draining.append(t + drain_s)   # drained: no new requests, finish streams
        min_ready = min(min_ready, old_ready + new_ready)
        t += 1
    return (t if new_ready >= N else None), min_ready

N, START, DRAIN = 16, 300, 120     # 16 replicas; 5 min to Ready from a warm local cache; 2 min drain
print(f"N={N} replicas, start-to-Ready {START} s, drain {DRAIN} s")
print(f"{'strategy':44s} {'spare':>5s} {'duration':>9s} {'min Ready':>9s}")
cases = [
    ("rolling surge 1, unavailable 0", 1, 0, 0),
    ("rolling surge 1, unavailable 0", 1, 0, 1),
    ("rolling surge 0, unavailable 1", 0, 1, 0),
    ("rolling surge 0, unavailable 4 (25%)", 0, 4, 0),
    ("rolling k8s default: surge 25%, unavail 25%", math.ceil(0.25 * N), math.floor(0.25 * N), 0),
    ("rolling surge 4, unavailable 0", 4, 0, 4),
    ("blue-green (surge N, unavailable 0)", N, 0, N),
]
for name, s, u, spare in cases:
    dur, low = roll(N, s, u, spare, START, DRAIN)
    d = f"{dur/60:7.1f} m" if dur else "  stuck"
    print(f"{name:44s} {spare:5d} {d:>9s} {low:9d}")
assert roll(N, 1, 0, 0, START, DRAIN)[0] is None, "surge without a free GPU must deadlock"
```

```console
$ python3 rollout.py
N=16 replicas, start-to-Ready 300 s, drain 120 s
strategy                                     spare  duration min Ready
rolling surge 1, unavailable 0                   0     stuck        16
rolling surge 1, unavailable 0                   1   110.0 m        16
rolling surge 0, unavailable 1                   0   112.0 m        15
rolling surge 0, unavailable 4 (25%)             0    28.0 m        12
rolling k8s default: surge 25%, unavail 25%      0    28.0 m        12
rolling surge 4, unavailable 0                   4    26.0 m        16
blue-green (surge N, unavailable 0)             16     5.0 m        16
```

Reading the table, for 16 replicas that each take 5 minutes to start and 2 minutes to drain:

- **No spare node, surge 1, unavailable 0: stuck.** This is the deadlock, and the assertion at the bottom checks it.
- **One spare node is enough to roll safely, and very slowly.** With surge 1 and one spare, each step waits for one start and one drain: 16 × (5 + 2) minutes ≈ 110 minutes at full capacity the whole time. Taking a replica down first (surge 0, unavailable 1) takes about as long and runs at 15/16 capacity throughout.
- **Faster means losing capacity or holding spares.** Unavailable 4 finishes in 28 minutes at 12/16 capacity (75%) for the duration. Four spare nodes with surge 4 finish in 26 minutes at full capacity. The Kubernetes default (25%/25%) behaves like unavailable 4 when there's no spare node, because the surge pods can't be placed.
- **Blue-green needs a second fleet.** With 16 spare nodes the whole deploy is one start, 5 minutes. On a leased fleet, nobody keeps 16 idle nodes for this.

So on a fixed fleet the deploy plan is a choice of the form: hold $s$ spare nodes and roll at full capacity in about $\lceil N/s \rceil \times (T_{\text{start}} + T_{\text{drain}})$, or roll during a traffic trough with maxUnavailable set to whatever capacity you can lose at that hour. The [autoscaling section](#on-a-fixed-fleet-move-gpus-between-models) adds a third option: the spare nodes during a deploy are whatever batch work you can preempt. Every term scales with $T_{\text{start}}$, which is the case for section 3.

Baseten's [rolling deployments](https://docs.baseten.co/deployment/rolling-deployments) expose the same two numbers with different defaults: `max_surge_percent` defaults to 25 and `max_unavailable_percent` to 0 (range 0-50), and old replicas retire only as new ones become ready. Its example: with 100 old replicas and 25% unavailable, it requests 25 new ones and drops the old version to 75; if only 5 of the new ones become ready, only 5 more old ones retire. That rule, retirement tracks readiness, is what keeps a bad build (one that never becomes ready) from taking capacity with it. Baseten also has `stabilization_time_seconds` (default 0, up to 3600) to pause after each traffic shift. Modal's default deploy is rolling in the same sense: ["existing containers will continue handling inputs (using the previous version of the App) until new containers have completed their cold start"](https://modal.com/docs/guide/managing-deployments).

**On a leased fleet:** the rolling-update arithmetic applies unchanged, and there is no free surge capacity, so decide per deploy whether you are spending spares or capacity, and schedule big rollouts into troughs.

### Draining in-flight streams

An LLM request can stream for minutes. Killing a replica cuts every stream on it mid-sentence. Retrying doesn't help: tokens have already been sent ([no retry after first byte](../inference/serving-hands-on.md#timeouts-retries-and-ejection)). A replica being removed must **drain**: stop receiving new requests, let the in-flight ones finish, then exit.

The mechanics on Kubernetes ([pod termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination)): the pod is removed from Service endpoints, the kubelet runs the container's `preStop` hook if there is one, then sends SIGTERM to process 1, and sends SIGKILL when `terminationGracePeriodSeconds` (default 30 s) runs out. The grace period includes the preStop hook. So the server must, on SIGTERM: fail its readiness check, refuse new requests with a retryable status, wait for in-flight streams, then exit. And the grace period must be longer than the longest stream you allow. `drain.py` is a server that does this and a client that checks no stream was cut:

```python
# drain.py - graceful drain on SIGTERM: fail readiness, refuse new work, finish in-flight streams, exit.
#   python3 drain.py            (starts a server subprocess, streams through it, sends SIGTERM mid-stream)
import asyncio, os, signal, subprocess, sys, time

PORT = 8765
async def server():
    inflight, draining = 0, asyncio.Event()
    async def handle(reader, writer):
        nonlocal inflight
        line = (await reader.readline()).decode()
        while (await reader.readline()) not in (b"\r\n", b""): pass       # skip headers
        path = line.split()[1]
        if path == "/ready":                     # readiness: what the router / kubelet polls
            code = "503 Service Unavailable" if draining.is_set() else "200 OK"
            writer.write(f"HTTP/1.1 {code}\r\nContent-Length: 0\r\n\r\n".encode())
        elif draining.is_set():                  # a request that slipped past the router
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nContent-Length: 0\r\n\r\n")
        else:                                    # /generate?n=K: stream K "tokens", 0.1 s apart
            inflight += 1
            n = int(path.split("n=")[1])
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n")
            for i in range(n):
                writer.write(f"tok{i} ".encode()); await writer.drain(); await asyncio.sleep(0.1)
            writer.write(b"[DONE]\n")
            inflight -= 1
        await writer.drain(); writer.close()
    srv = await asyncio.start_server(handle, "127.0.0.1", PORT)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, draining.set)
    print(f"[server] up, pid {os.getpid()}", flush=True)
    await draining.wait()
    t0 = time.time()
    print(f"[server] SIGTERM: readiness -> 503, {inflight} streams in flight", flush=True)
    while inflight:                              # the grace period must be longer than this loop
        await asyncio.sleep(0.05)
    print(f"[server] drained in {time.time()-t0:.1f} s, exiting", flush=True)
    srv.close()

async def get(path):
    r, w = await asyncio.open_connection("127.0.0.1", PORT)
    w.write(f"GET {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode()); await w.drain()
    data = await r.read(); w.close()
    return data.decode().split("\r\n")[0], data.decode().split("\r\n\r\n", 1)[-1]

async def client(proc):
    await asyncio.sleep(0.3)
    streams = [asyncio.create_task(get(f"/generate?n={n}")) for n in (5, 20, 40)]   # 0.5, 2 and 4 s long
    await asyncio.sleep(0.3)
    proc.send_signal(signal.SIGTERM); print("[client] sent SIGTERM", flush=True)
    await asyncio.sleep(0.1)
    print("[client] /ready ->", (await get("/ready"))[0])
    print("[client] new /generate ->", (await get("/generate?n=3"))[0])
    for s in streams:
        status, body = await s
        print(f"[client] stream: {status}, {len(body.split())-1} tokens, ends with {body.split()[-1]}")
        assert body.strip().endswith("[DONE]"), "a stream was cut off"

if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        asyncio.run(server())
    else:
        proc = subprocess.Popen([sys.executable, __file__, "serve"])
        asyncio.run(client(proc))
        print("[client] server exit code", proc.wait())
```

```console
$ python3 drain.py
[server] up, pid 88105
[client] sent SIGTERM
[server] SIGTERM: readiness -> 503, 3 streams in flight
[server] drained in 3.7 s, exiting
[client] /ready -> HTTP/1.1 503 Service Unavailable
[client] new /generate -> HTTP/1.1 503 Service Unavailable
[client] stream: HTTP/1.1 200 OK, 5 tokens, ends with [DONE]
[client] stream: HTTP/1.1 200 OK, 20 tokens, ends with [DONE]
[client] stream: HTTP/1.1 200 OK, 40 tokens, ends with [DONE]
[client] server exit code 0
```

SIGTERM arrived with three streams in flight. Readiness went to 503 at once, a new request got `503 Retry-After: 1` (a router retries that on another replica, since no token was sent), and all three streams ran to completion; the 4-second one finished 3.7 s after the signal. Then the process exited 0, well before any SIGKILL.

The Kubernetes side, **untested**:

```yaml
# untested sketch; field names per the Kubernetes Pod and Deployment API
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate: {maxSurge: 0, maxUnavailable: 1}       # full fleet: take one down first
  template:
    spec:
      terminationGracePeriodSeconds: 660        # > max generation time (cap max_tokens so this holds)
      containers:
      - name: vllm
        lifecycle:
          preStop:
            exec:                               # give endpoint removal time to reach every router
              command: ["sleep", "15"]
        readinessProbe:
          httpGet: {path: /ready, port: 8080}   # your sidecar's endpoint, flipped after warmup (below)
          periodSeconds: 5
```

The `sleep` in `preStop` covers a race: endpoint removal and SIGTERM happen in parallel, and a router that hasn't seen the removal yet keeps sending requests for a few seconds. Two things make draining bounded: cap `max_tokens` server-side so the longest generation has a known duration, and size the grace period from that. Don't rely on vLLM to drain for you: confirm what your engine version does on SIGTERM before trusting it with long streams.

### Warm up before joining the router

A new replica should take its first real request only after it is warm (section 3, [warmup](#warmup-readiness-and-joining-the-router)). On Kubernetes that means the readiness probe checks something that becomes true only after warmup. A small startup wrapper is enough: start the engine, poll its `/health`, send warmup requests that cover your shapes, run the [output-quality probe](#output-quality-probes) below, then create the ready file. `minReadySeconds` on the Deployment adds a further soak: a pod must stay ready that long before the rollout counts it as available.

### Canaries gated on metrics

A canary's value is the comparison, so it must be automated and statistically meaningful. Together's [canary rollouts](https://together.ai/blog/canary-rollouts-upgrade-models-in-production-without-downtime) are a good public design:

- The default ladder is "5% → 25% → 50% → 100%". At each step: scale the target up, pass a health gate, shift traffic, wait, drain the source, wait an interval, check the metric gates.
- Gates compare the target's `router_error_rate` and `router_latency` with the source's (a relative regression check) or with fixed thresholds. A p95 gate needs at least 20 requests and a p99 gate at least 100, or the gate reports that metrics are unavailable.
- "There is no automatic abort: a confirmed regression always parks the rollout for a human." Rollback is a rollout in reverse.
- In their demo, an upgrade from Qwen2.5-7B-Instruct to Qwen3.5-9B raised p95 latency from 734 ms to 1,740 ms, 137% against a 25% budget. The rollout stopped at 10% of traffic, and the reverse rollout completed 5 min 49 s after it started.

Traffic splitting itself needs one detail right. If the split is by replica count, a canary with one replica and an old version with 15 gets 1/16 of traffic whatever percentage you set. Together's router [computes "each deployment's effective capacity as `weight × ready_replicas`"](https://together.ai/blog/configuring-dedicated-model-inference) and routes in proportion to it.

Baseten went the other way: ["Canary promotions are deprecated for models, and you can no longer turn them on"](https://docs.baseten.co/deployment/deployments), replaced by the rolling deployments above. A rolling deploy whose retirement tracks readiness, with a stabilization pause, is a canary with fewer knobs: the first batch of new replicas *is* the canary, and the pause is its soak time.

**On a leased fleet:** canaries are cheap here, since one canary replica is one node out of the existing fleet. The gates are what you build: error rate and latency compared with the old version, a minimum sample size, a pause for a human on regression, and the output-quality probes below.

### Output-quality probes

Error-rate and latency gates catch a replica that fails or slows down. They don't catch a replica that returns HTTP 200 with worse text. Anthropic's [postmortem of three issues in August-September 2025](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) describes exactly that. A misconfiguration deployed to TPU servers "caused an error during token generation", with symptoms including "producing Thai or Chinese characters in response to English prompts". A routing change sent some requests to servers configured for a different context window; "at the worst impacted hour on August 31, 16% of Sonnet 4 requests were affected". Standard evaluations missed both for weeks. Among the remediations: "We've added detection tests for unexpected character outputs to our deployment process", and quality evaluations that "we will run … continuously on true production systems".

So a new replica, and a canary, should pass a quality probe before it joins the router. The cheapest one has two parts: a fixed set of prompts decoded greedily and compared with golden outputs recorded from the known-good build, and a detector for the specific failure seen above. `probe.py` runs both on the CPU model, against a good build and a "bad build" whose sampler adds a bias to a block of 50,000 rare-script token ids:

```python
# probe.py - an output-quality probe for a new replica: fixed prompts, greedy decoding, compared
# with golden outputs recorded from the known-good version, plus a cheap "wrong script" detector.
#   /tmp/absorb-venv/bin/python probe.py
# The "bad deploy" adds a small bias to a block of rare token ids in the sampler: HTTP 200, fluent-ish,
# wrong. Health checks pass for both.
import json, os, unicodedata, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
PATH = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16).eval()
PROMPTS = ["The capital of France is", "def fibonacci(n):", "Water boils at a temperature of",
           "The three primary colors are", "In 1969, Neil Armstrong", "import numpy as np\nx = np.arange(",
           "The opposite of hot is", "A haiku about autumn:\n"]

def generate(prompt, bias=None):
    ids = tok(prompt, return_tensors="pt").input_ids                 # (1, T)
    for _ in range(16):
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()                # (V,)
        if bias is not None: logits += bias                          # the "sampler bug"
        ids = torch.cat([ids, logits.argmax().view(1, 1)], dim=1)
    return tok.decode(ids[0, -16:])

def foreign_fraction(text):                  # share of letters outside Latin script
    letters = [c for c in text if c.isalpha()]
    return sum("LATIN" not in unicodedata.name(c, "") for c in letters) / max(1, len(letters))

def probe(bias=None, golden=None):
    outs = [generate(p, bias) for p in PROMPTS]
    exact = sum(o == g for o, g in zip(outs, golden)) if golden else len(outs)
    worst = max(foreign_fraction(o) for o in outs)
    return outs, exact, worst

golden, _, _ = probe()                       # recorded once from the known-good build
bias = torch.zeros(model.config.vocab_size)
bias[100_000:150_000] = 6.0                  # mostly CJK/Thai/rare-script tokens in this vocabulary
for name, b in (("good build", None), ("bad build", bias)):
    outs, exact, worst = probe(b, golden)
    ok = exact >= len(PROMPTS) - 1 and worst < 0.05      # tolerate one numerics-level difference
    print(f"{name}: {exact}/{len(PROMPTS)} match golden, max foreign-script fraction {worst:.2f} -> "
          f"{'PASS' if ok else 'FAIL: keep it out of the router'}")
    bad = [(p, o) for p, o, g in zip(PROMPTS, outs, golden) if o != g]
    for p, o in bad: print(f"   {json.dumps(p)} -> {json.dumps(o, ensure_ascii=False)}")
```

```console
$ /tmp/drvenv/bin/python probe.py
good build: 8/8 match golden, max foreign-script fraction 0.00 -> PASS
bad build: 6/8 match golden, max foreign-script fraction 1.00 -> FAIL: keep it out of the router
   "In 1969, Neil Armstrong" -> " and Buzz Aldrin became the first humans to踏上地球表面的第一脚踏足迹"
   "A haiku about autumn:\n" -> "秋風吹拂葉々飘揚秋雨淅淅沥沥秋陽"
```

The bad build passes any health check, and 6 of its 8 answers are byte-identical to the good build's: the corruption only shows when a rare token wins, which is why it was hard to see in production. The two probes that caught it switched script mid-sentence. In practice:

- **Golden comparisons are exact only for the same kernels and batch shapes.** A new engine version or a different batch size changes floating-point reduction order and can change a greedy token ([determinism and batch invariance](../inference/serving-engine-internals.md#determinism-and-batch-invariance)). Compare per-token log-probabilities against a tolerance, or the KL divergence between old and new next-token distributions on the probe prompts, rather than strings, when the engine changes.
- **Probe with real distributions.** A few hundred prompts sampled (with consent and privacy controls) from your traffic, plus targeted checks: language and script of the answer matches the prompt, JSON parses, tool calls are well-formed, answers to a small eval set stay within their noise band.
- **Run the probe continuously, not only at deploy.** Several of Anthropic's issues came from changes outside the model (load balancing, a compiler), and they only reached some replicas. A probe that every replica answers every few minutes, compared fleet-wide, catches the one bad replica.

### Rollback

Rollback is a deploy, with every cost above: on a full fleet with $T_{\text{start}} = 5$ min, rolling 16 replicas back one at a time takes nearly two hours. Three things make it fast:

1. **Keep the old version's artifacts on every node** (image, weights, compile cache) until the new version has run cleanly for a day. Then a rollback replica starts at the fast-path $T_{\text{start}}$, not the cold one.
2. **Roll back with the capacity you have.** During an incident you can usually afford maxUnavailable well above the deploy-time setting.
3. **Keep a fraction of the old version running** through the first hours of a deploy, as Together's reverse rollout and blue-green both do. Shifting traffic back to replicas that are already warm takes seconds.


## 5. Recovery

### Detecting a sick GPU

GPUs fail more often than people expect. [Failure-rate arithmetic](../orchestration/internals.md#failure-rate-arithmetic) works it out from the Llama 3 training run: on the order of one interruption per 50,000 GPU-hours, so a 256-node fleet sees a node-level failure about once a day. The signals, from cheap to expensive:

- **Kernel log errors.** The NVIDIA driver reports faults as Xid lines in the kernel log ([Xid errors](../compute/gpu-software-stack.md#xid-errors-reading-finding-and-looking-them-up)). An agent on each node that tails the log and maps Xid numbers to actions reacts within seconds.
- **ECC and row remapping counters** ([ECC errors](../compute/gpu-software-stack.md#ecc-errors-and-row-remapping-in-nvidia-smi--q)). Together writes that ECC errors in GPU memory are ["the most common thing we see"](https://www.together.ai/blog/99-9-uptime-for-inference).
- **The engine itself.** A replica that crashes with "uncorrectable ECC error encountered" or fails `cuInit` is a GPU signal, not just an application crash.
- **Active diagnostics.** `dcgmi diag` levels 1-4 ([running diagnostics](../compute/gpu-software-stack.md#running-diagnostics-with-dcgmi-diag)). Modal notes that level 4 "takes around an hour", so at boot it runs only light checks (`systemctl` and `nvidia-smi` queries and a read/write on one randomly chosen GPU), and on instances it holds longer it runs weekly active checks: DCGM diag level 2, a burn test and a local NCCL all-reduce ([GPU health post](https://modal.com/blog/gpu-health)). Together points out the cost: "You can't exercise a GPU that's already serving traffic", so active checks run "in the gaps between workloads".

### Drain the whole host, or replace one GPU?

An 8-GPU node serving one TP8 replica loses the replica when any one GPU fails, so the question only arises for nodes with several smaller replicas. Modal's answer is blunt: on a failed check they ["mark the entire host unhealthy, drain it, and then either dispose of it or reinstall it"](https://modal.com/blog/gpu-health), and call per-GPU reset "overcomplicated and no guarantee of recovery". The reasons carry over to a leased fleet: a GPU fault often has a host-level cause (a PCIe switch, NVLink, power, the driver), a GPU reset needs every process on that GPU stopped anyway, and a node in a half-working state is a source of [gray failures](control-planes.md#slow-vs-dead-in-practice). Drain the node with the [cordon-and-evict](control-planes.md#draining) flow, move its replicas to a spare, and run full diagnostics on it out of the serving path.

The exception is a large node packed with many single-GPU replicas, where draining costs eight replicas for one bad GPU. There, cordon the node so nothing new lands, take only the affected replica out, and drain the rest at the next convenient moment. Don't restart the bad GPU's replica on the same node.

### Liveness vs readiness for an engine that takes minutes to start

The [probes section](../orchestration/kubernetes-from-zero.md#probes) of the Kubernetes chapter defines the three probe types. For a slow-starting engine:

- **Startup probe**: a budget of 2-3× your measured p99 $T_{\text{start}}$, so a slow start isn't killed halfway and restarted from zero (the most common LLM crash loop in [serving's failure modes](../inference/serving-hands-on.md#common-failure-modes)).
- **Readiness**: gated on warmup and the quality probe, and failed during drain. Readiness failing removes the replica from routing without killing it, which is what you want for "overloaded" and "draining".
- **Liveness**: only for "the process is wedged and only a restart helps". Keep it local and cheap: the engine's own `/health`, never a check that depends on another service. A liveness probe that fails fleet-wide when a dependency hiccups restarts every replica at once, and each restart costs a full $T_{\text{start}}$. That is how a 30-second outage of something else becomes a 10-minute outage of serving.

### MTTR arithmetic

**MTTR** (mean time to repair, here: to restore capacity) is the sum of detection, ejection from the router, and getting a replacement serving. `recovery.py` adds up the stages for one 8-GPU node serving a TP8 70B replica, under four setups, then turns failure rates into lost capacity and spare counts:

```python
# recovery.py - MTTR of a lost replica, and how many spare nodes a fixed fleet needs.
#   python3 recovery.py
import math

# ---- 1. MTTR: a sum of serial stages, for one 8-GPU node serving one TP8 replica
detect = {"probe (10 s period x 3 failures)": 30, "Xid/ECC watcher on the node": 5}
paths = {   # stage -> seconds after detection
    "no spare: wait for repair":       [("eject from router", 1), ("repair / RMA turnaround", 24 * 3600)],
    "spare node, cold (network disk, no caches)": [("eject from router", 1), ("schedule", 5), ("image pull", 130),
                                        ("weights 140 GB @ 0.32 GB/s", 438), ("engine init, no caches (assumed)", 180),
                                        ("warmup", 30)],
    "spare node, weights pre-staged":  [("eject from router", 1), ("schedule", 5), ("image cached", 0),
                                        ("weights 140 GB @ 12 GB/s NVMe", 12), ("engine init, cached compile (assumed)", 60),
                                        ("warmup", 30)],
    "N+1: a spare replica already serving": [("eject from router", 1)],
}
print("time to restore full capacity after one node dies")
for how, stages in paths.items():
    for dname, d in detect.items():
        total = d + sum(s for _, s in stages)
        print(f"  {how:42s} detect via {dname:34s} {total/60:8.1f} min")

# ---- 2. failure rate -> expected failures and lost replica-hours per year
mtbf_gpu_h = 50_000                          # per GPU incl. its share of host/network (Llama 3 order of magnitude)
mtbf_node_h = mtbf_gpu_h / 8                 # one GPU failing takes the 8-GPU node (and its replica) out
for nodes in (4, 32, 256):
    f_per_year = nodes * 8760 / mtbf_node_h
    print(f"\n{nodes:3d} nodes: {f_per_year:5.1f} node failures/year, one every {8760/f_per_year:6.0f} h")
    for how, stages in paths.items():
        mttr_h = (30 + sum(s for _, s in stages)) / 3600
        print(f"     {how:42s} lost replica-hours/year {f_per_year*mttr_h:8.1f}"
              f"  = {100*f_per_year*mttr_h/(nodes*8760):.3f}% of capacity")

# ---- 3. spares: nodes out for repair at once ~ Poisson(N * T_repair / MTBF_node)
def poisson_quantile(lam, q):
    p, k = 0.0, 0
    while True:
        p += math.exp(-lam) * lam ** k / math.factorial(k)
        if p >= q: return k
        k += 1
print("\nspare nodes so that a broken node can be replaced immediately 99.9% of the time")
print(f"{'nodes':>5} {'repair':>7} {'mean out':>8} {'spares':>6} {'share':>6}")
for nodes in (4, 32, 256):
    for repair_h in (24, 24 * 7):
        lam = nodes * repair_h / mtbf_node_h
        k = poisson_quantile(lam, 0.999)
        print(f"{nodes:5d} {repair_h:6d}h {lam:8.2f} {k:6d} {100*k/nodes:5.1f}%")

# ---- 4. flaky starts: retries are part of recovery
for p_fail in (0.001, 0.004):               # cuInit flake rate on one cloud; boot failure rate on another
    for k in (8, 64, 512):
        print(f"P(all {k:3d} starts succeed first try | p_fail={p_fail}) = {(1-p_fail)**k:.3f}")
```

```console
$ python3 recovery.py
time to restore full capacity after one node dies
  no spare: wait for repair                  detect via probe (10 s period x 3 failures)     1440.5 min
  no spare: wait for repair                  detect via Xid/ECC watcher on the node          1440.1 min
  spare node, cold (network disk, no caches) detect via probe (10 s period x 3 failures)       13.6 min
  spare node, cold (network disk, no caches) detect via Xid/ECC watcher on the node            13.2 min
  spare node, weights pre-staged             detect via probe (10 s period x 3 failures)        2.3 min
  spare node, weights pre-staged             detect via Xid/ECC watcher on the node             1.9 min
  N+1: a spare replica already serving       detect via probe (10 s period x 3 failures)        0.5 min
  N+1: a spare replica already serving       detect via Xid/ECC watcher on the node             0.1 min

  4 nodes:   5.6 node failures/year, one every   1562 h
     no spare: wait for repair                  lost replica-hours/year    134.6  = 0.384% of capacity
     spare node, cold (network disk, no caches) lost replica-hours/year      1.3  = 0.004% of capacity
     spare node, weights pre-staged             lost replica-hours/year      0.2  = 0.001% of capacity
     N+1: a spare replica already serving       lost replica-hours/year      0.0  = 0.000% of capacity

 32 nodes:  44.9 node failures/year, one every    195 h
     no spare: wait for repair                  lost replica-hours/year   1076.8  = 0.384% of capacity
     spare node, cold (network disk, no caches) lost replica-hours/year     10.1  = 0.004% of capacity
     spare node, weights pre-staged             lost replica-hours/year      1.7  = 0.001% of capacity
     N+1: a spare replica already serving       lost replica-hours/year      0.4  = 0.000% of capacity

256 nodes: 358.8 node failures/year, one every     24 h
     no spare: wait for repair                  lost replica-hours/year   8614.5  = 0.384% of capacity
     spare node, cold (network disk, no caches) lost replica-hours/year     81.1  = 0.004% of capacity
     spare node, weights pre-staged             lost replica-hours/year     13.8  = 0.001% of capacity
     N+1: a spare replica already serving       lost replica-hours/year      3.1  = 0.000% of capacity

spare nodes so that a broken node can be replaced immediately 99.9% of the time
nodes  repair mean out spares  share
    4     24h     0.02      1  25.0%
    4    168h     0.11      2  50.0%
   32     24h     0.12      2   6.2%
   32    168h     0.86      5  15.6%
  256     24h     0.98      5   2.0%
  256    168h     6.88     16   6.2%
P(all   8 starts succeed first try | p_fail=0.001) = 0.992
P(all  64 starts succeed first try | p_fail=0.001) = 0.938
P(all 512 starts succeed first try | p_fail=0.001) = 0.599
P(all   8 starts succeed first try | p_fail=0.004) = 0.968
P(all  64 starts succeed first try | p_fail=0.004) = 0.774
P(all 512 starts succeed first try | p_fail=0.004) = 0.128
```

What it says:

- **Without a spare, MTTR is the repair turnaround.** A day per failure is 0.38% of fleet capacity lost on average, at any fleet size, and it is lost in whole replicas at random times, including peak.
- **With a spare node, MTTR is $T_{\text{start}}$.** Cold, from network disk with no caches, about 13.5 minutes; with weights pre-staged on its NVMe and a compile cache, about 2 minutes. Detection is a small part either way; reading Xid events instead of waiting for three failed probes saves 25 s.
- **N+1 replicas make MTTR the ejection time.** If one more replica than peak demand needs is already serving, losing one costs nothing but the requests in flight on it. The router's [passive ejection](../inference/serving-hands-on.md#timeouts-retries-and-ejection) and the client's retry-before-first-byte handle the rest.
- **Spares come from a Poisson tail.** The number of nodes out for repair at once is roughly Poisson with mean $N \cdot T_{\text{repair}} / \text{MTBF}_{\text{node}}$ (the same model as the [hot spares](../orchestration/internals.md#hot-spares) section). For 32 nodes and a 24-hour repair turnaround, 2 spares cover 99.9% of the time; with week-long RMAs it is 5. Small fleets pay the most proportionally: 4 nodes need 1-2 spares, 25-50%. Correlated failures (a rack's power, a bad driver rollout) break the independence assumption, so treat these as floors.
- **Starts fail too, and retries are part of MTTR.** At a 0.1% per-start failure rate (Modal's L4 `cuInit` number), a 64-replica restart has a 6% chance that at least one replica needs a retry; at a 0.4% boot failure rate (Modal's Cloud A, where 99.6% of launches boot) a 512-node bring-up almost certainly has failures. Retry fast, on a different node.

**On a leased fleet:** this is where a spare node earns its keep twice. It turns MTTR from a repair turnaround into a $T_{\text{start}}$, and it is the surge capacity for [deploys](#surge-and-unavailability-on-a-full-fleet). Keep it running preemptible batch work so it isn't idle. Size it from the Poisson table, and take the fast path seriously: pre-staged weights and caches on the spare are what make its $T_{\text{start}}$ two minutes and not fifteen.


## 6. Autoscaling

An **autoscaler** is a reconciler whose desired replica count comes from a measurement: every few seconds it reads a load signal, computes how many replicas that load needs, and asks for that many ([reconciliation](control-planes.md#desired-state-and-reconciliation)). Three choices define one: the signal, the control law that turns the signal into a count, and the damping that keeps it from oscillating.

### The signal: concurrency and queues, not GPU utilization

The [orchestration chapter](../orchestration/internals.md#autoscale-on-queue-depth-and-kv-cache-use-not-cpu) explains why NVML's "GPU utilization" is the wrong signal: it is the fraction of time any kernel ran, so a decode-bound engine reads near 100% whether it holds 4 sequences or 200. GKE's [autoscaling guidance for LLM inference](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling) rejects it for the same reason (it "does not measure how much work is being done while the GPU is active"), rejects GPU memory because preallocating servers make it scale up but never down, and recommends **queue size** (start with a threshold of 3-5 waiting requests per replica and raise it until latency hits your target) or **batch size**.

Together ran the comparison on live-shaped traffic ([autoscaling post](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference)): Qwen3.5-9B, 1 to 3 replicas allowed, a sine wave of 12-48 requests per second with 80 req/s spikes, under three different scaling signals:

| Signal and target | Replicas | Replica-minutes | Requests | Errors |
|---|---|---|---|---|
| `inflight_requests` = 8 per replica | scaled 1 → 2 → 3 | 26 | 40.6k | 536 |
| `ttft` p95 = 300 ms | never scaled | 18 | 46.4k | 6 |
| `gpu_utilization` = 75% | never scaled | 18 | 46.5k | 2 |

Only in-flight requests reacted. The engine's continuous batching absorbed the extra load into longer per-request latency rather than higher TTFT, and the bursty load never pushed utilization past 75%. (The error and request counts differ between runs in ways the post doesn't fully explain; what matters is that two of the three signals never triggered.) This is why every managed platform scales on **in-flight requests per replica**, and why Baseten's docs say flatly that its ["Target utilization is not GPU utilization"](https://docs.baseten.co/deployment/autoscaling): it is the fraction of request slots in use.

KV-cache usage (`vllm:kv_cache_usage_perc`) is a useful second signal, because long prompts can fill the cache with few requests. llm-d's variant autoscaler uses both: a replica counts as saturated when [KV usage reaches 0.80 or the queue reaches 5](https://llm-d.ai/docs/0.7/architecture/advanced/autoscaling/wva), and it scales up when average spare KV falls below 0.10 or spare queue below 3.

### Control laws and their defaults

Nearly every system uses the same proportional law. Kubernetes' HPA [computes](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

$$
\text{desired} = \left\lceil \text{current} \times \frac{\text{observed metric}}{\text{target metric}} \right\rceil
$$

which, for a concurrency signal, is the same as $\lceil \text{total in-flight} / \text{target per replica} \rceil$. Baseten writes it as $\lceil \text{concurrent} / (\text{concurrency\_target} \times \text{target\_utilization}) \rceil$: with a target of 10 and 70% utilization, 25 concurrent requests need $\lceil 25/7 \rceil = 4$ replicas. The differences are in the target, the averaging window and the damping:

| System | Signal | Defaults | Damping |
|---|---|---|---|
| Kubernetes HPA | any metric via an adapter | 15 s sync period, 10% tolerance | scale-down stabilization 300 s; scale-up by the larger of +100% or +4 pods per 15 s |
| [KEDA](https://keda.sh/docs/latest/reference/scaledobject-spec/) | Prometheus queries and many others | `pollingInterval` 30 s | `cooldownPeriod` 300 s, for scaling to zero only; 1↔N is the HPA's |
| [Knative KPA](https://knative.dev/docs/serving/autoscaling/kpa-specific/) | in-flight requests per pod | [target 100 at 70% utilization](https://knative.dev/docs/serving/autoscaling/concurrency/) | 60 s stable window; a "panic" mode on a 6 s window (10% of stable) when load reaches 200% of capacity, during which it does not scale down |
| [Baseten](https://docs.baseten.co/deployment/autoscaling) | in-flight requests, streams count until they end | `concurrency_target` 1, `target_utilization_percentage` 70, 60 s averaging window | `scale_down_delay` 900 s; at most 50% of replicas removed per step |
| [Fireworks](https://docs.fireworks.ai/deployments/autoscaling) | load fraction, tokens/s, RPS or concurrent requests | scale-up window 30 s | scale-down window 10 min; scale-to-zero window 1 h |
| [Together](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference) | `inflight_requests` (and others) | target 8 per replica | scale-down window 300 s |
| [Modal](https://modal.com/docs/guide/concurrent-inputs) | inputs per container | `target_inputs`, with bursts up to `max_inputs` (e.g. 80 and 96, "a 20% burst") | [`scaledown_window`](https://modal.com/docs/guide/cold-start) 60 s by default, 2 s to 20 min |
| llm-d WVA | KV usage, queue length | KV 0.80, queue 5 | adds one replica at a time, to the cheapest hardware variant |

Three patterns run through the table:

- **Target below the hard limit.** Knative's and Baseten's 70%, Modal's `target_inputs` < `max_inputs`. The gap is headroom for the load that arrives while a new replica starts. A replica's hard limit should be the batch size at which it still meets its latency target, found by a [concurrency sweep](../inference/serving-hands-on.md#concurrency-sweep-finding-the-knee); Baseten's book says the concurrency target and the batch size ["should match"](https://www.baseten.co/inference-engineering/book/07-production/autoscaling/). Knative's default target of 100 concurrent requests per pod is far too high for an LLM replica and must be changed.
- **Fast up, slow down.** Scale-up windows are seconds to half a minute; scale-down windows are 5 to 15 minutes. Removing a replica is cheap, but getting it back costs $T_{\text{start}}$. Together's docs describe the failure of a short scale-down window as a "sawtooth": "the classic symptom of a scale-down window shorter than your traffic's natural rhythm". Baseten's book puts the cause plainly: "If you can't spin up replicas fast, it's hard to confidently scale down, leading to over-provisioning."
- **Scale-down is gradual.** Baseten removes at most half the replicas per step; the HPA's default policy and Knative's max scale-down rate do similar things.

A KEDA trigger for vLLM, on the same pattern as the [orchestration chapter's example](../orchestration/internals.md#autoscale-on-queue-depth-and-kv-cache-use-not-cpu) but on running plus waiting requests (**untested**):

```yaml
# untested; KEDA ScaledObject + HPA behavior passthrough. Target = 70% of a replica's 32 slots.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {name: llama-70b}
spec:
  scaleTargetRef: {name: llama-70b}
  minReplicaCount: 4
  maxReplicaCount: 16                     # this model's share of the fleet
  pollingInterval: 15
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:   {stabilizationWindowSeconds: 0}
        scaleDown:
          stabilizationWindowSeconds: 900
          policies: [{type: Percent, value: 25, periodSeconds: 300}]
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: sum(vllm:num_requests_running{model_name="llama-70b"}) + sum(vllm:num_requests_waiting{model_name="llama-70b"})
      threshold: "22"                     # per replica: ceil(32 * 0.7)
```

**On a leased fleet:** the control laws (HPA, KEDA, Knative's stable and panic windows, the managed platforms' targets and windows) apply unchanged; they are just arithmetic on a signal. What doesn't carry over is scaling to zero through Knative's activator or a vendor's wake-up path: on nodes you pay for anyway, keep a floor of replicas (`minReplicaCount`, Knative's min-scale) and hand surplus nodes to batch work instead.

### A discrete-time autoscaler simulator

How much does $T_{\text{start}}$ change what autoscaling can do? `autoscale_sim.py` runs six hours of bursty traffic against a pool of replicas. Each replica holds 16 concurrent sequences; a request holds a slot for a lognormal generation time (median 15 s); requests that find no free slot wait in a FIFO queue. Every 15 s the autoscaler computes $\lceil (\text{in flight} + \text{queued}) / (16 \times \text{target}) \rceil$, starts replicas that become ready $T_{\text{start}}$ later, and, after the load has stayed below capacity for a whole scale-down window, drains the least-loaded replicas. Starting and draining replicas are paid for. The model may use up to 40 replicas, its share of the fleet.

```python
# autoscale_sim.py - discrete-time autoscaler simulator: bursty arrivals, replicas that take
# STARTUP seconds to become ready, target-concurrency scaling, scale-down window, draining.
#   python3 autoscale_sim.py
# Pure stdlib. One step = 1 s. Everything is seeded, so the numbers repeat.
import heapq, math, random

SLOTS = 16            # max concurrent sequences per replica (engine max_num_seqs, sized for the latency SLO)
MAX_REPLICAS = 40     # the GPUs this model may use on the fixed fleet
MIN_REPLICAS = 2
EVAL_EVERY = 15       # autoscaler period, s (HPA's default sync period)
HOURS = 6

def trace(seed=1):
    """Arrivals per second: a slow 'diurnal' swell plus random bursts (a launch, a batch client)."""
    rng = random.Random(seed)
    T = HOURS * 3600
    base = [3.0 + 2.0 * math.sin(2 * math.pi * t / (4 * 3600)) for t in range(T)]   # 1..5 req/s
    t = 0
    while t < T:
        t += int(rng.expovariate(1 / 1500))                  # a burst every ~25 min on average
        dur, mult = rng.randint(120, 600), rng.uniform(2.0, 3.5)
        for u in range(t, min(T, t + dur)):
            base[u] *= mult
    return [rng.poisson(l) if hasattr(rng, "poisson") else poisson(rng, l) for l in base]

def poisson(rng, lam):
    k, p, L = 0, 1.0, math.exp(-lam)                         # Knuth; fine for lam < 30
    while True:
        p *= rng.random()
        if p <= L: return k
        k += 1

def simulate(arrivals, startup, window, target_util, seed=2, static=None, buffer=0):
    rng = random.Random(seed)
    # replica: dict(state='starting'|'ready'|'draining', ready_at, inflight)
    reps = [dict(state="ready", ready_at=0, inflight=0) for _ in range(static or min(MAX_REPLICAS, MIN_REPLICAS + buffer))]
    done = []                                                # heap of (finish_time, replica_index)
    queue = []                                               # arrival times of waiting requests (FIFO)
    waits, gpu_s, below_since = [], 0, None
    for t, n_new in enumerate(arrivals):
        while done and done[0][0] <= t:                      # completions free slots
            _, i = heapq.heappop(done); reps[i]["inflight"] -= 1
        for r in reps:                                       # starting replicas become ready
            if r["state"] == "starting" and t >= r["ready_at"]: r["state"] = "ready"
        queue.extend([t] * n_new)
        # admit FIFO into the least-loaded ready replica with a free slot
        ready = [i for i, r in enumerate(reps) if r["state"] == "ready"]
        qi = 0
        while qi < len(queue) and ready:
            i = min(ready, key=lambda j: reps[j]["inflight"])
            if reps[i]["inflight"] >= SLOTS: break
            reps[i]["inflight"] += 1
            waits.append(t - queue[qi]); qi += 1
            gen = max(2, int(rng.lognormvariate(math.log(15), 0.8)))   # generation time, median 15 s
            heapq.heappush(done, (t + gen, i))
        del queue[:qi]
        # drained replicas leave the fleet
        reps_alive = [r["state"] != "draining" or r["inflight"] > 0 for r in reps]
        if not all(reps_alive):
            keep = [i for i, a in enumerate(reps_alive) if a]
            remap = {old: new for new, old in enumerate(keep)}
            reps = [reps[i] for i in keep]
            done = [(f, remap[i]) for f, i in done]; heapq.heapify(done)
        gpu_s += len(reps)                                   # starting and draining GPUs are paid for too
        if static or t % EVAL_EVERY: continue
        # ---- the autoscaler: desired = ceil(load / (SLOTS * target_util)), load = in flight + queued
        load = sum(r["inflight"] for r in reps if r["state"] == "ready") + len(queue)
        desired = max(MIN_REPLICAS, min(MAX_REPLICAS, math.ceil(load / (SLOTS * target_util)) + buffer))  # buffer: warm spares above demand
        active = [r for r in reps if r["state"] in ("starting", "ready")]
        if desired > len(active):
            below_since = None
            for _ in range(desired - len(active)):
                reps.append(dict(state="starting", ready_at=t + startup, inflight=0))
        elif desired < len(active):
            below_since = t if below_since is None else below_since
            if t - below_since >= window:                    # held below for a whole window: scale down
                victims = sorted((r for r in active if r["state"] == "ready"), key=lambda r: r["inflight"])
                for r in victims[: len(active) - desired]:
                    r["state"] = "draining"                  # no new requests; in-flight ones finish
                below_since = None
        else:
            below_since = None
    waits.sort()
    p = lambda q: waits[min(len(waits) - 1, int(q * len(waits)))]
    return dict(p50=p(0.50), p95=p(0.95), p99=p(0.99), gpu_h=gpu_s / 3600, n=len(waits))

if __name__ == "__main__":
    arr = trace()
    print(f"{HOURS} h trace: {sum(arr):,} requests, mean {sum(arr)/len(arr):.2f} req/s, "
          f"peak 60 s mean {max(sum(arr[i:i+60]) for i in range(0, len(arr)-60, 10))/60:.2f} req/s")
    peak_static = simulate(arr, 0, 0, 1.0, static=MAX_REPLICAS)
    print(f"static {MAX_REPLICAS} replicas: p95 wait {peak_static['p95']} s, p99 {peak_static['p99']} s, "
          f"{peak_static['gpu_h']:.0f} GPU-h")
    print()
    print("target utilization 0.7 (Baseten/Knative default)")
    print(f"{'startup s':>9} {'window s':>8} {'p50':>4} {'p95':>5} {'p99':>5} {'GPU-h':>6}")
    for startup in (30, 120, 300, 600):
        for window in (60, 600):
            r = simulate(arr, startup, window, 0.7)
            print(f"{startup:9d} {window:8d} {r['p50']:4d} {r['p95']:5d} {r['p99']:5d} {r['gpu_h']:6.0f}")
    print()
    print("warm buffer needed: the fewest extra idle replicas (above target 0.7) for p95 queue wait <= 2 s")
    print(f"{'startup s':>9} {'buffer':>6} {'p95':>5} {'p99':>5} {'GPU-h':>6}  vs static {MAX_REPLICAS}")
    for startup in (30, 120, 300, 600):
        for buf in range(0, MAX_REPLICAS + 1, 2):
            r = simulate(arr, startup, 600, 0.7, buffer=buf)
            if r["p95"] <= 2: break
        print(f"{startup:9d} {buf:6d} {r['p95']:5d} {r['p99']:5d} {r['gpu_h']:6.0f}  "
              f"{100*r['gpu_h']/peak_static['gpu_h']:4.0f}%")
    a = simulate(arr, 120, 600, 0.7); b = simulate(arr, 120, 600, 0.7)
    assert a == b, "simulation must be deterministic"
```

```console
$ python3 autoscale_sim.py
6 h trace: 115,576 requests, mean 5.35 req/s, peak 60 s mean 31.23 req/s
static 40 replicas: p95 wait 0 s, p99 0 s, 240 GPU-h

target utilization 0.7 (Baseten/Knative default)
startup s window s  p50   p95   p99  GPU-h
       30       60    0     7    12     73
       30      600    0     6    11    100
      120       60    0    50    58     82
      120      600    0    39    48    120
      300       60   12   136   179    113
      300      600    0   127   136    126
      600       60  102   392   420    166
      600      600    0   378   418    139

warm buffer needed: the fewest extra idle replicas (above target 0.7) for p95 queue wait <= 2 s
startup s buffer   p95   p99  GPU-h  vs static 40
       30      4     2     6    120    50%
      120     18     1     7    187    78%
      300     20     2    50    195    81%
      600     30     1    17    231    96%
```

### Reading the simulator

- **Queue wait tracks startup time.** At a 70% target, p95 queue wait is 6-7 s with 30 s starts, 39-50 s with 2-minute starts, over 2 minutes with 5-minute starts and over 6 minutes with 10-minute starts. A burst that arrives faster than replicas can start waits for them; the target's 30% headroom absorbs only the first part of it.
- **The scale-down window trades GPU-hours for stability, and the trade flips with slow starts.** With 30 s starts, a 60 s window is cheapest (73 vs 100 GPU-hours) and barely worse on latency. With 10-minute starts the short window costs *more* GPU-hours (166 vs 139) and has a worse median: the pool scales down between bursts and pays a full start, billed while starting, for each new one. That is Together's sawtooth.
- **Startup time sets the over-provisioning.** The second table asks how many always-on extra replicas (a warm buffer, like Modal's `buffer_containers`) each startup time needs to keep p95 queue wait at 2 s or below. With 30 s starts, 4 extra replicas do it at 120 GPU-hours, half of what a static fleet sized for peak (40 replicas, 240 GPU-hours) costs. With 2-minute starts it takes 18, 78% of static. With 10-minute starts it takes 30, and the autoscaler saves 4% over not autoscaling at all. **Past a few minutes of startup, autoscaling for a latency target degenerates into static provisioning for the peak.** Even with the buffer, the 99th percentile stays at 17-50 s for 5- and 10-minute starts, against 6-7 s for the faster ones.

The simulator is deliberately simple: one model, one queue, uniform replicas, no prefix-cache affinity, and generation times that don't depend on load. Its job is to show the shape of the trade, not to size your fleet. To size yours, replace `trace()` with your own per-second arrival counts, set `SLOTS` from your concurrency sweep, and set `startup` from your measured p99 $T_{\text{start}}$, not the p50: Baseten's docs advise pre-warming "using your p99 startup time as the lead time".

### On a fixed fleet: move GPUs between models

On a fixed fleet, scaling one model up means scaling something else down: another model, or batch work. The total is constant, so the useful question is how fast and how often you can move a node from one role to another. Two effects compete. Modal's [capacity post](https://modal.com/blog/the-future-of-ai-needs-more-flexible-gpu-capacity) calls the gain demand pooling: provision for "the peak of the sum" rather than "the sum of the peaks". The cost is a swap: a node changing models serves nobody for one drain plus one $T_{\text{start}}$.

`realloc.py` simulates a week on a 32-node fleet shared by two models with offset daily peaks (a chat model peaking mid-day, a code model peaking late evening). A static 16/16 split is compared with moving nodes every 5 minutes to demand + 1, with three swap times, and with a variant that tolerates a surplus of up to 2 nodes before shrinking a model (hysteresis). Nodes assigned to neither model run preemptible batch work.

```python
# realloc.py - a fixed fleet shared by two models plus batch work: static split vs moving nodes.
#   python3 realloc.py
# One step = 5 min. Each node serves one replica of either model, or runs preemptible batch work.
# A node that changes role is useless for SWAP seconds (drain + load the other model).
import math, random

NODES, STEP, DAYS = 32, 300, 7
rng = random.Random(3)
steps = DAYS * 24 * 3600 // STEP
def wave(t, phase, lo, hi):                  # diurnal demand in replicas, peak at `phase` hours
    h = (t * STEP / 3600) % 24
    return lo + (hi - lo) * max(0.0, math.cos(2 * math.pi * (h - phase) / 24)) ** 2
chat = [max(1, round(wave(t, 14, 3, 22) * rng.uniform(0.85, 1.15))) for t in range(steps)]  # peaks mid-day
code = [max(1, round(wave(t, 22, 3, 18) * rng.uniform(0.85, 1.15))) for t in range(steps)]  # peaks late evening
print(f"{DAYS} days, {NODES} nodes. peak demand: chat {max(chat)}, code {max(code)}, "
      f"sum of peaks {max(chat)+max(code)}, peak of sum {max(c+k for c, k in zip(chat, code))}")

def run(policy, swap_s, headroom=1, hyst=0):
    alloc = {"chat": 16, "code": 16}         # nodes serving each model
    short = idle = free_h = moves = 0
    for c, k in zip(chat, code):
        dem = {"chat": c, "code": k}
        lost = 0.0
        if policy != "static":
            want = {m: min(NODES, dem[m] + headroom) for m in dem}
            if sum(want.values()) > NODES:               # not enough: scale both down in proportion
                f = NODES / sum(want.values()); want = {m: int(want[m] * f) for m in want}
            for m in dem:                                # shrink first (hysteresis: tolerate a small surplus)
                if want[m] < alloc[m] - hyst: alloc[m] = want[m]
            for m in dem:                                # then grow into free nodes (taken from batch)
                delta = min(want[m] - alloc[m], NODES - sum(alloc.values()))
                if delta > 0:
                    moves += delta; lost += delta * swap_s / STEP
                    alloc[m] += delta
        free = NODES - sum(alloc.values())
        short += sum(max(0, dem[m] - alloc[m]) for m in dem) + lost   # replica-steps of demand unserved
        idle += sum(max(0, alloc[m] - dem[m]) for m in dem)           # serving a model, but no demand for it
        free_h += free                                                 # not assigned: batch can use these
    h = STEP / 3600
    return short * h, idle * h, free_h * h, moves

print(f"{'policy':10s} {'swap':>6s} {'unserved replica-h':>18s} {'idle, assigned':>14s} {'free for batch':>14s} {'moves':>6s}")
for policy, swap, hyst in (("static", 0, 0), ("dynamic", 600, 0), ("dynamic", 120, 0), ("dynamic", 10, 0),
                           ("dyn+hyst2", 600, 2), ("dyn+hyst2", 120, 2)):
    s, i, f, m = run(policy, swap, hyst=hyst)
    print(f"{policy:10s} {swap:5d}s {s:18.1f} {i:14.0f} {f:14.0f} {m:6d}")
total = NODES * DAYS * 24
print(f"(fleet total {total} node-h)")
```

```console
$ python3 realloc.py
7 days, 32 nodes. peak demand: chat 25, code 21, sum of peaks 46, peak of sum 28
policy       swap unserved replica-h idle, assigned free for batch  moves
static         0s              158.7           3100              0      0
dynamic      600s              192.2            336           2606   1153
dynamic      120s               38.4            336           2606   1153
dynamic       10s                3.2            336           2606   1153
dyn+hyst2    600s              130.7            646           2296    784
dyn+hyst2    120s               26.1            646           2296    784
(fleet total 5376 node-h)
```

- **The static split both under-serves and wastes.** The chat model's peak (25 replicas) doesn't fit in its 16 nodes, so 159 replica-hours of demand go unserved, while 3,100 node-hours (58% of the fleet) sit assigned to a model with no demand for them.
- **Pooling helps only if swaps are fast.** The peak of the sum is 28 nodes, so moving nodes can serve everything. With 10 s swaps, which is sleep-mode territory, unserved demand drops to 3 replica-hours and 2,606 node-hours become free for batch work. With 10-minute swaps, moving nodes is *worse* than not moving them (192 vs 159 unserved): every move loses a node for two of the 5-minute steps, and the policy moves 1,153 times a week chasing noise.
- **Hysteresis is cheap and helps.** Tolerating a surplus of two nodes cuts moves by a third, and with 10-minute swaps it beats the static split (131 vs 159) while still freeing 2,296 node-hours for batch.

So the order is: make swaps fast (section 3: pre-staged weights and caches, sleep mode, LoRA adapters on a shared base), then reallocate with hysteresis, then fill the rest with work that can be evicted in seconds. Batch inference, evaluations, offline embedding jobs and RL rollouts all qualify ([RL infrastructure](rl-infrastructure.md)); they need to checkpoint or tolerate being killed, and the placement layer needs [preemption with priorities](control-planes.md#constraints-affinity-spreading-and-preemption).

**On a leased fleet:** llm-d's WVA does a version of this already: it [adds one replica to the cheapest variant](https://llm-d.ai/docs/0.7/architecture/advanced/autoscaling/wva) when a model saturates, so on a fleet with several GPU types it spends the cheapest free capacity first. The managed platforms' scale-down windows and buffers carry over unchanged; their scale-to-zero does not, because a node you don't use is still paid for, and the cheaper equivalent is to give it to batch work.


## 7. Bursting and serverless

### When serverless beats reserved

Sometimes demand exceeds the fleet: a launch, a customer's backfill job, an RL run that needs rollouts this week. Modal's [pricing post](https://modal.com/blog/how-to-price-serverless) gives the rule for whether to rent that capacity per second or reserve it. Reserved capacity costs $R_r$ per GPU-hour for the peak at all times; serverless costs $R_s$ per GPU-hour actually used. So

$$
\frac{C_{\text{reserved}}}{C_{\text{serverless}}} = \frac{R_r}{R_s} \times \frac{\text{peak}}{\text{average}}
$$

and serverless is cheaper when the peak-to-average ratio exceeds the reservation discount $R_s/R_r$. The post observes "discounts on the order of 2-5x", and peak-to-average ratios for inference, training and agentic development "in excess of 5:1, even 10:1".

The useful answer is usually neither: reserve a baseline $b$ and burst above it. For a demand trace, the cost-optimal $b$ follows from a one-line argument. Raising $b$ by one replica costs $R_r$ per hour, and saves $R_s$ per hour in the hours when demand is above $b$. So the optimum is where the fraction of time demand exceeds $b$ equals $R_r / R_s$. `burst.py` checks this against the simulator's trace:

```python
# burst.py - reserve a baseline, burst the rest: the cost-optimal split for a demand trace.
#   python3 burst.py        (reuses the arrival trace from autoscale_sim.py)
import math
from autoscale_sim import trace, SLOTS

arr = trace()
MEAN_GEN_S = 15 * math.exp(0.8 ** 2 / 2)       # mean of the lognormal generation time in the simulator
per_min = [sum(arr[i:i + 60]) / 60 for i in range(0, len(arr), 60)]
demand = [math.ceil(l * MEAN_GEN_S / (SLOTS * 0.7)) for l in per_min]   # Little's law: replicas busy per minute
peak, avg = max(demand), sum(demand) / len(demand)
print(f"replicas needed per minute: mean {avg:.1f}, peak {peak}, peak/avg {peak/avg:.2f}")

def cost(b, r_s):            # reserved price 1 per replica-minute; on-demand/serverless price r_s
    return b * len(demand) + r_s * sum(max(0, d - b) for d in demand)

print(f"{'burst price / reserved':>22} {'all reserved':>12} {'all burst':>9} {'best b':>6} {'hybrid':>7}  P(d > b*)")
for r_s in (1.5, 2, 3, 5):
    best = min(range(peak + 1), key=lambda b: cost(b, r_s))
    above = sum(d > best for d in demand) / len(demand)
    print(f"{r_s:22.1f} {cost(peak, r_s):12.0f} {cost(0, r_s):9.0f} {best:6d} {cost(best, r_s):7.0f}  {above:.2f} (1/r_s = {1/r_s:.2f})")
    # Modal's rule: reserved-for-peak / all-serverless = (R_r/R_s) * (peak/avg)
    assert abs(cost(peak, r_s) / cost(0, r_s) - (1 / r_s) * (peak / avg)) < 1e-9
```

```console
$ python3 burst.py
replicas needed per minute: mean 10.4, peak 56, peak/avg 5.39
burst price / reserved all reserved all burst best b  hybrid  P(d > b*)
                   1.5        20160      5610      6    4870  0.67 (1/r_s = 0.67)
                   2.0        20160      7480      8    5590  0.48 (1/r_s = 0.50)
                   3.0        20160     11220      9    6786  0.33 (1/r_s = 0.33)
                   5.0        20160     18700     11    8900  0.20 (1/r_s = 0.20)
```

The trace has a peak-to-average ratio of 5.4, so reserving for the peak loses to all-serverless even when serverless costs 5x as much per hour (20,160 vs 18,700), as Modal's ratio predicts, and the `assert` checks the formula. Reserving a baseline does better than either at every price ratio: at 3x, reserve 9 replicas (demand is above 9 a third of the time) and burst the rest, 40% cheaper than all-serverless and three times cheaper than reserving for the peak. The column `P(d > b*)` matches $1/r_s$, the rule above.

The model leaves out the thing this chapter is about: burst capacity has a cold start, paid at the moment of the burst. Use the [simulator](#a-discrete-time-autoscaler-simulator) with the burst provider's $T_{\text{start}}$ to see what the burst costs in queueing, and buy burst capacity with a short start, or start it ahead of predictable peaks.

**On a leased fleet:** your fleet *is* the reserved baseline, so this tells you whether its size is right (the fraction of time demand exceeds it should be near $R_r/R_s$) and how much to rely on burst capacity above it.

### Scale-to-zero, vendor by vendor

What happens to a request when a deployment has zero replicas differs a lot between vendors, and it decides whether scale-to-zero is usable for interactive traffic at all:

| Platform | Request to a deployment at zero | Source |
|---|---|---|
| Baseten | waits for a replica; "startup time becomes part of that request's latency". Startup minutes are billed | [cold starts](https://docs.baseten.co/deployment/autoscaling/cold-starts), [autoscaling](https://docs.baseten.co/deployment/autoscaling) |
| Fireworks | immediate HTTP 503 `DEPLOYMENT_SCALING_UP`; "Requests to a scaled-to-zero deployment are not queued", so the client retries. Deployments with min replicas 0 are deleted after 7 days without traffic | [autoscaling](https://docs.fireworks.ai/deployments/autoscaling) |
| Together | no automatic wake: "requests to a stopped deployment return an error rather than triggering a start" | [autoscaling post](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference) |
| Modal | "By default, Modal Functions will scale to zero" ([scaling docs](https://modal.com/docs/guide/scale)), and containers scale back after `scaledown_window` (60 s by default); `min_containers` sets a floor | [cold start](https://modal.com/docs/guide/cold-start) |
| Knative / KServe | the activator "is responsible for receiving and buffering requests for inactive revisions", so requests wait while the first pod starts | [Knative docs](https://knative.dev/docs/serving/load-balancing/target-burst-capacity/) |

With an LLM $T_{\text{start}}$ measured in minutes, a request that queues behind a scale-from-zero usually times out anyway, so scale-to-zero suits development deployments, low-traffic internal tools and batch jobs, and not interactive endpoints.

**On a leased fleet:** scale-to-zero saves nothing on nodes you've already paid for. Its equivalent is "sleep and hand the GPUs to batch", with a wake time of seconds.

### Spot plus on-demand

Spot (preemptible) cloud GPUs are cheaper and can be taken back at any time. SkyServe ([Mao et al., EuroSys 2025](https://arxiv.org/abs/2411.01438)) serves models on a mix: it spreads spot replicas across regions and clouds so that preemptions are less correlated, over-provisions spot replicas, and falls back to on-demand when spot is unavailable. The paper reports cost reduced "by 43% on average" compared with on-demand replicas. The [policy](https://docs.skypilot.ai/en/latest/serving/spot-policy.html) is a few lines of SkyPilot configuration (**untested**):

```yaml
# untested; SkyServe service spec
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 10
    target_qps_per_replica: 2.5          # SkyServe scales on QPS, a cruder signal than in-flight requests
    base_ondemand_fallback_replicas: 1   # always keep one on-demand replica
    dynamic_ondemand_fallback: true      # launch on-demand when spot is unavailable
resources:
  accelerators: H100:1
  use_spot: true
```

Every preemption is a recovery event with a full $T_{\text{start}}$, and spot replicas start on cloud nodes that have none of your caches. Everything in section 3 applies twice over.

**On a leased fleet:** this is a burst strategy for the capacity above your fleet. Keep the interactive baseline on your own nodes and send spillover and preemption-tolerant work to spot.


## 8. What infrastructure to build on

All of the above can run on very different stacks. Which one depends on how many nodes and models you have and what else runs on the cluster:

| Situation | Build on | Why |
|---|---|---|
| A few nodes, one to three models | Plain containers or systemd units running vLLM or SGLang, one per GPU group, behind a small router ([the gateway chapter](../inference/serving-hands-on.md#a-gateway-in-front-of-several-replicas)). Deploys are a script that drains and restarts one unit at a time | Nothing to operate. Independent replicas behind a simple router scale well: for INTELLECT-3, Prime Intellect found vLLM's built-in multi-node data parallelism "quickly plateaued", deployed each inference node as an independent server with one client per node, and saw that "inference throughput scales linearly with the number of nodes used" ([technical report](https://arxiv.org/html/2512.16144v1), §2.1.4) |
| Many nodes, many models, frequent deploys | Kubernetes: Deployments or LeaderWorkerSets, an inference gateway, KEDA or a variant autoscaler | Rolling updates, probes, draining, preemption and autoscaling are built in. llm-d, KServe and NVIDIA Dynamo are all built on it |
| Training and RL on the same cluster as serving | Slurm for training jobs, with vLLM launched inside jobs for rollouts and evals ([Slurm from zero](../orchestration/slurm/from-zero.md)); serving on a Kubernetes partition, or on nodes Slurm doesn't manage | Slurm schedules gangs and queues well. It has no rolling updates, readiness-gated routing or request-driven autoscaling, which makes it the wrong tool for a public API |
| Untrusted code (agent sandboxes, user-supplied code, RL environments) | gVisor or Firecracker microVMs for that part of the system only ([running untrusted code safely](../foundations/containers-and-virtualization.md#10-running-untrusted-code-safely)) | The isolation boundary is needed where the code is untrusted. Your own inference engine is trusted code, and runc is the fastest runtime for it |

**On a leased fleet:** the Prime Intellect result applies directly. Their inference nodes were their own, and the lesson (many independent replicas behind a simple router before one big distributed engine) costs nothing to adopt.

**Why Modal built its own runtime, and why you don't need to.** Modal runs every customer's container under gVisor and built checkpoint/restore, a lazy image filesystem and a GPU buffer pool on top. Two requirements drove that. First, multi-tenant security: strangers' code shares Modal's hosts, so a user-space kernel between them and the host kernel is worth its cost ([gVisor's trade-offs](../foundations/containers-and-virtualization.md#gvisor-a-kernel-in-user-space)). Second, scale-to-zero economics: Modal only earns when customers' containers run, so it has to start arbitrary containers in seconds from nothing, and owning the kernel is what let it snapshot and restore arbitrary process state. gVisor is the enabler of Modal's snapshots, not the source of their speed; the speed comes from restoring instead of re-initializing, which [CRIU with cuda-checkpoint](#process-and-gpu-snapshots) does on ordinary runc.

A leased fleet serving its own models has neither requirement. Its code is trusted, so runc is the right runtime. Its nodes are paid for whether busy or not, so it needs fast *swaps and recoveries*, not starts from zero, and the cheap tools for those are pre-pulled images, pre-staged weights, persistent compile caches, sleep mode and runtime LoRA.


## Startup-time checklist for a leased fleet

In the order to do them. Each saving is the published or measured number this chapter cites; yours will differ, so time your stages first and skip items whose stage is already small.

1. **Instrument the stages** (`startup_stages.py`, vLLM's startup log lines, a timestamp at ready and at the first routed token). Chart p50 and p99 per deploy. Without this the rest is guessing. Baseten's dashboard reports exactly this, per phase.
2. **Pre-pull serving images on every node** with `imagePullPolicy: IfNotPresent` ([pre-pulled images](https://kubernetes.io/docs/concepts/containers/images/#pre-pulled-images)). Saves the whole pull: 130-141 s for a 14 GB vLLM image in the Microsoft study.
3. **Pre-stage weights on local NVMe before every deploy**, with a `.complete` marker the rollout waits for. Takes weight time from storage speed to NVMe speed: 70B bf16 goes from 438 s (default loader on a network disk) to about 12 s at fal's 10-15 GB/s NVMe tier (`loadmath.py`). Keep the previous version's files for rollback.
4. **Use a parallel loader for anything not pre-staged**: `--load-format runai_streamer` with `--model-loader-extra-config '{"concurrency": 16}'`. 15 GB went from 47.99 s to 14.34 s on gp3 and to 4.88 s from S3 ([NVIDIA](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/)). Set `HF_XET_HIGH_PERFORMANCE=1` for Hub downloads and `HF_HUB_OFFLINE=1` on serving nodes.
5. **Persist and ship compile caches** per (model, engine version, GPU type): `VLLM_CACHE_ROOT` on local disk, built in CI, copied with the weights; `TORCHINDUCTOR_CACHE_DIR` for SGLang. Replicate measured 2-3x faster boots ([torch.compile caching](https://replicate.com/blog/torch-compile-caching)); fal's compile + warmup went from 202.9 s to 77.79 s with its cache; on CPU here, 9.4 s to 3.1 s.
6. **Capture fewer CUDA graph sizes** if your batch sizes are bounded (`--cudagraph-capture-sizes`, `--max-cudagraph-capture-size`; SGLang `--cuda-graph-max-bs-decode`). Graph capture time scales with the number of sizes captured. No published fleet number; measure before and after with the startup log.
7. **Gate readiness on warmup and a quality probe**, not on the port being open, and give the startup probe 2-3x your p99 start. Prevents slow first requests and the crash loop that restarts from zero; `probe.py` catches a sampler corruption that health checks pass.
8. **Drain properly**: SIGTERM handling that fails readiness and finishes streams, `preStop` sleep, `terminationGracePeriodSeconds` above your capped maximum generation time. No saving in startup time, but no cut streams in any deploy (`drain.py`).
9. **Keep spare nodes warm, with everything pre-staged.** Size from the Poisson table in `recovery.py` (2 spares for 32 nodes at a 24-hour repair turnaround). MTTR goes from the repair turnaround to about 2 minutes. Run preemptible batch work on them meanwhile.
10. **Swap models without restarting**: vLLM `--enable-sleep-mode` (wake from level 1 in 0.26-0.82 s for small models and "~3-6s for large models" against 37.6-58.1 s cold starts, per [vLLM](https://vllm.ai/blog/sleep-mode)), SGLang `--enable-memory-saver`, and runtime LoRA loading for fine-tunes of one base. This is what makes the reallocation in `realloc.py` pay off.
11. **Scale-down windows of 5-15 minutes and a warm buffer sized from your p99 start** (`autoscale_sim.py`). Prevents the sawtooth; the shorter your starts after items 2-10, the smaller the buffer.
12. **Only then consider process snapshots** (CRIU + cuda-checkpoint, Dynamo Snapshot) for single-GPU replicas that start often: 3-10x on small models ([Modal](https://modal.com/blog/gpu-mem-snapshots), [NVIDIA](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/)), much less on 70B-class models ([Parasail](https://www.parasail.io/blogs/cutting-cold-start-latency-snapshotting)). Free the KV cache before the dump and keep a normal-start fallback.


## You can now read

- [Operating inference](operating-inference.md): rate limits, tenancy, SLOs and cost, which assume the deploy, recovery and scaling machinery here.
- [RL infrastructure](rl-infrastructure.md), where rollout fleets sleep, wake and swap weights constantly, and [getting compute](getting-compute.md), where the reserved-vs-burst arithmetic decides what to lease.
- The vendor posts cited above, critically: you can now ask which stage a "10x faster cold start" removed, whether it applies to a model your size, and whether a leased fleet has that stage at all.
- The [llm-d](https://llm-d.ai), [KServe](https://kserve.github.io/website/) and [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) docs on autoscaling and snapshots, and the [Knative](https://knative.dev/docs/serving/autoscaling/) and [HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/) docs, as variations on one control law.

## Further reading

Papers:

- Fu et al., "ServerlessLLM: Low-Latency Serverless Inference for Large Language Models", OSDI 2024. Loading-optimized checkpoints and multi-tier caches.
- Mao et al., "SkyServe: Serving AI Models across Regions and Clouds with Spot Instances", EuroSys 2025, [arXiv:2411.01438](https://arxiv.org/abs/2411.01438).
- Kabakibo, Trivedi and Wang, ["Breaking the Ice: Analyzing Cold Start Latency in vLLM"](https://research.ibm.com/publications/breaking-the-ice-analyzing-cold-start-latency-in-vllm), MLSys 2026.
- Thompson et al., ["Seekable OCI: Lazy-Loading Container Images via Range-Request Indexing"](https://cdn.amazon.science/84/96/51ea2e0d40fca8cfaee9328278b9/soci-paper.pdf), AWS, 2026.
- Prime Intellect, ["INTELLECT-3: Technical Report"](https://arxiv.org/html/2512.16144v1), 2025. Section 2.1.4 on independent inference servers.

Engineering posts:

- Modal: ["How we achieved truly serverless GPUs"](https://modal.com/blog/truly-serverless-gpus), ["Fast, lazy container loading"](https://modal.com/blog/jono-containers-talk), ["Memory snapshots"](https://modal.com/blog/mem-snapshots), ["GPU memory snapshots"](https://modal.com/blog/gpu-mem-snapshots), ["Keeping 20,000 GPUs healthy"](https://modal.com/blog/gpu-health), ["How to price serverless GPUs"](https://modal.com/blog/how-to-price-serverless).
- Baseten: ["How the Baseten Delivery Network makes cold starts fast"](https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/); [autoscaling](https://docs.baseten.co/deployment/autoscaling), [cold start](https://docs.baseten.co/deployment/autoscaling/cold-starts) and [rolling deployment](https://docs.baseten.co/deployment/rolling-deployments) docs.
- Together: ["Autoscaling endpoints for LLM inference"](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference), ["Canary rollouts"](https://together.ai/blog/canary-rollouts-upgrade-models-in-production-without-downtime), ["What does 99.9% uptime mean for inference?"](https://www.together.ai/blog/99-9-uptime-for-inference).
- fal: ["Cut cold starts"](https://fal.ai/docs/examples/serverless/cut-cold-starts). Replicate: ["Torch compile caching for inference speed"](https://replicate.com/blog/torch-compile-caching).
- NVIDIA: ["Checkpointing CUDA applications with CRIU"](https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/), ["Dynamo Snapshot"](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/), ["Run:ai Model Streamer"](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/).
- Microsoft: ["Dissecting LLM container cold start"](https://techcommunity.microsoft.com/blog/linuxandopensourceblog/dissecting-llm-container-cold-start-where-the-time-actually-goes/4508831).
- vLLM: ["Sleep mode"](https://vllm.ai/blog/sleep-mode); docs on [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile.html), [optimization levels](https://docs.vllm.ai/en/latest/design/optimization_levels/) and [sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html).
- Anthropic: ["A postmortem of three recent issues"](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues), 2025.
- GKE: ["Best practices for autoscaling LLM inference workloads with GPUs"](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling).
