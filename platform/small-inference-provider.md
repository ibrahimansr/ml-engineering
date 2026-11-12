# Running a Small Inference Provider: A Worked Example

This chapter puts the whole book to work on one concrete business. A small company trains and serves models for other startups, who are its customers. Some customers want a fine-tune of an open model behind an API. Some want a base model at a lower price than the big providers charge. One or two want a model nobody else may touch. The company leases two 8-GPU NVIDIA B300 nodes from a GPU provider on a one-year contract: 16 GPUs, each with 288 GB of HBM. Nothing else. No third node, no second region, and no autoscaling into a cloud it already pays for.

Every mechanism used here was taught earlier, and every one is linked rather than re-derived. What this chapter adds is the arithmetic for *this* setup and the decisions that follow from it:

1. **Capacity.** What fits on a B300 and on a node, how much KV cache is left, roughly how many tokens per second each model gives, and a planner that proposes a placement of customer models on 16 GPUs.
2. **Packing.** Why LoRA fine-tunes on shared bases are the business, and full fine-tunes are an exception with a different price.
3. **The pipeline from a training checkpoint to a served model**, with a quality gate that compares a quantized model to its BF16 original on the customer's own evals.
4. **Training and serving on the same 16 GPUs**, and what each sharing policy buys over a synthetic day.
5. **Customer isolation**: caches, weights, logs, keys and noisy neighbours.
6. **Losing one of two nodes**, what each mitigation costs, and what uptime to put in a contract.
7. **The first 30 days**, step by step, each step linked to the section that teaches it.
8. **When to lease node 3.**

The three chapters this one leans on most each end in a checklist for a leased fleet: [the tuning checklist](../inference/serving-efficiently.md#tuning-checklist-for-a-leased-fleet), [the startup-time checklist](deploys-recovery-autoscaling.md#startup-time-checklist-for-a-leased-fleet) and [the operations checklist](operating-inference.md#operations-checklist-for-a-leased-fleet). They are written for fleets from a couple of nodes to a few dozen. Two nodes change the answers in specific ways: a node is half the fleet, a spare node doubles the bill, block granularity matters more than control laws, and the most valuable tool is a GPU-hour budget per customer. This chapter turns those checklists into one plan for this setup and points back to them rather than repeating them.

The company is fictional, and so are its customers ("kestrel", "rook", "sable", "wren"). Vendors appear only as sources of published numbers and practice.

## Prerequisites

- [Accelerators](../compute/accelerators.md): the [TFLOPS table](../compute/accelerators.md#tflops-comparison-table), the [measured MAMF and MSMF table](../compute/accelerators.md#mamf-and-msmf-comparison-table) (this chapter uses the book's own B300 measurements), the [memory table](../compute/accelerators.md#memory-comparison-table) and [GB vs GiB](../compute/accelerators.md#gb-gib-and-the-capacity-you-actually-get).
- [The GPU node software stack](../compute/gpu-software-stack.md): [MIG and MPS](../compute/gpu-software-stack.md#sharing-one-gpu-mig-and-mps), [nccl-tests](../compute/gpu-software-stack.md#nccl-tests-single-node-then-two-then-all), [`dcgmi diag`](../compute/gpu-software-stack.md#running-diagnostics-with-dcgmi-diag), the [bring-up runbook](../compute/gpu-software-stack.md#9-bring-up-order-the-runbook) and the [node acceptance script](../compute/gpu-software-stack.md#10-a-node-acceptance-script).
- [Training arithmetic](../training/arithmetic.md): [FP8 training](../training/arithmetic.md#fp8-training), [FP4 formats](../training/arithmetic.md#fp4-formats) and [optimizer-state memory](../training/arithmetic.md#optimizer-state-memory).
- [Fault tolerance](../training/fault-tolerance-and-reproducibility.md): [checkpoints](../training/fault-tolerance-and-reproducibility.md#checkpoints), [graceful stops](../training/fault-tolerance-and-reproducibility.md#graceful-stops-kill-and-save-switches) and [forced preemption](../training/fault-tolerance-and-reproducibility.md#dealing-with-forced-job-preemption).
- [Inference Internals I](../inference/arithmetic.md): [max concurrent sequences](../inference/arithmetic.md#max-concurrent-sequences), [the decode-latency bound](../inference/arithmetic.md#decode-latency-lower-bound), [TP communication in decode](../inference/arithmetic.md#tensor-parallel-communication-in-decode), [MoE arithmetic](../inference/arithmetic.md#moe-inference-arithmetic), [FP8 scaling](../inference/arithmetic.md#fp8-e4m3-and-its-scaling-granularities), [NVFP4 and MXFP4](../inference/arithmetic.md#nvfp4-and-mxfp4) and [what the bounds leave out](../inference/arithmetic.md#what-the-bounds-leave-out).
- [Inference Internals II](../inference/serving-engine-internals.md): [prefix caching](../inference/serving-engine-internals.md#prefix-caching) and [multi-LoRA serving](../inference/serving-engine-internals.md#multi-lora-serving).
- [Serving Efficiently on a Fixed Fleet](../inference/serving-efficiently.md): [TP vs DP vs EP](../inference/serving-efficiently.md#tp-vs-dp-vs-ep-the-layout-per-model-size), [FP8 weights and KV](../inference/serving-efficiently.md#fp8-weights-and-fp8-kv-cache), [packing many models](../inference/serving-efficiently.md#packing-many-models-onto-a-fixed-fleet) and [backfill](../inference/serving-efficiently.md#priority-filling-idle-capacity-with-batch-work).
- [Deploys, Recovery and Autoscaling](deploys-recovery-autoscaling.md): [the stages of a replica start](deploys-recovery-autoscaling.md#the-stages), [weights](deploys-recovery-autoscaling.md#weights), [rolling updates on a full fleet](deploys-recovery-autoscaling.md#surge-and-unavailability-on-a-full-fleet), [output-quality probes](deploys-recovery-autoscaling.md#output-quality-probes), [MTTR arithmetic](deploys-recovery-autoscaling.md#mttr-arithmetic), [moving GPUs between models](deploys-recovery-autoscaling.md#on-a-fixed-fleet-move-gpus-between-models) and [burst economics](deploys-recovery-autoscaling.md#when-serverless-beats-reserved).
- [Operating an Inference Service](operating-inference.md): [noisy neighbours](operating-inference.md#dedicated-vs-shared-deployments-noisy-neighbours), [SLOs per tier](operating-inference.md#slos-per-tier), [nines and capacity](operating-inference.md#what-999-vs-9999-means-and-the-capacity-it-needs), [batch](operating-inference.md#5-batch-and-offline-inference), [pricing](operating-inference.md#6-pricing-and-unit-economics), [continuous evals](operating-inference.md#continuous-evals-and-canary-prompts-in-production) and [security](operating-inference.md#9-security-basics-for-an-inference-api).
- [RL infrastructure](rl-infrastructure.md) (training jobs that share an inference fleet) and [Getting Compute](getting-compute.md) (leases, contracts, burn-in, cluster evaluation).

**What was run.** Every Python block with a file name in its first line was run on an Apple M4 laptop (macOS, CPython 3.14.5) from one scratch directory, `/tmp/sip`, and its output is pasted verbatim. `capacity.py`, `planner.py`, `lora_or_full.py`, `share_sim.py` and `failure_plan.py` use only the standard library; `planner.py` imports `capacity.py`, so keep them together. `quality_gate.py` ran in a virtualenv with PyTorch 2.15 (CPU) and `transformers` 5.22.2 on `Qwen/Qwen2.5-0.5B-Instruct` from the local Hugging Face cache; its quantized candidates are **simulated** (weights rounded to the FP8 or NVFP4 grid and kept in fp32), which is said again where it matters. From its output, only the weight-loading progress bars and one PyTorch platform warning were removed. No GPU was available. **Every `vllm`, `nvidia-smi`, `llmcompressor` and shell command is untested.** Flag and API names were checked against current docs and source in November 2026. The capacity numbers are **roofline bounds** from the book's measured B300 numbers, not benchmarks: a real engine reaches some fraction of them, and the planner assumes half. Prices were read on the linked pages in November 2026. The lease price (\$4.00 per GPU-hour) is an **assumption**, stated wherever it is used.

## 1. The hardware, confirmed

Before any arithmetic, pin down what a B300 is. Three sources agree, and one detail needs care.

| Quantity | Value | Source |
|---|---|---|
| HBM per GPU | 288 GB HBM3e | [memory table](../compute/accelerators.md#memory-comparison-table) |
| HBM bandwidth per GPU | 8.0 TB/s | [memory table](../compute/accelerators.md#memory-comparison-table); NVIDIA's [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) doesn't list it |
| Dense BF16 / FP8 / NVFP4, spec | 2,250 / 4,500 / 13,500 TFLOPS | [TFLOPS table](../compute/accelerators.md#tflops-comparison-table), from NVIDIA's 108 PFLOPS dense FP4 per 8 GPUs |
| Measured MSMF BF16 / FP8 / NVFP4 | 1,519 / 2,969 / 7,434 TFLOPS | [MAMF and MSMF table](../compute/accelerators.md#mamf-and-msmf-comparison-table) |
| Measured MAMF BF16 / FP8 / NVFP4 | 1,892 / 3,608 / 8,909 TFLOPS | same |
| NVLink per GPU | 1.8 TB/s; 14.4 TB/s per 8-GPU board | [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) |
| Total GPU memory per 8-GPU system | "2.1 TB" | [HGX page](https://www.nvidia.com/en-us/data-center/hgx/), [DGX B300 page](https://www.nvidia.com/en-us/data-center/dgx-b300/) |
| Networking (DGX B300) | 8 single-port ConnectX-8, "Up to 800 Gb/s" each | [DGX B300 page](https://www.nvidia.com/en-us/data-center/dgx-b300/) |
| Local NVMe (DGX B300) | "8x 3.84 TB NVMe E1.S" | same |
| System power (DGX B300) | "~14 kW" | same |

The detail is the memory size. "288 GB" and "2.1 TB per 8 GPUs" disagree unless you read the units: 288 GB is $288 \times 10^9$ bytes, which is 268 GiB, and $8 \times 268 = 2{,}146$ GiB, about 2.1 TiB. NVIDIA's MIG profile for the whole GPU is called `7g.269gb`, which matches the GiB reading ([MIG user guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)). The scripts below use $288 \times 10^9$ bytes and vLLM's default `gpu_memory_utilization` of 0.9, so 259 GB per GPU is usable for weights and KV. Check `nvidia-smi --query-gpu=memory.total --format=csv` on your nodes on day one; that is the number that counts.

Three more things to note about the measured rows:

- **MSMF, not the spec, is the compute number to plan with.** The B300's sustained BF16 matmul rate is 67.5% of the spec ([MSMF %](../compute/accelerators.md#mamf-and-msmf-comparison-table)), because a saturated GPU drops its clock to stay within its power limit ([why MSMF is below the spec](../compute/accelerators.md#why-msmf-is-well-below-the-spec-and-mamf-above-msmf)).
- **The BF16 and FP8 rows were measured with the node's other GPUs idle** (`Siblings = no`), which can read higher than a busy node sustains. The NVFP4 row was measured with siblings busy. Re-measure on your own nodes during bring-up ([section 8](#8-the-first-30-days)).
- **NVFP4 needs the right kernels.** The book's NVFP4 MSMF of 7,434 needs cuBLAS 13.4 (torch 2.14.1+cu132); with cuBLAS 13.1 the same GPU reached 6,410, because the older library picks a kernel that doesn't use B300's faster FP4 path ([note 19](../compute/accelerators.md#mamf-and-msmf-comparison-table)). Your serving engine's NVFP4 kernels have the same dependency.

**MIG on B300.** NVIDIA documents these profiles for the B300 288 GB product ([MIG user guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)):

| Profile | Compute | Memory | Max instances per GPU |
|---|---|---|---|
| `1g.34gb` | 1/7 | 1/8 | 7 |
| `1g.67gb` | 1/7 | 2/8 | 4 |
| `2g.67gb` | 2/7 | 2/8 | 3 |
| `3g.135gb` | 3/7 | 4/8 | 2 |
| `4g.135gb` | 4/7 | 4/8 | 1 |
| `7g.269gb` | 7/7 | 8/8 | 1 |

(plus a `1g.34gb+me` variant with media engines). The guide lists these for "the B300 288 GB product" without naming the HGX board, so confirm with `nvidia-smi mig -lgip` on your nodes before you plan around them. Two things the guide doesn't say, and which the planner below assumes: that a slice's HBM bandwidth scales with its share of memory (1/8 of 8 TB/s for a `1g.34gb`), and that any combination whose compute and memory fractions fit is placeable. Real MIG placement has fixed start positions, so some combinations that fit by arithmetic aren't allowed; `nvidia-smi mig -lgipp` lists the valid placements.

**The lease.** Sixteen B300s at an assumed \$4.00 per GPU-hour on a one-year contract cost $16 \times 4.00 \times 730 = \$46{,}720$ a month, \$560,640 a year. For scale, the list prices for renting a B300 by the hour, read in November 2026, are \$6.64 on demand at [Prime Intellect](https://www.primeintellect.ai/compute), \$0.001972 per second (\$7.10 per hour) on [Modal](https://modal.com/pricing), and \$11.49 for a dedicated endpoint at [Together](https://www.together.ai/pricing). Those three numbers come back in [section 7](#7-planning-for-the-loss-of-a-node), where they price the alternatives to a third node.

## 2. Capacity: what fits on 16 B300s

### Memory budgets per GPU and per node

The per-GPU budget is the formula from [max concurrent sequences](../inference/arithmetic.md#max-concurrent-sequences): usable HBM, minus this GPU's share of the weights, is what the KV cache gets. The calculator below applies it to four representative customer models in BF16, FP8 and NVFP4, picks the smallest tensor-parallel degree that leaves room for at least 128k tokens of KV, and then computes roofline decode and prefill rates the way [the decode-latency bound](../inference/arithmetic.md#decode-latency-lower-bound) does. The four models are Llama-3.1-8B and Llama-3.3-70B (configs as in [Internals I](../inference/arithmetic.md#real-configs)), GLM-4.5-Air as the ~100B-class MoE (106B total, 12B active per the [model card](https://huggingface.co/zai-org/GLM-4.5-Air), MIT license; 46 layers, 8 KV heads of 128, 128 routed experts with 8 active, from its `config.json`), and DeepSeek-V3 (671B, 37B active, MLA). Two small Qwen2.5 models (1.5B and 3B, from their `config.json` files) are in `MODELS` for the planner but not printed.

Four modelling choices, all visible in the code:

- **Not everything gets quantized.** Embeddings and the LM head stay in BF16, as in the [llm-compressor examples](https://github.com/vllm-project/llm-compressor/tree/main/examples) (`ignore=["lm_head"]`). For Llama-3.3-70B that is 2.1B of the 70.55B parameters.
- **NVFP4 costs 4.5 bits per weight**: a 4-bit value plus one FP8 scale per 16 values, as NVIDIA [describes it](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) (the per-tensor FP32 scale is negligible). The bits-per-weight arithmetic is in [Internals I](../inference/arithmetic.md#bits-per-weight-including-metadata).
- **KV is BF16 with BF16 weights and FP8 otherwise**, the [recommended pairing](../inference/serving-efficiently.md#fp8-weights-and-fp8-kv-cache). With tensor parallelism above the number of KV heads, the KV is replicated. DeepSeek-V3 is modelled with data-parallel attention, so each rank holds its own sequences' latents ([why](../inference/arithmetic.md#max-concurrent-sequences)).
- **The all-reduce floor uses the book's 8x B200 measurement**, 23.5 µs per small all-reduce ([TP communication](../inference/arithmetic.md#tensor-parallel-communication-in-decode)). Nobody has measured it on B300 for this book; B300 has the same NVLink generation, but measure it during bring-up.

```python
# capacity.py - what fits on a B300 and on an 8x B300 node, KV room, and roofline decode/prefill rates.
#   python3 capacity.py          (planner.py imports the tables and functions below)
# Hardware numbers are the book's: compute/accelerators.md spec + measured MSMF. Bounds, not benchmarks.
import math

HBM = 288e9            # bytes per GPU (memory table); MIG's full-GPU profile is 7g.269gb, i.e. ~268 GiB
HBM_BW = 8.0e12        # bytes/s per GPU (memory table)
MSMF = {"bf16": 1519e12, "fp8": 2969e12, "nvfp4": 7434e12}   # measured sustained matmul FLOP/s
ALPHA = 23.5e-6        # s per small all-reduce: measured on 8x B200 (Internals I). B300 not measured.
UTIL = 0.90            # vLLM's default gpu_memory_utilization
BYTES_W = {"bf16": 2.0, "fp8": 1.0, "nvfp4": 4.5 / 8}   # NVFP4: 4-bit value + one FP8 scale per 16 values
BYTES_KV = {"bf16": 2, "fp8": 1, "nvfp4": 1}            # BF16 KV with BF16 weights, FP8 KV otherwise

# N: total params; N16: params left in BF16 (embeddings, lm_head); Nact: params per token;
# Nexp: routed-expert params; E, k: experts and experts per token; L: layers;
# kv: KV elements per token per layer (2 * kv_heads * head_dim, or 576 for DeepSeek's MLA latent);
# nkv: KV heads (TP above this replicates KV); af: attention FLOPs per decoded token per context token per layer;
# lin: sum of (in + out) dims of the 7 linear layers per block, for LoRA sizing (0: not used with LoRA here)
MODELS = {
    "qwen-1.5b":   dict(N=1.54e9, N16=0.233e9, Nact=1.54e9, Nexp=0, E=1, k=1, L=28, kv=2*2*128, nkv=2, af=4*12*128, lin=41216),
    "qwen-3b":     dict(N=3.09e9, N16=0.311e9, Nact=3.09e9, Nexp=0, E=1, k=1, L=36, kv=2*2*128, nkv=2, af=4*16*128, lin=51968),
    "llama-8b":    dict(N=8.03e9, N16=1.051e9, Nact=8.03e9, Nexp=0, E=1, k=1, L=32, kv=2*8*128, nkv=8, af=4*32*128, lin=81920),
    "llama-70b":   dict(N=70.55e9, N16=2.101e9, Nact=70.55e9, Nexp=0, E=1, k=1, L=80, kv=2*8*128, nkv=8, af=4*64*128, lin=161792),
    "glm-air-106b": dict(N=106e9, N16=1.241e9, Nact=12e9, Nexp=99.65e9, E=128, k=8, L=46, kv=2*8*128, nkv=8, af=4*96*128, lin=0),
    "dsv3-671b":   dict(N=671e9, N16=1.853e9, Nact=37e9, Nexp=653.9e9, E=256, k=8, L=61, kv=576, nkv=None, af=2*128*1088, lin=0),
}

def weights_per_gpu(m, dt, P):                       # bytes
    return ((m["N"] - m["N16"]) * BYTES_W[dt] + m["N16"] * 2) / P

def kv_per_token_per_gpu(m, dt, P):                  # bytes of KV one token costs on each GPU of the replica
    total = m["L"] * m["kv"] * BYTES_KV[dt]
    if m["nkv"] is None:                             # MLA with data-parallel attention: each rank holds its own seqs
        return total / P
    return total / min(P, m["nkv"])                  # TP shards KV heads; past nkv ranks the KV is replicated

def kv_tokens(m, dt, P, hbm=HBM, extra=0.0):         # tokens of KV the replica can hold; extra = other bytes/GPU
    budget = hbm * UTIL - weights_per_gpu(m, dt, P) - extra
    return max(0.0, budget) / kv_per_token_per_gpu(m, dt, P)

def step_time(m, dt, P, B, T, bw=1.0, fl=1.0):      # one decode step, B seqs, T tokens of context each
    # bw, fl: fraction of the GPU's HBM bandwidth and FLOPs the replica gets (1.0, or a MIG slice)
    touched = m["E"] * (1 - (1 - m["k"] / m["E"]) ** B) / m["E"] if m["Nexp"] else 0.0
    wbytes = ((m["N"] - m["Nexp"] - m["N16"]) * BYTES_W[dt] + m["N16"] * 2 + m["Nexp"] * touched * BYTES_W[dt]) / P
    t_mem = (wbytes + B * T * kv_per_token_per_gpu(m, dt, P)) / (HBM_BW * bw)
    t_cmp = (2 * m["Nact"] * B + m["L"] * m["af"] * T * B) / (P * MSMF[dt] * fl)
    return max(t_mem, t_cmp) + (2 * m["L"] * ALPHA if P > 1 else 0.0)

def prefill_tok_s(m, dt, P, T, fl=1.0):              # compute-bound prefill, average context T/2
    return P * MSMF[dt] * fl / (2 * m["Nact"] + m["L"] * m["af"] * T / 2)

def smallest_tp(m, dt, min_tokens=128_000):
    for P in (1, 2, 4, 8, 16):
        if kv_tokens(m, dt, P) >= min_tokens:
            return P
    return None

if __name__ == "__main__":
    T = 8192
    print(f"B300: {HBM/1e9:.0f} GB, {UTIL:.0%} usable = {HBM*UTIL/1e9:.0f} GB/GPU, {HBM_BW/1e12:.0f} TB/s; "
          f"node = 8 GPUs = {8*HBM*UTIL/1e9:.0f} GB usable")
    print(f"{'model':13s} {'dtype':5s} {'weights':>8s} {'TP':>3s} {'W/GPU':>6s} {'KV/GPU':>6s} {'KV tok':>7s} "
          f"{'@8k':>5s} {'@32k':>5s} {'B':>4s} {'TPOT':>6s} {'dec tok/s/GPU':>13s} {'pre tok/s/GPU':>13s}")
    for name, m in MODELS.items():
        if name.startswith("qwen"):
            continue
        for dt in ("bf16", "fp8", "nvfp4"):
            P = smallest_tp(m, dt)
            wtot = weights_per_gpu(m, dt, 1) / 1e9
            if P is None:
                print(f"{name:13s} {dt:5s} {wtot:7.0f}G  does not fit on 16 GPUs"); continue
            tok = kv_tokens(m, dt, P)
            B = int(min(tok / T, 256))
            st = step_time(m, dt, P, B, T)
            print(f"{name:13s} {dt:5s} {wtot:7.0f}G {P:3d} {weights_per_gpu(m, dt, P)/1e9:5.0f}G "
                  f"{(HBM*UTIL - weights_per_gpu(m, dt, P))/1e9:5.0f}G {tok/1e6:6.2f}M {tok/8192:5.0f} {tok/32768:5.0f} "
                  f"{B:4d} {st*1e3:5.1f}ms {B/st/P:13,.0f} {prefill_tok_s(m, dt, P, T)/P:13,.0f}")
    # the same 70B on more GPUs than it needs: KV room grows, but so does the all-reduce floor
    m = MODELS["llama-70b"]
    print("\nllama-70b fp8, B=64 at 8k context, by TP degree")
    for P in (1, 2, 4, 8):
        st = step_time(m, "fp8", P, 64, T)
        print(f"  TP{P}: KV {kv_tokens(m, 'fp8', P)/1e6:5.2f}M tok, step {st*1e3:5.2f} ms "
              f"(all-reduce floor {2*m['L']*ALPHA*1e3 if P > 1 else 0:.2f} ms), {64/st/P:6,.0f} tok/s/GPU")
    assert smallest_tp(MODELS["llama-8b"], "bf16") == 1 and smallest_tp(MODELS["dsv3-671b"], "bf16") == 8
```

```console
$ python3 capacity.py
B300: 288 GB, 90% usable = 259 GB/GPU, 8 TB/s; node = 8 GPUs = 2074 GB usable
model         dtype  weights  TP  W/GPU KV/GPU  KV tok   @8k  @32k    B   TPOT dec tok/s/GPU pre tok/s/GPU
llama-8b      bf16       16G   1    16G   243G   1.86M   226    57  226  32.3ms         6,988        83,427
llama-8b      fp8         9G   1     9G   250G   3.82M   466   116  256  18.3ms        13,978       163,065
llama-8b      nvfp4       6G   1     6G   253G   3.86M   472   118  256  17.9ms        14,275       408,294
llama-70b     bf16      141G   1   141G   118G   0.36M    44    11   43  32.1ms         1,341        10,004
llama-70b     fp8        73G   1    73G   187G   1.14M   139    35  138  32.2ms         4,281        19,554
llama-70b     nvfp4      43G   1    43G   216G   1.32M   161    40  161  32.3ms         4,977        48,960
glm-air-106b  bf16      212G   1   212G    47G   0.25M    31     8   30  28.7ms         1,046        45,669
glm-air-106b  fp8       107G   1   107G   152G   1.61M   197    49  196  32.3ms         6,066        89,264
glm-air-106b  nvfp4      61G   1    61G   198G   2.10M   256    64  256  32.4ms         7,908       223,505
dsv3-671b     bf16     1342G   8   168G    91G  10.41M  1271   318  256  26.1ms         1,225        10,579
dsv3-671b     fp8       673G   4   168G    91G  10.36M  1264   316  256  26.2ms         2,444        20,677
dsv3-671b     nvfp4     380G   2   190G    69G   3.94M   480   120  256  31.2ms         4,100        51,772

llama-70b fp8, B=64 at 8k context, by TP degree
  TP1: KV  1.14M tok, step 19.82 ms (all-reduce floor 0.00 ms),  3,229 tok/s/GPU
  TP2: KV  2.72M tok, step 13.67 ms (all-reduce floor 3.76 ms),  2,341 tok/s/GPU
  TP4: KV  5.88M tok, step  8.71 ms (all-reduce floor 3.76 ms),  1,836 tok/s/GPU
  TP8: KV 12.21M tok, step  6.24 ms (all-reduce floor 3.76 ms),  1,283 tok/s/GPU
```

Columns: `weights` is the whole model; `TP` the smallest tensor-parallel degree that leaves 128k tokens of KV; `W/GPU` and `KV/GPU` the per-GPU split of the 259 GB; `KV tok` the tokens of KV the replica holds; `@8k` and `@32k` how many full-length sequences that is; `B` the batch used for the rate (KV-limited at 8k, capped at 256); `TPOT` the bound on time per output token at that batch; the last two columns are bounds per GPU.

What this says about the hardware the company leased:

- **A 70B dense model in BF16 fits on one GPU.** 141 GB of weights leaves 118 GB of KV, 44 sequences of 8k. On an 80 GB H100 the same model needs TP4 for weights alone. On B300 the question for a 70B model is never whether it fits but how much KV you want: FP8 gives 1.14M tokens on one GPU, 3.2x the BF16 figure, because it halves both the weights and every token of KV.
- **The ~100B MoE fits on one GPU in FP8** (107 GB) and is barely usable in BF16 (47 GB of KV, 31 sequences of 8k). GLM-4.5-Air in NVFP4 on one B300 holds 2.1M tokens of KV.
- **The 671B MoE takes half a node in FP8.** 673 GB on 4 GPUs, 168 GB each. In BF16 it takes a whole node; in NVFP4, two GPUs. A customer who asks for DeepSeek-V3 is asking for a quarter of the fleet in FP8 per replica, or half the fleet for two replicas on different nodes. [Section 3](#3-packing-many-customers-lora-first) comes back to that.
- **Replicas beat bigger TP on throughput per GPU**, as [Serving Efficiently](../inference/serving-efficiently.md#tp-vs-dp-vs-ep-the-layout-per-model-size) argued for H100s. The second table makes it concrete: for the 70B FP8 model at batch 64, TP1 gets 3,229 tokens/s per GPU and TP8 gets 1,283. TP8's step is three times faster (6.24 vs 19.82 ms), but 3.76 ms of it is the all-reduce floor ($2L \times 23.5\ \mu\text{s} = 160 \times 23.5\ \mu\text{s}$), and it uses eight times the GPUs. On B300, TP is a latency tool, and a KV-capacity tool for long contexts, not a throughput tool.
- **NVFP4 barely moves decode for dense models at this batch and context.** 4,977 vs 4,281 tokens/s per GPU for the 70B model: at 8k context, the KV reads dominate the step, and NVFP4 shrinks only the weights. It moves prefill a lot (48,960 vs 19,554 tokens/s per GPU), because prefill is compute-bound and B300's FP4 MSMF is 2.5x its FP8 MSMF. NVFP4 KV, which the Prime Intellect result in the [tuning checklist](../inference/serving-efficiently.md#tuning-checklist-for-a-leased-fleet) used, would be the lever for decode.

### KV room and concurrency at typical contexts

The `@8k` and `@32k` columns are the ceiling on concurrent full-length sequences per replica. Agents and coding assistants run long contexts, so it is worth reading the 32k column as the planning number. A 70B FP8 replica holds 35 sequences of 32k on one GPU. That is the one place where TP2 makes sense for a 70B model on B300: 2.72M tokens of KV instead of 1.14M, 2.4x more (each GPU holds half the weights and half the KV heads), at 2,341 rather than 3,229 tokens/s per GPU. If a customer's sessions are KV-bound, [KV capacity is the bottleneck](../inference/serving-efficiently.md#kv-capacity-is-the-bottleneck-for-agentic-workloads) and TP2 can win on goodput despite losing on the roofline. The planner below does the trade-off per deployment, from that deployment's context length.

### Rough tokens per second

The rates are bounds. [What the bounds leave out](../inference/arithmetic.md#what-the-bounds-leave-out) lists why a real engine reaches less: kernel launch gaps, achieved HBM bandwidth below spec, host scheduling, prefill interference. It also says what "good" is: above about 70% of the HBM roofline for decode. The planner below assumes 50%, a conservative number for a team that hasn't tuned yet. Replace it with your measured fraction as soon as you have one ([achieved.py](../inference/arithmetic.md#what-the-bounds-leave-out) backs it out of a measured TPOT).

With that derate, a rough fleet-wide figure: 16 GPUs serving Llama-3.3-70B in FP8 at batch 64, TP1, give about $16 \times 3{,}229 \times 0.5 \approx 26{,}000$ output tokens/s, or 93 million output tokens an hour if they ran flat out. The same 16 GPUs serving the 8B model in FP8 at batch 256 give about $16 \times 13{,}978 \times 0.5 \approx 112{,}000$ tokens/s. Real traffic isn't flat, and some of the fleet must be headroom, which is what the planner works out.

### A placement planner

The planner takes a list of customer deployments with their peak traffic and latency targets and proposes a placement on 2 nodes × 8 GPUs. For each deployment it tries every shape (MIG slices for models under 10B parameters, then TP1, TP2, TP4 and TP8 inside one node) and keeps the one that needs the fewest GPUs. For a shape it:

1. Computes KV room after weights and resident LoRA adapters, and requires room for at least four full-context sequences.
2. Finds the largest batch whose decode step meets the deployment's TPOT target, the [knob a real engine sets with `max_num_seqs`](../inference/serving-efficiently.md#the-batch-knobs-max_num_seqs-max_num_batched_tokens-and-the-chunked-prefill-budget).
3. Derives the replica's output rate (roofline × 50%, × 90% if it carries LoRA adapters), and its prefill rate.
4. Adds decode time and uncached prefill time at peak to get how busy one replica would be, and sizes replicas to stay at or below 70% busy. That 30% headroom absorbs bursts and the queueing that [Little's law](../inference/arithmetic.md#littles-law) predicts near saturation.

LoRA customers on the same base are one deployment (a **LoRA pool**) with the sum of their traffic. The 90% factor is Fireworks' published number for multi-LoRA throughput, which can "approach 90% of base model performance" ([Fireworks](https://fireworks.ai/blog/multi-lora)). Adapters are sized as in [Internals II](../inference/serving-engine-internals.md#multi-lora-serving), at rank 64 with 16 resident per replica. That memory comes out of the KV budget.

Placement spreads each deployment's replicas over both nodes, so a deployment with two or more replicas survives a node failure, and reports which deployments would lose every replica if each node died. MIG slices are packed onto shared GPUs first-fit, with replica $j$ of a deployment going to node $j \bmod 2$.

```python
# planner.py - propose a placement of customer models on 2 nodes x 8 B300, and report headroom.
#   python3 planner.py           (imports capacity.py from the same directory)
# Rates are roofline bounds from capacity.py times EFF. Replace EFF with your measured fraction.
import math, sys
from capacity import MODELS, HBM, kv_tokens, step_time, prefill_tok_s, weights_per_gpu

NODES, GPUS = 2, 8
EFF = 0.5            # fraction of the roofline a real engine reaches (assumption; measure it, Internals I)
TARGET = 0.7         # plan replicas to run at most 70% busy at peak
LORA_EFF = 0.9       # multi-LoRA throughput vs base ("approach 90%", Fireworks)
MAX_LORAS, LORA_RANK = 16, 64   # adapters resident per replica, and the rank the kernels are sized for
GiB = 2**30
MIG = [("1g.34gb", 1, 1, 34 * GiB), ("2g.67gb", 2, 2, 67 * GiB), ("3g.135gb", 3, 4, 135 * GiB)]
#      name, compute sevenths, memory eighths, bytes   (B300 288GB profiles, NVIDIA MIG user guide)

def lora_bytes(m, r=LORA_RANK):                      # rank-r adapter on all 7 linear layers, BF16
    return m["L"] * r * m["lin"] * 2

def shapes(d):
    """Candidate replica shapes: MIG slices first, then whole-GPU TP groups inside one node."""
    m = MODELS[d["model"]]
    extra = MAX_LORAS * lora_bytes(m) if d["adapters"] else 0.0
    for name, c7, m8, mem in MIG if m["N"] < 10e9 else []:   # MIG only for small models; bw ~ memory share,
                                                             # FLOPs ~ compute share (assumed)
        yield dict(kind="mig", name=name, P=1, gpus=max(c7 / 7, m8 / 8), c7=c7, m8=m8, hbm=mem, bw=m8 / 8, fl=c7 / 7, extra=extra)
    for P in (1, 2, 4, 8):
        yield dict(kind="tp", name=f"TP{P}", P=P, gpus=P, hbm=HBM, bw=1.0, fl=1.0, extra=extra / P)

def size(d, s):
    """Replicas of shape s that carry deployment d's peak at <= TARGET busy, or None if it can't."""
    m, dt, T = MODELS[d["model"]], d["dtype"], d["ctx"]
    tok = kv_tokens(m, dt, s["P"], hbm=s["hbm"], extra=s["extra"])
    if tok < 4 * T:                                  # want room for at least 4 full-context sequences
        return None
    B = 1
    while B < min(tok // T, 256) and step_time(m, dt, s["P"], B + 1, T, s["bw"], s["fl"]) <= d["tpot"]:
        B += 1
    st = step_time(m, dt, s["P"], B, T, s["bw"], s["fl"])
    if st > d["tpot"]:
        return None
    eff = EFF * (LORA_EFF if d["adapters"] else 1.0)
    dec = eff * B / st                               # output tok/s per replica
    pre = eff * prefill_tok_s(m, dt, s["P"], T, s["fl"])
    busy = d["out"] / dec + d["out"] * d["in_per_out"] * (1 - d["hit"]) / pre   # replica-seconds per second
    n = max(d.get("min_rep", 1), math.ceil(busy / TARGET))
    return dict(shape=s, n=n, B=B, step=st, busy=busy, dec=dec, kv=tok)

def plan(deploys):
    picks = []
    for d in deploys:
        best = min((r for s in shapes(d) if (r := size(d, s))), key=lambda r: (r["n"] * r["shape"]["gpus"], r["shape"]["P"], r["shape"]["kind"] == "mig"), default=None)
        if best is None:
            sys.exit(f"{d['name']}: no shape meets TPOT {d['tpot']*1e3:.0f} ms")
        picks.append((d, best))
    # pack MIG slices onto shared GPUs, first fit on compute sevenths and memory eighths;
    # replica j of a deployment goes to a MIG GPU on node j % NODES, so replicas don't share a node
    mig_gpus = []                                    # each: [node, used sevenths, used eighths, [labels]]
    for d, r in picks:
        if r["shape"]["kind"] != "mig": continue
        for j in range(r["n"]):
            fits = lambda g: g[0] == j % NODES and g[1] + r["shape"]["c7"] <= 7 and g[2] + r["shape"]["m8"] <= 8
            g = next((g for g in mig_gpus if fits(g)), None)
            if g is None:
                g = [j % NODES, 0, 0, []]; mig_gpus.append(g)
            g[1] += r["shape"]["c7"]; g[2] += r["shape"]["m8"]; g[3].append(f"{d['name']}:{r['shape']['name']}")
    # MIG GPUs go to their node; TP groups (largest first) to the node holding fewest of their replicas
    nodes = [[] for _ in range(NODES)]               # each: list of (label, gpus)
    free = lambda i: GPUS - sum(g for _, g in nodes[i])
    for g in mig_gpus:
        if free(g[0]) < 1:
            return picks, nodes, "MIG GPU"
        nodes[g[0]].append(("MIG[" + " ".join(g[3]) + "]", 1))
    items = [(d["name"], r["shape"]["P"]) for d, r in picks if r["shape"]["kind"] == "tp" for _ in range(r["n"])]
    for label, P in sorted(items, key=lambda x: -x[1]):
        same = lambda i: sum(l == label for l, _ in nodes[i])
        ok = [i for i in range(NODES) if free(i) >= P]
        if not ok:
            return picks, nodes, label
        nodes[min(ok, key=lambda i: (same(i), -free(i)))].append((label, P))
    return picks, nodes, None

def report(title, deploys):
    print(f"=== {title}")
    picks, nodes, failed = plan(deploys)
    print(f"{'deployment':18s} {'model':12s} {'dtype':5s} {'shape':8s} {'n':>2s} {'GPUs':>5s} {'B':>4s} {'TPOT':>6s} "
          f"{'tok/s/rep':>9s} {'peak out':>8s} {'busy@peak':>9s}")
    for d, r in picks:
        print(f"{d['name']:18s} {d['model']:12s} {d['dtype']:5s} {r['shape']['name']:8s} {r['n']:2d} "
              f"{r['n']*r['shape']['gpus']:5.2f} {r['B']:4d} {r['step']*1e3:5.1f}ms {r['dec']:9,.0f} {d['out']:8,d} "
              f"{r['busy']/r['n']:9.0%}")
    need = sum(P for n in nodes for _, P in n)
    if failed:
        want = sum(r["n"] * r["shape"]["gpus"] for _, r in picks)
        print(f"DOES NOT FIT: needs {want:.1f} GPUs before packing losses; ran out at {failed}"); return None
    for i, n in enumerate(nodes):
        print(f"node {i}: " + ", ".join(f"{l}({P})" for l, P in n) + f"  | free {GPUS - sum(P for _, P in n)}")
    print(f"fleet: {need}/{NODES*GPUS} GPUs placed, {NODES*GPUS-need} free")
    for i in range(NODES):                           # what survives if node i dies?
        lost = [d["name"] for d, r in picks if not any(any(l == d["name"] or d["name"] + ":" in l for l, _ in nodes[j]) for j in range(NODES) if j != i)]
        print(f"  lose node {i}: no replica left for {lost if lost else 'nobody'}")
    return picks, nodes

# Peak output tok/s per deployment; in_per_out = prompt tokens per output token; hit = prefix-cache hit rate
def D(name, model, dtype, out, in_per_out, hit, ctx, tpot_ms, adapters=0, **kw):
    return dict(name=name, model=model, dtype=dtype, out=out, in_per_out=in_per_out, hit=hit, ctx=ctx,
                tpot=tpot_ms / 1e3, adapters=adapters, **kw)

LORA_FIRST = [
    D("8b-lora-pool", "llama-8b", "fp8", 6000, 6, 0.5, 8192, 40, adapters=14, min_rep=2),
    D("70b-lora-pool", "llama-70b", "fp8", 2000, 8, 0.5, 16384, 50, adapters=6, min_rep=2),
    D("70b-ft-kestrel", "llama-70b", "fp8", 600, 5, 0.3, 16384, 50),
    D("glm-air-api", "glm-air-106b", "nvfp4", 2500, 10, 0.6, 32768, 50, min_rep=2),
    D("3b-ft-rook", "qwen-3b", "bf16", 600, 20, 0.2, 4096, 30, min_rep=2),
    D("1.5b-ft-sable", "qwen-1.5b", "bf16", 300, 10, 0.2, 2048, 30, min_rep=2),
    D("1.5b-ft-wren", "qwen-1.5b", "bf16", 200, 10, 0.2, 2048, 30),
]
# the same customers if every LoRA customer had insisted on a full fine-tune (traffic split evenly)
FULL_FT = [d for d in LORA_FIRST if not d["adapters"]]
FULL_FT += [D(f"8b-ft-{i:02d}", "llama-8b", "fp8", 6000 // 14, 6, 0.5, 8192, 40) for i in range(14)]
FULL_FT += [D(f"70b-ft-{i:02d}", "llama-70b", "fp8", 2000 // 6, 8, 0.5, 16384, 50) for i in range(6)]

if __name__ == "__main__":
    m = MODELS["llama-8b"]
    print(f"rank-{LORA_RANK} LoRA: 8B {lora_bytes(m)/1e6:.0f} MB, 70B {lora_bytes(MODELS['llama-70b'])/1e6:.0f} MB; "
          f"{MAX_LORAS} resident cost {MAX_LORAS*lora_bytes(MODELS['llama-70b'])/1e9:.1f} GB on a 70B replica")
    a = report("LoRA-first: fine-tunes as adapters on shared bases", LORA_FIRST)
    print()
    b = report("Full fine-tunes for everyone", FULL_FT)
    assert a is not None and b is None               # the point of the section
    assert abs(lora_bytes(m, 16) / 2 - 41.9e6) < 0.1e6   # matches Internals II's 41.9M-param rank-16 adapter
```

The customer list in `LORA_FIRST` is the company's book of business at month three: 14 startups with LoRA fine-tunes of Llama-3.1-8B, 6 with LoRA fine-tunes of Llama-3.3-70B, one ("kestrel") with a full fine-tune of the 70B model, a shared GLM-4.5-Air API, and three small full fine-tunes of Qwen2.5 models used as classifiers and routers. The traffic numbers are peak output tokens per second, made up for the example; the ratios of prompt to output tokens and the prefix-cache hit rates are the kind of thing [trace_stats.py](../inference/serving-efficiently.md#what-to-compute-from-a-trace) measures from real logs.

```console
$ python3 planner.py
rank-64 LoRA: 8B 336 MB, 70B 1657 MB; 16 resident cost 26.5 GB on a 70B replica
=== LoRA-first: fine-tunes as adapters on shared bases
deployment         model        dtype shape     n  GPUs    B   TPOT tok/s/rep peak out busy@peak
8b-lora-pool       llama-8b     fp8   TP1       2  2.00  256  18.3ms     6,290    6,000       60%
70b-lora-pool      llama-70b    fp8   TP1       5  5.00   59  28.9ms       919    2,000       63%
70b-ft-kestrel     llama-70b    fp8   TP1       2  2.00   69  32.2ms     1,070      600       40%
glm-air-api        glm-air-106b nvfp4 TP1       4  4.00   64  32.3ms       992    2,500       67%
3b-ft-rook         qwen-3b      bf16  1g.34gb   2  0.29  157  29.9ms     2,627      600       41%
1.5b-ft-sable      qwen-1.5b    bf16  1g.34gb   2  0.29  256  18.1ms     7,067      300        6%
1.5b-ft-wren       qwen-1.5b    bf16  1g.34gb   1  0.14  256  18.1ms     7,067      200        8%
node 0: MIG[3b-ft-rook:1g.34gb 1.5b-ft-sable:1g.34gb 1.5b-ft-wren:1g.34gb](1), 8b-lora-pool(1), 70b-lora-pool(1), 70b-lora-pool(1), 70b-lora-pool(1), 70b-ft-kestrel(1), glm-air-api(1), glm-air-api(1)  | free 0
node 1: MIG[3b-ft-rook:1g.34gb 1.5b-ft-sable:1g.34gb](1), 8b-lora-pool(1), 70b-lora-pool(1), 70b-lora-pool(1), 70b-ft-kestrel(1), glm-air-api(1), glm-air-api(1)  | free 1
fleet: 15/16 GPUs placed, 1 free
  lose node 0: no replica left for ['1.5b-ft-wren']
  lose node 1: no replica left for nobody

=== Full fine-tunes for everyone
deployment         model        dtype shape     n  GPUs    B   TPOT tok/s/rep peak out busy@peak
70b-ft-kestrel     llama-70b    fp8   TP1       2  2.00   69  32.2ms     1,070      600       40%
glm-air-api        glm-air-106b nvfp4 TP1       4  4.00   64  32.3ms       992    2,500       67%
3b-ft-rook         qwen-3b      bf16  1g.34gb   2  0.29  157  29.9ms     2,627      600       41%
1.5b-ft-sable      qwen-1.5b    bf16  1g.34gb   2  0.29  256  18.1ms     7,067      300        6%
1.5b-ft-wren       qwen-1.5b    bf16  1g.34gb   1  0.14  256  18.1ms     7,067      200        8%
8b-ft-00           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-01           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-02           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-03           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-04           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-05           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-06           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-07           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-08           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-09           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-10           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-11           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-12           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
8b-ft-13           llama-8b     fp8   1g.34gb   2  0.29   44  32.7ms       673      428       37%
70b-ft-00          llama-70b    fp8   TP1       1  1.00   69  32.2ms     1,070      333       46%
70b-ft-01          llama-70b    fp8   TP1       1  1.00   69  32.2ms     1,070      333       46%
70b-ft-02          llama-70b    fp8   TP1       1  1.00   69  32.2ms     1,070      333       46%
70b-ft-03          llama-70b    fp8   TP1       1  1.00   69  32.2ms     1,070      333       46%
70b-ft-04          llama-70b    fp8   TP1       1  1.00   69  32.2ms     1,070      333       46%
70b-ft-05          llama-70b    fp8   TP1       1  1.00   69  32.2ms     1,070      333       46%
DOES NOT FIT: needs 16.7 GPUs before packing losses; ran out at 70b-ft-04
```

How to read the first plan:

- **Everything is TP1 or smaller.** With 259 GB per GPU, no deployment on this list needs tensor parallelism to fit or to meet a 50 ms TPOT target. Every replica is an independent single-GPU engine behind a router, the layout [Prime Intellect found to scale linearly](../inference/serving-efficiently.md#many-independent-replicas-vs-one-multi-node-engine). One consequence is a simple failure domain: one bad GPU takes out one replica, not a node's worth.
- **The 70B LoRA pool runs at batch 59, not 138.** Sixteen resident rank-64 adapters cost 26.5 GB, and the 16k context halves the sequences that fit, so KV is what limits the batch. A customer who wants longer contexts on this pool moves it toward TP2 (twice the KV per replica, [see above](#kv-room-and-concurrency-at-typical-contexts)).
- **The three small fine-tunes cost less than one GPU together.** Five `1g.34gb` slices hold them on two shared GPUs, one per node. The 1.5B "sable" model runs at 6% busy at peak; it has two replicas because its contract says so, not because it needs the throughput.
- **The headroom is one GPU and the 30% built into each pool.** 15 of 16 GPUs are placed. The busiest pool would run at 67% at peak if the traffic forecasts are right.
- **The N-1 check finds one exposure**: "wren", which bought the cheapest tier with one replica, is down while node 0 is. That's a contract decision, not a bug, and [section 7](#7-planning-for-the-loss-of-a-node) puts it in the tier table. The check is only about whether *some* replica survives. How much capacity survives is a different question, also in section 7.

The planner is deliberately crude. It ignores prefill/decode interference, it assumes a uniform context length per deployment, its MIG packing doesn't know the hardware's placement rules, and its rates are 50% of a roofline. What it gives you is the shape of the answer and the arithmetic to check a proposal against. When you have [a replayed trace](../inference/serving-efficiently.md#a-replayable-benchmark) and measured per-replica goodput, replace `dec` and `pre` with measured numbers and keep the rest.

### MIG slices for small models

MIG is how a 16-GPU fleet serves small models without giving each one a 288 GB GPU. A Qwen2.5-1.5B model needs about 3 GB of weights; on a whole B300 it would leave 99% of the memory as KV that its traffic never uses. A `1g.34gb` slice gives it hardware-isolated memory and compute, a seventh of the SMs and (by assumption) an eighth of the bandwidth, which the planner's table shows is still enough for about 7,000 tokens/s per slice after its 50% derate, at an 18 ms step.

MIG is the right tool here for three reasons beyond packing: the [isolation is in hardware](../compute/gpu-software-stack.md#sharing-one-gpu-mig-and-mps), so one tenant's out-of-memory error or illegal access can't take down another's; each slice is a separate CUDA device, so each engine is an ordinary process with `CUDA_VISIBLE_DEVICES=MIG-<uuid>`; and slices can be sold. The costs: reconfiguring MIG needs the GPU idle (and may need a reset), so the layout is chosen at bring-up and changed rarely, and a slice can't borrow idle SMs from its neighbours the way processes under [MPS](../compute/gpu-software-stack.md#sharing-one-gpu-mig-and-mps) can. Use MPS or plain co-location (two engines at `--gpu-memory-utilization 0.45`, as in [Serving Efficiently](../inference/serving-efficiently.md#multiplexing-and-swapping-whole-models)) for your own cooperating processes, and MIG between tenants.

```bash
# untested. One B300 split into seven 1g.34gb slices, one small model per slice.
sudo nvidia-smi -i 15 -mig 1                          # MIG mode on GPU 15 (GPU must be idle)
nvidia-smi mig -i 15 -lgip                            # confirm the B300 profiles listed above
nvidia-smi mig -i 15 -lgipp                           # valid placements, which the planner ignores
sudo nvidia-smi mig -i 15 -cgi 1g.34gb,1g.34gb,1g.34gb -C
nvidia-smi -L                                         # MIG device UUIDs
CUDA_VISIBLE_DEVICES=MIG-<uuid> vllm serve /mnt/nvme/models/rook-3b/v7 --port 8101 --gpu-memory-utilization 0.9
```

## 3. Packing many customers: LoRA first

### Shared bases vs dedicated replicas

The second plan is the same customers if every one of the 20 LoRA customers had insisted on a full fine-tune: same traffic, split evenly. It doesn't fit, and the way it fails is the business lesson.

- **The 8B customers survive on MIG.** Fourteen 8B full fine-tunes in FP8 fit in `1g.34gb` slices, two each, about 4 GPUs. Each runs at batch 44 because a 34 GB slice has room for only that much KV, and at 37% busy.
- **The 70B customers each cost a whole GPU**, regardless of traffic. Six 70B full fine-tunes at 333 tokens/s each need six GPUs at 46% busy, against five GPUs at 63% busy for the LoRA pool that carries the same 2,000 tokens/s. And that's with one replica each. A customer who also wants to survive a node failure needs two, and six such customers need twelve GPUs, three quarters of the fleet.
- **The fleet stops at 16.7 GPUs of demand**, before the per-customer floors for redundancy, before any headroom for growth, before any GPU is left for training.

The mechanism is the per-deployment floor. A full fine-tune is a separate set of weights, so it needs its own replica (a slice, a GPU, or a TP group), sized for its own peak, idle in its own troughs. A LoRA adapter rides on a replica that exists anyway, costs a few hundred MB to 1.7 GB of memory, and shares a batch with every other customer on that base ([Serving Efficiently](../inference/serving-efficiently.md#multi-lora-many-fine-tunes-on-one-base)). Prime Intellect's hosted training product runs on exactly this design, with "multi-tenant LoRA deployments" that allow "shared hardware across runs, enabling high efficiency and per-token pricing", and says that "running dedicated, fine-tuned deployments of large Mixture of Experts models per customer is infeasible and wasteful" ([Prime Intellect](https://www.primeintellect.ai/blog/lab)).

The same arithmetic disposes of the 671B request from [section 2](#memory-budgets-per-gpu-and-per-node). A dedicated DeepSeek-V3 deployment in FP8 needs 4 GPUs per replica and two replicas to survive a node: half the fleet for one customer. On 16 GPUs the honest answer is to route that customer to a provider that serves the model per token, or to price a dedicated deployment at half the lease plus margin, on the same term as the lease.

### The business consequence

On a fixed lease, every replica floor is a cost the company pays every hour. Translate the floors into money with the assumed \$4.00 per GPU-hour:

| What the customer buys | GPUs it pins | Lease cost per month |
|---|---|---|
| 70B full fine-tune, one replica | 1 | $730 \times 4.00 = \$2{,}920$ |
| 70B full fine-tune, two replicas (survives a node) | 2 | \$5,840 |
| 8B full fine-tune, two `1g.34gb` slices | 2/7 | about \$830 |
| 671B dedicated, FP8, two replicas | 8 | \$23,360 |
| any LoRA on a pool that already exists | 0 | adapter memory, plus ~10% of the throughput it uses |

That table is a price list's floor. Three consequences for how the company sells:

1. **Default every fine-tune to LoRA**, priced per token like the base model, and make it the path of least resistance in the training product. Fireworks prices multi-LoRA serving the same as the base, "Inference prices are the same as a single base model", whether you serve one LoRA or hundreds ([Fireworks](https://fireworks.ai/blog/multi-lora)). The cost basis makes that possible: the pool's GPUs are paid for by its aggregate traffic.
2. **Price full fine-tunes as dedicated capacity**, per replica-month, with a minimum term that matches the lease. [Operating Inference](operating-inference.md#dedicated-vs-shared-deployments-noisy-neighbours) makes the case: dedicated capacity sold by the hour shifts utilization risk onto the provider, and a customer who leaves after a month leaves the GPU idle for the eleven months the company still pays for.
3. **Keep the number of bases small.** Every distinct base model is another pool with its own floor and its own headroom. Two or three bases (an 8B, a 70B and one MoE here) are what 16 GPUs can hold with redundancy. A customer who wants LoRA on a fourth base is a full-fine-tune customer in disguise.

### Deciding with evals

"LoRA when quality allows" needs a test, agreed with the customer before training starts. The procedure:

1. **The customer supplies the eval**, or you build it with them: held-out items from their task, with a scoring rule they accept (written before the models are trained, so neither side tunes to it).
2. **Agree a non-inferiority margin**: how much worse than a full fine-tune the LoRA may be. Two points of accuracy is a common kind of number; the customer's product decides it.
3. **Train both** on the same data, a LoRA at a rank the pool supports and a full fine-tune, score both on the same items, and compare *paired* per-item results. Pairing matters because most items are scored the same by both models; the variance that counts is in the items where they differ.
4. **Decide on the confidence interval, not the point estimate**: LoRA if the upper end of the 95% interval on (full − LoRA) is below the margin, full fine-tune if the lower end is above it, otherwise more eval items.

```python
# lora_or_full.py - is the LoRA good enough? A paired non-inferiority test on the customer's eval set.
#   python3 lora_or_full.py
# The per-item scores are SIMULATED (seeded): replace them with your two models' per-item results.
import random

MARGIN = 0.02          # the customer accepts up to 2 points worse than the full fine-tune (agreed up front)

def simulate(n, p_full, p_lora, rho_flip, seed):
    """Per-item 0/1 scores for both models on the same items. Most items agree; a few flip."""
    rng = random.Random(seed)
    full = [rng.random() < p_full for _ in range(n)]
    lora = [f if rng.random() > rho_flip else (rng.random() < p_lora) for f in full]
    return full, lora

def paired_bootstrap(full, lora, B=10_000, seed=0):
    rng = random.Random(seed)
    d = [int(f) - int(l) for f, l in zip(full, lora)]            # +1: full right, LoRA wrong
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, means[int(0.025 * B)], means[int(0.975 * B)]

print(f"{'eval items':>10s} {'full':>6s} {'LoRA':>6s} {'diff':>6s} {'95% CI':>16s}  decision (margin {MARGIN:.0%})")
for n, p_lora, rho, seed in ((60, 0.80, 0.25, 1), (600, 0.80, 0.25, 2), (600, 0.55, 0.5, 3), (3000, 0.80, 0.25, 4)):
    full, lora = simulate(n, 0.82, p_lora, rho_flip=rho, seed=seed)
    diff, lo, hi = paired_bootstrap(full, lora)
    if hi < MARGIN:
        verdict = "LoRA is non-inferior: serve it on the shared base"
    elif lo > MARGIN:
        verdict = "LoRA is worse by more than the margin: full fine-tune (dedicated price)"
    else:
        verdict = "inconclusive: get more eval items before deciding"
    print(f"{n:10d} {sum(full)/n:6.1%} {sum(lora)/n:6.1%} {diff:+6.1%} [{lo:+6.1%}, {hi:+6.1%}]  {verdict}")
```

```console
$ python3 lora_or_full.py
eval items   full   LoRA   diff           95% CI  decision (margin 2%)
        60  80.0%  81.7%  -1.7% [ -8.3%,  +3.3%]  inconclusive: get more eval items before deciding
       600  81.8%  82.2%  -0.3% [ -2.7%,  +2.0%]  inconclusive: get more eval items before deciding
       600  83.2%  72.7% +10.5% [ +7.0%, +14.0%]  LoRA is worse by more than the margin: full fine-tune (dedicated price)
      3000  81.6%  80.7%  +0.9% [ -0.1%,  +1.9%]  LoRA is non-inferior: serve it on the shared base
```

Sixty items can't decide anything at a 2-point margin: the interval is 11.6 points wide. Six hundred items nearly decide the equal-quality case; the interval's upper end lands right on the margin, so it still says "inconclusive". Three thousand decide it. A clearly worse LoRA shows up with 600. Most startups arrive with fewer than a few hundred labelled examples, so the practical rule is to agree the margin and the eval size before training, and to offer to build the eval with them. When the eval can't decide, the default is LoRA: it can be upgraded to a full fine-tune later, while a dedicated replica, once sold, is hard to take back. Seeds and their meaning are in [Serving LLMs in Practice](../inference/serving-hands-on.md#seeds-and-why-one-run-is-an-anecdote); a real comparison should also train each variant with more than one seed.

## 4. From a training checkpoint to a served model

### The pipeline

Every customer model goes through the same nine steps between "training finished" and "serving all traffic". Each is a gate: a failure stops the pipeline and leaves the previous version serving.

1. **Checkpoint.** The training job writes a final checkpoint with its config, data hashes and git SHA ([reproducibility](../training/fault-tolerance-and-reproducibility.md#reproducibility)).
2. **Merge or keep the LoRA.** Keep it as an adapter by default; merge only for customers who get a dedicated replica (below).
3. **Convert** to the serving format: Hugging Face safetensors with the base model's tokenizer and chat template copied verbatim, pre-sharded per rank if TP > 1 ([pre-sharding](../inference/serving-hands-on.md#speeding-up-model-loading-time)).
4. **Quantize** to FP8 or NVFP4, with calibration data where the scheme needs it.
5. **Quality gate** against the BF16 model on the customer's evals.
6. **Stage weights** on every serving node's NVMe, with a `.complete` marker ([weights](deploys-recovery-autoscaling.md#weights)).
7. **Rolling deploy** with a canary replica and readiness gated on warmup ([rolling updates on a full fleet](deploys-recovery-autoscaling.md#surge-and-unavailability-on-a-full-fleet)).
8. **Canary prompts** on the new replica, compared with golden outputs ([output-quality probes](deploys-recovery-autoscaling.md#output-quality-probes)).
9. **Rollback** to the previous version, kept on NVMe ([rollback](deploys-recovery-autoscaling.md#rollback)).

### Merge or keep the adapter

For a LoRA customer, *keep the adapter*. Loading it is a runtime API call on a running pool ([runtime LoRA loading](../inference/serving-efficiently.md#multi-lora-many-fine-tunes-on-one-base)), so a new version is a few hundred megabytes on disk and a load call, not a replica restart. The adapter stays in BF16 while the base is FP8 or NVFP4. Two cautions. The base the adapter was trained against is the BF16 base, and it is served on a quantized base, so the quality gate must run on the *quantized base plus the adapter*, not on BF16 alone. And the adapter's rank must fit the pool's `--max-lora-rank`, which sizes the kernels for every adapter on the replica ([Serving Efficiently](../inference/serving-efficiently.md#multi-lora-many-fine-tunes-on-one-base)); a rank-256 adapter belongs on a different pool.

For a full fine-tune or a customer who pays for a dedicated replica, *merge* ($W' = W + BA$), then quantize the merged weights. Merging removes the LoRA kernels' overhead (the 10% in the planner). The catch is that quantization error is now applied to the fine-tuned weights, not the base. Fireworks reports that when "LoRA is merged into the base model" and then quantized, "the quality is significantly degraded" without quantization-aware training ([FireAttention V4](https://fireworks.ai/blog/fireattention-v4-fp4-b200)). That is the case the gate below exists for.

```python
# untested. Merge a LoRA into its base with PEFT, in BF16, on CPU or one GPU.
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("/models/llama-3.3-70b-instruct", dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "/ckpt/kestrel/lora-final").merge_and_unload()
merged.save_pretrained("/ckpt/kestrel/merged-bf16", max_shard_size="5GB")
AutoTokenizer.from_pretrained("/models/llama-3.3-70b-instruct").save_pretrained("/ckpt/kestrel/merged-bf16")
```

### Convert and quantize

The two formats that matter on B300 need different amounts of calibration:

- **FP8 with dynamic activation scales** needs none. Weights get per-channel scales computed from the weights themselves; activations are scaled per token at runtime. llm-compressor's FP8 example is `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])` with `oneshot(model=model, recipe=recipe)` and no dataset ([example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/llama3_example.py)). The scaling choices are explained in [FP8 E4M3 scaling granularities](../inference/arithmetic.md#fp8-e4m3-and-its-scaling-granularities).
- **NVFP4 (W4A4)** quantizes activations to FP4 too, and needs calibration data to set the activations' global scale. llm-compressor's NVFP4 example uses `scheme="NVFP4"`, `ignore=["lm_head"]`, and a `oneshot` call with a dataset, `max_seq_length=2048` and `num_calibration_samples=20` ([example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/llama3_example.py)). The format itself is in [NVFP4 and MXFP4](../inference/arithmetic.md#nvfp4-and-mxfp4).

Calibration data should be the customer's traffic, not a generic chat set. A model fine-tuned on legal contracts has activation ranges a chat dataset never exercises. Use a sample of the customer's training prompts (they already gave you those) or of logged production prompts if the contract allows. Never use the eval set: the gate must be measured on data the quantizer didn't see.

```python
# untested. NVFP4 with calibration on the customer's own prompts (llm-compressor oneshot API).
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
src = "/ckpt/kestrel/merged-bf16"
model, tok = AutoModelForCausalLM.from_pretrained(src), AutoTokenizer.from_pretrained(src)
calib = load_dataset("json", data_files="/data/kestrel/train_prompts.jsonl", split="train").shuffle(seed=0).select(range(512))
oneshot(model=model, dataset=calib, recipe=QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"]),
        max_seq_length=2048, num_calibration_samples=512)
model.save_pretrained("/ckpt/kestrel/nvfp4", save_compressed=True); tok.save_pretrained("/ckpt/kestrel/nvfp4")
```

The output is a compressed-tensors checkpoint that vLLM detects and loads without a `--quantization` flag. A 70B merge-and-quantize should fit on one B300, since 141 GB of BF16 weights fit (untested), which matters for [section 5](#5-training-and-serving-on-the-same-16-gpus): it is a job that takes one GPU off serving for a while, not a node.

### The quality gate

The gate compares the candidate (quantized, possibly merged, possibly with an adapter) with the BF16 reference on the customer's eval set, and passes only if all of these hold:

- **Distribution distance on the reference answers.** Teacher-force the reference answers through both models and compute, per answer token, the KL divergence from reference to candidate and whether their top-1 tokens agree. This is the most sensitive signal and needs no generation. It needs logprobs, which is why it runs offline on your own models: Fireworks notes that it fell back to MMLU for cross-engine comparisons because "not all LLM implementations return log probabilities" ([FireAttention V4](https://fireworks.ai/blog/fireattention-v4-fp4-b200)).
- **The customer's task metric**, under the same decoding the product uses (greedy here), on the same items, with paired flips counted: how many items went right→wrong, and how many wrong→right.
- **Thresholds written down before looking at candidates.** NVIDIA reports "1% or less accuracy degradation" from FP8 to NVFP4 on DeepSeek-R1-0528 ([NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)); that is a vendor number on a 671B model, and a customer's 8B fine-tune can do much worse. Set the thresholds per customer, in the contract if their product depends on it.

The script below is the gate, run on Qwen2.5-0.5B-Instruct on CPU, with three candidates: FP8 (per-channel E4M3), NVFP4 (16-element blocks, E4M3 block scales, a per-tensor FP32 scale) and a deliberately broken INT4 with one scale per tensor. The candidates are **simulated**: weights are rounded to each format's grid and stored back in fp32, so the script measures the effect of weight rounding only. A real W8A8 or W4A4 engine also quantizes activations and uses different kernels, so its numbers will differ. The gate logic is unchanged when you feed it a real engine's logprobs.

```python
# quality_gate.py - gate a quantized candidate against the BF16 reference on a customer's eval set.
#   /tmp/absorb-venv/bin/python quality_gate.py
# Model: Qwen/Qwen2.5-0.5B-Instruct from the local Hugging Face cache, on CPU.
# The candidates are SIMULATED quantizations (weights rounded to FP8 / NVFP4 grids and stored back in
# fp32), not real low-precision kernels: they reproduce the weight rounding error, not the activation
# quantization or the kernels. The gate logic is the same for a real engine's outputs.
import math, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Gate thresholds, written down BEFORE looking at candidates. Per-token numbers are on the reference answers.
MAX_KL = 0.05          # mean KL(ref || cand), nats per answer token
MIN_TOP1 = 0.90        # fraction of answer positions where both models' argmax agree
MAX_ACC_DROP = 0.025   # task accuracy may drop by at most 2.5 points ...
MAX_NET_FLIPS = 1      # ... and at most 1 more item may go right->wrong than wrong->right

EVAL = [  # (question, accepted answer) - stands in for the customer's held-out eval set
    ("What is 7 times 8?", "56"), ("What is 12 plus 15?", "27"), ("What is 100 minus 37?", "63"),
    ("What is 9 times 9?", "81"), ("What is 144 divided by 12?", "12"), ("What is 25 plus 17?", "42"),
    ("What is 6 times 7?", "42"), ("What is 50 minus 18?", "32"), ("What is 11 times 11?", "121"),
    ("What is 90 divided by 3?", "30"), ("What is 8 times 12?", "96"), ("What is 33 plus 44?", "77"),
    ("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"), ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Spain?", "Madrid"), ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"), ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Kenya?", "Nairobi"), ("What is the capital of Peru?", "Lima"),
    ("What is the capital of Norway?", "Oslo"), ("What is the capital of Turkey?", "Ankara"),
    ("What is the chemical symbol for gold?", "Au"), ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"), ("What is the chemical symbol for oxygen?", "O"),
    ("How many days are in a leap year?", "366"), ("How many legs does a spider have?", "8"),
    ("How many continents are there?", "7"), ("How many minutes are in an hour?", "60"),
    ("Which planet is closest to the Sun?", "Mercury"), ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the largest ocean on Earth?", "Pacific"), ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"), ("What is the freezing point of water in Celsius?", "0"),
    ("What is the square root of 64?", "8"), ("In which year did World War II end?", "1945"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
def prompt_ids(q):
    msgs = [{"role": "user", "content": q + " Answer in a few words."}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]

def load():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).float().eval()

# ---- fake quantizers: (out, in) weight -> same shape, values on the target grid
def q_fp8(w):                                       # E4M3, one scale per output channel
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
    return (w / s).to(torch.float8_e4m3fn).float() * s

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
def to_e2m1(x):                                     # round |x| <= 6 to the nearest FP4 (E2M1) value
    idx = (x.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    return E2M1[idx] * x.sign()

def q_nvfp4(w):                                     # blocks of 16 along `in`, FP8 E4M3 block scale, FP32 tensor scale
    o, i = w.shape
    g = w.abs().max() / (6.0 * 448.0)               # tensor scale so every block scale fits in E4M3
    b = w.reshape(o, i // 16, 16)
    s = (b.abs().amax(-1, keepdim=True) / 6.0 / g).to(torch.float8_e4m3fn).float().clamp(min=2**-9) * g
    return (to_e2m1(b / s) * s).reshape(o, i)

def q_int4_tensor(w):                               # the "bad" candidate: INT4 with ONE scale per tensor
    s = w.abs().max() / 7.0
    return (w / s).round().clamp(-8, 7) * s

def quantize(model, fn):                            # decoder Linear layers only; embeddings / lm_head stay
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and ".layers." in name:
                mod.weight.copy_(fn(mod.weight))
    return model

@torch.no_grad()
def answer_logprobs(model):                         # teacher-forced log-softmax over each reference answer
    out = []
    for q, a in EVAL:
        p = prompt_ids(q)
        a_ids = tok(" " + a if a[0].isdigit() else a, return_tensors="pt")["input_ids"]
        ids = torch.cat([p, a_ids], 1)
        logits = model(ids).logits[0, p.shape[1] - 1:-1]        # (len(a), vocab): predictions of answer tokens
        out.append((torch.log_softmax(logits.float(), -1), a_ids[0]))
    return out

@torch.no_grad()
def task_correct(model):                            # greedy, 16 new tokens, case-insensitive containment
    res, texts = [], []
    for q, a in EVAL:
        p = prompt_ids(q)
        g = model.generate(p, attention_mask=torch.ones_like(p), max_new_tokens=16, do_sample=False)
        t = tok.decode(g[0, p.shape[1]:], skip_special_tokens=True)
        res.append(a.lower() in t.lower()); texts.append(t)
    return res, texts

t0 = time.time()
ref = load()
ref_lp, (ref_ok, ref_txt) = answer_logprobs(ref), task_correct(ref)
print(f"reference (bf16 weights, fp32 compute): accuracy {sum(ref_ok)}/{len(EVAL)} = {sum(ref_ok)/len(EVAL):.1%}"
      f"  [{time.time()-t0:.0f} s]")
print("  sample:", repr(EVAL[0][0]), "->", repr(ref_txt[0]))

print(f"\n{'candidate':12s} {'KL nats/tok':>11s} {'top1 agree':>10s} {'dNLL/tok':>9s} {'acc':>6s} {'R->W':>4s} {'W->R':>4s}  verdict")
for cname, fn in (("fp8", q_fp8), ("nvfp4", q_nvfp4), ("int4-tensor", q_int4_tensor)):
    cand = quantize(load(), fn)
    kl = agree = dnll = n = 0.0
    for (lr, ids), (lc, _) in zip(ref_lp, answer_logprobs(cand)):
        kl += (lr.exp() * (lr - lc)).sum().item()                    # sum over positions of KL(ref || cand)
        agree += (lr.argmax(-1) == lc.argmax(-1)).sum().item()
        dnll += (lr.gather(1, ids[:, None]) - lc.gather(1, ids[:, None])).sum().item()
        n += len(ids)
    ok, txt = task_correct(cand)
    rw = sum(r and not c for r, c in zip(ref_ok, ok)); wr = sum(c and not r for r, c in zip(ref_ok, ok))
    acc = sum(ok) / len(EVAL)
    fails = [why for bad, why in ((kl / n > MAX_KL, "KL"), (agree / n < MIN_TOP1, "top1"),
             (sum(ref_ok) / len(EVAL) - acc > MAX_ACC_DROP, "acc"), (rw - wr > MAX_NET_FLIPS, "flips")) if bad]
    print(f"{cname:12s} {kl/n:11.4f} {agree/n:10.1%} {dnll/n:+9.4f} {acc:6.1%} {rw:4d} {wr:4d}  "
          f"{'PASS' if not fails else 'FAIL (' + ', '.join(fails) + ')'}")
    for i in [i for i, (r, c) in enumerate(zip(ref_ok, ok)) if r != c][:2]:
        print(f"    {EVAL[i][0]!r}: ref {ref_txt[i]!r} | cand {txt[i]!r}")
print(f"[total {time.time()-t0:.0f} s]")
```

```console
$ /tmp/absorb-venv/bin/python quality_gate.py
reference (bf16 weights, fp32 compute): accuracy 31/40 = 77.5%  [21 s]
  sample: 'What is 7 times 8?' -> '64'

candidate    KL nats/tok top1 agree  dNLL/tok    acc R->W W->R  verdict
fp8               0.0293      91.5%   +0.0446  82.5%    0    2  PASS
    'What is the capital of Italy?': ref 'Florence.' | cand 'The capital of Italy is Rome.'
    'What is the capital of Kenya?': ref 'Khartoum' | cand 'The capital of Kenya is Nairobi.'
nvfp4             0.2381      81.9%   -0.2596  95.0%    0    7  FAIL (KL, top1)
    'What is 7 times 8?': ref '64' | cand '7 x 8 = 56.'
    'What is 100 minus 37?': ref '93' | cand 'The result of 100 minus 37 is 63.'
int4-tensor      16.3061       0.0%  +13.2109   2.5%   30    0  FAIL (KL, top1, acc, flips)
    'What is 12 plus 15?': ref '12 + 15 = 27' | cand ' ".$_steen bådeiationExceptionallisoisrishayoutenussoles okoerus schöne acompañaatorialfrage'
    'What is 9 times 9?': ref '9 squared equals 81.' | cand '石膏ák益 décidé岙蒴🏽ье ".$_enção部副 INDIRECT调剂swireträgthole'
[total 997 s]
```

`dNLL/tok` is the candidate's negative log-likelihood of the reference answers minus the reference model's: positive means the candidate finds the right answers less likely.

The three verdicts are the three kinds of result a gate sees in practice:

- **FP8 passes**, with a mean KL of 0.029 nats per token and 91.5% top-1 agreement. It also flipped two items from wrong to right. That is noise, not an improvement: on 40 items, the standard error of a 77.5% accuracy is $\sqrt{0.775 \times 0.225 / 40} = 6.6$ points.
- **NVFP4 fails, and scores higher.** Accuracy went from 77.5% to 95.0%, seven items flipped wrong→right and none the other way, and the gate still says no, because the KL is 0.24 and the top-1 agreement 82%. Look at the outputs: the reference answers tersely, as the prompt asked ("64"); the NVFP4 model answers in sentences that show the working ("7 x 8 = 56."), and the containment check rewards that. The candidate is a *different model*. A customer whose product parses one-word answers would see breakage where the accuracy column shows a gain. That is the reason the gate measures sameness first and the task metric second, and the reason a score that rises after quantization is a reason to look at outputs, not to ship. (A 0.5B model is also much more sensitive to 4-bit rounding than the 70B-class models vendors report on; on this model, this simulation says NVFP4 needs quantization-aware training or a different recipe.)
- **The broken INT4 produces garbage**, and every check fails. One scale per tensor lets a single outlier weight set the step size for millions of others ([why group scales exist](../inference/arithmetic.md#quantization-from-scratch-scales-zero-points-groups)). A deploy pipeline should never get this far with one, but a conversion bug, a wrong config or a tokenizer mismatch produces outputs that look exactly like this, and the gate is the cheapest place to catch it.

The thresholds above are illustrations. For a real customer: an eval of several hundred items or more, so a 2-point threshold is above the noise ([lora_or_full.py](#deciding-with-evals) showed what 60 items can and can't decide); KL and top-1 measured on their eval and on a sample of their production prompts; and the decoding settings their product uses. Also run the gate on the *served* path once the candidate is staged: the same prompts through the real engine at the batch sizes it serves, because kernels and batch shape change numerics ([numerics across hardware](operating-inference.md#numerics-differences-across-hardware)).

### Stage, roll, canary, roll back

From here on, the deploy chapter's machinery applies unchanged. On two nodes it has three peculiarities.

- **Stage on both nodes before starting.** Each B300 node's NVMe takes a 70B FP8 checkpoint (73 GB) in about 6 s at 12 GB/s ([loadmath.py](deploys-recovery-autoscaling.md#weights)), so the replica start, not the copy, is the critical path. Copy to `/mnt/nvme/models/<customer>/<version>/`, write `.complete` last, keep the previous version for rollback, and keep the weights encrypted at rest ([section 6](#6-customer-isolation)).
- **There is no surge capacity.** With 15 of 16 GPUs placed, a rolling update can't add a replica before removing one, so it runs with `maxSurge: 0, maxUnavailable: 1` per pool, and the pool runs one replica short while each replica restarts ([surge and unavailability](deploys-recovery-autoscaling.md#surge-and-unavailability-on-a-full-fleet)). Roll during the trough. The pool's 30% headroom from the planner is what makes one-short safe for big pools. A small pool is different: the 8B pool's two replicas run at 60% busy at peak, so while one restarts the other would need 120% of a replica. Roll small pools at night.
- **For a LoRA, the canary is an adapter, not a replica.** Load the new adapter version under a new name (`acme-v8`) next to `acme-v7`, send a percentage of the customer's traffic or only the canary prompts to it at the gateway, compare, then switch the name the customer's key resolves to. Rollback is switching the name back. Nothing restarts.

```bash
# untested. LoRA canary on a running vLLM pool: load v8 beside v7, probe it, then flip the alias in the gateway.
# Runtime loading needs the server started with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True; keep these endpoints internal.
curl -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "acme-v8", "lora_path": "/mnt/nvme/adapters/acme/v8"}'
python3 probe.py --model acme-v8 --golden /mnt/nvme/adapters/acme/v8/golden.jsonl   # deploys chapter's probe
# gateway: acme -> acme-v8 for 5% of acme's requests; watch acme's SLIs and quality signals; then 100%
curl -X POST localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name": "acme-v7"}'
```

Canary signals are the ones from [Operating Inference](operating-inference.md#continuous-evals-and-canary-prompts-in-production): per-replica (or per-adapter) canary prompts with checkable answers, error and latency SLIs, and a passive check on live outputs. Gate the promotion on all three, and keep the previous version loaded until the new one has served a full daily cycle.

## 5. Training and serving on the same 16 GPUs

The company trains models too: customers' fine-tunes, its own evals, sometimes an RL run ([RL infrastructure](rl-infrastructure.md)). On 16 GPUs it can't afford a separate training cluster, so training and serving share the fleet. The question is how.

### Three policies

1. **Static split.** Some GPUs serve, the rest train, always. Simple, and wasteful in both directions: serving is sized for peak and idle at night; training never gets more than its share.
2. **Time-sharing.** Serving gets the fleet during the day, training gets most of it at night. Matches a diurnal load shape, but it is a schedule, not a response: a 3 a.m. burst from a customer's batch job finds the serving side at night size.
3. **Preemptible training.** Serving takes what it needs plus a margin, training gets the rest, and serving can take GPUs back at any time. Training jobs must checkpoint quickly and restart at a different size. This is the [move-GPUs-between-models](deploys-recovery-autoscaling.md#on-a-fixed-fleet-move-gpus-between-models) idea with training as the second tenant, and the [preemption with priorities](control-planes.md#constraints-affinity-spreading-and-preemption) the control-planes chapter describes.

### What a switch costs

A GPU moving from training to serving is useless for: the time to stop the training job cleanly and save its state, plus the serving replica's $T_{\text{start}}$. From the [MTTR arithmetic](deploys-recovery-autoscaling.md#mttr-arithmetic), $T_{\text{start}}$ for a 70B replica is about 2.3 minutes with weights pre-staged on NVMe and a cached compile, and 13.6 minutes cold, from a network disk without caches. With a one-minute checkpoint save, a switch costs 3.3 minutes pre-staged and 14.6 minutes cold. For four GPUs that is $4 \times 3.3 / 60 = 0.22$ GPU-hours or $4 \times 14.6 / 60 = 0.97$ GPU-hours per switch. Small per switch; what matters is how long serving runs short meanwhile.

The other direction (serving to training) costs a drain, bounded by the longest generation you allow ([draining](deploys-recovery-autoscaling.md#draining-in-flight-streams)), plus restarting the training job at its new size: loading the checkpoint, rebuilding the process group. The whole job pays that, not just the GPUs that moved.

How long a save takes depends on what is saved. A full fine-tune of the 70B model with AdamW in mixed precision keeps about 16 bytes per parameter: BF16 weights and gradients (4), an FP32 master copy (4) and AdamW's two FP32 moments (8, from the [optimizer-state table](../training/arithmetic.md#optimizer-state-memory)). That is $70.55 \times 10^9 \times 16 = 1.13$ TB across the job, of which roughly 1 TB (everything but the gradients) is checkpointed. Spread over a node's 8 local NVMe drives (8 × 3.84 TB on a DGX B300), at an assumed aggregate 20 GB/s of writes, that is about 50 s. A LoRA job's state is the adapter and its optimizer state: for a rank-64 adapter on the 70B model, 1.7 GB of BF16 weights ([planner.py](#a-placement-planner)) and a few GB with optimizer state, saved in seconds. **LoRA training jobs are the ideal preemptible tenant**: small state, fast restarts, and they usually fit on one or a few GPUs.

### Checkpoint on preempt

Without a save on preemption, a preempted job loses everything since its last periodic checkpoint: half the interval on average. With a 30-minute interval that is 15 minutes of the whole job's GPUs, every time serving reclaims capacity. The fix is the one the training chapters use for Slurm preemption: catch a signal, save, exit ([graceful stops](../training/fault-tolerance-and-reproducibility.md#graceful-stops-kill-and-save-switches), [forced preemption](../training/fault-tolerance-and-reproducibility.md#dealing-with-forced-job-preemption)). In this setup the "scheduler" is the company's own control loop. It sends SIGUSR1 (or writes a save-switch file) to the training job, waits for the save to finish or a deadline (two minutes) to pass, and only then starts the serving replica. Write checkpoints to local NVMe first and copy them to object storage in the background, so the save is bounded by NVMe speed.

```bash
# untested. The reclaim sequence the control loop runs for one 4-GPU block.
kill -USR1 "$(cat /run/train/kestrel-ft.pid)"                  # training job saves and exits on SIGUSR1
timeout 120 bash -c 'until [ -f /mnt/nvme/ckpt/kestrel-ft/latest/.complete ]; do sleep 2; done' \
  || echo "save missed its deadline: reclaiming anyway, job resumes from its periodic checkpoint"
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /mnt/nvme/models/llama-3.3-70b-fp8 --port 8004 ...   # replica starts
```

### Filling idle serving GPUs

Between preemptible training and the serving peak there is a second filler: batch inference and eval jobs on the serving replicas themselves, at low priority, capped in flight ([backfill](../inference/serving-efficiently.md#priority-filling-idle-capacity-with-batch-work), [batch with deadlines](operating-inference.md#filling-idle-capacity-with-a-deadline-queue)). Batch work is finer-grained than training: it fills a replica's idle fraction request by request, so it uses the capacity a 4-GPU block granularity can't give to training. The company's own work qualifies (evals of every candidate model, synthetic data generation, RL rollouts), and so does a customer-facing batch tier sold at a discount.

### A simulator

`share_sim.py` runs one synthetic weekday on 16 GPUs moved in blocks of 4. Serving demand (GPUs needed at the SLO) follows a daily curve peaking at 13 GPUs, a little under the 15 that `planner.py` places with headroom, plus three customer bursts, one of them at 3 a.m. The policies are the three above, the preemptible one in four variants: cold vs pre-staged serving starts, with and without checkpoint-on-preempt, and with batch backfill using 70% of idle serving capacity. The switch costs are the deploy chapter's $T_{\text{start}}$ values and stated assumptions.

```python
# share_sim.py - 16 GPUs shared by serving and training: static split, night training, preemptible
# training (with and without checkpoint-on-preempt and pre-staged serving), and batch backfill.
#   python3 share_sim.py
# One step = 5 min, one synthetic weekday. GPUs move in blocks of 4 (a TP group or half a node).
import math, random

GPUS, BLOCK, STEP = 16, 4, 5                      # GPUs, GPUs per block, minutes per step
STEPS = 24 * 60 // STEP
rng = random.Random(7)

def need(t):                                       # serving GPUs needed at the SLO (planner.py: ~15 at peak)
    h = t * STEP / 60
    base = 3.5 + 8.5 * max(0.0, math.sin(math.pi * (h - 7) / 15)) ** 1.5  # quiet 22:00-07:00, peak mid-afternoon
    return base * rng.uniform(0.9, 1.1)
NEED = [need(t) for t in range(STEPS)]
for start_h, mins, extra in ((3.0, 40, 5), (10.5, 30, 4), (19.0, 45, 5)):   # customer bursts: a backfill job, a launch
    for t in range(int(start_h * 60 / STEP), int((start_h * 60 + mins) / STEP)):
        NEED[t] += extra
NEED = [min(GPUS, n) for n in NEED]

# switch costs in minutes. Serving start: deploys chapter recovery.py (pre-staged 2.3 min, cold 13.6 min).
DRAIN = 1.0           # finish in-flight streams before a serving block leaves (assumed cap on generation time)
TRAIN_RESTART = 3.0   # training job re-launch at a new size: load checkpoint, NCCL init (assumed)
CKPT_SAVE = 1.0       # checkpoint-on-preempt: save sharded state to local NVMe (assumed)
CKPT_EVERY = 30.0     # periodic checkpoint interval when there is no save-on-preempt
BACKFILL = 0.7        # fraction of idle serving GPU-time batch/eval work can use (assumed; serving-efficiently backfill.py)

def run(policy, t_start=2.3, save_on_preempt=True, backfill=False):
    serve_blocks = {"static": 3, "night": 1}.get(policy, 4)   # blocks serving at midnight
    dead = [0.0] * (GPUS // BLOCK)                 # minutes left of switching for each block (useless meanwhile)
    surplus_steps = 0
    short = served = train = batch = lost = short_min = 0.0
    switches = 0
    for t in range(STEPS):
        h = t * STEP / 60
        n = NEED[t]
        if policy == "static":
            want = 3
        elif policy == "night":
            want = 1 if h < 8 else 4
        else:                                      # preemptible: serving gets need + 2 GPUs margin, rounded up
            want = min(4, math.ceil((n + 2) / BLOCK))
            if want < serve_blocks:                # give blocks back to training only after 30 min of surplus
                surplus_steps += 1
                if surplus_steps < 30 // STEP:
                    want = serve_blocks
            else:
                surplus_steps = 0
        if want != serve_blocks:
            switches += 1
            moved = abs(want - serve_blocks)
            if want > serve_blocks:                # training -> serving: save (or lose) work, then start serving
                cost = (CKPT_SAVE if save_on_preempt else 0) + t_start
                lost += (CKPT_EVERY / 2 / 60) * BLOCK * (4 - serve_blocks) * (not save_on_preempt)
            else:                                  # serving -> training: drain
                cost = DRAIN
            for b in range(min(want, serve_blocks), max(want, serve_blocks)):
                dead[b] = cost
            for b in range(max(want, serve_blocks), 4):   # the whole training job restarts at its new size
                dead[b] = max(dead[b], TRAIN_RESTART + (CKPT_SAVE if save_on_preempt and want > serve_blocks else 0))
            serve_blocks = want
        live = [max(0.0, 1 - dead[b] / STEP) for b in range(4)]    # usable fraction of this step per block
        dead = [max(0.0, d - STEP) for d in dead]
        cap = BLOCK * sum(live[:serve_blocks])
        hrs = STEP / 60
        served += min(n, cap) * hrs
        short += max(0.0, n - cap) * hrs
        short_min += STEP * (n > cap + 1e-9)
        train += BLOCK * sum(live[serve_blocks:]) * hrs
        if backfill:
            batch += BACKFILL * max(0.0, cap - n) * hrs
    train -= lost
    return served, short, short_min, train, batch, switches

print(f"serving need: mean {sum(NEED)/STEPS:.1f} GPUs, peak {max(NEED):.1f}, trough {min(NEED):.1f}; "
      f"day = {GPUS*24} GPU-h")
print(f"{'policy':42s} {'served':>6s} {'unmet':>5s} {'short min':>9s} {'train':>5s} {'batch':>5s} {'useful':>6s} {'moves':>5s}")
rows = [("static 12 serve / 4 train", dict(policy="static")),
        ("night training (00-08: 4 serve / 12 train)", dict(policy="night")),
        ("preemptible, cold serving start 13.6 min", dict(policy="preempt", t_start=13.6)),
        ("preemptible, no checkpoint-on-preempt", dict(policy="preempt", save_on_preempt=False)),
        ("preemptible, pre-staged start 2.3 min", dict(policy="preempt")),
        ("preemptible + batch backfill", dict(policy="preempt", backfill=True))]
res = {}
for name, kw in rows:
    s, u, sm, tr, b, sw = res[name] = run(**kw)
    print(f"{name:42s} {s:6.1f} {u:5.1f} {sm:9.0f} {tr:5.1f} {b:5.1f} {(s+tr+b)/(GPUS*24):6.0%} {sw:5d}")
print("(GPU-hours per day; 'useful' = served + training + batch over 384 GPU-h)")
assert res["preemptible + batch backfill"][1] <= res["static 12 serve / 4 train"][1] + 1e-9
```

```console
$ python3 share_sim.py
serving need: mean 6.8 GPUs, peak 13.1, trough 3.2; day = 384 GPU-h
policy                                     served unmet short min train batch useful moves
static 12 serve / 4 train                   162.7   1.1       130  96.0   0.0    67%     0
night training (00-08: 4 serve / 12 train)  160.7   3.1        60  96.0   0.0    67%     1
preemptible, cold serving start 13.6 min    163.6   0.2        25 103.8   0.0    70%    10
preemptible, no checkpoint-on-preempt       163.8   0.0         0  96.9   0.0    68%    10
preemptible, pre-staged start 2.3 min       163.8   0.0         0 103.8   0.0    70%    10
preemptible + batch backfill                163.8   0.0         0 103.8  79.9    90%    10
(GPU-hours per day; 'useful' = served + training + batch over 384 GPU-h)
```

`served` and `unmet` are GPU-hours of serving demand met and missed; `short min` is the minutes in which serving had less capacity than it needed, which is what an SLO sees; `train` is useful training GPU-hours after restarts and lost work; `useful` is everything useful over the 384 GPU-hours in a day.

What it says:

- **The static split misses its SLO for 130 minutes a day** while giving training only 4 GPUs. Twelve serving GPUs cover most of the day, but the afternoon peak and the bursts land on a fixed size. Only 1.1 GPU-hours go unmet, which looks small, and is spread over two hours of degraded service.
- **Night training is worse for the 3 a.m. burst** (3.1 GPU-hours unmet, all in the hour when serving is at night size) and gives training no more GPU-hours than the static split. A schedule can't react.
- **Preemptible training with pre-staged serving meets every minute of demand and gives training 8% more** (103.8 vs 96 GPU-hours). With cold serving starts the bursts catch the fleet short for 25 minutes; pre-staging weights and compile caches on every node is what makes preemption safe for the SLO. Without checkpoint-on-preempt, training loses 6.9 GPU-hours a day to work it has to redo, which wipes out most of preemption's gain.
- **Training's gain is small; batch's is large.** Preemption moves whole 4-GPU blocks, keeps a 2-GPU margin and waits 30 minutes before giving capacity back, so much of the idle serving capacity stays idle: average demand is 6.8 GPUs, but serving holds about 11.7 on average ($16 - 103.8/24$, counting GPUs mid-switch as serving). Batch work on the serving replicas uses that idle fraction request by request, and takes the day from 70% useful to 90%. On 16 GPUs, the order of priorities is: pre-stage serving (so preemption is safe), add checkpoint-on-preempt (so it is cheap), and fill serving's idle time with batch and evals (where most of the gain is).

The block size is the lever this simulator holds fixed. With TP1 serving replicas, blocks could be single GPUs, and training jobs that fit on 1-2 GPUs (most LoRA fine-tunes of an 8B model, and a 70B LoRA with the base in FP8) could take GPUs one at a time. Try `BLOCK = 1` with a matching `want`; the simulator is short enough to change.

## 6. Customer isolation

Customers are startups who are often competitors of each other. Every piece of shared state between them is a possible leak, and on a multi-tenant fleet almost everything is shared. The controls, in the order a small provider should build them:

### Prefix caches: salt per tenant

A prefix-cache hit has a lower TTFT, so a client that can time its requests can test whether some prefix is in the cache, which means someone else sent it. Song et al. built this into an attack: "a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches", and showed "the feasibility of stealing system prompts and those produced by peer users" ([arXiv 2409.20002](https://arxiv.org/abs/2409.20002)). The fix is a per-tenant salt mixed into the first block hash, which vLLM exposes as a per-request `cache_salt` that "prevents timing-based attacks where an adversary could infer cached content by observing latency differences" ([vLLM prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html)). [Operating Inference](operating-inference.md#isolation-between-tenants-adapters-and-caches) has the mechanism and a runnable demo (`cache_salt.py`).

For this company: the gateway sets `cache_salt` from the organization ID on every request and ignores any value the client sends. LoRA pools need it as much as base-model pools: the adapter ID is already part of the block hash ([Serving Efficiently](../inference/serving-efficiently.md#multi-lora-many-fine-tunes-on-one-base)), which separates tenants on different adapters but not two keys of the same organization, nor two tenants on the base model. The cost is losing cross-tenant cache sharing of identical prefixes, such as a popular agent framework's system prompt. Share across tenants only prefixes the company publishes itself. Salt CPU-offloaded KV ([KV offloading](../inference/serving-efficiently.md#kv-offloading-to-cpu-or-disk)) the same way: it is the same cache on slower memory.

### Weights: separate storage and encryption

A customer's fine-tuned weights are their product. Treat them like customer data, not like build artifacts:

- **One storage prefix per customer, one encryption key per customer**, in an object store with server-side encryption under customer-specific keys, and an access policy that lets only the deploy pipeline's role read it. A per-customer key makes deletion at contract end a key deletion that you can prove ("crypto-shredding"), and limits what one leaked credential exposes.
- **On the nodes**, keep `/mnt/nvme/models` and `/mnt/nvme/adapters` on an encrypted volume (dm-crypt/LUKS), so a drive returned in an RMA ([dealing with accelerator failures](getting-compute.md#dealing-with-accelerator-failures)) carries no readable weights. Ask the provider what happens to drives when a node is swapped; it belongs in the contract ([security questions](getting-compute.md#security)).
- **Load adapters only from your own storage**, by an ID the gateway resolves from the customer's key, never from a path or URL in the request ([adapters](operating-inference.md#isolation-between-tenants-adapters-and-caches)). Namespace adapter names by organization (`acme/support-v8`), so two tenants can't collide and an error message for another tenant's adapter is a 404.
- **Know where each customer's weights are allowed to go.** The overflow plan in [section 7](#7-planning-for-the-loss-of-a-node) moves models to another provider during an outage. That is a transfer of the customer's weights to a third party and needs to be in their contract, or excluded for them.

### Logging and retention per contract

[Operating Inference](operating-inference.md#prompt-and-response-logging-and-retention) sets the default: log metadata (token counts, timings, finish reasons, adapter, replica), not content. On a small provider, the contract decides the exceptions, so make them configuration rather than code: a per-organization record with `log_content` (off by default; on only for a time-limited debugging window the customer requested), `retention_days` for each store (request log, batch inputs and outputs, traces, idempotency records), and `training_use` (whether their prompts may be used for calibration or anything else; default no). The pipeline in [section 4](#convert-and-quantize) reads `training_use` before it takes a calibration sample from production logs. Test deletion: a script that, for a given organization, finds every copy in every store and checks that its age is under the retention period.

### API keys and scopes

Keys are stored as hashes, carry a public ID, and are scoped by endpoint and by model, as in [api_keys.py](operating-inference.md#api-keys-and-their-scope). For this company, add two scopes on top of that chapter's list: `train` (start fine-tuning jobs, which spend GPU-hours, the scarcest thing it has) and `weights:download` (export a customer's own fine-tuned weights, if the contract allows it). A CI key gets `inference` on one model and nothing else. Every key belongs to an organization, and the organization ID is what drives the cache salt, the adapter namespace, the rate limits and the logging policy, so the gateway never trusts any of those from the request itself.

### Noisy-neighbour limits

On a shared pool, one customer's traffic changes everyone's latency ([noisy neighbours](operating-inference.md#dedicated-vs-shared-deployments-noisy-neighbours)). With 16 GPUs there is no spare pool to move a noisy customer to, so the limits have to be in place before the first one appears:

- **Per-organization limits on in-flight tokens**, not just requests, so a tenant's long contexts can't fill a pool's KV cache ([KV pressure](operating-inference.md#dedicated-vs-shared-deployments-noisy-neighbours)). For the 70B LoRA pool, with 5 replicas of 59 × 16k-token sequences, a cap of one fifth of the pool's KV per organization leaves any single customer able to fill at most one replica's worth.
- **A cap on prompt length per tier** and chunked prefill sized so a 100k-token prompt can't stall a shared batch for seconds ([neighbour.py](operating-inference.md#dedicated-vs-shared-deployments-noisy-neighbours) showed one long prompt making 32 other streams' ITL 3.6-11x worse for over two seconds).
- **Fair queuing across organizations** at the gateway instead of FCFS, keyed on organization ([Virtual Token Counter](operating-inference.md#virtual-token-counter)), and per-organization TPM limits with reserve-and-refund ([tpm_limiter.py](operating-inference.md#estimating-tokens-before-generation-reservation-and-refund)).
- **Ramp limits** for new customers and for step changes in traffic, so one launch can't take a pool ([operations checklist](operating-inference.md#operations-checklist-for-a-leased-fleet), item 4).
- **Per-adapter concurrency in LoRA pools**: `--max-loras` caps distinct adapters per step, so a burst from many small customers' adapters at once queues requests; give the pool `--max-loras` at least the number of adapters active in a typical second of your trace ([Serving Efficiently](../inference/serving-efficiently.md#multi-lora-many-fine-tunes-on-one-base)).

## 7. Planning for the loss of a node

### What a node failure costs

With two nodes, one failure is half the fleet. The deploy chapter's [MTTR arithmetic](deploys-recovery-autoscaling.md#mttr-arithmetic) uses about one interruption per 50,000 GPU-hours (the Llama 3 training run, via the [failure-rate arithmetic](../orchestration/internals.md#failure-rate-arithmetic)), and treats any GPU's failure as taking its node out, since a node with a faulty GPU is drained whole ([drain the whole host](deploys-recovery-autoscaling.md#drain-the-whole-host-or-replace-one-gpu)). For two 8-GPU nodes that is a node MTBF of 6,250 hours and 2.8 node failures a year: a 21% chance of at least one in any given month.

How long the node stays out depends on the provider's replacement process, which is the first of the [three most important questions](getting-compute.md#the-three-most-important-questions) to ask before signing. The script below prices the three ways to cover the gap: a third node (N+1), bursting to someone else's GPUs, and degrading by customer tier.

```python
# failure_plan.py - two nodes, one fails: what each mitigation costs per year, and what uptime to promise.
#   python3 failure_plan.py
# Failure rate: ~1 interruption per 50,000 GPU-hours (Llama 3, via the deploys chapter's recovery.py).
# Prices: lease is an ASSUMPTION; burst prices are list prices read November 2026 (links in the text).
import math

GPUS_PER_NODE, NODES = 8, 2
MTBF_NODE_H = 50_000 / GPUS_PER_NODE             # any one GPU failing takes its node's replicas out
LEASE = 4.00                                      # $/GPU-hour, 1-year B300 lease (assumption)
BURST = {"per-second serverless B300 (Modal list)": 0.001972 * 3600,
         "on-demand B300 (Prime Intellect list)": 6.64,
         "dedicated endpoint B300 (Together list)": 11.49}
HOURS = 8760

fails = NODES * HOURS / MTBF_NODE_H
print(f"node MTBF {MTBF_NODE_H:,.0f} h -> {fails:.1f} node failures/year on {NODES} nodes; "
      f"P(>=1 in a 30-day month) = {1 - math.exp(-NODES * 720 / MTBF_NODE_H):.0%}")

print("\nhours per year with only one node, by provider repair turnaround")
for repair in (24, 72, 168):
    down = fails * repair
    both = fails * (repair / MTBF_NODE_H) * repair / 2     # the other node fails mid-repair; mean overlap repair/2
    print(f"  repair {repair:3d} h: {down:5.0f} h at half capacity ({down/HOURS:.2%} of the year); "
          f"both down ~{both*60:.1f} min/year")

print("\ncost per year of covering the lost node (8 GPUs) for the whole repair")
node3 = GPUS_PER_NODE * LEASE * HOURS
print(f"  N+1: a third leased node                     ${node3:>9,.0f}  (fixed; it can run training and batch meanwhile)")
for name, price in BURST.items():
    for repair in (24, 168):
        hrs = fails * repair
        print(f"  burst, {name:40s} repair {repair:3d} h: ${GPUS_PER_NODE * price * hrs:>8,.0f}")

# degradation tiers: with 8 GPUs left, who gets served? (peak GPU needs from planner.py's LORA_FIRST)
tiers = [("tier 1: dedicated + SLA", ["70b-ft-kestrel", "3b-ft-rook", "1.5b-ft-sable"], 2 + 0.29 + 0.29),
         ("tier 2: shared API, LoRA pools", ["8b-lora-pool", "70b-lora-pool", "glm-air-api"], 2 + 5 + 4),
         ("tier 3: best effort, batch", ["1.5b-ft-wren", "batch/eval"], 0.14)]
left = GPUS_PER_NODE
print("\none node down at peak: 8 GPUs left, filled in tier order")
for name, who, gpus in tiers:
    got = min(left, gpus); left -= got
    print(f"  {name:32s} needs {gpus:5.2f} GPUs at peak, gets {got:5.2f} ({got/gpus:.0%})")

# what a single site can promise: node failures are covered, the site is not
print("\nmonthly uptime of 'some capacity is serving' if a site-wide outage (power, network, a bad")
print("fleet-wide deploy) happens k times a year for h hours each:")
for k, h in ((1, 4), (2, 8), (4, 12)):
    print(f"  {k} x {h:2d} h/year -> worst affected month {1 - h / 720:.3%}, annual {1 - k * h / HOURS:.3%}")

# Modal's rule from the deploys chapter: reserve capacity b only while P(demand > b) >= R_r / R_s
for name, price in BURST.items():
    print(f"node 3 pays for itself vs {name:40s} if needed > {LEASE / price:.0%} of hours")
assert node3 > GPUS_PER_NODE * max(BURST.values()) * fails * 168   # burst is cheaper even at week-long repairs
```

```console
$ python3 failure_plan.py
node MTBF 6,250 h -> 2.8 node failures/year on 2 nodes; P(>=1 in a 30-day month) = 21%

hours per year with only one node, by provider repair turnaround
  repair  24 h:    67 h at half capacity (0.77% of the year); both down ~7.8 min/year
  repair  72 h:   202 h at half capacity (2.30% of the year); both down ~69.8 min/year
  repair 168 h:   471 h at half capacity (5.38% of the year); both down ~379.8 min/year

cost per year of covering the lost node (8 GPUs) for the whole repair
  N+1: a third leased node                     $  280,320  (fixed; it can run training and batch meanwhile)
  burst, per-second serverless B300 (Modal list)  repair  24 h: $   3,821
  burst, per-second serverless B300 (Modal list)  repair 168 h: $  26,746
  burst, on-demand B300 (Prime Intellect list)    repair  24 h: $   3,574
  burst, on-demand B300 (Prime Intellect list)    repair 168 h: $  25,016
  burst, dedicated endpoint B300 (Together list)  repair  24 h: $   6,184
  burst, dedicated endpoint B300 (Together list)  repair 168 h: $  43,289

one node down at peak: 8 GPUs left, filled in tier order
  tier 1: dedicated + SLA          needs  2.58 GPUs at peak, gets  2.58 (100%)
  tier 2: shared API, LoRA pools   needs 11.00 GPUs at peak, gets  5.42 (49%)
  tier 3: best effort, batch       needs  0.14 GPUs at peak, gets  0.00 (0%)

monthly uptime of 'some capacity is serving' if a site-wide outage (power, network, a bad
fleet-wide deploy) happens k times a year for h hours each:
  1 x  4 h/year -> worst affected month 99.444%, annual 99.954%
  2 x  8 h/year -> worst affected month 98.889%, annual 99.817%
  4 x 12 h/year -> worst affected month 98.333%, annual 99.452%
node 3 pays for itself vs per-second serverless B300 (Modal list)  if needed > 56% of hours
node 3 pays for itself vs on-demand B300 (Prime Intellect list)    if needed > 60% of hours
node 3 pays for itself vs dedicated endpoint B300 (Together list)  if needed > 35% of hours
```

### The three options, priced

**N+1, a third node.** At the assumed lease price it costs \$280,320 a year, half again the current fleet, and covers a node failure completely: MTTR becomes a router ejection ([N+1 replicas](deploys-recovery-autoscaling.md#mttr-arithmetic)). It isn't idle meanwhile; it runs training and batch. As insurance alone it is the most expensive option by a factor of 10 to 78. It makes sense when there is enough work to keep it busy anyway, which is [section 9](#9-what-to-lease-next)'s question.

**Overflow to rented GPUs.** Covering 8 GPUs for the whole repair costs \$3,600 to \$6,200 a year at a 24-hour turnaround and \$25,000 to \$43,000 at a week, at the list prices in the script. Even at the worst combination it is a sixth of the third node. This is the [deploy chapter's burst rule](deploys-recovery-autoscaling.md#when-serverless-beats-reserved) applied to failures: capacity needed a few percent of the time should be rented when needed. The catches are not in the price:

- **Availability isn't guaranteed.** Per-second and on-demand B300 capacity may not be there at the moment a node fails. Have accounts and quotas set up at two providers in advance, and test that you can get 8 GPUs.
- **Cold start, paid by users.** A fresh cloud GPU has none of your caches. Together measured 145 s from create to READY for an 18 GB custom fine-tune plus 26-40 s more until the first token through routing ([deploys chapter](deploys-recovery-autoscaling.md#what-published-breakdowns-say)); a 70B model from object storage takes longer. Keep a pre-built image and the weights of tier-2 pools in the burst provider's region, encrypted, so the burst start is minutes and not an hour.
- **Weights leave your infrastructure.** For LoRA pools and the base-model API that's your own base weights plus customer adapters; for dedicated customers it's their model. The contract has to allow it ([section 6](#weights-separate-storage-and-encryption)). Customers who don't allow it get the degradation tier instead.
- **The same model, exactly.** Same revision, same quantization, same chat template, same tokenizer, checked with the canary prompts ([multi-region and failover](operating-inference.md#multi-region-and-failover)). A fallback that serves a subtly different model is a silent quality incident.
- **For base-model traffic, per-token serverless may be cheaper than GPUs.** If the GLM-4.5-Air API's traffic can go to a provider that serves the same model per token, the failover costs only the tokens it carries. Check the model is on their list; Together's serverless page, for example, lists gpt-oss-120b at \$0.15 per million input tokens and \$0.60 per million output tokens, but not GLM-4.5-Air ([pricing](https://www.together.ai/pricing)).

**Degradation by tier.** With one node down at peak, 8 GPUs are left. Tier-1 customers (dedicated replicas and paid SLAs) need 2.58 GPUs and get all of them, because the planner put one of each of their replicas on each node. The shared pools need 11 GPUs and get 5.42: about half their peak capacity, enforced by the gateway's per-organization limits (customers get 429s with `Retry-After` past their reduced share, not timeouts; [429 vs 503](operating-inference.md#what-the-client-sees-on-overload-429-vs-503)). Best effort gets nothing until the node is back. Peak lasts a few hours a day; at the 3 a.m. trough the surviving node carries everything. Degradation costs nothing extra in GPUs and costs goodput, SLO budget and possibly SLA credits for tier 2 at peak, which is why the tiers have to be in the contracts before the first failure.

The combination that fits a 2-node company: **tier-1 customers placed across both nodes (survive on their own), overflow for tier-2 pools to rented GPUs where contracts allow, degradation for everything else, and no third node bought as insurance.**

### What to put in customer SLAs

The second half of the script is the important one for contracts. A node failure is covered, because tier-1 replicas are on both nodes. A *site* failure is not: both nodes are in one data center, on one provider's network, behind one gateway the company runs. Power, network, the provider's control plane or a bad fleet-wide deploy of your own take everything down at once. One 4-hour site outage in a year leaves the affected month at 99.44%; two 8-hour outages leave it at 98.9%.

That matches what [Operating Inference](operating-inference.md#what-999-vs-9999-means-and-the-capacity-it-needs) concluded for one data center, "A fleet in one data center can honestly offer 99%, maybe 99.5%", and what Together contracts for its own single-data-center deployments, an SLA of 99% (the 99.9% tier needs two facilities). What the company can offer, written in the language of [SLOs per tier](operating-inference.md#slos-per-tier):

| Tier | Who | Availability SLA (monthly) | Capacity during a node failure | Notes |
|---|---|---|---|---|
| 1 | dedicated replicas, enterprise | 99.5% | 100% of contracted replicas | replicas on both nodes; credits if missed |
| 2 | shared API and LoRA pools | 99% | best effort, at least ~50% of peak rate limits | per-org limits shrink proportionally |
| 3 | best effort, batch | none | none until repair | batch deadlines in days, not hours |

Define availability on the [goodput SLI measured at completion](operating-inference.md#measure-at-completion-not-at-the-gateway), not on "the endpoint answered", so a degraded hour counts honestly. Exclude announced maintenance windows, and announce them: driver and engine upgrades on two nodes mean one node at a time in the trough. And write the provider's own SLA into the analysis: if the GPU provider credits only after a node is down for 24 hours ([Service Level Agreement](getting-compute.md#service-level-agreement)), the 24-hour repair row is the best case, not the typical one.

## 8. The first 30 days

The order matters: each step makes the next one measurable. Effort is engineer-days for a team of two or three, rough.

| Days | Step | What "done" means | Teaches it | Effort |
|---|---|---|---|---|
| 1-2 | Preflight and inventory: record every GPU's serial and UUID, driver, firmware, NIC and NVMe layout; confirm `memory.total` and the MIG profiles | an inventory file per node, in git | [preflight](getting-compute.md#preflight-what-to-collect-before-you-start), [taking inventory](../compute/gpu-software-stack.md#taking-inventory-of-a-node-layer-by-layer), [the three questions](getting-compute.md#the-three-most-important-questions) | 1 |
| 2-3 | Software stack: pinned driver, Fabric Manager, DCGM, a fresh venv with the newest PyTorch | `nvidia-smi`, `dcgmi discovery -l` and NCCL versions recorded | [bring-up runbook](../compute/gpu-software-stack.md#9-bring-up-order-the-runbook), [a current, isolated software stack](getting-compute.md#a-current-isolated-software-stack) | 1-2 |
| 3-5 | Burn-in: DCGM diag level 3-4 on every GPU, the node acceptance script, a sustained matmul on all 16 GPUs for 48 hours | no Xid, no row remaps, no GPU slower than the median by more than a few percent | [burn-in](getting-compute.md#accelerators-need-to-be-burned-in), [acceptance script](../compute/gpu-software-stack.md#10-a-node-acceptance-script), [`dcgmi diag`](../compute/gpu-software-stack.md#running-diagnostics-with-dcgmi-diag) | 1 (48 h of wall time) |
| 4-6 | NCCL tests: intra-node all-reduce on each node, then the two-node pair; small-message latency for the all-reduce floor | `busbw` and latency tables, compared with the provider's numbers | [nccl-tests](../compute/gpu-software-stack.md#nccl-tests-single-node-then-two-then-all), [network](getting-compute.md#a-network), [latency](getting-compute.md#latency-every-collective-no-target) | 1 |
| 5-7 | MAMF and MSMF on every GPU, BF16, FP8 and NVFP4, siblings busy | per-GPU table; replace the book's B300 numbers in `capacity.py` with yours | [compute](getting-compute.md#b-compute), [the MAMF finder](../compute/accelerators.md#the-mamf-finder-mamf-finderpy) | 1 |
| 6-7 | Storage: `fio` on local NVMe and on the provider's object store; encrypted volumes for model and adapter directories | read/write GB/s; `loadmath.py` re-run with your numbers | [storage](getting-compute.md#c-storage), [weights](deploys-recovery-autoscaling.md#weights) | 1 |
| 7 | Write the verdict; escalate anything that missed; keep good nodes on reboot | a signed-off cluster report | [write the verdict](getting-compute.md#write-the-verdict), [keep your good accelerators](getting-compute.md#keep-your-good-accelerators-on-reboot) | 0.5 |
| 8-10 | First engine: one 70B FP8 replica on one GPU and one 8B pool; measure startup stages; persist compile caches; pre-stage weights | per-stage startup times; $T_{\text{start}}$ from NVMe under 3 min | [startup checklist](deploys-recovery-autoscaling.md#startup-time-checklist-for-a-leased-fleet) items 1-6 | 2-3 |
| 9-12 | Measure the real roofline fraction: replay a synthetic trace at several batch sizes; back out achieved HBM fraction; set `EFF` in the planner | measured TPOT and tok/s per replica, not bounds | [a replayable benchmark](../inference/serving-efficiently.md#a-replayable-benchmark), [sweep](../inference/serving-efficiently.md#a-methodical-sweep-one-variable-at-a-time), [achieved.py](../inference/arithmetic.md#what-the-bounds-leave-out) | 3 |
| 10-14 | Gateway: keys with scopes, per-org salt, TPM limits with reserve-and-refund, fair queuing, 429/503 with `Retry-After`, per-request log at completion | the [operations checklist](operating-inference.md#operations-checklist-for-a-leased-fleet) items 1-3, 8, 11 | [Operating Inference](operating-inference.md) sections 1-3 and 9 | 4 |
| 12-16 | Observability: engine and DCGM metrics, the SLO report, burn-rate alerts, the watchdog runbook, canary prompts per replica and per adapter | a dashboard and a pager that would have fired on each injected fault | [metrics](operating-inference.md#metrics), [slo_report.py](operating-inference.md#measure-at-completion-not-at-the-gateway), [incidents](operating-inference.md#8-incidents-and-on-call), [quality probes](operating-inference.md#continuous-evals-and-canary-prompts-in-production) | 3 |
| 15-18 | The deploy pipeline: merge/convert/quantize jobs, `quality_gate.py` on real engine outputs, staging with `.complete`, LoRA canaries, drain, rollback drill | one adapter and one full model shipped and rolled back end to end | [section 4](#4-from-a-training-checkpoint-to-a-served-model), [deploys](deploys-recovery-autoscaling.md#4-deploying-a-new-model-version-on-a-fixed-fleet) | 3-4 |
| 17-20 | Failure drills: kill a replica, drain a node, burst one pool to the overflow provider and back | measured MTTR per path; a tested failover runbook | [recovery](deploys-recovery-autoscaling.md#5-recovery), [section 7](#7-planning-for-the-loss-of-a-node) | 2 |
| 18-22 | Training on the fleet: preemptible LoRA training with save-on-SIGUSR1, the reclaim control loop, backfill queue for evals | a training job preempted and resumed with no lost steps | [section 5](#5-training-and-serving-on-the-same-16-gpus), [graceful stops](../training/fault-tolerance-and-reproducibility.md#graceful-stops-kill-and-save-switches), [backfill](../inference/serving-efficiently.md#priority-filling-idle-capacity-with-batch-work) | 3 |
| 21-25 | First customers: one LoRA customer and one dedicated customer, with evals agreed, the gate run, tiers and retention in the contract | two customers in production with SLO reports | [section 3](#deciding-with-evals), [section 6](#6-customer-isolation) | 3 |
| 25-30 | Tuning from real traffic: trace stats, prefix-aware routing, FP8 KV, batch knobs, re-plan with measured numbers | a re-run `planner.py` with measured rates and the real customer list | [tuning checklist](../inference/serving-efficiently.md#tuning-checklist-for-a-leased-fleet) items 1-6 | 3 |

That is about 35-40 engineer-days in 30 calendar days, which a team of two can do only if nothing goes wrong in burn-in. Things will. The parts that can slip without hurting customers are the training integration (days 18-22) and tuning (25-30); the parts that can't are burn-in, the gateway's isolation controls and the quality gate, because they protect the first customers.

## 9. What to lease next

The [burst rule](deploys-recovery-autoscaling.md#when-serverless-beats-reserved) gives the threshold: a reserved GPU at price $R_r$ beats a rented one at $R_s$ when you'd need it more than $R_r/R_s$ of the time. At the assumed \$4.00 lease against \$7.10 per-second B300s or \$6.64 on demand, that's 56-60% of hours ([failure_plan.py](#what-a-node-failure-costs)). Against dedicated endpoints at \$11.49, 35%. So node 3 is justified when the company would otherwise rent a node's worth of GPUs for more than about half of all hours. Signals that it's getting close:

- **The planner stops fitting the pipeline of signed customers.** Re-run `planner.py` with measured rates and next quarter's contracted traffic. When the plan places 16 of 16 GPUs with no free GPU and the pools above 70% busy at peak, the next customer doesn't fit.
- **Serving is short at peak on normal days.** The SLO report shows latency-budget burn at peak with no incident behind it, and the gateway sheds tier-2 traffic with 503s weekly.
- **Bursting stops being occasional.** Overflow hours are logged per month; when they climb past a third of the hours in a month, the burst rule is about to flip.
- **Training is starved.** Preemptible training gets fewer GPU-hours than the customers' fine-tune queue needs, and fine-tunes wait days to start. In `share_sim.py` terms, `train` falls below the queue's demand.
- **Tier-1 demand outgrows one node.** If dedicated customers' replicas no longer fit on one node, a node failure takes some tier-1 capacity down, and the SLA in [section 7](#what-to-put-in-customer-slas) no longer holds.
- **A customer needs a model that doesn't fit with redundancy**: a 671B MoE as a dedicated deployment, or a 70B at 128k context on TP2 pairs.

When the signals line up, the lease terms decide most of the cost. [Getting Compute](getting-compute.md#contracts) has the full list; the items that matter most at this size:

- **Same spine, same region, same generation.** Node 3 should be on the same network spine as the first two ([same network spine](getting-compute.md#all-your-nodes-on-the-same-network-spine)) and in the same region, so the weights, NVMe staging and the two-node NCCL baseline carry over ([same region](getting-compute.md#keep-all-accelerators-in-the-same-region)). Ask about expansion when signing the first lease ([do you think you will need to expand?](getting-compute.md#do-you-think-you-will-need-to-expand)), not a year later.
- **Coterminous terms.** Align node 3's end date with the first two, so the next decision (renew, move to the next generation, buy) is made once for the whole fleet. A new node with a three-year term on a one-year fleet is a commitment to a generation that may be obsolete halfway through ([next-generation migration](getting-compute.md#next-generation-accelerator-migration)).
- **Replacement terms in hours.** For a 3-node fleet a one-week RMA is a third of capacity for a week. Negotiate replacement time and credits per node ([dealing with accelerator failures](getting-compute.md#dealing-with-accelerator-failures), [SLA](getting-compute.md#service-level-agreement)); it's worth more than a lower hourly price.
- **A break clause** ([contract-breaking clause](getting-compute.md#a-contract-breaking-clause)), and the right to move to the next generation ([when it is worth upgrading](getting-compute.md#when-is-it-worth-upgrading-gpus)).
- **Burn-in before acceptance**, as for the first two ([burn-in](getting-compute.md#accelerators-need-to-be-burned-in)).

There is also the third answer: not a node but a bigger burst arrangement, such as reserved hours with the overflow provider, or a provider's dedicated endpoint for the one large model that doesn't fit. On 16 GPUs, the cheapest capacity is often the capacity you didn't lease.

## You can now read

- Any GPU provider's spec sheet, and turn it into a capacity table for your own model mix: per-GPU budgets, KV room, roofline rates, and which MIG slices exist.
- Vendor posts on multi-LoRA serving ([Fireworks](https://fireworks.ai/blog/multi-lora), [Baseten](https://www.baseten.co/blog/how-to-serve-10-000-fine-tuned-llms-from-a-single-gpu/), [Prime Intellect Lab](https://www.primeintellect.ai/blog/lab)) as statements about per-deployment floors and pricing, not only about kernels.
- Quantization announcements ([NVIDIA NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [FireAttention V4](https://fireworks.ai/blog/fireattention-v4-fp4-b200)) critically: what model, what eval, and whether the result would pass a gate on your customer's eval.
- GPU lease offers and SLAs, with the failure, burst and tier arithmetic to say what an extra node or a shorter replacement time is worth to you.
- Customer contracts for an inference product: what availability a single site supports, what the tiers promise during a failure, and where the customer's weights and logs may go.

## Further reading

- NVIDIA, [MIG user guide: supported profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html); [HGX platform specifications](https://www.nvidia.com/en-us/data-center/hgx/); [DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/); ["Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
- vLLM, [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html) and [prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html); [llm-compressor examples](https://github.com/vllm-project/llm-compressor/tree/main/examples).
- Song et al., ["The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems"](https://arxiv.org/abs/2409.20002), arXiv 2409.20002.
- Chen et al., "Punica: Multi-Tenant LoRA Serving", arXiv 2310.18547; Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters", arXiv 2311.03285.
- Fireworks, ["Multi-LoRA"](https://fireworks.ai/blog/multi-lora) and ["FireAttention V4"](https://fireworks.ai/blog/fireattention-v4-fp4-b200); Prime Intellect, ["Introducing Lab"](https://www.primeintellect.ai/blog/lab); Together, ["What does 99.9% uptime mean for inference?"](https://www.together.ai/blog/99-9-uptime-for-inference); Modal, ["How to price serverless GPUs"](https://modal.com/blog/how-to-price-serverless).
- Price pages read in November 2026: [Modal](https://modal.com/pricing), [Prime Intellect compute](https://www.primeintellect.ai/compute), [Together](https://www.together.ai/pricing).
- Zhipu AI, [GLM-4.5-Air model card](https://huggingface.co/zai-org/GLM-4.5-Air).
- In this repo: [Accelerators](../compute/accelerators.md), [The GPU node software stack](../compute/gpu-software-stack.md), [Inference Internals I](../inference/arithmetic.md) and [II](../inference/serving-engine-internals.md), [Serving Efficiently](../inference/serving-efficiently.md), [Deploys, Recovery and Autoscaling](deploys-recovery-autoscaling.md), [Operating an Inference Service](operating-inference.md), [RL Infrastructure](rl-infrastructure.md) and [Getting Compute](getting-compute.md).
