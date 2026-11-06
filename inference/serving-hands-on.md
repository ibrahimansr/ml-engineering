# Serving LLMs in Practice

## Prerequisites

This chapter is the hands-on companion to two theory chapters. It does not repeat them; it links to them whenever a number or mechanism comes from there.

- [Inference Internals I: The Arithmetic of LLM Inference](arithmetic.md) - rooflines, KV cache sizing, the decode latency bound, Little's law and the worked capacity-planning example.
- [Inference Internals II: Inside a Serving Engine](serving-engine-internals.md) - PagedAttention, continuous batching, chunked prefill, prefix caching, speculative decoding, serving-time parallelism, disaggregation and the engine metrics table.
- [Inference overview](README.md) - vocabulary: prefill, decode, TTFT, TPOT/ITL, percentiles.
- [Concurrency](../foundations/concurrency.md) - especially [asyncio](../foundations/concurrency.md#asyncio) and [producer/consumer and backpressure](../foundations/concurrency.md#producerconsumer-and-backpressure). Every client and server in this chapter is an asyncio program.
- [Networking](../foundations/networking.md) - TCP, [HTTP/1.1](../foundations/networking.md#http11-from-raw-sockets), and why one long-lived connection per stream matters.
- [Distributed Systems](../foundations/distributed-systems.md) - [tail latency](../foundations/distributed-systems.md#stragglers-and-tail-latency), [timeouts](../foundations/distributed-systems.md#heartbeats-timeouts-and-slow-vs-dead) and [consistent hashing](../foundations/distributed-systems.md#consistent-hashing-sharding-that-survives-membership-changes), all three of which the gateway uses.
- Helpful for the deployment sections: [Containers and Virtualization](../foundations/containers-and-virtualization.md) and [Kubernetes from Zero](../orchestration/kubernetes-from-zero.md).

The theory chapters tell you what a serving engine should be able to do. This one is about running one: which flags to pass, how to put load on it, how to read the numbers that come back, how to put a router in front of several copies, and what breaks.

Two kinds of code appear below, and they are labelled:

- **Ran on a Mac.** A mock OpenAI-compatible server, a load generator, a gateway, a metrics watcher and a retry client: about 450 lines of Python in total, all reproduced in full. They ran on an Apple Silicon MacBook with Python 3.14.5, fastapi 0.162.0, uvicorn 0.54.0 and httpx 0.28.1. All the output you see was copied from those runs. The mock has no model in it. It simulates time with a step-cost model shaped like the roofline in [Internals I](arithmetic.md#decode-latency-lower-bound), which is enough to reproduce the qualitative behaviours that matter (queueing, saturation, prefill stalls, cache affinity) using only a CPU.
- **Untested (needs NVIDIA GPUs).** Every `vllm`, `sglang`, Ray, Kubernetes and Docker command. Flag names were checked against the vLLM and SGLang docs as of November 2026 (`docs.vllm.ai/en/latest`, `docs.sglang.io`). Both projects rename flags and change defaults between releases, so treat `--help` on your installed version as the authority.

Setup for the runnable parts:

```bash
python3 -m venv /tmp/serving-lab/venv
/tmp/serving-lab/venv/bin/pip install fastapi uvicorn httpx
cd /tmp/serving-lab   # put the .py files from this chapter here
```



## The shape of a serving deployment

Before any commands, here is the picture the rest of the chapter fills in:

```
  clients (SDKs, apps,            gateway / router                  engine replicas
  your load generator)            (FastAPI, Envoy,                  (vLLM, SGLang ...)
                                   sglang_router, llm-d)
 +-----------+   HTTP + SSE    +----------------------+   HTTP    +--------------------+
 | async     | --------------> |  pick a replica:     | --------> | API server process |
 | client    | <-- tokens ---- |  rr / least / prefix |           |  tokenize, detok   |
 | semaphore |                 |  timeouts, retries,  |           +---------+----------+
 +-----------+                 |  cancellation        |                     | IPC
                               +----------+-----------+           +---------v----------+
                                          |                       | engine core:       |
                                          +---- /metrics -------> | scheduler + GPU    |
                                                Prometheus        | workers (TP/PP/DP) |
                                                                  +--------------------+
```

Terms:

- A **replica** is one complete copy of the model that can serve requests by itself. It may span 1 GPU, 8 GPUs (TP=8) or several nodes (TP x PP).
- The **engine** is the process group that owns the GPUs and runs the scheduler loop from [Internals II](serving-engine-internals.md#request-lifecycle).
- The **API server** is the HTTP front end inside the engine package. It speaks the OpenAI REST protocol, tokenizes, and streams tokens back.
- The **gateway** (also called router or load balancer) is a separate service that sends each request to one of several replicas.
- **SSE** (Server-Sent Events) is the streaming format. The response is a single HTTP response that never closes early, and each event in it is a line `data: <json>` followed by a blank line.

Your load generator measures the whole path end to end. An engine's own metrics only see the part inside the engine. When the two disagree, the difference is time spent in the gateway, the network or the API server.



## Running vLLM

[vLLM](https://github.com/vllm-project/vllm) is the most widely deployed open-source engine. Its internals are described in [Internals II](serving-engine-internals.md). This section covers the operator's view.

### Install and first launch

```bash
# untested here (needs Linux + NVIDIA GPU). vLLM wheels bundle a specific torch/CUDA build:
# install into a fresh environment, never into one that already has a different torch.
uv venv --python 3.12 ~/venvs/vllm && source ~/venvs/vllm/bin/activate
uv pip install vllm --torch-backend=auto

# smallest useful launch: one GPU, OpenAI-compatible server on :8000
vllm serve Qwen/Qwen3-8B --port 8000
```

Sketched expected startup log (from memory of 2025-2026 builds; exact wording varies):

```
INFO ... Resolved architecture: Qwen3ForCausalLM
INFO ... Loading weights took 6.1 seconds
INFO ... Available KV cache memory: 52.3 GiB
INFO ... GPU KV cache size: 380,000 tokens
INFO ... Maximum concurrency for 40,960 tokens per request: 9.3x
INFO ... Capturing CUDA graphs ...
INFO ... Starting vLLM API server on http://0.0.0.0:8000
```

The two lines to read every time are **KV cache size** and **maximum concurrency**. Check them against the KV sizing in [Internals I](arithmetic.md#max-concurrent-sequences) before you send any traffic. If the number of tokens is far below what you computed, either something else is holding GPU memory or `--max-model-len` is forcing a large reservation.

The same server from the official container image (untested here):

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B
```

`--ipc=host` is there because tensor-parallel workers exchange data through `/dev/shm` (shared memory), and Docker's default 64 MB shared-memory size is too small for that. Why this matters is covered in [Containers and Virtualization](../foundations/containers-and-virtualization.md).

### The flags that matter

Each of these flags changes either *what fits in memory* or *how the scheduler fills a step*:

| Flag | What it controls | How to choose |
|-|-|-|
| `--tensor-parallel-size` / `-tp` | Shards every layer across N GPUs, with an all-reduce per layer ([cost](arithmetic.md#tensor-parallel-communication-in-decode)) | The smallest N whose weights plus a useful KV budget fit. Keep it within one NVLink domain (one node). |
| `--pipeline-parallel-size` / `-pp` | Splits layers into N stages | Use when a model does not fit in one node, or when the interconnect between GPUs is PCIe only |
| `--data-parallel-size` / `-dp` | N independent engine replicas behind one API server; MoE layers can shard across them | Mostly for MoE with `--enable-expert-parallel`. For dense models, separate replicas behind a gateway are simpler. |
| `--max-model-len` | Longest prompt + output accepted | Set it to what your product needs, not to the model's maximum. Any request over the limit gets HTTP 400. |
| `--gpu-memory-utilization` | Fraction of GPU memory vLLM may use. Weights, activations and CUDA graphs come out of it first; the remainder becomes KV cache | Default was 0.9 for most releases. The latest docs show 0.92 (version-sensitive). Lower it if another process shares the GPU. |
| `--max-num-seqs` | Upper limit on sequences running in one step (decode batch size) | Raising it increases throughput and ITL together. Lower it to protect ITL, or when you see preemptions. |
| `--max-num-batched-tokens` | Token budget per step, which is effectively the chunked-prefill chunk size | Smaller means smoother ITL and slower TTFT; larger means the opposite. Measured below in [chunked prefill](#chunked-prefill-trades-ttft-for-itl-tail). |
| `--enable-prefix-caching` | Automatic prefix caching ([internals](serving-engine-internals.md#hash-based-block-caching-vllm-apc)) | On by default in V1. Leave it on. |
| `--enable-chunked-prefill` | Splits long prompts across steps | Always on in V1 |
| `--quantization`, `--kv-cache-dtype` | Weight and KV formats ([arithmetic](arithmetic.md#quantization-internals)) | See below |
| `--speculative-config` | Speculative decoding ([internals](serving-engine-internals.md#speculative-decoding)) | See below |
| `--served-model-name`, `--api-key` | The name clients pass as `model`, and a bearer-token check | Always set both in anything shared |

`--max-num-seqs` and `--max-num-batched-tokens` have defaults that depend on the vLLM version and the GPU (older releases used 256 sequences; V1 picks larger values on large GPUs). Read the effective values from the startup log rather than assuming them.

A production-shaped launch for a 70B dense model on one 8x H100 node (untested):

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --served-model-name llama-70b --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --port 8000
```

### Quantized weights

There are two ways to get quantized weights:

1. **Pre-quantized checkpoints** (AWQ, GPTQ, FP8 or NVFP4 in `compressed-tensors` or ModelOpt format). vLLM reads `quantization_config` from the model's `config.json` and picks the kernel automatically. You pass no flag.
2. **Online FP8**: `--quantization fp8` on a bf16 checkpoint quantizes the weights to FP8 at load time with per-tensor scales. This is convenient for experiments. For production, a checkpoint calibrated offline is the safer choice.

```bash
# untested. Pre-quantized: no flag, detected from config.json
vllm serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic -tp 4
# online FP8 weights + FP8 KV cache from a bf16 checkpoint
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8 --kv-cache-dtype fp8
```

Which format pays off on which GPU (FP8 needs Hopper or newer; INT4 weight-only helps at small batch and can hurt at large batch) is worked out in [Internals I](arithmetic.md#when-the-kernel-becomes-dequant-bound). Run your eval on the quantized model before you ship it. Throughput numbers say nothing about quality.

### Speculative decoding

```bash
# untested. EAGLE-3 draft head; JSON keys as documented for --speculative-config
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 3}'

# n-gram / prompt-lookup speculation: no draft model, good for code edits and RAG that copy from the prompt
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'
```

Turn it on only after measuring. Speculation wins at low batch sizes and can lose at high ones ([why](serving-engine-internals.md#when-speculation-hurts-at-high-batch)). To check, run the load generator below at your real concurrency with speculation on and off, read the acceptance-rate counters (`spec_decode_*` in `/metrics`), and keep it only if ITL and throughput both improve at the concurrency you actually serve.

### vLLM's own benchmark client

vLLM ships a load generator that implements the same measurements as the one we build below, and it is the one to use against real engines:

```bash
# untested. Open-loop at 10 req/s, at most 128 in flight, synthetic 1024-in / 128-out
vllm bench serve --model meta-llama/Llama-3.3-70B-Instruct --served-model-name llama-70b \
  --base-url http://localhost:8000 \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 1000 --request-rate 10 --max-concurrency 128 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
```

The reason to build our own anyway is that you will need to change it: to replay your real prompt-length distribution, add shared system prompts, call your gateway with authentication, or record per-request traces. It is about 100 lines.



## Running SGLang

[SGLang](https://github.com/sgl-project/sglang) uses the same OpenAI-compatible API. It differs from vLLM in its scheduler and in RadixAttention ([internals](serving-engine-internals.md#radixattention-sglang)). Most vLLM flags have an SGLang equivalent under a different name:

| Concept | vLLM | SGLang |
|-|-|-|
| launch | `vllm serve MODEL` | `python -m sglang.launch_server --model-path MODEL` |
| default port | 8000 | 30000 |
| tensor / pipeline / data parallel | `-tp`, `-pp`, `-dp` | `--tp-size` (`--tp`), `--pp-size`, `--dp-size` |
| expert parallel | `--enable-expert-parallel` | `--ep-size` |
| memory fraction for weights+KV | `--gpu-memory-utilization` | `--mem-fraction-static` |
| max running sequences | `--max-num-seqs` | `--max-running-requests` |
| context limit | `--max-model-len` | `--context-length` |
| chunked prefill chunk | `--max-num-batched-tokens` | `--chunked-prefill-size` (`-1` disables) |
| prefix cache off | negated `--enable-prefix-caching` | `--disable-radix-cache` |
| KV dtype | `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e5m2` / `fp8_e4m3` |
| Prometheus metrics | on by default | `--enable-metrics` |

The mapping is approximate. For example, `mem-fraction-static` covers weights plus KV pool, and the reserve for activations is accounted for differently than in vLLM.

```bash
# untested
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 \
  --mem-fraction-static 0.85 --max-running-requests 128 --context-length 16384 \
  --chunked-prefill-size 4096 --enable-metrics

# EAGLE-3 speculative decoding
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

The EAGLE-3 draft repository name above comes from SGLang's docs examples as I remember them. Check it before you use it.

SGLang also ships its own router (`sglang_router`), which provides the cache-aware routing we build by hand [below](#a-gateway-in-front-of-several-replicas):

```bash
# untested; policy names as of 2025 releases: random, round_robin, cache_aware, power_of_two
python -m sglang_router.launch_router --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy cache_aware --port 8000
```

How to choose between the two engines: both are fast, and either one's lead on a given model changes from month to month. Pick the one that supports your model and features (structured output, LoRA, your quantization format, your parallelism layout) on the day you deploy, then benchmark *your* traffic on both with the same client. The [overview's framework comparison](README.md#how-to-choose-an-inference-framework) covers the rest.



## The OpenAI-compatible API and streaming

Every engine in this chapter implements a subset of the OpenAI REST API. Three endpoints matter:

- `POST /v1/chat/completions` - takes `messages`. The server applies the model's chat template (the Jinja template from `tokenizer_config.json` that turns messages into one prompt string).
- `POST /v1/completions` - takes a raw `prompt` string and applies no template. Use it for base models and for benchmarks where you want exact control over the prompt.
- `GET /v1/models`, plus the non-OpenAI `GET /health` and `GET /metrics`.

A streaming request with `"stream": true` returns `text/event-stream`. This is the real wire format, captured from the mock server built in the next section (ran on Mac):

```bash
curl -sN localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mock-llm", "messages": [{"role": "user", "content": "hello there how are you"}],
  "max_tokens": 3, "stream": true, "stream_options": {"include_usage": true}}'
```

```
data: {"id": "chatcmpl-1451727143f9", "object": "chat.completion.chunk", "created": 1793913301, "model": "mock-llm", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}

data: {"id": "chatcmpl-1451727143f9", "object": "chat.completion.chunk", "created": 1793913301, "model": "mock-llm", "choices": [{"index": 0, "delta": {"content": "tok1 "}, "finish_reason": null}]}

data: {"id": "chatcmpl-1451727143f9", "object": "chat.completion.chunk", "created": 1793913301, "model": "mock-llm", "choices": [{"index": 0, "delta": {"content": "tok2 "}, "finish_reason": null}]}

data: {"id": "chatcmpl-1451727143f9", "object": "chat.completion.chunk", "created": 1793913301, "model": "mock-llm", "choices": [{"index": 0, "delta": {"content": "tok3 "}, "finish_reason": null}]}

data: {"id": "chatcmpl-1451727143f9", "object": "chat.completion.chunk", "created": 1793913301, "model": "mock-llm", "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}

data: {"id": "chatcmpl-1451727143f9", "object": "chat.completion.chunk", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

data: [DONE]
```

Things in that stream that trip people up:

- The **first chunk carries only the role** and an empty `content`. A client that starts its TTFT clock at the first chunk measures the time the server took to *accept* the request, not the time to the first token. Our load generator only timestamps chunks with non-empty content.
- **One chunk is not always one token.** The detokenizer emits text only once it forms valid UTF-8 and has passed stop-string checks, and some servers coalesce several tokens into one chunk under load. An ITL computed from chunk gaps is therefore a *chunk* interval. Use `usage.completion_tokens` for token counts.
- The **usage chunk has `choices: []`** and arrives only when you ask for it with `stream_options.include_usage`. Code that does `ev["choices"][0]` without checking crashes on it.
- `[DONE]` is a sentinel string, not JSON.
- `finish_reason` is `"stop"` (EOS or stop string), `"length"` (`max_tokens` reached) or `"abort"`/`null` on cancellation, depending on the engine. For benchmarks you usually want a fixed output length. vLLM accepts the non-standard `ignore_eos: true` for this, and `vllm bench serve` sets it.

The official `openai` Python SDK works against any of these servers, because it only needs a `base_url`:

```python
# untested against a real engine here; same protocol as the mock
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
stream = client.chat.completions.create(model="llama-70b", stream=True, max_tokens=64,
                                        messages=[{"role": "user", "content": "Say hi"}])
for ev in stream:
    if ev.choices and ev.choices[0].delta.content:
        print(ev.choices[0].delta.content, end="", flush=True)
```

For load generation we use raw `httpx` instead of the SDK. Raw httpx lets us control connection pooling, gives us per-chunk timestamps, and keeps a layer of client-side overhead out of the measurements.



## A mock server you can run anywhere

You need something to point a load generator at that behaves *qualitatively* like an engine, runs on a laptop, and has knobs you can turn. The mock below includes a miniature version of the [unified scheduler](serving-engine-internals.md#the-unified-scheduler):

- **Admission**: first-come-first-served (FCFS), up to `--max-num-seqs` sequences running; the rest wait.
- **Per-step token budget** `--max-num-batched-tokens`: each running decode contributes 1 token, and prompts are prefilled in chunks out of whatever budget remains. This is vLLM V1's policy of scheduling decodes first and then chunked prefill.
- **A step-cost model** standing in for the GPU:

  ```
  step_time = t0 + c_decode * n_decoding + c_prefill * n_prefill_tokens
  ```

  `t0` (8 ms) plays the role of reading the weights, which is paid once per step regardless of batch size. `c_decode` (0.25 ms per sequence) plays the KV read, and `c_prefill` (0.02 ms per token) plays prefill compute. This is the `TPOT(B) = t_0 + B * c_kv` model from [Internals I](arithmetic.md#the-throughput-latency-pareto), with prefill added.
- **A block-hash prefix cache** with LRU eviction, following the same chained-hash scheme as [vLLM APC](serving-engine-internals.md#hash-based-block-caching-vllm-apc). Cached prompt tokens skip their prefill cost.
- **Streaming in the OpenAI SSE format**, `--max-model-len` validation returning HTTP 400, a Prometheus-text `/metrics`, and **cancellation**: when the client disconnects, the sequence is removed from the batch.

"Tokens" are whitespace-separated words, and the output is `tok1 tok2 ...`.

```python
# mock_server.py - a fake OpenAI-compatible LLM server with a toy continuous-batching engine.
# No model: "tokens" are words, and time is simulated with a step-cost model
#   step_time = T0 + C_DECODE * n_decoding + C_PREFILL * n_prefill_tokens
# which has the same shape as the roofline model in inference/arithmetic.md.
import argparse, asyncio, hashlib, json, time, uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8001)
p.add_argument("--max-num-seqs", type=int, default=32)               # cap on running sequences
p.add_argument("--max-num-batched-tokens", type=int, default=2048)   # per-step token budget (chunked prefill)
p.add_argument("--max-model-len", type=int, default=8192)            # prompt + output limit
p.add_argument("--t0-ms", type=float, default=8.0)       # fixed per-step cost (weights read + launch overhead)
p.add_argument("--c-decode-ms", type=float, default=0.25)  # per decoding sequence (KV read)
p.add_argument("--c-prefill-ms", type=float, default=0.02) # per prompt token prefilled
p.add_argument("--prefix-cache-blocks", type=int, default=4096)  # 0 disables prefix caching
args = p.parse_args()

BLOCK = 16  # tokens per KV block, as in vLLM


class Seq:
    def __init__(self, prompt_tokens, max_tokens):
        self.prompt_len = len(prompt_tokens)
        self.prefill_left = len(prompt_tokens)
        self.max_tokens = max_tokens
        self.generated = 0
        self.q = asyncio.Queue()   # engine -> HTTP handler; None means finished
        self.cancelled = False


class Engine:
    def __init__(self):
        self.waiting, self.running = deque(), []
        self.wake = asyncio.Event()
        self.cache = OrderedDict()  # block-hash -> None, LRU order
        self.m = dict(prompt_tokens=0, generation_tokens=0, prefix_queries=0, prefix_hits=0,
                      requests_finished=0, requests_cancelled=0, steps=0)

    def prefix_lookup(self, toks):
        # Hash chain over full blocks: block i's key covers tokens [0, 16*(i+1)), so a hit
        # means the whole prefix matches. Same idea as vLLM automatic prefix caching.
        h, hit, n_full = hashlib.sha1(), 0, len(toks) // BLOCK
        miss = False
        for i in range(n_full):
            h.update(" ".join(toks[i * BLOCK:(i + 1) * BLOCK]).encode())
            k = h.hexdigest()
            if not miss and k in self.cache:
                hit += BLOCK
                self.cache.move_to_end(k)
            else:
                miss = True
                if args.prefix_cache_blocks:
                    self.cache[k] = None
                    if len(self.cache) > args.prefix_cache_blocks:
                        self.cache.popitem(last=False)
        self.m["prefix_queries"] += len(toks)
        self.m["prefix_hits"] += hit
        return hit

    def submit(self, toks, max_tokens):
        s = Seq(toks, max_tokens)
        if args.prefix_cache_blocks:
            s.prefill_left -= min(self.prefix_lookup(toks), len(toks) - 1)  # always compute >= 1 token
        self.m["prompt_tokens"] += len(toks)
        self.waiting.append(s)
        self.wake.set()
        return s

    async def loop(self):
        while True:
            if not self.running and not self.waiting:
                self.wake.clear()
                await self.wake.wait()
            # admission: FCFS up to max_num_seqs
            while self.waiting and len(self.running) < args.max_num_seqs:
                self.running.append(self.waiting.popleft())
            self.running = [s for s in self.running if not s.cancelled]
            # schedule: decodes first (1 token each), then prefill chunks from the remaining budget
            budget = args.max_num_batched_tokens
            emit, n_dec, n_pre = [], 0, 0
            for s in self.running:
                if s.prefill_left == 0 and budget > 0:
                    emit.append(s); n_dec += 1; budget -= 1
            for s in self.running:
                if s.prefill_left > 0 and budget > 0:
                    n = min(s.prefill_left, budget)
                    s.prefill_left -= n; budget -= n; n_pre += n
                    if s.prefill_left == 0:
                        emit.append(s)  # the last prefill chunk produces the first token
            if not emit and not n_pre:
                await asyncio.sleep(0)
                continue
            await asyncio.sleep((args.t0_ms + args.c_decode_ms * n_dec + args.c_prefill_ms * n_pre) / 1e3)
            self.m["steps"] += 1
            for s in emit:
                if s.cancelled:
                    continue
                s.generated += 1
                self.m["generation_tokens"] += 1
                s.q.put_nowait(f"tok{s.generated} ")
                if s.generated >= s.max_tokens:
                    s.q.put_nowait(None)
                    self.m["requests_finished"] += 1
                    self.running.remove(s)


engine = Engine()


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(engine.loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "mock-llm", "object": "model"}]}


@app.get("/metrics")
async def metrics():
    g = dict(num_requests_running=len(engine.running), num_requests_waiting=len(engine.waiting))
    lines = [f"mock:{k} {v}" for k, v in g.items()]
    lines += [f"mock:{k}_total {v}" for k, v in engine.m.items()]
    return PlainTextResponse("\n".join(lines) + "\n")


def chunk(rid, created, delta, finish=None):
    return "data: " + json.dumps({
        "id": rid, "object": "chat.completion.chunk", "created": created, "model": "mock-llm",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    toks = " ".join(m.get("content", "") for m in body.get("messages", [])).split()
    max_tokens = int(body.get("max_tokens") or 64)
    if len(toks) + max_tokens > args.max_model_len:
        # vLLM also returns 400 here, with a message naming the limit
        return JSONResponse(status_code=400, content={"error": {
            "message": f"prompt ({len(toks)}) + max_tokens ({max_tokens}) > max_model_len ({args.max_model_len})",
            "type": "BadRequestError"}})
    s = engine.submit(toks, max_tokens)
    rid, created = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time())

    if not body.get("stream"):
        out = []
        while (t := await s.q.get()) is not None:
            out.append(t)
        return {"id": rid, "object": "chat.completion", "created": created, "model": "mock-llm",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(out)},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(toks), "completion_tokens": len(out),
                          "total_tokens": len(toks) + len(out)}}

    async def gen():
        try:
            yield chunk(rid, created, {"role": "assistant", "content": ""})
            while (t := await s.q.get()) is not None:
                yield chunk(rid, created, {"content": t})
            yield chunk(rid, created, {}, "length")
            if (body.get("stream_options") or {}).get("include_usage"):
                yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "choices": [],
                                             "usage": {"prompt_tokens": s.prompt_len,
                                                       "completion_tokens": s.generated}}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Runs on normal completion AND when Starlette cancels us because the client went away.
            if s.generated < s.max_tokens:
                s.cancelled = True
                engine.m["requests_cancelled"] += 1
                if s in engine.waiting:
                    engine.waiting.remove(s)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
```

How cancellation works, because it is the part people get wrong: Starlette runs the `gen()` async generator to produce the response body. When the client's TCP connection closes, Starlette notices the `http.disconnect` event and stops iterating the generator, which runs its `finally` block. That block marks the sequence cancelled, and the engine loop drops it at the next step. vLLM does the equivalent: on disconnect, its API server calls `abort` on the engine, and the KV blocks are freed. If your gateway does not propagate the disconnect, which is the bug we avoid [below](#cancellation-end-to-end), the GPU keeps generating tokens nobody will read.

Smoke test (ran on Mac):

```bash
./venv/bin/python mock_server.py --port 8001 &
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mock-llm","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
curl -s localhost:8001/metrics
```

```
{"id":"chatcmpl-0c4c7fbc95f1","object":"chat.completion","created":1793913302,"model":"mock-llm","choices":[{"index":0,"message":{"role":"assistant","content":"tok1 tok2 tok3 tok4 "},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":4,"total_tokens":5}}
mock:num_requests_running 0
mock:num_requests_waiting 0
mock:prompt_tokens_total 6
mock:generation_tokens_total 7
mock:prefix_queries_total 6
mock:prefix_hits_total 0
mock:requests_finished_total 2
mock:requests_cancelled_total 0
mock:steps_total 7
```

(The counters include the 3-token streaming request from the previous section.)



## Writing a load generator

### What to measure

The per-request quantities are defined in [Internals I](arithmetic.md#definitions). Here is how each is computed at the client:

```
t0         = time the request was *issued* (or *arrived*, for open-loop - see below)
stamps[i]  = time chunk i with non-empty content arrived
TTFT       = stamps[0] - t0
ITL        = stamps[i+1] - stamps[i]        (every gap, all requests pooled)
E2E        = stamps[-1] - t0
throughput = total content chunks / wall time,  completed requests / wall time
```

Report **percentiles**, never just means ([why](README.md#percentiles)). Always report the **error count** next to the latencies: a run with 5% timeouts has excellent p99 latency over the 95% of requests that succeeded.

### Closed loop vs open loop

There are two fundamentally different ways to generate load, and confusing them is the single most common benchmarking error:

- **Closed loop** (`--concurrency N`): N simulated users, each sending its next request only after the previous one finished. Offered load adapts to the server. When the server slows down, users send less. This measures *throughput at a fixed concurrency*, and that throughput can never exceed what the server sustains. It hides queueing, because a slow server is simply sent fewer requests. (This is "coordinated omission": the load generator coordinates with the system it is measuring.)
- **Open loop** (`--rate R`): requests arrive at times drawn from a Poisson process with rate R. **Poisson arrivals** means independent arrivals with exponentially distributed gaps, which is a reasonable model for many independent users. Arrivals do not wait for completions, so if R exceeds capacity the queue grows without bound and latency grows with the length of the run. This is what production traffic does, so it is the right way to measure *latency at a given QPS*.

A production load generator needs both. Use closed loop to find the saturation throughput, and open loop to measure latency at the request rates you will actually see.

### The code

```python
# loadgen.py - streaming load generator for any OpenAI-compatible /v1/chat/completions endpoint.
# Closed loop (--concurrency N: N users, each sends the next request when the last finishes)
# or open loop (--rate R: Poisson arrivals at R req/s, capped at --concurrency in flight).
import argparse, asyncio, json, random, time

import httpx

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8001")
p.add_argument("--model", default="mock-llm")
p.add_argument("-n", "--num-requests", type=int, default=200)
p.add_argument("-c", "--concurrency", type=int, default=8)
p.add_argument("--rate", type=float, default=0.0, help="req/s, Poisson; 0 = closed loop")
p.add_argument("--prompt-words", type=int, default=256)
p.add_argument("--max-tokens", type=int, default=128)
p.add_argument("--n-prefixes", type=int, default=0, help="shared system prompts (0 = all unique)")
p.add_argument("--prefix-words", type=int, default=1024)
p.add_argument("--timeout", type=float, default=60.0)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
rng = random.Random(args.seed)
VOCAB = [f"w{i}" for i in range(5000)]
PREFIXES = [" ".join(rng.choices(VOCAB, k=args.prefix_words)) for _ in range(args.n_prefixes)]


def make_messages():
    user = " ".join(rng.choices(VOCAB, k=args.prompt_words))
    msgs = [{"role": "user", "content": user}]
    if PREFIXES:
        msgs.insert(0, {"role": "system", "content": rng.choice(PREFIXES)})
    return msgs


async def one(client, out, t0=None):
    body = {"model": args.model, "messages": make_messages(), "max_tokens": args.max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = t0 or time.perf_counter()  # open loop passes the arrival time, so client-side queueing counts
    stamps = []  # arrival time of each chunk that carried content
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                await r.aread()
                out.append({"err": f"HTTP {r.status_code}"})
                return
            async for line in r.aiter_lines():  # SSE: "data: {...}" lines separated by blank lines
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev["choices"] and ev["choices"][0]["delta"].get("content"):
                    stamps.append(time.perf_counter())
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        out.append({"err": type(e).__name__})
        return
    if not stamps:
        out.append({"err": "no tokens"})
        return
    out.append({"ttft": stamps[0] - t0, "e2e": stamps[-1] - t0, "n": len(stamps),
                "itl": [b - a for a, b in zip(stamps, stamps[1:])]})


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))] if xs else float("nan")


async def main():
    out = []
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=args.timeout) as client:
        t_start = time.perf_counter()
        if args.rate > 0:
            # Open loop: arrivals do not wait for completions. The semaphore is client-side
            # backpressure: beyond `concurrency` in flight, new arrivals queue here (and that
            # wait is counted in their TTFT, because the user is waiting too).
            sem = asyncio.Semaphore(args.concurrency)

            async def gated():
                t_arrive = time.perf_counter()
                async with sem:
                    await one(client, out, t_arrive)
            tasks = []
            for _ in range(args.num_requests):
                tasks.append(asyncio.create_task(gated()))
                await asyncio.sleep(rng.expovariate(args.rate))
            await asyncio.gather(*tasks)
        else:
            left = iter(range(args.num_requests))

            async def user():
                for _ in left:  # shared iterator: users pull the next request id
                    await one(client, out)
            await asyncio.gather(*(user() for _ in range(args.concurrency)))
        wall = time.perf_counter() - t_start

    ok = [r for r in out if "ttft" in r]
    errs = len(out) - len(ok)
    itl = [x for r in ok for x in r["itl"]]
    ntok = sum(r["n"] for r in ok)
    mode = f"rate={args.rate}/s cap={args.concurrency}" if args.rate else f"concurrency={args.concurrency}"
    print(f"{mode}  ok={len(ok)} err={errs}  wall={wall:.1f}s  "
          f"req/s={len(ok)/wall:.2f}  out_tok/s={ntok/wall:.0f}")
    print(f"{'metric':8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}   (ms)")
    for name, xs in [("TTFT", [r["ttft"] for r in ok]), ("ITL", itl), ("E2E", [r["e2e"] for r in ok])]:
        print(f"{name:8} " + " ".join(f"{1e3*pct(xs, q):8.1f}" for q in (50, 90, 99, 100)))


asyncio.run(main())
```

Design choices worth copying:

- **One `httpx.AsyncClient` with `max_connections = concurrency`.** A connection pool smaller than your concurrency silently becomes the real concurrency cap, with the extra requests queueing inside httpx. A pool that is far larger is harmless.
- **In open-loop mode, `t0` is the arrival time, not the send time.** If the client-side semaphore is full, the request waits, and a real user would be waiting too. Starting the clock after the semaphore is coordinated omission again, just on the client side.
- **The arrival RNG is seeded** (`--seed`), and the same RNG generates the prompts. That makes runs reproducible. It also means each seed is one *realization* of the Poisson process, and the seed affects results near saturation, as shown below.
- **Everything happens on one event loop.** At high concurrency a Python client can become the bottleneck, because JSON parsing of thousands of chunks per second competes with timestamping. If client CPU nears 100%, the ITL you measure includes client scheduling delay. Run several client processes, or parse fewer fields.



## Reading the results

All the runs in this section were made on the Mac against the mock (default flags: `max-num-seqs=32`, `max-num-batched-tokens=2048`, step model 8 ms + 0.25 ms/seq + 0.02 ms/prefill-token), with 256-word prompts and 128 output tokens unless stated otherwise. These numbers describe the mock, not any GPU. The *shapes* are the point.

### Concurrency sweep: finding the knee

```bash
./venv/bin/python mock_server.py --port 8001 &
for c in 1 8 32 64; do ./venv/bin/python loadgen.py -c $c -n $((c*8>64?c*8:64)); echo; done
```

```
concurrency=1  ok=64 err=0  wall=78.1s  req/s=0.82  out_tok/s=105
metric        p50      p90      p99      max   (ms)
TTFT         16.6     24.8     38.7     38.7
ITL           9.3     10.1     19.0    124.4
E2E        1205.0   1293.8   1400.6   1400.6

concurrency=8  ok=64 err=0  wall=11.6s  req/s=5.53  out_tok/s=708
metric        p50      p90      p99      max   (ms)
TTFT         18.2     24.4    108.1    108.1
ITL          11.0     12.2     31.4     99.2
E2E        1426.7   1498.6   1522.0   1522.0

concurrency=32  ok=256 err=0  wall=19.1s  req/s=13.43  out_tok/s=1719
metric        p50      p90      p99      max   (ms)
TTFT         50.6    166.1    396.6    406.4
ITL          17.1     22.0     32.1     48.6
E2E        2338.7   2453.9   2714.0   2730.9

concurrency=64  ok=512 err=0  wall=37.6s  req/s=13.62  out_tok/s=1743
metric        p50      p90      p99      max   (ms)
TTFT       2396.7   2497.8   2585.4   2607.4
ITL          17.0     21.0     66.0    359.3
E2E        4697.7   4808.8   4878.8   4924.2
```

How to read it, line by line:

1. **c=1 gives the latency floor.** The model predicts an ITL of `8 + 0.25 = 8.25 ms`, and we measured a p50 of 9.3 ms. The extra ~1 ms is the cost of an asyncio sleep plus the HTTP and JSON path on macOS. On a real engine this gap is the CPU overhead discussed in [Internals II](serving-engine-internals.md#why-cpu-overhead-matters-at-decode), and comparing measured ITL at c=1 to the [roofline bound](arithmetic.md#decode-latency-lower-bound) is the first number you should compute for any deployment.
2. **From c=1 to c=8, throughput rises 6.7x (105 to 708 tok/s) while ITL rises 18% (9.3 to 11.0 ms).** This is the memory-bound regime: the weight read (`t0`) is shared across the batch, so throughput is almost free here.
3. **At c=32 the batch fills `max-num-seqs`.** The model predicts a 16 ms step and a ceiling of `32 / 16 ms = 2000 tok/s`. We measured 17.1 ms and 1719 tok/s. The missing ~14% goes to steps that also carry prefill chunks, plus loop overhead.
4. **At c=64 throughput is flat (1743 tok/s) and TTFT jumps 47x, from a p50 of 50.6 ms to 2397 ms.** Thirty-two requests sit in `waiting` at all times. Each waits for a full running request to finish (about `128 x 17 ms = 2.2 s`), and TTFT p50 is almost exactly that. *A TTFT that equals one E2E of another request is the signature of admission queueing*, and engine metrics show it as `num_requests_waiting > 0` and a large `request_queue_time_seconds`. More concurrency past the knee buys no throughput at all; it only adds latency.
5. **The p99/max columns are noisy.** One spike of 124 ms at c=1 is one event in 8,000 gaps, probably a GC pause or a macOS scheduling hiccup. Before you build a theory on a max value, rerun.

### Open loop: what latency looks like at a given QPS

```bash
for R in 8 10 12 16; do ./venv/bin/python loadgen.py --rate $R -c 256 -n 300; done
```

```
rate=8.0/s cap=256  ok=300 err=0  wall=38.8s  req/s=7.74  out_tok/s=991
TTFT         28.4     35.6     45.7     47.5
ITL          12.7     16.3     21.2     54.0
E2E        1685.4   1941.1   2074.3   2085.1

rate=10.0/s cap=256  ok=300 err=0  wall=31.6s  req/s=9.50  out_tok/s=1216
TTFT         32.1     42.7    321.5    533.5
ITL          14.0     18.9     27.7    103.7
E2E        1880.4   2263.9   2430.9   2581.9

rate=12.0/s cap=256  ok=300 err=0  wall=26.7s  req/s=11.22  out_tok/s=1436
TTFT         34.3    161.2    752.1    920.8
ITL          16.1     20.6     26.3     32.5
E2E        2182.4   2355.8   2997.7   3130.9

rate=16.0/s cap=256  ok=300 err=0  wall=24.0s  req/s=12.50  out_tok/s=1600
TTFT        948.2   3118.2   3533.8   3639.7
ITL          17.2     22.0     27.8     58.4
E2E        3269.8   5285.3   5794.5   5876.7
```

(Header lines trimmed. Columns are p50, p90, p99, max in ms. The 10 req/s point was run separately against a freshly started server; the other three ran back to back against one server.)

- At **16 req/s**, above the ~13.6 req/s saturation we measured in closed loop, the queue grows for the whole run: arrivals exceed completions by about 2.4 req/s. The p90 TTFT of 3.1 s is a function of *how long you ran the test*, not a property of the server. A longer run would show a larger number. If a latency percentile changes with the run length, you are above capacity.
- **ITL barely moves from 10 to 16 req/s** (p99 27.7 to 27.8 ms) while TTFT explodes. That is because `max-num-seqs` caps the decode batch. The engine protects running requests and makes new requests wait. Whether that is the behaviour you want is a product decision, and the `max-num-seqs` knob is what makes it.
- Compare closed loop at c=32 (13.4 req/s, TTFT p99 397 ms) with open loop at 12 req/s (TTFT p99 752 ms). The open-loop run serves *fewer* requests per second and has *worse* tail latency, because Poisson bursts momentarily exceed capacity. A closed-loop benchmark never generates those bursts.

### Seeds, and why one run is an anecdote

The same rates rerun with two more seeds (TTFT p99 and ITL p99 in ms; "realized" is completed requests over wall time):

| rate | seed | realized req/s | TTFT p99 | ITL p99 |
|-:|-:|-:|-:|-:|
| 10 | 0 | 9.50 | 321.5 | 27.7 |
| 10 | 1 | 8.25 | 76.8 | 25.2 |
| 10 | 2 | 9.05 | 41.9 | 20.9 |
| 12 | 0 | 11.22 | 752.1 | 26.3 |
| 12 | 1 | 9.81 | 58.6 | 25.0 |
| 12 | 2 | 10.58 | 83.8 | 25.6 |

At 12 req/s, TTFT p99 ranges from 59 to 752 ms across seeds, a factor of 13. Three hundred Poisson arrivals is a small sample, and the seed-0 realization happened to contain more bursts. (Realized rate also includes the drain time after the last arrival, so it reads low.) Near the knee, tail latency is dominated by the arrival realization. The practical rules:

- Run at least 3 seeds and report the spread. Run long enough to get thousands of requests per point.
- Report the realized arrival rate next to the target rate.
- Set SLO capacity at a rate where *every* seed passes, which is what the capacity section does below.

### Chunked prefill trades TTFT for ITL tail

The same mock, long prompts (4,000 words, which cost `4000 x 0.02 = 80 ms` to prefill in one step), open loop at 4 req/s, prefix cache off. One variable changes: the per-step token budget.

```bash
for B in 100000 512; do
  ./venv/bin/python mock_server.py --port 8001 --max-num-batched-tokens $B --prefix-cache-blocks 0 &
  ./venv/bin/python loadgen.py --rate 4 -c 64 -n 160 --prompt-words 4000 --max-tokens 64
  pkill -f mock_server.py
done
```

```
== max-num-batched-tokens=100000
rate=4.0/s cap=64  ok=160 err=0  wall=35.2s  req/s=4.54  out_tok/s=291
metric        p50      p90      p99      max   (ms)
TTFT        106.2    217.1    295.5    306.3
ITL          10.9     15.2     96.5    249.6
E2E        1175.7   1717.8   2225.4   2262.4
== max-num-batched-tokens=512
rate=4.0/s cap=64  ok=160 err=0  wall=35.5s  req/s=4.50  out_tok/s=288
metric        p50      p90      p99      max   (ms)
TTFT        325.2    700.7   1187.6   1314.0
ITL          19.9     21.3     24.7     49.2
E2E        1407.4   1953.9   2512.1   2639.5
```

- **No chunking** (budget 100k): whenever a 4,000-token prompt is admitted, every running request's next token waits the full 80 ms prefill. Several prompts admitted together cost 160 ms or more, which shows up as ITL p99 96.5 ms and max 250 ms. The median ITL stays a fast 10.9 ms. This is the [prefill stall](serving-engine-internals.md#the-stall).
- **Chunks of 512**: ITL p99 falls 3.9x (96.5 to 24.7 ms) and max 5x. In exchange, the median ITL rises (each step carries `512 x 0.02 = 10.2 ms` of prefill) and TTFT p50 triples, because each prompt now needs about 8 steps to prefill and pays `t0` on every one.
- In the mock, prefill tokens cost the same however you batch them, plus the per-step `t0`. On a real GPU, prefill chunks piggyback on the decode step's weight read, so chunking costs less TTFT than shown here ([Sarathi-Serve](serving-engine-internals.md#sarathi-and-sarathi-serve)). The direction of the trade is the same. **Pick the budget from your SLO**: if your SLO is on ITL p99 (chat UIs, voice), use small chunks; if it is on TTFT (short answers, classification), use large chunks.

One more lesson from this experiment, which I got wrong on the first try: I first ran it in **closed loop** (16 users, 96 requests). All 16 users started at once, so their prefills were batched together and they finished together, and the whole run marched in lockstep waves. During a wave's prefill nobody was decoding, so the stall was invisible. Unchunked looked *better*: ITL p99 22.8 vs 26.3 ms, TTFT p50 1259 vs 1347 ms, 7.47 vs 5.73 req/s. The stall showed up only as a single 1209 ms max-ITL outlier. Closed-loop clients synchronize with each other, which hides interference effects. Use open loop for anything involving prefill/decode interaction.



## Concurrency and client-side backpressure

**Backpressure** is any mechanism that makes a fast producer slow down to match a slower consumer ([general treatment](../foundations/concurrency.md#producerconsumer-and-backpressure)). In LLM serving it exists at four layers, and you want it at the *outermost* one that knows enough to act:

| Layer | Mechanism | Failure if missing |
|-|-|-|
| your client | `asyncio.Semaphore(N)` on in-flight requests, connection pool size | unbounded tasks, memory growth, every request times out together |
| gateway | per-replica max in-flight, 429 when full, queue with deadline | one replica queues 1,000 requests while another idles |
| engine API server | none by default; it accepts everything | waiting queue grows without bound; TTFT goes to minutes |
| engine scheduler | `max-num-seqs`, KV budget, preemption | (this one always exists) |

The engine's waiting queue is unbounded and has no deadlines. A request that has waited 90 s, by which time its client has already given up, still gets prefilled unless the disconnect propagates. So *bounding concurrency and enforcing deadlines is the client's and the gateway's job*.

The semaphore pattern from the load generator, by itself:

```python
sem = asyncio.Semaphore(64)          # at most 64 in flight from this process

async def call(client, body):
    async with sem:                  # the 65th caller waits here, not in the engine's queue
        return await client.post("/v1/chat/completions", json=body, timeout=30)
```

When the server says "slow down" (HTTP 429 Too Many Requests, or 503 Service Unavailable), back off **with jitter**. Without jitter, every client retries at the same instant and the retries arrive as a synchronized wave, a "thundering herd". Capped exponential backoff with full jitter, run against the gateway from the next section with its only upstream dead (ran on Mac):

```python
# retry.py - client-side retry with capped exponential backoff and full jitter.
# Retry only what is safe to retry: connect errors, 429 and 503, before any token was consumed.
import asyncio, random, time, httpx

async def post_with_retry(client, url, body, attempts=5, base=0.2, cap=5.0):
    for i in range(attempts):
        try:
            r = await client.post(url, json=body)
            if r.status_code not in (429, 503):
                return r
            # Retry-After is in seconds when present; otherwise back off exponentially with jitter
            delay = float(r.headers.get("retry-after", 0)) or random.uniform(0, min(cap, base * 2 ** i))
            why = f"HTTP {r.status_code}"
        except httpx.ConnectError:
            delay, why = random.uniform(0, min(cap, base * 2 ** i)), "connect error"
        print(f"t={time.perf_counter()-T0:5.2f}s attempt {i+1}: {why}", end="")
        if i == attempts - 1:
            print()
            break
        print(f", sleeping {delay:.2f}s")
        await asyncio.sleep(delay)
    raise RuntimeError(f"gave up after {attempts} attempts")

async def main():
    body = {"model": "mock-llm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            await post_with_retry(c, "http://127.0.0.1:8000/v1/chat/completions", body)
        except RuntimeError as e:
            print(e)

random.seed(0); T0 = time.perf_counter()
asyncio.run(main())
```

```
t= 0.06s attempt 1: HTTP 503, sleeping 0.17s
t= 0.23s attempt 2: HTTP 503, sleeping 0.30s
t= 0.54s attempt 3: HTTP 503, sleeping 0.34s
t= 0.88s attempt 4: HTTP 503, sleeping 0.41s
t= 1.30s attempt 5: HTTP 503
gave up after 5 attempts
```

Rules for retries in LLM serving specifically:

- **Retry only before the first token is consumed.** After that a retry either duplicates output the user has already seen, or throws away work the GPU already did.
- **Retries multiply load exactly when the system is overloaded.** Three client retries times three gateway retries is up to 9x amplification. Retry at one layer only, and cap the total with a **retry budget** (for example, retries may be at most 10% of requests).
- **Set a total deadline, not just a per-attempt timeout.** A 30 s timeout with 5 attempts is a 150 s worst case.



## A gateway in front of several replicas

Once you have more than one replica, something has to pick one for each request. That sounds like a solved problem (nginx, Envoy, a Kubernetes Service), but LLM requests differ from ordinary web requests in three ways that break the usual defaults:

1. **Requests are long and wildly uneven.** A request lasts seconds to minutes, and costs differ by 100x between short and long prompts or outputs. Round robin assumes requests are similar in cost.
2. **Replicas hold state that makes some requests cheaper on one replica than another.** That state is the prefix cache. Sending a conversation back to the replica that holds its KV skips its prefill.
3. **The response streams.** Buffering proxies break TTFT, idle timeouts kill long generations, and a disconnect has to propagate upstream.

### The gateway code

```python
# gateway.py - a minimal LLM gateway: one OpenAI-compatible endpoint in front of N replicas.
# Routing: rr (round robin), least (fewest in-flight), prefix (rendezvous hash of the prompt
# prefix, with a bounded-load fallback). Connect/first-byte timeouts, retry-before-first-byte,
# passive ejection of dead replicas, and cancellation propagated upstream.
import argparse, asyncio, hashlib, itertools, json, time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8000)
p.add_argument("--upstreams", default="http://127.0.0.1:8001,http://127.0.0.1:8002")
p.add_argument("--policy", choices=["rr", "least", "prefix"], default="least")
p.add_argument("--prefix-chars", type=int, default=2000)  # how much of the prompt identifies a "prefix"
p.add_argument("--load-slack", type=int, default=8)       # prefix policy: max in-flight above the least-loaded
p.add_argument("--retries", type=int, default=2)
p.add_argument("--first-byte-timeout", type=float, default=10.0)
args = p.parse_args()

UP = args.upstreams.split(",")
inflight = {u: 0 for u in UP}
down_until = {u: 0.0 for u in UP}
rr = itertools.cycle(UP)
client = httpx.AsyncClient(timeout=httpx.Timeout(connect=1.0, read=30.0, write=5.0, pool=5.0),
                           limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200))
app = FastAPI()


def healthy(exclude):
    now = time.monotonic()
    ups = [u for u in UP if down_until[u] <= now and u not in exclude]
    return ups or [u for u in UP if u not in exclude]  # all down: try anyway rather than fail closed


def pick(body, exclude):
    ups = healthy(exclude)
    if not ups:
        return None
    if args.policy == "rr":
        for u in rr:
            if u in ups:
                return u
    least = min(ups, key=lambda u: inflight[u])
    if args.policy == "least":
        return least
    # prefix: rendezvous (highest-random-weight) hashing. Same prefix -> same replica, and removing
    # a replica only remaps the keys that lived on it. Bounded load: if the sticky replica is far
    # busier than the least-loaded one, give up cache affinity for this request.
    key = "".join(m.get("content", "") for m in body.get("messages", []))[: args.prefix_chars]
    sticky = max(ups, key=lambda u: hashlib.sha1((u + key).encode()).digest())
    return sticky if inflight[sticky] <= inflight[least] + args.load_slack else least


@app.get("/health")
async def health():
    return {"upstreams": {u: {"inflight": inflight[u], "down": down_until[u] > time.monotonic()} for u in UP}}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    tried = set()
    for attempt in range(args.retries + 1):
        u = pick(body, tried)
        if u is None:
            break
        tried.add(u)
        inflight[u] += 1
        try:
            up_req = client.build_request("POST", u + "/v1/chat/completions", json=body)
            r = await asyncio.wait_for(client.send(up_req, stream=True), args.first_byte_timeout)
        except (httpx.TransportError, asyncio.TimeoutError):
            inflight[u] -= 1
            down_until[u] = time.monotonic() + 5.0  # passive health check: eject for 5 s
            continue
        if r.status_code >= 500 or r.status_code == 429:
            await r.aclose(); inflight[u] -= 1
            continue  # retryable: nothing has reached the client yet
        if r.status_code != 200 or not body.get("stream"):
            data = await r.aread(); await r.aclose(); inflight[u] -= 1
            return JSONResponse(status_code=r.status_code, content=json.loads(data))

        async def relay(r=r, u=u):
            # From the first forwarded byte on, no retries: the client has already seen tokens.
            # If the client disconnects, Starlette cancels this generator; `finally` closes the
            # upstream response, which closes the TCP connection, which makes the engine abort.
            try:
                async for raw in r.aiter_raw():
                    yield raw
            finally:
                await r.aclose()
                inflight[u] -= 1
        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"x-upstream": u, "x-attempts": str(attempt + 1)})
    return JSONResponse(status_code=503, content={"error": {"message": "no upstream available"}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
```

The three policies:

- **`rr`** - round robin. It is stateless and fair by count, and blind to both load and cache.
- **`least`** - the fewest in-flight requests *as seen by this gateway*. With several gateway instances, each sees only its own requests. Real systems either scrape `num_requests_running` and `kv_cache_usage_perc` from the engines, or use "power of two choices": sample two replicas and pick the less loaded, which comes close to global least-load without global state.
- **`prefix`** - rendezvous hashing on the first `prefix-chars` characters of the prompt, so the same system prompt always maps to the same replica. Rendezvous (highest-random-weight) hashing is a form of [consistent hashing](../foundations/distributed-systems.md#consistent-hashing-sharding-that-survives-membership-changes): when a replica is removed, only the keys that lived on it move. A pure hash creates hot spots when one prefix is popular, so the policy is **bounded-load**: if the sticky replica has more than `load-slack` requests above the least-loaded one, the request goes to the least-loaded replica and gives up the cache hit. Production routers (SGLang's `cache_aware`, llm-d's scheduler, NVIDIA Dynamo's KV router) do the same with a real radix tree or block-hash index of what each replica holds, rather than a hash of the first N characters. The theory is in [Internals II](serving-engine-internals.md#cache-aware-routing-across-replicas).

### Routing experiment

Three mock replicas, each with a prefix cache of 2,048 blocks (32k tokens). The workload is 12 distinct 3,000-word system prompts, each followed by a unique 64-word user message, 32 output tokens, 24 concurrent users and 360 requests. Each policy runs on freshly started replicas (ran on Mac):

```bash
for POL in rr least prefix; do
  for port in 8011 8012 8013; do ./venv/bin/python mock_server.py --port $port --prefix-cache-blocks 2048 & done
  ./venv/bin/python gateway.py --port 8000 --policy $POL \
      --upstreams http://127.0.0.1:8011,http://127.0.0.1:8012,http://127.0.0.1:8013 &
  ./venv/bin/python loadgen.py --url http://127.0.0.1:8000 -c 24 -n 360 \
      --n-prefixes 12 --prefix-words 3000 --prompt-words 64 --max-tokens 32
  # then scrape prefix_hits_total / prefix_queries_total from each replica, and kill everything
done
```

```
== policy=rr
concurrency=24  ok=360 err=0  wall=7.7s  req/s=46.69  out_tok/s=1494
TTFT         28.9    155.5    532.2    665.1
ITL          11.0     19.7     52.9    132.3
E2E         474.6    645.6   1166.4   1259.1
== policy=least
concurrency=24  ok=360 err=0  wall=7.4s  req/s=48.73  out_tok/s=1559
TTFT         28.2    151.6    506.9    577.1
ITL          11.1     13.2     51.7     59.2
E2E         435.3    607.2   1066.6   1098.1
== policy=prefix
concurrency=24  ok=360 err=0  wall=6.1s  req/s=58.90  out_tok/s=1885
TTFT         24.1     83.3    356.4    359.1
ITL          11.0     13.0     16.5     53.0
E2E         377.1    444.8    754.1    754.9
```

Prefix-cache hit rate, summed over the three replicas' counters:

| policy | hit rate | req/s | TTFT p90 | ITL p99 |
|-|-:|-:|-:|-:|
| rr | 75.4% | 46.7 | 155.5 ms | 52.9 ms |
| least | 76.8% | 48.7 | 151.6 ms | 51.7 ms |
| prefix | 92.5% | 58.9 | 83.3 ms | 16.5 ms |

Where the hit rates come from (checked with python):

- The best possible hit rate per request is `187 full blocks x 16 / 3064 tokens = 97.7%`. Block 188 mixes prefix and user text, so it never hits.
- With prefix routing, each prefix misses once on one replica (12 cold misses), giving a ceiling of `348/360 x 97.7% = 94.4%`. We measured 92.5%; the gap is the bounded-load fallback sending some requests off their sticky replica.
- With rr, each prefix misses once on *each* replica (36 cold misses), giving a ceiling of 87.9%. It reaches only 75.4% because each replica now needs 12 prefixes x 187 = 2,244 blocks plus the churn of unique user blocks, which is more than its 2,048-block capacity. LRU eviction thrashes. **Spraying requests across replicas divides the effective cache size by the number of replicas.**
- `least` behaves almost like `rr` here because the requests are uniform. Least-load earns its keep when request costs vary, which this workload does not exercise.

The result: prefix-aware routing gives 26% more throughput and a 3.2x better ITL p99 on identical hardware. The ITL gain comes from fewer long prefills interrupting decode steps. This is one seed and one workload shape, and the size of the win scales with the share of prompt tokens that are shareable prefix. With all-unique prompts it would be zero.

### Timeouts, retries and ejection

The gateway's timeout and retry rules, and why each one is there:

| Rule | Value in code | Why |
|-|-|-|
| connect timeout | 1 s | A replica that does not accept a TCP connection within 1 s is down or overloaded. Try another one. |
| first-byte timeout | 10 s | Covers queueing plus prefill. Set it from your TTFT SLO with margin, not from E2E. |
| read timeout (between chunks) | 30 s | Detects a hung stream. It must exceed the worst *gap* between chunks, not the generation length. |
| retry on connect error, 5xx, 429 | up to 2 other replicas | safe, because the client has received nothing yet |
| no retry after first byte | - | Tokens have been sent. Retrying would duplicate them. |
| passive ejection | 5 s after a connect failure | Stops every request from paying the connect timeout to a dead replica. Active `/health` probes are the complement. |

A gateway configured with one dead upstream (`:8009`, nothing listening) and one live replica, under load (ran on Mac):

```bash
./venv/bin/python mock_server.py --port 8001 &
./venv/bin/python gateway.py --port 8000 --policy rr --upstreams http://127.0.0.1:8009,http://127.0.0.1:8001 &
./venv/bin/python loadgen.py --url http://127.0.0.1:8000 -c 8 -n 64 --max-tokens 32
curl -s localhost:8000/health
```

```
concurrency=8  ok=64 err=0  wall=3.2s  req/s=20.01  out_tok/s=640
metric        p50      p90      p99      max   (ms)
TTFT         51.5     77.7     92.1     92.1
ITL          10.9     11.3     30.4     45.4
E2E         392.2    415.0    431.3    431.3
{"upstreams":{"http://127.0.0.1:8009":{"inflight":0,"down":true},"http://127.0.0.1:8001":{"inflight":0,"down":false}}}
```

No request failed. Round robin sent about half of the first requests to the dead port, got a connection refused, retried on `:8001`, and ejected `:8009`. On localhost, connection refused returns in microseconds. A host that is down on a real network instead swallows packets, so every attempt costs the full 1 s connect timeout, which is exactly why ejection matters.

### Cancellation end to end

The client below reads 5 tokens of a 5,000-token generation through the gateway and then closes the connection. It then reads the engine's counters (ran on Mac):

```python
# cancel_demo.py - open a stream through the gateway, read 5 tokens, hang up; then ask the engine what happened.
import asyncio, httpx

async def main():
    body = {"model": "mock-llm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5000, "stream": True}
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json=body) as r:
            print("HTTP", r.status_code, "routed to", r.headers.get("x-upstream"), "after", r.headers.get("x-attempts"), "attempt(s)")
            n = 0
            async for line in r.aiter_lines():
                n += line.startswith("data: ")
                if n == 6:  # role chunk + 5 tokens
                    break
        # leaving the `async with` closes the connection = client cancellation
        await asyncio.sleep(0.5)
        print((await c.get("http://127.0.0.1:8001/metrics")).text)

asyncio.run(main())
```

```
HTTP 200 routed to http://127.0.0.1:8001 after 2 attempt(s)
mock:num_requests_running 0
mock:num_requests_waiting 0
mock:prompt_tokens_total 1
mock:generation_tokens_total 5
mock:prefix_queries_total 1
mock:prefix_hits_total 0
mock:requests_finished_total 0
mock:requests_cancelled_total 1
mock:steps_total 6
```

The engine generated 5 tokens, not 5,000. The disconnect travelled client → gateway (Starlette stops `relay()`) → `r.aclose()` closes the upstream connection → the mock's `gen()` `finally` block runs → the sequence leaves the batch. (The `2 attempt(s)` header is the round-robin retry past the dead `:8009` again.) Break any link in that chain and an abandoned request keeps its batch slot and KV blocks until `max_tokens`. Common ways that happens: a proxy that buffers the whole response, a gateway that reads upstream with a non-streaming client, or `asyncio.shield` around the upstream call.

The first version of this demo asked for `max_tokens: 10000` and got `HTTP 400`, because the mock enforces `--max-model-len 8192` exactly as vLLM does. If a client sees 400s only on some requests, check prompt length plus `max_tokens` against the server's limit first.

### Production gateways

What this ~100-line gateway leaves out, and where to find it:

- **Streaming-aware proxies**: Envoy and nginx work if you disable response buffering (`proxy_buffering off;` in nginx) and raise idle and stream timeouts. Otherwise TTFT becomes E2E.
- **LLM-aware routers**: `sglang_router` (cache-aware, also does PD routing), the vLLM production-stack router, llm-d (a Kubernetes-native scheduler with KV-cache- and load-aware scoring), NVIDIA Dynamo, and the Kubernetes Gateway API Inference Extension (endpoint picking from model-server metrics). As of 2025-2026 these are all moving quickly; compare them on your traffic.
- **Auth, per-tenant rate limits, request logging, priority classes.** These belong in the gateway, not the engine. How such services are built is covered in [Services and Control Planes](../platform/control-planes.md).



## Multi-node serving

A model that does not fit in one node's memory has to span several nodes. The [parallelism theory](serving-engine-internals.md#serving-time-parallelism) gives the standard rule: **TP inside a node (NVLink), PP or EP across nodes (InfiniBand/RoCE)**, because TP does an all-reduce on every layer and PP sends one activation per stage boundary. For example, Llama-3.1-405B in bf16 is about 810 GB of weights, which is more than 8x H100 (640 GB) can hold. The two common layouts are TP=8 x PP=2 on two H100 nodes, or FP8 on one node (405 GB).

### vLLM on Ray

vLLM uses Ray to place workers on other nodes. **Ray** is a distributed runtime: you start one head process and N worker processes, and Ray schedules Python actors onto them. Every node must have the *same* vLLM, torch and NCCL versions and the same model path, which in practice means the same container image.

```bash
# untested. On every node: same image, model pre-downloaded to the same path.
# Point NCCL and Gloo at the fast NIC, or they may pick a management interface
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0   # interface names are site-specific
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# node 0 (head)
ray start --head --port=6379
# node 1..N-1
ray start --address=10.0.0.1:6379
# check: should list 16 GPUs
ray status

# on the head node only
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend ray --max-model-len 16384
```

vLLM's docs include a `run_cluster.sh` helper that starts the Ray head and workers inside the vLLM container. Diagnosing a multi-node NCCL setup that hangs at startup is a networking problem, not a vLLM problem. Run `nccl-tests` across the two nodes first ([network benchmarks](../network/benchmarks/README.md)), and set `NCCL_DEBUG=INFO` to see which interface and transport NCCL picked.

### Multi-node data parallel (MoE)

For large MoE models (DeepSeek-V3 class), the layout that serves the most tokens is **DP attention + EP experts** ([why](serving-engine-internals.md#dp-attention-and-expert-parallelism-for-moe)). vLLM launches it with one command per node instead of Ray. The commands below follow the vLLM data-parallel deployment docs, with 2 nodes x 8 GPUs, DP=16 and EP across all 16 (untested):

```bash
# node 0 (10.0.0.1): API server + DP ranks 0-7
vllm serve deepseek-ai/DeepSeek-V3 --enable-expert-parallel \
  --data-parallel-size 16 --data-parallel-size-local 8 \
  --data-parallel-address 10.0.0.1 --data-parallel-rpc-port 13345
# node 1: headless (engines only), ranks 8-15
vllm serve deepseek-ai/DeepSeek-V3 --enable-expert-parallel --headless \
  --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 \
  --data-parallel-address 10.0.0.1 --data-parallel-rpc-port 13345
```

If the single API server becomes the bottleneck at large DP, `--api-server-count N` scales it out on the head node. SGLang's equivalent spans nodes with `--nnodes/--node-rank/--dist-init-addr`:

```bash
# untested. Run on both nodes, changing only --node-rank
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 \
  --tp 16 --nnodes 2 --node-rank 0 --dist-init-addr 10.0.0.1:50000 \
  --ep-size 16 --trust-remote-code
```

The large-scale EP variants (DeepEP all-to-all, two-batch overlap, EPLB) have many version-specific flags. Follow the engine's DeepSeek deployment guide for the exact release you run.

### Disaggregated prefill and decode

When disaggregation pays off is worked out in [Internals II](serving-engine-internals.md#disaggregated-prefill-and-decode). In short: long prompts, strict ITL, enough scale to keep two pools busy, and RDMA between the pools. As of November 2026, both engines ship it behind a KV-transfer layer and a router that sends each request first to a prefill instance and then to a decode instance.

vLLM with the NIXL connector, following the `NixlConnector` usage docs (untested):

```bash
# prefill instance (producer)
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-8B --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
# decode instance (consumer)
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-8B --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
# toy router from the vLLM repo (tests/v1/kv_connector/nixl_integration/toy_proxy_server.py)
python toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 --decoder-hosts localhost --decoder-ports 8200
```

SGLang with the Mooncake or NIXL transfer backend (untested):

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-ib-device mlx5_0
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30001 --base-gpu-id 1 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-ib-device mlx5_0
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --port 8000
```

When prefill and decode run on different hosts, set `VLLM_NIXL_SIDE_CHANNEL_HOST` (vLLM) and make sure `UCX_NET_DEVICES` or `--disaggregation-ib-device` names the RDMA NICs. Over plain TCP the transfer can cost more than the prefill it saves ([arithmetic](serving-engine-internals.md#kv-transfer-cost)). The vLLM toy proxy is explicitly a test harness. At production scale people put llm-d or Dynamo in front, and both orchestrate the P/D pools on Kubernetes.

Benchmark disaggregation against a well-tuned colocated baseline with chunked prefill, using the open-loop generator at your SLO. Measure **goodput** (requests per second meeting *both* the TTFT and ITL targets), not raw throughput.



## Deploying on Kubernetes

The Kubernetes concepts themselves (Pods, Deployments, Services, probes, GPU device plugins, node selectors) are covered in [Kubernetes from Zero](../orchestration/kubernetes-from-zero.md). The LLM-specific parts of a manifest are what this section covers. Untested sketch:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: {name: llama-8b}
spec:
  replicas: 2
  selector: {matchLabels: {app: llama-8b}}
  template:
    metadata: {labels: {app: llama-8b}}
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest          # pin a version tag in production
        args: ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--max-model-len", "16384"]
        ports: [{containerPort: 8000}]
        resources: {limits: {nvidia.com/gpu: 1}}
        env: [{name: HF_TOKEN, valueFrom: {secretKeyRef: {name: hf, key: token}}}]
        volumeMounts:
        - {name: shm, mountPath: /dev/shm}       # TP workers need more than the 64 MB default
        - {name: hf-cache, mountPath: /root/.cache/huggingface}
        startupProbe:                            # weights + CUDA graph capture can take minutes
          httpGet: {path: /health, port: 8000}
          periodSeconds: 10
          failureThreshold: 90                   # 15 minutes before the kubelet gives up
        readinessProbe:                          # gates traffic from the Service
          httpGet: {path: /health, port: 8000}
          periodSeconds: 5
        livenessProbe:
          httpGet: {path: /health, port: 8000}
          periodSeconds: 10
          failureThreshold: 6
      terminationGracePeriodSeconds: 120          # let in-flight generations finish on scale-down
      volumes:
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: 16Gi}}
      - {name: hf-cache, persistentVolumeClaim: {claimName: hf-cache}}
```

The parts that are specific to LLM serving:

- **Startup probe with a long budget.** Without one, the liveness probe kills the pod while it is still loading 140 GB of weights, and it never comes up. This crash loop is common.
- **Model weights on a shared volume or pre-pulled to local NVMe.** Downloading from the Hub on every pod start is slow, and it gets rate-limited when you scale 20 pods at once ([storage](../foundations/storage-and-data.md)).
- **Grace period ≥ your longest generation.** Rolling updates otherwise cut streams mid-response.
- **A plain `Service` load-balances per TCP connection, not per request.** With HTTP keep-alive, one busy client can stick to one pod. Put an LLM-aware router or the Gateway API Inference Extension in front.
- **Autoscale on engine metrics** (`num_requests_waiting`, KV usage), not on CPU or GPU utilization. GPU utilization reads near 100% at any nonzero load.



## Monitoring engine metrics

What each engine metric means and when to act on it is in the [Internals II observability table](serving-engine-internals.md#observability). The practical part is collecting the metrics and turning counters into rates.

A minimal watcher that scrapes Prometheus text and differentiates the counters (ran on Mac):

```python
# watch_metrics.py - poll a Prometheus-text /metrics endpoint and print gauges and counter rates.
import sys, time, httpx

url, every, n = sys.argv[1], 2.0, int(sys.argv[2])

def scrape():
    out = {}
    for line in httpx.get(url).text.splitlines():
        if line and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            out[k] = float(v)
    return out

prev, tp = scrape(), time.time()
print(f"{'t':>4} {'running':>8} {'waiting':>8} {'gen_tok/s':>10} {'prefix_hit%':>12}")
for i in range(n):
    time.sleep(every)
    cur, tc = scrape(), time.time()
    d = lambda k: cur[k] - prev[k]  # counters only go up; rates come from differences
    hit = 100 * d("mock:prefix_hits_total") / max(1, d("mock:prefix_queries_total"))
    print(f"{(i+1)*every:4.0f} {cur['mock:num_requests_running']:8.0f} {cur['mock:num_requests_waiting']:8.0f} "
          f"{d('mock:generation_tokens_total')/(tc-tp):10.0f} {hit:12.1f}")
    prev, tp = cur, tc
```

Run against one mock replica under closed-loop load at concurrency 64, with 4 shared 512-word system prompts:

```bash
./venv/bin/python loadgen.py -c 64 -n 320 --n-prefixes 4 --prefix-words 512 > lg.out &
./venv/bin/python watch_metrics.py http://127.0.0.1:8001/metrics 10
```

```
   t  running  waiting  gen_tok/s  prefix_hit%
   2       32       32       1600         62.5
   4       32       32       1734         66.7
   6       32       32       1733         66.7
   8       32       32       1717         66.7
  10       32       32       1737         66.7
  12       32       32       1721         66.7
  14       32       32       1879          0.0
  16       32       32       1735         66.7
  18       32       32       1723         66.7
  20       32       32       1735         66.7
```

```
concurrency=64  ok=320 err=0  wall=23.6s  req/s=13.58  out_tok/s=1738
metric        p50      p90      p99      max   (ms)
TTFT       2389.3   2448.2   2639.8   2660.7
ITL          17.1     17.7     50.5     59.4
E2E        4677.3   4742.2   4914.9   4969.3
```

Reading it:

- `running == max-num-seqs` and `waiting > 0` on every sample means the replica is saturated and admission-bound. The client's 2.4 s TTFT is almost all queue time.
- The engine-side token rate (~1730 tok/s) matches the client-side 1738 tok/s. If these disagree, tokens are being lost or delayed between the engine and the client (a buffering proxy, or client CPU).
- The 66.7% hit rate is exactly `512 / (512 + 256)`, the system-prompt share of each prompt. That is a ceiling you can compute *before* deploying from your traffic's prompt structure.
- The `0.0` row at t=14 is an artifact of rates computed over short windows: closed-loop users finish in waves, and no new request was admitted in that 2 s window, so there were no prefix queries at all. The code prints 0 when the denominator is 0. Prometheus has the same issue: `rate()` over a window shorter than the event spacing is noise.

For a real deployment, Prometheus scrapes each replica and you query **histograms** for percentiles (untested; metric names as in vLLM ~0.10-0.11, which [have been renamed before](serving-engine-internals.md#observability)):

```yaml
# prometheus.yml
scrape_configs:
- job_name: vllm
  scrape_interval: 5s
  static_configs: [{targets: ["10.0.0.11:8000", "10.0.0.12:8000"]}]
```

```promql
# p99 TTFT across replicas, 5-minute window
histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
# queue depth per replica: alert if > 0 for 10 min
vllm:num_requests_waiting
# output tokens/s per replica
rate(vllm:generation_tokens_total[1m])
# prefix cache hit rate
rate(vllm:prefix_cache_hits_total[5m]) / rate(vllm:prefix_cache_queries_total[5m])
```

Engine histograms have fixed buckets. If your SLO is 300 ms and the nearest bucket boundaries are 250 ms and 500 ms, `histogram_quantile` interpolates between them, and the p99 it reports can be off by up to that gap. For SLO accounting, also log the client-side per-request numbers.



## Capacity planning from measured numbers

[Internals I](arithmetic.md#worked-sizing-example) sizes a deployment from roofline bounds. Here the same procedure runs from *measurements* instead, using the mock data above as a stand-in for a real replica:

1. **Fix the workload**: prompt and output length distributions, prefix structure. (Here: 256-word prompts, 128 output tokens, no shared prefix.)
2. **Fix the SLO as percentiles**: TTFT p99 ≤ 500 ms, ITL p99 ≤ 30 ms.
3. **Run an open-loop sweep on one replica with several seeds.** Find the highest rate where *every* seed meets the SLO. From the tables above, 10 req/s passes on all 3 seeds (worst TTFT p99 321.5 ms, worst ITL p99 27.7 ms), while 12 req/s fails on seed 0 (TTFT p99 752 ms). The per-replica capacity at this SLO is therefore **10 req/s**.
4. **Sanity-check with Little's law and the saturation point**, then size:

```python
import math
peak_qps = 150          # target peak, req/s
per_replica = 10        # highest measured rate meeting SLO on all 3 seeds
e2e_mean = 1.88         # s, p50 E2E at 10 req/s (seed 0) as proxy for mean
n_out = 128
L = per_replica * e2e_mean
print(f"Little's law in-flight per replica: {L:.1f} (max-num-seqs=32 -> {L/32:.0%} of slots)")
base = math.ceil(peak_qps / per_replica)
print(f"replicas for peak: {base}, +1 for failure/rolling deploy: {base+1}")
print(f"output tok/s per replica at SLO: {per_replica*n_out}  (measured 1216 at realized 9.5 req/s)")
saturated = 1740
print(f"fraction of saturated throughput used at SLO: {per_replica*n_out/saturated:.0%}")
gpu_hr, gpus_per_replica = 2.0, 1
cost = (base+1)*gpus_per_replica*gpu_hr
print(f"cost/hour {cost:.0f} $, per 1M output tokens at peak: {cost/(peak_qps*n_out*3600/1e6):.3f} $")
print(f"overload @16 req/s: excess {16-13.6:.1f} req/s -> queue grows ~{(16-13.6)*20:.0f} requests over a 20 s run")
```

```
Little's law in-flight per replica: 18.8 (max-num-seqs=32 -> 59% of slots)
replicas for peak: 15, +1 for failure/rolling deploy: 16
output tok/s per replica at SLO: 1280  (measured 1216 at realized 9.5 req/s)
fraction of saturated throughput used at SLO: 74%
cost/hour 32 $, per 1M output tokens at peak: 0.463 $
overload @16 req/s: excess 2.4 req/s -> queue grows ~48 requests over a 20 s run
```

What the numbers say:

- **The SLO costs about 26% of peak throughput.** You run at 74% of saturation so that Poisson bursts drain before they become p99 violations. This "headroom" is not waste; it is the price of the tail SLO, and a tighter p99 costs more of it.
- **Little's law says 18.8 requests in flight per replica**, comfortably below `max-num-seqs=32`. If it came out above 32, the replica would be admission-bound at your target rate and the sweep would have shown it.
- **N+1** covers one replica failing or being rolled during a deploy. At small replica counts, N+1 is a large fraction of the cost. Use N+2 if a single failure would push the remaining replicas past the knee.
- **Re-measure when anything changes**: model, engine version, quantization, `max-num-seqs`, the prompt length distribution or the prefix share. The prefix share changed throughput by 26% in the routing experiment.
- **Cross-check the measured per-replica number against the roofline** from [Internals I](arithmetic.md#decode-latency-lower-bound). Measured decode throughput far below the bound (for example under 50% of HBM bandwidth at batch 32) means there is engine or CPU overhead to fix before you buy more GPUs.



## Common failure modes

| Symptom | Likely cause | Check / fix |
|-|-|-|
| OOM or "no available memory for cache blocks" at startup | another process on the GPU, `gpu-memory-utilization` too high, or `max-model-len` too large for the KV that's left | `nvidia-smi`; lower `--max-model-len` or `--gpu-memory-utilization`; add TP |
| startup error: max model len greater than KV capacity | the KV cache cannot hold even one max-length sequence | lower `--max-model-len`, FP8 KV, or more GPUs ([sizing](arithmetic.md#max-concurrent-sequences)) |
| pod crash-loops while loading | liveness probe fires before weights finish loading | startup probe with a long budget (see Kubernetes section) |
| TP=2+ hangs at startup or dies with a bus error in Docker | `/dev/shm` too small, or NCCL picked the wrong interface or P2P path | `--ipc=host` / memory-backed `emptyDir`; `NCCL_DEBUG=INFO`; `NCCL_P2P_DISABLE=1` to test |
| multi-node launch hangs | firewall, wrong `NCCL_SOCKET_IFNAME`, mismatched versions across nodes | `nccl-tests` across nodes first; same image everywhere |
| TTFT p50 ≈ one other request's E2E | admission queueing (`waiting > 0`) | more replicas, or a higher `max-num-seqs` if KV and ITL allow |
| ITL p99 ≫ p50 | prefill stalls, preemption, Python GC | smaller `max-num-batched-tokens`; watch `num_preemptions_total`; P/D disaggregation |
| throughput collapses at high load, preemptions rising | KV oversubscribed; sequences are evicted and recomputed | lower `max-num-seqs`, FP8 KV, shorter `max-model-len` |
| TTFT measured at the client ≈ E2E | a proxy buffers the stream | `proxy_buffering off`; check with `curl -N` through every hop |
| HTTP 400 on some requests | prompt + `max_tokens` > `max-model-len` (we hit this above) | clamp `max_tokens` in the gateway; return a clear error |
| garbage or never-ending output from a chat model | wrong or missing chat template, or `/v1/completions` used with a chat model | use `/v1/chat/completions`; check the template via `--chat-template` |
| GPU busy, nobody listening | client cancellations not propagated | `requests_cancelled` should track client disconnects; see [cancellation](#cancellation-end-to-end) |
| latency spike after a deploy | cold prefix cache; CUDA graph capture / torch.compile on the first requests | warm up with representative traffic before marking the pod ready |
| retries make an overload worse | retry amplification at several layers | retry at one layer, with jitter and a retry budget |
| benchmark looks great, production does not | closed-loop benchmark, unrealistic prompt lengths, no shared prefixes, one seed | open loop, replay real length distributions, 3+ seeds |

One check catches a large share of these before users do: **after any change, run the same load-generator command with the same seed and diff the summary table against the last known-good run.** Keep that command and its output in the repo next to the deployment config.



## You can now read

- [vLLM's documentation](https://docs.vllm.ai) on engine arguments, data-parallel deployment, the NIXL connector and the metrics list, and know which knob moves which number.
- [SGLang's server arguments](https://docs.sglang.io/advanced_features/server_arguments.html) and PD-disaggregation docs.
- The [llm-d](https://github.com/llm-d/llm-d) and [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) architecture docs, as answers to the routing, cancellation and disaggregation problems built by hand here.
- Benchmark write-ups from engine vendors, critically: you can now ask whether a benchmark is closed or open loop, which percentile it reports, how many seeds it used and what prefix share its workload had.
- [RL Infrastructure](../platform/rl-infrastructure.md), where rollout generation is exactly this serving stack under a very different load pattern (huge bursts, all requests sharing a prompt, weights updated every step).



## Further reading

- vLLM docs: engine arguments https://docs.vllm.ai/en/latest/configuration/engine_args.html; data-parallel deployment https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html; NIXL connector https://docs.vllm.ai/en/latest/features/nixl_connector_usage.html (all checked November 2026, "latest" developer docs).
- SGLang docs: server arguments https://docs.sglang.io/advanced_features/server_arguments.html; PD disaggregation https://docs.sglang.io/advanced_features/pd_disaggregation.html.
- Zhong et al., "DistServe", arXiv 2401.09670 - goodput as the metric, and the P/D split.
- Agrawal et al., "Sarathi-Serve", arXiv 2403.02310 - the chunked-prefill trade measured above, on real GPUs.
- Zheng et al., "SGLang", arXiv 2312.07104 - RadixAttention, the basis of cache-aware routing.
- Dean & Barroso, "The Tail at Scale", CACM 56(2), 2013 - hedged requests, tail tolerance, why p99 dominates at fan-out.
- Gil Tene, "How NOT to Measure Latency" (talk, 2015) - coordinated omission, the closed-loop trap.
- Thaler & Ravishankar, "Using Name-Based Mappings to Increase Hit Rates", IEEE/ACM ToN 1998 - rendezvous (HRW) hashing.
- Mitzenmacher, "The Power of Two Choices in Randomized Load Balancing", IEEE TPDS 2001.
- Marc Brooker, "Exponential Backoff And Jitter", AWS Architecture Blog, 2015.
- Kubernetes Gateway API Inference Extension: https://gateway-api-inference-extension.sigs.k8s.io/
- In this repo: [Inference overview](README.md), [Inference Internals I](arithmetic.md), [Inference Internals II](serving-engine-internals.md), [Kubernetes from Zero](../orchestration/kubernetes-from-zero.md), [Containers and Virtualization](../foundations/containers-and-virtualization.md), [RL Infrastructure](../platform/rl-infrastructure.md).
