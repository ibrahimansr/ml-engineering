# Operating an Inference Service

Picture a small company that sells LLM inference over an HTTP API. It leases a fixed fleet of GPUs, say a few dozen 8-GPU H100 nodes on a one-year contract, and runs open-weight models on them with vLLM or SGLang. Customers send it requests with API keys. It bills them per token, and some large customers rent whole replicas by the hour. Earlier chapters built the engine ([Inference Internals II](../inference/serving-engine-internals.md)), tuned it ([Serving Efficiently](../inference/serving-efficiently.md)), put a gateway in front of it ([Serving LLMs in Practice](../inference/serving-hands-on.md)), and covered how replicas are started, rolled out, recovered and scaled ([Deploys, Recovery and Autoscaling](deploys-recovery-autoscaling.md)). This chapter is about the layer between those engines and the customers, the part that makes a fleet of engines a business:

- what the API promises a client, including what it says when it is overloaded;
- who is allowed to use how much (rate limits and quotas);
- whose request runs next when there is not enough GPU for everyone (fairness and priority);
- what "working" means, as a number (SLOs, goodput, error budgets);
- what to do with the GPUs at 4 a.m. when nobody is chatting (batch work);
- what a million tokens costs you and what to charge for it;
- how you find out that the service is returning HTTP 200 with worse text;
- what pages the on-call engineer, and what they do next;
- the security basics that are specific to an inference API.

One fact runs through the whole chapter: **on a leased fleet, the GPU-hours are already paid for.** A serverless provider that rents GPUs by the second can hand an idle GPU back and stop paying. You can't. So your cost per token is set mostly by one number, the fraction of leased GPU-time doing paid work (**utilization**), and any work you can put on an otherwise idle GPU earns money for no extra GPU cost. Many techniques below come from vendors that run differently: per-second serverless (Modal), multi-cloud capacity markets (Baseten, Fireworks), or very large first-party fleets (OpenAI, Anthropic). Each time one comes up, a short **On a leased fleet** note says whether it carries over and what the equivalent is.

Conventions:

- Every script starts with a comment naming its file. All of them are pure Python with only the standard library, run with CPython 3.14.5 on an Apple Silicon Mac, from one scratch directory (`/tmp/opinf`). Some import earlier ones (`tpm_limiter.py` imports `token_bucket.py`), so keep them together.
- Output under a script is real output from running it. No block in this chapter needs a GPU or a real engine.
- Simulations are seeded, so you get the same numbers. Several end with `assert`s that fail if you break the property the section is about.
- Prices are quoted with the date they were read (November 2026) and a link. They change often; check them before you reuse them.

## Prerequisites

This chapter links to the following instead of repeating them:

- [Inference Internals I](../inference/arithmetic.md): [prefill and decode](../inference/arithmetic.md#prefill-and-decode), [TTFT, TPOT, ITL, E2E and goodput](../inference/arithmetic.md#definitions), [percentiles](../inference/arithmetic.md#percentiles), [Little's law](../inference/arithmetic.md#littles-law), [the decode-latency bound and the \$/1M-token conversion](../inference/arithmetic.md#decode-latency-lower-bound), [KV cache sizing](../inference/arithmetic.md#kv-cache-sizing) and [worked capacity sizing](../inference/arithmetic.md#worked-sizing-example).
- [Inference Internals II](../inference/serving-engine-internals.md): [scheduler policies](../inference/serving-engine-internals.md#policies), [admission by KV budget](../inference/serving-engine-internals.md#admission-control-by-kv-budget), [preemption](../inference/serving-engine-internals.md#preemption-swap-vs-recompute), [prefix caching](../inference/serving-engine-internals.md#hash-based-block-caching-vllm-apc), [cache-aware routing](../inference/serving-engine-internals.md#cache-aware-routing-across-replicas), [batch invariance](../inference/serving-engine-internals.md#determinism-and-batch-invariance), [multi-LoRA serving](../inference/serving-engine-internals.md#multi-lora-serving) and [the engine metrics table](../inference/serving-engine-internals.md#observability).
- [Serving LLMs in Practice](../inference/serving-hands-on.md): [the OpenAI-compatible API and SSE streaming](../inference/serving-hands-on.md#the-openai-compatible-api-and-streaming), [client-side backpressure and retries](../inference/serving-hands-on.md#concurrency-and-client-side-backpressure), [the gateway and its timeouts](../inference/serving-hands-on.md#timeouts-retries-and-ejection), [cancellation](../inference/serving-hands-on.md#cancellation-end-to-end), [engine monitoring](../inference/serving-hands-on.md#monitoring-engine-metrics), [capacity planning](../inference/serving-hands-on.md#capacity-planning-from-measured-numbers) and [the failure-mode table](../inference/serving-hands-on.md#common-failure-modes).
- [Serving Efficiently](../inference/serving-efficiently.md): engine tuning, routing, speculative decoding, long context and multimodal serving. Throughput numbers in the pricing section are what that chapter's work buys you.
- [Control Planes](control-planes.md): [idempotency keys](control-planes.md#idempotency-keys), [token buckets and per-tenant limits](control-planes.md#bounded-queues-load-shedding-and-per-tenant-limits), [retry budgets](control-planes.md#retries-and-retry-budgets), [status codes](control-planes.md#errors-and-status-codes), [SLIs, SLOs and burn-rate alerts](control-planes.md#slis-slos-and-error-budgets), [Prometheus](control-planes.md#metrics-with-prometheus), [tracing](control-planes.md#tracing), [dashboards](control-planes.md#dashboards-and-alerting), [on-call](control-planes.md#on-call-and-incident-debugging) and [cells](control-planes.md#partial-outages-and-cells).
- [Deploys, Recovery and Autoscaling](deploys-recovery-autoscaling.md): startup time, rollouts and canaries, recovery and autoscaling. This chapter assumes replicas come and go and does not explain how.
- [Distributed Systems](../foundations/distributed-systems.md): [idempotency](../foundations/distributed-systems.md#idempotency), [retries with backoff and jitter](../foundations/distributed-systems.md#retries-with-exponential-backoff-and-jitter), and [availability, SLOs and nines](../foundations/distributed-systems.md#availability-slos-and-nines).

**What was run.** Every Python block with output was run as shown: the token bucket, the TPM limiter with reservation and refund, hierarchical limits, the FCFS/VTC/RPM fairness simulator, the noisy-neighbour arithmetic, the request-log generator and SLO/goodput report, the nines-and-capacity table, the deadline batch queue, the cost calculator, the quality probe, the incident watchdog and the API-key/cache-salt demo. None of it needs a GPU: the engines are simulated with step-cost models of the same shape as the one in [Serving LLMs in Practice](../inference/serving-hands-on.md#a-mock-server-you-can-run-anywhere). Vendor numbers were re-read on the linked pages in November 2026.


## 1. The API surface

### What the customer is buying

Almost every LLM API today speaks the OpenAI wire format: `POST /v1/chat/completions` with a JSON body of `messages`, `max_tokens`, sampling parameters and `stream`, answered either with one JSON object or with a Server-Sent Events (SSE) stream of chunks. [Serving LLMs in Practice](../inference/serving-hands-on.md#the-openai-compatible-api-and-streaming) took that format apart chunk by chunk. Being compatible with it is not optional for a small provider: customers switch to you by changing a `base_url` in the official SDK, and anything that SDK cannot parse is a support ticket.

Compatibility on the happy path is easy, because vLLM and SGLang already serve the format. The work is in everything the engine does not do. Here is what happens to one request in a production API, in order:

```
client --TLS--> edge / load balancer
                 |
                 v
              gateway: authenticate key -> resolve org, tier, model, adapter
                       validate (context length, max_tokens, parameters)
                       rate limits (key, org, global)        -> 429 + Retry-After
                       admission (is there capacity?)        -> 503 + Retry-After
                       fair queue / priority                 (section 3)
                 |
                 v
              router: pick a replica (prefix affinity, load)  (Serving Efficiently)
                 |
                 v
              engine: prefill, decode, stream tokens back
                 |
                 v
              gateway: count tokens, settle the rate limit, write the request log,
                       emit metrics, bill                      (sections 2, 4, 7)
```

The engine is one box in that list. The rest is this chapter.

### Streaming and cancellation

Two details about streaming matter more for an operator than for the engine.

**The HTTP status is sent before the work is done.** A streaming response starts with `200 OK` and headers, and only then do the tokens arrive. If the replica dies halfway through, the client has already seen a 200. The OpenAI format has no standard way to report that, so providers send an error event in the stream or just close it, and the client sees a stream that ended without `finish_reason` and without the usage chunk. Your access logs will record the request as a success. Section 4 shows how far apart "success by status code" and "success at completion" drift on a realistic log, and why you have to measure the second one.

**Cancellation has to reach the engine, and billing has to agree with it.** When a client disconnects, the engine must stop generating, or an abandoned request keeps its batch slot and KV blocks until `max_tokens` ([the cancellation chain](../inference/serving-hands-on.md#cancellation-end-to-end) shows the whole path). The billing question is what to charge. The usual answer is to charge for the prompt (the prefill was done) and for the output tokens generated before the disconnect was noticed. Your engine knows that count even though the client never received the usage chunk, so take it from the engine side of the gateway, not from what was relayed. Log `finish_reason = "client_cancel"` so these requests can be left out of your availability numbers (the client chose to leave) and still be counted in your token metrics (the GPU did the work).

### Idempotency and retries

A client sends a request, the connection drops before the reply, and the client retries. For a non-streaming chat completion that is a second generation and a second bill. Two rules make this safe:

- **Accept an idempotency key on non-streaming and batch endpoints.** The mechanism is the one in [Control Planes](control-planes.md#idempotency-keys): the client sends an `Idempotency-Key` header, you store the key with the response for a bounded time (per tenant), and a retry with the same key gets the stored response instead of a new generation. Two LLM-specific points apply. Sampling is random, so a regenerated answer differs from the first one; returning the stored response is the only way the client gets the answer it was billed for. And storing responses is storing customer content, so the stored copy falls under your retention policy (section 9).
- **Streams cannot be replayed, so retry only before the first byte.** Once tokens have reached the client, a retry either duplicates them or throws away work. The rule from [Serving LLMs in Practice](../inference/serving-hands-on.md#concurrency-and-client-side-backpressure) holds for the whole stack: the gateway may retry a request on another replica only while nothing has been sent to the client.

Retries also amplify overload, because every layer that retries multiplies load at the moment the system can least take it. Retry at one layer, cap retries with a [retry budget](control-planes.md#retries-and-retry-budgets), and in your SDK documentation tell customers which errors are safe to retry.

### Timeouts for long generations

LLM requests are long. A reasoning model that thinks for 20,000 tokens at 50 tokens/s takes 400 s, and a single request can be cheap to start and expensive to finish. Every hop on the path has a timeout, and the shortest one wins:

| Hop | What usually cuts it | What to do |
|---|---|---|
| client SDK | a default total timeout | document the maximum generation time; recommend streaming for long outputs |
| corporate proxies, browsers | idle timeouts on a connection with no bytes | send SSE comment lines (`: keepalive`) during long prefills, so the connection is never idle |
| your load balancer | idle and maximum-duration limits | raise them above your longest allowed generation; check them, since cloud defaults are short |
| gateway → engine | a first-byte timeout and a between-chunks timeout | set the first-byte timeout from your TTFT SLO, and the between-chunks timeout from the worst inter-token gap, never from the total length ([gateway timeouts](../inference/serving-hands-on.md#timeouts-retries-and-ejection)) |
| your own deadline | none, by default | enforce a server-side maximum (e.g. `max_tokens` cap per tier) so one request cannot hold a slot for an hour |

For work that really takes many minutes, a synchronous HTTP request is the wrong shape. Offer an asynchronous one: `POST` returns a request ID immediately, and the result is fetched later or pushed to a webhook. Baseten's async inference is a documented example ([docs](https://docs.baseten.co/inference/async)): a request can wait in its queue for 10 minutes by default and up to 72 hours, sync requests take priority ("when the model has no free slots, the load balancer rejects async work with `429`"), only 500/502/503/504 are retried, and webhook delivery is attempted twice. One sentence in those docs is worth copying into your own design review: "If all delivery attempts fail, your model output is permanently lost." If you push results, store them until delivery is confirmed.

**On a leased fleet.** The async pattern carries over unchanged and is more valuable for you than for a serverless provider, because async requests are exactly the deferrable work that fills idle leased GPUs (section 5). Baseten's "sync wins" rule is the policy you want too.

### What the client sees on overload: 429 vs 503

Clients have to decide two things from an error: whether to retry, and when. The general table of status codes is in [Control Planes](control-planes.md#errors-and-status-codes). For an inference API the important split is between **"you sent too much"** and **"we have too little"**:

- **429 Too Many Requests**: *the client* exceeded a limit it was given (rate limit, quota, spend cap). It is the client's to fix, and it should come with a `Retry-After` header saying how many seconds to wait. A token bucket can compute that number exactly (next section).
- **503 Service Unavailable**: *you* can't serve a request that was within its limits. The fleet is overloaded or a dependency is down. It also gets `Retry-After`, set from how long you expect the overload to last, plus jitter so the retries don't all arrive together.

Providers draw this line in public:

- Fireworks returns 429 when you are over your rate limit and `503 Service Overloaded` when it is shedding load, and says plainly that "Staying within your rate limits does not guarantee that every request succeeds" ([rate limits doc](https://docs.fireworks.ai/serverless/rate-limits)). It tells clients to honor `Retry-After` and otherwise use exponential backoff with jitter.
- Baseten's Model APIs use a non-standard **529**: "A `529 Overloaded` response means that the Model API has no available serving capacity", and "This response can occur even when your organization remains within its rate limits" ([pricing and limits](https://docs.baseten.co/inference/model-apis/pricing-and-limits)).
- Anthropic returns 429 for rate limits with a `retry-after` header, and also uses 429 when a monthly spend cap is reached, but then *without* `retry-after`, because retrying will not help until the month ends; the body carries `error_code: enforced_spend_limit_reached` so clients can tell the two apart ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)).

The Anthropic case points at a rule: **put a machine-readable reason in the error body**, not just a status code. Something like `rate_limit_exceeded` with the name of the limit that was hit (`key_tpm`, `org_rpm`), `overloaded`, `spend_cap_reached` or `context_length_exceeded`. Clients and your support team both use it.

Two more client-visible rules:

- **Reject early, not late.** A request that will be shed should be shed at the gateway in milliseconds, before it has a replica, not after it has waited 30 s in an engine queue. A fast 503 lets the client retry elsewhere or back off. A slow 503 has already wasted the client's time and your queue slot.
- **Never return 400 for a capacity problem, or 503 for a bad request.** A 400 is never retried. A client that gets 400 for "prompt too long for this replica's `max-model-len`" when another replica would have taken it has been told something false. And a 503 for a malformed request makes well-behaved clients retry it forever.


## 2. Rate limiting and quotas

### Requests per minute vs tokens per minute

A **rate limit** caps how fast one client may consume the service: so much per minute, enforced continuously. A **quota** caps total consumption over a long period (tokens per day, dollars per month). Both protect the fleet from one client and protect the client from runaway costs.

Counting *requests* per minute (RPM) is the classic web API limit, and for LLMs it is the wrong unit on its own. One request can be 50 tokens in and 10 out, or 200,000 in and 8,000 out, a difference of thousands of times in GPU cost. So LLM APIs meter **tokens per minute** (TPM), usually alongside RPM, and the big providers split TPM further:

- Anthropic meters RPM, input tokens per minute (ITPM) and output tokens per minute (OTPM) per model class. Its ITPM counts only *uncached* input: "With a 2,000,000 ITPM limit and an 80 percent cache hit rate, you could effectively process 10,000,000 total input tokens per minute" ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)).
- Fireworks keeps three per-model meters: total prompt TPM, uncached prompt TPM and generated TPM ([rate limits doc](https://docs.fireworks.ai/serverless/rate-limits)).
- Baseten's Model APIs use a single TPM that counts input and output, and "Cached and uncached input tokens count equally toward token rate limits" ([pricing and limits](https://docs.baseten.co/inference/model-apis/pricing-and-limits)).
- OpenAI meters RPM, RPD, TPM, TPD and images per minute, at organization and project level ([rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits)).

The split follows the cost structure. Prefill and decode use the GPU very differently ([Inference Internals I](../inference/arithmetic.md#prefill-and-decode)): a cached input token costs almost nothing, an uncached input token costs a little compute, and an output token costs a full decode step's share of memory bandwidth. Section 6 puts numbers on that. Separate meters let you protect the scarce resource (decode slots) without throttling the cheap one. Not counting cache hits against the limit also rewards clients for structuring prompts so they cache well, which is good for your fleet.

**On a leased fleet.** Copy the split meters. Input, cached input and output have very different costs on your hardware, the same as on theirs. A single TPM that counts a cached token like an output token (the Baseten choice) is simpler to explain, but it throttles cache-friendly customers you would rather encourage.

### A token bucket

The standard limiter is a **token bucket**. A bucket holds up to `capacity` tokens (here "token" means a unit of permission, which for a TPM limit happens to be an LLM token) and refills continuously at `rate` per second. A request that costs `n` is admitted only if `n` are in the bucket, and then they are removed. `capacity` is the largest allowed burst and `rate` is the sustained limit. Anthropic says its API works this way: "your capacity is continuously replenished up to your maximum limit, rather than being reset at fixed intervals", and "a rate of 60 requests per minute (RPM) might be enforced as 1 request per second" ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)).

[Control Planes](control-planes.md#bounded-queues-load-shedding-and-per-tenant-limits) used a token bucket to protect a scheduler. The version here adds the one thing an API needs on top: when it rejects, it computes how long until the request *would* fit. That number is the `Retry-After` header.

```python
# token_bucket.py - a token bucket that also tells the client how long to wait (Retry-After)
import math

class TokenBucket:
    """Holds up to `capacity` tokens and refills at `rate` tokens/s.
    A request costing n tokens is admitted only if n tokens are in the bucket."""
    def __init__(self, capacity, rate, now=0.0):
        self.capacity, self.rate = capacity, rate
        self.level, self.t = capacity, now            # start full: a new key may burst once

    def _refill(self, now):
        self.level = min(self.capacity, self.level + (now - self.t) * self.rate)
        self.t = now

    def try_take(self, n, now):
        """Return (admitted, retry_after_seconds)."""
        self._refill(now)
        if n > self.capacity:                         # can never fit: that is a 400, not a 429
            raise ValueError(f"cost {n} exceeds bucket capacity {self.capacity}")
        if self.level >= n:
            self.level -= n
            return True, 0.0
        return False, (n - self.level) / self.rate    # exact time until n tokens are available

if __name__ == "__main__":
    # "60 RPM" as a bucket: rate 1 request/s, burst 10 (Anthropic enforces RPM this way; see text)
    b = TokenBucket(capacity=10, rate=1.0)
    t, admitted, waits = 0.0, 0, []
    for i in range(25):                               # 25 requests in the first 0.5 s
        ok, ra = b.try_take(1, now=t)
        admitted += ok
        if not ok: waits.append(ra)
        t += 0.02
    print(f"burst of 25 in 0.5 s: admitted={admitted}, rejected={25 - admitted}")
    print(f"Retry-After on first/last rejection: {waits[0]:.2f} s / {waits[-1]:.2f} s "
          f"-> header value {math.ceil(waits[0])} (integer seconds, rounded up)")
    # a well-behaved client that waits exactly Retry-After succeeds on its first retry
    ok, ra = b.try_take(1, now=t)
    ok2, _ = b.try_take(1, now=t + ra)
    print(f"retry at t+Retry-After ({ra:.2f} s): admitted={ok2}")
    # sustained: one minute at 2 req/s offered -> ~60 + burst admitted
    b, adm = TokenBucket(10, 1.0), 0
    for k in range(120):
        adm += b.try_take(1, now=k * 0.5)[0]
    print(f"60 s at 2 req/s offered: admitted={adm} (= 60 refilled + 10 burst, give or take one)")
    assert 69 <= adm <= 71
```

```console
$ python3 token_bucket.py
burst of 25 in 0.5 s: admitted=10, rejected=15
Retry-After on first/last rejection: 0.80 s / 0.52 s -> header value 1 (integer seconds, rounded up)
retry at t+Retry-After (0.50 s): admitted=True
60 s at 2 req/s offered: admitted=69 (= 60 refilled + 10 burst, give or take one)
```

Three things to notice:

- **The burst is a product decision.** With `capacity` equal to a full minute of `rate`, a client can spend a whole minute's allowance in one second, and a fleet of clients that all do so at the top of the minute is a synchronized spike. Smaller buckets smooth load; larger ones are friendlier to bursty clients. Anthropic's "60 RPM might be enforced as 1 request per second" is a small bucket.
- **`Retry-After` is exact, so honor it exactly.** The client that waited 0.50 s got in on its first retry. HTTP's `Retry-After` is in whole seconds, so round up. Add a little jitter on the server side too, so that the 15 rejected clients don't all come back in the same millisecond.
- **A request bigger than the bucket is not a 429.** It can never be admitted, however long the client waits, so return a 400 that says which limit it exceeds. Otherwise a well-behaved client retries it forever.

Anthropic adds one more kind of limit on top of this: a **ramp limit**, which caps how fast an organization's usage may *grow*. "Your ramp limit is your organization's highest per-minute usage of the model in the recent past, plus a step", it rises as successful usage grows, and exceeding it returns a 429 with `error_code: ramp_limit_exceeded` even under the rate limit ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)). The reason is capacity: a customer whose limit is ten times its usage can still go from 10% to 100% in one minute, and no fleet has spare capacity for every customer's limit at once. Fireworks does something similar in slow motion: limits grow 25% when recent usage exceeds half of the limit, at most once every 15 minutes, and shrink 20% after three days of low use ([rate limits doc](https://docs.fireworks.ai/serverless/rate-limits)).

**On a leased fleet.** Ramp limits matter *more* for you. A serverless provider can buy more GPUs within minutes when a customer ramps (at a price). You can't add GPUs this month. A ramp limit is how you turn "a customer quadrupled their traffic at 10:00" from an outage for everyone into a slower ramp for one customer. Implement it as a second bucket whose capacity tracks the customer's recent peak usage plus a step.

### Estimating tokens before generation: reservation and refund

A TPM limiter has a problem that an RPM limiter does not: at admission time it does not know the cost. The prompt can be tokenized (or estimated from its character count), but the output length is unknown until generation stops. There are three policies, and providers use all of them:

1. **Charge at completion.** Admit while the bucket is positive, and subtract the actual tokens when the request finishes. Nothing is wasted, but a burst of long requests all gets admitted before any of them is charged.
2. **Reserve `max_tokens` and keep it.** Charge `prompt_estimate + max_tokens` at admission. Never overshoots, but most requests use a fraction of `max_tokens`, so most of the limit is spent on tokens never generated.
3. **Reserve, then refund.** Charge `prompt_estimate + max_tokens` at admission and, at completion, give back what was not used (or take more, if the prompt estimate was low).

OpenAI documents the reservation: "Your rate limit is calculated as the maximum of `max_tokens` and the estimated number of tokens based on the character count of your request", and advises setting `max_tokens` "as close to your expected response size as possible" ([rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits)). Anthropic took the other route for output: "OTPM rate limits are evaluated in real time as output tokens are produced, counting only the actual tokens generated. The `max_tokens` parameter does not factor into OTPM rate limit calculations" ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)). That is policy 1 done continuously, debiting as each token is generated, which needs a limiter in the token stream rather than only at the door.

The simulation puts one organization with a 100,000 TPM limit under a realistic shape of traffic: a batch script fires 200 requests at once, then steady traffic follows at 1 request per second. Each request has an estimated 1,000-token prompt (the real count is 0.8 to 1.25 times that), asks for `max_tokens = 4096`, and actually generates about 300 tokens on average.

```python
# tpm_limiter.py - tokens-per-minute limiting when output length is unknown at admission:
#   end   : admit while the bucket is positive, charge actual tokens at completion
#   hold  : reserve prompt_estimate + max_tokens at admission, never refund
#   refund: reserve prompt_estimate + max_tokens at admission, refund the unused part at completion
import random
from token_bucket import TokenBucket

TPM = 100_000                  # tokens per minute for this org (input + output, one meter)
DT, T_END = 0.1, 600.0         # 100 ms ticks, 10 simulated minutes
DECODE_S = 0.02                # 50 tok/s per stream once running

def workload(seed=0):
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(200):       # a batch script fires 200 requests at t=0 ...
        reqs.append(0.0)
    while t < T_END:           # ... then steady traffic at 1 req/s (Poisson)
        t += rng.expovariate(1.0)
        reqs.append(t)
    out = []
    for t0 in reqs:
        est = 1000                                     # prompt estimate from character count
        prompt = int(est * rng.uniform(0.8, 1.25))     # the tokenizer's real count differs
        n_out = min(4096, int(rng.expovariate(1 / 300)) + 1)   # mean ~300, capped by max_tokens
        out.append(dict(t0=t0, est=est, prompt=prompt, max_tokens=4096, n_out=n_out))
    return out

def run(policy):
    bucket = TokenBucket(capacity=TPM, rate=TPM / 60)
    backlog = workload()                       # client-side queue, sorted by arrival time
    inflight, done, served = [], [], []        # served: (t, tokens) when a request finishes
    peak_inflight, t = 0, 0.0
    while t < T_END:
        bucket._refill(t)
        for r in [r for r in inflight if r["t_done"] <= t]:      # completions: settle
            inflight.remove(r)
            actual = r["prompt"] + r["n_out"]
            if policy == "end":
                bucket.level -= actual                           # may go negative: debt
            elif policy == "refund":
                bucket.level = min(TPM, bucket.level + r["held"] - actual)   # refund (or extra debit)
            served.append((t, actual))
        while backlog and backlog[0]["t0"] <= t:                 # admit in arrival order
            r = backlog[0]
            if policy == "end":
                if bucket.level <= 0: break
            else:
                r["held"] = r["est"] + r["max_tokens"]
                if bucket.level < r["held"]: break
                bucket.level -= r["held"]
            backlog.pop(0)
            r["t_done"] = t + 0.2 + r["n_out"] * DECODE_S
            inflight.append(r)
        peak_inflight = max(peak_inflight, sum(r["prompt"] + r["n_out"] for r in inflight))
        t = round(t + DT, 1)
    per_min = [sum(n for (ts, n) in served if m * 60 <= ts < (m + 1) * 60) for m in range(10)]
    return per_min, peak_inflight

print(f"limit: {TPM:,} tokens/min.  minute-by-minute tokens actually served:")
for policy in ("end", "hold", "refund"):
    per_min, peak = run(policy)
    print(f"{policy:6s} " + " ".join(f"{x/1000:5.0f}k" for x in per_min)
          + f" | min 2-10 avg {sum(per_min[1:]) / 9 / TPM:4.0%} of limit, peak in flight {peak / 1000:.0f}k")
```

```console
$ python3 tpm_limiter.py
limit: 100,000 tokens/min.  minute-by-minute tokens actually served:
end      260k   107k    82k    85k    71k   101k    82k    89k    73k    83k | min 2-10 avg  86% of limit, peak in flight 258k
hold      45k    28k    25k    24k    25k    24k    25k    28k    21k    30k | min 2-10 avg  26% of limit, peak in flight 25k
refund   153k   105k    99k   104k    92k   102k    98k    98k    96k    83k | min 2-10 avg  98% of limit, peak in flight 27k
```

Reading the three rows:

- **Charge at completion** let all 200 burst requests in before any had finished, so 258k tokens of work were in flight against a 100k/minute limit, and 260k tokens were served in the first minute. Then the debt (the bucket went far negative) blocked the client for a while. The limit is right on average and wrong exactly when it matters, during a burst.
- **Reserve and keep** never overshoots and serves about a quarter of the limit. Each request reserves about 5,100 tokens and uses about 1,300, so the client pays for a limit four times larger than what it gets. Customers notice, and they respond by lowering `max_tokens`, which truncates their answers.
- **Reserve and refund** keeps in-flight work near the limit (27k at peak) and still serves 98% of the limit over minutes 2-10. The 153k in the first minute is not an overshoot: the bucket started full (100k of burst) and refilled another 100k during that minute. That is the burst you configured.

Production details:

- **Settle with the engine's token counts**, not your estimate. The `usage` the engine reports at the end of each request is what refunds and bills use, and it must be logged with the request (section 7).
- **Refund on every exit path**: completion, client cancel, engine error, timeout. A refund that is only on the success path leaks limit on every failure, which is when clients retry most.
- **Allow small debt.** When the prompt estimate was low, the settlement takes the bucket slightly negative. Don't reject a request that already ran. Let the debt delay the next one.
- **Cap `max_tokens` per tier** so a single reservation can't take a whole minute's limit.

**On a leased fleet.** Reserve-and-refund is the one to build. It is what keeps a client's in-flight work bounded, and in-flight work is what consumes your fixed KV capacity. Anthropic's real-time OTPM debit is better for the client, but it needs the limiter in the token path of every stream; build it once you have the request log working.

### Per-key, per-org and global limits

Limits nest. An **API key** belongs to a **project** or workspace, which belongs to an **organization** (the paying customer), which shares the **fleet** with everyone else. Each level gets its own buckets:

- **Per key**: stops one leaked or buggy key (a CI job in a loop) from taking the whole organization's allowance.
- **Per org**: what the customer pays for, and what their tier says.
- **Global** (per model): what your fleet can actually serve at its SLO, from measured goodput (section 4), not a guess. It is the limit that turns overload into fast 503s at the gateway instead of slow queueing in the engines.

Anthropic documents the same nesting: workspace limits inside the organization limit, where "Organization-wide limits always apply, even if workspace limits add up to more" ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)); OpenAI limits at both organization and project level ([rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits)).

A request must fit in every bucket on its path, and it must take from all of them or none, otherwise a request rejected by the org limit has already spent the key's tokens:

```python
# hier_limits.py - per-key, per-org and global limits: a request must fit in every bucket on its path
from token_bucket import TokenBucket

class Limits:
    def __init__(self):
        self.b = {}
    def add(self, name, per_min):
        self.b[name] = TokenBucket(capacity=per_min, rate=per_min / 60)
    def try_take(self, path, n, now):
        """path = ["key:abc", "org:acme", "global"]. All-or-nothing: check every level, then take."""
        waits = {}
        for name in path:
            b = self.b[name]; b._refill(now)
            if b.level < n:
                waits[name] = (n - b.level) / b.rate
        if waits:
            name = max(waits, key=waits.get)          # tell the client about the slowest bucket
            return False, name, waits[name]
        for name in path:
            self.b[name].level -= n
        return True, None, 0.0

L = Limits()
L.add("global", 300_000)                  # what the fleet can actually serve (from measured goodput)
L.add("org:acme", 200_000); L.add("org:zeta", 200_000)
L.add("key:acme-prod", 200_000); L.add("key:acme-ci", 50_000); L.add("key:zeta-prod", 200_000)
path = lambda key: [f"key:{key}", f"org:{key.split('-')[0]}", "global"]

for key, n in [("acme-ci", 40_000), ("acme-ci", 40_000),        # CI key hits its own cap
               ("acme-prod", 150_000), ("acme-prod", 20_000),    # org cap: 40k + 150k + 20k > 200k
               ("zeta-prod", 100_000), ("zeta-prod", 100_000)]:  # global cap: 290k used, 10k left
    ok, who, wait = L.try_take(path(key), n, now=0.0)
    print(f"{key:10s} {n:7,} -> " + ("admitted" if ok else f"429, limited by {who}, Retry-After {wait:.1f} s"))
```

```console
$ python3 hier_limits.py
acme-ci     40,000 -> admitted
acme-ci     40,000 -> 429, limited by key:acme-ci, Retry-After 36.0 s
acme-prod  150,000 -> admitted
acme-prod   20,000 -> 429, limited by org:acme, Retry-After 3.0 s
zeta-prod  100,000 -> admitted
zeta-prod  100,000 -> 429, limited by global, Retry-After 18.0 s
```

The last line deserves a second look. The two organizations' limits add up to 400k against a global 300k: the fleet is **oversold**. That is normal and deliberate, the same bet an airline makes, because customers rarely all peak together. But it means the global rejection is a different kind of event from the others: zeta was within everything it paid for. That one is *your* capacity problem, so it should be a 503 (overloaded), not a 429, and it should page someone if it persists. The code returns the name of the limiting bucket so the gateway can pick the status code and the error reason.

In a real gateway these buckets live in shared state, because many gateway replicas admit requests for the same organization. The usual design is Redis, with one atomic script that checks and takes from all buckets on the path (Redis runs a Lua script atomically, so no other client sees a half-applied take). A cheaper alternative is to split each organization's rate across the N gateway replicas and keep the buckets in memory; it is exact only if the load balancer spreads each organization evenly. [Control Planes](control-planes.md#bounded-queues-load-shedding-and-per-tenant-limits) discusses both.

### Why per-client RPM limits waste capacity

Rate limits are often the only fairness mechanism a young API has: give every client a limit equal to its share of capacity, and nobody can take more than their share. The problem is that a limit is enforced even when nobody else wants the capacity. If only one client is active, it is still held to its share and the rest of the fleet idles. A scheduler with that property is not **work-conserving**: it leaves capacity unused while work is waiting.

Sheng et al. measured this in "Fairness in Serving Large Language Models" ([arXiv 2401.00588](https://arxiv.org/abs/2401.00588)), on a trace from LMSYS Chatbot Arena. Their table 2, in output tokens per second:

| Scheduler | Max service difference | Throughput (tok/s) |
|---|---|---|
| FCFS | 759.97 | 777 |
| VTC | 368.40 | 779 |
| RPM(5) | 143.86 | 340 |
| RPM(20) | 446.76 | 694 |
| RPM(30) | 693.66 | 747 |

A 5-requests-per-minute limit per client makes service the most even, at 340 tok/s instead of 779: $1 - 340/779 = 56\%$ of the capacity unused. Looser limits waste less and are less fair. The paper describes RPM limiting as working "as an FCFS approach with admission control" rather than as a fair scheduler, and notes it "may not be work-conserving when the number of active clients is low". VTC, the scheduler in the second row, gets better fairness than FCFS at the same throughput. The next section builds it.

**On a leased fleet.** This result is sharper for you than for anyone with elastic capacity. Unused leased capacity is money already spent for nothing. Keep rate limits for what they are good at (stopping abuse and runaway clients, enforcing what a tier paid for), set them generously, and get fairness from the scheduler instead.


## 3. Fairness and priority across tenants

### FCFS and its failure

Rate limits decide what gets *in*. Once requests are in, something decides which one gets the next free batch slot. Most engines use **first-come, first-served** (FCFS): the waiting queue is ordered by arrival ([engine policies](../inference/serving-engine-internals.md#policies)). Under overload FCFS gives each tenant service in proportion to how much it sends. A tenant that sends three times as much gets three times the service, and a light tenant's request waits behind the heavy tenant's queue. That last part is the one customers notice: the light tenant did nothing wrong and still sees its TTFT go from 300 ms to 30 s.

What you want is closer to **max-min fairness**: a tenant that asks for less than an equal share gets everything it asks for, and the tenants that ask for more split what is left equally. Network routers solved this for packets decades ago with **fair queuing**, which tracks how much service each flow has received and serves the flow that has received least. The complication for LLMs is that the cost of a request is not known when it is admitted (output length again), and that input and output tokens cost different amounts.

### Virtual Token Counter

The **Virtual Token Counter** (VTC) scheduler from the Sheng et al. paper ([arXiv 2401.00588](https://arxiv.org/abs/2401.00588)) adapts fair queuing to continuous batching:

- **Service is weighted tokens**: $W = w_p \cdot n_{\text{in}} + w_q \cdot n_{\text{out}}$. The paper uses $w_p = 1$ and $w_q = 2$, following OpenAI's pricing at the time, because output tokens are the more expensive ones.
- **Each client has a counter** of the service it has received. When a batch slot frees up, the scheduler admits the head request of the client with the *smallest* counter.
- **Charge as you go.** Input tokens are added to the counter at admission. Output tokens are added one decode step at a time, as they are generated. So the unknown output length is never needed in advance.
- **Lift the counter when a client returns.** A client that was idle would otherwise come back with a low counter and get a burst of service as "back pay". When a client goes from no queued requests to some, its counter is raised to at least the smallest counter among queued clients (or, if none are queued, to that of the last client to leave the queue). In the paper's words, credits "cannot be carried over".

The paper proves that two clients that are both continuously backlogged never differ in received service by more than $2 \cdot \max(w_p L_{\text{in}}, w_q M)$, where $L_{\text{in}}$ is the maximum input length and $M$ the maximum number of tokens in a running batch, and that no work-conserving non-preemptive scheduler can guarantee better than $w_q M$.

The simulator below is a toy continuous-batching engine: a KV budget of 10,000 tokens, every request 256 tokens in and 256 out, one decode step every 20 ms. It can therefore run 19 requests at once and finish at most 3.71 requests per second. Three clients send Poisson traffic:

- **A** is heavy (4 req/s, more than the whole engine can do by itself) and is the only one active for the first 60 s.
- **B** is medium (2 req/s) and joins at 60 s.
- **C** is light (0.3 req/s, well under a third of capacity) and joins at 60 s.

It runs FCFS, VTC, and FCFS with a per-client request-rate limit set to a third of capacity each (the "fair share" RPM policy).

```python
# vtc_sim.py - FCFS vs Virtual Token Counter (VTC) vs per-client request-rate limits
# on a toy continuous-batching engine. Service is measured as in the VTC paper:
#   W = w_p * input_tokens + w_q * output_tokens,  w_p = 1, w_q = 2
import random
from collections import deque

W_P, W_Q = 1, 2
M = 10_000                 # KV capacity in tokens; a request reserves input + max_tokens
STEP = 0.02                # one decode step = 20 ms: every running request emits 1 token
N_IN, N_OUT = 256, 256     # every request: 256 in, 256 out (so cost differences are pure arrival rate)
PHASE2, T_END = 60.0, 180.0
RATES = {"A": 4.0, "B": 2.0, "C": 0.3}    # req/s; only A is active before t = 60 s

def arrivals(seed):
    rng, out = random.Random(seed), []
    for c, lam in RATES.items():
        t = 0.0 if c == "A" else PHASE2
        while True:
            t += rng.expovariate(lam)
            if t >= T_END: break
            out.append((t, c))
    return sorted(out)

def simulate(policy, seed=0, rpm_rate=None):
    arr = deque(arrivals(seed))
    q = {c: deque() for c in RATES}           # per-client FIFO of waiting requests (arrival times)
    fifo = deque()                            # global arrival order, for FCFS / RPM
    vtc = {c: 0.0 for c in RATES}             # virtual counters
    last_left = 0.0                           # counter of the last client whose queue emptied
    tokens = {c: 1.0 for c in RATES}          # per-client request buckets for the RPM policy
    running, used = [], 0                     # running: [client, tokens_left, t_arrival]
    served = {c: 0.0 for c in RATES}          # service W since PHASE2
    lat = {c: [] for c in RATES}              # queue wait (admission - arrival) since PHASE2
    out_tok = {"p1": 0, "p2": 0}
    max_gap_ab, t = 0.0, 0.0
    while t < T_END:
        while arr and arr[0][0] <= t:                     # arrivals
            ta, c = arr.popleft()
            if policy == "vtc" and not q[c]:              # client becomes backlogged: lift counter
                backlogged = [vtc[k] for k in RATES if q[k]]
                vtc[c] = max(vtc[c], min(backlogged) if backlogged else last_left)
            q[c].append(ta); fifo.append((ta, c))
        if policy == "rpm":
            for c in RATES: tokens[c] = min(1.0, tokens[c] + rpm_rate * STEP)
        while True:                                       # admission, non-preemptive
            if used + N_IN + N_OUT > M: break
            if policy == "vtc":
                cands = [c for c in RATES if q[c]]
                if not cands: break
                c = min(cands, key=lambda k: vtc[k])
                ta = q[c].popleft(); fifo.remove((ta, c))
                vtc[c] += W_P * N_IN                      # input charged at admission
                if not q[c]: last_left = vtc[c]
            else:
                pick = None
                for (ta, c) in fifo:                      # FCFS; RPM skips clients out of tokens
                    if policy == "fcfs" or tokens[c] >= 1: pick = (ta, c); break
                if pick is None: break
                fifo.remove(pick); ta, c = pick; q[c].popleft()
                if policy == "rpm": tokens[c] -= 1
            running.append([c, N_OUT, ta]); used += N_IN + N_OUT
            if t >= PHASE2:
                served[c] += W_P * N_IN; lat[c].append(t - ta)
        for r in running:                                 # one decode step
            r[1] -= 1
            if policy == "vtc": vtc[r[0]] += W_Q          # output charged per generated token
            if t >= PHASE2: served[r[0]] += W_Q
            out_tok["p2" if t >= PHASE2 else "p1"] += 1
        for r in [r for r in running if r[1] == 0]:
            running.remove(r); used -= N_IN + N_OUT
        if t >= PHASE2:
            max_gap_ab = max(max_gap_ab, abs(served["A"] - served["B"]))
        t = round(t + STEP, 2)
    p2 = T_END - PHASE2
    return dict(thr1=out_tok["p1"] / PHASE2, thr2=out_tok["p2"] / p2,
                share={c: served[c] / p2 for c in RATES},
                wait={c: sum(v) / len(v) if v else float("nan") for c, v in lat.items()},
                gap=max_gap_ab)

cap = M // (N_IN + N_OUT)
print(f"engine: {cap} concurrent requests, {cap / (N_OUT * STEP):.2f} req/s max; offered in phase 2: "
      f"{sum(RATES.values()):.1f} req/s (A {RATES['A']}, B {RATES['B']}, C {RATES['C']})")
print(f"{'policy':10s} {'out tok/s':>9s} {'out tok/s':>9s} | {'service/s (W) since t=60':^26s} | "
      f"{'mean queue wait, s':^20s} | max|W_A-W_B|")
print(f"{'':10s} {'A alone':>9s} {'all 3':>9s} | {'A':>8s} {'B':>8s} {'C':>8s} | {'A':>6s} {'B':>6s} {'C':>6s} |")
res = {}
for name, pol, kw in [("FCFS", "fcfs", {}), ("VTC", "vtc", {}),
                      ("RPM-limit", "rpm", {"rpm_rate": cap / (N_OUT * STEP) / 3})]:
    r = res[name] = simulate(pol, **kw)
    print(f"{name:10s} {r['thr1']:9.0f} {r['thr2']:9.0f} | " + " ".join(f"{r['share'][c]:8.0f}" for c in "ABC")
          + " | " + " ".join(f"{r['wait'][c]:6.1f}" for c in "ABC") + f" | {r['gap']:10,.0f}")

bound = 2 * max(W_P * N_IN, W_Q * M)
print(f"VTC paper bound on the gap between two backlogged clients: 2*max(w_p*L_in, w_q*M) = {bound:,}")
assert res["VTC"]["gap"] <= bound < res["FCFS"]["gap"]
assert res["VTC"]["wait"]["C"] < 1.0 < res["FCFS"]["wait"]["C"]
```

```console
$ python3 vtc_sim.py
engine: 19 concurrent requests, 3.71 req/s max; offered in phase 2: 6.3 req/s (A 4.0, B 2.0, C 0.3)
policy     out tok/s out tok/s |  service/s (W) since t=60  |  mean queue wait, s  | max|W_A-W_B|
             A alone     all 3 |        A        B        C |      A      B      C |
FCFS             858       950 |     1787      960      109 |   28.3   29.2   30.1 |     99,308
VTC              858       950 |     1349     1325      182 |   38.4   14.7    0.4 |      3,536
RPM-limit        299       678 |      936      922      182 |   79.6   29.2    0.3 |      1,880
VTC paper bound on the gap between two backlogged clients: 2*max(w_p*L_in, w_q*M) = 40,000
```

What each policy did:

- **FCFS** kept the engine full (950 output tok/s is 19 streams at 50 tok/s) and handed out service in proportion to demand: A got almost twice B's service, and the gap between them grew to 99,308 weighted tokens by the end and would keep growing. The light client C waited **30 s** on average for each request, the same as everyone else, because it queued behind A's backlog.
- **VTC** kept the same throughput in both phases, split the contended capacity almost equally between A and B (1,349 vs 1,325), and kept their gap at 3,536, inside the paper's bound of 40,000. C waited **0.4 s**: it always had the smallest counter, so its requests went to the front. A waits longest, which is correct, since A is the client sending more than the whole engine can serve.
- **The RPM limit** was the most even (gap 1,880) and threw away capacity in both phases. With A alone it served 299 tok/s instead of 858, about two thirds of the engine idle while A's requests waited 80 s. With all three active it served 678 instead of 950, because C uses less than its third and nobody may use the rest. This is the paper's RPM result in miniature.

(One quirk of the toy: C's service is lower under FCFS than under the other two only because C's queue was still draining at the end of the run. The gap in queue wait is the real effect.)

What a production version adds:

- **Where it runs.** VTC needs to control which request enters the batch next, so it belongs where the queue is. Open-source engines mostly do FCFS or priority in their core ([engine policies](../inference/serving-engine-internals.md#policies)), so the practical place is a queue in the gateway or router that holds requests until a replica has room and releases them in VTC order. That only works if the engines themselves are not queueing much, so cap each replica's in-flight requests near its running batch size, and let the waiting happen in the gateway, where fairness is applied.
- **Weights per tier.** Weighted fair queuing gives a tenant with weight 2 twice the share of a tenant with weight 1 when both are backlogged. Divide each client's counter increments by its weight, and paying more buys a bigger share under contention without starving anyone.
- **Cost weights from your own costs.** $w_q/w_p$ should be your output-to-input cost ratio. Section 6 derives it from the hardware. It comes out much higher than 2, and cached input should cost close to nothing.

**On a leased fleet.** VTC is a scheduler, not a capacity model, so it applies unchanged. It matters more for you than for an elastic provider: when you are oversold and overloaded, you cannot buy your way out of the contention, so who waits is decided by the scheduler. VTC is also work-conserving, which is the property that keeps leased GPUs busy when only one tenant is active.

### Priority classes: interactive, batch and best-effort

Fairness among equals is one problem; traffic of different *kinds* is another. Most inference products end up with three classes:

| Class | Example | Latency target | Admission under overload | Pricing |
|---|---|---|---|---|
| **interactive** | chat, coding assistants, agents | TTFT in hundreds of ms, steady TPOT | admitted first; shed last | list price, or a premium for a priority tier |
| **batch** | evaluations, synthetic data, nightly document processing | a deadline in hours | waits for spare capacity; expires at the deadline | discounted (section 5) |
| **best-effort** | free tier, internal experiments | none | shed first, with a fast 503 | free or cheapest |

Order the queue by class first, then by VTC within a class. Strict priority between classes is simple and usually right for interactive over batch, since batch has hours of slack. Two guards keep it from going wrong:

- **Reserve headroom for the top class.** If batch work fills every slot, an interactive burst has to wait for batch requests to finish, which at a few hundred tokens each is seconds. Cap the lower classes at, say, 90% of slots or KV, so interactive arrivals always find room. The batch queue in section 5 does exactly this.
- **Don't let a priority tier become the default.** Anthropic offers a Priority Tier with its own rate-limit headers ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)), and Fireworks a Priority tier that "is prioritized during high load" and shares the same rate limits as regular requests ([rate limits doc](https://docs.fireworks.ai/serverless/rate-limits)). If everyone buys priority, there is no lower class left to shed. Sell priority as a capacity commitment you have actually reserved, and price it so.

vLLM's `--scheduling-policy priority` orders admission and preemption by a per-request priority ([engine policies](../inference/serving-engine-internals.md#policies)), so the class can be passed down to the engine as well as enforced in the gateway queue.

**On a leased fleet.** Classes are how a fixed fleet serves a variable load. Interactive demand peaks during the day; batch and best-effort fill the rest. Without classes, you size the fleet for the interactive peak and waste the trough. With them, the trough is sold as batch.

### Preemption

**Preemption** means taking resources away from work that is already running. It happens at two levels.

**Inside an engine**, when the KV cache fills, the scheduler evicts a running sequence and later either swaps its KV back from host memory or recomputes it ([swap vs recompute](../inference/serving-engine-internals.md#preemption-swap-vs-recompute)). With priority scheduling, the victim is the lowest-priority, most recently admitted sequence. The cost is real: a recomputed victim pays its prefill again. A steady rate of preemptions means the engine is oversubscribed, and you should admit less (section 8, KV exhaustion).

**Across the fleet**, a whole replica is taken away from a lower class: batch replicas are drained to make room for an interactive surge. Fireworks sells capacity on these terms. Preemptible deployments cost "50% off the on-demand GPU rate", Fireworks "can take the GPUs back at any time", and "A replica can be preempted mid-request without warning", so in-flight requests fail and the client must retry ([on-demand deployments doc](https://docs.fireworks.ai/guides/ondemand-deployments)).

Rules that keep preemption cheap:

- **Preempt the cheapest work to redo.** A batch request 20 tokens into its output has lost almost nothing. One 3,000 tokens in has lost a lot. Prefer victims by tokens generated so far, not only by class.
- **Never preempt an interactive stream that has started.** The client has seen tokens. Preempting means a broken response, or a re-run with different sampling.
- **Make batch requests restartable.** A batch request that is preempted goes back in the queue, not to the customer as an error. Since batch requests are not streamed, re-running from scratch is invisible to the customer except in completion time.

**On a leased fleet.** You don't sell your GPUs back to a cloud, so fleet-level preemption is a move between your own classes: batch replicas give way to interactive. The Fireworks model, a discounted class that can be interrupted at any time, is the right one for the work you put on spare capacity.

### Dedicated vs shared deployments, noisy neighbours

A **shared** (multi-tenant) deployment runs many customers' requests in the same batches on the same replicas, priced per token. A **dedicated** deployment gives one customer whole replicas, priced per GPU-hour. Both are common. Together, for example, sells serverless inference (largely best effort), dedicated GPUs, and in between them Provisioned Throughput: reserved per-model capacity at token prices, committed for a month or more, with a 99% uptime SLA, through the same API ([announcement, July 2026](https://www.together.ai/blog/provisioned-throughput), [docs](https://docs.together.ai/docs/inference/provisioned-throughput)). Fireworks' docs say that on dedicated deployments the prompt cache is shared across all requests by default, while its serverless caches are separate per account "to prevent data leakage and timing attacks" ([prompt caching doc](https://docs.fireworks.ai/guides/prompt-caching)).

Sharing is what makes per-token pricing cheap. Tenants' peaks don't coincide, so the fleet is sized for the peak of the sum rather than the sum of the peaks, and everyone's requests fill one large batch. The cost is **noisy neighbours**: one tenant's traffic degrades another's in a shared batch. The most visible case is a **prefill stall**: a long prompt is prefilled in chunks interleaved with everyone's decode steps, and each of those steps is slower. With the step-cost model of the mock engine from [Serving LLMs in Practice](../inference/serving-hands-on.md#a-mock-server-you-can-run-anywhere):

```python
# neighbour.py - one tenant's 100k-token prompt, chunked into a shared batch, seen by 32 other streams.
# Step-cost model from the mock engine in Serving LLMs in Practice: t = 8 ms + 0.25 ms*decodes + 0.02 ms*prefill tokens
T0, C_DEC, C_PRE = 8.0, 0.25, 0.02
decodes = 32
quiet = T0 + C_DEC * decodes
for chunk in (8192, 2048):
    steps = -(-100_000 // chunk)
    busy = T0 + C_DEC * decodes + C_PRE * chunk
    print(f"chunk {chunk:5d}: {steps:3d} steps at {busy:5.1f} ms instead of {quiet:.1f} ms -> every other stream's "
          f"ITL is {busy / quiet:.1f}x for {steps * busy / 1000:.1f} s")
```

```console
$ python3 neighbour.py
chunk  8192:  13 steps at 179.8 ms instead of 16.0 ms -> every other stream's ITL is 11.2x for 2.3 s
chunk  2048:  49 steps at  57.0 ms instead of 16.0 ms -> every other stream's ITL is 3.6x for 2.8 s
```

One customer's long-document request makes 32 other customers' streams stutter for over two seconds. Smaller chunks spread the damage thinner. Routing long prompts to replicas reserved for them, or to disaggregated prefill ([Serving Efficiently](../inference/serving-efficiently.md)), removes it. The other mechanisms:

- **KV pressure.** A tenant with long contexts fills the KV cache, which caps everyone's batch size and causes preemptions. Per-tenant caps on in-flight tokens (not just requests) bound this.
- **Cache eviction.** A tenant streaming many unique prompts evicts other tenants' cached prefixes, and their TTFT rises.
- **Rate-limit sharing at the global level** (the 503 in `hier_limits.py`).

When a customer needs guaranteed latency, a custom model, or isolation for compliance, they get dedicated capacity. Section 6 works out when dedicated is cheaper for them.

**On a leased fleet.** A dedicated replica sold by the hour shifts utilization risk to the customer: you are paid whether or not they use it. That is attractive when your own utilization is uncertain. Sell dedicated replicas on terms that match your lease (monthly, not hourly) if you can. Otherwise an hourly customer who leaves leaves you holding the GPUs.


## 4. SLOs and goodput

### SLOs per tier

The vocabulary is from [Distributed Systems](../foundations/distributed-systems.md#availability-slos-and-nines) and [Control Planes](control-planes.md#slis-slos-and-error-budgets): a **service level indicator** (SLI) is a measured number about what users experience, a **service level objective** (SLO) is a target for it over a window, and the **error budget** is the fraction the SLO allows to fail. An **SLA** is an SLO written into a contract with a penalty.

For an LLM API, "the request succeeded" is not enough as an SLI, because a request that streams at 2 tokens per second has succeeded and is useless. The SLI that matches what a user feels is per request: **the request completed, its TTFT was under a target, and its TPOT was under a target.** Both latency metrics are defined in [Inference Internals I](../inference/arithmetic.md#definitions). Targets differ per tier, because the tiers sell different things. The numbers below are the ones this chapter's examples use. They are a design choice, not a standard:

| Tier | TTFT target | TPOT target | Good-request SLO | Notes |
|---|---|---|---|---|
| interactive | 1 s | 50 ms (20 tok/s, faster than reading speed) | 99% of requests good, 30 days | what chat and agents need |
| standard | 5 s | 100 ms | 99% | cheaper per token; tolerates queueing |
| batch | (none) | (none) | 99% of requests completed by the deadline | section 5 |

Write the SLO per request *class* where prompt length varies a lot: a 100k-token prompt cannot have the same TTFT as a 1k-token one, since prefill alone takes longer. Either scale the TTFT target with prompt length (for example 1 s plus 0.1 ms per input token) or report SLOs separately per prompt-length bucket.

### Goodput

DistServe ([arXiv 2401.09670](https://arxiv.org/abs/2401.09670)) made **goodput** the standard metric for LLM serving: "the maximum rate that can be served within both TTFT and TPOT constraints on each GPU". In operation, goodput is requests per second that *met the SLO*. Throughput counts every completed request, including slow ones, and you can always raise throughput by batching more and making everyone slower. Goodput can't be gamed that way. Report it per tier and per GPU, and use it, not throughput, as the capacity number in your global rate limit (section 2) and in capacity planning ([Serving LLMs in Practice](../inference/serving-hands-on.md#capacity-planning-from-measured-numbers)).

### Measure at completion, not at the gateway

Where you measure decides what you see. Together's post on what uptime means for inference puts it in one sentence: "We measure at inference completion, not at the gateway" ([Together, July 2026](https://www.together.ai/blog/99-9-uptime-for-inference)). A request that reached the load balancer and then failed on the GPU counts as downtime in their accounting. The same post names silent corruption from VRAM ECC errors as "the most common thing we see; they corrupt weights silently", which is a failure that no status code shows at all (section 7).

Here is why the difference is large for streaming APIs, with real numbers from a simulated hour. `make_log.py` writes one JSON line per request, the per-request log section 7 asks you to keep, with three problems injected: replica `r3` has a failing GPU from minute 20 to 40 and 30% of its streams die after the 200 has been sent; minutes 45-50 are overloaded; and one customer is over quota.

```python
# make_log.py - write one hour of synthetic per-request logs (JSON lines) with three injected problems:
#   1. replica r3 has a failing GPU from minute 20 to 40: 30% of its streams die mid-generation
#   2. minutes 45-50 are overloaded: queueing adds 0-3 s to TTFT and the gateway sheds 5% with 503
#   3. a customer over quota gets 429s (their fault, not ours)
import json, random
rng = random.Random(7)
REPLICAS = ["r0", "r1", "r2", "r3"]
with open("requests.jsonl", "w") as f:
    t, i = 0.0, 0
    while t < 3600:
        t += rng.expovariate(10.0)                       # 10 req/s
        i += 1
        tier = "interactive" if rng.random() < 0.7 else "standard"
        tenant = rng.choice(["acme", "zeta", "orbit", "quota-hog"])
        rec = dict(id=f"req_{i:06d}", ts=round(t, 3), tenant=tenant, tier=tier,
                   n_in=int(rng.lognormvariate(7, 1)), max_tokens=1024)
        overloaded = 45 * 60 <= t < 50 * 60
        if tenant == "quota-hog" and rng.random() < 0.2:
            rec.update(http=429, finish="rate_limited", n_out=0); f.write(json.dumps(rec) + "\n"); continue
        if overloaded and rng.random() < 0.05:
            rec.update(http=503, finish="shed", n_out=0); f.write(json.dumps(rec) + "\n"); continue
        rep = rng.choice(REPLICAS)
        ttft = 0.15 + rec["n_in"] / 20000 + (rng.uniform(0, 3) if overloaded else rng.expovariate(1 / 0.05))
        tpot = rng.gauss(0.030, 0.004) + (0.02 if overloaded else 0)
        n_out = min(1024, int(rng.expovariate(1 / 250)) + 1)
        finish = "stop" if n_out < 1024 else "length"
        if rep == "r3" and 20 * 60 <= t < 40 * 60 and rng.random() < 0.3:
            n_out, finish = rng.randint(1, n_out), "error"   # stream dies after headers were sent
        elif rng.random() < 0.02:
            n_out, finish = rng.randint(1, n_out), "client_cancel"
        rec.update(http=200, replica=rep, n_out=n_out, finish=finish, ttft_ms=round(ttft * 1e3, 1),
                   tpot_ms=round(tpot * 1e3, 2), e2e_ms=round((ttft + (n_out - 1) * tpot) * 1e3, 1))
        f.write(json.dumps(rec) + "\n")
print(open("requests.jsonl").readline().strip())
print(sum(1 for _ in open("requests.jsonl")), "records")
```

```console
$ python3 make_log.py
{"id": "req_000001", "ts": 0.039, "tenant": "acme", "tier": "interactive", "n_in": 225, "max_tokens": 1024, "http": 200, "replica": "r2", "n_out": 23, "finish": "stop", "ttft_ms": 205.0, "tpot_ms": 32.35, "e2e_ms": 916.6}
36159 records
```

The report computes, per tier: availability as an access log sees it (non-5xx status codes), availability at completion (finished with `stop` or `length`), latency percentiles, throughput, goodput, and the **burn rate**, the rate at which the error budget is being spent relative to plan (1.0 means the budget lasts exactly the window; the alerting arithmetic is in [Control Planes](control-planes.md#slis-slos-and-error-budgets)). The customer's own 429s and cancellations are left out of the denominator: they are not failures of the service.

```python
# slo_report.py - availability, latency percentiles, goodput and error-budget burn from a request log
import json, math
from collections import defaultdict

SLO = {  # per tier: a request is GOOD if it completed and met both latency targets
    "interactive": dict(ttft_ms=1000, tpot_ms=50, target=0.99),
    "standard":    dict(ttft_ms=5000, tpot_ms=100, target=0.99),
}
recs = [json.loads(l) for l in open("requests.jsonl")]
span_s = recs[-1]["ts"] - recs[0]["ts"]

def pct(xs, p):                                   # nearest-rank percentile
    xs = sorted(xs); return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]

def counted(r):
    # out of the denominator: the customer's own 429s and the customer's own cancellations
    return r["finish"] not in ("rate_limited", "client_cancel")

def completed(r):
    return r["http"] == 200 and r["finish"] in ("stop", "length")

def good(r):
    s = SLO[r["tier"]]
    return completed(r) and r["ttft_ms"] <= s["ttft_ms"] and r["tpot_ms"] <= s["tpot_ms"]

print(f"{len(recs):,} requests over {span_s / 60:.0f} min\n")
print(f"{'tier':12s} {'n':>6s} {'avail@gw':>9s} {'avail@done':>10s} {'TTFT p50/p99 ms':>16s} "
      f"{'TPOT p99':>8s} {'thru r/s':>8s} {'goodput':>8s} {'good %':>7s} {'burn':>5s}")
for tier, s in SLO.items():
    rs = [r for r in recs if r["tier"] == tier and counted(r)]
    ok_gw = sum(r["http"] < 500 for r in rs) / len(rs)       # what an access log shows: status codes
    ok_done = sum(completed(r) for r in rs) / len(rs)        # measured at inference completion
    done = [r for r in rs if completed(r)]
    g = sum(good(r) for r in rs)
    burn = (1 - g / len(rs)) / (1 - s["target"])
    print(f"{tier:12s} {len(rs):6d} {ok_gw:9.3%} {ok_done:10.3%} "
          f"{pct([r['ttft_ms'] for r in done], 50):7.0f}/{pct([r['ttft_ms'] for r in done], 99):<8.0f} "
          f"{pct([r['tpot_ms'] for r in done], 99):8.1f} {len(done) / span_s:8.2f} {g / span_s:8.2f} "
          f"{g / len(rs):7.2%} {burn:5.1f}")

# where did the bad requests come from? group by 5-minute bucket and by replica
print("\nbad interactive requests by 5-minute window (only windows with >1% bad):")
by_win = defaultdict(lambda: [0, 0])
for r in recs:
    if r["tier"] == "interactive" and counted(r):
        w = int(r["ts"] // 300); by_win[w][0] += 1; by_win[w][1] += not good(r)
for w, (n, bad) in sorted(by_win.items()):
    if bad / n > 0.01: print(f"  min {w * 5:2d}-{w * 5 + 5:2d}: {bad:4d}/{n:4d} bad ({bad / n:5.1%})")
print("mid-stream errors by replica (HTTP 200, finish=error):",
      {rep: sum(1 for r in recs if r.get("replica") == rep and r["finish"] == "error")
       for rep in ("r0", "r1", "r2", "r3")})
```

```console
$ python3 slo_report.py
36,159 requests over 60 min

tier              n  avail@gw avail@done  TTFT p50/p99 ms TPOT p99 thru r/s  goodput  good %  burn
interactive   23601   99.542%    97.098%     266/2877         54.7     6.37     5.89  89.86%  10.1
standard      10145   99.517%    97.122%     267/2953         54.6     2.74     2.74  97.12%   2.9

bad interactive requests by 5-minute window (only windows with >1% bad):
  min 20-25:  140/2030 bad ( 6.9%)
  min 25-30:  143/1955 bad ( 7.3%)
  min 30-35:  154/1944 bad ( 7.9%)
  min 35-40:  169/2019 bad ( 8.4%)
  min 45-50: 1736/1974 bad (87.9%)
mid-stream errors by replica (HTTP 200, finish=error): {'r0': 0, 'r1': 0, 'r2': 0, 'r3': 820}
```

What the report says:

- **The access log says 99.54%; completion says 97.10%.** The 820 streams that died on `r3` all returned HTTP 200, so a status-code SLI never saw them. Measured at the gateway, this hour looks like a minor overload blip. Measured at completion, the interactive tier lost 2.9% of requests to a hardware fault.
- **Goodput is 5.89 req/s against 6.37 completed.** Adding the latency targets removes another 8% of the requests, almost all from the overloaded window, where 88% of interactive requests missed TTFT or TPOT while still returning 200.
- **The interactive tier burned its budget at 10x** for this hour. The standard tier, with looser latency targets, was hurt only by the failures (burn 2.9). The same incident costs different tiers different amounts of budget, which is why budgets are per tier.
- **The breakdowns point at the causes.** Bad requests by time window give the two incidents; mid-stream errors by replica give the bad GPU. That is why `replica` belongs in every log line.

The practical rule: **compute SLIs from a log written at completion**, with the finish reason and the engine's token counts, and treat dashboards built on access-log status codes as a lower bound on how bad things are.

### Error budgets

An error budget turns an SLO into a decision rule: while the budget lasts, ship changes; when it is spent, stop and fix reliability. The alerting arithmetic (multi-window burn-rate alerts) is in [Control Planes](control-planes.md#slis-slos-and-error-budgets). Three things are specific to inference:

- **Latency misses spend budget.** With a goodput SLI, an overloaded hour spends budget even if nothing errored. That is the right incentive: it makes capacity a reliability problem, which it is.
- **Budgets are per tier, and per model.** A new model with a buggy chat template can burn its own budget while the others are fine. Aggregating them hides it.
- **Spend budget on purpose.** Deploys, engine upgrades and kernel changes are where most incidents come from (section 7's postmortem is a list of them). A team with budget left can afford a canary on a risky engine upgrade; a team without should freeze.

### What 99.9% vs 99.99% means, and the capacity it needs

An availability target is a number of minutes, and the minutes decide the architecture:

```python
# nines_capacity.py - what an availability target allows per 30 days, and the GPUs each design needs
import math
MIN_30D = 30 * 24 * 60
for a in (0.99, 0.999, 0.9999):
    print(f"{a:.2%}: {(1 - a) * MIN_30D:6.1f} min of full outage per 30 days "
          f"(or {(1 - a) * 1e6:5.0f} failed requests per million)")

peak_rps, per_replica_rps, gpus_per_replica = 120, 10, 8   # per-replica rate measured AT the SLO
n = math.ceil(peak_rps / per_replica_rps)
designs = [
    ("one site, N+1 replicas (survives a node)", n + 1),
    ("two sites, each carries full peak (survives a site)", 2 * (n + 1)),
    ("three regions, any two carry peak (survives a region)", 3 * math.ceil((n + 1) / 2)),
    ("four regions, any three carry peak", 4 * math.ceil((n + 1) / 3)),
]
print(f"\npeak {peak_rps} req/s, {per_replica_rps} req/s per {gpus_per_replica}-GPU replica -> N = {n}")
for name, reps in designs:
    print(f"  {name:54s} {reps:3d} replicas = {reps * gpus_per_replica:4d} GPUs "
          f"({reps / n:.2f}x the peak requirement)")
```

```console
$ python3 nines_capacity.py
99.00%:  432.0 min of full outage per 30 days (or 10000 failed requests per million)
99.90%:   43.2 min of full outage per 30 days (or  1000 failed requests per million)
99.99%:    4.3 min of full outage per 30 days (or   100 failed requests per million)

peak 120 req/s, 10 req/s per 8-GPU replica -> N = 12
  one site, N+1 replicas (survives a node)                13 replicas =  104 GPUs (1.08x the peak requirement)
  two sites, each carries full peak (survives a site)     26 replicas =  208 GPUs (2.17x the peak requirement)
  three regions, any two carry peak (survives a region)   21 replicas =  168 GPUs (1.75x the peak requirement)
  four regions, any three carry peak                      20 replicas =  160 GPUs (1.67x the peak requirement)
```

43 minutes a month (99.9%) is one bad deploy and its rollback, or one node failure handled by a human. 4.3 minutes (99.99%) is less than the time it takes most engines to load a large model, so a single replica restart in the critical path breaks it: at four nines, failover has to be automatic and the capacity to fail over to has to be running already.

Together's post lays out what each level has to survive ([Together, July 2026](https://www.together.ai/blog/99-9-uptime-for-inference)):

- **99%** must survive node failures ("GPU hardware faults, driver crashes, thermal events"): health checks, draining and fast replica replacement in one data center.
- **99.9%** must survive losing a data center: "model weights deployed across two facilities, enough capacity on each side to absorb the full load", with live traffic to both sites, not a cold standby.
- **99.99%** must survive a regional outage, which "typically calls for multi-region deployment with AZ redundancy and reserved failover capacity".

Together reports delivering 99.9% measured at completion, with a contractual SLA of 99% for single-data-center deployments and 99.9% for multi-data-center ones. The capacity table shows what that ladder costs. Surviving a site means owning the peak twice (2.17x). Spreading over more failure domains is cheaper, because each one only has to cover a fraction of a lost peer: with $k$ domains that must survive the loss of one, the overhead is $k/(k-1)$, so 1.75x with three and 1.67x with four. The catch is that every domain has to be big enough to be worth running.

**On a leased fleet.** This is the most expensive line in your plan, because spare capacity on a lease is paid every hour whether or not it's used. Two things soften it. First, the redundancy doesn't have to sit idle: run batch and best-effort work on the failover capacity (section 5) and preempt it when a site fails, so the "spare" GPUs earn money until they're needed. Second, don't promise more nines than your lease geography supports. A fleet in one data center can honestly offer 99%, maybe 99.5%. Selling 99.9% on it is selling an SLA you can't meet.

### Multi-region and failover

Fireworks published a seven-day production comparison: "request success rate went from 99.269% for a single-region deployment to 99.992% on a multi-region deployment" ([Fireworks, September 2026](https://fireworks.ai/blog/multi-region-deployment)). In minutes per 30 days, that is roughly the difference between 5 hours and 3.5 minutes of failed requests. The authors are careful about what it proves: "These are observational signals, not an A/B test or a universal uptime promise", "traffic was concentrated in a few accounts and paired-account outcomes were mixed", and the sensitivity checks "did not make it causal". Take it as evidence of direction, not as a number to promise. Their design is worth copying anyway: "Fireworks plans placement ahead of traffic, so requests do not need to consult the scheduler or probe every region", "DNS directs requests to healthy gateways based on proximity", and "On a qualifying failure, the gateway can try another destination if routing policy permits". That last one is the retry-before-first-byte rule from section 1 at the scale of regions.

What goes wrong with failover in practice:

- **The surviving region gets the whole load at once.** Its prefix caches are cold for the new traffic, its TTFT rises, and if it was sized only for its own peak, it overloads. Failover capacity must be sized and *warm*.
- **Retries from the failed region arrive as a herd.** Every client whose request failed retries at the same moment. Retry budgets and jitter at the client and the gateway turn that wave into a ramp.
- **Failback is a second failover.** Moving traffic back to the recovered region is the same cold-cache event in reverse. Shift it gradually.
- **Cross-region moves break cache affinity.** A conversation whose prefix is cached in one region pays full prefill in another. Route sessions stickily to a region and move them only on failure.

**On a leased fleet.** Multi-region means leases in several regions, which a small company may not have. The common compromise is a primary leased fleet with a fallback to a serverless provider of the same open model for overflow and disasters: you pay per token only when you use it, and the failover capacity is "reserved" in someone else's fleet. Test that the fallback works (same model revision, same chat template, same tokenizer) before you need it.


## 5. Batch and offline inference

### Batch APIs

A **batch API** accepts a file of many requests, processes them asynchronously within a completion window, and returns a file of results. The two largest providers price it the same way:

- OpenAI: "50% cost discount compared to synchronous APIs"; the completion window "can only be set to `24h`"; up to 50,000 requests and 200 MB per batch; unfinished requests are cancelled when a batch expires and "You will be charged for tokens consumed from any completed requests"; batch rate limits "are separate from existing per-model rate limits" ([batch guide](https://developers.openai.com/api/docs/guides/batch)).
- Anthropic: "reducing costs by 50 percent", "most batches completing within 1 hour", "Batches expire if processing does not complete within 24 hours", up to 100,000 requests or 256 MB per batch, results available for 29 days, and expired requests are not billed ([batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)). The 50% stacks with prompt-caching discounts ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)). Batches have their own rate limits, including a cap on requests waiting in the processing queue, and are not subject to ramp limits ([rate limits doc](https://platform.claude.com/docs/en/api/rate-limits)).

Why a provider can afford half price: a deadline in hours lets it run the work whenever capacity is free, at the large batch sizes that give the most tokens per GPU-second ([the throughput-latency Pareto](../inference/arithmetic.md#the-throughput-latency-pareto)), and stop it the moment interactive traffic needs the GPUs back. The interactive tier pays for the peak; batch buys the trough.

**On a leased fleet.** The product shape (a file in, a file out, a 24-hour window, half price, separate limits) is what customers already know from OpenAI and Anthropic, so copy it. What differs is your reason for offering it: those providers can also shift batch across a huge fleet and many regions; you have one fleet and its daily trough, which is exactly what the 24-hour window lets you use.

### Fireworks swapped its batch API for preemptible capacity

Fireworks' batch inference page now opens with "**Batch is paused.**": new jobs are not accepted, stored batch data "will be permanently deleted", and users are pointed to a guide on migrating batch workloads to preemptible deployments ([batch inference docs](https://docs.fireworks.ai/guides/batch-inference)). While it ran, it was priced at 50% off serverless. Preemptible deployments are also 50% off the on-demand GPU rate, can be taken back at any time, and can lose a replica mid-request ([on-demand deployments doc](https://docs.fireworks.ai/guides/ondemand-deployments)). Fireworks' pricing page still lists "50% off input and output for jobs that don't need an immediate answer" for batch ([pricing](https://fireworks.ai/pricing)), so the two pages disagree as of November 2026.

The shift is from selling *tokens later* to selling *GPUs that may vanish*. For the provider, the second is simpler: no job queue, no per-token accounting of partial work, no deadline promise. For the customer, it is more work: they run their own engine and their own retry logic.

**On a leased fleet.** Both models work on a fixed fleet, and they are two ways of selling the same idle capacity. A batch API keeps the scheduling and the deadline promise on your side, which is more engineering and more control. Selling preemptible replicas pushes that to the customer, and is simpler if your customers are capable of running engines. Either way, the economics are better for you than for a provider with elastic capacity: the GPUs in your trough are already paid for, so batch revenue at a discount is almost pure margin, as long as it never delays interactive traffic.

### Filling idle capacity with a deadline queue

A **deadline queue** holds batch jobs, each with a submission time, a deadline and an amount of work, and spends whatever capacity interactive traffic leaves on them. The scheduling rule that matters is **earliest deadline first** (EDF): among queued jobs, serve the one due soonest. On a single resource, EDF is optimal in a precise sense: if any order meets every deadline, EDF does.

The second rule is **admission control**: at submission time, check whether the forecast spare capacity before the job's deadline covers this job plus everything already queued that is due no later. If not, reject the job *now*. A customer who learns at submission that a job won't finish can split it or pay for priority. A customer who learns 24 hours later, from an expired batch, has lost a day.

The simulation runs two days of a fleet with 1,000 units of capacity per minute, interactive load that swings daily between 300 and 850, and a 10% reserve kept free for interactive bursts. Batch jobs arrive at random, 80% with a 24-hour deadline and 20% with a 1-hour deadline (a "fast batch" product), at two load levels:

```python
# batch_queue.py - fill a fixed fleet's idle capacity with deadline-bound batch work.
# Units: fleet capacity is 1000 "units"/minute (say 1 unit = 1k output tokens at the batch size you run).
# Interactive traffic always goes first; batch gets what is left, minus a burst reserve.
import math, random

C, RESERVE = 1000, 0.10                    # capacity per minute; keep 10% free for interactive bursts
DAY = 24 * 60

def interactive(m):                        # diurnal load: 300 at night, 850 at the daily peak
    return 575 + 275 * math.sin(2 * math.pi * (m % DAY) / DAY - math.pi / 2)

def spare(m):
    return max(0.0, C * (1 - RESERVE) - interactive(m))

def jobs(gap, seed=1):
    rng, out, m = random.Random(seed), [], 0.0
    while m < 2 * DAY:
        m += rng.expovariate(1 / gap)      # a job every `gap` minutes on average
        window = 60 if rng.random() < 0.2 else DAY     # 20% "1-hour" jobs, 80% "24-hour" jobs
        size = rng.lognormvariate(math.log(2000), 0.8) if window == DAY else rng.lognormvariate(math.log(300), 0.5)
        out.append(dict(sub=int(m), deadline=int(m) + window, size=size, left=size))
    return out

def run(order, admission, gap):
    js, queue, done_units, expired_units, rejected_units, util = jobs(gap), [], 0.0, 0.0, 0.0, []
    nxt = 0
    for m in range(2 * DAY + DAY):         # run a third day with no new jobs, so every deadline passes
        while nxt < len(js) and js[nxt]["sub"] <= m:
            j = js[nxt]; nxt += 1
            if admission:
                # admit only if the spare capacity forecast before the deadline covers this job plus
                # the work already queued that is due no later (the EDF feasibility test)
                ahead = sum(q["left"] for q in queue if q["deadline"] <= j["deadline"])
                cap = sum(spare(k) for k in range(m, j["deadline"]))
                if ahead + j["size"] > cap:
                    rejected_units += j["size"]; continue     # tell the customer now, not in 24 h
            queue.append(j)
        for j in [j for j in queue if j["deadline"] <= m]:     # expiry: unfinished part is not billed
            expired_units += j["left"]; queue.remove(j)
        queue.sort(key=(lambda j: j["deadline"]) if order == "edf" else (lambda j: j["sub"]))
        s, used = spare(m), 0.0
        for j in queue:
            take = min(j["left"], s - used)
            j["left"] -= take; used += take; done_units += take
            if used >= s: break
        queue = [j for j in queue if j["left"] > 1e-9]
        if m < 2 * DAY: util.append((interactive(m) + used) / C)
    total = sum(j["size"] for j in js)
    short = [j for j in js if j["deadline"] - j["sub"] == 60]
    short_ok = sum(j["left"] <= 1e-9 for j in short) / len(short)     # rejected ones keep left > 0
    return done_units / total, expired_units / total, rejected_units / total, short_ok, sum(util) / len(util)

base = sum(interactive(m) for m in range(2 * DAY)) / (2 * DAY) / C
print(f"interactive only: fleet utilization {base:.1%}, peak {max(interactive(m) for m in range(DAY)) / C:.0%}")
for gap in (6, 4):
    print(f"\nbatch offered: one job every {gap} min ({sum(j['size'] for j in jobs(gap)) / (2 * DAY) / C:.0%} of capacity)")
    print(f"{'policy':26s} {'units done':>10s} {'expired':>8s} {'rejected':>9s} {'1h jobs on time':>16s} {'fleet util':>11s}")
    for name, order, adm in [("FIFO", "fifo", False), ("EDF", "edf", False),
                             ("EDF + admission control", "edf", True)]:
        d, e, r, s1, u = run(order, adm, gap)
        print(f"{name:26s} {d:10.1%} {e:8.1%} {r:9.1%} {s1:16.1%} {u:11.1%}")
```

```console
$ python3 batch_queue.py
interactive only: fleet utilization 57.5%, peak 85%

batch offered: one job every 6 min (36% of capacity)
policy                     units done  expired  rejected  1h jobs on time  fleet util
FIFO                            97.6%     2.4%      0.0%            14.0%       87.3%
EDF                            100.0%     0.0%      0.0%           100.0%       87.3%
EDF + admission control        100.0%     0.0%      0.0%           100.0%       87.3%

batch offered: one job every 4 min (55% of capacity)
policy                     units done  expired  rejected  1h jobs on time  fleet util
FIFO                            84.4%    15.6%      0.0%            13.2%       89.0%
EDF                             84.4%    15.6%      0.0%           100.0%       89.0%
EDF + admission control         83.9%     0.6%     15.6%           100.0%       89.0%
```

Reading it:

- **Batch takes fleet utilization from 57.5% to 87-89%.** Nothing else in this chapter moves cost per token as much. Section 6 shows why: cost per token is inversely proportional to utilization.
- **At moderate load, EDF meets every deadline and FIFO doesn't.** FIFO let 86% of the 1-hour jobs miss, because they queued behind large 24-hour jobs that had hours of slack. EDF ran them first and still finished everything else in time.
- **At overload, the order can't save you, but admission control changes who finds out and when.** Without it, 15.6% of the work expires after waiting up to a day. With it, the same amount is rejected at submission and only 0.6% expires. The forecast here is perfect (the interactive load is a known curve). Yours won't be, so admit against a conservative forecast, for example last week's same-hour load plus a margin, and expect some expiries.

What a real batch system adds around this core:

- **Requests, not jobs, are the unit of work.** A batch file is thousands of independent requests. Schedule requests across replicas, bill each completed one, and on expiry cancel only the unfinished ones (both OpenAI and Anthropic bill completed requests and not expired ones).
- **Run batch at the throughput-optimal configuration.** Batch requests don't care about TPOT, so they can run at much larger batch sizes. If you can, put batch on replicas configured for it, or let the batch class fill the slots that the interactive SLO leaves free on mixed replicas.
- **Give batch its own rate limits** and its own queue cap, as both providers do, so a huge batch submission can't touch interactive admission.
- **Preempt, don't fail.** When interactive load rises, batch requests that are preempted go back to the queue (section 3).
- **Store results durably** before reporting completion, and keep them only as long as your retention policy says (section 9).

**On a leased fleet.** Of all the techniques in this chapter, this one is closest to free money. The two published 50% discounts show what customers expect to pay. Your marginal cost for batch run in the trough is close to zero, because the GPU-hours are sunk; the only real costs are the engineering and the power above idle. So the discount you can offer is set by what the market pays for batch, not by your costs, and every batch token is margin.


## 6. Pricing and unit economics

### From GPU-hour to dollars per million tokens

Cost per token comes from three numbers: what a GPU-hour costs, how many tokens a replica produces per second while busy, and what fraction of the time it is busy. For a replica of $G$ GPUs leased at $p$ dollars per GPU-hour, with throughput $X$ tokens/s while busy and utilization $u$ (the fraction of leased time spent on paid work):

$$
\text{cost per 1M tokens} = \frac{G \cdot p}{3600 \cdot X \cdot u} \cdot 10^6
$$

This is the conversion from [Inference Internals I](../inference/arithmetic.md#decode-latency-lower-bound) with $u$ added, and $u$ is the term operators control. Halving utilization doubles cost per token, whatever the engine does.

Input and output tokens go through different phases. Prefill processes a prompt's tokens in parallel and is compute-bound; decode produces one token per sequence per step and is memory-bound ([prefill and decode](../inference/arithmetic.md#prefill-and-decode)). So a replica has two throughputs: $X_p$ input tokens/s when prefilling and $X_d$ output tokens/s when decoding at the batch size the TPOT SLO allows. Charging each token the GPU-time it takes gives

$$
\text{cost of a request} = \frac{G \cdot p}{3600 \cdot u} \left( \frac{n_{\text{in}} - n_{\text{cached}}}{X_p} + \frac{n_{\text{out}}}{X_d} \right)
$$

which treats prefill and decode time as additive. That is exact for disaggregated serving and a good approximation for chunked prefill.

The calculator below uses one replica of a 70B dense model on 8 H100s, with FP8 weights and KV cache. The inputs are assumptions, and the comments say where each comes from: $X_p$ from the H100's measured FP8 matmul ceiling ([MAMF table](../compute/accelerators.md#mamf-and-msmf-comparison-table)) at 50% efficiency and $2N$ FLOPs per token; $X_d$ as 60% of the roofline bound for batch 64 at 4k context in [Inference Internals I's table](../inference/arithmetic.md#decode-latency-lower-bound); and \$2.00 per GPU-hour, the rate that chapter uses. Put your measured numbers in instead; [Serving LLMs in Practice](../inference/serving-hands-on.md#capacity-planning-from-measured-numbers) shows how to measure per-replica throughput at an SLO.

```python
# cost_calc.py - $/1M tokens from GPU price, throughput and utilization; break-even and margin.
# Every input below is an ASSUMPTION for one replica of a 70B dense model, FP8 weights and KV,
# on 8x H100 (see the text for where each comes from). Replace them with your own measurements.
GPU_HR      = 2.00        # $/GPU-hour on the lease
GPUS        = 8           # GPUs per replica
X_PREFILL   = 43_900      # input tok/s per replica: 8 x 1535 TFLOP/s (FP8 MAMF) x 50% / (2 x 70e9)
X_DECODE    = 4_430       # output tok/s per replica at the TPOT SLO: 60% of the 7,384 tok/s roofline bound
KV_BYTES    = 163_840     # KV bytes per token, 70B (80 layers, 8 KV heads, d=128) with FP8 KV
HBM_GB      = 80          # per GPU

def per_m(x_tok_s, util):
    """$ per 1M tokens of a phase that runs at x_tok_s when busy, if the fleet is busy `util` of the time."""
    return GPUS * GPU_HR / 3600 / x_tok_s / util * 1e6

print("cost per 1M tokens (what it costs YOU, before margin):")
print(f"{'utilization':>11s} {'input':>8s} {'output':>8s} {'out/in':>7s}")
for u in (0.3, 0.5, 0.7, 0.9):
    i, o = per_m(X_PREFILL, u), per_m(X_DECODE, u)
    print(f"{u:11.0%} {i:8.3f} {o:8.3f} {o / i:7.1f}")

# --- a request's cost is GPU-seconds: input at prefill speed, output at decode speed
def request_cost(n_in, n_out, util, cached=0):
    return (n_in - cached) * per_m(X_PREFILL, util) / 1e6 + n_out * per_m(X_DECODE, util) / 1e6
print(f"\none chat turn, 3,000 in / 400 out at 60% util: ${request_cost(3000, 400, 0.6):.5f}  "
      f"(output is {400 * per_m(X_DECODE, .6) / (3000 * per_m(X_PREFILL, .6) + 400 * per_m(X_DECODE, .6)):.0%} of it)")

# --- prompt caching: a hit skips prefill compute but the KV had to be HELD somewhere meanwhile
hbm_gb_hr = GPU_HR / HBM_GB                       # attribute the whole GPU price to its HBM: upper bound
gb_per_mtok = KV_BYTES * 1e6 / 1e9
hold_5min = gb_per_mtok * hbm_gb_hr * 5 / 60
recompute = per_m(X_PREFILL, 0.6)
print(f"\nKV for 1M tokens = {gb_per_mtok:.0f} GB. Holding it in HBM for 5 min: ${hold_5min:.3f}; "
      f"recomputing it (prefill at 60% util): ${recompute:.3f}")
print(f"-> HBM residency breaks even with recompute after {recompute / (hold_5min / 5):.1f} min unless it is reused")
print(f"-> reloading it from host DRAM over PCIe 5 x16 (~55 GB/s measured) takes {gb_per_mtok / 55:.1f} s "
      f"vs {1e6 / X_PREFILL:.1f} s to recompute")

# --- break-even utilization and margin sensitivity for a price list
P_IN, P_OUT = 0.60, 1.80                          # YOUR list price, $/1M tokens (a choice, not a market quote)
MIX = 7.5                                         # input tokens per output token in your traffic
def blended_cost(u): return (MIX * per_m(X_PREFILL, u) + per_m(X_DECODE, u)) / (MIX + 1)
blended_price = (MIX * P_IN + P_OUT) / (MIX + 1)
u_star = blended_cost(1.0) / blended_price
print(f"\nprice ${P_IN}/${P_OUT} per 1M in/out, mix {MIX}:1 -> blended ${blended_price:.3f}/1M tokens; "
      f"break-even utilization {u_star:.0%}")
print(f"{'util':>5s} " + " ".join(f"{'x' + str(m):>7s}" for m in (0.6, 0.8, 1.0, 1.2)) + "   (gross margin at price multiplier)")
for u in (0.2, 0.3, 0.4, 0.6, 0.8):
    row = [1 - blended_cost(u) / (blended_price * m) for m in (0.6, 0.8, 1.0, 1.2)]
    print(f"{u:5.0%} " + " ".join(f"{x:7.0%}" for x in row))

# --- dedicated (per GPU-hour) vs per-token, from the customer's side
DED_GPU_HR = 4.00                                 # what you charge for a dedicated GPU-hour
ded_hr = DED_GPU_HR * GPUS
tok_hr_full = 3600 / (MIX / X_PREFILL + 1 / X_DECODE) * (MIX + 1)   # tokens/h one replica can do at 100%
breakeven_tok = ded_hr / blended_price * 1e6
print(f"\ndedicated replica ${ded_hr:.0f}/h vs per-token at ${blended_price:.3f}/1M: customer breaks even at "
      f"{breakeven_tok / 1e6:.0f}M tokens/h = {breakeven_tok / tok_hr_full:.0%} of what the replica can do")
assert abs(blended_cost(u_star) - blended_price) < 1e-9
```

```console
$ python3 cost_calc.py
cost per 1M tokens (what it costs YOU, before margin):
utilization    input   output  out/in
        30%    0.337    3.344     9.9
        50%    0.202    2.007     9.9
        70%    0.145    1.433     9.9
        90%    0.112    1.115     9.9

one chat turn, 3,000 in / 400 out at 60% util: $0.00118  (output is 57% of it)

KV for 1M tokens = 164 GB. Holding it in HBM for 5 min: $0.341; recomputing it (prefill at 60% util): $0.169
-> HBM residency breaks even with recompute after 2.5 min unless it is reused
-> reloading it from host DRAM over PCIe 5 x16 (~55 GB/s measured) takes 3.0 s vs 22.8 s to recompute

price $0.6/$1.8 per 1M in/out, mix 7.5:1 -> blended $0.741/1M tokens; break-even utilization 28%
 util    x0.6    x0.8    x1.0    x1.2   (gross margin at price multiplier)
  20%   -133%    -75%    -40%    -17%
  30%    -55%    -17%      7%     22%
  40%    -17%     13%     30%     42%
  60%     22%     42%     53%     61%
  80%     42%     56%     65%     71%

dedicated replica $32/h vs per-token at $0.741/1M: customer breaks even at 43M tokens/h = 56% of what the replica can do
```

### Why input is cheaper than output

With these numbers, an output token costs about 10x an input token, because $X_p / X_d \approx 10$: prefill runs at the compute roofline and decode at the bandwidth roofline. Public prices have the same shape with a smaller ratio. As of November 2026, Anthropic lists Claude Sonnet 4.6 at \$3 input and \$15 output per million tokens ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)), and Baseten lists GPT OSS 120B at \$0.10 / \$0.50 and DeepSeek V4.1 Flash at \$0.30 / \$1.20 ([pricing](https://www.baseten.co/pricing/)): ratios of 5x, 5x and 4x. The VTC paper's $w_q = 2$ followed OpenAI's ratio at the time ([arXiv 2401.00588](https://arxiv.org/abs/2401.00588)).

Why price ratios are lower than raw compute ratios:

- **Input tokens also cost during decode.** Every input token sits in the KV cache for the whole generation and is read at every decode step. Long prompts shrink the batch size the KV cache allows and slow each step ([KV cache sizing](../inference/arithmetic.md#kv-cache-sizing)), which lowers $X_d$ for everyone in the batch. A long prompt's real cost is mostly KV-seconds, not prefill FLOPs.
- **Agentic traffic is input-heavy.** Ratios of input to output tokens of 10:1 or more are common for agents that resend a long history every turn. In the calculator's chat turn, 3,000 in and 400 out, output is still 57% of the cost; with a 30,000-token history it would be a small fraction. Pricing input near cost would make the most common workloads unprofitable.
- **Prices are set by the market as well as cost.** Anyone can serve an open model, so the floor is set by the cheapest competent competitor.

The practical consequence: price per token class (input, cached input, output) from *your* measured costs, then check them against market prices. If your output price is far below your output cost and input far above, your margins depend on your traffic mix staying the same.

### Prompt-caching discounts and their cost basis

A **prompt cache** keeps the KV of a prompt prefix so a later request with the same prefix skips its prefill ([prefix caching](../inference/serving-engine-internals.md#hash-based-block-caching-vllm-apc)). Providers pass part of the saving on:

- Anthropic: a 5-minute cache write costs 1.25x the base input price, a 1-hour write 2x, and a cache read 0.1x on most models (0.05x on Opus 5.5 and Sonnet 5.5). Caching pays off "after one cache read" for the 5-minute cache and "after two cache reads" for the 1-hour cache, and the multipliers stack with the batch discount ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)).
- OpenAI caches automatically from 1,024 tokens (for GPT-5.6 and later), discounts cached input "up to 95%", and does not share caches across organizations. Its docs also reveal how routing works: a `prompt_cache_key` influences routing, and "traffic above 15 requests per minute can lead to overflow routing", because cached state lives on individual machines ([prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)).
- Fireworks bills cached input at 50% off by default ([pricing](https://fireworks.ai/pricing)); Baseten lists DeepSeek V4.1 Flash cached input at \$0.007 against \$0.30 uncached ([pricing](https://www.baseten.co/pricing/)).

The cost basis has two sides, and the calculator prints both. A hit saves the prefill (\$0.169 per million tokens at 60% utilization in this example). But the KV had to be *held* somewhere between requests. Held in HBM, the memory that also bounds your batch size, 1M tokens of a 70B model's FP8 KV is 164 GB, and attributing the GPU's whole price to its 80 GB of HBM, keeping it for 5 minutes costs \$0.341: twice the cost of recomputing it. Unless that prefix is read again within about 2.5 minutes, HBM residency loses money. Host DRAM changes the answer: reloading those 164 GB over PCIe 5 at the ~55 GB/s [measured in the interconnects chapter](../network/interconnects.md) takes 3 s, versus 23 s to recompute.

So the economics of a prompt cache are a tiering problem. Hot prefixes (system prompts shared by thousands of requests per minute) belong in HBM. A conversation that might come back in a few minutes belongs in host memory or on local NVMe. The pricing follows: the write premium (1.25x, 2x) pays for holding the KV, longer retention costs more, and the deep read discounts are possible because the read is a memory copy, not a prefill.

**On a leased fleet.** Host DRAM and local NVMe on leased GPU nodes are already paid for and mostly idle, so the marginal cost of a host-memory KV tier is close to zero until it fills. That makes deep cached-input discounts cheaper for you than the HBM arithmetic suggests, as long as the router sends returning conversations to the node that holds their KV ([cache-aware routing](../inference/serving-engine-internals.md#cache-aware-routing-across-replicas)). The cache hit rate is a revenue number: track it per tenant.

### Dedicated vs per-token pricing

**Per-token** pricing puts utilization risk on you: idle time is your loss. **Dedicated** pricing, per GPU-hour, puts it on the customer. Published on-demand H100 prices, read in November 2026, give the range customers compare you against: Modal at \$0.001097 per second (\$3.95 per hour) ([pricing](https://modal.com/pricing)), Baseten at \$0.10833 per minute (\$6.50 per hour) ([pricing](https://www.baseten.co/pricing/)), and Fireworks at \$8.00 per hour ([pricing](https://fireworks.ai/pricing)). These are per-second or per-minute billing on capacity the vendor can reassign. A one-year lease costs much less per hour than any of them, and the difference is what pays for your idle time.

From the customer's side, the calculator's last line is the break-even: a dedicated replica at \$4.00 per GPU-hour costs the same as per-token at the example prices once the customer uses 56% of what the replica can do. Customers above that should go dedicated; customers below it are better off per token, and you are better off with them per token too, since you can pool their idle time with others'.

### Break-even utilization and margin sensitivity

The margin table is the one to keep on the wall. At the example prices, the fleet loses money below 28% utilization, makes 30% gross margin at 40% and 53% at 60%. Utilization moves margin more than any plausible change in price: going from 40% to 60% utilization is worth more than a 20% price increase. That is the economic argument for everything in sections 3 and 5: fair scheduling that keeps the batch full, priority classes, and batch work in the trough.

**On a leased fleet.** The lease converts GPU cost into a fixed cost, so the cost per token falls as $1/u$ for as long as demand lasts. A serverless provider can't get below its per-second GPU price however busy it is. You can, which is the reason to lease, and the reason you lose money in a month when demand drops. Track utilization weekly, as revenue.


## 7. Observability and quality

### The per-request log

One line per request, written when the request ends (completion, error, cancel or timeout), is the source for SLOs, billing, debugging and abuse detection. `make_log.py` wrote a minimal one. A production record adds the fields you will want during an incident:

| Field | Why |
|---|---|
| `request_id`, `trace_id` | join the log to traces and to the customer's support ticket |
| `org`, `project`, `key_id` (never the key), `tier`, `model`, `adapter` | per-tenant SLOs, billing, abuse |
| `replica`, `node`, `gpu_ids`, `engine_version`, `model_revision` | find the bad replica, the bad GPU, the bad deploy |
| `ts_arrival`, `ts_admitted`, `ts_first_token`, `ts_done` | queue time, TTFT, E2E, all computed, never estimated |
| `n_in`, `n_cached`, `n_out`, `max_tokens` | billing and limit settlement, from the engine's `usage` |
| `finish_reason`, `http_status`, `error_code` | completion-based availability (section 4) |
| `retry_of`, `idempotency_key` | separating retries from new traffic |

Billing and SLOs must come from the same record. If billing reads one pipeline and reliability another, they will disagree and you will not know which is right. Prompt and response *content* does not go in this log by default (section 9).

### Metrics

Prometheus-style metrics give you the same information as the log aggregated in real time. The engine metrics, and how to scrape them, are in [Serving LLMs in Practice](../inference/serving-hands-on.md#monitoring-engine-metrics) and [the engine metrics table](../inference/serving-engine-internals.md#observability). What the operator layer adds:

- **Queue depth and the age of the oldest waiting request**, per class, at the gateway queue as well as the engines. Queue age shows a stuck queue that queue length hides.
- **KV usage with the preemption rate.** Usage near 100% alone is fine; near 100% with preemptions rising is oversubscription (section 8).
- **TTFT and TPOT percentiles per tier and per tenant.** Per-tenant labels are a cardinality problem: thousands of tenants times histogram buckets times replicas will overload Prometheus ([label cardinality](control-planes.md#metrics-with-prometheus)). Label the top tenants by traffic explicitly, put everyone else in `other`, and get per-tenant detail for the rest from the request log.
- **Goodput and budget burn per tier**, computed from the log stream.
- **Limit rejections by reason** (`key_tpm`, `org_rpm`, `global`): a rise in `global` means your capacity is short; a rise in one org's limits is that customer's problem.

Baseten's metrics docs carry one caveat worth repeating: concurrency and queue-size panels are sampled gauges, so short spikes between samples can be missed ([metrics docs](https://docs.baseten.co/observability/metrics)). Modal's GPU-health post gives a second: their metrics are "aggregated at the container level, so they are less effective at spotting a single bad GPU" among eight ([Modal, December 2025](https://modal.com/blog/gpu-health)). Per-GPU signals need per-GPU export.

### Tracing through gateway, router and engine

A trace follows one request across processes as a tree of timed spans ([tracing](control-planes.md#tracing)). For an inference request the spans are: gateway (auth, limits, queue wait), router (replica choice and why: prefix match length, load), engine (waiting, prefill, decode), and the return path. Pass the W3C `traceparent` header from the gateway to the router to the engine's HTTP server, and record the router's decision as span attributes. Engine-internal span support differs between engine versions; check yours. Sample traces (1-10%, plus every slow or failed request), and keep the `trace_id` in the request log so you can jump from a bad log line to its trace.

### Silent quality regressions

The incidents that hurt most return HTTP 200. In September 2025 Anthropic published a postmortem of three bugs that degraded Claude's output quality over August and early September ([Anthropic, September 2025](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)). It opens with "We never reduce model quality due to demand, time of day, or server load", and the problems were infrastructure bugs:

1. **A routing bug.** From August 5, some Sonnet 4 requests were sent to servers configured for an upcoming 1M-token context window, at first 0.8% of them. A routine load-balancing change on August 29 increased the share; at the worst hour, on August 31, it was 16%. About 30% of Claude Code users had at least one message misrouted. **Sticky routing** made it worse: "once a request was served by the incorrect server, subsequent follow-ups were likely to be served by the same incorrect server."
2. **Output corruption.** A misconfiguration deployed to TPU servers on August 25 caused errors during token generation, such as "producing Thai or Chinese characters in response to English prompts". It affected Opus 4.1 and Opus 4 until August 28 and Sonnet 4 until the rollback on September 2.
3. **A compiler bug in approximate top-k.** Next-token probabilities were computed in bf16, but the XLA compiler for TPUs could run some operations in fp32 under a flag (`xla_allow_excess_precision`) that defaults to true, so different operations disagreed on which token was most likely. The fix was to switch from approximate to exact top-k and standardize some operations on fp32: "Model quality is non-negotiable, so we accepted the minor efficiency impact."

Why it took weeks to find: the evaluations didn't capture the degradation, partly because "Claude often recovers well from isolated mistakes"; privacy controls limit engineers' access to user conversations; and "Each bug produced different symptoms on different platforms at different rates". Among the remediations: running quality evaluations "continuously on true production systems".

The lessons for any operator:

- **Routing is a quality change** when replica pools differ (context length, quantization, engine version, hardware). Treat a load-balancer change like a model change.
- **Stickiness concentrates harm.** Session affinity, which you want for prefix caching, sends the same users back to the same bad replica.
- **Sampling and kernel "optimizations" need numerical tests** against a reference implementation before deploy, comparing next-token distributions, not just benchmark scores.
- **Canary gates on latency and errors can't see any of this.** Together's canary rollouts gate on three router metrics, error rate, latency and in-flight requests ([Together, September 2026](https://together.ai/blog/canary-rollouts-upgrade-models-in-production-without-downtime)); a regression in text quality passes all three. Rollouts themselves are covered in [Deploys, Recovery and Autoscaling](deploys-recovery-autoscaling.md). What this chapter adds is a quality signal for them to gate on.

### Continuous evals and canary prompts in production

Two kinds of quality signal can run all the time, per replica:

- **Canary prompts**: a fixed set of prompts with checkable answers (arithmetic, exact-match extraction, a JSON schema, a unit test for generated code), sent to every replica on a schedule at temperature 0. They catch a replica that has become worse at the task. They are expensive per request and few in number, so they are statistically weak.
- **Passive checks on live traffic**: cheap properties computed on every response without storing it, such as the share of English prompts whose reply contains characters from another script (the postmortem's symptom), the share of JSON-mode replies that fail to parse, the share hitting `max_tokens`, repetition rate, and the mean log-probability of sampled tokens where the engine returns it. They have huge sample sizes and only see what you thought to measure.

Compare each replica against the others, not against a fixed threshold: a bad replica stands out from its peers, while traffic changes move all replicas together. The simulation has 8 replicas; replica `r5` picks up a sampling bug at minute 30 that drops its canary pass rate from 97% to 90% and makes 2% of its live replies contain unexpected scripts:

```python
# quality_probe.py - catch a replica that returns HTTP 200 with worse text.
# Two signals per replica: (1) canary prompts with checkable answers, (2) a character-script
# anomaly rate on live English traffic (the symptom in Anthropic's Aug 2025 postmortem).
import math, random, unicodedata

def scripts(text):
    """Set of Unicode scripts in a string, from character names ('THAI CHARACTER ...' -> 'THAI')."""
    out = set()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "UNKNOWN")
            out.add("LATIN" if name.startswith("LATIN") else name.split()[0])
    return out

def anomalous(prompt, reply):
    """English prompt, reply contains a script the prompt does not: suspicious."""
    return bool(scripts(reply) - scripts(prompt) - {"LATIN"})

assert not anomalous("Say hello", "Hello there!")
assert anomalous("Say hello", "Hello สวัสดี there!")
assert anomalous("Fix this Python", "def f(x): return x  # 修复")

def z_lower(k, n, k0, n0):
    """Two-proportion z: is this replica's pass rate k/n below everyone else's k0/n0?"""
    p = (k + k0) / (n + n0)
    se = math.sqrt(p * (1 - p) * (1 / n + 1 / n0)) or 1e-9
    return (k / n - k0 / n0) / se

rng = random.Random(3)
REPLICAS = [f"r{i}" for i in range(8)]
BAD, START = "r5", 30                       # r5 picks up a sampling bug at minute 30
PASS_OK, PASS_BAD = 0.97, 0.90              # canary pass rates
ANOM_OK, ANOM_BAD = 0.0005, 0.02            # share of live replies with an unexpected script
PROBES, LIVE = 10, 400                      # per replica per minute
stats = {r: dict(k=0, n=0, a=0, m=0) for r in REPLICAS}
fired = {}
for minute in range(120):
    for r in REPLICAS:
        bad = r == BAD and minute >= START
        s = stats[r]
        s["k"] += sum(rng.random() < (PASS_BAD if bad else PASS_OK) for _ in range(PROBES)); s["n"] += PROBES
        s["a"] += sum(rng.random() < (ANOM_BAD if bad else ANOM_OK) for _ in range(LIVE)); s["m"] += LIVE
    if minute % 5 == 4:                      # every 5 min, compare each replica with the rest (last 30 min
        for r in REPLICAS:                   # would be better; cumulative keeps the code short)
            o = [stats[x] for x in REPLICAS if x != r]
            zc = z_lower(stats[r]["k"], stats[r]["n"], sum(x["k"] for x in o), sum(x["n"] for x in o))
            za = -z_lower(stats[r]["a"], stats[r]["m"], sum(x["a"] for x in o), sum(x["m"] for x in o))
            for sig, z in (("canary", zc), ("script", za)):
                if z < -4 and (r, sig) not in fired:       # |z| > 4: few false alarms across 8 replicas x 2 signals
                    fired[(r, sig)] = minute
                    print(f"minute {minute:3d}: {r} flagged by {sig:6s} z={z:6.1f}  "
                          f"({minute - START + 1} min after the bug, if {r} is the bad one)")
print("false alarms:", [k for k in fired if k[0] != BAD])
for sig in ("canary", "script"):
    assert (BAD, sig) in fired
```

```console
$ python3 quality_probe.py
minute  34: r5 flagged by script z=  -9.8  (5 min after the bug, if r5 is the bad one)
minute  69: r5 flagged by canary z=  -4.8  (40 min after the bug, if r5 is the bad one)
false alarms: []
```

The passive check found the bad replica in 5 minutes; the canary prompts needed 40, with 10 probes per replica per minute. Detecting a drop from 97% to 90% takes a few hundred samples, and a cumulative count dilutes a new problem with the clean history (a sliding window would be faster). Run both. The passive checks catch the known symptoms quickly on all traffic. The canaries catch the regressions you didn't think to write a detector for. A slower, broader eval (a held-out benchmark run against one replica of each pool every few hours) catches what's left, and its results should be compared with the same eval run against the reference deployment at rollout time.

### Numerics differences across hardware

The same model and weights give different outputs on different hardware and software, for two reasons. **Batch dependence**: kernels pick reduction orders from the batch shape, so the same request gives different floating-point results depending on other users' traffic ([batch invariance](../inference/serving-engine-internals.md#determinism-and-batch-invariance)). **Implementation differences**: different GPU generations, TP degrees, quantization formats, attention kernels and, as the postmortem shows, compiler precision flags.

Small differences are normal; the job is to tell them apart from bugs. Measure them once per pool against a reference: run a fixed prompt set through both with teacher forcing (feed the same tokens, compare the next-token distributions), and record the KL divergence and top-1 agreement per pool. That gives you a baseline, and a new deploy whose divergence is several times the baseline is a bug until shown otherwise. Fireworks evaluated its FP4 kernels with MMLU-style accuracy instead of KL because not every engine returns log-probabilities ([FireAttention V4](https://fireworks.ai/blog/fireattention-v4-fp4-b200)); if your engine does return them, the distribution comparison is far more sensitive.

**On a leased fleet.** A fixed fleet is usually more uniform than a multi-cloud one, which makes per-replica comparison easier: replicas should agree closely, so outliers stand out. The exceptions are the moments you mix hardware (a new node type joining) or software (an engine upgrade rolled out halfway). Those are the times to run the full eval against both pools.


## 8. Incidents and on-call

The general practice (mitigate first, then understand, then fix; runbook per page; blameless postmortems) is in [Control Planes](control-planes.md#on-call-and-incident-debugging), and [the failure-mode table](../inference/serving-hands-on.md#common-failure-modes) in Serving LLMs in Practice covers startup and configuration failures. These are the incidents that page an inference operator once the service is running. Each runbook entry is: symptom, how to confirm, how to mitigate, how to fix for good.

| Incident | Symptom | Confirm | Mitigate | Fix |
|---|---|---|---|---|
| **KV exhaustion** | ITL p99 and TTFT rise together, throughput falls | KV usage near 100% *and* preemptions rising ([admission by KV budget](../inference/serving-engine-internals.md#admission-control-by-kv-budget)); look for one tenant's longer prompts or larger `max_tokens` | lower the per-replica in-flight cap at the gateway so requests wait in the fair queue instead of thrashing; cap that tenant's in-flight tokens | per-tenant in-flight token limits; long-prompt replicas; FP8 KV; capacity |
| **Stuck engine** | a replica holds requests and produces nothing; `/health` stays green; its TTFT hits the first-byte timeout | running > 0 and the generated-token counter flat for a minute | eject it from the router; capture `py-spy dump` and `nvidia-smi` *before* restarting, since the evidence dies with the process | from the dump (NCCL hang, scheduler deadlock, hung GPU); add "running but not producing" to router health |
| **Bad GPU** | one replica or node with more errors, slower steps or worse quality than its peers; Xid errors, rising ECC counters | per-replica breakdowns (sections 4 and 7) | drain and cordon the *whole host* | provider replaces the part |
| **OOM on long prompts** | crashes or errors only on some requests, often after a customer starts sending longer documents or many images | failing requests' `n_in` in the log; engine error output | lower the accepted input length at the gateway with a clear 400; route long prompts to replicas with smaller prefill chunks | test the advertised maximum context end to end, at full batch, before advertising it |
| **Thundering herd after a deploy** | errors and latency spike as a rollout finishes or the gateway restarts | request rate above organic traffic; `retry_of` set in the log; cold new replicas | retry budgets; `Retry-After` with jitter; shift traffic gradually | warm-up and rollout pacing ([Deploys, Recovery and Autoscaling](deploys-recovery-autoscaling.md)); SDKs that honor `Retry-After` |
| **Cache-cold routing** | TTFT rises everywhere with no traffic change | prefix hit rate per replica before and after; changed replica membership remaps a hash ring ([consistent hashing](../foundations/distributed-systems.md#consistent-hashing-sharding-that-survives-membership-changes)) | roll back the router change; caches rewarm in minutes if routing is stable | bounded-load prefix affinity ([Serving Efficiently](../inference/serving-efficiently.md)); change membership a few replicas at a time |

Two of these have published field experience behind them. On bad GPUs, Together names ECC errors "the most common thing we see; they corrupt weights silently" ([Together, July 2026](https://www.together.ai/blog/99-9-uptime-for-inference)). Modal, running over 20,000 GPUs, marks the whole host unhealthy, drains it and disposes of or reinstalls it, rather than resetting single GPUs, which is "overcomplicated and no guarantee of recovery"; it runs light checks at boot, passive checks (`dmesg`, `dcgmi health`) continuously and active checks weekly (DCGM diagnostics level 2, a GPU burn test, a local NCCL all-reduce), and concludes that "It's underappreciated how unreliable GPUs are" ([Modal, December 2025](https://modal.com/blog/gpu-health)).

The detection side of these entries can be automated from the scrapes every replica already exports. A watchdog that maps three of them to their runbook entries, on synthetic scrapes:

```python
# watchdog.py - turn engine scrapes into runbook entries. Each scrape is what /metrics gives you
# every 10 s per replica (vLLM-style names; counters are cumulative, so rules use differences).
RULES = []
def rule(f): RULES.append(f); return f

@rule
def stuck_engine(w):   # requests are running but no token has been produced for 60 s
    if all(s["running"] > 0 for s in w[-6:]) and w[-1]["gen_tokens"] == w[-7]["gen_tokens"]:
        return "STUCK ENGINE: eject from router, capture py-spy + nvidia-smi, then restart"

@rule
def kv_exhaustion(w):  # KV full and the scheduler is evicting running sequences
    if w[-1]["kv_usage"] > 0.97 and w[-1]["preemptions"] - w[-7]["preemptions"] > 20:
        return "KV EXHAUSTION: lower max-num-seqs / admit less; check for one tenant's long prompts"

@rule
def cache_cold(w):     # prefix hit rate collapsed relative to 5 minutes ago
    def hit(a, b): return (b["hits"] - a["hits"]) / max(1, b["queries"] - a["queries"])
    if len(w) >= 37 and hit(w[-7], w[-1]) < 0.5 * hit(w[-37], w[-31]):
        return "CACHE-COLD ROUTING: check the router's replica set / hash ring changed (deploy?)"

def scrapes(kind, n=40):
    s, out = dict(running=30, gen_tokens=0, kv_usage=0.7, preemptions=0, hits=0, queries=0), []
    for i in range(n):
        s = dict(s)
        s["gen_tokens"] += 0 if (kind == "stuck" and i >= 30) else 15_000
        s["queries"] += 50_000
        s["hits"] += 10_000 if (kind == "cold" and i >= 33) else 35_000
        if kind == "kv" and i >= 30: s["kv_usage"], s["preemptions"] = 0.99, s["preemptions"] + 8
        out.append(s)
    return out

for name, kind in [("r0", "healthy"), ("r1", "stuck"), ("r2", "kv"), ("r3", "cold")]:
    w = scrapes(kind)
    alerts = [msg for f in RULES for t in range(7, len(w) + 1) if (msg := f(w[:t]))]
    print(f"{name} ({kind:7s}): {alerts[0] if alerts else 'ok'}")
```

```console
$ python3 watchdog.py
r0 (healthy): ok
r1 (stuck  ): STUCK ENGINE: eject from router, capture py-spy + nvidia-smi, then restart
r2 (kv     ): KV EXHAUSTION: lower max-num-seqs / admit less; check for one tenant's long prompts
r3 (cold   ): CACHE-COLD ROUTING: check the router's replica set / hash ring changed (deploy?)
```

The stuck-engine rule is worth having on day one: the standard engine health endpoint does not catch it, and it is the incident that turns one hung process into minutes of timeouts for every request routed there.

**On a leased fleet.** You can't replace a bad GPU by asking for another instance; you open a ticket with the provider and wait. Two practices follow. Keep at least one spare node's worth of capacity so a cordoned node doesn't cost you capacity during the wait (run batch on it meanwhile). And put the provider's hardware replacement time into your lease negotiation, since it directly sets how long you run without the node.


## 9. Security basics for an inference API

### API keys and their scope

An API key is a bearer credential: whoever holds it can spend the customer's money. The basics:

- **Store a hash, not the key.** Show the secret once at creation and store only a hash of it. A database leak then leaks no usable keys. Compare in constant time.
- **Put a public ID in the key.** A key like `sk_live_<id>_<secret>` lets you find the record without a scan, and lets logs, dashboards and support refer to keys by ID. Secret scanners (GitHub's, for example) find keys with a recognizable prefix in public repositories.
- **Scope keys**: which endpoints (inference, batch, fine-tuning, admin), which models and adapters, which limits and spend cap. A CI key should not be able to start a fine-tune.
- **Return the same 401 for unknown, revoked and wrong keys**, and 404 for another tenant's model or adapter, so an attacker can't use error messages to learn what exists.


A plain SHA-256 is enough here because the secret is 256 random bits; slow password hashes (bcrypt, scrypt) exist for low-entropy human passwords. The key ID and the order of the set in the output differ on every run:

```python
# api_keys.py - API keys stored as hashes with scopes, and per-tenant salting of prefix-cache block hashes
import hashlib, hmac, secrets

DB = {}   # key_id -> record. In production: a table; the secret itself is never stored.

def create_key(org, scopes, models, tpm):
    key_id = secrets.token_hex(4)                       # public, shows up in logs and dashboards
    secret = secrets.token_urlsafe(32)                  # 256 bits, shown to the customer exactly once
    DB[key_id] = dict(org=org, scopes=set(scopes), models=set(models), tpm=tpm, revoked=False,
                      sha256=hashlib.sha256(secret.encode()).hexdigest())
    return f"sk_live_{key_id}_{secret}"

def authorize(header, scope, model):
    try:
        _, _, key_id, secret = header.removeprefix("Bearer ").split("_", 3)
    except ValueError:
        return 401, "malformed key"
    rec = DB.get(key_id)
    digest = hashlib.sha256(secret.encode()).hexdigest()
    if rec is None or rec["revoked"] or not hmac.compare_digest(digest, rec["sha256"]):
        return 401, "invalid key"                      # same answer for unknown, revoked and wrong
    if scope not in rec["scopes"]:
        return 403, f"key lacks scope {scope}"
    if model not in rec["models"]:
        return 404, f"model {model} not found"          # don't confirm that other tenants' models exist
    return 200, rec["org"]

k = create_key("acme", ["inference"], ["llama-70b", "acme/lora-support-v3"], tpm=200_000)
print("issued:", k[:22] + "...", "| stored:", {kk: v for kk, v in DB[k.split('_')[2]].items() if kk != "sha256"})
print(authorize("Bearer " + k, "inference", "llama-70b"))
print(authorize("Bearer " + k, "inference", "zeta/lora-legal-v1"))   # another tenant's adapter
print(authorize("Bearer " + k, "batch", "llama-70b"))
print(authorize("Bearer " + k[:-1] + "x", "inference", "llama-70b"))
DB[k.split("_")[2]]["revoked"] = True
print(authorize("Bearer " + k, "inference", "llama-70b"))
```

```console
$ python3 api_keys.py
issued: sk_live_cfd20703_jKATe... | stored: {'org': 'acme', 'scopes': {'inference'}, 'models': {'llama-70b', 'acme/lora-support-v3'}, 'tpm': 200000, 'revoked': False}
(200, 'acme')
(404, 'model zeta/lora-legal-v1 not found')
(403, 'key lacks scope batch')
(401, 'invalid key')
(401, 'invalid key')
```

### Prompt and response logging, and retention

Prompts and responses are customer data: source code, contracts, medical notes. What you keep, for how long and who can read it is a product decision with legal weight, and it should be written down before the first customer arrives:

- **Log metadata by default, content by exception.** The request log in section 7 has token counts and timings, not text. Content logging is opt-in per customer (for debugging their integration), time-limited, and access-controlled.
- **Every copy has a retention period**: the request log, idempotency records with stored responses, batch inputs and outputs (Anthropic keeps batch results for 29 days ([batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing))), prompt caches in host memory or on disk, and traces, which may capture prompt snippets as span attributes. Delete on schedule, and test deletion.
- **Plan for debugging without content.** The Anthropic postmortem names privacy controls as one reason the bugs took weeks to reproduce. The tools that help are the ones in this chapter: per-replica statistics, passive checks computed on content without storing it, canary prompts that are yours, and customer-submitted feedback with explicit consent.

### The inference endpoint is an egress path

In August 2026 Prime Intellect described how a model in an "offline" evaluation sandbox reached the internet anyway ([Prime Intellect, August 2026](https://www.primeintellect.ai/blog/universal-offline-sandbox-escape)). The sandbox had no web access, but the agent could reach the inference API through an interception proxy (`verifiers`' `InterceptionServer`) with an authorized key, since it needed the model to run. It sent a normal Responses API request using the `file_url` input, "the file input capability to fetch remote content", and the provider fetched a GitHub API URL on its behalf. The result came back in the model's context, and the agent found the answer it was not supposed to have.

For an operator, the lesson is that **any feature that makes your server fetch a URL is an egress path** for whoever can send you requests: image URLs in multimodal requests, file inputs, tool calls run server-side. It is a server-side request forgery (SSRF) surface too: a URL pointing at `169.254.169.254` or an internal service makes your server fetch it from inside your network. The post lists the state of the engines: TensorRT-LLM disables remote fetching by default from v1.3.0rc15 and Dynamo from v1.4.0, SGLang added an opt-in allowlist for media domains in v0.5.18, and vLLM has had an equivalent allowlist since v0.11.0. Its recommendation applies to every operator: harden inference deployments used internally, "rather than only public-facing endpoints".

What to do:
- Disable remote fetching in the engine unless a product needs it; if it does, fetch through a proxy with a domain allowlist and no route to internal addresses.
- Treat internal endpoints (RL rollouts, evals, staging) like public ones: keys, limits, logs.

### Isolation between tenants' adapters and caches

Multi-LoRA serving ([Inference Internals II](../inference/serving-engine-internals.md#multi-lora-serving)) and prefix caching share GPU state between tenants, which makes isolation a property of your routing and naming:

- **Adapters**: authorize every request's adapter against the key's allowed models (`api_keys.py` does), namespace adapter names by organization so two tenants can't collide, and load adapters only from your own storage, never from a URL in the request.
- **Prefix caches leak through timing.** A cache hit has a lower TTFT, so a client that can time requests can test whether some prefix is cached, which means someone else sent it. Song et al. turned this into an attack, "a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches", and showed "the feasibility of stealing system prompts and those produced by peer users", including in black-box tests on public services ([arXiv 2409.20002](https://arxiv.org/abs/2409.20002)). Fireworks documents the risk for its dedicated deployments, where one cache is shared by all requests by default: "potentially, an adversary can learn that a particular prompt is cached by observing the response time"; its serverless caches are per account, and an `x-prompt-cache-isolation-key` header separates caches within a deployment ([prompt caching doc](https://docs.fireworks.ai/guides/prompt-caching)). OpenAI's caches "are not shared across organizations" ([prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)).

The fix in vLLM is a per-request `cache_salt`, which is hashed into the first block so that only requests with the same salt can reuse those cached blocks. The docs say this "prevents timing-based attacks where an adversary could infer cached content by observing latency differences" ([vLLM prefix caching design doc](https://docs.vllm.ai/en/latest/design/prefix_caching.html)). Set it in the gateway from the organization ID, never from the client. What salting does to the chained block hashes:

```python
# cache_salt.py - per-tenant salting of prefix-cache block hashes
import hashlib

# Prefix caching hashes each block of 16 tokens chained with the previous block's hash. Mixing a
# per-tenant salt into the FIRST block makes identical prompts from different tenants share nothing.
def block_hashes(tokens, salt="", block=16):
    h, out = salt.encode(), []
    for i in range(0, len(tokens) - len(tokens) % block, block):
        h = hashlib.sha256(h + repr(tokens[i:i + block]).encode()).digest()
        out.append(h.hex()[:8])
    return out

system_prompt = list(range(48))                        # the same 48-token system prompt
print("no salt  :", block_hashes(system_prompt), block_hashes(system_prompt) == block_hashes(system_prompt))
a, b = block_hashes(system_prompt, salt="org:acme"), block_hashes(system_prompt, salt="org:zeta")
print("salted   :", a, "vs", b, "shared blocks:", len(set(a) & set(b)))
```

```console
$ python3 cache_salt.py
no salt  : ['be2af200', '137dc9a1', 'ad7dee3a'] True
salted   : ['26334de0', 'b44c15f7', '2260193a'] vs ['916aa9a0', '6c579ac7', '0edf6fdf'] shared blocks: 0
```

Salting per organization gives up cross-tenant sharing of identical prefixes, such as a popular open-source agent's system prompt sent by many customers. That sharing is real savings, and it is exactly the information the side channel leaks. Share across tenants only for prefixes you publish yourself.

**On a leased fleet.** Nothing changes for security, except that you, not a cloud's managed service, own every layer: the node, the engine flags, the proxy. The Prime Intellect engines list is a checklist of flags to verify on your own deployment.


## Operations checklist for a leased fleet

In rough order of when a small team should do them. Each item names the section that explains it and the source it is based on.

1. **Compute SLIs at completion from a per-request log**, with finish reason, engine token counts and replica on every line (section 4; [Together](https://www.together.ai/blog/99-9-uptime-for-inference)).
2. **Return 429 for the client's limits and 503 for your capacity, both with `Retry-After`**, plus a machine-readable reason in the body (section 1; [Fireworks](https://docs.fireworks.ai/serverless/rate-limits), [Anthropic](https://platform.claude.com/docs/en/api/rate-limits)).
3. **Meter input, cached input and output separately, with reserve-and-refund for `max_tokens`** (section 2; [Anthropic](https://platform.claude.com/docs/en/api/rate-limits), [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits)).
4. **Add ramp limits**, so one customer's step change can't take the fleet (section 2; [Anthropic](https://platform.claude.com/docs/en/api/rate-limits)).
5. **Detect stuck engines** (running > 0, tokens flat) and eject them, and capture `py-spy` before restarting (section 8).
6. **Health-check GPUs passively all the time and actively every week; on failure, drain the whole host** (section 8; [Modal](https://modal.com/blog/gpu-health)).
7. **Disable remote URL fetching in the engines**, or allowlist it, and harden internal endpoints like public ones (section 9; [Prime Intellect](https://www.primeintellect.ai/blog/universal-offline-sandbox-escape)).
8. **Salt prefix caches per organization** (section 9; [vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching.html), [arXiv 2409.20002](https://arxiv.org/abs/2409.20002)).
9. **Run passive quality checks on all traffic and canary prompts per replica**, and compare replicas with each other (section 7; [Anthropic postmortem](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)).
10. **Treat routing and load-balancer changes as quality changes**, and gate rollouts on a quality signal as well as latency and errors (section 7; [Anthropic postmortem](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues), [Together](https://together.ai/blog/canary-rollouts-upgrade-models-in-production-without-downtime)).
11. **Replace FCFS with fair queuing in the gateway**, and keep rate limits generous (section 3; [arXiv 2401.00588](https://arxiv.org/abs/2401.00588)).
12. **Sell the trough: a batch tier with deadlines, EDF and admission control**, preempted by interactive traffic (section 5; [OpenAI](https://developers.openai.com/api/docs/guides/batch), [Anthropic](https://platform.claude.com/docs/en/build-with-claude/batch-processing), [Fireworks](https://docs.fireworks.ai/guides/ondemand-deployments)).
13. **Track utilization weekly as the number that sets cost per token**, and price from measured throughput (section 6).
14. **Store async results until delivery is confirmed** (section 1; [Baseten](https://docs.baseten.co/inference/async)).
15. **Promise only the nines your lease geography supports**, and run batch on your redundancy (section 4; [Together](https://www.together.ai/blog/99-9-uptime-for-inference), [Fireworks](https://fireworks.ai/blog/multi-region-deployment)).


## You can now read

- Provider rate-limit, batch and pricing pages ([Anthropic](https://platform.claude.com/docs/en/api/rate-limits), [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits), [Fireworks](https://docs.fireworks.ai/serverless/rate-limits)) as descriptions of the schedulers and limiters behind them, and tell which policy each one implements.
- The [VTC paper](https://arxiv.org/abs/2401.00588) and its follow-ups on fair LLM serving, and judge whether a proposed scheduler is work-conserving.
- Vendor reliability posts ([Together](https://www.together.ai/blog/99-9-uptime-for-inference), [Fireworks](https://fireworks.ai/blog/multi-region-deployment), [Modal](https://modal.com/blog/gpu-health)) critically: where they measure, over what window, and what they don't claim.
- Postmortems like [Anthropic's](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues), and write the detector that would have caught each bug.
- [RL Infrastructure](rl-infrastructure.md), where the inference service's biggest internal customer is a trainer, and the [Prime Intellect escape](https://www.primeintellect.ai/blog/universal-offline-sandbox-escape) explains why its endpoints need the same hardening as public ones.

## Further reading

Papers:

- Sheng et al., ["Fairness in Serving Large Language Models"](https://arxiv.org/abs/2401.00588) (VTC), arXiv 2401.00588.
- Zhong et al., ["DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"](https://arxiv.org/abs/2401.09670), arXiv 2401.09670.
- Song et al., ["The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems"](https://arxiv.org/abs/2409.20002), arXiv 2409.20002.
- Demers, Keshav and Shenker, "Analysis and Simulation of a Fair Queueing Algorithm", SIGCOMM 1989: the network ancestor of VTC.
- Liu and Layland, "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment", JACM 1973: earliest-deadline-first.

Provider documentation and posts (all read in November 2026):

- Anthropic: [rate limits](https://platform.claude.com/docs/en/api/rate-limits), [pricing](https://platform.claude.com/docs/en/about-claude/pricing), [batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing), ["A postmortem of three recent issues"](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) (September 2025).
- OpenAI: [rate limits](https://developers.openai.com/api/docs/guides/rate-limits), [batch](https://developers.openai.com/api/docs/guides/batch), [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
- Fireworks: [serverless rate limits](https://docs.fireworks.ai/serverless/rate-limits), [on-demand and preemptible deployments](https://docs.fireworks.ai/guides/ondemand-deployments), [batch inference (paused)](https://docs.fireworks.ai/guides/batch-inference), [prompt caching](https://docs.fireworks.ai/guides/prompt-caching), [pricing](https://fireworks.ai/pricing), ["Inside Fireworks multi-region deployments"](https://fireworks.ai/blog/multi-region-deployment) (September 2026), [FireAttention V4](https://fireworks.ai/blog/fireattention-v4-fp4-b200).
- Together: ["What does 99.9% uptime mean for inference?"](https://www.together.ai/blog/99-9-uptime-for-inference) (July 2026), [canary rollouts](https://together.ai/blog/canary-rollouts-upgrade-models-in-production-without-downtime) (September 2026), [Provisioned Throughput](https://www.together.ai/blog/provisioned-throughput) (July 2026).
- Baseten: [async inference](https://docs.baseten.co/inference/async), [Model API pricing and limits](https://docs.baseten.co/inference/model-apis/pricing-and-limits), [metrics](https://docs.baseten.co/observability/metrics), [pricing](https://www.baseten.co/pricing/).
- Modal: ["Keeping 20,000 GPUs healthy"](https://modal.com/blog/gpu-health) (December 2025), [pricing](https://modal.com/pricing).
- Prime Intellect: ["Uncovering a universal offline sandbox escape"](https://www.primeintellect.ai/blog/universal-offline-sandbox-escape) (August 2026).
- vLLM: [prefix caching design, including cache isolation](https://docs.vllm.ai/en/latest/design/prefix_caching.html).
- Google, [*Site Reliability Engineering*](https://sre.google/sre-book/table-of-contents/) and [*The Site Reliability Workbook*](https://sre.google/workbook/table-of-contents/), for SLOs and error budgets.
- In this repo: [Control Planes](control-planes.md), [Deploys, Recovery and Autoscaling](deploys-recovery-autoscaling.md), [Serving Efficiently](../inference/serving-efficiently.md), [Serving LLMs in Practice](../inference/serving-hands-on.md), [Inference Internals I](../inference/arithmetic.md) and [II](../inference/serving-engine-internals.md).
