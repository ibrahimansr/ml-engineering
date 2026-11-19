# Collectives: Laptop Measurement and Transports

Reference material split out of [Collective Communication Internals and Cost Models](collectives-internals.md) so the main chapter reads faster. The sections stand alone: read them when you need them. Earlier chapters link straight to the section they use.

## Measuring collectives on a laptop with gloo

Everything so far is a model. This section measures real collectives - real processes, real sockets, the real `torch.distributed` API - on a laptop with no GPU, using the same `algbw`/`busbw` arithmetic as nccl-tests. Then it shows the commands for doing the same on GPUs.

**What these numbers are and aren't.** On one machine, gloo's "network" is the loopback interface (`lo0` on macOS, `lo` on Linux): a virtual network device inside the kernel that hands bytes from one socket to another. So a loopback gloo benchmark measures memory copies through the kernel's TCP stack, the CPU doing the reduction, process scheduling and gloo's own overhead. It does **not** measure any network, NIC or switch, and the bandwidth numbers below say nothing about what a cluster will do. What does transfer is the *shape*: the latency floor, the ramp around $n_{1/2}$, the plateau, how `busbw` normalizes across rank counts, and how noisy a shared machine is.

### Launching a process group on one machine

A *process group* is the set of processes that take part in collectives together; each process is a *rank*. They find each other through a *rendezvous*: rank 0 opens a small TCP key-value store (the `TCPStore`) at `MASTER_ADDR:MASTER_PORT`, every rank connects to it and publishes its own address, and then the backend (here gloo) opens direct connections between ranks. Background in [How PyTorch distributed bootstraps](../foundations/networking.md#how-pytorch-distributed-bootstraps).

Two ways to launch, both tested on the M4:

```python
# [torch] gloo_smoke.py - spawn 4 ranks from one script with torch.multiprocessing
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
def run(rank, world):
    dist.init_process_group("gloo", rank=rank, world_size=world)
    x = torch.full((4,), float(rank))                 # (4,)
    dist.all_reduce(x)                                # sum over ranks: 0+1+2+3
    out = [torch.empty(2) for _ in range(world)]      # world x (2,)
    dist.all_gather(out, torch.full((2,), float(rank)))
    y = torch.empty(2); dist.reduce_scatter_tensor(y, torch.ones(2 * world))   # (2w,) -> (2,)
    a = torch.arange(world, dtype=torch.float) + 10 * rank                     # (w,)
    b = torch.empty(world); dist.all_to_all_single(b, a)                       # (w,) -> (w,)
    if rank == 0: print(x.tolist(), [o.tolist() for o in out], y.tolist(), b.tolist())
    dist.destroy_process_group()
if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29511")
    mp.spawn(run, args=(4,), nprocs=4)
```

```
$ /tmp/mle-venv/bin/python gloo_smoke.py
[6.0, 6.0, 6.0, 6.0] [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]] [4.0, 4.0] [0.0, 10.0, 20.0, 30.0]
```

```python
# [torch] tr_allreduce.py - the same collective under torchrun, which sets RANK/WORLD_SIZE/MASTER_* for you
import torch, torch.distributed as dist
dist.init_process_group("gloo")                       # reads env:// variables set by torchrun
x = torch.full((4,), float(dist.get_rank()))          # (4,)
dist.all_reduce(x)
print(f"rank {dist.get_rank()}/{dist.get_world_size()}: {x.tolist()}")
dist.destroy_process_group()
```

```
$ OMP_NUM_THREADS=1 /tmp/mle-venv/bin/torchrun --nproc_per_node 4 \
      --master_addr 127.0.0.1 --master_port 29541 tr_allreduce.py
W1105 16:31:51.309000 62120 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.
rank 0/4: [6.0, 6.0, 6.0, 6.0]
rank 1/4: [6.0, 6.0, 6.0, 6.0]
rank 3/4: [6.0, 6.0, 6.0, 6.0]
rank 2/4: [6.0, 6.0, 6.0, 6.0]
```

What I needed on macOS 15.7 / torch 2.15.0, so you don't have to rediscover it:

- With `mp.spawn`, `MASTER_ADDR=127.0.0.1` plus any free `MASTER_PORT` was enough. `MASTER_ADDR=localhost` also worked. `GLOO_SOCKET_IFNAME=lo0` (which tells gloo which network interface to bind) wasn't needed here, but is harmless, and is the fix on machines where the hostname resolves to an address the other ranks can't reach.
- `torchrun --standalone` **hung** (killed after 45s and again after 120s) while printing repeated `[c10d] The IPv6 network addresses of (...ip6.arpa, ...) cannot be retrieved` warnings - the rendezvous store failing to reverse-resolve `::1`, which is what this Mac's hostname resolves to. Setting `GLOO_SOCKET_IFNAME=lo0` did not fix it, because it's the rendezvous, not gloo, that's stuck. Passing `--master_addr 127.0.0.1 --master_port <port>` instead of `--standalone` worked.
- `OMP_NUM_THREADS=1` (or `torch.set_num_threads(1)` in each rank) stops 8 ranks from each starting a 10-thread intra-op pool on a 10-core machine - see [Scheduling and context switches](../foundations/concurrency.md#scheduling-and-context-switches).
- On this build gloo's transport is libuv (the error messages below come from `gloo/transport/uv/`), where Linux builds use gloo's TCP transport. Same sockets underneath; I didn't compare them.

### An nccl-tests-style benchmark for gloo

nccl-tests' conventions, from [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md): for all-reduce the reported size $S$ is the per-rank buffer and $\text{busbw} = \text{algbw} \cdot 2(p-1)/p$; for all-gather and reduce-scatter $S$ is the full, un-sharded buffer and the factor is $(p-1)/p$. The script uses those, takes the slowest rank's time (a collective isn't done until the last rank is done), and sweeps sizes by 4x from 4 bytes to 256MiB:

```python
# gloo_bench.py - nccl-tests-style algbw/busbw for gloo collectives on CPU [torch]
# usage: python gloo_bench.py --world 4 --coll all_reduce --max-bytes 268435456
import argparse, os, time, torch, torch.distributed as dist, torch.multiprocessing as mp

# busbw factor and "size S" convention from nccl-tests doc/PERFORMANCE.md:
#   all_reduce: S = per-rank buffer, factor 2(p-1)/p
#   all_gather / reduce_scatter: S = full (un-sharded) buffer, factor (p-1)/p
FACTOR = {"all_reduce": lambda p: 2 * (p - 1) / p,
          "all_gather": lambda p: (p - 1) / p,
          "reduce_scatter": lambda p: (p - 1) / p}

def make_op(coll, S, p):
    n = max(S // 4, 1)                          # fp32 elements in the full buffer
    if coll == "all_reduce":
        x = torch.ones(n)                       # (n,)
        return n * 4, lambda: dist.all_reduce(x)
    n = max(n - n % p, p)                       # sharded collectives: divisible by p
    if coll == "all_gather":
        inp, out = torch.ones(n // p), torch.empty(n)     # (n/p,) -> (n,)
        return n * 4, lambda: dist.all_gather_into_tensor(out, inp)
    if coll == "reduce_scatter":
        inp, out = torch.ones(n), torch.empty(n // p)     # (n,) -> (n/p,)
        return n * 4, lambda: dist.reduce_scatter_tensor(out, inp)

def worker(rank, p, coll, max_bytes):
    torch.set_num_threads(1)                    # p processes share 10 cores; avoid oversubscription
    dist.init_process_group("gloo", rank=rank, world_size=p)
    S = 4 if coll == "all_reduce" else 4 * p
    if rank == 0:
        print(f"# gloo {coll}  world_size={p}  torch {torch.__version__}")
        print(f"{'size(B)':>11} {'time(us)':>10} {'algbw(GB/s)':>12} {'busbw(GB/s)':>12}")
    while S <= max_bytes:
        nbytes, op = make_op(coll, S, p)
        iters = max(5, min(200, int(2e8 // max(nbytes, 1))))   # ~fixed bytes per measurement
        for _ in range(3): op()                 # warmup (connection setup, allocations)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters): op()
        t = torch.tensor([(time.perf_counter() - t0) / iters])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank defines the collective's time
        t = t.item()
        algbw = nbytes / t / 1e9
        busbw = algbw * FACTOR[coll](p)
        if rank == 0:
            print(f"{nbytes:>11} {t*1e6:>10.1f} {algbw:>12.3f} {busbw:>12.3f}", flush=True)
        S *= 4
    dist.destroy_process_group()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--coll", default="all_reduce", choices=list(FACTOR))
    ap.add_argument("--max-bytes", type=int, default=256 * 2**20)
    a = ap.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")   # loopback; see text
    os.environ.setdefault("MASTER_PORT", "29500")
    mp.spawn(worker, args=(a.world, a.coll, a.max_bytes), nprocs=a.world)
```

```bash
cd /tmp/mle-coll
for w in 2 4 8; do /tmp/mle-venv/bin/python gloo_bench.py --world $w --coll all_reduce | tee ar_${w}ranks.txt; done
```

### All-reduce results, 2/4/8 ranks

Real output from the M4, the three runs merged into one table (time in µs, bandwidths in GB/s; fp32, so 4 bytes is one element):

```
            |  2 ranks                 |  4 ranks                 |  8 ranks
    size(B) | time(us) algbw  busbw    | time(us) algbw  busbw    | time(us) algbw  busbw
          4 |      201 0.000  0.000    |      431 0.000  0.000    |      819 0.000  0.000
         16 |      403 0.000  0.000    |      846 0.000  0.000    |     1771 0.000  0.000
         64 |      424 0.000  0.000    |     1286 0.000  0.000    |     4354 0.000  0.000
        256 |      351 0.001  0.001    |     1137 0.000  0.000    |     3947 0.000  0.000
       1024 |      354 0.003  0.003    |     1498 0.001  0.001    |     3392 0.000  0.001
       4096 |      369 0.011  0.011    |     1545 0.003  0.004    |     3389 0.001  0.002
      16384 |      380 0.043  0.043    |     1179 0.014  0.021    |     3476 0.005  0.008
      65536 |      445 0.147  0.147    |     1343 0.049  0.073    |     3456 0.019  0.033
     262144 |      475 0.552  0.552    |     2476 0.106  0.159    |     3915 0.067  0.117
    1048576 |      852 1.231  1.231    |     1969 0.532  0.799    |     4976 0.211  0.369
    4194304 |     2782 1.508  1.508    |     5754 0.729  1.093    |     9132 0.459  0.804
   16777216 |    10089 1.663  1.663    |    17434 0.962  1.443    |    27631 0.607  1.063
   67108864 |    38265 1.754  1.754    |    92179 0.728  1.092    |   115066 0.583  1.021
  268435456 |   148509 1.808  1.808    |   265452 1.011  1.517    |   807635 0.332  0.582
```

How to read it:

- **Latency floor.** Below ~64KB the time doesn't depend on size: ~350-420µs for 2 ranks, ~0.9-1.5ms for 4, ~3.4-4.3ms for 8. That's $\alpha_\text{total}$, and it grows with $p$ (more sequential steps, more processes to schedule on a loaded machine). A GPU all-reduce on NVLink has a floor of ~15µs ([Latency](measuring-and-debugging.md#latency)) - 20-300x lower, because no kernel socket, syscall or process wakeup sits on the path.
- **Plateau.** `busbw` tops out at ~1.8GB/s (2 ranks), ~1.5 (4) and ~1.0 (8). In a contention-free world with a fixed per-link $\beta$, `busbw` would be the same for all three; here it isn't, because the ranks share 10 cores (4 fast, 6 slow), one memory system and one kernel, so adding ranks adds contention rather than links.
- **Noise.** Earlier runs of the same script on the same machine gave plateaus of 2.03GB/s (2 ranks), 1.32 and 1.50GB/s (4 ranks, two runs) and 1.03GB/s (8 ranks), and 4-rank small-message latencies anywhere from 0.5 to 4.7ms. The machine was shared (load average 10-20). The 8-rank drop at 256MiB (1.06 -> 0.58GB/s `busbw` here, 0.79GB/s in the earlier run) appeared in both runs; I didn't investigate it (memory pressure is one guess: 8 ranks x 256MiB buffers plus gloo's internal copies, on 16GB).
- **`busbw` doing its job.** At 16MiB, `algbw` is 1.66 / 0.96 / 0.61GB/s for 2/4/8 ranks - it falls with $p$ as the [algbw vs busbw](collectives-internals.md#algbw-vs-busbw) section predicts - while `busbw` is 1.66 / 1.44 / 1.06.

### Fitting alpha and beta to the curve

Two parameters describe the whole curve surprisingly well. A crude fit - $\alpha$ from the median of the small sizes, $\beta$ from the slope of the two largest:

```python
# fit_ab.py - fit alpha and beta to a gloo_bench.py output file
import sys, statistics
rows = [tuple(map(float, l.split())) for l in open(sys.argv[1]) if l.strip()[0].isdigit()]
S = [r[0] for r in rows]; t = [r[1] * 1e-6 for r in rows]          # bytes, seconds
alpha = statistics.median(ti for si, ti in zip(S, t) if si <= 4096)  # latency-bound points
beta = (t[-1] - t[-2]) / (S[-1] - S[-2])                             # slope of the two largest
print(f"alpha ~ {alpha*1e6:.0f} us   1/beta ~ {1/beta/1e9:.2f} GB/s (algbw asymptote)"
      f"   n_1/2 = alpha/beta ~ {alpha/beta/1e6:.2f} MB")
for si, ti in zip(S, t):
    print(f"{si:>11.0f}  measured {ti*1e6:>9.0f} us   model {(alpha + si*beta)*1e6:>9.0f} us")
```

```
$ python3 fit_ab.py ar_2ranks.txt
alpha ~ 361 us   1/beta ~ 1.83 GB/s (algbw asymptote)   n_1/2 = alpha/beta ~ 0.66 MB
          4  measured       201 us   model       361 us
         16  measured       403 us   model       361 us
         64  measured       424 us   model       361 us
        256  measured       351 us   model       361 us
       1024  measured       354 us   model       362 us
       4096  measured       369 us   model       363 us
      16384  measured       380 us   model       370 us
      65536  measured       445 us   model       397 us
     262144  measured       475 us   model       505 us
    1048576  measured       852 us   model       935 us
    4194304  measured      2782 us   model      2658 us
   16777216  measured     10089 us   model      9548 us
   67108864  measured     38265 us   model     37109 us
  268435456  measured    148509 us   model    147353 us
```

The model is within ~10% from 256KB up, and its $n_{1/2}$ of ~0.66MB is where the measured curve is at half its plateau. The misses at 4 and 16 bytes are real effects the model doesn't have (gloo may take a different code path for tiny buffers; I didn't check). This is the exercise to repeat on a real cluster: fit $\alpha$ and $\beta$ to `all_reduce_perf` output, and you have the two numbers you need to predict every bucket size your training job uses.

### All-gather and reduce-scatter

```bash
for c in all_gather reduce_scatter; do /tmp/mle-venv/bin/python gloo_bench.py --world 4 --coll $c; done
```

```
            |  all_gather, 4 ranks     |  reduce_scatter, 4 ranks
    size(B) | time(us) algbw  busbw    | time(us) algbw  busbw
         16 |      542 0.000  0.000    |      667 0.000  0.000
         64 |      455 0.000  0.000    |     1049 0.000  0.000
        256 |      480 0.001  0.000    |      992 0.000  0.000
       1024 |      450 0.002  0.002    |     1346 0.001  0.001
       4096 |      454 0.009  0.007    |     1163 0.004  0.003
      16384 |      469 0.035  0.026    |      918 0.018  0.013
      65536 |      464 0.141  0.106    |     1084 0.060  0.045
     262144 |      571 0.459  0.344    |     1324 0.198  0.148
    1048576 |      970 1.081  0.810    |     2234 0.469  0.352
    4194304 |     2959 1.417  1.063    |     5920 0.708  0.531
   16777216 |     9226 1.818  1.364    |    18272 0.918  0.689
   67108864 |    38036 1.764  1.323    |    74292 0.903  0.677
  268435456 |   139170 1.929  1.447    |   288328 0.931  0.698
```

All-gather reaches ~1.4GB/s `busbw`, similar to the 4-rank all-reduce; reduce-scatter only ~0.7GB/s, and its small-message latency is about twice all-gather's. Per the model they move the same bytes, so the 2x is an implementation property of gloo's reduce-scatter in this build - I haven't read how `ProcessGroupGloo` implements it, so I can't tell you why. That itself is the lesson: **the model tells you what to expect; a 2x miss tells you where to look**.

### The same measurement on GPUs: nccl-tests and torch-dist-bench

On a GPU node the measurement is the same, with NCCL instead of gloo. The repo's [`torch-dist-bench.py`](measuring-and-debugging.md#torch-dist-bench) is the PyTorch-level tool, and [nccl-tests](measuring-and-debugging.md#nccl-tests) is the reference; their setup and flags are documented in [Measuring and debugging the network](measuring-and-debugging.md). **[GPU, untested here]** - typical command lines:

```bash
# nccl-tests: build once, then sweep 8B..8GB by 2x on all 8 local GPUs, one process per GPU
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make -j
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8        # -b min bytes, -e max bytes, -f factor, -g GPUs per process

# PyTorch-level, same conventions, from this repo
python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    network/benchmarks/torch-dist-bench.py

# see NCCL's choices while it runs
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8
# measure one algorithm/protocol at a time - for understanding, not for production
NCCL_ALGO=Ring NCCL_PROTO=Simple ./build/all_reduce_perf -b 1M -e 1G -f 2 -g 8
NCCL_ALGO=Tree ./build/all_reduce_perf -b 1M -e 1G -f 2 -g 8
```

How to read these lines, if [GPU software stack](../compute/gpu-software-stack.md#nccl-tests-single-node-then-two-then-all) is not fresh: `git clone ... && make` downloads and compiles the C/CUDA sources of nccl-tests ([toolkit](../foundations/toolkit.md#reading-c), git and Make sections); `./build/all_reduce_perf` is the resulting program. `python -m torch.distributed.run` is the same thing as the `torchrun` command, and `--rdzv_endpoint localhost:6000 --rdzv_backend c10d` tells it where the rendezvous store lives (the `MASTER_ADDR:MASTER_PORT` idea from above; [orchestration internals](../orchestration/internals.md#c10d-rendezvous) goes deeper). A line that starts with `NAME=value NAME2=value2 command` sets those environment variables for that one command only ([environment variables](../foundations/linux-and-os.md#environment-variables-and-path)). NCCL reads all of its configuration this way: `NCCL_DEBUG=INFO` makes it log its decisions to stderr, `NCCL_DEBUG_SUBSYS` picks which parts log (`INIT` setup, `GRAPH` topology and rings, `TUNING` algorithm choice), and `NCCL_ALGO` / `NCCL_PROTO` restrict which algorithm and protocol it may use (both explained in [NCCL internals](collectives-internals.md#nccl-internals) below). The full list is NCCL's [environment variables page](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

The output has the same `algbw`/`busbw` columns as the gloo script (per operation, out-of-place and in-place); compare the plateau `busbw` with the per-direction link rate, as in [algbw vs busbw](collectives-internals.md#algbw-vs-busbw).

## Transports

### InfiniBand

InfiniBand is a network technology built for RDMA, separate from Ethernet. It is lossless by construction: link-level, credit-based flow control means a sender never transmits unless the receiver has buffer space, so packets are not dropped under congestion - they wait. Routing is computed by the *subnet manager* (a software service, running on a switch or a host, that discovers the fabric and programs every switch's forwarding tables). NVIDIA's IB switches support **adaptive routing**, which picks among equal-cost output ports per packet based on queue occupancy rather than a static hash; the NIC handles the resulting out-of-order delivery for RDMA. Adaptive routing turns "two flows hashed onto the same uplink" from a persistent 2x slowdown into transient queueing - see [InfiniBand adaptive routing](measuring-and-debugging.md#infiniband-adaptive-routing). IB also has congestion control (based on FECN/BECN - forward and backward explicit congestion notification bits that switches set in packet headers, the IB counterpart of ECN), but tuning it is rarely exposed to users.

The NCCL-visible failure semantics come from RC (reliable connected) queue pairs - a QP is the pair of send/receive queues the NIC keeps for one RDMA connection: an RDMA op that isn't acknowledged within the local ACK timeout is retransmitted, up to a retry count, after which the QP goes to error and NCCL reports a fatal error. `NCCL_IB_TIMEOUT` sets the exponent: the timeout is $4.096\ \mu\text{s} \cdot 2^{\text{NCCL\_IB\_TIMEOUT}}$, so 20 gives ~4.3s and 22 gives ~17.2s per attempt, multiplied by `NCCL_IB_RETRY_CNT` retries. (I believe the default timeout exponent is 20 in recent versions; older versions used a lower value - check your version's docs.)

```python
# ib_timeout.py - how long before an unacknowledged RDMA op kills the job
for to in (14, 18, 20, 22):
    t = 4.096e-6 * 2 ** to
    print(f"NCCL_IB_TIMEOUT={to}: {t:7.3f} s per attempt, x7 retries = {7*t:6.1f} s")
```

```
NCCL_IB_TIMEOUT=14:   0.067 s per attempt, x7 retries =    0.5 s
NCCL_IB_TIMEOUT=18:   1.074 s per attempt, x7 retries =    7.5 s
NCCL_IB_TIMEOUT=20:   4.295 s per attempt, x7 retries =   30.1 s
NCCL_IB_TIMEOUT=22:  17.180 s per attempt, x7 retries =  120.3 s
```

The `x7` assumes `NCCL_IB_RETRY_CNT=7`, which I believe is the default (7 is also the maximum the IB retry-count field can hold); the exact accounting of attempts vs retries is a detail I haven't verified. The point is the order of magnitude: a dead link takes tens of seconds to surface as an error at the default, and minutes at 22.

**[GPU, untested]** - the IB-level checks before blaming NCCL. These tools were introduced in [InfiniBand for operators](../compute/gpu-software-stack.md#infiniband-for-operators-ports-the-subnet-manager-lids) and [perftest](../compute/gpu-software-stack.md#perftest-measuring-one-link); in short, an *HCA* (host channel adapter) is an IB NIC and shows up as a device named like `mlx5_0`, `ibstat` prints each HCA port's state and rate, `ibv_devinfo` is the same view through the verbs library (the RDMA programming interface whose queue pairs NCCL uses, see [InfiniBand](../foundations/networking.md#infiniband)), and `ib_write_bw` from the perftest package measures raw RDMA-write bandwidth between two NICs with no NCCL involved - start it on one node, then point the same command at that node's address from the other:

```bash
ibstat                              # each HCA port: State: Active, Physical state: LinkUp, Rate: 400
ibv_devinfo | grep -E "hca_id|state|active_mtu"
ib_write_bw -d mlx5_0 --report_gbits          # server; on the peer: ib_write_bw -d mlx5_0 <server> --report_gbits
```

### RoCEv2

RoCEv2 (RDMA over Converged Ethernet) carries the IB transport inside ordinary UDP/IP packets (UDP destination port 4791), so it runs on Ethernet switches - but the RC transport assumes near-lossless delivery, because its loss recovery (go-back-N in most NICs: on a lost packet, resend it *and everything sent after it*) is expensive. Two mechanisms make Ethernet behave:

- **PFC** (Priority Flow Control, 802.1Qbb): a switch whose ingress queue for a priority fills sends a PAUSE upstream for that priority. Lossless, but coarse: a PAUSE stops *all* traffic of that class on the link, not just the flow causing it, so congestion spreads backwards (head-of-line blocking, congestion trees), and under pathological conditions PFC pause storms and deadlocks are possible.
- **ECN + DCQCN**: switches mark packets when queues exceed a threshold; the receiver NIC returns congestion notification packets; the sender NIC cuts its rate multiplicatively and recovers over time ([Zhu et al., Congestion Control for Large-Scale RDMA Deployments](https://doi.org/10.1145/2785956.2787484), SIGCOMM 2015). The idea is to throttle before PFC triggers, keeping PFC as a last-resort backstop.

The tuning surface is large (ECN thresholds, DCQCN rate parameters, PFC headroom, traffic classes - `NCCL_IB_TC`, `NCCL_IB_GID_INDEX` select class and RoCE version), and mistuning shows up as low or erratic collective bandwidth rather than errors. Meta describes running RoCE for training at scale, including that DCQCN tuning was not the main lever for them and that collective-library-level flow control and routing mattered more ([Gangidi et al., RDMA over Ethernet for Distributed AI Training at Meta Scale](https://doi.org/10.1145/3651890.3672233), SIGCOMM 2024). Vendor Ethernet fabrics for AI (Spectrum-X, see [README](interconnects.md#spectrum-x-ethernet)) add per-packet adaptive routing on the switch and reordering in the NIC, i.e. they import the IB approach.

**[GPU, untested]** - the NCCL side of a RoCE setup is mostly selecting the right GID (the RoCE address entry: v1 or v2, IPv4 or IPv6) and traffic class; the values depend on the cluster's switch configuration, so the numbers below are placeholders to be replaced with what your network team configured. (The `for` loop prints every GID table entry of port 1 of `mlx5_0` from sysfs; [RoCE in one page](../compute/gpu-software-stack.md#roce-in-one-page) shows what the table looks like. `export` sets a variable for every later command in this shell, so the training job launched afterwards inherits it. `NCCL_SOCKET_IFNAME` is covered in [Which interface does NCCL pick?](../foundations/networking.md#which-interface-does-nccl-pick).)

```bash
# which GID index is RoCE v2 on this port
for i in /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/*; do echo "$i: $(cat $i 2>/dev/null)"; done
export NCCL_IB_GID_INDEX=3      # placeholder: the index whose type reads "RoCE v2"
export NCCL_IB_TC=106           # placeholder: the traffic class mapped to the lossless priority
export NCCL_SOCKET_IFNAME=eth0  # interface for NCCL's bootstrap TCP connections
```

### Flow collisions and ECMP

Without adaptive routing ([ECMP](../foundations/networking.md#ecmp) has the basics), a switch with several equal-cost uplinks chooses one per flow (a flow is the sequence of packets of one connection) by hashing header fields - this is ECMP, equal-cost multi-path routing - (for RoCEv2, typically the 5-tuple including the UDP source port, which the NIC derives from the QP). Collective traffic has **low entropy** (few distinct flows, so few distinct hash inputs): a ring all-reduce over $N$ nodes with 8 channels is a handful of long-lived, equally sized *elephant flows* (large, long-lived flows, as opposed to the many short "mice" of web traffic, which hashing spreads well). Hashing $k$ such flows onto $k$ uplinks:

- $P(\text{no collision}) = k!/k^k$ - 0.24% for $k = 8$, ~1e-6 for $k = 16$.
- Expected load on the most loaded uplink (simulated, 20k trials): 2.59 flows for $k = 8$, 3.07 for $k = 16$.

A ring is synchronous - every step waits for the slowest hop - so the whole collective runs at the rate of the most-shared link. With 2.59 flows sharing the worst uplink, the expected bandwidth is $1/2.59 \approx 39\%$ of the NIC rate, for the whole ring, until the flows are re-hashed. That's the mechanism behind "same job, same nodes, `busbw` varies 2x between runs" on ECMP fabrics.

Mitigations, all of which increase entropy or remove hashing:

- [`NCCL_IB_QPS_PER_CONNECTION`](measuring-and-debugging.md#nccl_ib_qps_per_connection): split each connection over $q$ QPs (each with its own UDP source port), each carrying $1/q$ of the traffic. Simulated expected worst-uplink load drops from 2.59 to 1.76 ($q = 4$) and 1.37 ($q = 16$) flows' worth for $k = 8$. Costs QP state and some GPU memory.
- `NCCL_IB_SPLIT_DATA_ON_QPS`: whether NCCL round-robins messages over those QPs or splits each message across them (semantics vary by version - check the env docs).
- Adaptive routing (IB, Spectrum-X and similar), which doesn't depend on the hash at all.
- Rail-optimized topology, which keeps ring traffic below the tier where the hashing happens.

The simulation is a balls-into-bins model with uniform hashing; real hash functions on real headers can be better or worse, so treat these as typical rather than predictive.

```python
# ecmp_sim.py - balls into bins: k equal elephant flows hashed onto k uplinks
import math, random
random.seed(0)
def worst_load(k, q=1, trials=20_000):
    """each flow is split over q QPs (sub-flows of weight 1/q), each hashed independently."""
    tot = 0.0
    for _ in range(trials):
        load = [0.0] * k
        for _ in range(k * q): load[random.randrange(k)] += 1 / q
        tot += max(load)
    return tot / trials
for k in (8, 16):
    print(f"k={k:>2}: P(no collision) = k!/k^k = {math.factorial(k)/k**k:.2e}   "
          f"E[max load] = {worst_load(k):.2f} flows  -> ring runs at ~{1/worst_load(k):.0%} of NIC rate")
for q in (4, 16):
    print(f"k=8, QPS_PER_CONNECTION={q:>2}: E[max load] = {worst_load(8, q):.2f}")
```

```
k= 8: P(no collision) = k!/k^k = 2.40e-03   E[max load] = 2.60 flows  -> ring runs at ~39% of NIC rate
k=16: P(no collision) = k!/k^k = 1.13e-06   E[max load] = 3.08 flows  -> ring runs at ~32% of NIC rate
k=8, QPS_PER_CONNECTION= 4: E[max load] = 1.76
k=8, QPS_PER_CONNECTION=16: E[max load] = 1.37
```

(Seeded Monte Carlo; the third digit moves with the seed - earlier unseeded runs gave 2.59 and 3.07.)

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).
