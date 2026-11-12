# Collective Communication Internals and Cost Models

## Prerequisites

Read these first if the systems vocabulary is new to you:

- [Toolkit](../foundations/toolkit.md) - terminal basics, git and Make (the nccl-tests build is `git clone` + `make`), JSON (the profiler trace parsed near the end), and the "reading C" section (NCCL is a C library; `ncclConfig_t` and friends are C structs).
- [Computer architecture](../foundations/computer-architecture.md) - GPU memory (HBM), SMs, caches and PCIe; every bandwidth in this chapter is one of these.
- [Linux and the OS](../foundations/linux-and-os.md) - processes, NUMA, sysfs and drivers, which is where NCCL reads the machine's topology from; also [environment variables](../foundations/linux-and-os.md#environment-variables-and-path), which is how every `NCCL_*` setting below is passed.
- [Concurrency](../foundations/concurrency.md) - memory fences, flags, polling and streams; the NCCL protocols and compute-comm overlap are concurrency problems.
- [Networking](../foundations/networking.md) - links, switches, packets, flow control, RDMA and routing; needed for the topology and transport sections. In particular [HPC networking: RDMA, InfiniBand, RoCE, EFA](../foundations/networking.md#hpc-networking-rdma-infiniband-roce-efa) (HCAs, verbs, queue pairs, PFC/ECN), [ECMP](../foundations/networking.md#ecmp) and [how PyTorch distributed bootstraps](../foundations/networking.md#how-pytorch-distributed-bootstraps).
- [Distributed systems](../foundations/distributed-systems.md) - ranks, collectives, synchronization, stragglers and timeouts, and [data parallel vs model parallel](../foundations/distributed-systems.md#data-parallel-vs-model-parallel) for the one-paragraph versions of DDP, ZeRO/FSDP, tensor and pipeline parallelism used throughout.
- [Transformer architectures](../models/architectures.md) - the MLP shapes used in the async-TP example and [Mixture of Experts](../models/architectures.md#mixture-of-experts), whose expert routing is what generates all-to-alls.
- [GPU execution model](../compute/accelerator/execution-model.md) - SMs, warps, CTAs (thread blocks), kernels, GEMMs and [wave quantization](../compute/accelerator/execution-model.md#why-dims-want-multiples-of-8-16-64-128); a NCCL channel is one CTA, and SM contention is about these.
- [GPU software stack](../compute/gpu-software-stack.md) - where NCCL, nccl-tests and the InfiniBand tools come from: [InfiniBand for operators](../compute/gpu-software-stack.md#infiniband-for-operators-ports-the-subnet-manager-lids) (`ibstat`), [perftest](../compute/gpu-software-stack.md#perftest-measuring-one-link) (`ib_write_bw`), [the NCCL environment variables an operator sets](../compute/gpu-software-stack.md#the-environment-variables-an-operator-sets) and [nccl-tests](../compute/gpu-software-stack.md#nccl-tests-single-node-then-two-then-all).

Forward links (later chapters; this one defines just enough where it first uses them): [Slurm from zero](../orchestration/slurm/from-zero.md) (`srun`, used in two multi-node command blocks), [orchestration internals](../orchestration/internals.md#c10d-rendezvous) (torchrun's rendezvous), [model parallelism derivations](../training/model-parallelism/derivations.md) (the full treatment of DDP buckets, FSDP, TP/SP and EP volumes) and [profiling and optimization](../training/profiling-and-optimization.md) (reading real training traces).

## Introduction

This chapter is about the theory underneath the measurements in the rest of this section: why a collective costs what it costs, how NCCL (NVIDIA's collective communication library, the backend PyTorch uses for GPU collectives) turns a topology into a schedule, and how to predict the number before running the benchmark. It starts by showing what each point-to-point, one-sided and collective pattern does ([The communication patterns](#the-communication-patterns)), and assumes you know what the hardware is ([Interconnects](interconnects.md)). The next chapter, [Measuring and debugging the network](measuring-and-debugging.md), is the practical side: the benchmarks that produce the GPU numbers quoted here, and what to do when they come out wrong. Nothing here replaces measuring - the point is to know what number to expect, so that you can tell when a measurement is wrong.

All arithmetic in this chapter was checked with `python3`. Where a parameter value is an assumption rather than a spec, it is labelled as such. No benchmark numbers are invented here - where a GPU measurement appears, it is quoted from elsewhere in this repo, with a link.

If you are new to systems, the [collectives introduction in Networking](../foundations/networking.md#collective-communication-an-intuitive-introduction) shows what each collective computes on three tiny arrays and walks one 3-rank ring all-reduce by hand. Read it first; this chapter starts where it stops.

### How to use the code in this chapter

The chapter is meant to be read with a terminal open. Almost every section has a small program you can run:

- **Pure Python** blocks need only `python3`. They are simulators: they move numbers between pretend ranks exactly as the algorithm does, print every step, and `assert` that the final answer equals the true sum. If you can read the simulator, you understand the algorithm.
- **[torch]** blocks need `pip install torch numpy` in a virtualenv and run on a laptop CPU. They use `torch.distributed` with the **gloo** backend (PyTorch's CPU collective library, which talks over ordinary TCP sockets), so you can launch 2, 4 or 8 real processes on one machine and call real collectives.
- **[GPU, untested]** blocks need NVIDIA GPUs, NCCL and usually Linux. They were **not run** while writing this chapter; they follow the documented commands and APIs, but treat them as untested.

The outputs pasted below come from one Apple M4 laptop (4 performance + 6 efficiency cores, 16GB RAM, macOS 15.7), Python 3.14 and PyTorch 2.15.0, while other jobs were also using the machine (load average 10-20 on 10 cores during the runs). Treat every timing as one noisy anecdote. The simulators' outputs are deterministic (or seeded) and reproduce exactly.

```bash
python3 -m venv /tmp/mle-venv && /tmp/mle-venv/bin/pip install torch numpy
mkdir -p /tmp/mle-coll && cd /tmp/mle-coll      # put the scripts from this chapter here
/tmp/mle-venv/bin/python -c "import torch, torch.distributed as d; print(torch.__version__, d.is_gloo_available())"
# 2.15.0 True
```

## The communication patterns

Before costing anything, here is what each pattern *does*. There are three families:

- **Point-to-point**: one sender, one receiver.
- **One-sided**: one rank reads or writes another rank's memory, and the other rank doesn't take part in the call.
- **Collective**: every rank in a group takes part. The data either flows from many ranks to one, from one to many, or from many to many.

Below, each pattern gets a picture and its PyTorch call, and one script at the end of the section runs them all on 4 CPU processes so you can see the data move. The pictures come from NVIDIA's 2015 NCCL introduction by Cliff Woolley ([slides](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)). Each picture shows the buffers before (top or left) and after the call; a box is one rank's buffer, a letter is a block of data, and in the reduce-type pictures a sum like `A+B+C+D` means "the elementwise reduction of those blocks". The PyTorch function names link to the [`torch.distributed` docs](https://docs.pytorch.org/docs/stable/distributed.html).

### Point-to-point

Point-to-point is the simplest pattern: one sender, one receiver. [Pipeline parallelism](../training/model-parallelism/derivations.md#pipeline-parallelism) is the typical user: the GPU holding one stage of layers finishes its forward pass and sends the activations to the GPU holding the next stage, so the first GPU calls `send` and the second calls `recv`.

PyTorch has a blocking pair, `dist.send(tensor, dst)` / `dist.recv(tensor, src)`, which return only when the transfer is done, and a non-blocking pair, `dist.isend` / `dist.irecv`, which return a request handle at once; you call `.wait()` on it before touching the buffer ([PyTorch's point-to-point tutorial](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html#id1)). The non-blocking form is what lets every rank send and receive at the same time without deadlocking: with blocking calls, if every rank first calls `send`, everyone waits for a receiver that never comes. `dist.batch_isend_irecv` takes a list of `dist.P2POp` objects and issues them together as one batch, which is how pipeline schedules send and receive in the same step; [Other collectives](measuring-and-debugging.md#other-collectives) measures what this costs on a GPU node. In the script below, each rank sends its tensor to the next rank around a ring and receives from the previous one.

### One-sided communication

Collectives and `send`/`recv` are *two-sided*: every rank that takes part posts a matching call. *One-sided* communication, also called RMA (remote memory access), lets one rank write into or read from another rank's memory without that peer posting a `recv`. That is the useful shape for irregular access - an embedding lookup, a weight copy, routing one token to one expert - where making every rank join an all-to-all would move more data than the algorithm needs.

From PyTorch 2.14, the NCCL backend exposes this as a *window*: a region of each rank's memory that is registered so that peers can `Put` data into it or `Get` data out of it, using NCCL's `Put`/`Get` APIs. Creating the window is still collective - every rank in the group must call it, in the same order - and the Python entry points are experimental and underscore-prefixed (`dist._new_window`, `dist._supports_window`). Use a window when the access pattern is sparse; keep two-sided collectives when every rank participates anyway ([PyTorch 2.14 release blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)).

What the API looks like in the PyTorch 2.15.0 build used for this chapter (CPU only, so the gloo answer is the only one checked):

```python
# [torch] window_api.py - the one-sided API surface; gloo has no windows, NCCL is needed for the real thing
import os, inspect, torch.distributed as dist, torch._C._distributed_c10d as c10d
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
dist.init_process_group("gloo", rank=0, world_size=1)
print(inspect.signature(dist._new_window))
print("gloo supports windows:", dist._supports_window())
print("Window methods:", [m for m in dir(c10d.Window) if not m.startswith("_")])
```

```
(tensor: torch.Tensor | None = None, group: torch.distributed.distributed_c10d.ProcessGroup | None = None) -> torch._C._distributed_c10d.Window
gloo supports windows: False
Window methods: ['get_attr', 'map_remote_tensor', 'put', 'signal', 'tensor_deregister', 'tensor_register', 'wait_signal']
```

How to read this: `_new_window` takes an optional tensor to register and an optional group, and returns a `Window`; its docstring says that if you omit the tensor, the later `tensor_register` call is collective too. gloo answers `False`, so on a CPU you can inspect the API but not use it. The method names are what this build exposes (`put` writes into a peer's registered memory, `signal`/`wait_signal` let a peer learn that a write has landed); how they behave on NCCL is **untested here**, and as an experimental API it may change between releases.

### Collective communications

A collective involves either many senders and one receiver, one sender and many receivers, or many senders and many receivers. In PyTorch each process is usually tied to one accelerator, and collectives run over a *process group*: the set of ranks that take part in the call. The default group is all ranks (`dist.init_process_group` creates it); `dist.new_group(ranks)` makes smaller ones, and one process can belong to several groups - for example a tensor-parallel group inside the node and a data-parallel group across nodes.

#### Broadcast

Copy one rank's buffer (the *root's*) to every other rank - for example to send the initial weights from rank 0 to everyone.

![broadcast](images/collective-broadcast-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![broadcast](images/collective-broadcast-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

`dist.broadcast(tensor, src, group)` copies `tensor` from rank `src` into `tensor` on all other ranks ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast)). The algorithms are in [Broadcast: binomial tree vs scatter+all-gather vs pipelined chain](#broadcast-binomial-tree-vs-scatterall-gather-vs-pipelined-chain).

#### Gather

Every rank sends its buffer to one destination rank, which ends up with all of them side by side.

![gather](images/collective-gather-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![gather](images/collective-gather-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

`dist.gather(tensor, gather_list, dst, group)` copies one `tensor` from every rank in the group into `gather_list` on rank `dst`; other ranks pass `None` ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.gather)).

#### All-gather

Like gather, but every rank ends up with everyone's buffer.

![all-gather](images/collective-all-gather-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![all-gather](images/collective-all-gather-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

ZeRO (DeepSpeed) and FSDP use it to reassemble the sharded weights of a layer just before its `forward` and `backward` ([ZeRO and FSDP](../training/model-parallelism/derivations.md#zero-and-fsdp)).

`dist.all_gather(tensor_list, tensor, group)` copies `tensor` from every rank into `tensor_list` on every rank ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)); `dist.all_gather_into_tensor` is the same with one flat output tensor, used in [Ring all-gather](#ring-all-gather).

#### Reduce

Combine every rank's buffer elementwise with an operation (sum, max, ...) and put the result on one destination rank.

![reduce](images/collective-reduce-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![reduce](images/collective-reduce-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

`dist.reduce(tensor, dst, op, group)` applies `op` across every rank's `tensor` and stores the result in `tensor` on rank `dst` ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.reduce)). The operations PyTorch supports include `AVG`, `SUM`, `PRODUCT`, `MIN`, `MAX`, `BAND`, `BOR` and `BXOR` (bitwise and/or/xor) - [full list](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp); not every backend supports every op on every dtype, so check the table on that page for yours. The script below shows a trap: after a `reduce`, the buffer on the *other* ranks is not their input any more, it holds whatever partial sum the algorithm left there.

#### All-reduce

Same as reduce, but every rank gets the result.

![all-reduce](images/collective-all-reduce-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![all-reduce](images/collective-all-reduce-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

[DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) uses it to sum the gradients of all ranks ([Data parallelism (DDP)](../training/model-parallelism/derivations.md#data-parallelism-ddp)). `dist.all_reduce(tensor, op, group)` is `reduce` with the result stored on every rank ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)).

#### Scatter

The reverse of gather: the root holds a list of pieces and rank `i` receives piece `i`.

![scatter](images/collective-scatter-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![scatter](images/collective-scatter-2.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

`dist.scatter(tensor, scatter_list, src, group)` copies `scatter_list[i]` from rank `src` into `tensor` on rank `i` ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.scatter)).

#### Reduce-scatter

Reduce, but instead of one rank getting the whole result, the result is split into $p$ shards and rank $i$ gets shard $i$.

![reduce-scatter](images/collective-reduce-scatter.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

ZeRO and FSDP use it for the gradients: each rank only needs the reduced gradient of the shard of parameters it owns. It is not a cheaper drop-in for [all-reduce](#all-reduce) - each rank ends up with only its own reduced shard rather than the full result. That is exactly what ZeRO wants, and it's why it costs half the data movement: a ring all-reduce *is* a reduce-scatter followed by an all-gather ([Ring all-reduce](#ring-all-reduce)), so dropping the all-gather phase halves the bytes on the wire.

`dist.reduce_scatter(output, input_list, op, group, async_op)` reduces, then scatters a list of tensors to all ranks in the group ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter)); `dist.reduce_scatter_tensor` is the flat-tensor version.

#### All-to-all

Every rank sends a different piece to every other rank: rank $r$'s $j$-th piece goes to rank $j$, so the data is transposed across the ranks.

![all-to-all](images/collective-all-to-all-1.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

![all-to-all](images/collective-all-to-all.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

DeepSpeed-Ulysses sequence parallelism uses it to switch attention from sequence-sharded to head-sharded and back ([Context parallelism (CP)](../training/model-parallelism/derivations.md#context-parallelism-cp)), and MoE [expert parallelism](../training/model-parallelism/derivations.md#expert-parallelism) uses it to send each token to the GPU holding its expert and to bring the result back.

`dist.all_to_all(output_tensor_list, input_tensor_list, group)` sends `input_tensor_list[j]` to rank `j` and fills `output_tensor_list[i]` with what rank `i` sent ([doc](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all)); `dist.all_to_all_single` is the flat-tensor version, and the one gloo supports.

#### All of them on four CPU processes

The script starts 4 processes on one machine with the gloo backend (the launch mechanics are explained in [Launching a process group on one machine](#launching-a-process-group-on-one-machine)). Rank $r$ starts with the tensor `[10r+1, 10r+2]`, so you can tell from any value where it came from:

```python
# [torch] patterns.py - every pattern in this section on 4 CPU ranks (gloo)
# run: python patterns.py
import os, torch, torch.distributed as dist, torch.multiprocessing as mp

def show(rank, name, t):
    # rank-ordered printing: everyone waits its turn at a barrier
    for r in range(dist.get_world_size()):
        if r == rank: print(f"{name:15s} rank {rank}: {t}", flush=True)
        dist.barrier()

def worker(rank, p):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    dist.init_process_group("gloo", rank=rank, world_size=p)
    x = torch.tensor([10 * rank + 1, 10 * rank + 2])          # (2,) rank r holds [10r+1, 10r+2]

    # point-to-point: a ring of isend/irecv, rank r -> r+1 (non-blocking, then wait)
    got = torch.empty_like(x)
    reqs = [dist.isend(x, (rank + 1) % p), dist.irecv(got, (rank - 1) % p)]
    for q in reqs: q.wait()
    show(rank, "isend/irecv", got)

    t = x.clone(); dist.broadcast(t, src=0);                      show(rank, "broadcast", t)
    out = [torch.empty_like(x) for _ in range(p)] if rank == 0 else None
    dist.gather(x, out, dst=0);                                   show(rank, "gather", out)
    out = [torch.empty_like(x) for _ in range(p)]
    dist.all_gather(out, x);                                      show(rank, "all_gather", out)
    t = x.clone(); dist.reduce(t, dst=0, op=dist.ReduceOp.SUM);   show(rank, "reduce (dst 0)", t)
    t = x.clone(); dist.all_reduce(t, op=dist.ReduceOp.MAX);      show(rank, "all_reduce MAX", t)
    t = torch.empty(1, dtype=x.dtype)
    chunks = [torch.tensor([100 + i]) for i in range(p)] if rank == 0 else None
    dist.scatter(t, chunks, src=0);                               show(rank, "scatter", t)
    full = torch.arange(p) + 10 * rank                            # (p,) one element per destination
    t = torch.empty(1, dtype=full.dtype)
    dist.reduce_scatter_tensor(t, full);                          show(rank, "reduce_scatter", t)
    t = torch.empty_like(full)
    dist.all_to_all_single(t, full);                              show(rank, "all_to_all", t)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(4,), nprocs=4)
```

Output (PyTorch 2.15.0, gloo, Apple M4; PyTorch warnings removed):

```
isend/irecv     rank 0: tensor([31, 32])
isend/irecv     rank 1: tensor([1, 2])
isend/irecv     rank 2: tensor([11, 12])
isend/irecv     rank 3: tensor([21, 22])
broadcast       rank 0: tensor([1, 2])
broadcast       rank 1: tensor([1, 2])
broadcast       rank 2: tensor([1, 2])
broadcast       rank 3: tensor([1, 2])
gather          rank 0: [tensor([1, 2]), tensor([11, 12]), tensor([21, 22]), tensor([31, 32])]
gather          rank 1: None
gather          rank 2: None
gather          rank 3: None
all_gather      rank 0: [tensor([1, 2]), tensor([11, 12]), tensor([21, 22]), tensor([31, 32])]
all_gather      rank 1: [tensor([1, 2]), tensor([11, 12]), tensor([21, 22]), tensor([31, 32])]
all_gather      rank 2: [tensor([1, 2]), tensor([11, 12]), tensor([21, 22]), tensor([31, 32])]
all_gather      rank 3: [tensor([1, 2]), tensor([11, 12]), tensor([21, 22]), tensor([31, 32])]
reduce (dst 0)  rank 0: tensor([64, 68])
reduce (dst 0)  rank 1: tensor([63, 66])
reduce (dst 0)  rank 2: tensor([52, 54])
reduce (dst 0)  rank 3: tensor([31, 32])
all_reduce MAX  rank 0: tensor([31, 32])
all_reduce MAX  rank 1: tensor([31, 32])
all_reduce MAX  rank 2: tensor([31, 32])
all_reduce MAX  rank 3: tensor([31, 32])
scatter         rank 0: tensor([100])
scatter         rank 1: tensor([101])
scatter         rank 2: tensor([102])
scatter         rank 3: tensor([103])
reduce_scatter  rank 0: tensor([60])
reduce_scatter  rank 1: tensor([64])
reduce_scatter  rank 2: tensor([68])
reduce_scatter  rank 3: tensor([72])
all_to_all      rank 0: tensor([ 0, 10, 20, 30])
all_to_all      rank 1: tensor([ 1, 11, 21, 31])
all_to_all      rank 2: tensor([ 2, 12, 22, 32])
all_to_all      rank 3: tensor([ 3, 13, 23, 33])
```

How to read this output, one block per pattern:

- `isend/irecv`: rank 1 got rank 0's `[1, 2]`, and rank 0 got rank 3's `[31, 32]` - each rank received from its left neighbour.
- `broadcast`: everyone now holds the root's `[1, 2]`.
- `gather`: only rank 0 has the list, in rank order; the others passed `None` and still have `None`.
- `all_gather`: the same list everywhere.
- `reduce`: rank 0 has the true sum, $1+11+21+31 = 64$ and $2+12+22+32 = 68$. Ranks 1-3 hold leftover partial sums ($63 = 11+21+31$, $52 = 21+31$, and rank 3 its own input): gloo reduced *in place* along a chain toward rank 0. Never read the buffer of a non-destination rank after `reduce`.
- `all_reduce MAX`: every rank holds the elementwise maximum, rank 3's `[31, 32]`.
- `scatter`: rank $i$ got `100 + i`, the $i$-th element of rank 0's list.
- `reduce_scatter`: rank $r$ starts with `[10r, 10r+1, 10r+2, 10r+3]`; rank $i$ ends with the sum of element $i$ over all ranks, $60 + 4i$ (e.g. rank 0: $0+10+20+30 = 60$).
- `all_to_all`: rank $j$ ends with element $j$ of every rank, `[j, 10+j, 20+j, 30+j]` - the transpose whose cost is derived in the algorithm section on all-to-all below.

The rest of the chapter is about how these are implemented and what they cost.

## The alpha-beta model

### Definitions

A *cost model* is a formula that predicts how long an operation takes from a few hardware parameters. It's deliberately crude: the goal is to be right to within a factor of ~2 with two numbers, so that you can reason about designs before you have the hardware.

The Hockney / alpha-beta model charges a point-to-point message (one sender, one receiver) of $m$ bytes:

$$
T(m) = \alpha + m \cdot \beta
$$

Two separate things cost time when you send a message. There's a fixed cost you pay even for a 1-byte message - launching the operation, the NIC noticing it, the first bit crossing the switches, the receiver noticing it arrived. That's *latency*, $\alpha$. Then there's a cost proportional to size, because the link can only push so many bytes per second. That's the inverse of *bandwidth*, $\beta$. The model just adds them. It's the right first model because those two costs come from different hardware and you fix them in different ways: latency with fewer, larger messages or fewer hops; bandwidth with faster links or fewer bytes.

- $\alpha$ - per-message latency (seconds): software launch, NIC (network interface card, the adapter connecting a machine to the network) doorbell, switch hops, synchronization flags.
- $\beta$ - inverse bandwidth (seconds/byte) of the bottleneck link, $\beta = 1/B$.
- $\gamma$ - optional per-byte reduction cost (seconds/byte) for collectives that reduce.

Notation used throughout (a *rank* is one participating process, normally one per GPU; a *collective* is an operation that every rank in a group calls together, such as an *all-reduce*, which sums a buffer across all ranks and leaves the sum on every rank): $p$ ranks, $n$ bytes per rank (the input buffer size of an all-reduce), $N$ nodes, $G$ GPUs per node, $P = N \cdot G$.

Two derived quantities do most of the work:

- $n_{1/2} = \alpha_\text{total} / \beta_\text{eff}$ - the message size at which a collective reaches half of its asymptotic bandwidth. Below it you are latency-bound and a faster link doesn't help; above it you are bandwidth-bound and fewer calls don't help.
- The crossover size between two algorithms $A$ and $B$: $n^* = (\alpha_A - \alpha_B) / (\beta_B - \beta_A)$, where each $\alpha$ and $\beta$ is the collective-level total (not per message).

The single-message version, with an assumed $\alpha = 5\ \mu\text{s}$ and a 50GBps link (one 400Gbps NIC):

```python
# alpha_beta.py - one message: T(m) = alpha + m*beta, and the effective bandwidth m/T
alpha, B = 5e-6, 50e9                 # assumed: 5us per message, 50GB/s link (one 400Gbps NIC)
beta = 1 / B
n_half = alpha / beta                 # size at which you get half the link rate
print(f"n_1/2 = alpha/beta = {n_half/1e3:.0f} KB")
for m in [8, 1_000, 64_000, 250_000, 1_000_000, 16_000_000, 1_000_000_000]:
    T = alpha + m * beta
    print(f"{m:>13,} B  T = {T*1e6:10.1f} us   effective {m/T/1e9:6.2f} GB/s  ({m/T/B:6.1%} of link)")
```

```
n_1/2 = alpha/beta = 250 KB
            8 B  T =        5.0 us   effective   0.00 GB/s  (  0.0% of link)
        1,000 B  T =        5.0 us   effective   0.20 GB/s  (  0.4% of link)
       64,000 B  T =        6.3 us   effective  10.19 GB/s  ( 20.4% of link)
      250,000 B  T =       10.0 us   effective  25.00 GB/s  ( 50.0% of link)
    1,000,000 B  T =       25.0 us   effective  40.00 GB/s  ( 80.0% of link)
   16,000,000 B  T =      325.0 us   effective  49.23 GB/s  ( 98.5% of link)
1,000,000,000 B  T =    20005.0 us   effective  49.99 GB/s  (100.0% of link)
```

That S-shaped curve - near zero, then a ramp around $n_{1/2}$, then flat at the link rate - is the shape of every bandwidth-vs-size plot from nccl-tests, and you'll measure it yourself in [Measuring collectives on a laptop](#measuring-collectives-on-a-laptop-with-gloo).

The model assumes full-duplex links (a link is a physical connection between two devices; full-duplex means it carries traffic in both directions at full speed simultaneously, so a rank can send and receive $m$ bytes simultaneously in time $\alpha + m \cdot \beta$), no contention, and homogeneous links. Each of those assumptions breaks somewhere in this chapter, and the breakage is usually what explains a disappointing measurement.

### Lower bounds

Before deriving algorithms, the bounds they are measured against:

- **Latency lower bound** for any collective that needs information from every rank to reach every rank: $\lceil \log_2 p \rceil \cdot \alpha$, since the number of ranks holding a given piece of information can at most double each round.
- **Bandwidth lower bound for all-reduce**: $2 \cdot (p-1)/p \cdot n \cdot \beta$. Sketch: think of the output as $p$ shards and assign each shard to the rank that finishes reducing it. A rank's data has to leave it twice over: its contributions to the $p-1$ shards it doesn't own ($(p-1)/p \cdot n$ bytes, a reduce-scatter), and - since the other ranks need the final result of the shard it owns, while what it sent earlier were only partial inputs - it must also receive the $p-1$ reduced shards it doesn't own ($(p-1)/p \cdot n$, an all-gather), with symmetric traffic going out. Counted per direction per rank that's $2 \cdot (p-1)/p \cdot n$; the rigorous version has to rule out cleverer schemes, which is what the paper does. Patarasuk and Yuan prove this bound and that a ring achieves it ([Bandwidth optimal all-reduce algorithms for clusters of workstations](https://doi.org/10.1016/j.jpdc.2008.09.002), JPDC 2009).
- **Reduce-scatter and all-gather**: $(p-1)/p \cdot n \cdot \beta$ each - every rank receives or sends $p-1$ shards of size $n/p$.
- **Broadcast**: $n \cdot \beta$ - the root must at least send $n$ bytes once, everyone must receive $n$.

These bounds are what `busbw` normalizes against, which is why they come first. See [algbw vs busbw](#algbw-vs-busbw).

A *lower bound* is a time no algorithm can beat on the given hardware, so it tells you how much room is left: if your measurement is at 90% of the bound, stop optimizing the algorithm. Note how the all-reduce bandwidth bound barely moves with $p$ - adding ranks doesn't add bytes per rank - while the latency bound keeps growing (in the code and output, RS = reduce-scatter and AG = all-gather, abbreviations used for the rest of the chapter):

```python
# lower_bounds.py - the floor any algorithm is measured against, p ranks, n bytes, link B
import math
alpha, B, n = 5e-6, 50e9, 1e9
for p in (2, 8, 64, 1024):
    lat = math.ceil(math.log2(p)) * alpha
    ar = 2 * (p - 1) / p * n / B
    rs = (p - 1) / p * n / B
    print(f"p={p:>4}: latency >= {lat*1e6:5.0f}us   all-reduce bw >= {ar*1e3:6.2f}ms   "
          f"RS/AG bw >= {rs*1e3:6.2f}ms   broadcast bw >= {n/B*1e3:.0f}ms")
```

```
p=   2: latency >=     5us   all-reduce bw >=  20.00ms   RS/AG bw >=  10.00ms   broadcast bw >= 20ms
p=   8: latency >=    15us   all-reduce bw >=  35.00ms   RS/AG bw >=  17.50ms   broadcast bw >= 20ms
p=  64: latency >=    30us   all-reduce bw >=  39.38ms   RS/AG bw >=  19.69ms   broadcast bw >= 20ms
p=1024: latency >=    50us   all-reduce bw >=  39.96ms   RS/AG bw >=  19.98ms   broadcast bw >= 20ms
```

## Deriving the classic algorithms

Each collective can be implemented by several algorithms, and libraries like NCCL switch between them per call, using internal heuristics, unless the user overrides them ([The tuning model](#the-tuning-model)). The canonical reference for everything in this section is Thakur, Rabenseifner and Gropp, [Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521) (IJHPCA 2005). The formulas below follow its conventions, with $n$ the per-rank buffer size.

### Ring reduce-scatter

Split the buffer into $p$ shards of $n/p$ bytes. At step $k$ ($k = 0, \ldots, p-2$), rank $r$ sends shard $(r-k) \bmod p$ to rank $r+1$, which adds it to its own copy of that shard. After $p-1$ steps, rank $r$ holds the fully reduced shard $(r+1) \bmod p$.

Each step moves $n/p$ bytes per rank, and all ranks send simultaneously on disjoint links:

$$
T_\text{RS,ring} = (p-1)\,\alpha + \frac{p-1}{p}\, n\beta + \frac{p-1}{p}\, n\gamma
$$

Why a ring: each rank talks only to two neighbours, so every link carries exactly one chunk per step in each direction, nothing is ever waiting on a busy link, and no rank is a hotspot. The price is $p-1$ sequential steps - the reason for everything else in this chapter.

The simulator below does both ring phases (reduce-scatter, then the [all-gather](#ring-all-gather)) on 4 pretend ranks. Each "rank" is a Python list of $p$ chunks (one number per chunk, to keep it readable), and every step is one round in which all ranks send at the same time:

```python
# ring_allreduce_sim.py - pure Python, no dependencies
import random

def ring_allreduce(data, verbose=True):
    """data[r] is rank r's buffer, a list of p chunks (one number per chunk here)."""
    p = len(data)
    buf = [list(d) for d in data]          # each rank's working copy
    sent = [0] * p                          # chunks sent per rank
    # Phase 1: reduce-scatter. Step k: rank r sends chunk (r-k) mod p to r+1, which ADDS.
    for k in range(p - 1):
        msgs = [(r, (r + 1) % p, (r - k) % p, buf[r][(r - k) % p]) for r in range(p)]
        for src, dst, c, val in msgs:       # all sends in a step happen "at once":
            buf[dst][c] += val              # msgs captured every value before any write
            sent[src] += 1
        if verbose:
            print(f"RS step {k}: " + "  ".join(f"r{s}->r{d} c{c}" for s, d, c, _ in msgs))
    if verbose:  # now rank r owns the fully reduced chunk (r+1) mod p
        print("after RS: " + "  ".join(f"r{r} owns c{(r+1)%p}={buf[r][(r+1)%p]}" for r in range(p)))
    # Phase 2: all-gather. Step k: rank r sends chunk (r+1-k) mod p to r+1, which OVERWRITES.
    for k in range(p - 1):
        msgs = [(r, (r + 1) % p, (r + 1 - k) % p, buf[r][(r + 1 - k) % p]) for r in range(p)]
        for src, dst, c, val in msgs:
            buf[dst][c] = val
            sent[src] += 1
        if verbose:
            print(f"AG step {k}: " + "  ".join(f"r{s}->r{d} c{c}" for s, d, c, _ in msgs))
    return buf, sent

if __name__ == "__main__":
    p = 4
    data = [[10 ** r * (c + 1) for c in range(p)] for r in range(p)]   # rank r: [1,2,3,4]*10^r
    for r in range(p): print(f"rank {r} start: {data[r]}")
    out, sent = ring_allreduce(data)
    expect = [sum(col) for col in zip(*data)]
    for r in range(p): print(f"rank {r} end:   {out[r]}")
    assert all(o == expect for o in out)
    # bytes check: each rank sent 2(p-1) chunks of n/p -> 2(p-1)/p * n
    print("chunks sent per rank:", sent, "= 2(p-1) =", 2 * (p - 1))
    for p in [2, 3, 5, 8, 16, 33]:          # random sizes, silent
        d = [[random.randint(-9, 9) for _ in range(p)] for _ in range(p)]
        o, s = ring_allreduce(d, verbose=False)
        assert all(x == [sum(c) for c in zip(*d)] for x in o) and s == [2 * (p - 1)] * p
    print("ok: p in {2,3,5,8,16,33}")
```

```
rank 0 start: [1, 2, 3, 4]
rank 1 start: [10, 20, 30, 40]
rank 2 start: [100, 200, 300, 400]
rank 3 start: [1000, 2000, 3000, 4000]
RS step 0: r0->r1 c0  r1->r2 c1  r2->r3 c2  r3->r0 c3
RS step 1: r0->r1 c3  r1->r2 c0  r2->r3 c1  r3->r0 c2
RS step 2: r0->r1 c2  r1->r2 c3  r2->r3 c0  r3->r0 c1
after RS: r0 owns c1=2222  r1 owns c2=3333  r2 owns c3=4444  r3 owns c0=1111
AG step 0: r0->r1 c1  r1->r2 c2  r2->r3 c3  r3->r0 c0
AG step 1: r0->r1 c0  r1->r2 c1  r2->r3 c2  r3->r0 c3
AG step 2: r0->r1 c3  r1->r2 c0  r2->r3 c1  r3->r0 c2
rank 0 end:   [1111, 2222, 3333, 4444]
rank 1 end:   [1111, 2222, 3333, 4444]
rank 2 end:   [1111, 2222, 3333, 4444]
rank 3 end:   [1111, 2222, 3333, 4444]
chunks sent per rank: [6, 6, 6, 6] = 2(p-1) = 6
ok: p in {2,3,5,8,16,33}
```

Read one column of the RS steps: chunk `c0` starts at rank 0, goes to rank 1 (step 0, sum $1+10$), then rank 2 (step 1, $+100$), then rank 3 (step 2, $+1000$), and ends up complete at rank 3 - the `r3 owns c0=1111` in the line after the RS steps. Every chunk takes the same $p-1$ hops, just starting at a different rank, which is why all links are busy at once.

### Ring all-gather

Same pattern, no reduction - at each step a rank forwards the shard it most recently received:

$$
T_\text{AG,ring} = (p-1)\,\alpha + \frac{p-1}{p}\, n\beta
$$

Here $n$ is the size of the gathered output on each rank, which is also how `nccl-tests` reports the size for all-gather.

In the simulator above, the AG phase is the `=` (overwrite) loop: the chunk a rank just finished is forwarded unchanged, and after $p-1$ hops every rank has every chunk. In PyTorch the all-gather with exactly this size convention is `all_gather_into_tensor`, where each rank contributes $n/p$ and receives $n$:

```python
# [torch] the shapes of all-gather and reduce-scatter; launch with 4 ranks as in
# "Measuring collectives on a laptop" below
p = dist.get_world_size()
shard = torch.full((1024,), float(dist.get_rank()))   # (n/p,)   what I own
full = torch.empty(1024 * p)                          # (n,)     what everyone gets
dist.all_gather_into_tensor(full, shard)              # ring AG: (p-1)/p * n bytes in, per rank
mine = torch.empty(1024)                              # (n/p,)
dist.reduce_scatter_tensor(mine, full)                # ring RS: (p-1)/p * n bytes out, per rank
```

Both moved $(p-1)/p \cdot n$ bytes per rank, half of an all-reduce. (Reminder from [Data parallel vs model parallel](../foundations/distributed-systems.md#data-parallel-vs-model-parallel): *DDP* - DistributedDataParallel - keeps a full copy of the model on every rank and all-reduces the gradients; *FSDP* / *ZeRO-3* keeps only a $1/p$ shard of parameters, gradients and optimizer state per rank, all-gathers each layer's weights just before using them and reduce-scatters the gradients. The full derivation is in [ZeRO and FSDP](../training/model-parallelism/derivations.md#zero-and-fsdp).) That's the arithmetic behind FSDP/ZeRO-3 "costing the same as DDP plus one extra all-gather": DDP does one all-reduce ($2 \cdot (p-1)/p \cdot n$), FSDP does an all-gather in forward, another in backward, and a reduce-scatter ($3 \cdot (p-1)/p \cdot n$), i.e. 1.5x the bytes.

### Ring all-reduce

All-reduce = reduce-scatter followed by all-gather:

$$
T_\text{AR,ring} = 2(p-1)\,\alpha + 2\,\frac{p-1}{p}\, n\beta + \frac{p-1}{p}\, n\gamma
$$

Ring all-reduce is done like the [ring broadcast](#ring-broadcast-in-nccls-notation): the message is split into many small messages, and each GPU sends a small message to the next GPU in parallel with the others. It needs twice as many steps as a broadcast, because each byte has to travel the ring twice - once to be reduced, once to be distributed - so the message crosses the wire twice (walked through by hand in [Ring all-reduce, step by step](../foundations/networking.md#ring-all-reduce-step-by-step)). The whole buffer is also first split into chunks, which NCCL processes one after another (see [Chunking and pipelining](#chunking-and-pipelining)). Here is the reduction of the first chunk travelling around the ring:

![ring all-reduce chunk movement](images/all-reduce-ring-chunk1.png)

([source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf))

Then the next chunk is done, and so on until every chunk is reduced:

![ring-based all-reduce chunk 2](images/all-reduce-ring-chunk2.png)

([source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf))

The bandwidth term hits the lower bound exactly, which is why the ring is the default for large messages. The latency term grows linearly in $p$, which is why it isn't the default at scale.

The $\gamma$ term on a GPU is a memory-bound elementwise add. A ring step reads the local shard and the received shard and writes the sum, so roughly 3 bytes of HBM (high-bandwidth memory, the GPU's own DRAM) traffic per byte on the wire. With H100 HBM3 at 3.35TBps against NVLink (NVIDIA's GPU-to-GPU link, much faster than PCIe) 4 at 450GBps per direction, the ratio is ~7.4x, so $\gamma$ is about $3/7.4 \approx 0.4$ of $\beta$ in the worst case - not negligible, but NCCL fuses it into the receive and it largely overlaps with the transfer. It becomes visible with low channel counts (fewer SMs doing the adds) and during heavy compute overlap (the HBM is shared).

```python
# [torch] gamma.py - what a CPU backend pays per byte to copy vs to reduce (single thread)
import time, torch
torch.set_num_threads(1)
n = 64 * 2**20 // 4                          # 64MiB of fp32
a, b, out = torch.ones(n), torch.ones(n), torch.empty(n)   # (n,) each

def bench(f, reps=20):
    f(); t0 = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t0) / reps

t_copy = bench(lambda: out.copy_(a))                 # read n*4 B, write n*4 B
t_add  = bench(lambda: torch.add(a, b, out=out))     # read 2*n*4 B, write n*4 B
nb = n * 4
print(f"copy 64MiB: {t_copy*1e3:6.2f} ms  -> {nb/t_copy/1e9:5.1f} GB/s of payload")
print(f"add  64MiB: {t_add*1e3:6.2f} ms  -> {nb/t_add/1e9:5.1f} GB/s of payload")
print(f"gamma (s/byte) ~ {t_add/nb:.2e}   vs  beta of a 400Gbps NIC = {1/50e9:.2e}")
```

```
copy 64MiB:   2.41 ms  ->  27.8 GB/s of payload
add  64MiB:   2.62 ms  ->  25.6 GB/s of payload
gamma (s/byte) ~ 3.90e-11   vs  beta of a 400Gbps NIC = 2.00e-11
```

On one M4 core, reducing costs about as much as copying, and $\gamma$ is about 2x the $\beta$ of a 400Gbps NIC - a CPU core couldn't keep up with that NIC's reductions. That's one reason GPU collectives do the reduction on the GPU, many SMs at once, next to the data. For the gloo runs [below](#measuring-collectives-on-a-laptop-with-gloo) it's irrelevant: they reach ~2GB/s, an order of magnitude below this $\gamma$, so their bottleneck is the socket path, not the adds.

### Broadcast: binomial tree vs scatter+all-gather vs pipelined chain

A *broadcast* copies one rank's (the *root's*) buffer to every other rank - used e.g. to send initial weights from rank 0. Three designs, with different trade-offs between number of steps and bytes per step:

- **Binomial tree**: $T = \lceil \log_2 p \rceil \cdot (\alpha + n \cdot \beta)$. Latency-optimal, but every round sends the whole buffer, so it's $\log_2 p$ times off the bandwidth bound.
- **Scatter + all-gather** (van de Geijn): $T = (\log_2 p + p - 1) \cdot \alpha + 2 \cdot (p-1)/p \cdot n \cdot \beta$. Within $2\times$ of the bandwidth bound for any $p$.
- **Pipelined chain** (NCCL's ring broadcast, see [Ring broadcast in NCCL's notation](#ring-broadcast-in-nccls-notation) below): split $n$ into $S$ chunks and stream them down a chain of $p$ ranks. The last chunk arrives after $p-2+S$ chunk steps:

$$
T_\text{bcast,chain}(S) = (p - 2 + S)\left(\alpha + \frac{n\beta}{S}\right)
$$

Minimizing over $S$: $dT/dS = 0$ gives $S^* = \sqrt{(p-2) \cdot n \cdot \beta / \alpha}$. For $p = 128$, $n = 1\ \text{GB}$, $\beta = 1/(50\ \text{GBps})$, $\alpha = 14\ \mu\text{s}$ (assumed), $S^* \approx 424$ chunks of ~2.4MB each, and $T \approx 33.6\ \text{ms}$ against the $n \cdot \beta = 20\ \text{ms}$ bound, versus $140\ \text{ms}$ for the binomial tree. The gap to the bound is the `sqrt` fill cost, $\approx 2 \cdot \sqrt{(p-2) \cdot \alpha \cdot n \cdot \beta}$ for large $S$. This is the same pipelining that makes NCCL's ring and tree reach their asymptotic bandwidth - chunking is not an optimization on top of the algorithm, it is what makes the bandwidth term of the algorithm achievable at all.

#### Ring broadcast in NCCL's notation

NVIDIA's original NCCL slides present the same chain with the latency term dropped, which makes the bandwidth argument easy to see. Given:

- $N$: bytes to broadcast
- $B$: bandwidth of each link
- $k$: number of GPUs

A naive broadcast sends the whole message one hop at a time: each step takes $N/B$ (the time to push the whole message over one link), and the message needs $k-1$ hops to reach the last GPU, so the total is $(k-1) \cdot N/B$.

Here is how a ring-based broadcast is performed:

![ring-based broadcast](images/broadcast-ring.png)
[source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

The ring algorithm splits $N$ into $S$ messages. At each step $N/(S \cdot B)$ is sent, $S$ times less than the naive algorithm sends per step. The total time to broadcast $N$ bytes to $k$ GPUs is

$$
S \cdot \frac{N}{S \cdot B} + (k-2) \cdot \frac{N}{S \cdot B} = \frac{N \cdot (S + k - 2)}{S \cdot B}
$$

How to read this: the first term is the $S$ pieces leaving the source one after another, and the second is the $k-2$ further hops the last piece still needs to reach the far end of the ring, while every earlier piece is already moving ahead of it. If the pieces are small enough that $S \gg k$, then $S + k - 2 \approx S$ and the total is about $N/B$ - the time to send the message over a single link, however many GPUs there are. This is $T_\text{bcast,chain}$ above with $\alpha = 0$, $p = k$ and $\beta = 1/B$; the $\alpha$ term is what stops you from making $S$ arbitrarily large.

*Pipelining* means overlapping successive pieces of work: while rank 1 forwards chunk 1 to rank 2, rank 0 is already sending chunk 2 to rank 1, like an assembly line. The simulator shows both broadcasts, and checks the $p-2+S$ step count the formula relies on:

```python
# bcast_sim.py - binomial-tree broadcast and pipelined-chain broadcast, step by step
import math

def binomial_bcast(p, root=0):
    have = {root}; rnd = 0
    while len(have) < p:
        sends = [(r, r + 2 ** rnd) for r in sorted(have) if r + 2 ** rnd < p]  # (root=0)
        print(f"round {rnd}: " + "  ".join(f"r{a}->r{b}" for a, b in sends) + "   (whole buffer each)")
        have |= {b for _, b in sends}; rnd += 1
    assert rnd == math.ceil(math.log2(p))
    return rnd

def chain_bcast(p, S, verbose=False):
    """rank 0 -> 1 -> ... -> p-1, buffer split into S chunks. Returns number of chunk-steps."""
    got = [0] * p; got[0] = S; step = 0
    while got[-1] < S:
        # every rank i>0 that is behind its predecessor receives one more chunk, simultaneously
        moves = [i for i in range(1, p) if got[i] < got[i - 1]]
        for i in moves: got[i] += 1
        step += 1
        if verbose: print(f"step {step}: chunks held = {got}")
    return step

if __name__ == "__main__":
    print("binomial tree, p=8:"); binomial_bcast(8)
    print("pipelined chain, p=4, S=3 chunks:")
    steps = chain_bcast(4, 3, verbose=True); print("steps =", steps, " formula p-2+S =", 4 - 2 + 3)
    for p, S in [(8, 1), (8, 16), (128, 424)]:
        assert chain_bcast(p, S) == p - 2 + S
    print("ok: chain takes p-2+S chunk-steps")
```

```
binomial tree, p=8:
round 0: r0->r1   (whole buffer each)
round 1: r0->r2  r1->r3   (whole buffer each)
round 2: r0->r4  r1->r5  r2->r6  r3->r7   (whole buffer each)
pipelined chain, p=4, S=3 chunks:
step 1: chunks held = [3, 1, 0, 0]
step 2: chunks held = [3, 2, 1, 0]
step 3: chunks held = [3, 3, 2, 1]
step 4: chunks held = [3, 3, 3, 2]
step 5: chunks held = [3, 3, 3, 3]
steps = 5  formula p-2+S = 5
ok: chain takes p-2+S chunk-steps
```

And the numbers in the paragraph above:

```python
import math
p, n, beta, alpha = 128, 1e9, 1 / 50e9, 14e-6                 # alpha assumed
S = math.sqrt((p - 2) * n * beta / alpha)                       # optimal chunk count
T = (p - 2 + S) * (alpha + n * beta / S)
print(f"S* = {S:.0f} chunks of {n/S/1e6:.2f}MB   T = {T*1e3:.1f}ms   bound n*beta = {n*beta*1e3:.0f}ms"
      f"   binomial = {math.log2(p)*(alpha + n*beta)*1e3:.0f}ms")
# S* = 424 chunks of 2.36MB   T = 33.6ms   bound n*beta = 20ms   binomial = 140ms
```

### Recursive halving and doubling

For power-of-two $p$:

- **Recursive halving reduce-scatter**: at round $k$, exchange half of your remaining data with the partner at distance $p/2^{k+1}$ and reduce. Round $k$ moves $n/2^{k+1}$ bytes, total $\log_2 p$ rounds:

$$
T_\text{RS,rh} = \log_2(p)\,\alpha + \frac{p-1}{p}\, n\beta + \frac{p-1}{p}\, n\gamma
$$

- **Recursive doubling all-gather**: the mirror image:

$$
T_\text{AG,rd} = \log_2(p)\,\alpha + \frac{p-1}{p}\, n\beta
$$

- **Rabenseifner all-reduce** = halving RS + doubling AG: $2 \cdot \log_2(p) \cdot \alpha + 2 \cdot (p-1)/p \cdot n \cdot \beta + (p-1)/p \cdot n \cdot \gamma$. Bandwidth-optimal *and* logarithmic latency - on paper it dominates the ring.
- **Recursive doubling all-reduce** (exchange the full buffer each round): $\log_2(p) \cdot (\alpha + n \cdot \beta + n \cdot \gamma)$. Best for tiny messages.

The trick behind all of these is *pairing by bit*: in round $k$ rank $r$ talks to rank $r \mathbin{\text{XOR}} 2^k$ (the rank whose number differs in exactly bit $k$). After round 0 each pair of ranks has the same information, after round 1 each group of 4, and so on - the group size doubles every round, which is what hits the $\log_2 p$ latency bound. Halving goes the other way: start with the farthest partner and halve the data each round. Simulated on 8 ranks with 8 chunks each:

```python
# rhd_sim.py - recursive doubling and Rabenseifner (halving RS + doubling AG), power-of-two p
def recursive_doubling_allreduce(data, verbose=True):
    p = len(data); buf = [list(d) for d in data]; rounds = 0; dist = 1
    while dist < p:                                  # log2(p) rounds
        new = [[a + b for a, b in zip(buf[r], buf[r ^ dist])] for r in range(p)]  # swap FULL buffer
        if verbose: print(f"RD round {rounds}: partner = r XOR {dist}; each rank sends all {len(buf[0])} chunks")
        buf, dist, rounds = new, dist * 2, rounds + 1
    return buf

def rabenseifner_allreduce(data, verbose=True):
    p = len(data); n = len(data[0]); buf = [list(d) for d in data]
    lo, hi = [0] * p, [n] * p                        # each rank's live window [lo, hi)
    sent = [0] * p
    dist = p // 2
    while dist >= 1:                                 # reduce-scatter by recursive halving
        msgs = []
        for r in range(p):
            mid = (lo[r] + hi[r]) // 2
            keep = (lo[r], mid) if r & dist == 0 else (mid, hi[r])   # lower rank keeps lower half
            give = (mid, hi[r]) if r & dist == 0 else (lo[r], mid)
            msgs.append((r, r ^ dist, give, keep))
        snap = [list(b) for b in buf]
        for src, dst, (g0, g1), _ in msgs:
            for i in range(g0, g1): buf[dst][i] += snap[src][i]
            sent[src] += g1 - g0
        for r, _, _, (k0, k1) in msgs: lo[r], hi[r] = k0, k1
        if verbose: print(f"RS-halving dist={dist}: each rank sends {msgs[0][2][1]-msgs[0][2][0]} chunks to r XOR {dist}")
        dist //= 2
    if verbose: print("after RS: " + "  ".join(f"r{r} owns [{lo[r]},{hi[r]})" for r in range(p)))
    dist = 1
    while dist < p:                                  # all-gather by recursive doubling
        snap = [list(b) for b in buf]; sl, sh = list(lo), list(hi)
        for r in range(p):
            q = r ^ dist
            for i in range(sl[q], sh[q]): buf[r][i] = snap[q][i]
            sent[q] += sh[q] - sl[q]
            lo[r], hi[r] = min(sl[r], sl[q]), max(sh[r], sh[q])
        if verbose: print(f"AG-doubling dist={dist}: each rank sends {sh[0]-sl[0]} chunks to r XOR {dist}")
        dist *= 2
    return buf, sent

if __name__ == "__main__":
    import random
    p = 8; n = 8                                     # 8 ranks, 8 chunks per rank
    data = [[random.randint(0, 9) for _ in range(n)] for _ in range(p)]
    expect = [sum(c) for c in zip(*data)]
    out = recursive_doubling_allreduce(data); assert all(o == expect for o in out)
    out, sent = rabenseifner_allreduce(data); assert all(o == expect for o in out)
    print("chunks sent per rank:", sent, "  2(p-1)/p*n =", 2 * (p - 1) * n // p)
    for p in [2, 4, 16, 64]:
        d = [[random.randint(-9, 9) for _ in range(p)] for _ in range(p)]; e = [sum(c) for c in zip(*d)]
        assert all(o == e for o in recursive_doubling_allreduce(d, False))
        o, s = rabenseifner_allreduce(d, False); assert all(x == e for x in o) and s == [2 * (p - 1)] * p
    print("ok: p in {2,4,8,16,64}")
```

```
RD round 0: partner = r XOR 1; each rank sends all 8 chunks
RD round 1: partner = r XOR 2; each rank sends all 8 chunks
RD round 2: partner = r XOR 4; each rank sends all 8 chunks
RS-halving dist=4: each rank sends 4 chunks to r XOR 4
RS-halving dist=2: each rank sends 2 chunks to r XOR 2
RS-halving dist=1: each rank sends 1 chunks to r XOR 1
after RS: r0 owns [0,1)  r1 owns [1,2)  r2 owns [2,3)  r3 owns [3,4)  r4 owns [4,5)  r5 owns [5,6)  r6 owns [6,7)  r7 owns [7,8)
AG-doubling dist=1: each rank sends 1 chunks to r XOR 1
AG-doubling dist=2: each rank sends 2 chunks to r XOR 2
AG-doubling dist=4: each rank sends 4 chunks to r XOR 4
chunks sent per rank: [14, 14, 14, 14, 14, 14, 14, 14]   2(p-1)/p*n = 14
ok: p in {2,4,8,16,64}
```

Recursive doubling sent $3 \times 8 = 24$ chunks per rank in 3 rounds; Rabenseifner sent $4+2+1+1+2+4 = 14$ in 6 rounds; the ring would send 14 in 14 rounds. That's the whole latency/bandwidth trade-off in one line of output. Notice also *which* rounds are expensive: the 4-chunk exchanges are with `r XOR 4`, the farthest partner - the point of the next paragraph.

Why NCCL doesn't use Rabenseifner as its main algorithm: the partner at round $k$ is at distance $p/2^{k+1}$, so on a hierarchical topology the first rounds - the ones moving the most data - cross the slowest links, and on a fat-tree (the usual cluster network: nodes plug into *leaf* switches, and leaves are joined through a layer of *spine* switches) those long-distance exchanges collide in the spine. A ring keeps every transfer on a nearest-neighbour link that NCCL picked to be fast. NCCL 2.23 did add PAT (Parallel Aggregated Trees), a Bruck-like logarithmic algorithm (Bruck's algorithm does an all-gather in $\lceil \log_2 p \rceil$ rounds, each rank sending to the rank $2^k$ places away in round $k$, like recursive doubling but for any $p$) for all-gather and reduce-scatter, but restricted to one GPU per node ([NVIDIA blog](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/)).

### All-to-all

In an *all-to-all*, every rank sends a different piece of data to every other rank - it's a transpose of a $p \times p$ matrix of blocks, with row $r$ living on rank $r$. Mixture-of-experts layers use it to send each token to the GPU holding its expert ([Mixture of Experts](../models/architectures.md#mixture-of-experts); spreading the experts over GPUs is called *expert parallelism*, EP, derived in [Expert parallelism](../training/model-parallelism/derivations.md#expert-parallelism)). Each rank holds $p$ blocks of $n/p$ bytes, block $j$ destined for rank $j$ ($n$ is the per-rank send buffer).

- **Pairwise exchange**: $p-1$ rounds, round $k$ sends to $r+k$ and receives from $r-k$:

$$
T_\text{A2A,pairwise} = (p-1)\,\alpha + \frac{p-1}{p}\, n\beta
$$

- **Bruck**: $\log_2 p$ rounds, each sending half the buffer: $\log_2(p) \cdot \alpha + (n/2) \cdot \log_2(p) \cdot \beta$. Wins only when $\alpha$ dominates.

Unlike the other collectives, all-to-all has no locality to exploit - the $(p-1)/p$ term is the fraction of the buffer that has to leave the rank, and on a multi-node job almost all of it has to leave the node. For $P = 1024$ GPUs over $N = 128$ nodes, $(P-G)/P = 99.2\%$ of each rank's buffer crosses the scale-out network, so at 50GBps per GPU NIC the bound is $0.992 \cdot n / 50\ \text{GBps} \approx 19.8\ \text{ms}$ per GB. NVLink bandwidth is irrelevant to it, and the network's bisection bandwidth is the real denominator: with an oversubscription of $o{:}1$ above the leaf switches, any traffic that crosses the spine gets $1/o$ of the NIC rate. All-to-all is the collective that exposes oversubscription and incast; all-reduce mostly doesn't, because the ring keeps traffic on a few fixed paths. (*Oversubscription* is a switch tier having less uplink capacity than downlink capacity; *incast* is many senders converging on one receiver's link at once, so their packets queue in the switch in front of it.)

```python
# a2a_sim.py - pairwise-exchange all-to-all: round k, rank r sends block (r+k) to rank r+k
def pairwise_a2a(send, verbose=True):
    p = len(send); recv = [[None] * p for _ in range(p)]
    for r in range(p): recv[r][r] = send[r][r]            # own block never moves
    for k in range(1, p):                                 # p-1 rounds
        pairs = [(r, (r + k) % p) for r in range(p)]
        for src, dst in pairs: recv[dst][src] = send[src][dst]
        if verbose: print(f"round {k}: " + "  ".join(f"r{s}->r{d}" for s, d in pairs))
    return recv

if __name__ == "__main__":
    p = 4
    send = [[f"{r}{j}" for j in range(p)] for r in range(p)]   # block "rj" = from rank r, for rank j
    recv = pairwise_a2a(send)
    for r in range(p): print(f"rank {r}: sent {send[r]}  got {recv[r]}")
    assert all(recv[j][r] == send[r][j] for r in range(p) for j in range(p))   # a transpose
    # per rank: sends p-1 of its p blocks -> (p-1)/p of the buffer leaves the rank
    P, G = 1024, 8
    print(f"fraction of buffer leaving the node, P={P}, G={G}: {(P - G) / P:.3f}")
```

```
round 1: r0->r1  r1->r2  r2->r3  r3->r0
round 2: r0->r2  r1->r3  r2->r0  r3->r1
round 3: r0->r3  r1->r0  r2->r1  r3->r2
rank 0: sent ['00', '01', '02', '03']  got ['00', '10', '20', '30']
rank 1: sent ['10', '11', '12', '13']  got ['01', '11', '21', '31']
rank 2: sent ['20', '21', '22', '23']  got ['02', '12', '22', '32']
rank 3: sent ['30', '31', '32', '33']  got ['03', '13', '23', '33']
fraction of buffer leaving the node, P=1024, G=8: 0.992
```

Every round pairs each rank with a *different* partner, so unlike the ring the traffic pattern sweeps across the whole network: in round $k$ on a multi-node job, almost every flow crosses a node boundary, and many cross the spine. The real call in PyTorch is `dist.all_to_all_single(out, inp)` with `inp` and `out` of shape `(p * block,)`; gloo supports it on CPU (checked with 4 ranks on the M4).

### Double binary tree

A single pipelined binary tree for all-reduce (reduce up, broadcast down) has logarithmic latency, but a leaf only ever sends during the reduce and only ever receives during the broadcast, so half the ranks use half their links. Sanders, Speck and Träff ([Two-tree algorithms for full bandwidth broadcast, reduction and scan](https://doi.org/10.1016/j.parco.2009.09.001), Parallel Computing 2009) fix this with two trees over the same ranks such that each rank is interior in at most one of them and the two trees each carry half the data. NCCL adopted this in 2.4 ([Massively Scale Your Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/), Jeaugey 2019).

Why a tree at all: a ring passes the message from neighbour to neighbour, so the number of steps grows with the number of GPUs $k$, and the base latency grows with it. A tree fans the message out instead, so the hop count grows with $\log k$. That is why the two swap places as you scale: ring keeps the higher peak bandwidth, tree the lower base latency, and for a fixed message size a large enough $k$ makes tree the faster of the two ([The ring/tree crossover](#the-ringtree-crossover) puts numbers on it). A single binary tree would waste half the machines: half the nodes are leaves, and a leaf only ever receives in the broadcast half, so its outbound bandwidth never contributes. The *double binary tree* builds two trees over the same GPUs, arranged so that no node is an interior node in both and at most one node is a leaf in both. Both trees carry traffic at the same time, which recovers the full bandwidth. The second tree is the first mirrored when the node count is even, or shifted by one position when it is odd.

Two complementary binary trees - every rank is a leaf in one and an interior node in the other:

![NCCL double binary tree](images/nccl-double-binary-tree.png)

([source](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/))

NVIDIA's measured small-message latency, tree vs ring, as the node count grows:

![NCCL tree vs ring latency on Summit](images/nccl-tree-vs-ring-latency-summit.png)

([source](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/))

Bandwidth: count what one rank moves. In the tree where it is interior it receives $n/2$ from each of 2 children and sends $n/2$ to its parent during the reduce, then receives $n/2$ from its parent and sends $n/2$ to each child during the broadcast. In the tree where it's a leaf it sends $n/2$ up and receives $n/2$ down. Totals per direction:

$$
\begin{aligned}
\text{sent} &= \underbrace{n/2 + n}_\text{interior} + \underbrace{n/2}_\text{leaf} = 2n \\
\text{received} &= \underbrace{n + n/2}_\text{interior} + \underbrace{n/2}_\text{leaf} = 2n
\end{aligned}
$$

So with all of a rank's links sharing one per-direction bandwidth $B$ (which is the case for the NIC of a node), $T_\text{bw} \approx 2n/B$ - the same as the ring's $2 \cdot (p-1)/p \cdot n/B$ up to the $(p-1)/p$ factor. The latency term is the pipeline depth, $\sim 2 \cdot \log_2(p)$ hops (up then down):

$$
T_\text{AR,dbtree} \approx 2\log_2(p)\,\alpha + 2n\beta \quad (+\ \text{pipeline fill})
$$

A *binary tree* here is just a parent pointer per rank, with at most two children each. The construction below is the one I read in NCCL's `src/graph/trees.cc` (tree 0 from the lowest set bit of the rank; tree 1 is tree 0 mirrored, $r \to p-1-r$, for even $p$ and shifted by one for odd $p$) - the property check doesn't depend on that reading being exact, it checks the invariant directly. The simulator then runs an all-reduce with half the buffer on each tree and counts elements per rank:

```python
# dbtree_sim.py - double binary tree all-reduce, pure Python
def btree_parent(r, p):
    """Binary tree over ranks 0..p-1 from the lowest set bit of r. Rank 0 is the root and
    has one child (the largest power of two < p); odd ranks are leaves."""
    if r == 0: return -1
    bit = r & -r                                  # lowest set bit of r
    up = (r ^ bit) | (bit << 1)                   # parent one level up
    if up >= p: up = r ^ bit                      # clipped at the right edge
    return up

def double_tree(p):
    t0 = [btree_parent(r, p) for r in range(p)]
    if p % 2 == 0:                                # even p: mirror ranks  r -> p-1-r
        m = lambda r: p - 1 - r
    else:                                         # odd p: shift ranks by one
        m = lambda r: (r + 1) % p
    t1 = [-1] * p
    for r in range(p):                            # tree 1 = tree 0 with relabelled ranks
        par = t0[r]
        t1[m(r)] = -1 if par == -1 else m(par)
    return t0, t1

def children(parent):
    ch = [[] for _ in parent]
    for r, u in enumerate(parent):
        if u >= 0: ch[u].append(r)
    return ch

def tree_allreduce(data, parent, sent, recv):
    """reduce up to the root, then broadcast down. data[r] = this tree's half of rank r's buffer."""
    p = len(data); ch = children(parent); root = parent.index(-1)
    acc = [list(d) for d in data]
    def reduce(r):                                # post-order: children first
        for c in ch[r]:
            reduce(c)
            acc[r] = [a + b for a, b in zip(acc[r], acc[c])]
            sent[c] += len(acc[c]); recv[r] += len(acc[c])
    reduce(root)
    out = [None] * p
    def bcast(r, val):
        out[r] = val
        for c in ch[r]:
            sent[r] += len(val); recv[c] += len(val)
            bcast(c, val)
    bcast(root, acc[root])
    return out

if __name__ == "__main__":
    import random
    p = 8
    t0, t1 = double_tree(p)
    for name, t in (("tree0", t0), ("tree1", t1)):
        ch = children(t)
        print(name, "parent:", t, " interior:", [r for r in range(p) if ch[r]])
    for p in range(2, 1025):                       # property check
        t0, t1 = double_tree(p); c0, c1 = children(t0), children(t1)
        # the property that matters for bandwidth: at most 2 children in total, so a rank
        # receives at most 2 half-buffers during the reduce and sends at most 2 in the broadcast
        assert all(len(c0[r]) + len(c1[r]) <= 2 for r in range(p)), p
        if p % 2 == 0:                            # even p: interior in at most one tree
            assert all(not (c0[r] and c1[r]) for r in range(p)), p
    print("ok: <= 2 children per rank across both trees, p = 2..1024")
    p, n = 8, 16                                  # 8 ranks, buffer of 16 elements
    data = [[random.randint(0, 9) for _ in range(n)] for _ in range(p)]
    t0, t1 = double_tree(p); sent, recv = [0] * p, [0] * p
    h0 = tree_allreduce([d[: n // 2] for d in data], t0, sent, recv)  # first half on tree 0
    h1 = tree_allreduce([d[n // 2:] for d in data], t1, sent, recv)   # second half on tree 1
    expect = [sum(c) for c in zip(*data)]
    assert all(h0[r] + h1[r] == expect for r in range(p))
    print("elements sent per rank:", sent, " received:", recv, " (2n =", 2 * n, ")")
```

```
tree0 parent: [-1, 2, 4, 2, 0, 6, 4, 6]  interior: [0, 2, 4, 6]
tree1 parent: [1, 3, 1, 7, 5, 3, 5, -1]  interior: [1, 3, 5, 7]
ok: <= 2 children per rank across both trees, p = 2..1024
elements sent per rank: [16, 32, 32, 32, 32, 32, 32, 16]  received: [16, 32, 32, 32, 32, 32, 32, 16]  (2n = 32 )
```

With 8 ranks, tree 0's interior nodes are the even ranks and tree 1's are the odd ones, so the two trees really are complementary. Every rank moves at most $2n$ per direction, as derived above; ranks 0 and 7 move only $n$ because each is the single-child root of one tree and a leaf of the other. One correction that the check turned up: for odd $p$ the statement "each rank is interior in at most one tree" is not literally true of this construction - rank 0 has one child in each tree - but the property that bandwidth actually needs, at most two children in total across both trees, holds for every $p$ from 2 to 1024.

In theory, then, the double binary tree is bandwidth-optimal *and* logarithmic. In practice NCCL's tree reaches a lower fraction of peak than its ring - a NCCL maintainer: "ring is superior in terms of peak bandwidth (except on 2 nodes), tree is superior in terms of base latency" ([NCCL #790](https://github.com/NVIDIA/nccl/issues/790), quoted in [`NCCL_ALGO`](measuring-and-debugging.md#nccl_algo)). Plausible reasons: an interior node receives from two children at once and must reduce both inputs, which needs more SM (streaming multiprocessor, one of the GPU's ~100+ independent compute cores) and memory bandwidth per byte; the fan-in creates 2:1 incast on a NIC; and the tree edges are longer-distance than ring edges, so they collide more in the fabric. I haven't found NVIDIA documentation that quantifies the split between these, so treat the list as hypotheses.

NCCL builds the tree at node granularity, not GPU granularity: inside a node the GPUs form a chain, and the double binary tree spans the nodes. A maintainer gives the latency models as $2 \cdot (G-1) \cdot l + 2 \cdot \log_2(N) \cdot L$ for the tree and $N \cdot (G-1) \cdot l + N \cdot L$ for one ring traversal ([NCCL #1049](https://github.com/NVIDIA/nccl/issues/1049); $l$ intra-node hop latency, $L$ network hop latency). The ring all-reduce is two traversals, so $2 \cdot (N \cdot (G-1) \cdot l + N \cdot L)$ up to the $-1$ terms.

Historically NCCL implemented the tree only for all-reduce; the other collectives use rings (or PAT, above, and NVLS or CollNet, the in-switch reduction schemes described in [NVLS](#nvls-nvlink-sharp) and [CollNet and IB SHARP](#collnet-and-ib-sharp), where available).

### Hierarchical all-reduce

The standard 2-level decomposition for $N$ nodes of $G$ GPUs:

1. Intra-node reduce-scatter over $G$ GPUs: each GPU ends with $n/G$ bytes, reduced over its node.
2. Inter-node all-reduce of those $n/G$ bytes among the $N$ GPUs with the same local rank - one independent all-reduce per rail.
3. Intra-node all-gather.

$$
\begin{aligned}
T_\text{hier} &= 2(G-1)\,\alpha_l + 2\,\frac{G-1}{G}\, n\beta_l + T_\text{AR,inter}(N, n/G) && \text{steps 1 and 3, plus step 2} \\
&= \ldots + 2(N-1)\,\alpha_L + 2\,\frac{N-1}{N}\, \frac{n}{G}\,\beta_L && \text{step 2 as a ring}
\end{aligned}
$$

where $\beta_l = 1/B_\text{nvlink}$ and $\beta_L = 1/B_{\text{nic per GPU}}$. Note what this does to the inter-node term: each GPU sends $2 \cdot (N-1)/N \cdot n/G$ through its own NIC, so the node as a whole sends $2 \cdot (N-1)/N \cdot n$ through its $G$ NICs - exactly the bandwidth lower bound for an all-reduce over $N$ nodes with node-level bandwidth $G \cdot B_\text{nic}$. The inter-node bytes don't shrink, they get spread across all the NICs. Steps 1-3 can be pipelined over chunks, in which case the time is bounded below by the slowest stage rather than the sum.

The same three steps, simulated with numpy on 4 nodes x 8 GPUs, counting bytes on each kind of link (a *rail* is the set of GPUs with the same local index across nodes, e.g. GPU 3 of every node, which in a rail-optimized cluster share a leaf switch):

```python
# hier_sim.py - 2-level all-reduce on N nodes x G GPUs, counting NVLink vs NIC bytes [numpy]
import numpy as np

def ring_rs(bufs):
    """in-place ring reduce-scatter over a list of p equal 1-D arrays; returns bytes sent per rank.
    afterwards rank r's chunk (r+1) % p is fully reduced."""
    p = len(bufs); ch = np.array_split(np.arange(bufs[0].size), p); sent = 0
    for k in range(p - 1):
        vals = [bufs[r][ch[(r - k) % p]].copy() for r in range(p)]   # all ranks send at once
        for r in range(p): bufs[(r + 1) % p][ch[(r - k) % p]] += vals[r]
        sent += vals[0].nbytes
    return sent

def ring_ag(bufs):
    """in-place ring all-gather; assumes rank r holds valid chunk (r+1) % p (output of ring_rs)."""
    p = len(bufs); ch = np.array_split(np.arange(bufs[0].size), p); sent = 0
    for k in range(p - 1):
        vals = [bufs[r][ch[(r + 1 - k) % p]].copy() for r in range(p)]
        for r in range(p): bufs[(r + 1) % p][ch[(r + 1 - k) % p]] = vals[r]
        sent += vals[0].nbytes
    return sent

N, G, n_elem = 4, 8, 4096                        # 4 nodes x 8 GPUs, 4096 fp32 per GPU = 16KiB
rng = np.random.default_rng(0)
x = rng.standard_normal((N, G, n_elem)).astype(np.float32)   # (N, G, n)
expect = x.sum(axis=(0, 1))                                    # (n,)
buf = x.copy()

# step 1: intra-node reduce-scatter (NVLink). GPU g ends owning shard (g+1) % G of its node's sum
nvl = sum(ring_rs([buf[i, g] for g in range(G)]) for i in range(N)) / N
# step 2: per-rail inter-node all-reduce of the owned shard (NIC). rail g = GPU g of every node
shards = np.array_split(np.arange(n_elem), G)
nic = 0
for g in range(G):
    idx = shards[(g + 1) % G]
    rail = [buf[i, g, idx].copy() for i in range(N)]          # N arrays of n/G elements
    nic += ring_rs(rail) + ring_ag(rail)
    for i in range(N): buf[i, g, idx] = rail[i]
nic /= G
# step 3: intra-node all-gather (NVLink)
nvl += sum(ring_ag([buf[i, g] for g in range(G)]) for i in range(N)) / N

assert np.allclose(buf, expect, atol=1e-4)
nb = n_elem * 4
print(f"per GPU: NVLink bytes sent {nvl:.0f}  (2(G-1)/G*n = {2*(G-1)/G*nb:.0f})")
print(f"per GPU: NIC bytes sent    {nic:.0f}  (2(N-1)/N*n/G = {2*(N-1)/N*nb/G:.0f})")
print(f"node total over its G NICs: {G*nic:.0f}  (2(N-1)/N*n = {2*(N-1)/N*nb:.0f}, the bound for N nodes)")
```

```
per GPU: NVLink bytes sent 28672  (2(G-1)/G*n = 28672)
per GPU: NIC bytes sent    3072  (2(N-1)/N*n/G = 3072)
node total over its G NICs: 24576  (2(N-1)/N*n = 24576, the bound for N nodes)
```

The NVLink side moves the full $2 \cdot (G-1)/G \cdot n$ per GPU; the NIC side moves $1/G$ of a flat all-reduce's bytes per GPU, and the $G$ NICs of a node together move exactly the all-reduce bound for $N$ nodes.

NCCL's flat multi-channel ring already gets the same inter-node bandwidth term: it builds one ring per NIC (more precisely, per channel - NCCL's unit of parallelism, one GPU thread block driving one ring, see [Channels](#channels) - with channels mapped onto NICs), and each ring crosses each node boundary once. What hierarchy buys you is the latency term - $2 \cdot (N-1)$ network hops instead of $2 \cdot (P-1)$ total hops - and the freedom to use a different algorithm at each level (NVLS inside, tree or SHARP across). NCCL's `NVLSTree` and `CollNetDirect` are exactly such hierarchical schedules.

### Summary of costs

Per-rank buffer $n$, $p$ ranks, contention-free:

| Collective     | Algorithm             | Latency term      | Bandwidth term          |
| :------------- | :-------------------- | :---------------- | :---------------------- |
| reduce-scatter | ring                  | $(p-1) \cdot \alpha$         | $(p-1)/p \cdot n \cdot \beta$         |
| reduce-scatter | recursive halving     | $\log_2(p) \cdot \alpha$       | $(p-1)/p \cdot n \cdot \beta$         |
| all-gather     | ring                  | $(p-1) \cdot \alpha$         | $(p-1)/p \cdot n \cdot \beta$         |
| all-gather     | recursive doubling    | $\log_2(p) \cdot \alpha$       | $(p-1)/p \cdot n \cdot \beta$         |
| all-reduce     | ring                  | $2 \cdot (p-1) \cdot \alpha$       | $2 \cdot (p-1)/p \cdot n \cdot \beta$       |
| all-reduce     | Rabenseifner          | $2 \cdot \log_2(p) \cdot \alpha$     | $2 \cdot (p-1)/p \cdot n \cdot \beta$       |
| all-reduce     | recursive doubling    | $\log_2(p) \cdot \alpha$       | $\log_2(p) \cdot n \cdot \beta$         |
| all-reduce     | double binary tree    | $\sim 2 \cdot \log_2(p) \cdot \alpha$    | $\sim 2 \cdot n \cdot \beta$                |
| all-reduce     | in-network (SHARP: switches that do the reduction themselves, see [below](#nvls-nvlink-sharp)) | $\sim 2 \cdot \text{depth} \cdot \alpha_\text{sw}$   | $\sim n \cdot \beta$                  |
| broadcast      | binomial tree         | $\log_2(p) \cdot \alpha$       | $\log_2(p) \cdot n \cdot \beta$         |
| broadcast      | pipelined chain       | $(p-2+S) \cdot \alpha$       | $(p-2+S)/S \cdot n \cdot \beta$       |
| all-to-all     | pairwise              | $(p-1) \cdot \alpha$         | $(p-1)/p \cdot n \cdot \beta$         |
| all-to-all     | Bruck                 | $\log_2(p) \cdot \alpha$       | $\log_2(p)/2 \cdot n \cdot \beta$       |

The $\gamma$ terms are omitted; add $(p-1)/p \cdot n \cdot \gamma$ to every reducing algorithm except SHARP, where the switch does the arithmetic.

The table is easier to absorb as a calculator. This one evaluates the four all-reduce formulas over message sizes and rank counts, marks the fastest, and finds the crossovers by bisection. $\alpha = 5\ \mu\text{s}$ and $B = 50\ \text{GBps}$ are assumptions for illustration, not NCCL's parameters:

```python
# cost_model.py - alpha-beta(-gamma) all-reduce cost for the classic algorithms
import math

def t_ring(p, n, a, b, g=0):  return 2*(p-1)*a + 2*(p-1)/p*n*b + (p-1)/p*n*g
def t_rabens(p, n, a, b, g=0): return 2*math.log2(p)*a + 2*(p-1)/p*n*b + (p-1)/p*n*g
def t_rd(p, n, a, b, g=0):    return math.log2(p)*(a + n*b + n*g)
def t_dbtree(p, n, a, b, g=0, eta=1.0):          # eta = achieved bw fraction vs ring
    return 2*math.log2(p)*a + 2*n*b/eta + n*g

ALGS = {"ring": t_ring, "rabenseifner": t_rabens, "rec-doubling": t_rd, "dbtree": t_dbtree}

def crossover(f, g, p, a, b, lo=1, hi=1e13):
    """smallest n (bytes) at which f becomes cheaper than g, by bisection on log scale; None if never."""
    d = lambda n: f(p, n, a, b) - g(p, n, a, b)
    if (d(lo) < 0) == (d(hi) < 0): return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if (d(mid) < 0) == (d(lo) < 0): lo = mid
        else: hi = mid
    return hi

def fmt(x):
    for u, s in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if x >= u: return f"{x/u:.3g}{s}"
    return f"{x:.3g}B"

if __name__ == "__main__":
    a, B = 5e-6, 50e9; b = 1 / B                     # assumed: 5us per message, 50GBps per direction
    sizes = [4, 4096, 2**20, 2**24, 2**28, 2**30]
    for p in (8, 64, 1024):
        print(f"\np={p}  alpha={a*1e6:.0f}us  B={B/1e9:.0f}GBps   time in us, * = fastest")
        print(f"{'size':>8} " + " ".join(f"{k:>13}" for k in ALGS))
        for n in sizes:
            ts = {k: f(p, n, a, b) * 1e6 for k, f in ALGS.items()}
            best = min(ts, key=ts.get)
            print(f"{fmt(n):>8} " + " ".join(f"{ts[k]:>12.1f}{'*' if k == best else ' '}" for k in ALGS))
        for f, g in (("rabenseifner", "rec-doubling"), ("ring", "rec-doubling"), ("ring", "dbtree")):
            x = crossover(ALGS[f], ALGS[g], p, a, b)
            print(f"  {f} beats {g} above: {fmt(x) if x else 'never (in 1B..10TB)'}")
    # closed form check: n* = (alpha_A - alpha_B) / (beta_B - beta_A)
    p = 8; aA, aB = 2*math.log2(p)*a, math.log2(p)*a; bA, bB = 2*(p-1)/p*b, math.log2(p)*b
    print(f"\nclosed-form rabenseifner/rec-doubling crossover at p=8: {fmt((aA-aB)/(bB-bA))}")
```

```
p=8  alpha=5us  B=50GBps   time in us, * = fastest
    size          ring  rabenseifner  rec-doubling        dbtree
      4B         70.0          30.0          15.0*         30.0 
   4.1KB         70.1          30.1          15.2*         30.2 
  1.05MB        106.7          66.7*         77.9          71.9 
  16.8MB        657.2         617.2*       1021.6         701.1 
   268MB       9465.2        9425.2*      16121.1       10767.4 
  1.07GB      37651.0       37611.0*      64439.5       42979.7 
  rabenseifner beats rec-doubling above: 600KB
  ring beats rec-doubling above: 2.2MB
  ring beats dbtree above: 8MB

p=64  alpha=5us  B=50GBps   time in us, * = fastest
    size          ring  rabenseifner  rec-doubling        dbtree
      4B        630.0          60.0          30.0*         60.0 
   4.1KB        630.2          60.2          30.5*         60.2 
  1.05MB        671.3         101.3*        155.8         101.9 
  16.8MB       1290.6         720.6*       2043.3         731.1 
   268MB      11199.6       10629.6*      32242.3       10797.4 
  1.07GB      42908.6       42338.6*     128879.0       43009.7 
  rabenseifner beats rec-doubling above: 372KB
  ring beats rec-doubling above: 7.44MB
  ring beats dbtree above: 912MB

p=1024  alpha=5us  B=50GBps   time in us, * = fastest
    size          ring  rabenseifner  rec-doubling        dbtree
      4B      10230.0         100.0          50.0*        100.0 
   4.1KB      10230.2         100.2          50.8*        100.2 
  1.05MB      10271.9         141.9*        259.7         141.9 
  16.8MB      10900.4         770.4*       3405.4         771.1 
   268MB      20956.9       10826.9*      53737.1       10837.4 
  1.07GB      53137.7       43007.7*     214798.4       43049.7 
  rabenseifner beats rec-doubling above: 312KB
  ring beats rec-doubling above: 63.6MB
  ring beats dbtree above: 259GB

closed-form rabenseifner/rec-doubling crossover at p=8: 600KB
```

What to take from it:

- **Tiny messages**: recursive doubling wins everywhere - fewest rounds, and bandwidth doesn't matter at 4 bytes.
- **The ring's latency term is what kills it at scale**: at $p=1024$ a 4-byte ring all-reduce costs 10ms in this model, 200x Rabenseifner.
- **On paper Rabenseifner dominates the ring at every size**, and the double binary tree is within a hair of it. In this model, the ring only beats the double binary tree at enormous sizes (8MB at $p=8$, 259GB at $p=1024$), purely from its $(p-1)/p$ factor. The model is wrong in a specific way - it assumes every pair of ranks is connected by an equally fast, uncontended link - and the next sections ([why NCCL doesn't use Rabenseifner](#recursive-halving-and-doubling), [tree efficiency](#double-binary-tree), [the ring/tree crossover](#the-ringtree-crossover)) are about what happens when you drop that assumption.

## algbw vs busbw

[nccl-tests](measuring-and-debugging.md#nccl-tests) reports two bandwidths, and the [`torch-dist-bench.py`](measuring-and-debugging.md#torch-dist-bench) benchmark here reports the same pair. The definitions are in nccl-tests' [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md):

$$
\begin{aligned}
\text{algbw} &= S / t \\
\text{busbw} &= \text{algbw} \cdot \operatorname{factor}(\text{collective}, p)
\end{aligned}
$$

where $S$ is the size nccl-tests reports (for all-gather and reduce-scatter that's the full, un-sharded buffer; for all-reduce it's the per-rank buffer) and $t$ the elapsed time.

`algbw` is the right number for "how long will my $S$-byte call take", but it isn't comparable across rank counts: a ring all-reduce over 2 ranks moves $1 \cdot S$ per rank and over 1024 ranks moves $\sim 2 \cdot S$, so `algbw` falls by ~2x with $p$ even when the links run at full speed. `busbw` divides that out. The factor is the bandwidth term of the bandwidth-optimal algorithm, divided by $S \cdot \beta$:

| Collective     | Bytes each rank must send (optimal) | busbw factor  | $p=8$  | $p=1024$ |
| :------------- | :---------------------------------- | :------------ | -----: | -------: |
| all-reduce     | $2 \cdot (p-1)/p \cdot S$                     | $2 \cdot (p-1)/p$   |  1.750 |    1.998 |
| reduce-scatter | $(p-1)/p \cdot S$                       | $(p-1)/p$     |  0.875 |    0.999 |
| all-gather     | $(p-1)/p \cdot S$                       | $(p-1)/p$     |  0.875 |    0.999 |
| all-to-all     | $(p-1)/p \cdot S$                       | $(p-1)/p$     |  0.875 |    0.999 |
| broadcast      | $S$ (root)                          | $1$           |  1.000 |    1.000 |
| reduce         | $S$                                 | $1$           |  1.000 |    1.000 |
| send/recv      | $S$                                 | $1$           |  1.000 |    1.000 |

The derivation for all-reduce, concretely: the ring moves $2 \cdot (p-1)/p \cdot S$ bytes per rank through that rank's slowest link at $B$, so $t \ge 2 \cdot (p-1)/p \cdot S/B$, i.e. $\text{algbw} \le B \cdot p/(2 \cdot (p-1))$, i.e. $\text{busbw} = \text{algbw} \cdot 2 \cdot (p-1)/p \le B$. So **`busbw` is an estimate of the per-rank, per-direction bandwidth of the bottleneck link under the assumption that the bandwidth-optimal point-to-point algorithm was used**. Compare it to the *unidirectional* per-GPU link rate: 450GBps for NVLink 4, 900GBps for NVLink 5, and for inter-node runs the per-GPU NIC rate (50GBps for a 400Gbps NIC) - not the duplex figure, and not the node aggregate. See [Unidirectional vs Bidirectional (Duplex)](interconnects.md#unidirectional-vs-bidirectional-duplex).

In code - this is all a benchmark has to do once it has a time $t$ (*algbw* is "algorithm bandwidth", bytes of user buffer per second; *busbw* is "bus bandwidth", the per-link rate it implies):

```python
# busbw.py - nccl-tests conventions (doc/PERFORMANCE.md): algbw = S/t, busbw = algbw * factor
FACTOR = {"all_reduce": lambda p: 2 * (p - 1) / p, "reduce_scatter": lambda p: (p - 1) / p,
          "all_gather": lambda p: (p - 1) / p, "all_to_all": lambda p: (p - 1) / p,
          "broadcast": lambda p: 1.0, "reduce": lambda p: 1.0, "sendrecv": lambda p: 1.0}
def bw(coll, S, t, p):
    algbw = S / t
    return algbw, algbw * FACTOR[coll](p)
for k, f in FACTOR.items(): print(f"{k:>14}: p=8 {f(8):.3f}   p=1024 {f(1024):.3f}")
# a 1GB all-reduce over 8 GPUs that took 4.77ms:
a, b = bw("all_reduce", 1e9, 4.77e-3, 8)
print(f"algbw {a/1e9:.0f} GB/s  busbw {b/1e9:.0f} GB/s  <- compare busbw with 450GB/s NVLink 4")
```

```
all_reduce: p=8 1.750   p=1024 1.998
reduce_scatter: p=8 0.875   p=1024 0.999
    all_gather: p=8 0.875   p=1024 0.999
    all_to_all: p=8 0.875   p=1024 0.999
     broadcast: p=8 1.000   p=1024 1.000
        reduce: p=8 1.000   p=1024 1.000
      sendrecv: p=8 1.000   p=1024 1.000
algbw 210 GB/s  busbw 367 GB/s  <- compare busbw with 450GB/s NVLink 4
```

The last line is the measured 8x H200 ring number from [SHARP](interconnects.md#sharp) run backwards: 4.77ms for 1GB is only 210GBps of `algbw`, which looks like less than half of NVLink 4, but it's 367GBps of `busbw`, i.e. 82% of the 450GBps link.

Three ways `busbw` misleads, all consequences of that "assuming" clause:

1. **In-network reduction**. With NVLS or SHARP, each rank sends $\sim S$ rather than $2 \cdot (p-1)/p \cdot S$, but nccl-tests still multiplies by $2 \cdot (p-1)/p$. That's how an 8x H100 node reports 480GBps `busbw` on a 450GBps link ([SHARP](interconnects.md#sharp), [nccl-tests #153](https://github.com/NVIDIA/nccl-tests/issues/153#issuecomment-1628415956)). The wire-level number for NVLS is closer to $\text{algbw} \cdot (1 + 1/p)$ - see [NVLS](#nvls-nvlink-sharp).
2. **Hierarchical runs**. On $N$ nodes the inter-node bottleneck carries $2 \cdot (N-1)/N \cdot S / G$ per NIC, not $2 \cdot (P-1)/P \cdot S$. `busbw` with $p = P$ then measures neither link: it's the bandwidth the per-GPU path would need under a flat ring. For $N$ large this is close to the per-GPU NIC rate, which is why inter-node `busbw` is usually compared with it, but it's an approximation, and at small $N$ the intra-node links contribute a visible share - see [Inter-node speed depends on intra-node speed](interconnects.md#inter-node-speed-depends-on-intra-node-speed).
3. **Latency-bound sizes**. Below $n_{1/2}$, `busbw` mostly measures $\alpha$. A curve of `busbw` vs size is really a plot of $S / (\alpha_\text{total} + S \cdot \beta_\text{eff})$.

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
- **`busbw` doing its job.** At 16MiB, `algbw` is 1.66 / 0.96 / 0.61GB/s for 2/4/8 ranks - it falls with $p$ as the [algbw vs busbw](#algbw-vs-busbw) section predicts - while `busbw` is 1.66 / 1.44 / 1.06.

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

How to read these lines, if [GPU software stack](../compute/gpu-software-stack.md#nccl-tests-single-node-then-two-then-all) is not fresh: `git clone ... && make` downloads and compiles the C/CUDA sources of nccl-tests ([toolkit](../foundations/toolkit.md#reading-c), git and Make sections); `./build/all_reduce_perf` is the resulting program. `python -m torch.distributed.run` is the same thing as the `torchrun` command, and `--rdzv_endpoint localhost:6000 --rdzv_backend c10d` tells it where the rendezvous store lives (the `MASTER_ADDR:MASTER_PORT` idea from above; [orchestration internals](../orchestration/internals.md#c10d-rendezvous) goes deeper). A line that starts with `NAME=value NAME2=value2 command` sets those environment variables for that one command only ([environment variables](../foundations/linux-and-os.md#environment-variables-and-path)). NCCL reads all of its configuration this way: `NCCL_DEBUG=INFO` makes it log its decisions to stderr, `NCCL_DEBUG_SUBSYS` picks which parts log (`INIT` setup, `GRAPH` topology and rings, `TUNING` algorithm choice), and `NCCL_ALGO` / `NCCL_PROTO` restrict which algorithm and protocol it may use (both explained in [NCCL internals](#nccl-internals) below). The full list is NCCL's [environment variables page](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

The output has the same `algbw`/`busbw` columns as the gloo script (per operation, out-of-place and in-place); compare the plateau `busbw` with the per-direction link rate, as in [algbw vs busbw](#algbw-vs-busbw).

## NCCL internals

The source of truth is the code (`src/graph/` for topology and search, `src/graph/tuning.cc` for the cost model, `src/device/` for the kernels) and the [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/). Hu et al., [Demystifying NCCL](https://arxiv.org/abs/2507.04786) (2025), is the best secondary source, and is where the protocol table in [Protocols: Simple, LL, LL128](#protocols-simple-ll-ll128) comes from. Internals change between releases - verify anything below against the version you run.

### Channels

A channel is NCCL's unit of parallelism: one CUDA thread block (CTA - a group of threads scheduled together onto one SM, part of a *kernel*, i.e. one function launched to run on the GPU) on each GPU, with its own ring or tree, its own peer connections and its own FIFO buffers. A collective's buffer is split across channels, each channel runs the algorithm independently on its slice. Multiple channels exist for two reasons:

1. **One SM can't saturate NVLink**. A single CTA's load/store throughput is far below 450-900GBps, so NCCL needs many SMs to drive the links.
2. **Multiple NICs and multiple paths**. Each channel's ring can enter and leave the node through a different NIC; NCCL typically uses at least one channel per NIC, often two.

The channel count is chosen per call by the tuner, and is reduced for small messages (fewer CTAs means less launch and synchronization overhead). That is one reason the measured `busbw` curve ramps more slowly than the one-$\alpha$-one-$\beta$ model predicts: the effective $\beta$ at mid sizes is that of fewer channels. You can bound it with [`NCCL_MIN_CTAS` / `NCCL_MAX_CTAS`](measuring-and-debugging.md#nccl_min_ctas-and-nccl_max_ctas) (formerly `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`, which are still accepted in the versions I've seen) or per communicator via `ncclConfig_t` (a *communicator* is NCCL's object for one group of ranks - the NCCL side of a PyTorch process group, created by `ncclCommInitRank`; `ncclConfig_t` is a C struct of options passed when creating it - see the "reading C" section of the [toolkit](../foundations/toolkit.md#reading-c)). Forcing more channels than the tuner picks rarely helps bandwidth on NVLink and costs SMs; forcing fewer is the standard knob to free SMs for overlapped compute (see [SM contention](#sm-contention)).

Why split work into channels at all, rather than one big kernel? Because a GPU is ~100+ independent cores, and a single core (SM) can only issue so many loads and stores per cycle. To move 450GBps you need many SMs each moving a slice - which is exactly a channel. **[GPU, untested]** - how to see and bound the channel count:

```bash
# how many channels NCCL picked, and the ring order of each (look for "Channel 00/.." lines)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  torchrun --nproc_per_node 8 network/benchmarks/torch-dist-bench.py 2>&1 | grep -E "Channel [0-9]+/" | head

# bound it: e.g. leave more SMs to the GEMMs you overlap with
NCCL_MIN_CTAS=4 NCCL_MAX_CTAS=8 torchrun --nproc_per_node 8 train.py
```

The channel count is also visible as the number of CTAs in the NCCL kernel's launch configuration in an Nsight Systems trace ([Nsight Systems](../compute/accelerator/execution-model.md#nsight-systems-where-does-the-time-go)).

### Topology detection and graph search

At `ncclCommInitRank`, NCCL builds a graph of the node: GPUs, NVLink and NVSwitch (a switch chip that connects many GPUs' NVLinks so every pair can talk at full speed) connections (from NVML, NVIDIA's driver query library), PCIe (the standard motherboard bus connecting GPUs, NICs and CPUs) switches and host bridges (from sysfs, the Linux kernel's file-based view of devices under `/sys`), CPUs/NUMA nodes (NUMA - non-uniform memory access - means each CPU socket has its own local memory and devices, and reaching another socket's is slower), and NICs with their speeds. You can dump it with `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` and override it with `NCCL_TOPO_FILE` - cloud providers ship topology files for VM types where sysfs lies about PCIe placement.

It then searches that graph for rings, trees and NVLS/CollNet patterns, trying to maximize per-channel bandwidth subject to constraints: a ring must enter and leave a node through NICs, prefer a NIC that is on the same PCIe switch as the GPU it serves, avoid crossing the CPU interconnect, and (with [`NCCL_CROSS_NIC=0/2`](measuring-and-debugging.md#nccl_cross_nic)) use the same NIC index on every node so the ring stays on one rail. `NCCL_GRAPH_DUMP_FILE=/tmp/graph.xml` writes the result; `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH` logs the chosen rings and trees as `Channel 00/16 : 0 1 2 3 4 5 6 7` style lines. If the search finds only low-bandwidth patterns - typically because a GPU and its NIC sit under different host bridges - every collective is capped by that, and no tuning variable will fix it.

**[GPU, untested]** - the commands to see what NCCL sees:

```bash
nvidia-smi topo -m                         # GPU/NIC matrix: NV# (NVLink), PIX/PXB (same PCIe switch), PHB, SYS
lspci -tv | less                           # the raw PCIe tree the matrix is derived from
NCCL_TOPO_DUMP_FILE=/tmp/topo.xml NCCL_GRAPH_DUMP_FILE=/tmp/graph.xml NCCL_DEBUG=INFO \
  torchrun --nproc_per_node 8 network/benchmarks/torch-dist-bench.py
grep -E "<gpu|<nic|<pci" /tmp/topo.xml | head -30   # what NCCL detected
grep -E "<graph|<channel" /tmp/graph.xml | head      # the rings/trees it built from it
```

Think of the graph search as a constrained shortest-path problem done once at startup: it's slow (it tries many candidate rings) and it's why `ncclCommInitRank` can take seconds on big machines, but the result is reused for every collective on that communicator.

### The tuning model

For each `(algorithm, protocol)` pair NCCL estimates

$$
\text{time}(\text{size}) = \operatorname{latency}(\text{algo}, \text{proto}, \text{nRanks}, \text{nNodes}) + \frac{\text{size}}{\operatorname{bandwidth}(\text{algo}, \text{proto}, \text{topology})}
$$

and picks the minimum - this is the "model of the latency and bandwidth of each algorithm/protocol combination" described in [NCCL #790](https://github.com/NVIDIA/nccl/issues/790). The latency is built from a per-`(algo, proto)` base latency plus a per-hop hardware latency times the hop count of the algorithm, with separate per-hop values for NVLink, PCIe and the network. The values live in `src/graph/tuning.cc`. As I recall them for recent versions (unverified, and they change across versions), the per-hop numbers are of the order of (`LL`, `LL128` and `Simple` are NCCL's three *protocols* - different ways for a receiver to know that data has arrived, trading latency for bandwidth - explained in [Protocols](#protocols-simple-ll-ll128) below): NVLink ring hop ~0.6µs `LL` / ~1.9µs `LL128` / ~3.4µs `Simple`; network hop ~2.7µs `LL` / ~4µs `LL128` / ~14µs `Simple`; base latencies of a few to ~15µs. I use those orders of magnitude in the [worked numbers](#worked-numbers) below and label them as assumptions. They are model parameters, not measurements - but they are the parameters NCCL itself uses to choose, so they're the right ones for predicting what NCCL will choose.

A toy version of that selection for an 8-GPU NVLink 4 ring, with the assumed per-hop numbers above. `LL128`'s base latency (8.4µs) and its 93.75% efficiency are my assumptions, the efficiency from the line format in [Protocols](#protocols-simple-ll-ll128); the real model has more terms and per-architecture tables:

```python
# tuner_toy.py - NCCL-style (algo, proto) selection with ASSUMED parameters, ring only, 8 GPUs NVLink 4
# time = base + hops * per_hop  +  size / (busbw_link * efficiency / busbw_factor)
p, B = 8, 450e9                          # ranks, NVLink 4 per-direction bytes/s
hops = 2 * (p - 1)                       # ring all-reduce: 14 sequential steps
PROTO = {  # name: (base_us, per_hop_us, bandwidth efficiency)  -- assumed orders of magnitude
    "LL":     (6.6, 0.6, 0.50),
    "LL128":  (8.4, 1.9, 0.9375),
    "Simple": (8.4, 3.4, 1.00),
}
def t_us(proto, n):
    base, hop, eff = PROTO[proto]
    algbw = B * eff / (2 * (p - 1) / p)     # bytes/s the caller sees
    return base + hops * hop + n / algbw * 1e6

for proto in PROTO:
    base, hop, eff = PROTO[proto]
    lat = base + hops * hop; algbw = B * eff / (2 * (p - 1) / p)
    print(f"{proto:>6}: latency {lat:5.1f} us   algbw {algbw/1e9:6.1f} GB/s   n_1/2 {lat*1e-6*algbw/1e6:5.1f} MB")
print(f"\n{'size':>10} " + " ".join(f"{k:>9}" for k in PROTO) + "   pick")
for n in [2**k for k in range(10, 31, 2)]:
    ts = {k: t_us(k, n) for k in PROTO}
    print(f"{n:>10} " + " ".join(f"{ts[k]:>9.1f}" for k in PROTO) + f"   {min(ts, key=ts.get)}")
# LL vs Simple crossover, ignoring LL128: (lat_S - lat_LL) / (1/bw_LL - 1/bw_S)
lat = {k: PROTO[k][0] + hops * PROTO[k][1] for k in PROTO}
bw = {k: B * PROTO[k][2] / 1.75 for k in PROTO}
print(f"\nLL/Simple crossover: {(lat['Simple'] - lat['LL']) * 1e-6 / (1/bw['LL'] - 1/bw['Simple']) / 1e6:.1f} MB")
```

```
    LL: latency  15.0 us   algbw  128.6 GB/s   n_1/2   1.9 MB
 LL128: latency  35.0 us   algbw  241.1 GB/s   n_1/2   8.4 MB
Simple: latency  56.0 us   algbw  257.1 GB/s   n_1/2  14.4 MB

      size        LL     LL128    Simple   pick
      1024      15.0      35.0      56.0   LL
      4096      15.0      35.0      56.0   LL
     16384      15.1      35.1      56.1   LL
     65536      15.5      35.3      56.3   LL
    262144      17.0      36.1      57.0   LL
   1048576      23.2      39.3      60.1   LL
   4194304      47.6      52.4      72.3   LL
  16777216     145.5     104.6     121.2   LL128
  67108864     537.0     313.4     317.0   LL128
 268435456    2102.8    1148.5    1099.9   Simple
1073741824    8366.3    4489.0    4231.7   Simple

LL/Simple crossover: 10.5 MB
```

The pattern - `LL` for small, `LL128` in the middle, `Simple` for large - is the qualitative behaviour NCCL shows; the exact switch points depend on parameters I'm not certain of, and on NVLS, which this toy leaves out.

A sanity check against measurement: the model's ring `LL` latency on 8 GPUs is $6.6 + 2 \cdot 7 \cdot 0.6 = 15.0\ \mu\text{s}$, and [Latency](measuring-and-debugging.md#latency) measures 15µs for every all-reduce up to 512KiB on 8x H200. The agreement is closer than the precision of the parameters justifies, so read it as "the right order", not as validation.

`NCCL_ALGO` and `NCCL_PROTO` restrict the set this minimum is taken over. Use them to measure each candidate on your hardware, not to tune production (see [`NCCL_ALGO`](measuring-and-debugging.md#nccl_algo)). A tuner plugin (`NCCL_TUNER_PLUGIN`) can replace the model; AWS ships one, which is why the B200 sweep in [SHARP](interconnects.md#sharp) switches at a different GPU count than the H200 one.

### Protocols: Simple, LL, LL128

The algorithm decides who sends what to whom. Orthogonal to it, the *protocol* decides how each of those transfers is synchronized - how a receiver knows data has arrived - and it swings performance as much as the algorithm does. NCCL has three of them; the protocol is the `proto` half of the `Algo NVLS proto SIMPLE` lines that [`NCCL_ALGO`](measuring-and-debugging.md#nccl_algo) shows you how to print.

Columns are ordered by ascending per-hop latency, which happens to be ascending bandwidth utilization too:

| Property              | LL                | LL128                          | Simple            |
| :-------------------- | :---------------- | :----------------------------- | :---------------- |
| Design goal           | low latency       | low latency and high bandwidth | high bandwidth    |
| Synchronization       | flag-based        | flag-based                     | memory fences     |
| Payload per unit      | 4B data + 4B flag | 120B data + 8B flag            | large data chunks |
| Bandwidth utilization | 25-50% of peak    | ~95% of peak                   | near peak         |
| Latency per hop       | ~1µs              | ~2µs                           | ~6µs              |

Source: Hu et al., [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786) (2025), Table I.

How to read the table: each column is one protocol. "Payload per unit" is what one synchronized write carries, "bandwidth utilization" is the fraction of the link's peak that protocol can reach, and "latency per hop" is the fixed cost it adds each time data moves from one GPU to the next - the protocol's contribution to $\alpha$. The rest of this section explains each row.

**Simple** writes a chunk into the receiver's FIFO slot (a slot in a first-in-first-out ring buffer shared by sender and receiver), issues a memory fence (an instruction that waits until all of this thread's previous writes are visible to other devices before letting later writes proceed; see [Memory models in brief](../foundations/concurrency.md#part-6-memory-models-in-brief)), then writes a flag (a step counter). The receiver polls the flag, then reads the chunk. The fence has to wait for all prior writes to become visible, which over NVLink or PCIe costs microseconds - so `Simple` amortizes it over large chunks and reaches near-peak bandwidth, but its $\alpha$ is the largest.

**LL** (low latency) removes the fence by making every store self-validating. The FIFO is a sequence of 16-byte lines, each `{data0, flag0, data1, flag1}` of 4 bytes each, written with a single 16-byte volatile store. The flag is the step number; the receiver spins on the line until both flags equal the expected step, then knows both data words are valid. This works because the line is written by one store instruction that the interconnect delivers without tearing (splitting it into parts that could arrive separately), so a matching flag implies the data next to it arrived in the same transaction. Why a flag per 4 bytes of data rather than one per line: the protocol needs every word of data to be validated by a flag that travelled in the same indivisible write, and pairing each 4-byte data word with a 4-byte flag in an 8-byte unit is the arrangement that only relies on 8-byte writes being atomic. That's my reading of the code's structure, not something NVIDIA documents. The cost is that half of every line is flag: 8 data bytes per 16 bytes moved, a 50% ceiling before anything else, and the narrow store pattern pushes the achieved fraction lower again - the "25-50% of peak" in the table. Across the network there is a second cost, according to Hu et al.: for the receiver to poll the flag, the `LL` staging buffer sits in host memory where the CPU can see it, which rules out GPUDirect RDMA (the NIC writing straight into GPU memory, see [PCIe, NUMA and GPUDirect RDMA](#pcie-numa-and-gpudirect-rdma)). Acceptable when latency is all that matters, wasteful otherwise.

**LL128** applies the same idea to 128-byte lines: 120 bytes of data and 8 bytes of flag, $120/128 = 93.75\%$ efficiency. A warp writes the line cooperatively, and correctness depends on the interconnect delivering a 128-byte write as a unit, without reordering it against the flag. NVLink provides that; NCCL enables `LL128` only on platforms where it knows it does, and disables it on others (some PCIe/CPU combinations) rather than risk silent corruption. `NCCL_PROTO=^LL128` turning a hang or a wrong result into a correct one is a known diagnostic for exactly this. So `LL128` is what you see on NVLink systems, and its strength is intra-node. Over a network the per-128-byte synchronization turns a large transfer into millions of small operations, and `Simple` takes the lead again.

The LL line format, packed with `struct` so you can see the bytes:

```python
# ll_line.py - the shape of an LL and an LL128 line, and why the efficiencies are 50% and 93.75%
import struct
step = 7                                       # flag value = the step number the receiver expects
line = struct.pack("<IIII", 0xAAAA, step, 0xBBBB, step)   # {data0, flag0, data1, flag1}, 16 bytes
d0, f0, d1, f1 = struct.unpack("<IIII", line)
ready = (f0 == step and f1 == step)            # receiver: spin until both flags match this step
print(f"LL line {len(line)}B  data {d0:#x} {d1:#x}  ready={ready}  efficiency {8/16:.2%}")
print(f"LL128 line 128B: 120B data + 8B flag -> efficiency {120/128:.2%}")
```

```
LL line 16B  data 0xaaaa 0xbbbb  ready=True  efficiency 50.00%
LL128 line 128B: 120B data + 8B flag -> efficiency 93.75%
```

**Why protocol choice is size-dependent**: in the $\alpha\text{-}\beta$ frame, `LL` has $\alpha \approx 0.6\ \mu\text{s/hop}$, $\beta \approx 1/(0.5 \cdot B)$; `Simple` has $\alpha \approx 3.4\ \mu\text{s/hop}$, $\beta \approx 1/B$. For 8 GPUs on NVLink 4 with the assumed parameters, `LL` costs $15\ \mu\text{s} + n/128.6\ \text{GBps}$ (algbw) and `Simple` costs $56\ \mu\text{s} + n/257\ \text{GBps}$, crossing at $n \approx 10.5\ \text{MB}$. `LL128` fills the middle. The measured switch points are earlier than that (`Ring LL` up to 1MiB then `NVLS Simple` from 2MiB on H200, per [`NCCL_ALGO`](measuring-and-debugging.md#nccl_algo)) because NVLS joins the candidate set.

### Chunking and pipelining

Each channel has, per peer, a FIFO of `NCCL_STEPS` slots (8 in the versions I've read). For `Simple` the FIFO is `NCCL_BUFFSIZE` bytes (default 4MiB), so a slot is 512KiB; `LL` and `LL128` have their own, smaller buffers. A ring all-reduce is executed as a loop over the buffer in pieces of `nRanks · chunkSize` per channel; within a piece it does the $2 \cdot (p-1)$ steps from the derivation above, each on one chunk. Up to `NCCL_STEPS` steps can be in flight between a sender and receiver, so as long as the per-hop latency is less than ~8 chunk transfer times, the ring streams at link rate and the $2 \cdot (p-1) \cdot \alpha$ fill cost is paid once per collective, not once per loop.

That's the condition behind a few practical effects:

- At 50GBps a 512KiB chunk takes 10.5µs, so 8 slots cover ~84µs of round-trip latency. A long-latency path (many switch hops, congested fabric) that exceeds that stalls the pipeline, and `busbw` drops even at large sizes.
- Raising `NCCL_BUFFSIZE` helps high-latency fabrics and costs GPU memory per peer per channel.
- Lowering it, or a very high channel count with small per-channel slices, makes the fill/drain cost a larger share.

The arithmetic behind the first bullet, for three link speeds:

```python
# chunking.py - does the 8-slot FIFO cover the round trip?
BUFFSIZE, STEPS = 4 * 2**20, 8            # NCCL_BUFFSIZE default, NCCL_STEPS (as read in source)
slot = BUFFSIZE // STEPS
for B in (50e9, 100e9, 450e9):
    t_chunk = slot / B
    print(f"B={B/1e9:>4.0f}GB/s  slot {slot//1024}KiB  chunk time {t_chunk*1e6:5.1f}us  "
          f"8 slots hide up to {STEPS*t_chunk*1e6:5.1f}us of latency")
```

```
B=  50GB/s  slot 512KiB  chunk time  10.5us  8 slots hide up to  83.9us of latency
B= 100GB/s  slot 512KiB  chunk time   5.2us  8 slots hide up to  41.9us of latency
B= 450GB/s  slot 512KiB  chunk time   1.2us  8 slots hide up to   9.3us of latency
```

Faster links shrink the latency a fixed FIFO can hide, which is one reason buffer sizes and protocols are re-tuned per hardware generation.

The FIFO itself is the classic bounded producer/consumer queue from [Queues](../foundations/concurrency.md#queues), with flags instead of a lock. A miniature of the `Simple` protocol with Python threads - the sender writes data, then the flag; the receiver spins on the flag, reads, then frees the slot:

```python
# fifo_flags.py - NCCL's "Simple" protocol in miniature: a FIFO of STEPS slots, data then flag
# (Python threads; the GIL hides the memory-ordering problem the real fence solves)
import threading, time
STEPS, NCHUNKS, CHUNK = 8, 32, 4                 # slots in flight, chunks to send, elements/chunk
slots = [None] * STEPS                           # the receiver's FIFO buffer
head = [0]                                       # flag: number of chunks the sender has posted
tail = [0]                                       # flag: number of chunks the receiver has consumed
waits = {"sender": 0, "receiver": 0}

def sender():
    for step in range(NCHUNKS):
        while step - tail[0] >= STEPS:           # FIFO full: wait for the receiver to free a slot
            waits["sender"] += 1; time.sleep(0)
        slots[step % STEPS] = [step] * CHUNK     # 1. write the data into the slot
        # (on a GPU: __threadfence_system() here, so the data is visible before the flag)
        head[0] = step + 1                       # 2. publish the flag

def receiver(out):
    for step in range(NCHUNKS):
        while head[0] <= step:                   # spin until the flag says chunk `step` is there
            waits["receiver"] += 1; time.sleep(0)
        out.extend(slots[step % STEPS])          # read the data
        time.sleep(0.0005)                       # pretend the reduction takes a while
        tail[0] = step + 1                       # tell the sender the slot is free

out = []
ts = [threading.Thread(target=sender), threading.Thread(target=receiver, args=(out,))]
for t in ts: t.start()
for t in ts: t.join()
assert out == [s for s in range(NCHUNKS) for _ in range(CHUNK)]
print(f"delivered {NCHUNKS} chunks in order through {STEPS} slots; "
      f"sender spun {waits['sender']} times (FIFO full), receiver spun {waits['receiver']} times")
```

```
$ python3 fifo_flags.py
delivered 32 chunks in order through 8 slots; sender spun 77937 times (FIFO full), receiver spun 0 times
```

(Spin counts vary run to run.) The receiver is slow here, so the sender spends its time blocked on a full FIFO - *backpressure*, the slow stage throttling the fast one. On a GPU the same spinning happens in a CTA that occupies an SM, which is why an idle-looking NCCL kernel still costs compute resources.

### NVLS (NVLink SHARP)

On NVSwitch 3 and later (H100 and newer), the switch can reduce. The GPU-side interface is a multicast object (one memory address that maps to a buffer on every GPU in a group, so a single write reaches all of them) (`cuMulticastCreate`) mapped into every participant, plus the PTX (NVIDIA's GPU assembly language) instructions `multimem.ld_reduce` (the switch reads the same address from every GPU in the group, reduces, and returns the sum), `multimem.st` (one store fanned out to every GPU) and `multimem.red`. That's why NCCL's log says `Created Multicast group` - see [SHARP](interconnects.md#sharp).

An NVLS all-reduce is, per GPU: `multimem.ld_reduce` its $n/p$ shard (the switch pulls $n/p$ from each of the $p$ GPUs and returns $n/p$), then `multimem.st` the result (the GPU sends $n/p$, the switch delivers it to all $p$). Per GPU per direction:

$$
\begin{aligned}
\text{sent} &= \underbrace{\tfrac{n}{p} \cdot p}_{p\ \text{contributions over}\ p\ \text{shards}} + \underbrace{\tfrac{n}{p}}_\text{store} = n\,(1 + 1/p) \\
\text{received} &= \underbrace{\tfrac{n}{p}}_\text{reduced shard} + \underbrace{n}_{\text{results of all}\ p\ \text{shards}} = n\,(1 + 1/p)
\end{aligned}
$$

That counts the GPU's own contribution and own copy as crossing its NVLink. If the switch doesn't need them to (I don't know whether it does - the PTX documentation describes semantics, not wire traffic), it's $n \cdot (p-1)/p$ instead. So the ideal time is between $n \cdot (p-1)/p \cdot \beta$ and $n \cdot (p+1)/p \cdot \beta$, against $2 \cdot (p-1)/p \cdot n \cdot \beta$ for the ring - an ideal speedup between $1.56\times$ ($2 \cdot 7/9$) and $2.0\times$ for $p = 8$.

```python
# nvls_bytes.py - bytes per GPU per direction, ring vs NVLS, p = 8
p, n, B = 8, 1e9, 450e9
ring = 2 * (p - 1) / p * n
lo, hi = (p - 1) / p * n, (p + 1) / p * n
print(f"ring {ring/1e9:.3f}GB -> {ring/B*1e3:.2f}ms   NVLS {lo/1e9:.3f}-{hi/1e9:.3f}GB -> "
      f"{lo/B*1e3:.2f}-{hi/B*1e3:.2f}ms   ideal speedup {ring/hi:.2f}x-{ring/lo:.2f}x")
print(f"measured: 480/367 = {480/367:.2f}x;  1GB at busbw 480 -> {1.75*n/480e9*1e3:.2f}ms")
```

```
ring 1.750GB -> 3.89ms   NVLS 0.875-1.125GB -> 1.94-2.50ms   ideal speedup 1.56x-2.00x
measured: 480/367 = 1.31x;  1GB at busbw 480 -> 3.65ms
```

The measured speedup on 8x H200 is $480/367 = 1.31\times$ ([SHARP](interconnects.md#sharp)). For a 1GB all-reduce that's 3.65ms measured against an ideal of 1.94-2.5ms. The gap is plausibly the switch's reduction throughput (NVIDIA quotes 450GBps of "reduction bandwidth" for the DGX H100 NVSwitch fabric, which is suspiciously close to one GPU's link rate) plus the SMs NCCL dedicates to NVLS - I have not found a source that pins it down.

`NVLSTree` composes NVLS inside the node with the inter-node tree; it [requires IB (InfiniBand) or RoCE](https://github.com/NVIDIA/nccl/issues/1031#issuecomment-1773965518). On rack-scale systems (NVL72) NVLS spans the whole NVLink domain.

### CollNet and IB SHARP

CollNet is NCCL's interface to in-network reduction on the scale-out fabric, implemented through a plugin (for InfiniBand SHARP, NVIDIA's `nccl-rdma-sharp-plugins`). Two variants:

- **CollNetDirect**: every GPU in a node talks to the network directly - intra-node reduce-scatter, each GPU sends its $n/G$ shard into the SHARP aggregation tree via its own NIC, the switches reduce across nodes and return $n/G$, then intra-node all-gather.
- **CollNetChain**: GPUs in a node form a chain to one NIC, for systems with fewer NICs.

Cost of the inter-node part for CollNetDirect: each GPU sends $n/G$ up and receives $n/G$ down, concurrently in opposite directions, plus the aggregation tree depth:

$$
\begin{aligned}
T_\text{inter,SHARP} &\approx 2 \cdot \text{depth} \cdot \alpha_\text{switch} + \frac{n}{G}\,\beta_L \\
T_\text{inter,ring} &\approx 2(N-1)\,\alpha_L + 2\,\frac{N-1}{N}\,\frac{n}{G}\,\beta_L
\end{aligned}
$$

So asymptotically SHARP halves the inter-node bandwidth term (the "N+1 sends instead of 2N" in [SHARP](interconnects.md#sharp)) and replaces the $O(N)$ latency with $O(\text{depth})$, where depth is 2-3 for a fat-tree. Constraints that make the speedup smaller in practice: the switch's aggregation throughput and the number of concurrent SHARP trees/jobs it supports are finite, SHARP resources must be allocated by the subnet manager / aggregation manager, and the supported reduction dtypes and ops are limited. Treat a 2x inter-node gain as the ceiling.

The two formulas at 128 nodes, with an assumed 1µs per switch level and depth 3 (I don't have a sourced number for SHARP's per-level latency):

```python
# sharp_inter.py - inter-node part per GPU, N nodes x G GPUs, n per GPU
N, G, n, B, a_L, a_sw, depth = 128, 8, 1e9, 50e9, 14e-6, 1e-6, 3   # a_sw assumed
sharp = 2 * depth * a_sw + (n / G) / B
ring = 2 * (N - 1) * a_L + 2 * (N - 1) / N * (n / G) / B
print(f"SHARP {sharp*1e3:.2f}ms   ring {ring*1e3:.2f}ms   ratio {ring/sharp:.2f}x")
```

```
SHARP 2.51ms   ring 8.52ms   ratio 3.40x
```

More than 2x here - but the extra beyond 2x is the ring's latency term, which a tree inter-node schedule would also remove ([Worked numbers](#1024-gpus-128-nodes-8x-400gbps-ib-per-node) has the tree at 5.20ms). The bandwidth-only gain is the 2x.

### PXN

PXN (PCI x NVLink, NCCL 2.12) lets a GPU send to a remote GPU through a *different* local GPU's NIC: data goes over NVLink to the GPU whose NIC is on the destination's rail, and that GPU's proxy (a CPU helper thread NCCL runs per GPU to drive the NIC) posts the RDMA (remote direct memory access: the NIC reads or writes the remote machine's memory directly, without involving the remote CPU). Two benefits ([Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/)):

1. **Rail alignment**: traffic from GPU $i$ on node A to GPU $j$ on node B can travel on rail $j$ the whole way, instead of entering rail $i$ and crossing to rail $j$ in the spine.
2. **Message aggregation**: messages from all local GPUs to the same remote NIC can be combined, reducing the number of in-flight connections and small RDMA ops.

This matters mostly for all-to-all and point-to-point patterns, where without PXN every GPU would talk to every remote NIC. `NCCL_PXN_DISABLE=1` turns it off; it trades NVLink bandwidth (cheap) for spine bandwidth (expensive, possibly oversubscribed).

**[GPU, untested]** - the A/B test for whether PXN matters to your job is an all-to-all benchmark with and without it. Preview: this uses Slurm, the cluster scheduler taught in [Slurm from zero](../orchestration/slurm/from-zero.md#srun-launching-a-step). Just enough to follow: `srun` starts copies of a program across the nodes Slurm has given you; `-N 2` asks for 2 nodes and `--ntasks-per-node 8` for 8 processes on each, one per GPU (hence `-g 1`, one GPU per process). For that to work nccl-tests must be compiled with MPI support (`make MPI=1 ...`, see [nccl-tests in the GPU software stack chapter](../compute/gpu-software-stack.md#nccl-tests-single-node-then-two-then-all)); MPI is the classic HPC library the 16 processes use to find each other.

```bash
# multi-node, e.g. 2 nodes x 8 GPUs under SLURM with MPI-built nccl-tests
srun -N 2 --ntasks-per-node 8 ./build/alltoall_perf -b 1M -e 1G -f 2 -g 1
NCCL_PXN_DISABLE=1 srun -N 2 --ntasks-per-node 8 ./build/alltoall_perf -b 1M -e 1G -f 2 -g 1
```

### Buffer registration and symmetric memory

By default NCCL copies user data into its own FIFO buffers and the peer copies it out - an extra pass through HBM and, for the network, the proxy works from NCCL-owned registered memory. User buffer registration (`ncclCommRegister`, or automatic registration of buffers captured in CUDA graphs - see [CUDA graphs](../compute/accelerator/execution-model.md#cuda-graphs)) lets NCCL work directly on the user buffer: for NVLS the user buffer is mapped into the multicast object, for IB it's registered with the NIC, avoiding the copy and freeing SMs. PyTorch exposes this through its NCCL memory pool APIs. Requirements and limitations (allocation through `ncclMemAlloc` / the cuMem APIs - CUDA's low-level virtual-memory driver calls (`cuMemCreate`, `cuMemMap`) that let one physical allocation be mapped into other GPUs and devices - which collectives benefit) vary by version - check the [NCCL user buffer registration docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html).

**[GPU, untested]** In PyTorch the route is a CUDA memory pool backed by NCCL's allocator, registered with the communicator. The method names below exist in the torch 2.15 type stubs (`ProcessGroupNCCL.mem_allocator`, `register_mem_pool`, `torch.cuda.MemPool`, `torch.cuda.use_mem_pool`); I didn't run it, and the API is young:

```python
import torch, torch.distributed as dist
dist.init_process_group("nccl")
backend = dist.distributed_c10d._get_default_group()._get_backend(torch.device("cuda"))
pool = torch.cuda.MemPool(backend.mem_allocator)      # allocations come from ncclMemAlloc
with torch.cuda.use_mem_pool(pool):
    grad = torch.empty(64 * 2**20, device="cuda", dtype=torch.bfloat16)   # (64M,)
backend.register_mem_pool(pool)                       # NCCL can now work on grad in place
dist.all_reduce(grad)
```

Symmetric memory (NCCL 2.27+, `ncclCommWindowRegister`) goes further: every rank registers a buffer at the same offset of a collective window, so a kernel can address any peer's buffer directly over NVLink without a handshake. This removes per-call synchronization overhead, which is why its gains are largest at small sizes - see [Symmetric memory](interconnects.md#symmetric-memory) for the measurements. In $\alpha\text{-}\beta$ terms, it attacks $\alpha$, not $\beta$.

**[GPU, untested]** PyTorch's (private, version-sensitive) entry point allocates a tensor in symmetric memory and exchanges handles once:

```python
import torch, torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem   # private module, as of torch 2.15
t = symm_mem.empty(1 << 20, dtype=torch.bfloat16, device=f"cuda:{dist.get_rank()}")   # (1M,)
hdl = symm_mem.rendezvous(t, dist.group.WORLD)          # one-time handshake; afterwards peers'
                                                        # buffers are directly addressable
```

## Topology

### NVLink and NVSwitch generations

Per-GPU unidirectional NVLink bandwidth, from the table in [NVLink](interconnects.md#nvlink) (NVIDIA's [NVLink page](https://www.nvidia.com/en-us/data-center/nvlink/)):

| Generation | GPU              | Links | Uni-dir. per GPU | Switch domain                    |
| :--------- | :--------------- | ----: | ---------------: | :------------------------------- |
| NVLink 3   | A100             |    12 |          300GBps | 8 GPUs (NVSwitch 2)              |
| NVLink 4   | H100/H200        |    18 |          450GBps | 8 GPUs, up to 256 with NVLink Switch System |
| NVLink 5   | B200/GB200/B300  |    18 |          900GBps | 8 (HGX) or 72 (NVL72)            |
| NVLink 6   | Rubin (announced)|    36 |         1500GBps | announced                        |

The NVLink 6 figure is preliminary. NVIDIA first announced 3.6TBps duplex per Rubin GPU (260TB/s per NVL72 rack); later reports cite a revision to 3TBps duplex (216TB/s per rack), which is the 1500GBps per direction in the table ([ServeTheHome](https://www.servethehome.com/?p=106378)). Check NVIDIA's current Vera Rubin NVL72 spec page before using it in a calculation.

Note that the "900GBps" usually quoted for NVLink 4 is the duplex sum; the number that goes into $\beta$ for a collective is 450GBps. NVL72: 72 GPUs x 1.8TBps duplex = 129.6TBps, the "130TB/s" NVIDIA quotes for [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/). On an NVL72, a ring all-reduce over 72 GPUs has bandwidth term $2 \cdot 71/72 \cdot n/900\ \text{GBps}$, i.e. the full NVLink rate at 9x the scale of an HGX node, and the inter-node problem starts at the rack boundary instead of the board.

```python
# nvlink_gens.py - per-direction rate is what goes into beta
for gen, uni in (("NVLink 3", 300), ("NVLink 4", 450), ("NVLink 5", 900)):
    print(f"{gen}: uni {uni}GB/s  duplex {2*uni}GB/s  8-GPU ring 1GB all-reduce >= {1.75e9/(uni*1e9)*1e3:.2f}ms")
print(f"NVL72: 72 x 1.8TB/s = {72*1.8:.1f}TB/s;  72-GPU ring bw term {2*71/72*1e9/900e9*1e3:.2f}ms per GB")
```

```
NVLink 3: uni 300GB/s  duplex 600GB/s  8-GPU ring 1GB all-reduce >= 5.83ms
NVLink 4: uni 450GB/s  duplex 900GB/s  8-GPU ring 1GB all-reduce >= 3.89ms
NVLink 5: uni 900GB/s  duplex 1800GB/s  8-GPU ring 1GB all-reduce >= 1.94ms
NVL72: 72 x 1.8TB/s = 129.6TB/s;  72-GPU ring bw term 2.19ms per GB
```

### PCIe, NUMA and GPUDirect RDMA

The inter-node path for a GPU's data is: HBM → GPU's PCIe interface → PCIe switch → NIC → wire, with GPUDirect RDMA letting the NIC DMA (direct memory access - a device copying memory without the CPU) to and from GPU memory without staging through host memory. Its bandwidth is the minimum along the path, and the path is where most "the NIC is fine but NCCL is slow" problems live:

- **GPU and NIC under the same PCIe switch** (`PIX`/`PXB` in `nvidia-smi topo -m`): peer-to-peer traffic stays inside the switch. DGX-style nodes pair each GPU with a NIC this way.
- **Through the CPU root complex** (`PHB`) or **across sockets** (`SYS`): peer-to-peer either degrades or is disabled, and NCCL may fall back to staging through host memory. `NCCL_NET_GDR_LEVEL` sets the maximum distance at which NCCL will use GPUDirect RDMA.
- **PCIe ACS** (Access Control Services, a PCIe switch feature that makes device-to-device traffic go up to the CPU's root complex so the IOMMU - the memory-protection unit for devices from [IOMMU, VFIO and PCIe passthrough](../foundations/containers-and-virtualization.md#iommu-vfio-and-pcie-passthrough) - can check it) forces peer-to-peer traffic up to the root complex for IOMMU checks; see [Disable Access Control Services](measuring-and-debugging.md#disable-access-control-services).

A 400Gbps NIC needs 50GBps, which a PCIe 5 x16 link (~64GBps per direction raw) can carry; an 800Gbps NIC (100GBps) needs PCIe 6 x16 or two x16 links. If GPU and NIC share an upstream PCIe link, their traffic can contend there.

PCIe bandwidth from first principles: lanes x transfer rate x encoding efficiency (PCIe 3.0 and later use 128b/130b encoding: 128 payload bits per 130 on the wire). These are raw numbers before packet headers and flow-control overhead, which take several percent more:

```python
for gen, gts in (("PCIe 4", 16), ("PCIe 5", 32), ("PCIe 6", 64)):
    enc = 128 / 130 if gen != "PCIe 6" else 1.0   # PCIe 6: FLIT mode, no 128b/130b; FLIT overhead not modelled
    print(f"{gen} x16: ~{gts*16*enc/8:.0f} GB/s per direction")
print("400Gbps NIC needs", 400/8, "GB/s;  800Gbps needs", 800/8, "GB/s")
# PCIe 4 x16: ~32 GB/s per direction
# PCIe 5 x16: ~63 GB/s per direction
# PCIe 6 x16: ~128 GB/s per direction
# 400Gbps NIC needs 50.0 GB/s;  800Gbps needs 100.0 GB/s
```

**[GPU, untested]** - checking a real node's GPU-NIC placement and link widths:

```bash
nvidia-smi topo -m                                  # look at the GPUi <-> mlx5_j cells: PIX/PXB good, SYS bad
sudo lspci -vv -s <bus:dev.fn> | grep -E "LnkCap|LnkSta"   # -s: one device by its PCI address (first column of lspci); capable vs negotiated speed/width, e.g. x16 vs x8
cat /sys/class/net/<ifname>/device/numa_node        # which NUMA node a NIC hangs off
```

### Rail-optimized vs non-rail fat-tree

*Bisection bandwidth* is the total bandwidth across the narrowest cut that splits the nodes into two equal halves - the worst-case capacity for "half the cluster talks to the other half", which is what all-to-all does. A *non-blocking* fat-tree has enough parallel leaf-to-spine uplinks that any half of the nodes can talk to the other half at full speed, i.e. full bisection bandwidth. In a **rail-optimized** network, NIC $i$ of every node connects to leaf switch(es) $i$ - rail $i$. Traffic between same-index GPUs on different nodes needs one switch hop if the rail fits in one leaf, and never touches the spine. NCCL's rings (with `NCCL_CROSS_NIC` defaulting to "prefer same NIC") and the per-rail inter-node step of a hierarchical all-reduce produce exactly that traffic, so all-reduce, reduce-scatter and all-gather can avoid the spine almost entirely. Cross-rail traffic (all-to-all, pipeline sends between different local ranks - in *pipeline parallelism* each GPU holds a different block of layers and sends activations to the next one, see [Pipeline parallelism](../training/model-parallelism/derivations.md#pipeline-parallelism)) either goes through the spine or, with [PXN](#pxn), hops over NVLink first.

A fat tree - links get "fatter" towards the root so bisection bandwidth is preserved:

![fat tree network](images/fat-tree-network.png)

([source](https://commons.wikimedia.org/wiki/File:Fat_tree_network.svg))

In a **non-rail** (top-of-rack) fat-tree all NICs of a node land on the same leaf, and inter-node traffic between racks goes through the spine, where ECMP (hash-based choice among equal-cost paths) decides which uplink it takes (see [Flow collisions and ECMP](#flow-collisions-and-ecmp)). `NCCL_CROSS_NIC=1` is the hint for this design.

**Oversubscription** $o{:}1$ at a tier means the downlink capacity is $o$ times the uplink capacity. For a flow that crosses that tier the bound becomes $B_\text{nic} / o$ whenever all downlinks are busy. A rail-optimized fabric can be oversubscribed at the spine with little cost to all-reduce-dominated training, because rings don't use it; the same fabric can halve all-to-all throughput. So whether oversubscription matters depends on the collective mix, not just on the fabric: for MoE training with expert parallelism across nodes, compute the all-to-all bound with the bisection bandwidth, not the NIC bandwidth.

```python
# oversub.py - per-GPU bound for traffic that crosses an o:1 oversubscribed tier
B_nic = 50e9
for o in (1, 2, 3, 4):
    t_a2a = 0.992 * 1e9 / (B_nic / o)                 # 1GB all-to-all, 99.2% leaves the node (P=1024, G=8)
    print(f"{o}:1  cross-spine bw/GPU {B_nic/o/1e9:5.1f} GB/s   1GB all-to-all >= {t_a2a*1e3:5.1f} ms")
```

```
1:1  cross-spine bw/GPU  50.0 GB/s   1GB all-to-all >=  19.8 ms
2:1  cross-spine bw/GPU  25.0 GB/s   1GB all-to-all >=  39.7 ms
3:1  cross-spine bw/GPU  16.7 GB/s   1GB all-to-all >=  59.5 ms
4:1  cross-spine bw/GPU  12.5 GB/s   1GB all-to-all >=  79.4 ms
```

### How topology changes NCCL's choices

Some consequences of the graph search and tuning model, all visible in `NCCL_DEBUG_SUBSYS=GRAPH,TUNING` output:

- **PCIe-only nodes** (no NVLink): fewer channels, `LL128` often disabled, rings ordered to minimize host bridge crossings; `busbw` capped by the slowest PCIe hop.
- **NVSwitch nodes**: all GPU pairs equidistant, so any ring order works; NVLS becomes a candidate from ~5-6 GPUs up (measured switch points in [SHARP](interconnects.md#sharp)).
- **NIC count**: sets the minimum useful channel count for inter-node work and how many rings can cross the node boundary in parallel. A node with 8 GPUs but 4 NICs gets half the inter-node bandwidth per GPU, and the tuner knows it.
- **Node count**: shifts the ring/tree boundary to larger sizes as $N$ grows (see the crossover table in [Worked numbers](#worked-numbers)).
- **SHARP availability**: adds `CollNet*`/`NVLSTree` to the candidate set.

**[GPU, untested]** - to see the decisions rather than infer them:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,TUNING ./build/all_reduce_perf -b 8 -e 1G -f 4 -g 8 2>&1 \
  | grep -iE "pattern|channel|tree|ring|nvls|collnet" | head -40
```

The output format changes between NCCL versions, so grep loosely and read the lines rather than parsing them.

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

## Compute-communication overlap

The overlap story in [Comms and compute overlap](interconnects.md#comms-and-compute-overlap) is the timeline view. This section is what makes overlap less than perfect even when the timeline says it should be.

### Streams and what "async" means

A NCCL collective is a kernel. PyTorch launches it on a dedicated NCCL stream (a CUDA stream is a queue of GPU work that runs in order; work in different streams may run concurrently) per process group (the set of ranks a collective runs over), with event-based dependencies on the compute stream (input ready) and back (`work.wait()` makes the compute stream wait for the NCCL stream). Overlap happens when the GPU has kernels from both streams ready at once and enough resources to run them concurrently (streams and events in PyTorch: [Multiple streams in PyTorch](../foundations/concurrency.md#multiple-streams-in-pytorch)). The thing being overlapped in data-parallel training is usually a *gradient bucket*: DDP and FSDP group gradients into buckets of a few tens of MB and launch one collective per bucket as soon as backward has produced all of that bucket's gradients, so communication for the last layers runs while backward is still computing the earlier ones ([Bucketing and overlap](../training/model-parallelism/derivations.md#bucketing-and-overlap) derives the sizes). Three ways it silently doesn't:

1. A dependency forces serialization (you `wait()` too early, or the bucket isn't ready until the last layer's backward).
2. The collective's CTAs can't be scheduled because compute kernels occupy all SMs - the NCCL kernel starts late or runs with fewer resident CTAs.
3. Both run, but compete for HBM bandwidth and L2, so both slow down.

"Async" in PyTorch means the call returns a handle (a `Work` object) before the communication is done; you call `work.wait()` at the point where you actually need the result. With NCCL, `wait()` doesn't block the CPU - it inserts a dependency so the compute stream waits for the NCCL stream. **[GPU, untested]**:

```python
grad = torch.randn(64 * 2**20, device="cuda", dtype=torch.bfloat16)   # (64M,) a gradient bucket
work = dist.all_reduce(grad, async_op=True)    # enqueued on the NCCL stream, returns immediately
y = layer(x)                                   # enqueued on the compute stream: can run concurrently
work.wait()                                    # compute stream now waits for the all-reduce
grad.div_(dist.get_world_size())               # safe: ordered after the all-reduce
```

The CPU analogue runs on a laptop. With gloo, `async_op=True` hands the collective to a background thread, so the main thread can compute meanwhile - and the same three failure modes apply, with CPU cores and memory bandwidth playing the role of SMs and HBM:

```python
# overlap_gloo.py - async_op=True lets compute run while gloo moves bytes [torch]
import os, time, statistics, torch, torch.distributed as dist, torch.multiprocessing as mp
def worker(rank, p):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=p)
    grad = torch.ones(16 * 2**20 // 4)               # (4M,) fp32 = 16MiB "gradient bucket"
    a = torch.randn(512, 512)                         # (512, 512) input of the "next layer"
    def compute():
        for _ in range(40): a @ a                     # (512,512) @ (512,512)
    def comm():      dist.all_reduce(grad)
    def seq():       comm(); compute()
    def overlap():
        work = dist.all_reduce(grad, async_op=True)   # returns a Work handle immediately
        compute()                                     # runs while gloo's thread moves bytes
        work.wait()                                   # block until the reduced grad is ready
    res = {}
    for name, f in (("comm", comm), ("compute", compute), ("sequential", seq), ("overlapped", overlap)):
        ts = []
        for _ in range(12):
            dist.barrier(); t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        res[name] = statistics.median(ts[2:]) * 1e3   # drop 2 warmups, median of 10
    if rank == 0:
        print(f"p={p}: " + "  ".join(f"{k} {v:.1f}ms" for k, v in res.items())
              + f"  | ideal overlapped = max = {max(res['comm'], res['compute']):.1f}ms")
    dist.destroy_process_group()
if __name__ == "__main__":
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29523")
    p = int(os.environ.get("WORLD", 4))
    mp.spawn(worker, args=(p,), nprocs=p)
```

```
$ for w in 2 4; do for i in 1 2; do WORLD=$w /tmp/mle-venv/bin/python overlap_gloo.py; done; done
p=2: comm 5.8ms  compute 16.9ms  sequential 23.3ms  overlapped 16.8ms  | ideal overlapped = max = 16.9ms
p=2: comm 6.2ms  compute 14.9ms  sequential 22.1ms  overlapped 14.7ms  | ideal overlapped = max = 14.9ms
p=4: comm 15.7ms  compute 23.2ms  sequential 37.0ms  overlapped 27.7ms  | ideal overlapped = max = 23.2ms
p=4: comm 14.3ms  compute 25.4ms  sequential 37.4ms  overlapped 26.5ms  | ideal overlapped = max = 25.4ms
```

With 2 ranks on a 10-core machine there are spare cores for gloo's thread, and the overlap is perfect: overlapped equals compute alone. With 4 ranks, 4 compute threads plus 4 gloo threads plus everything else on the machine compete for cores and memory bandwidth, and overlap hides only part of the communication (overlapped ~27ms vs the ideal ~24ms) - mechanism 3 above, contention, in miniature. My first version of this script timed each phase once and gave results anywhere from "perfect overlap" to "overlap slower than sequential" on the shared machine; medians of 10 repetitions made it stable. That's worth remembering when you read a single-shot overlap measurement.

### SM contention

Each NCCL channel is a CTA occupying (part of) an SM for the whole duration of the collective, spinning on flags when it has nothing to do. An H100 SXM has 132 SMs, so 16 channels is ~12% and 32 channels ~24% of the GPU unavailable to the GEMMs (general matrix multiplies - the matmul kernels) you're overlapping with - and the GEMM's tile scheduling is computed for the full SM count, so losing 12% of SMs can cost more than 12% through wave quantization (a GEMM is split into tiles that run in "waves" of one tile per SM; if 1000 tiles run on 132 SMs that's 8 waves, the last one partly empty, and on 116 SMs it's 9 - a whole extra wave for a 12% SM loss). The opposite case is also common: a large GEMM fills every SM, the NCCL kernel can't get its CTAs resident, and the collective is delayed until the GEMM's waves drain.

Knobs: [`NCCL_MIN_CTAS` / `NCCL_MAX_CTAS`](measuring-and-debugging.md#nccl_min_ctas-and-nccl_max_ctas) or per-process-group `ncclConfig_t`; giving the communication stream a higher priority; frameworks that cap the GEMM's SM count (cuBLAS / CUTLASS SM-count hints) to leave room. The cost model: if a collective with $c$ channels at full speed takes $T_c$, and the overlapped GEMM takes $T_g$ on all SMs, the best case with SM partitioning is $\max\left(T_c(c),\ T_g \cdot \text{SMs}/(\text{SMs} - c)\right)$ - and you search over $c$.

```python
# sm_partition.py - pick the NCCL channel count c that minimises max(T_comm(c), T_gemm * SMs/(SMs-c))
SMS = 132                                      # H100 SXM
T_gemm = 2.0e-3                                # 2ms GEMM on all SMs (assumed)
T_comm_16 = 1.5e-3                             # collective takes 1.5ms with 16 channels (assumed)
def T_comm(c):                                 # assume bandwidth ~ linear in channels up to 16, flat after
    return T_comm_16 * 16 / min(c, 16)
best = min(range(1, 33), key=lambda c: max(T_comm(c), T_gemm * SMS / (SMS - c)))
for c in (4, 8, 12, 16, 24, 32):
    print(f"c={c:>2}: comm {T_comm(c)*1e3:.2f}ms  gemm {T_gemm*SMS/(SMS-c)*1e3:.2f}ms  "
          f"step {max(T_comm(c), T_gemm*SMS/(SMS-c))*1e3:.2f}ms")
print("best c =", best)
```

```
c= 4: comm 6.00ms  gemm 2.06ms  step 6.00ms
c= 8: comm 3.00ms  gemm 2.13ms  step 3.00ms
c=12: comm 2.00ms  gemm 2.20ms  step 2.20ms
c=16: comm 1.50ms  gemm 2.28ms  step 2.28ms
c=24: comm 1.50ms  gemm 2.44ms  step 2.44ms
c=32: comm 1.50ms  gemm 2.64ms  step 2.64ms
best c = 11
```

With these (assumed) inputs the optimum is around 11-12 channels: fewer and the collective becomes the critical path, more and the GEMM's slowdown dominates. The real $T_\text{comm}(c)$ curve is something you measure, by sweeping `NCCL_MAX_CTAS` in the overlapped setting.

### Copy-engine collectives

GPUs have DMA copy engines - dedicated hardware units for memory copies - that move data over NVLink without using SMs. A collective built from `cudaMemcpyAsync` between peer buffers (or symmetric memory mappings) uses no SM time, which is attractive for overlap. The constraints: copy engines can't reduce, so this works for all-gather and all-to-all (and broadcast), not for reduce-scatter or all-reduce; per-copy launch overhead makes them poor at small sizes; and the number of copy engines and their per-engine bandwidth limit how much of the NVLink you can drive. PyTorch's symmetric memory module has copy-engine based low-contention all-gather/reduce-scatter variants, and recent NCCL releases added copy-engine based collectives for some operations - (both unverified in detail; check the release notes of your PyTorch and NCCL versions).

The installed torch 2.15 does have them: `torch.distributed._symmetric_memory` exposes `_low_contention_all_gather` and `_low_contention_reduce_scatter` (private names; their signatures aren't documented, so I don't show a call). The plain copy-engine building block is a peer-to-peer copy, which uses DMA rather than SMs when peer access is enabled. **[GPU, untested]**:

```python
x0 = torch.randn(256 * 2**20 // 2, device="cuda:0", dtype=torch.bfloat16)   # (128M,) 256MB on GPU 0
x1 = torch.empty_like(x0, device="cuda:1")
s = torch.cuda.Stream(device="cuda:1")
with torch.cuda.stream(s):
    x1.copy_(x0, non_blocking=True)        # NVLink peer copy, executed by a copy engine
s.synchronize()
```

### Async TP / collective matmul

*Tensor parallelism* (TP) splits each weight matrix of a layer across $p$ GPUs, so every GPU does a slice of every matmul; *sequence parallelism* (SP) additionally shards the activations between those matmuls along the token dimension, so instead of all-reducing a layer's output TP+SP all-gathers the activations before a matmul and reduce-scatters after it. A *column-parallel* linear layer gives each GPU a slice of the output columns of $W$; a *row-parallel* one gives each GPU a slice of the input rows, so the partial results have to be summed. This is the one-paragraph version; [Tensor parallelism](../training/model-parallelism/derivations.md#tensor-parallelism-megatron) and [Megatron sequence parallelism](../training/model-parallelism/derivations.md#megatron-sequence-parallelism-sp) derive it. With that, TP+SP has, in every layer, $Y = \operatorname{AllGather}(X) \cdot W$ (column-parallel) and $Z = \operatorname{ReduceScatter}(Y \cdot W')$ (row-parallel), where the collective is on the critical path of the GEMM that consumes or produces it. Bucketing can't overlap these - the GEMM needs the gathered data. Decomposition can: split the all-gather into its $p$ ring steps and run the GEMM on each shard as it arrives:

```
for k in 0..p-1:
    Y[shard (r-k) mod p] = X_shard_current @ W      # compute on what we have
    X_shard_current = recv_from(r-1) while send_to(r+1)  # overlapped with the GEMM
```

This is the collective matmul in Wang et al., [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://doi.org/10.1145/3567955.3567959) (ASPLOS 2023), and PyTorch's async TP ([Distributed w/ TorchTitan: Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)), which implements the transfers with symmetric memory and copy engines so they don't take SMs from the GEMM.

Cost model, with $t_g = T_\text{gemm}/p$ per shard and $t_c = (n/p) \cdot \beta$ per shard transfer:

$$
\begin{aligned}
T_\text{sequential} &= (p-1)\,t_c + p\,t_g \\
T_\text{decomposed} &\approx t_g + (p-1)\max(t_g, t_c) && \text{first shard is local, then pipeline}
\end{aligned}
$$

Worked example, TP=8 on H100 for the fused gate+up projection of a 70B-class MLP (hidden 8192, FFN 28672): $X$ is $8192\ \text{tokens} \times 8192 \times 2\ \text{B} = 134\ \text{MB}$ gathered; the local GEMM is $2 \cdot 8192 \cdot 8192 \cdot (2 \cdot 28672/8) = 9.6 \times 10^{11}$ FLOP per rank, at an assumed 70% of the 989 TFLOPS dense BF16 peak (H100 SXM datasheet) that's 1.39ms. The all-gather receives $7/8 \cdot 134\ \text{MB} = 117\ \text{MB}$ at an assumed 80% of 450GBps: 0.33ms. Sequential 1.72ms; decomposed, since `t_c < t_g`, ~1.39ms - the all-gather is fully hidden, a 19% saving on this op. The real saving is smaller: eight GEMMs with $M = 1024$ each run at lower efficiency than one with $M = 8192$ (wave quantization, less reuse), and concurrent NVLink traffic takes HBM bandwidth from the GEMM. Whether the net is positive is an empirical question per shape; the model tells you the ceiling (here 19%) and that it shrinks as $t_c/t_g$ shrinks - i.e. async TP pays most at small token counts per rank and high TP degree, where the GEMM is short relative to the transfer.

That the decomposition gives the same answer is easy to check without GPUs - here $p$ pretend ranks in one process, each doing its shard GEMM as the activation shards rotate around the ring - followed by the cost numbers above:

```python
# collective_matmul.py - all-gather-then-matmul == ring-decomposed matmul, simulated in one process [torch]
import torch
torch.manual_seed(0)
p, T, H, F = 4, 32, 16, 24                     # ranks, tokens, hidden, ffn shard width
X = torch.randn(T, H)                          # (T, H) full activations, sharded by tokens
W = [torch.randn(H, F) for _ in range(p)]      # rank r holds column shard W[r]: (H, F)
Xs = list(X.chunk(p))                          # Xs[r]: (T/p, H) = rank r's token shard

ref = [X @ W[r] for r in range(p)]             # sequential: AllGather(X) then (T, H) @ (H, F) -> (T, F)

out = [torch.empty(T, F) for _ in range(p)]
cur = list(Xs)                                 # what each rank currently holds
for k in range(p):                             # p GEMMs, p-1 ring transfers
    for r in range(p):
        src = (r - k) % p                      # the shard rank r holds at step k came from rank r-k
        out[r][src * T // p:(src + 1) * T // p] = cur[r] @ W[r]   # (T/p, H) @ (H, F) -> (T/p, F)
    if k < p - 1:                              # "send to r+1, receive from r-1", overlapped with next GEMM
        cur = [cur[(r - 1) % p] for r in range(p)]
assert all(torch.allclose(out[r], ref[r], atol=1e-5) for r in range(p))
print("decomposed == all-gather + matmul for every rank")

# cost model from the text, TP=8 70B-class MLP
X_bytes = 8192 * 8192 * 2
t_g = 2 * 8192 * 8192 * (2 * 28672 / 8) / (0.7 * 989e12)
t_c = 7 / 8 * X_bytes / (0.8 * 450e9)
print(f"X={X_bytes/1e6:.0f}MB  gemm {t_g*1e3:.2f}ms  all-gather {t_c*1e3:.2f}ms  "
      f"sequential {(t_g+t_c)*1e3:.2f}ms  decomposed ~{max(t_g, t_g/8 + 7*max(t_g/8, t_c/7))*1e3:.2f}ms  "
      f"saving {1 - max(t_g, t_g/8 + 7*max(t_g/8, t_c/7))/(t_g+t_c):.0%}")
```

```
decomposed == all-gather + matmul for every rank
X=134MB  gemm 1.39ms  all-gather 0.33ms  sequential 1.72ms  decomposed ~1.39ms  saving 19%
```

**[GPU, untested]** PyTorch's async TP is enabled through `torch.compile` in recent versions (TorchTitan exposes it as a config flag); the switch I know of is `torch._inductor.config._micro_pipeline_tp = True` plus enabling symmetric memory for the TP group, but this is private, version-sensitive API - check the TorchTitan docs for your version rather than trusting this line.

### Verifying overlap in a trace

Overlap is a claim about a timeline, so check the timeline. With the PyTorch profiler ([Spotting it in a trace](../compute/accelerator/execution-model.md#spotting-it-in-a-trace)) or Nsight Systems (`nsys profile --trace=cuda,nvtx`, see [Nsight Systems](../compute/accelerator/execution-model.md#nsight-systems-where-does-the-time-go); the full workflow is in [profiling and optimization](../training/profiling-and-optimization.md)):

1. Find the NCCL kernels (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, `..._ReduceScatter_..._RING_SIMPLE` etc. - the name carries collective, op, dtype, algorithm and protocol, which is also a cheap way to see NCCL's choice without debug logs) and the compute kernels they're meant to overlap with.
2. Check that they actually run concurrently on different streams, not back-to-back.
3. Compare each kernel's duration overlapped vs in isolation. If the NCCL kernel is 2x longer when overlapped, the collective is being starved (SMs or HBM); if the GEMM is 15% longer, that's the cost of the overlap, and it belongs in the accounting.
4. Measure **exposed communication**: the time a NCCL kernel runs with no compute kernel running concurrently. That, not total NCCL time, is what overlap failed to hide. Meta's [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) computes communication-computation overlap from PyTorch profiler traces.

You can practise the whole workflow on CPU. Profile 3 overlapped gloo all-reduces, export the trace (a JSON file you can open in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`), and compute exposed communication from it:

```python
# trace_gloo.py - see collectives in a profiler trace (CPU + gloo) [torch]
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.profiler import profile, ProfilerActivity
def worker(rank, p):
    dist.init_process_group("gloo", rank=rank, world_size=p)
    grad, a = torch.ones(2**20), torch.randn(256, 256)     # (1M,), (256, 256)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            w = dist.all_reduce(grad, async_op=True)
            for _ in range(20): a @ a
            w.wait()
    if rank == 0:
        prof.export_chrome_trace("/tmp/mle-coll/trace_rank0.json")   # open in https://ui.perfetto.dev
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=6))
    dist.destroy_process_group()
if __name__ == "__main__":
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29524")
    mp.spawn(worker, args=(2,), nprocs=2)
```

```
----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
       gloo:all_reduce         0.00%       0.000us             0       6.580ms       2.193ms             3  
          aten::matmul         1.06%      26.588us        79.48%       1.992ms      33.202us            60  
              aten::mm        78.17%       1.959ms        78.42%       1.966ms      32.759us            60  
      c10d::allreduce_        20.52%     514.308us        20.52%     514.308us     171.436us             3  
    aten::resolve_conj         0.25%       6.210us         0.25%       6.210us       0.034us           180  
----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.506ms
```

```python
# exposed.py - exposed communication = time a collective runs with no compute running
import json, sys
ev = [e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("ph") == "X"]
iv = lambda pred: sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if pred(e["name"]))
comm = iv(lambda n: n.startswith("gloo:") or n.startswith("nccl:"))   # (start_us, end_us)
comp = iv(lambda n: n in ("aten::mm", "aten::addmm", "aten::bmm"))
def covered(a, b, spans):                     # length of [a,b) covered by the union of spans
    tot, cur = 0.0, a
    for s, e in spans:
        s, e = max(s, cur), min(e, b)
        if e > s: tot += e - s; cur = e
    return tot
total = sum(e - s for s, e in comm)
exposed = sum((e - s) - covered(s, e, comp) for s, e in comm)
print(f"collectives: {len(comm)}  total {total/1e3:.2f} ms  exposed {exposed/1e3:.2f} ms "
      f"({exposed/total:.0%} not hidden by compute)")
```

```
$ python3 exposed.py /tmp/mle-coll/trace_rank0.json
collectives: 3  total 6.58 ms  exposed 4.65 ms (71% not hidden by compute)
```

Two things to notice. The `gloo:all_reduce` row has 0 self CPU time but milliseconds of total time: it's the span from launch to completion, during which the CPU was free - exactly like a NCCL kernel on its stream. And most of that span wasn't covered by matmuls - 71% exposed in this run, 52% in an earlier one - because 20 small matmuls take less time than a 4MB gloo all-reduce. More compute per collective, or smaller buckets launched earlier, is what would hide it. On GPU traces the same interval arithmetic applies to NCCL kernels (`ncclDevKernel_*`) vs compute kernels.

A NCCL kernel's duration includes the time it spends waiting for the slowest peer to arrive, so a "slow all-reduce" in one rank's trace is often a straggler on another rank - look at the same collective across ranks before concluding the network is slow.

## Worked numbers

Assumptions, stated once:

- Payload $n = 1\ \text{GB}$ (10^9 bytes = 5·10^8 bf16 elements), per rank.
- NVLink 4: 450GBps per GPU per direction (the "900GBps" is the duplex sum).
- Inter-node: 128 nodes x 8 GPUs, 8x 400Gbps IB per node, rail-optimized, non-blocking. 400Gbps = 50GBps per GPU, 400GBps per node per direction.
- Per-hop latencies of the order of NCCL's tuning model (assumed, see [The tuning model](#the-tuning-model)): `Simple` ring hop 3.4µs on NVLink, 14µs over the network, base 8.4µs; tree NVLink hop 4µs, network hop 14µs.
- Contention-free, homogeneous, and link-rate bandwidth unless stated - so these are bounds, not predictions. Real `busbw` is typically ~80-90% of the link rate at best ([Real network throughput](interconnects.md#real-network-throughput)).

Every number in the tables below comes from this script, so you can change an assumption and see what moves:

```python
# worked_numbers.py - every number in "Worked numbers", from the stated assumptions
import math
GB = 1e9; us = 1e-6
n = 1 * GB
B_nvl, B_nic = 450 * GB, 50 * GB
P, N, G = 1024, 128, 8
base, hop_nvl, hop_net = 8.4 * us, 3.4 * us, 14 * us        # Simple ring (assumed)
tree_nvl, tree_net = 4 * us, 14 * us                           # tree hops (assumed)
ms = lambda t: f"{t*1e3:.2f}ms"

print("== 8 GPUs, one NVLink 4 node ==")
lat = 14 * hop_nvl; bw = 1.75 * n / B_nvl
print("ring ideal   ", f"{lat/us:.0f}us +", ms(bw), "=", ms(lat + bw))
print("ring @367    ", ms(1.75 * n / 367e9))
print("NVLS ideal   ", ms(7/8 * n / B_nvl), "-", ms(9/8 * n / B_nvl))
print("NVLS @480    ", ms(1.75 * n / 480e9))
a = base + 14 * hop_nvl; print(f"Simple n_1/2 = {a/us:.0f}us * {450/1.75:.0f}GBps = {a*B_nvl/1.75/1e6:.1f}MB")
print(f"LL     n_1/2 = 15us * {0.5*450/1.75:.0f}GBps = {15*us*0.5*B_nvl/1.75/1e6:.2f}MB")

print("\n== 1024 GPUs, 128 nodes ==")
ring_bw = 2 * (P - 1) / P * n / (G * B_nic)
ring_lat = base + (2 * (P - 1) - 2 * (N - 1)) * hop_nvl + 2 * (N - 1) * hop_net
print("flat ring bw ", ms(ring_bw), " nvlink side", ms(7/8 * 2*(P-1)/P * n / B_nvl))
print(f"flat ring lat {base/us:.1f} + {(2*(P-1)-2*(N-1))*hop_nvl/us:.0f} + {2*(N-1)*hop_net/us:.0f} us = {ms(ring_lat)}")
print("flat ring    ", ms(ring_bw + ring_lat), f" data share {ring_bw/(ring_bw+ring_lat):.0%}",
      f" n_1/2 {ring_lat * G*B_nic/(2*(P-1)/P)/GB:.2f}GB")
t_bw = 2 * n / (G * B_nic); t_lat = 2 * 7 * tree_nvl + 2 * math.log2(N) * tree_net
print("dbtree       ", ms(t_bw), "+", f"{t_lat/us:.0f}us", "=", ms(t_bw + t_lat),
      f" n_1/2 {t_lat * G*B_nic/2/1e6:.0f}MB")
intra = 7 * hop_nvl + 7/8 * n / B_nvl
inter_ring = 2*(N-1)*hop_net + 2*(N-1)/N * (n/G) / B_nic
inter_tree = 2*7*hop_net + 2 * (n/G) / B_nic
inter_sharp = (n/G) / B_nic
print("intra RS/AG  ", ms(intra), "| inter ring", ms(inter_ring), "| tree", ms(inter_tree), "| SHARP", ms(inter_sharp))
for name, inter in (("ring", inter_ring), ("tree", inter_tree), ("SHARP", inter_sharp)):
    print(f"hier {name:<6}  sequential {ms(2*intra + inter)}   pipelined {ms(max(intra, inter))}")

print("\n== ring/tree crossover, n* = (a_ring - a_tree) / (2/(eta*B) - 2(P-1)/P/B) ==")
Bn = G * B_nic
for Nn in (4, 16, 128):
    Pp = Nn * G
    a_r = base + (2*(Pp-1) - 2*(Nn-1)) * hop_nvl + 2*(Nn-1) * hop_net
    a_t = 2*7*tree_nvl + 2*math.log2(Nn)*tree_net
    row = [f"{(a_r - a_t) / (2/(eta*Bn) - 2*(Pp-1)/Pp/Bn) / 1e6:.0f}MB" for eta in (0.9, 0.7, 0.5)]
    print(f"N={Nn:>3}  ring {a_r/us:.0f}us  tree {a_t/us:.0f}us  n* {row}")

print("\n== misc ==")
print("bcast chain S* =", round(math.sqrt(126 * n / 50e9 / 14e-6)),
      " T =", ms((126 + math.sqrt(126*n/50e9/14e-6)) * (14e-6 + n/50e9/math.sqrt(126*n/50e9/14e-6))),
      " binomial", ms(7 * n / 50e9))
print("gamma/beta  =", round(3 / (3.35e12 / 450e9), 2), " HBM/NVLink =", round(3.35e12 / 450e9, 2))
print("NVL72 TBps  =", 72 * 1.8)
print("IB timeout  20 ->", round(4.096e-6 * 2**20, 2), "s   22 ->", round(4.096e-6 * 2**22, 2), "s")
print("512KiB @ 50GBps =", round(512 * 1024 / 50e9 * 1e6, 1), "us   x8 =", round(8 * 512 * 1024 / 50e9 * 1e6), "us")
print("NVLS ideal speedup p=8:", round(2*7/9, 2), "-", 2*7/8 / (7/8))
print("H100 SM share 16/132 =", f"{16/132:.1%}", " 32/132 =", f"{32/132:.1%}")
```

```
== 8 GPUs, one NVLink 4 node ==
ring ideal    48us + 3.89ms = 3.94ms
ring @367     4.77ms
NVLS ideal    1.94ms - 2.50ms
NVLS @480     3.65ms
Simple n_1/2 = 56us * 257GBps = 14.4MB
LL     n_1/2 = 15us * 129GBps = 1.93MB

== 1024 GPUs, 128 nodes ==
flat ring bw  5.00ms  nvlink side 3.89ms
flat ring lat 8.4 + 6093 + 3556 us = 9.66ms
flat ring     14.65ms  data share 34%  n_1/2 1.93GB
dbtree        5.00ms + 252us = 5.25ms  n_1/2 50MB
intra RS/AG   1.97ms | inter ring 8.52ms | tree 5.20ms | SHARP 2.50ms
hier ring    sequential 12.45ms   pipelined 8.52ms
hier tree    sequential 9.13ms   pipelined 5.20ms
hier SHARP   sequential 6.44ms   pipelined 2.50ms

== ring/tree crossover, n* = (a_ring - a_tree) / (2/(eta*B) - 2(P-1)/P/B) ==
N=  4  ring 283us  tree 112us  n* ['240MB', '74MB', '33MB']
N= 16  ring 1190us  tree 168us  n* ['1719MB', '468MB', '203MB']
N=128  ring 9657us  tree 252us  n* ['16782MB', '4379MB', '1879MB']

== misc ==
bcast chain S* = 424  T = 33.64ms  binomial 140.00ms
gamma/beta  = 0.4  HBM/NVLink = 7.44
NVL72 TBps  = 129.6
IB timeout  20 -> 4.29 s   22 -> 17.18 s
512KiB @ 50GBps = 10.5 us   x8 = 84 us
NVLS ideal speedup p=8: 1.56 - 2.0
H100 SM share 16/132 = 12.1%  32/132 = 24.2%
```

### 8 GPUs, one NVLink 4 node

| Algorithm      | Latency term                  | Bandwidth term                          | Total    |
| :------------- | :---------------------------- | :-------------------------------------- | -------: |
| ring (ideal)   | $14 \times 3.4\ \mu\text{s} = 48\ \mu\text{s}$           | $1.75 \times 1\ \text{GB} / 450\ \text{GBps} = 3.89\ \text{ms}$         |  3.94ms  |
| ring at measured 367GBps `busbw` | -           | $1.75 \times 1\ \text{GB} / 367\ \text{GBps}$                  |  4.77ms  |
| NVLS (ideal)   | small                         | $(7/8 \ldots 9/8) \times 1\ \text{GB} / 450\ \text{GBps}$         | 1.94-2.50ms |
| NVLS at measured 480GBps `busbw` | -           | $1.75 \times 1\ \text{GB} / 480\ \text{GBps}$                  |  3.65ms  |

(The measured `busbw` values are the 8x H200 numbers from [SHARP](interconnects.md#sharp); H100 and H200 share NVLink 4.)

A 1GB all-reduce is deep in the bandwidth regime on one node: latency is ~1% of the ring's time.

```python
# the 8-GPU rows, from the stated assumptions (also printed by worked_numbers.py)
lat, B = 14 * 3.4e-6, 450e9                     # 2(p-1)=14 ring steps x 3.4us Simple NVLink hop (assumed)
for name, bw in (("ideal", B), ("measured busbw 367GB/s", 367e9)):
    print(f"ring {name:>23}: {(lat if name == 'ideal' else 0) * 1e3 + 1.75e9 / bw * 1e3:.2f}ms")
print(f"latency share of the ideal ring: {lat / (lat + 1.75e9 / B):.1%}")
# ring                   ideal: 3.94ms
# ring  measured busbw 367GB/s: 4.77ms
# latency share of the ideal ring: 1.2%
```

The NVLink tree is not listed because inside one NVSwitch node NCCL's tree degenerates to a chain, which has no advantage there.

Where the regime changes, for the 8-GPU ring:

- `Simple`: $\alpha_\text{total} = 8.4 + 14 \cdot 3.4 = 56\ \mu\text{s}$, asymptotic $\text{algbw} = 450/1.75 = 257\ \text{GBps}$, so $n_{1/2} = 56\ \mu\text{s} \cdot 257\ \text{GBps} \approx 14.4\ \text{MB}$.
- `LL`: $\alpha_\text{total} = 15\ \mu\text{s}$, $\text{algbw} \approx 129\ \text{GBps}$ (50% efficiency), $n_{1/2} \approx 1.9\ \text{MB}$.

The measured 8x H200 curve in [How the bandwidth is measured](measuring-and-debugging.md#how-the-bandwidth-is-measured) reaches half its ~480GBps peak at ~16MiB (`busbw` 256GBps), consistent with the `Simple` estimate - again, closer than the inputs justify.

### 1024 GPUs, 128 nodes, 8x 400Gbps IB per node

$P = 1024$, $N = 128$, $G = 8$.

**Flat ring** (8 rings, one per NIC, each crossing every node boundary once):

$$
\begin{aligned}
\text{bandwidth} &= 2\,\frac{P-1}{P} \cdot \frac{n}{G \cdot 50\ \text{GBps}} = \frac{1.998 \cdot 1\ \text{GB}}{400\ \text{GBps}} = 5.00\ \text{ms} \\
\text{NVLink side of the same rings} &= \frac{7}{8} \cdot \frac{1.998 \cdot 1\ \text{GB}}{450\ \text{GBps}} = 3.89\ \text{ms} \quad (\text{not the bottleneck}) \\
\text{latency} &= 8.4\ \mu\text{s} + (2 \cdot 1023 - 2 \cdot 127) \cdot 3.4\ \mu\text{s} + 2 \cdot 127 \cdot 14\ \mu\text{s} \\
&= (8.4 + 6093 + 3556)\ \mu\text{s} = 9.66\ \text{ms} \\
\text{total} &\approx 14.7\ \text{ms}
\end{aligned}
$$

The ring's fill latency is twice its bandwidth term. Only 34% of the 1GB all-reduce's time is spent moving data at link rate, and the ring's $n_{1/2}$ here is ~1.9GB.

**Double binary tree** (chain inside the node, double binary tree over nodes):

$$
\begin{aligned}
\text{bandwidth} &= 2n / 400\ \text{GBps} = 5.00\ \text{ms} \\
\text{latency} &= 2 \cdot 7 \cdot 4\ \mu\text{s} + 2 \log_2(128) \cdot 14\ \mu\text{s} = 56 + 196 = 252\ \mu\text{s} \\
\text{total} &\approx 5.25\ \text{ms} \quad (n_{1/2} \approx 50\ \text{MB})
\end{aligned}
$$

**Hierarchical**, NVLink reduce-scatter / per-rail inter-node all-reduce of $n/8$ / NVLink all-gather:

$$
\begin{aligned}
\text{intra RS (and AG)} &= 7 \cdot 3.4\ \mu\text{s} + \frac{7}{8} \cdot \frac{1\ \text{GB}}{450\ \text{GBps}} = 1.97\ \text{ms each} \\
\text{inter, ring} &= 2 \cdot 127 \cdot 14\ \mu\text{s} + 2 \cdot \frac{127}{128} \cdot \frac{125\ \text{MB}}{50\ \text{GBps}} = 3.56 + 4.96 = 8.52\ \text{ms} \\
\text{inter, tree} &= 2 \cdot 7 \cdot 14\ \mu\text{s} + 2 \cdot \frac{125\ \text{MB}}{50\ \text{GBps}} = 0.20 + 5.00 = 5.20\ \text{ms} \\
\text{inter, SHARP} &\approx \text{depth} \cdot \alpha + \frac{125\ \text{MB}}{50\ \text{GBps}} \approx 2.50\ \text{ms}
\end{aligned}
$$

| Schedule                          | Sequential (no stage pipelining) | Pipelined bound (slowest stage) |
| :-------------------------------- | -------------------------------: | ------------------------------: |
| flat ring                         |                          14.7ms  |                         14.7ms  |
| double binary tree                |                          5.25ms  |                          5.25ms |
| hierarchical, ring inter-node     |                          12.5ms  |                          8.52ms |
| hierarchical, tree inter-node     |                          9.13ms  |                          5.20ms |
| hierarchical, SHARP inter-node    |                          6.44ms  |                          2.50ms |

What this says:

1. **At 1024 GPUs the flat ring is latency-bound even at 1GB.** That's the textbook reason NCCL uses trees at scale for all-reduce, and the inter-node bandwidth floor of $2 \cdot (N-1)/N \cdot n / 400\ \text{GBps} \approx 5\ \text{ms}$ is the same for every non-SHARP schedule - they differ only in latency and in how well the stages overlap.
2. **Hierarchy alone doesn't save bandwidth; it saves latency.** The inter-node bytes per node are the same; what changes is $2 \cdot (N-1) = 254$ network hops instead of $2 \cdot (P-1) = 2046$ total.
3. **Only in-network reduction moves the bandwidth floor**, by up to 2x on the inter-node part. Whether it reaches 2x depends on switch aggregation throughput, which I don't have a reliable figure for.
4. All of these assume a non-blocking, collision-free fabric. A single ECMP collision on a ring path in a non-adaptive fabric multiplies the ring's bandwidth term by the collision degree (see [Flow collisions and ECMP](#flow-collisions-and-ecmp)).

### The ring/tree crossover

With equal bandwidth terms the tree wins at all sizes; the crossover only exists because the tree reaches a lower fraction of peak. Write the tree's achieved bandwidth as $\eta$ times the ring's and solve $n^* = (\alpha_\text{ring} - \alpha_\text{tree}) / (2/(\eta \cdot B_\text{node}) - 2 \cdot (P-1)/P / B_\text{node})$:

| Nodes | Ring latency | Tree latency | $n^*$ at $\eta=0.9$ | $\eta=0.7$ | $\eta=0.5$ |
| ----: | -----------: | -----------: | --------------: | ------: | ------: |
|     4 |        283µs |        112µs |           240MB |    74MB |    33MB |
|    16 |       1190µs |        168µs |          1.7GB |   468MB |   203MB |
|   128 |       9657µs |        252µs |           17GB |   4.4GB |   1.9GB |

The one measured data point in this repo: on 4x P6-B200 nodes NCCL switches from `Tree LL128` to `Ring LL128` at 32MiB ([Inter-node speed depends on intra-node speed](interconnects.md#inter-node-speed-depends-on-intra-node-speed)). With the assumed latencies, that corresponds to $\eta \approx 0.5$ - a 2x lower tree bandwidth - or to latencies different from the ones assumed, or both; on EFA, with AWS's tuner plugin, the inputs are not NCCL's defaults anyway. The qualitative conclusion is robust to the parameters: the crossover moves up roughly in proportion to $N$, so at 100+ nodes the tree (or NVLSTree / SHARP) is the right all-reduce algorithm for every gradient bucket size people actually use.

The table is the "ring/tree crossover" block of `worked_numbers.py` above. To see how sensitive it is to $\eta$, solve for the $\eta$ that puts the 4-node crossover at the measured 32MiB:

```python
# eta_fit.py - which tree efficiency makes the 4-node crossover land at 32MiB?
a_r, a_t, B, P = 283e-6, 112e-6, 400e9, 32     # latencies from the table (assumed hop costs)
n = 32 * 2**20
# n = (a_r - a_t) / (2/(eta*B) - 2(P-1)/P/B)  ->  solve for eta
eta = 2 / B / ((a_r - a_t) / n + 2 * (P - 1) / P / B)
print(f"eta = {eta:.2f}")
# eta = 0.50
```

## Failure modes in cost-model terms

### Stragglers

A synchronous collective completes when the last rank arrives, so its effective latency is $\alpha + \max_i(d_i)$, where $d_i$ is rank $i$'s arrival delay (compute jitter, OS noise, a GC pause, a thermally throttled GPU). For iid Gaussian jitter with standard deviation $\sigma$, $E[\max]$ over $p$ ranks (simulated; Monte Carlo, so ±0.01): $1.42\sigma$ at $p = 8$, $2.60\sigma$ at $p = 128$, $3.24\sigma$ at $p = 1024$ - growing like $\sqrt{2 \ln p}$ asymptotically (3.72 at $p = 1024$; the bound converges slowly). Two consequences:

- The expected straggler penalty per collective grows with scale even if no individual rank gets worse, and it's paid on every synchronization point - per bucket, per layer for TP.
- One persistently slow rank (not jitter, a slow GPU) sets the pace for everyone, and appears in every *other* rank's trace as a long NCCL kernel. The slow rank itself shows a short NCCL kernel - it's the last to arrive, so it waits least.

The expected-max numbers come from a seeded simulation:

```python
# straggler_sim.py - E[max of p iid N(0,1)] : the expected wait for the last rank, in units of sigma
import math, random
random.seed(0)
for p in (8, 128, 1024):
    trials = 20_000 if p < 1024 else 4_000
    m = sum(max(random.gauss(0, 1) for _ in range(p)) for _ in range(trials)) / trials
    print(f"p={p:>5}: E[max] = {m:.2f} sigma   sqrt(2 ln p) = {math.sqrt(2*math.log(p)):.2f}")
```

```
p=    8: E[max] = 1.43 sigma   sqrt(2 ln p) = 2.04
p=  128: E[max] = 2.59 sigma   sqrt(2 ln p) = 3.12
p= 1024: E[max] = 3.25 sigma   sqrt(2 ln p) = 3.72
```

And the "slow rank looks fast" signature is easy to reproduce with real processes. Rank 2 does 200ms of extra "compute" before the all-reduce; every rank times only the all-reduce call:

```python
# straggler_gloo.py - one late rank makes every OTHER rank's all-reduce look slow [torch]
import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp
def worker(rank, p):
    dist.init_process_group("gloo", rank=rank, world_size=p)
    x = torch.ones(1024)                         # (1024,) fp32 = 4KiB, latency-bound
    dist.all_reduce(x); dist.barrier()           # warm up connections
    if rank == 2: time.sleep(0.200)              # rank 2 "computes" 200ms longer
    t0 = time.perf_counter(); dist.all_reduce(x); dt = time.perf_counter() - t0
    print(f"rank {rank}: all_reduce took {dt*1e3:7.1f} ms")
    dist.destroy_process_group()
if __name__ == "__main__":
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29521")
    mp.spawn(worker, args=(4,), nprocs=4)
```

```
$ /tmp/mle-venv/bin/python straggler_gloo.py | sort
rank 0: all_reduce took   206.1 ms
rank 1: all_reduce took   206.2 ms
rank 2: all_reduce took     1.1 ms
rank 3: all_reduce took   206.2 ms
```

Three ranks report a 200ms all-reduce of 4KB; the guilty rank reports under a millisecond. If you only look at rank 0's profile, the network looks broken.

### A slow link dragging a ring

In a ring each step waits for the slowest hop, so the bandwidth term is set by $\min_i(B_i)$, not the mean:

$$
T_\text{ring,bw} = 2\,\frac{p-1}{p} \cdot \frac{n}{\min_i B_i}
$$

One link at half speed - a NIC that negotiated down, a flapping cable retraining at a lower width, a PCIe link at x8, an ECMP collision - halves the bandwidth of the whole ring, i.e. of every collective whose ring traverses it. With 8 channels on 8 NICs, one bad NIC slows the one ring that uses it to 50%, and since the buffer is split evenly across channels, that ring finishes last and the collective takes ~2x as long. The tree is less sensitive in aggregate (a slow link only slows the subtree behind it), but still gated by the worst path. In a hierarchical schedule only the affected rail's inter-node all-reduce is slow - still the collective's critical path.

The diagnostic signature: `busbw` ~1/k of normal, reproducible, independent of message size above $n_{1/2}$, and dependent on which nodes are in the job.

```python
# slow_link.py - a synchronous ring runs at the speed of its slowest link
p, n = 8, 1e9                                  # 8 ranks, 1GB buffer
B = [50e9] * p                                 # link r -> r+1, bytes/s
B[5] = 25e9                                    # one NIC negotiated down to half speed
chunk = n / p
steps = 2 * (p - 1)
t_step = [chunk / b for b in B]                # every rank sends one chunk per step
t = steps * max(t_step)                        # a step ends when the slowest hop delivers
print(f"healthy ring {steps*chunk/50e9*1e3:.2f} ms   with one half-speed link {t*1e3:.2f} ms")
print(f"mean link bw {sum(B)/p/1e9:.2f} GB/s is irrelevant; min is {min(B)/1e9:.0f} GB/s")
```

```
healthy ring 35.00 ms   with one half-speed link 70.00 ms
mean link bw 46.88 GB/s is irrelevant; min is 25 GB/s
```

### Timeouts and hangs

Distinguish three layers:

- **Transport timeout**: an RDMA op not acknowledged after $4.096\ \mu\text{s} \cdot 2^{\text{NCCL\_IB\_TIMEOUT}} \cdot \text{NCCL\_IB\_RETRY\_CNT}$ (see [InfiniBand](#infiniband)) - the QP errors, NCCL reports an error, the job dies. This is a link or fabric failure.
- **Collective hang**: ranks disagree about which collective comes next (different call order, a rank skipped a collective, mismatched sizes). Nothing is lost on the wire; every rank waits forever on a flag that will never be written. NCCL can't detect this.
- **Framework watchdog**: PyTorch's `ProcessGroupNCCL` watchdog (a background thread that checks for collectives that never complete) aborts collectives exceeding the process group `timeout` (10 minutes by default for NCCL). `TORCH_NCCL_TRACE_BUFFER_SIZE` enables the flight recorder, which dumps the last collectives each rank issued - comparing sequence numbers across ranks finds the one that diverged.

A timeout is the only way a hung collective becomes visible, so the timeout value is a trade-off between false aborts during long legitimate waits (checkpointing on one rank, a slow first iteration with compilation) and wasted GPU time when a job really is hung.

A collective hang, reproduced with gloo: rank 1 skips an all-reduce and goes straight to a barrier. Nothing on the wire is wrong; every rank waits on a message that will never come, and only the process-group timeout (5s here) turns it into an error:

```python
# hang_gloo.py - a rank skips a collective: nobody errors until the timeout fires [torch]
import os, datetime, torch, torch.distributed as dist, torch.multiprocessing as mp
def worker(rank, p):
    dist.init_process_group("gloo", rank=rank, world_size=p, timeout=datetime.timedelta(seconds=5))
    x = torch.ones(4)
    dist.all_reduce(x)                           # everyone: fine
    try:
        if rank != 1:                            # BUG: rank 1 skips this all-reduce
            dist.all_reduce(x)
        dist.barrier()                           # rank 1 waits here instead
        print(f"rank {rank}: finished (wrong!)")
    except Exception as e:
        print(f"rank {rank}: {type(e).__name__}: {str(e).splitlines()[0][:400]}")
if __name__ == "__main__":
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29522")
    mp.spawn(worker, args=(3,), nprocs=3)
```

```
$ time /tmp/mle-venv/bin/python hang_gloo.py | sort
rank 0: RuntimeError: [/Users/runner/work/pytorch/pytorch/third_party/gloo/gloo/transport/uv/unbound_buffer.cc:65] Timed out waiting 5000ms for recv operation to complete
rank 1: RuntimeError: [/Users/runner/work/pytorch/pytorch/third_party/gloo/gloo/transport/uv/unbound_buffer.cc:65] Timed out waiting 5000ms for recv operation to complete
rank 2: RuntimeError: [/Users/runner/work/pytorch/pytorch/third_party/gloo/gloo/transport/uv/unbound_buffer.cc:101] Timed out waiting 5000ms for send operation to complete
# wall time 7.2s: 5s timeout plus startup
```

Note what the error does *not* say: which rank skipped which collective. Every rank just timed out on a send or receive. That's why the flight recorder exists. **[GPU, untested]** - for NCCL jobs:

```bash
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000          # keep the last 2000 collectives per rank (alias: TORCH_FR_BUFFER_SIZE)
export TORCH_NCCL_DUMP_ON_TIMEOUT=1               # write them out when the watchdog fires
export TORCH_FR_DUMP_TEMP_FILE=/tmp/nccl_trace_rank_   # dump file prefix, one file per rank
torchrun --nproc_per_node 8 train.py
# then compare the last sequence number each rank reached; the odd one out is the culprit.
# PyTorch ships an analyzer for these dumps: the `torchfrtrace` command
# (torch.distributed.flight_recorder.fr_trace; present in torch 2.15, flags not checked here).
```

### Localizing with per-pair tests

The cost model gives the strategy: a collective over a set of ranks is as slow as its worst link, so test subsets small enough that each test exercises few links.

1. **Within a node**: `nvbandwidth` or `p2pBandwidthLatencyTest` ([nvbandwidth](measuring-and-debugging.md#nvbandwidth)) for every GPU pair; an 8-GPU `all_reduce_perf` should reach the expected `busbw` for its NVLink generation.
2. **Per NIC**: `ib_write_bw` between the same NIC index on two nodes, for each of the 8 rails - tests the NIC, cable and rail switch path without NCCL.
3. **Per node pair**: a 2-node `all_reduce_perf`. All 8128 pairs of 128 nodes is too many to run serially, but on a non-blocking fabric $N/2$ disjoint pairs can run at once, and a round-robin tournament schedule covers all pairs in $N-1 = 127$ rounds. Usually you don't need all of them: pair node $i$ with a known-good reference node, or run one round of $N/2$ pairs and bisect any slow pair against two known-good nodes.
4. **Bisect the full job**: run the collective over halves of the node set; the half that's slow contains the bad link. $\log_2(128) = 7$ rounds to a node, if the problem is a single node rather than a spine link.

Spine problems don't localize to a node: they show up as pairs that are slow only when they cross a particular leaf boundary, so record which leaf each node is on before you start.

The round-robin schedule from step 3 is the "circle method" from sports tournaments: fix one node, rotate the rest one position per round:

```python
# tournament.py - round-robin pairing (circle method): every node pair once, N/2 disjoint pairs per round
def rounds(N):
    nodes = list(range(N))
    for _ in range(N - 1):
        yield [(nodes[i], nodes[N - 1 - i]) for i in range(N // 2)]
        nodes = [nodes[0]] + [nodes[-1]] + nodes[1:-1]          # rotate all but the first
N = 8
for k, r in enumerate(rounds(N)): print(f"round {k}: {r}")
seen = {frozenset(p) for r in rounds(128) for p in r}
assert len(seen) == 128 * 127 // 2 == 8128 and len(list(rounds(128))) == 127
print("128 nodes: 8128 pairs in 127 rounds of 64 disjoint pairs")
```

```
round 0: [(0, 7), (1, 6), (2, 5), (3, 4)]
round 1: [(0, 6), (7, 5), (1, 4), (2, 3)]
round 2: [(0, 5), (6, 4), (7, 3), (1, 2)]
round 3: [(0, 4), (5, 3), (6, 2), (7, 1)]
round 4: [(0, 3), (4, 2), (5, 1), (6, 7)]
round 5: [(0, 2), (3, 1), (4, 7), (5, 6)]
round 6: [(0, 1), (2, 7), (3, 6), (4, 5)]
128 nodes: 8128 pairs in 127 rounds of 64 disjoint pairs
```

**[GPU, untested]** - one round as SLURM jobs, each a 2-node `all_reduce_perf` (assumes nccl-tests built with MPI support and that SLURM's `--nodelist` can place each pair). Preview of [Slurm](../orchestration/slurm/from-zero.md#srun-launching-a-step) again (`srun` as in [PXN](#pxn); `--nodelist=$a,$b` names the exact two nodes). The shell parts: `while IFS=, read a b; do ... done < pairs.txt` reads `pairs.txt` line by line, splitting each line at the comma (`IFS=,`) into the variables `a` and `b`; the `&` at the end of the `srun` line runs it in the background so all pairs of the round run at once, and `wait` blocks until every background job has finished ([job control](../foundations/linux-and-os.md#job-control--jobs-fg-bg-ctrl-z)). The last line pulls the "Avg bus bandwidth" summary out of every log and sorts numerically on the third `:`-separated field (`sort -t: -k3 -n`), slowest first:

```bash
# pairs.txt: one "nodeA,nodeB" per line, from one round of the schedule above
while IFS=, read a b; do
  srun -N 2 --nodelist=$a,$b --ntasks-per-node 8 ./build/all_reduce_perf -b 1G -e 1G -g 1 \
       > pair_${a}_${b}.log 2>&1 &
done < pairs.txt; wait
grep -H "Avg bus bandwidth" pair_*.log | sort -t: -k3 -n | head    # slowest pairs first
```

## You can now read

- [Measuring and debugging the network](measuring-and-debugging.md) - run nccl-tests and `torch-dist-bench.py` on real hardware, compare the plateau `busbw` with the bounds here, and the practical side of the failure modes above (wrong interfaces, missing P2P, NCCL debug logging).
- [Model parallelism derivations](../training/model-parallelism/derivations.md) - where the all-reduce, all-gather, reduce-scatter and all-to-all volumes of DP, FSDP, TP, SP and EP come from; plug them into the cost models above.
- [Training across datacenters and the internet](../training/decentralized-training.md) - the same $\alpha$-$\beta$ reasoning with $\alpha$ in milliseconds and $\beta$ a thousand times worse, which is why those methods communicate rarely and compress.
- [Profiling and optimization](../training/profiling-and-optimization.md) - finding exposed communication in real training traces.

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).

## Further reading

Algorithms and cost models:

- R. Thakur, R. Rabenseifner, W. Gropp, [Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521), IJHPCA 2005 - the alpha-beta-gamma derivations for ring, recursive doubling/halving and Rabenseifner's algorithm.
- P. Patarasuk, X. Yuan, [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://doi.org/10.1016/j.jpdc.2008.09.002), JPDC 2009 - the all-reduce bandwidth lower bound.
- P. Sanders, J. Speck, J. L. Träff, [Two-tree algorithms for full bandwidth broadcast, reduction and scan](https://doi.org/10.1016/j.parco.2009.09.001), Parallel Computing 2009 - the double binary tree.
- E. Chan, M. Heimlich, A. Purkayastha, R. van de Geijn, [Collective communication: theory, practice, and experience](https://doi.org/10.1002/cpe.1206), CCPE 2007 - lower bounds and the scatter+all-gather family.

Communication patterns, with pictures:

- C. Woolley, [NCCL: Accelerated Multi-GPU Collective Communications](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf), SC15 - the source of the collective and ring diagrams in this chapter.
- [UvA Deep Learning Tutorials: communication primitives](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/High-performant_DL/Multi_GPU/hpdlmultigpu.html#Communication-Primitives) - another set of good visuals for the same patterns.
- [Writing distributed applications with PyTorch](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) - the point-to-point and collective APIs, tutorial style.

NCCL:

- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) and its [environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) page.
- Z. Hu et al., [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786), 2025 - a comprehensive analysis of NCCL: the protocol variants (Simple, LL, LL128), how data moves intra-node and inter-node, and the ring and tree algorithms.
- S. Jeaugey, [Massively Scale Your Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) - double binary trees in NCCL.
- [Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/) - PXN.
- [New Scaling Algorithm and Initialization with NVIDIA Collective Communications Library 2.23](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/) - PAT.
- [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) - the `algbw`/`busbw` definitions.
- NCCL issues [#790](https://github.com/NVIDIA/nccl/issues/790) (tuning model) and [#1049](https://github.com/NVIDIA/nccl/issues/1049) (ring and tree latency models).

Networks:

- Y. Zhu et al., [Congestion Control for Large-Scale RDMA Deployments](https://doi.org/10.1145/2785956.2787484), SIGCOMM 2015 - DCQCN.
- A. Gangidi et al., [RDMA over Ethernet for Distributed AI Training at Meta Scale](https://doi.org/10.1145/3651890.3672233), SIGCOMM 2024.
- [Adaptive Routing in InfiniBand Hardware](https://web-backend.simula.no/sites/default/files/publications/files/adaptive_routing_in_infiniband_hardware.pdf).

Overlap:

- S. Wang et al., [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://doi.org/10.1145/3567955.3567959), ASPLOS 2023 - collective matmul.
- [Distributed w/ TorchTitan: Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487).
- L.-W. Chang et al., [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858), 2024 (id unverified).
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis).

In this repo: [Interconnects](interconnects.md), [Measuring and debugging the network](measuring-and-debugging.md).
