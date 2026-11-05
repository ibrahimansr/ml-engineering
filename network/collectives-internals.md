# Collective Communication Internals and Cost Models

## Prerequisites

Read these first if the systems vocabulary is new to you:

- [Computer architecture](../foundations/computer-architecture.md) - GPU memory (HBM), SMs, caches and PCIe; every bandwidth in this chapter is one of these.
- [Linux and the OS](../foundations/linux-and-os.md) - processes, NUMA, sysfs and drivers, which is where NCCL reads the machine's topology from.
- [Concurrency](../foundations/concurrency.md) - memory fences, flags, polling and streams; the NCCL protocols and compute-comm overlap are concurrency problems.
- [Networking](../foundations/networking.md) - links, switches, packets, flow control, RDMA and routing; needed for the topology and transport sections.
- [Distributed systems](../foundations/distributed-systems.md) - ranks, collectives, synchronization, stragglers and timeouts.
- [Transformer architectures](../models/architectures.md) - tensor, sequence and expert parallelism, which are what generate the all-gathers, reduce-scatters and all-to-alls costed here.

## Introduction

This chapter is about the theory underneath the measurements in the rest of this section: why a collective costs what it costs, how NCCL (NVIDIA's collective communication library, the backend PyTorch uses for GPU collectives) turns a topology into a schedule, and how to predict the number before running the benchmark. It assumes you already know what the collectives do ([Communication Patterns](comms.md)), what the hardware is ([Inter-Node and Intra-Node Networking Hardware](README.md)) and how to measure it ([Network Benchmarks](benchmarks/README.md)). Nothing here replaces measuring - the point is to know what number to expect, so that you can tell when a measurement is wrong.

All arithmetic in this chapter was checked with `python3`. Where a parameter value is an assumption rather than a spec, it is labelled as such. No benchmark numbers are invented here - where a measured number appears, it is quoted from elsewhere in this repo, with a link.

## The alpha-beta model

### Definitions

The Hockney / alpha-beta model charges a point-to-point message of `m` bytes:

```
T(m) = α + m·β
```

- `α` - per-message latency (seconds): software launch, NIC (network interface card, the adapter connecting a machine to the network) doorbell, switch hops, synchronization flags.
- `β` - inverse bandwidth (seconds/byte) of the bottleneck link, `β = 1/B`.
- `γ` - optional per-byte reduction cost (seconds/byte) for collectives that reduce.

Notation used throughout (a *rank* is one participating process, normally one per GPU; a *collective* is an operation that every rank in a group calls together, such as an *all-reduce*, which sums a buffer across all ranks and leaves the sum on every rank): `p` ranks, `n` bytes per rank (the input buffer size of an all-reduce), `N` nodes, `G` GPUs per node, `P = N·G`.

Two derived quantities do most of the work:

- `n_1/2 = α_total / β_eff` - the message size at which a collective reaches half of its asymptotic bandwidth. Below it you are latency-bound and a faster link doesn't help; above it you are bandwidth-bound and fewer calls don't help.
- The crossover size between two algorithms `A` and `B`: `n* = (α_A - α_B) / (β_B - β_A)`, where each `α` and `β` is the collective-level total (not per message).

The model assumes full-duplex links (a link is a physical connection between two devices; full-duplex means it carries traffic in both directions at full speed simultaneously, so a rank can send and receive `m` bytes simultaneously in time `α + m·β`), no contention, and homogeneous links. Each of those assumptions breaks somewhere in this chapter, and the breakage is usually what explains a disappointing measurement.

### Lower bounds

Before deriving algorithms, the bounds they are measured against:

- **Latency lower bound** for any collective that needs information from every rank to reach every rank: `⌈log2 p⌉·α`, since the number of ranks holding a given piece of information can at most double each round.
- **Bandwidth lower bound for all-reduce**: `2·(p-1)/p · n·β`. Sketch: think of the output as `p` shards and assign each shard to the rank that finishes reducing it. A rank's data has to leave it twice over: its contributions to the `p-1` shards it doesn't own (`(p-1)/p · n` bytes, a reduce-scatter), and - since the other ranks need the final result of the shard it owns, while what it sent earlier were only partial inputs - it must also receive the `p-1` reduced shards it doesn't own (`(p-1)/p · n`, an all-gather), with symmetric traffic going out. Counted per direction per rank that's `2·(p-1)/p · n`; the rigorous version has to rule out cleverer schemes, which is what the paper does. Patarasuk and Yuan prove this bound and that a ring achieves it ([Bandwidth optimal all-reduce algorithms for clusters of workstations](https://doi.org/10.1016/j.jpdc.2008.09.002), JPDC 2009).
- **Reduce-scatter and all-gather**: `(p-1)/p · n·β` each - every rank receives or sends `p-1` shards of size `n/p`.
- **Broadcast**: `n·β` - the root must at least send `n` bytes once, everyone must receive `n`.

These bounds are what `busbw` normalizes against, which is why they come first. See [algbw vs busbw](#algbw-vs-busbw).

## Deriving the classic algorithms

The canonical reference for everything in this section is Thakur, Rabenseifner and Gropp, [Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521) (IJHPCA 2005). The formulas below follow its conventions, with `n` the per-rank buffer size.

### Ring reduce-scatter

Split the buffer into `p` shards of `n/p` bytes. At step `k` (`k = 0..p-2`), rank `r` sends shard `(r-k) mod p` to rank `r+1`, which adds it to its own copy of that shard. After `p-1` steps, rank `r` holds the fully reduced shard `(r+1) mod p`.

Each step moves `n/p` bytes per rank, and all ranks send simultaneously on disjoint links:

```
T_RS_ring = (p-1)·α + (p-1)/p · n·β + (p-1)/p · n·γ
```

### Ring all-gather

Same pattern, no reduction - at each step a rank forwards the shard it most recently received:

```
T_AG_ring = (p-1)·α + (p-1)/p · n·β
```

Here `n` is the size of the gathered output on each rank, which is also how `nccl-tests` reports the size for all-gather.

### Ring all-reduce

All-reduce = reduce-scatter followed by all-gather:

```
T_AR_ring = 2·(p-1)·α + 2·(p-1)/p · n·β + (p-1)/p · n·γ
```

One chunk travelling around the ring:

![ring all-reduce chunk movement](images/all-reduce-ring-chunk1.png)

([source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf))

The bandwidth term hits the lower bound exactly, which is why the ring is the default for large messages. The latency term grows linearly in `p`, which is why it isn't the default at scale.

The `γ` term on a GPU is a memory-bound elementwise add. A ring step reads the local shard and the received shard and writes the sum, so roughly 3 bytes of HBM (high-bandwidth memory, the GPU's own DRAM) traffic per byte on the wire. With H100 HBM3 at 3.35TBps against NVLink (NVIDIA's GPU-to-GPU link, much faster than PCIe) 4 at 450GBps per direction, the ratio is ~7.4x, so `γ` is about `3/7.4 ≈ 0.4` of `β` in the worst case - not negligible, but NCCL fuses it into the receive and it largely overlaps with the transfer. It becomes visible with low channel counts (fewer SMs doing the adds) and during heavy compute overlap (the HBM is shared).

### Broadcast: binomial tree vs scatter+all-gather vs pipelined chain

- **Binomial tree**: `T = ⌈log2 p⌉·(α + n·β)`. Latency-optimal, but every round sends the whole buffer, so it's `log2 p` times off the bandwidth bound.
- **Scatter + all-gather** (van de Geijn): `T = (log2 p + p - 1)·α + 2·(p-1)/p · n·β`. Within `2x` of the bandwidth bound for any `p`.
- **Pipelined chain** (what [comms.md](comms.md#broadcast-with-unidirectional-ring) shows for NCCL's ring broadcast): split `n` into `S` chunks and stream them down a chain of `p` ranks. The last chunk arrives after `p-2+S` chunk steps:

```
T_bcast_chain(S) = (p - 2 + S)·(α + n·β/S)
```

Minimizing over `S`: `dT/dS = 0` gives `S* = sqrt((p-2)·n·β/α)`. For `p = 128`, `n = 1GB`, `β = 1/(50GBps)`, `α = 14µs` (assumed), `S* ≈ 424` chunks of ~2.4MB each, and `T ≈ 33.6ms` against the `n·β = 20ms` bound, versus `140ms` for the binomial tree. The gap to the bound is the `sqrt` fill cost, `≈ 2·sqrt((p-2)·α·n·β)` for large `S`. This is the same pipelining that makes NCCL's ring and tree reach their asymptotic bandwidth - chunking is not an optimization on top of the algorithm, it is what makes the bandwidth term of the algorithm achievable at all.

### Recursive halving and doubling

For power-of-two `p`:

- **Recursive halving reduce-scatter**: at round `k`, exchange half of your remaining data with the partner at distance `p/2^(k+1)` and reduce. Round `k` moves `n/2^(k+1)` bytes, total `log2 p` rounds:

```
T_RS_rh = log2(p)·α + (p-1)/p · n·β + (p-1)/p · n·γ
```

- **Recursive doubling all-gather**: the mirror image:

```
T_AG_rd = log2(p)·α + (p-1)/p · n·β
```

- **Rabenseifner all-reduce** = halving RS + doubling AG: `2·log2(p)·α + 2·(p-1)/p · n·β + (p-1)/p · n·γ`. Bandwidth-optimal *and* logarithmic latency - on paper it dominates the ring.
- **Recursive doubling all-reduce** (exchange the full buffer each round): `log2(p)·(α + n·β + n·γ)`. Best for tiny messages.

Why NCCL doesn't use Rabenseifner as its main algorithm: the partner at round `k` is at distance `p/2^(k+1)`, so on a hierarchical topology the first rounds - the ones moving the most data - cross the slowest links, and on a fat-tree (the usual cluster network: nodes plug into *leaf* switches, and leaves are joined through a layer of *spine* switches) those long-distance exchanges collide in the spine. A ring keeps every transfer on a nearest-neighbour link that NCCL picked to be fast. NCCL 2.23 did add PAT (Parallel Aggregated Trees), a Bruck-like logarithmic algorithm for all-gather and reduce-scatter, but restricted to one GPU per node ([NVIDIA blog](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/)).

### All-to-all

Each rank holds `p` blocks of `n/p` bytes, block `j` destined for rank `j` (`n` is the per-rank send buffer).

- **Pairwise exchange**: `p-1` rounds, round `k` sends to `r+k` and receives from `r-k`:

```
T_A2A_pairwise = (p-1)·α + (p-1)/p · n·β
```

- **Bruck**: `log2 p` rounds, each sending half the buffer: `log2(p)·α + (n/2)·log2(p)·β`. Wins only when `α` dominates.

Unlike the other collectives, all-to-all has no locality to exploit - the `(p-1)/p` term is the fraction of the buffer that has to leave the rank, and on a multi-node job almost all of it has to leave the node. For `P = 1024` GPUs over `N = 128` nodes, `(P-G)/P = 99.2%` of each rank's buffer crosses the scale-out network, so at 50GBps per GPU NIC the bound is `0.992 · n / 50GBps ≈ 19.8ms` per GB. NVLink bandwidth is irrelevant to it, and the network's bisection bandwidth is the real denominator: with an oversubscription of `o:1` above the leaf switches, any traffic that crosses the spine gets `1/o` of the NIC rate. All-to-all is the collective that exposes oversubscription and incast; all-reduce mostly doesn't, because the ring keeps traffic on a few fixed paths.

### Double binary tree

A single pipelined binary tree for all-reduce (reduce up, broadcast down) has logarithmic latency, but a leaf only ever sends during the reduce and only ever receives during the broadcast, so half the ranks use half their links. Sanders, Speck and Träff ([Two-tree algorithms for full bandwidth broadcast, reduction and scan](https://doi.org/10.1016/j.parco.2009.09.001), Parallel Computing 2009) fix this with two trees over the same ranks such that each rank is interior in at most one of them and the two trees each carry half the data. NCCL adopted this in 2.4 ([Massively Scale Your Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/), Jeaugey 2019) - [comms.md](comms.md#tree) describes the construction.

Two complementary binary trees - every rank is a leaf in one and an interior node in the other:

![NCCL double binary tree](images/nccl-double-binary-tree.png)

([source](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/))

NVIDIA's measured small-message latency, tree vs ring, as the node count grows:

![NCCL tree vs ring latency on Summit](images/nccl-tree-vs-ring-latency-summit.png)

([source](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/))

Bandwidth: count what one rank moves. In the tree where it is interior it receives `n/2` from each of 2 children and sends `n/2` to its parent during the reduce, then receives `n/2` from its parent and sends `n/2` to each child during the broadcast. In the tree where it's a leaf it sends `n/2` up and receives `n/2` down. Totals per direction:

```
sent     = n/2 + n   (interior) + n/2 (leaf) = 2n
received = n + n/2   (interior) + n/2 (leaf) = 2n
```

So with all of a rank's links sharing one per-direction bandwidth `B` (which is the case for the NIC of a node), `T_bw ≈ 2n/B` - the same as the ring's `2·(p-1)/p · n/B` up to the `(p-1)/p` factor. The latency term is the pipeline depth, `~2·log2(p)` hops (up then down):

```
T_AR_dbtree ≈ 2·log2(p)·α + 2·n·β  (+ pipeline fill)
```

In theory, then, the double binary tree is bandwidth-optimal *and* logarithmic. In practice NCCL's tree reaches a lower fraction of peak than its ring - a NCCL maintainer: "ring is superior in terms of peak bandwidth (except on 2 nodes), tree is superior in terms of base latency" ([NCCL #790](https://github.com/NVIDIA/nccl/issues/790), quoted in [`NCCL_ALGO`](benchmarks/README.md#nccl_algo)). Plausible reasons: an interior node receives from two children at once and must reduce both inputs, which needs more SM (streaming multiprocessor, one of the GPU's ~100+ independent compute cores) and memory bandwidth per byte; the fan-in creates 2:1 incast on a NIC; and the tree edges are longer-distance than ring edges, so they collide more in the fabric. I haven't found NVIDIA documentation that quantifies the split between these, so treat the list as hypotheses.

NCCL builds the tree at node granularity, not GPU granularity: inside a node the GPUs form a chain, and the double binary tree spans the nodes. A maintainer gives the latency models as `2·(G-1)·l + 2·log2(N)·L` for the tree and `N·(G-1)·l + N·L` for one ring traversal ([NCCL #1049](https://github.com/NVIDIA/nccl/issues/1049); `l` intra-node hop latency, `L` network hop latency). The ring all-reduce is two traversals, so `2·(N·(G-1)·l + N·L)` up to the `-1` terms.

Historically NCCL implemented the tree only for all-reduce; the other collectives use rings (or PAT, NVLS, CollNet where available).

### Hierarchical all-reduce

The standard 2-level decomposition for `N` nodes of `G` GPUs:

1. Intra-node reduce-scatter over `G` GPUs: each GPU ends with `n/G` bytes, reduced over its node.
2. Inter-node all-reduce of those `n/G` bytes among the `N` GPUs with the same local rank - one independent all-reduce per rail.
3. Intra-node all-gather.

```
T_hier = 2·(G-1)·α_l + 2·(G-1)/G · n·β_l         # steps 1 and 3
       + T_AR_inter(N, n/G)                        # step 2, e.g. ring:
       = ... + 2·(N-1)·α_L + 2·(N-1)/N · (n/G)·β_L
```

where `β_l = 1/B_nvlink` and `β_L = 1/B_nic_per_gpu`. Note what this does to the inter-node term: each GPU sends `2·(N-1)/N · n/G` through its own NIC, so the node as a whole sends `2·(N-1)/N · n` through its `G` NICs - exactly the bandwidth lower bound for an all-reduce over `N` nodes with node-level bandwidth `G·B_nic`. The inter-node bytes don't shrink, they get spread across all the NICs. Steps 1-3 can be pipelined over chunks, in which case the time is bounded below by the slowest stage rather than the sum.

NCCL's flat multi-channel ring already gets the same inter-node bandwidth term: it builds one ring per NIC (more precisely, per channel, with channels mapped onto NICs), and each ring crosses each node boundary once. What hierarchy buys you is the latency term - `2·(N-1)` network hops instead of `2·(P-1)` total hops - and the freedom to use a different algorithm at each level (NVLS inside, tree or SHARP across). NCCL's `NVLSTree` and `CollNetDirect` are exactly such hierarchical schedules.

### Summary of costs

Per-rank buffer `n`, `p` ranks, contention-free:

| Collective     | Algorithm             | Latency term      | Bandwidth term          |
| :------------- | :-------------------- | :---------------- | :---------------------- |
| reduce-scatter | ring                  | `(p-1)·α`         | `(p-1)/p · n·β`         |
| reduce-scatter | recursive halving     | `log2(p)·α`       | `(p-1)/p · n·β`         |
| all-gather     | ring                  | `(p-1)·α`         | `(p-1)/p · n·β`         |
| all-gather     | recursive doubling    | `log2(p)·α`       | `(p-1)/p · n·β`         |
| all-reduce     | ring                  | `2·(p-1)·α`       | `2·(p-1)/p · n·β`       |
| all-reduce     | Rabenseifner          | `2·log2(p)·α`     | `2·(p-1)/p · n·β`       |
| all-reduce     | recursive doubling    | `log2(p)·α`       | `log2(p) · n·β`         |
| all-reduce     | double binary tree    | `~2·log2(p)·α`    | `~2·n·β`                |
| all-reduce     | in-network (SHARP: switches that do the reduction themselves, see [below](#nvls-nvlink-sharp)) | `~2·depth·α_sw`   | `~n·β`                  |
| broadcast      | binomial tree         | `log2(p)·α`       | `log2(p) · n·β`         |
| broadcast      | pipelined chain       | `(p-2+S)·α`       | `(p-2+S)/S · n·β`       |
| all-to-all     | pairwise              | `(p-1)·α`         | `(p-1)/p · n·β`         |
| all-to-all     | Bruck                 | `log2(p)·α`       | `log2(p)/2 · n·β`       |

The `γ` terms are omitted; add `(p-1)/p · n·γ` to every reducing algorithm except SHARP, where the switch does the arithmetic.

## algbw vs busbw

[nccl-tests](benchmarks/README.md#nccl-tests) reports two bandwidths, and the [`torch-dist-bench.py`](benchmarks/README.md#torch-dist-bench) benchmark here reports the same pair. The definitions are in nccl-tests' [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md):

```
algbw = S / t
busbw = algbw · factor(collective, p)
```

where `S` is the size nccl-tests reports (for all-gather and reduce-scatter that's the full, un-sharded buffer; for all-reduce it's the per-rank buffer) and `t` the elapsed time.

`algbw` is the right number for "how long will my `S`-byte call take", but it isn't comparable across rank counts: a ring all-reduce over 2 ranks moves `1·S` per rank and over 1024 ranks moves `~2·S`, so `algbw` falls by ~2x with `p` even when the links run at full speed. `busbw` divides that out. The factor is the bandwidth term of the bandwidth-optimal algorithm, divided by `S·β`:

| Collective     | Bytes each rank must send (optimal) | busbw factor  | `p=8`  | `p=1024` |
| :------------- | :---------------------------------- | :------------ | -----: | -------: |
| all-reduce     | `2·(p-1)/p · S`                     | `2·(p-1)/p`   |  1.750 |    1.998 |
| reduce-scatter | `(p-1)/p · S`                       | `(p-1)/p`     |  0.875 |    0.999 |
| all-gather     | `(p-1)/p · S`                       | `(p-1)/p`     |  0.875 |    0.999 |
| all-to-all     | `(p-1)/p · S`                       | `(p-1)/p`     |  0.875 |    0.999 |
| broadcast      | `S` (root)                          | `1`           |  1.000 |    1.000 |
| reduce         | `S`                                 | `1`           |  1.000 |    1.000 |
| send/recv      | `S`                                 | `1`           |  1.000 |    1.000 |

The derivation for all-reduce, concretely: the ring moves `2·(p-1)/p · S` bytes per rank through that rank's slowest link at `B`, so `t ≥ 2·(p-1)/p · S/B`, i.e. `algbw ≤ B · p/(2·(p-1))`, i.e. `busbw = algbw · 2·(p-1)/p ≤ B`. So **`busbw` is an estimate of the per-rank, per-direction bandwidth of the bottleneck link under the assumption that the bandwidth-optimal point-to-point algorithm was used**. Compare it to the *unidirectional* per-GPU link rate: 450GBps for NVLink 4, 900GBps for NVLink 5, and for inter-node runs the per-GPU NIC rate (50GBps for a 400Gbps NIC) - not the duplex figure, and not the node aggregate. See [Unidirectional vs Bidirectional (Duplex)](README.md#unidirectional-vs-bidirectional-duplex).

Three ways `busbw` misleads, all consequences of that "assuming" clause:

1. **In-network reduction**. With NVLS or SHARP, each rank sends `~S` rather than `2·(p-1)/p · S`, but nccl-tests still multiplies by `2·(p-1)/p`. That's how an 8x H100 node reports 480GBps `busbw` on a 450GBps link ([SHARP](README.md#sharp), [nccl-tests #153](https://github.com/NVIDIA/nccl-tests/issues/153#issuecomment-1628415956)). The wire-level number for NVLS is closer to `algbw · (1 + 1/p)` - see [NVLS](#nvls-nvlink-sharp).
2. **Hierarchical runs**. On `N` nodes the inter-node bottleneck carries `2·(N-1)/N · S / G` per NIC, not `2·(P-1)/P · S`. `busbw` with `p = P` then measures neither link: it's the bandwidth the per-GPU path would need under a flat ring. For `N` large this is close to the per-GPU NIC rate, which is why inter-node `busbw` is usually compared with it, but it's an approximation, and at small `N` the intra-node links contribute a visible share - see [Inter-node speed depends on intra-node speed](README.md#inter-node-speed-depends-on-intra-node-speed).
3. **Latency-bound sizes**. Below `n_1/2`, `busbw` mostly measures `α`. A curve of `busbw` vs size is really a plot of `S / (α_total + S·β_eff)`.

## NCCL internals

The source of truth is the code (`src/graph/` for topology and search, `src/graph/tuning.cc` for the cost model, `src/device/` for the kernels) and the [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/). Hu et al., [Demystifying NCCL](https://arxiv.org/abs/2507.04786) (2025), is the best secondary source, and is what [comms.md](comms.md#protocols) already draws the protocol table from. Internals change between releases - verify anything below against the version you run.

### Channels

A channel is NCCL's unit of parallelism: one CUDA thread block (CTA - a group of threads scheduled together onto one SM, part of a *kernel*, i.e. one function launched to run on the GPU) on each GPU, with its own ring or tree, its own peer connections and its own FIFO buffers. A collective's buffer is split across channels, each channel runs the algorithm independently on its slice. Multiple channels exist for two reasons:

1. **One SM can't saturate NVLink**. A single CTA's load/store throughput is far below 450-900GBps, so NCCL needs many SMs to drive the links.
2. **Multiple NICs and multiple paths**. Each channel's ring can enter and leave the node through a different NIC; NCCL typically uses at least one channel per NIC, often two.

The channel count is chosen per call by the tuner, and is reduced for small messages (fewer CTAs means less launch and synchronization overhead). That is one reason the measured `busbw` curve ramps more slowly than the one-`α`-one-`β` model predicts: the effective `β` at mid sizes is that of fewer channels. You can bound it with [`NCCL_MIN_CTAS` / `NCCL_MAX_CTAS`](benchmarks/README.md#nccl_min_ctas-and-nccl_max_ctas) (formerly `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`, which are still accepted in the versions I've seen) or per communicator via `ncclConfig_t`. Forcing more channels than the tuner picks rarely helps bandwidth on NVLink and costs SMs; forcing fewer is the standard knob to free SMs for overlapped compute (see [SM contention](#sm-contention)).

### Topology detection and graph search

At `ncclCommInitRank`, NCCL builds a graph of the node: GPUs, NVLink and NVSwitch (a switch chip that connects many GPUs' NVLinks so every pair can talk at full speed) connections (from NVML, NVIDIA's driver query library), PCIe (the standard motherboard bus connecting GPUs, NICs and CPUs) switches and host bridges (from sysfs, the Linux kernel's file-based view of devices under `/sys`), CPUs/NUMA nodes (NUMA - non-uniform memory access - means each CPU socket has its own local memory and devices, and reaching another socket's is slower), and NICs with their speeds. You can dump it with `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml` and override it with `NCCL_TOPO_FILE` - cloud providers ship topology files for VM types where sysfs lies about PCIe placement.

It then searches that graph for rings, trees and NVLS/CollNet patterns, trying to maximize per-channel bandwidth subject to constraints: a ring must enter and leave a node through NICs, prefer a NIC that is on the same PCIe switch as the GPU it serves, avoid crossing the CPU interconnect, and (with [`NCCL_CROSS_NIC=0/2`](benchmarks/README.md#nccl_cross_nic)) use the same NIC index on every node so the ring stays on one rail. `NCCL_GRAPH_DUMP_FILE=/tmp/graph.xml` writes the result; `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH` logs the chosen rings and trees as `Channel 00/16 : 0 1 2 3 4 5 6 7` style lines. If the search finds only low-bandwidth patterns - typically because a GPU and its NIC sit under different host bridges - every collective is capped by that, and no tuning variable will fix it.

### The tuning model

For each `(algorithm, protocol)` pair NCCL estimates

```
time(size) = latency(algo, proto, nRanks, nNodes) + size / bandwidth(algo, proto, topology)
```

and picks the minimum - this is the "model of the latency and bandwidth of each algorithm/protocol combination" described in [NCCL #790](https://github.com/NVIDIA/nccl/issues/790). The latency is built from a per-`(algo, proto)` base latency plus a per-hop hardware latency times the hop count of the algorithm, with separate per-hop values for NVLink, PCIe and the network. The values live in `src/graph/tuning.cc`. As I recall them for recent versions (unverified, and they change across versions), the per-hop numbers are of the order of: NVLink ring hop ~0.6µs `LL` / ~1.9µs `LL128` / ~3.4µs `Simple`; network hop ~2.7µs `LL` / ~4µs `LL128` / ~14µs `Simple`; base latencies of a few to ~15µs. I use those orders of magnitude in the [worked numbers](#worked-numbers) below and label them as assumptions. They are model parameters, not measurements - but they are the parameters NCCL itself uses to choose, so they're the right ones for predicting what NCCL will choose.

A sanity check against measurement: the model's ring `LL` latency on 8 GPUs is `6.6 + 2·7·0.6 = 15.0µs`, and [Latency](benchmarks/README.md#latency) measures 15µs for every all-reduce up to 512KiB on 8x H200. The agreement is closer than the precision of the parameters justifies, so read it as "the right order", not as validation.

`NCCL_ALGO` and `NCCL_PROTO` restrict the set this minimum is taken over. Use them to measure each candidate on your hardware, not to tune production (see [`NCCL_ALGO`](benchmarks/README.md#nccl_algo)). A tuner plugin (`NCCL_TUNER_PLUGIN`) can replace the model; AWS ships one, which is why the B200 sweep in [SHARP](README.md#sharp) switches at a different GPU count than the H200 one.

### Protocols: Simple, LL, LL128

The protocol decides how a receiver knows data has arrived. The table is in [comms.md](comms.md#protocols); here is why it looks the way it does.

**Simple** writes a chunk into the receiver's FIFO slot (a slot in a first-in-first-out ring buffer shared by sender and receiver), issues a memory fence (an instruction that waits until all of this thread's previous writes are visible to other devices before letting later writes proceed), then writes a flag (a step counter). The receiver polls the flag, then reads the chunk. The fence has to wait for all prior writes to become visible, which over NVLink or PCIe costs microseconds - so `Simple` amortizes it over large chunks and reaches near-peak bandwidth, but its `α` is the largest.

**LL** (low latency) removes the fence by making every store self-validating. The FIFO is a sequence of 16-byte lines, each `{data0, flag0, data1, flag1}` of 4 bytes each, written with a single 16-byte volatile store. The flag is the step number; the receiver spins on the line until both flags equal the expected step, then knows both data words are valid. This works because the line is written by one store instruction that the interconnect delivers without tearing (splitting it into parts that could arrive separately), so a matching flag implies the data next to it arrived in the same transaction. Why a flag per 4 bytes of data rather than one per line: the protocol needs every word of data to be validated by a flag that travelled in the same indivisible write, and pairing each 4-byte data word with a 4-byte flag in an 8-byte unit is the arrangement that only relies on 8-byte writes being atomic. That's my reading of the code's structure, not something NVIDIA documents. The cost is that half of every line is flag: 8 data bytes per 16 bytes moved, a 50% ceiling before anything else, and the narrow store pattern pushes the achieved fraction lower again - the "25-50% of peak" in comms.md.

**LL128** applies the same idea to 128-byte lines: 120 bytes of data and 8 bytes of flag, `120/128 = 93.75%` efficiency. A warp writes the line cooperatively, and correctness depends on the interconnect delivering a 128-byte write as a unit, without reordering it against the flag. NVLink provides that; NCCL enables `LL128` only on platforms where it knows it does, and disables it on others (some PCIe/CPU combinations) rather than risk silent corruption. `NCCL_PROTO=^LL128` turning a hang or a wrong result into a correct one is a known diagnostic for exactly this.

**Why protocol choice is size-dependent**: in the `α-β` frame, `LL` has `α ≈ 0.6µs/hop`, `β ≈ 1/(0.5·B)`; `Simple` has `α ≈ 3.4µs/hop`, `β ≈ 1/B`. For 8 GPUs on NVLink 4 with the assumed parameters, `LL` costs `15µs + n/128.6GBps` (algbw) and `Simple` costs `56µs + n/257GBps`, crossing at `n ≈ 10.5MB`. `LL128` fills the middle. The measured switch points are earlier than that (`Ring LL` up to 1MiB then `NVLS Simple` from 2MiB on H200, per [`NCCL_ALGO`](benchmarks/README.md#nccl_algo)) because NVLS joins the candidate set.

### Chunking and pipelining

Each channel has, per peer, a FIFO of `NCCL_STEPS` slots (8 in the versions I've read). For `Simple` the FIFO is `NCCL_BUFFSIZE` bytes (default 4MiB), so a slot is 512KiB; `LL` and `LL128` have their own, smaller buffers. A ring all-reduce is executed as a loop over the buffer in pieces of `nRanks · chunkSize` per channel; within a piece it does the `2·(p-1)` steps from the derivation above, each on one chunk. Up to `NCCL_STEPS` steps can be in flight between a sender and receiver, so as long as the per-hop latency is less than ~8 chunk transfer times, the ring streams at link rate and the `2·(p-1)·α` fill cost is paid once per collective, not once per loop.

That's the condition behind a few practical effects:

- At 50GBps a 512KiB chunk takes 10.5µs, so 8 slots cover ~84µs of round-trip latency. A long-latency path (many switch hops, congested fabric) that exceeds that stalls the pipeline, and `busbw` drops even at large sizes.
- Raising `NCCL_BUFFSIZE` helps high-latency fabrics and costs GPU memory per peer per channel.
- Lowering it, or a very high channel count with small per-channel slices, makes the fill/drain cost a larger share.

### NVLS (NVLink SHARP)

On NVSwitch 3 and later (H100 and newer), the switch can reduce. The GPU-side interface is a multicast object (one memory address that maps to a buffer on every GPU in a group, so a single write reaches all of them) (`cuMulticastCreate`) mapped into every participant, plus the PTX (NVIDIA's GPU assembly language) instructions `multimem.ld_reduce` (the switch reads the same address from every GPU in the group, reduces, and returns the sum), `multimem.st` (one store fanned out to every GPU) and `multimem.red`. That's why NCCL's log says `Created Multicast group` - see [SHARP](README.md#sharp).

An NVLS all-reduce is, per GPU: `multimem.ld_reduce` its `n/p` shard (the switch pulls `n/p` from each of the `p` GPUs and returns `n/p`), then `multimem.st` the result (the GPU sends `n/p`, the switch delivers it to all `p`). Per GPU per direction:

```
sent     = n/p · (p contributions spread over p shards → n) + n/p (store)   = n·(1 + 1/p)
received = n/p (reduced shard) + n (broadcast results of all p shards)      = n·(1 + 1/p)
```

That counts the GPU's own contribution and own copy as crossing its NVLink. If the switch doesn't need them to (I don't know whether it does - the PTX documentation describes semantics, not wire traffic), it's `n·(p-1)/p` instead. So the ideal time is between `n·(p-1)/p · β` and `n·(p+1)/p · β`, against `2·(p-1)/p · n·β` for the ring - an ideal speedup between `1.56x` (`2·7/9`) and `2.0x` for `p = 8`.

The measured speedup on 8x H200 is `480/367 = 1.31x` ([SHARP](README.md#sharp)). For a 1GB all-reduce that's 3.65ms measured against an ideal of 1.94-2.5ms. The gap is plausibly the switch's reduction throughput (NVIDIA quotes 450GBps of "reduction bandwidth" for the DGX H100 NVSwitch fabric, which is suspiciously close to one GPU's link rate) plus the SMs NCCL dedicates to NVLS - I have not found a source that pins it down.

`NVLSTree` composes NVLS inside the node with the inter-node tree; it [requires IB (InfiniBand) or RoCE](https://github.com/NVIDIA/nccl/issues/1031#issuecomment-1773965518). On rack-scale systems (NVL72) NVLS spans the whole NVLink domain.

### CollNet and IB SHARP

CollNet is NCCL's interface to in-network reduction on the scale-out fabric, implemented through a plugin (for InfiniBand SHARP, NVIDIA's `nccl-rdma-sharp-plugins`). Two variants:

- **CollNetDirect**: every GPU in a node talks to the network directly - intra-node reduce-scatter, each GPU sends its `n/G` shard into the SHARP aggregation tree via its own NIC, the switches reduce across nodes and return `n/G`, then intra-node all-gather.
- **CollNetChain**: GPUs in a node form a chain to one NIC, for systems with fewer NICs.

Cost of the inter-node part for CollNetDirect: each GPU sends `n/G` up and receives `n/G` down, concurrently in opposite directions, plus the aggregation tree depth:

```
T_inter_SHARP ≈ 2·depth·α_switch + (n/G)·β_L
T_inter_ring  ≈ 2·(N-1)·α_L     + 2·(N-1)/N · (n/G)·β_L
```

So asymptotically SHARP halves the inter-node bandwidth term (the "N+1 sends instead of 2N" in [SHARP](README.md#sharp)) and replaces the `O(N)` latency with `O(depth)`, where depth is 2-3 for a fat-tree. Constraints that make the speedup smaller in practice: the switch's aggregation throughput and the number of concurrent SHARP trees/jobs it supports are finite, SHARP resources must be allocated by the subnet manager / aggregation manager, and the supported reduction dtypes and ops are limited. Treat a 2x inter-node gain as the ceiling.

### PXN

PXN (PCI x NVLink, NCCL 2.12) lets a GPU send to a remote GPU through a *different* local GPU's NIC: data goes over NVLink to the GPU whose NIC is on the destination's rail, and that GPU's proxy (a CPU helper thread NCCL runs per GPU to drive the NIC) posts the RDMA (remote direct memory access: the NIC reads or writes the remote machine's memory directly, without involving the remote CPU). Two benefits ([Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/)):

1. **Rail alignment**: traffic from GPU `i` on node A to GPU `j` on node B can travel on rail `j` the whole way, instead of entering rail `i` and crossing to rail `j` in the spine.
2. **Message aggregation**: messages from all local GPUs to the same remote NIC can be combined, reducing the number of in-flight connections and small RDMA ops.

This matters mostly for all-to-all and point-to-point patterns, where without PXN every GPU would talk to every remote NIC. `NCCL_PXN_DISABLE=1` turns it off; it trades NVLink bandwidth (cheap) for spine bandwidth (expensive, possibly oversubscribed).

### Buffer registration and symmetric memory

By default NCCL copies user data into its own FIFO buffers and the peer copies it out - an extra pass through HBM and, for the network, the proxy works from NCCL-owned registered memory. User buffer registration (`ncclCommRegister`, or automatic registration of buffers captured in CUDA graphs) lets NCCL work directly on the user buffer: for NVLS the user buffer is mapped into the multicast object, for IB it's registered with the NIC, avoiding the copy and freeing SMs. PyTorch exposes this through its NCCL memory pool APIs. Requirements and limitations (allocation through `ncclMemAlloc` / cuMem APIs, which collectives benefit) vary by version - check the [NCCL user buffer registration docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html).

Symmetric memory (NCCL 2.27+, `ncclCommWindowRegister`) goes further: every rank registers a buffer at the same offset of a collective window, so a kernel can address any peer's buffer directly over NVLink without a handshake. This removes per-call synchronization overhead, which is why its gains are largest at small sizes - see [Symmetric memory](README.md#symmetric-memory) for the measurements. In `α-β` terms, it attacks `α`, not `β`.

## Topology

### NVLink and NVSwitch generations

Per-GPU unidirectional NVLink bandwidth, from the table in [NVLink](README.md#nvlink) (NVIDIA's [NVLink page](https://www.nvidia.com/en-us/data-center/nvlink/)):

| Generation | GPU              | Links | Uni-dir. per GPU | Switch domain                    |
| :--------- | :--------------- | ----: | ---------------: | :------------------------------- |
| NVLink 3   | A100             |    12 |          300GBps | 8 GPUs (NVSwitch 2)              |
| NVLink 4   | H100/H200        |    18 |          450GBps | 8 GPUs, up to 256 with NVLink Switch System |
| NVLink 5   | B200/GB200/B300  |    18 |          900GBps | 8 (HGX) or 72 (NVL72)            |
| NVLink 6   | Rubin (announced)|    36 |         1500GBps | announced                        |

Note that the "900GBps" usually quoted for NVLink 4 is the duplex sum; the number that goes into `β` for a collective is 450GBps. NVL72: 72 GPUs x 1.8TBps duplex = 129.6TBps, the "130TB/s" NVIDIA quotes for [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/). On an NVL72, a ring all-reduce over 72 GPUs has bandwidth term `2·71/72 · n/900GBps`, i.e. the full NVLink rate at 9x the scale of an HGX node, and the inter-node problem starts at the rack boundary instead of the board.

### PCIe, NUMA and GPUDirect RDMA

The inter-node path for a GPU's data is: HBM → GPU's PCIe interface → PCIe switch → NIC → wire, with GPUDirect RDMA letting the NIC DMA (direct memory access - a device copying memory without the CPU) to and from GPU memory without staging through host memory. Its bandwidth is the minimum along the path, and the path is where most "the NIC is fine but NCCL is slow" problems live:

- **GPU and NIC under the same PCIe switch** (`PIX`/`PXB` in `nvidia-smi topo -m`): peer-to-peer traffic stays inside the switch. DGX-style nodes pair each GPU with a NIC this way.
- **Through the CPU root complex** (`PHB`) or **across sockets** (`SYS`): peer-to-peer either degrades or is disabled, and NCCL may fall back to staging through host memory. `NCCL_NET_GDR_LEVEL` sets the maximum distance at which NCCL will use GPUDirect RDMA.
- **PCIe ACS** enabled on switches forces peer-to-peer traffic up to the root complex for IOMMU checks; see [Disable Access Control Services](benchmarks/README.md#disable-access-control-services).

A 400Gbps NIC needs 50GBps, which a PCIe 5 x16 link (~64GBps per direction raw) can carry; an 800Gbps NIC (100GBps) needs PCIe 6 x16 or two x16 links. If GPU and NIC share an upstream PCIe link, their traffic can contend there.

### Rail-optimized vs non-rail fat-tree

A *non-blocking* fat-tree has enough parallel leaf-to-spine uplinks that any half of the nodes can talk to the other half at full speed. In a **rail-optimized** network, NIC `i` of every node connects to leaf switch(es) `i` - rail `i`. Traffic between same-index GPUs on different nodes needs one switch hop if the rail fits in one leaf, and never touches the spine. NCCL's rings (with `NCCL_CROSS_NIC` defaulting to "prefer same NIC") and the per-rail inter-node step of a hierarchical all-reduce produce exactly that traffic, so all-reduce, reduce-scatter and all-gather can avoid the spine almost entirely. Cross-rail traffic (all-to-all, pipeline sends between different local ranks) either goes through the spine or, with [PXN](#pxn), hops over NVLink first.

A fat tree - links get "fatter" towards the root so bisection bandwidth is preserved:

![fat tree network](images/fat-tree-network.png)

([source](https://commons.wikimedia.org/wiki/File:Fat_tree_network.svg))

In a **non-rail** (top-of-rack) fat-tree all NICs of a node land on the same leaf, and inter-node traffic between racks goes through the spine, where ECMP (hash-based choice among equal-cost paths) decides which uplink it takes (see [Flow collisions and ECMP](#flow-collisions-and-ecmp)). `NCCL_CROSS_NIC=1` is the hint for this design.

**Oversubscription** `o:1` at a tier means the downlink capacity is `o` times the uplink capacity. For a flow that crosses that tier the bound becomes `B_nic / o` whenever all downlinks are busy. A rail-optimized fabric can be oversubscribed at the spine with little cost to all-reduce-dominated training, because rings don't use it; the same fabric can halve all-to-all throughput. So whether oversubscription matters depends on the collective mix, not just on the fabric: for MoE training with expert parallelism across nodes, compute the all-to-all bound with the bisection bandwidth, not the NIC bandwidth.

### How topology changes NCCL's choices

Some consequences of the graph search and tuning model, all visible in `NCCL_DEBUG_SUBSYS=GRAPH,TUNING` output:

- **PCIe-only nodes** (no NVLink): fewer channels, `LL128` often disabled, rings ordered to minimize host bridge crossings; `busbw` capped by the slowest PCIe hop.
- **NVSwitch nodes**: all GPU pairs equidistant, so any ring order works; NVLS becomes a candidate from ~5-6 GPUs up (measured switch points in [SHARP](README.md#sharp)).
- **NIC count**: sets the minimum useful channel count for inter-node work and how many rings can cross the node boundary in parallel. A node with 8 GPUs but 4 NICs gets half the inter-node bandwidth per GPU, and the tuner knows it.
- **Node count**: shifts the ring/tree boundary to larger sizes as `N` grows (see the crossover table in [Worked numbers](#worked-numbers)).
- **SHARP availability**: adds `CollNet*`/`NVLSTree` to the candidate set.

## Transports

### InfiniBand

InfiniBand is a network technology built for RDMA, separate from Ethernet. It is lossless by construction: link-level, credit-based flow control means a sender never transmits unless the receiver has buffer space, so packets are not dropped under congestion - they wait. Routing is computed by the subnet manager. NVIDIA's IB switches support **adaptive routing**, which picks among equal-cost output ports per packet based on queue occupancy rather than a static hash; the NIC handles the resulting out-of-order delivery for RDMA. Adaptive routing turns "two flows hashed onto the same uplink" from a persistent 2x slowdown into transient queueing - see [InfiniBand adaptive routing](benchmarks/README.md#infiniband-adaptive-routing). IB also has congestion control (FECN/BECN-based), but tuning it is rarely exposed to users.

The NCCL-visible failure semantics come from RC (reliable connected) queue pairs - a QP is the pair of send/receive queues the NIC keeps for one RDMA connection: an RDMA op that isn't acknowledged within the local ACK timeout is retransmitted, up to a retry count, after which the QP goes to error and NCCL reports a fatal error. `NCCL_IB_TIMEOUT` sets the exponent: the timeout is `4.096µs · 2^NCCL_IB_TIMEOUT`, so 20 gives ~4.3s and 22 gives ~17.2s per attempt, multiplied by `NCCL_IB_RETRY_CNT` retries. (I believe the default timeout exponent is 20 in recent versions; older versions used a lower value - check your version's docs.)

### RoCEv2

RoCEv2 (RDMA over Converged Ethernet) carries the IB transport inside ordinary UDP/IP packets (UDP destination port 4791), so it runs on Ethernet switches - but the RC transport assumes near-lossless delivery, because its loss recovery (go-back-N in most NICs) is expensive. Two mechanisms make Ethernet behave:

- **PFC** (Priority Flow Control, 802.1Qbb): a switch whose ingress queue for a priority fills sends a PAUSE upstream for that priority. Lossless, but coarse: a PAUSE stops *all* traffic of that class on the link, not just the flow causing it, so congestion spreads backwards (head-of-line blocking, congestion trees), and under pathological conditions PFC pause storms and deadlocks are possible.
- **ECN + DCQCN**: switches mark packets when queues exceed a threshold; the receiver NIC returns congestion notification packets; the sender NIC cuts its rate multiplicatively and recovers over time ([Zhu et al., Congestion Control for Large-Scale RDMA Deployments](https://doi.org/10.1145/2785956.2787484), SIGCOMM 2015). The idea is to throttle before PFC triggers, keeping PFC as a last-resort backstop.

The tuning surface is large (ECN thresholds, DCQCN rate parameters, PFC headroom, traffic classes - `NCCL_IB_TC`, `NCCL_IB_GID_INDEX` select class and RoCE version), and mistuning shows up as low or erratic collective bandwidth rather than errors. Meta describes running RoCE for training at scale, including that DCQCN tuning was not the main lever for them and that collective-library-level flow control and routing mattered more ([Gangidi et al., RDMA over Ethernet for Distributed AI Training at Meta Scale](https://doi.org/10.1145/3651890.3672233), SIGCOMM 2024). Vendor Ethernet fabrics for AI (Spectrum-X, see [README](README.md#spectrum-x-ethernet)) add per-packet adaptive routing on the switch and reordering in the NIC, i.e. they import the IB approach.

### Flow collisions and ECMP

Without adaptive routing, a switch with several equal-cost uplinks chooses one per flow (a flow is the sequence of packets of one connection) by hashing header fields - this is ECMP, equal-cost multi-path routing - (for RoCEv2, typically the 5-tuple including the UDP source port, which the NIC derives from the QP). Collective traffic has **low entropy**: a ring all-reduce over `N` nodes with 8 channels is a handful of long-lived, equally sized elephant flows. Hashing `k` such flows onto `k` uplinks:

- `P(no collision) = k!/k^k` - 0.24% for `k = 8`, ~1e-6 for `k = 16`.
- Expected load on the most loaded uplink (simulated, 20k trials): 2.59 flows for `k = 8`, 3.07 for `k = 16`.

A ring is synchronous - every step waits for the slowest hop - so the whole collective runs at the rate of the most-shared link. With 2.59 flows sharing the worst uplink, the expected bandwidth is `1/2.59 ≈ 39%` of the NIC rate, for the whole ring, until the flows are re-hashed. That's the mechanism behind "same job, same nodes, `busbw` varies 2x between runs" on ECMP fabrics.

Mitigations, all of which increase entropy or remove hashing:

- [`NCCL_IB_QPS_PER_CONNECTION`](benchmarks/README.md#nccl_ib_qps_per_connection): split each connection over `q` QPs (each with its own UDP source port), each carrying `1/q` of the traffic. Simulated expected worst-uplink load drops from 2.59 to 1.76 (`q = 4`) and 1.37 (`q = 16`) flows' worth for `k = 8`. Costs QP state and some GPU memory.
- `NCCL_IB_SPLIT_DATA_ON_QPS`: whether NCCL round-robins messages over those QPs or splits each message across them (semantics vary by version - check the env docs).
- Adaptive routing (IB, Spectrum-X and similar), which doesn't depend on the hash at all.
- Rail-optimized topology, which keeps ring traffic below the tier where the hashing happens.

The simulation is a balls-into-bins model with uniform hashing; real hash functions on real headers can be better or worse, so treat these as typical rather than predictive.

## Compute-communication overlap

The overlap story in [Comms and compute overlap](README.md#comms-and-compute-overlap) is the timeline view. This section is what makes overlap less than perfect even when the timeline says it should be.

### Streams and what "async" means

A NCCL collective is a kernel. PyTorch launches it on a dedicated NCCL stream (a CUDA stream is a queue of GPU work that runs in order; work in different streams may run concurrently) per process group (the set of ranks a collective runs over), with event-based dependencies on the compute stream (input ready) and back (`work.wait()` makes the compute stream wait for the NCCL stream). Overlap happens when the GPU has kernels from both streams ready at once and enough resources to run them concurrently. Three ways it silently doesn't:

1. A dependency forces serialization (you `wait()` too early, or the bucket isn't ready until the last layer's backward).
2. The collective's CTAs can't be scheduled because compute kernels occupy all SMs - the NCCL kernel starts late or runs with fewer resident CTAs.
3. Both run, but compete for HBM bandwidth and L2, so both slow down.

### SM contention

Each NCCL channel is a CTA occupying (part of) an SM for the whole duration of the collective, spinning on flags when it has nothing to do. An H100 SXM has 132 SMs, so 16 channels is ~12% and 32 channels ~24% of the GPU unavailable to the GEMMs (general matrix multiplies - the matmul kernels) you're overlapping with - and the GEMM's tile scheduling is computed for the full SM count, so losing 12% of SMs can cost more than 12% through wave quantization. The opposite case is also common: a large GEMM fills every SM, the NCCL kernel can't get its CTAs resident, and the collective is delayed until the GEMM's waves drain.

Knobs: [`NCCL_MIN_CTAS` / `NCCL_MAX_CTAS`](benchmarks/README.md#nccl_min_ctas-and-nccl_max_ctas) or per-process-group `ncclConfig_t`; giving the communication stream a higher priority; frameworks that cap the GEMM's SM count (cuBLAS / CUTLASS SM-count hints) to leave room. The cost model: if a collective with `c` channels at full speed takes `T_c`, and the overlapped GEMM takes `T_g` on all SMs, the best case with SM partitioning is `max(T_c(c), T_g · SMs/(SMs - c))` - and you search over `c`.

### Copy-engine collectives

GPUs have DMA copy engines - dedicated hardware units for memory copies - that move data over NVLink without using SMs. A collective built from `cudaMemcpyAsync` between peer buffers (or symmetric memory mappings) uses no SM time, which is attractive for overlap. The constraints: copy engines can't reduce, so this works for all-gather and all-to-all (and broadcast), not for reduce-scatter or all-reduce; per-copy launch overhead makes them poor at small sizes; and the number of copy engines and their per-engine bandwidth limit how much of the NVLink you can drive. PyTorch's symmetric memory module has copy-engine based low-contention all-gather/reduce-scatter variants, and recent NCCL releases added copy-engine based collectives for some operations - (both unverified in detail; check the release notes of your PyTorch and NCCL versions).

### Async TP / collective matmul

Tensor parallelism with sequence parallelism has, in every layer, `Y = AllGather(X) · W` (column-parallel) and `Z = ReduceScatter(Y · W')` (row-parallel), where the collective is on the critical path of the GEMM that consumes or produces it. Bucketing can't overlap these - the GEMM needs the gathered data. Decomposition can: split the all-gather into its `p` ring steps and run the GEMM on each shard as it arrives:

```
for k in 0..p-1:
    Y[shard (r-k) mod p] = X_shard_current @ W      # compute on what we have
    X_shard_current = recv_from(r-1) while send_to(r+1)  # overlapped with the GEMM
```

This is the collective matmul in Wang et al., [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://doi.org/10.1145/3567955.3567959) (ASPLOS 2023), and PyTorch's async TP ([Distributed w/ TorchTitan: Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)), which implements the transfers with symmetric memory and copy engines so they don't take SMs from the GEMM.

Cost model, with `t_g = T_gemm/p` per shard and `t_c = (n/p)·β` per shard transfer:

```
T_sequential = (p-1)·t_c + p·t_g
T_decomposed ≈ t_g + (p-1)·max(t_g, t_c)       # first shard is local, then pipeline
```

Worked example, TP=8 on H100 for the fused gate+up projection of a 70B-class MLP (hidden 8192, FFN 28672): `X` is `8192 tokens x 8192 x 2B = 134MB` gathered; the local GEMM is `2·8192·8192·(2·28672/8) = 9.6e11` FLOP per rank, at an assumed 70% of the 989 TFLOPS dense BF16 peak (H100 SXM datasheet) that's 1.39ms. The all-gather receives `7/8 · 134MB = 117MB` at an assumed 80% of 450GBps: 0.33ms. Sequential 1.72ms; decomposed, since `t_c < t_g`, ~1.39ms - the all-gather is fully hidden, a 19% saving on this op. The real saving is smaller: eight GEMMs with `M = 1024` each run at lower efficiency than one with `M = 8192` (wave quantization, less reuse), and concurrent NVLink traffic takes HBM bandwidth from the GEMM. Whether the net is positive is an empirical question per shape; the model tells you the ceiling (here 19%) and that it shrinks as `t_c/t_g` shrinks - i.e. async TP pays most at small token counts per rank and high TP degree, where the GEMM is short relative to the transfer.

### Verifying overlap in a trace

Overlap is a claim about a timeline, so check the timeline. With the PyTorch profiler or Nsight Systems (`nsys profile --trace=cuda,nvtx`):

1. Find the NCCL kernels (`ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, `..._ReduceScatter_..._RING_SIMPLE` etc. - the name carries collective, op, dtype, algorithm and protocol, which is also a cheap way to see NCCL's choice without debug logs) and the compute kernels they're meant to overlap with.
2. Check that they actually run concurrently on different streams, not back-to-back.
3. Compare each kernel's duration overlapped vs in isolation. If the NCCL kernel is 2x longer when overlapped, the collective is being starved (SMs or HBM); if the GEMM is 15% longer, that's the cost of the overlap, and it belongs in the accounting.
4. Measure **exposed communication**: the time a NCCL kernel runs with no compute kernel running concurrently. That, not total NCCL time, is what overlap failed to hide. Meta's [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) computes communication-computation overlap from PyTorch profiler traces.

A NCCL kernel's duration includes the time it spends waiting for the slowest peer to arrive, so a "slow all-reduce" in one rank's trace is often a straggler on another rank - look at the same collective across ranks before concluding the network is slow.

## Worked numbers

Assumptions, stated once:

- Payload `n = 1GB` (10^9 bytes = 5·10^8 bf16 elements), per rank.
- NVLink 4: 450GBps per GPU per direction (the "900GBps" is the duplex sum).
- Inter-node: 128 nodes x 8 GPUs, 8x 400Gbps IB per node, rail-optimized, non-blocking. 400Gbps = 50GBps per GPU, 400GBps per node per direction.
- Per-hop latencies of the order of NCCL's tuning model (assumed, see [The tuning model](#the-tuning-model)): `Simple` ring hop 3.4µs on NVLink, 14µs over the network, base 8.4µs; tree NVLink hop 4µs, network hop 14µs.
- Contention-free, homogeneous, and link-rate bandwidth unless stated - so these are bounds, not predictions. Real `busbw` is typically ~80-90% of the link rate at best ([Real network throughput](README.md#real-network-throughput)).

### 8 GPUs, one NVLink 4 node

| Algorithm      | Latency term                  | Bandwidth term                          | Total    |
| :------------- | :---------------------------- | :-------------------------------------- | -------: |
| ring (ideal)   | `14 x 3.4µs = 48µs`           | `1.75 x 1GB / 450GBps = 3.89ms`         |  3.94ms  |
| ring at measured 367GBps `busbw` | -           | `1.75 x 1GB / 367GBps`                  |  4.77ms  |
| NVLS (ideal)   | small                         | `(7/8 ... 9/8) x 1GB / 450GBps`         | 1.94-2.50ms |
| NVLS at measured 480GBps `busbw` | -           | `1.75 x 1GB / 480GBps`                  |  3.65ms  |

(The measured `busbw` values are the 8x H200 numbers from [SHARP](README.md#sharp); H100 and H200 share NVLink 4.)

A 1GB all-reduce is deep in the bandwidth regime on one node: latency is ~1% of the ring's time. The NVLink tree is not listed because inside one NVSwitch node NCCL's tree degenerates to a chain, which has no advantage there.

Where the regime changes, for the 8-GPU ring:

- `Simple`: `α_total = 8.4 + 14·3.4 = 56µs`, asymptotic `algbw = 450/1.75 = 257GBps`, so `n_1/2 = 56µs · 257GBps ≈ 14.4MB`.
- `LL`: `α_total = 15µs`, `algbw ≈ 129GBps` (50% efficiency), `n_1/2 ≈ 1.9MB`.

The measured 8x H200 curve in [How the bandwidth is measured](benchmarks/README.md#how-the-bandwidth-is-measured) reaches half its ~480GBps peak at ~16MiB (`busbw` 256GBps), consistent with the `Simple` estimate - again, closer than the inputs justify.

### 1024 GPUs, 128 nodes, 8x 400Gbps IB per node

`P = 1024`, `N = 128`, `G = 8`.

**Flat ring** (8 rings, one per NIC, each crossing every node boundary once):

```
bandwidth: 2·(P-1)/P · n / (G·50GBps) = 1.998 · 1GB / 400GBps      = 5.00ms
NVLink side of the same rings: 7/8 · 1.998 · 1GB / 450GBps        = 3.89ms  (not the bottleneck)
latency:   8.4µs + (2·1023 - 2·127)·3.4µs + 2·127·14µs
         = 8.4 + 6093 + 3556 µs                                    = 9.66ms
total                                                              ≈ 14.7ms
```

The ring's fill latency is twice its bandwidth term. Only 34% of the 1GB all-reduce's time is spent moving data at link rate, and the ring's `n_1/2` here is ~1.9GB.

**Double binary tree** (chain inside the node, double binary tree over nodes):

```
bandwidth: 2 · n / 400GBps                        = 5.00ms
latency:   2·7·4µs + 2·log2(128)·14µs = 56 + 196  = 252µs
total                                             ≈ 5.25ms   (n_1/2 ≈ 50MB)
```

**Hierarchical**, NVLink reduce-scatter / per-rail inter-node all-reduce of `n/8` / NVLink all-gather:

```
intra RS (and AG):  7·3.4µs + 7/8 · 1GB / 450GBps          = 1.97ms each
inter, ring:        2·127·14µs + 2·127/128 · 125MB / 50GBps = 3.56 + 4.96 = 8.52ms
inter, tree:        2·7·14µs   + 2 · 125MB / 50GBps         = 0.20 + 5.00 = 5.20ms
inter, SHARP:       ~depth·α  + 125MB / 50GBps              ≈ 2.50ms
```

| Schedule                          | Sequential (no stage pipelining) | Pipelined bound (slowest stage) |
| :-------------------------------- | -------------------------------: | ------------------------------: |
| flat ring                         |                          14.7ms  |                         14.7ms  |
| double binary tree                |                          5.25ms  |                          5.25ms |
| hierarchical, ring inter-node     |                          12.5ms  |                          8.52ms |
| hierarchical, tree inter-node     |                          9.13ms  |                          5.20ms |
| hierarchical, SHARP inter-node    |                          6.44ms  |                          2.50ms |

What this says:

1. **At 1024 GPUs the flat ring is latency-bound even at 1GB.** That's the textbook reason NCCL uses trees at scale for all-reduce, and the inter-node bandwidth floor of `2·(N-1)/N · n / 400GBps ≈ 5ms` is the same for every non-SHARP schedule - they differ only in latency and in how well the stages overlap.
2. **Hierarchy alone doesn't save bandwidth; it saves latency.** The inter-node bytes per node are the same; what changes is `2·(N-1) = 254` network hops instead of `2·(P-1) = 2046` total.
3. **Only in-network reduction moves the bandwidth floor**, by up to 2x on the inter-node part. Whether it reaches 2x depends on switch aggregation throughput, which I don't have a reliable figure for.
4. All of these assume a non-blocking, collision-free fabric. A single ECMP collision on a ring path in a non-adaptive fabric multiplies the ring's bandwidth term by the collision degree (see [Flow collisions and ECMP](#flow-collisions-and-ecmp)).

### The ring/tree crossover

With equal bandwidth terms the tree wins at all sizes; the crossover only exists because the tree reaches a lower fraction of peak. Write the tree's achieved bandwidth as `η` times the ring's and solve `n* = (α_ring - α_tree) / (2/(η·B_node) - 2·(P-1)/P / B_node)`:

| Nodes | Ring latency | Tree latency | `n*` at `η=0.9` | `η=0.7` | `η=0.5` |
| ----: | -----------: | -----------: | --------------: | ------: | ------: |
|     4 |        283µs |        112µs |           240MB |    74MB |    33MB |
|    16 |       1190µs |        168µs |          1.7GB |   468MB |   203MB |
|   128 |       9657µs |        252µs |           17GB |   4.4GB |   1.9GB |

The one measured data point in this repo: on 4x P6-B200 nodes NCCL switches from `Tree LL128` to `Ring LL128` at 32MiB ([Inter-node speed depends on intra-node speed](README.md#inter-node-speed-depends-on-intra-node-speed)). With the assumed latencies, that corresponds to `η ≈ 0.5` - a 2x lower tree bandwidth - or to latencies different from the ones assumed, or both; on EFA, with AWS's tuner plugin, the inputs are not NCCL's defaults anyway. The qualitative conclusion is robust to the parameters: the crossover moves up roughly in proportion to `N`, so at 100+ nodes the tree (or NVLSTree / SHARP) is the right all-reduce algorithm for every gradient bucket size people actually use.

## Failure modes in cost-model terms

### Stragglers

A synchronous collective completes when the last rank arrives, so its effective latency is `α + max_i(d_i)`, where `d_i` is rank `i`'s arrival delay (compute jitter, OS noise, a GC pause, a thermally throttled GPU). For iid Gaussian jitter with standard deviation `σ`, `E[max]` over `p` ranks (simulated): `1.42σ` at `p = 8`, `2.60σ` at `p = 128`, `3.24σ` at `p = 1024` - growing like `sqrt(2·ln p)` asymptotically (3.72 at `p = 1024`; the bound converges slowly). Two consequences:

- The expected straggler penalty per collective grows with scale even if no individual rank gets worse, and it's paid on every synchronization point - per bucket, per layer for TP.
- One persistently slow rank (not jitter, a slow GPU) sets the pace for everyone, and appears in every *other* rank's trace as a long NCCL kernel. The slow rank itself shows a short NCCL kernel - it's the last to arrive, so it waits least.

### A slow link dragging a ring

In a ring each step waits for the slowest hop, so the bandwidth term is set by `min_i(B_i)`, not the mean:

```
T_ring_bw = 2·(p-1)/p · n / min_i(B_i)
```

One link at half speed - a NIC that negotiated down, a flapping cable retraining at a lower width, a PCIe link at x8, an ECMP collision - halves the bandwidth of the whole ring, i.e. of every collective whose ring traverses it. With 8 channels on 8 NICs, one bad NIC slows the one ring that uses it to 50%, and since the buffer is split evenly across channels, that ring finishes last and the collective takes ~2x as long. The tree is less sensitive in aggregate (a slow link only slows the subtree behind it), but still gated by the worst path. In a hierarchical schedule only the affected rail's inter-node all-reduce is slow - still the collective's critical path.

The diagnostic signature: `busbw` ~1/k of normal, reproducible, independent of message size above `n_1/2`, and dependent on which nodes are in the job.

### Timeouts and hangs

Distinguish three layers:

- **Transport timeout**: an RDMA op not acknowledged after `4.096µs · 2^NCCL_IB_TIMEOUT · NCCL_IB_RETRY_CNT` (see [InfiniBand](#infiniband)) - the QP errors, NCCL reports an error, the job dies. This is a link or fabric failure.
- **Collective hang**: ranks disagree about which collective comes next (different call order, a rank skipped a collective, mismatched sizes). Nothing is lost on the wire; every rank waits forever on a flag that will never be written. NCCL can't detect this.
- **Framework watchdog**: PyTorch's `ProcessGroupNCCL` watchdog (a background thread that checks for collectives that never complete) aborts collectives exceeding the process group `timeout` (10 minutes by default for NCCL). `TORCH_NCCL_TRACE_BUFFER_SIZE` enables the flight recorder, which dumps the last collectives each rank issued - comparing sequence numbers across ranks finds the one that diverged.

A timeout is the only way a hung collective becomes visible, so the timeout value is a trade-off between false aborts during long legitimate waits (checkpointing on one rank, a slow first iteration with compilation) and wasted GPU time when a job really is hung.

### Localizing with per-pair tests

The cost model gives the strategy: a collective over a set of ranks is as slow as its worst link, so test subsets small enough that each test exercises few links.

1. **Within a node**: `nvbandwidth` or `p2pBandwidthLatencyTest` ([nvbandwidth](benchmarks/README.md#nvbandwidth)) for every GPU pair; an 8-GPU `all_reduce_perf` should reach the expected `busbw` for its NVLink generation.
2. **Per NIC**: `ib_write_bw` between the same NIC index on two nodes, for each of the 8 rails - tests the NIC, cable and rail switch path without NCCL.
3. **Per node pair**: a 2-node `all_reduce_perf`. All 8128 pairs of 128 nodes is too many to run serially, but on a non-blocking fabric `N/2` disjoint pairs can run at once, and a round-robin tournament schedule covers all pairs in `N-1 = 127` rounds. Usually you don't need all of them: pair node `i` with a known-good reference node, or run one round of `N/2` pairs and bisect any slow pair against two known-good nodes.
4. **Bisect the full job**: run the collective over halves of the node set; the half that's slow contains the bad link. `log2(128) = 7` rounds to a node, if the problem is a single node rather than a spine link.

Spine problems don't localize to a node: they show up as pairs that are slow only when they cross a particular leaf boundary, so record which leaf each node is on before you start.

## Further reading

Algorithms and cost models:

- R. Thakur, R. Rabenseifner, W. Gropp, [Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521), IJHPCA 2005 - the alpha-beta-gamma derivations for ring, recursive doubling/halving and Rabenseifner's algorithm.
- P. Patarasuk, X. Yuan, [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://doi.org/10.1016/j.jpdc.2008.09.002), JPDC 2009 - the all-reduce bandwidth lower bound.
- P. Sanders, J. Speck, J. L. Träff, [Two-tree algorithms for full bandwidth broadcast, reduction and scan](https://doi.org/10.1016/j.parco.2009.09.001), Parallel Computing 2009 - the double binary tree.
- E. Chan, M. Heimlich, A. Purkayastha, R. van de Geijn, [Collective communication: theory, practice, and experience](https://doi.org/10.1002/cpe.1206), CCPE 2007 - lower bounds and the scatter+all-gather family.

NCCL:

- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) and its [environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) page.
- Z. Hu et al., [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786), 2025.
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

In this repo: [Communication Patterns](comms.md), [Inter-Node and Intra-Node Networking Hardware](README.md), [Network Benchmarks](benchmarks/README.md), [Network Debug](debug/README.md).
