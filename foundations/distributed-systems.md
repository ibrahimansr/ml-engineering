# Foundations: Distributed Systems for ML Engineers

This chapter is for readers who know the ML math and use PyTorch every day, but who have never had to think about programs that run on many machines at once. You can get a long way on one GPU without that knowledge. You cannot train a frontier model without it. The first time a 512-GPU job hangs with no error message, or one slow node drags the whole cluster down, or two ranks quietly end up with different weights, you are debugging a distributed system, whether you meant to build one or not.

The chapter starts from zero and works up to what you need to read the [model parallelism](../training/model-parallelism/README.md), [fault tolerance](../training/fault-tolerance/README.md) and [orchestration](../orchestration/README.md) chapters, and after that the more advanced [parallelism derivations](../training/model-parallelism/derivations.md) and [orchestration internals](../orchestration/internals.md). The emphasis is on *why* things are built the way they are. Every odd-looking design in a distributed training stack (rendezvous, barriers, watchdog timeouts, the restart-the-whole-job reflex) exists because of one or two facts about distributed systems. Once you know those facts, the designs stop looking odd.

All arithmetic in this chapter was checked with `python3`. The runnable example at the end was run with PyTorch 2.15 on macOS with 2 and 4 CPU processes. Where a number depends on hardware, software version or an assumption, the text says so.

Companion chapters you may want open alongside this one: [Concurrency and Parallelism](./concurrency.md) (processes, threads, the GIL) and [Computer Networking](./networking.md) (TCP, ports, RDMA, and an intro to collectives).


## Why distribute at all

There are exactly two reasons to spread one training job across many accelerators:

1. **Memory:** the job does not fit on one device.
2. **Time:** the job fits, but would take too long.

Both are easy to put numbers on. Doing it once by hand is the best way to see why "just use a bigger GPU" stopped being an answer years ago.

### Memory: a 70B model vs one GPU

Take a 70-billion-parameter dense transformer trained with the standard mixed-precision Adam recipe. Per parameter you keep:

| What | dtype | bytes/param |
|---|---|---|
| weights used in forward/backward | bf16 | 2 |
| gradients | bf16 | 2 |
| master copy of weights | fp32 | 4 |
| Adam first moment (m) | fp32 | 4 |
| Adam second moment (v) | fp32 | 4 |
| **total** | | **16** |

(This is the accounting from the ZeRO paper. Some setups keep fp32 gradients and pay 18 bytes, others shard or quantize optimizer state and pay less. 16 is the usual starting point. See [the dtype chapter](../training/dtype.md) for the variations.)

```
weights alone (bf16):   70e9 x 2 bytes  =   140 GB
full training state:    70e9 x 16 bytes = 1,120 GB
one H100:                                    80 GB
minimum GPUs just to hold the state:  1120 / 80 = 14
```

Even the weights alone do not fit on one 80 GB card. The full training state needs at least 14 GPUs' worth of memory, *before* activations. Activation memory depends on batch size, sequence length and whether you recompute activations. For long sequences it can be as large as everything else combined. So the state of the model has to be split across devices. That splitting is model parallelism in its various forms (tensor, pipeline, ZeRO/FSDP sharding). It is not an optimization. Without it the model does not run at all.

### Time: one GPU vs a thousand

A standard estimate of training compute for a dense transformer is:

```
training FLOPs ~= 6 x N x D
    N = number of parameters
    D = number of training tokens
```

The 6 comes from about 2 FLOPs per parameter per token for the forward pass (one multiply and one add) and about 4 for the backward pass (gradients with respect to both activations and weights). The [training arithmetic chapter](../training/arithmetic.md) derives it properly.

Take N = 70e9 and D = 15e12 tokens (the token count reported for Llama 3):

```
FLOPs = 6 x 70e9 x 15e12 = 6.3e24
```

An H100 SXM has a dense bf16 peak of about 989 TFLOPS. Real training gets a fraction of peak, called MFU (Model FLOPs Utilization). Assume a respectable 40%:

```
achieved per GPU = 0.40 x 989e12 = 3.96e14 FLOP/s

1 GPU:        6.3e24 / 3.96e14 = 1.59e10 s  ~= 505 years
1,000 GPUs:   505 years / 1000             ~= 184 days
16,384 GPUs:  505 years / 16384            ~= 11 days
```

These assume perfect scaling, meaning a thousand GPUs run a thousand times faster. Real jobs lose some of that to communication and failures, which is what most of this chapter is about. Still, the order of magnitude makes the point: on one GPU this training run would outlast every person who might care about the result.

Cross-check (treat it as approximate). Meta's Llama 3.1 model card reports about 7.0M H100-hours for the 70B model. 6.3e24 / (7.0e6 x 3600 s) = 2.5e14 FLOP/s per GPU, which is about 25% of peak. That fits a large real job, which loses throughput to restarts, evals and imperfect overlap. The 6ND estimate gets you within a small factor of reality. That is what it is for.

So you distribute because you have to (memory) and because you want results this year (time). Everything else follows from the fact that you now have thousands of independent computers that must act as one.


## Core vocabulary

These terms show up in every launcher, log line and error message. Learn them once, precisely.

```
                    world (WORLD_SIZE = 8)
   +---------------- node 0 ----------------+  +---------------- node 1 ----------------+
   |  GPU0     GPU1     GPU2     GPU3        |  |  GPU0     GPU1     GPU2     GPU3        |
   |  rank 0   rank 1   rank 2   rank 3      |  |  rank 4   rank 5   rank 6   rank 7      |
   |  local 0  local 1  local 2  local 3     |  |  local 0  local 1  local 2  local 3     |
   +-----------------------------------------+  +-----------------------------------------+
            NODE_RANK = 0                                 NODE_RANK = 1
```

| Term | Meaning |
|---|---|
| **node** | One physical (or virtual) machine with its own OS, CPUs, RAM and usually 8 GPUs. Nodes talk to each other only over the network. |
| **process / worker** | One OS process running your training script. Normally there is one process per GPU. |
| **rank** | A process's global id, `0 .. WORLD_SIZE-1`. Unique across the whole job. |
| **local rank** | The process's id *within its node*, `0 .. nproc_per_node-1`. You use it to pick the GPU: `torch.cuda.set_device(local_rank)`. |
| **world size** | The total number of processes in the job. 2 nodes x 4 GPUs = 8. |
| **node rank** | The node's id within the job. With uniform nodes, `rank = node_rank x nproc_per_node + local_rank`. |
| **process group** | A subset of ranks that can do collectives together. The default group is "everyone". Model parallelism creates many smaller groups, for example "the 8 ranks holding shards of the same tensor" and "the ranks holding the same shard on different data replicas". |
| **backend** | The library that actually moves bytes for a process group: `nccl` for NVIDIA GPUs, `gloo` for CPU (and as a fallback), and vendor equivalents such as RCCL on AMD. |
| **SPMD** | Single Program, Multiple Data. Every rank runs *the same script*, and behaviour differs only through `rank` and the data it reads. This is the model PyTorch distributed, JAX and MPI all use. |
| **coordinator / master** | The well-known address (`MASTER_ADDR:MASTER_PORT`) that ranks contact to find each other at startup. Usually it lives on rank 0's node. It is a bootstrap point, not a boss: after startup the ranks are peers. |
| **launcher** | The thing that starts N copies of your script with the right environment variables: `torchrun`, `srun` (Slurm), `mpirun`, `deepspeed`, a Kubernetes operator. |

SPMD is worth dwelling on, because it is the source of a whole class of bugs. Since every rank runs the same code, any `if rank == 0:` that wraps a collective operation breaks the symmetry:

```python
if rank == 0:
    dist.all_reduce(t)   # rank 0 waits forever for ranks that never call it
```

Rank 0 enters the all-reduce and waits for partners that will never arrive. Nothing crashes. The job just stops. The rule is that **every rank must call the same collectives, in the same order, with compatible shapes.** Logging and checkpoint writes can be rank-conditional. Collectives cannot.


## The defining problem: partial failure

### The fallacies of distributed computing

In the 1990s, engineers at Sun Microsystems (L. Peter Deutsch and others) wrote down the assumptions that programmers new to distributed systems make without noticing, and that sooner or later turn out to be false:

1. The network is reliable.
2. Latency is zero.
3. Bandwidth is infinite.
4. The network is secure.
5. Topology doesn't change.
6. There is one administrator.
7. Transport cost is zero.
8. The network is homogeneous.

Each one has an ML version:

| Fallacy | How it bites in ML training |
|---|---|
| Network is reliable | A flaky InfiniBand link or a switch reboot stalls an all-reduce. Thousands of GPUs wait on one cable. |
| Latency is zero | Small collectives (for example per-layer all-reduces in tensor parallelism) are dominated by latency, not bandwidth. That is why TP is kept inside a node. |
| Bandwidth is infinite | Inter-node bandwidth is roughly an order of magnitude below intra-node NVLink. This sets which parallelism goes where. |
| Topology doesn't change | A node gets drained and replaced, so your rank-to-host mapping changes on restart. |
| One administrator | The cluster team, the storage team and the cloud provider each change things without telling you. |
| Network is homogeneous | One node has a different driver, NIC firmware or NCCL version, and it is the one that hangs. |

### Partial failure

On a single computer, failure is mostly all-or-nothing. If the machine dies, your program dies with it, and you know.

In a distributed system, **some parts fail while others keep running, and the parts that keep running often cannot tell what happened.** That is partial failure, and it is *the* defining problem of the field. Almost every mechanism in this chapter (timeouts, heartbeats, consensus, checkpoints) exists to cope with it.

Concretely, when rank 17 stops responding, the other ranks cannot tell which of these happened:

- the process crashed
- the node lost power
- the GPU fell off the bus and the process is stuck in a driver call
- the process is alive but stuck in a 40-second garbage collection or a slow filesystem read
- the network link between you and rank 17 is down, but rank 17 can still talk to others
- the message is just slow and will arrive in 100 ms

From the outside these all look the same: silence. We come back to this as the "slow vs dead" ambiguity when we get to heartbeats.

Synchronous training makes partial failure worse in a particular way. Every step needs *every* rank. So one failed component does not slow the job down by 1/N. It stops the job. Most web services are designed so that losing a replica loses capacity. A training job is designed so that losing a replica loses the job. That is a deliberate trade, made for simplicity and for mathematical equivalence with single-device SGD, but you pay for it in failure rate.

### Failure arithmetic

Suppose each node independently fails on average once every `M` days. That is its MTBF, mean time between failures. If failures are independent and memoryless (the usual exponential model), the failure rates of the nodes simply add:

```
job failure rate = (number of nodes) / M
job MTBF         = M / (number of nodes)
```

Worked example: say each node fails once every 1,000 days, about 2.7 years. That sounds very reliable. Now run a 1,000-node job:

```
job MTBF = 1000 days / 1000 nodes = 1 day = 24 hours
P(no failure in a given 24 h window) = exp(-24/24) = exp(-1) = 0.37
```

A node that you would call "rock solid" on its own gives you a job that **fails about once a day**, and has only a 37% chance of getting through any given day untouched.

Real data agrees. The Llama 3 paper reports 419 unexpected interruptions over a 54-day period on a 16,384-GPU H100 job, roughly one every 3.1 hours. Most were attributed to hardware, with GPUs and HBM the largest categories. Turned into per-GPU rates (these assume the job always had all 16,384 GPUs, which is approximately true):

```
per-GPU MTBF ~= 16384 x 54 days / 419 ~= 2,112 GPU-days ~= 5.8 years

at that per-GPU rate:
  8-GPU single node job:     MTBF ~= 2112 / 8    days ~= 0.7 years
  1,000-GPU job:             MTBF ~= 2112 / 1000 days ~= 51 hours
  8,000-GPU (1,000-node) job: MTBF ~= 2112 / 8000 days ~= 6.3 hours
```

Each GPU fails about once every 6 years. At scale you still lose a GPU every few hours. This is why the [fault tolerance chapter](../training/fault-tolerance/README.md) opens with "always plan to have more nodes than needed". At scale, failure is a steady background rate that you plan for, not a rare event.

### How often to checkpoint

If failures are routine, you need checkpoints, and you need to decide how often to take them. Checkpointing too often wastes time writing. Checkpointing too rarely wastes time redoing lost work. A classic first-order answer is Young's formula:

```
optimal checkpoint interval  T ~= sqrt(2 x C x MTBF)
    C = time to write one checkpoint (while training is paused)
```

Example: C = 5 minutes, job MTBF = 24 hours = 1,440 minutes:

```
T ~= sqrt(2 x 5 x 1440) = 120 minutes

overhead ~= C/T + T/(2 x MTBF)
         = 5/120 + 120/2880
         = 4.2% (writing) + 4.2% (expected lost work) ~= 8.3%
```

That is before restart time (scheduling, loading the checkpoint, warming up), which in practice is often the bigger term. Two levers come out of the formula. Make C smaller with asynchronous or sharded checkpoints, so `T` can shrink with it. Make MTBF larger by removing bad nodes before they join the job. See the [checkpoints chapter](../training/checkpoints/README.md).


## Communication models

### Shared memory vs message passing

There are two basic ways for parallel workers to cooperate:

- **Shared memory:** workers read and write the same memory. Threads in one process, or GPUs reading each other's memory over NVLink, work this way. It is fast and convenient, and it needs locks or other synchronization to avoid races. [Concurrency and Parallelism](./concurrency.md) covers this.
- **Message passing:** workers have private memory and cooperate only by sending each other messages. Separate processes on separate machines have no other option.

Distributed training is message passing. Even 8 GPUs inside one node are usually driven by 8 separate processes, and they exchange data through explicit operations (NCCL collectives over NVLink), not through shared Python objects. One practical consequence: **there is no global state.** A Python variable set on rank 0 does not exist on rank 3. If rank 3 needs it, someone has to send it.

### Synchronous vs asynchronous

- **Synchronous:** a sender waits until the communication completes, and everyone proceeds in lockstep.
- **Asynchronous:** a sender fires off a message and continues. Results arrive whenever they arrive.

There are two separate meanings here. Keep them apart:

1. **Algorithmic synchrony.** Do all workers apply the same gradient at the same step (synchronous SGD), or does each worker apply whatever possibly-stale gradient it has (asynchronous SGD, Hogwild-style)? Modern LLM training is almost entirely synchronous.
2. **API-level asynchrony.** Does the *function call* block? In PyTorch, `dist.all_reduce(t, async_op=True)` returns a handle immediately, and you call `.wait()` later. With NCCL, even "blocking" collectives are only enqueued on a CUDA stream, so the CPU runs ahead of the GPU. This is how frameworks overlap communication with computation. DDP starts all-reducing the gradients of the last layers while backward is still computing the gradients of earlier ones.

So a training job is algorithmically synchronous (everyone agrees on every step) and implemented asynchronously underneath (communication overlapped with compute).

### Bulk-synchronous parallel

Leslie Valiant's Bulk-Synchronous Parallel (BSP) model (1990) describes a computation as a sequence of *supersteps*. Each superstep is:

1. **local compute** on each worker, independently
2. **communication** between workers
3. a **barrier**: nobody starts the next superstep until everyone has finished this one

Synchronous data-parallel SGD maps onto it exactly:

```
          superstep k                              superstep k+1
  rank0  [fwd+bwd on shard 0]--+            +--[fwd+bwd]--+
  rank1  [fwd+bwd on shard 1]--+-all-reduce-+--[fwd+bwd]--+-all-reduce- ...
  rank2  [fwd+bwd on shard 2]--+  (barrier) +--[fwd+bwd]--+
  rank3  [fwd+bwd on shard 3]--+            +--[fwd+bwd]--+
                               ^
                      optimizer.step() can only happen
                      after everyone's gradient is summed
```

The all-reduce acts as the barrier. No rank can finish it until every rank has contributed.

Why BSP? It is easy to reason about. After every superstep, all replicas are in an identical, well-defined state, the same state a single huge GPU would have reached on the full batch. Convergence theory for SGD applies directly. Bugs reproduce. The cost is that each superstep is only as fast as the slowest worker.

### Stragglers and tail latency

Under BSP, step time is the **maximum** over ranks, not the average:

```
step_time = max(compute_time[r] for r in ranks) + communication
```

The maximum of many random variables lives in the tail of their distribution. Suppose each rank, on any given step, has a 0.1% chance of a hiccup (a GC pause, a page fault, a noisy neighbour, a thermally throttled GPU, a slow dataloader read):

```
P(at least one rank hiccups on a step):
   8 ranks:      1 - 0.999^8    =  0.8%
   1,000 ranks:  1 - 0.999^1000 = 63%
```

A 1-in-1000 event on one rank becomes the normal case for the job. This is "tail at scale" (Dean and Barroso's 2013 paper of that name is the classic reference). At scale your step time is set by your p99.9 worker, not your median one.

The worst case is a *persistent* straggler: one GPU that always runs 10% slow because of bad cooling, a degraded PCIe link or a misconfigured clock. It makes the whole job 10% slower, and the job's average throughput metrics will not say which GPU is to blame. You find it by measuring per-rank timings, or by running per-node benchmarks and removing the slow node. The [debug](../debug/README.md) and [fault tolerance](../training/fault-tolerance/README.md) chapters have tools for this.

### Barriers

A barrier is the simplest synchronization: "nobody passes until everybody arrives".

```python
dist.barrier()
```

Common legitimate uses:

- let rank 0 download or preprocess a dataset while the others wait, then everyone reads the result (see [`main_process_first.py`](../training/tools/main_process_first.py))
- make sure all ranks have finished writing their checkpoint shards before one rank writes a "checkpoint complete" marker
- line up ranks before a timing measurement

Two warnings. First, a barrier is a collective, so the SPMD rule applies: if one rank skips it, everyone else waits forever. Second, barriers hide bugs. Sprinkling `barrier()` calls until a hang "goes away" usually just moves the hang somewhere else. If you need a barrier to make something correct, write down why.


## Time and ordering

### Clocks drift

Every computer keeps time with a quartz crystal oscillator, and no two crystals run at exactly the same rate. Typical drift is on the order of tens of parts per million (it depends on hardware and temperature, so treat this as a ballpark):

```
20 ppm x 86,400 s/day = 1.7 seconds of drift per day
```

Left alone, two nodes' clocks would disagree by seconds within a day.

### NTP and its limits

Machines correct drift with NTP (Network Time Protocol, usually through `chrony` or `systemd-timesyncd` on Linux). NTP asks time servers for the time, estimates the network delay, and slowly adjusts the local clock. On a well-run data center network, clocks typically agree to within a fraction of a millisecond to a few milliseconds. Across the internet, or with misconfiguration, they can disagree by tens of milliseconds or much more. PTP (Precision Time Protocol), with hardware timestamping, can get to microseconds, but you should not assume your cluster runs it.

NTP can also step a clock, or slew it faster or slower than real time while it catches up. So even on one machine, wall-clock time (`time.time()`) can jump. For measuring durations, use a monotonic clock (`time.monotonic()` or `time.perf_counter()`), which never goes backwards. It is only meaningful on the machine that read it.

### Why you cannot trust timestamps across nodes

A GPU step takes a few hundred milliseconds. A collective can take microseconds to milliseconds. Cross-node clock error can be milliseconds. So when you merge logs from 64 nodes and sort by timestamp, the order of events that happened within a few milliseconds of each other on different nodes is essentially random. You can easily see "rank 12 received the tensor" logged before "rank 3 sent it".

Practical rules for debugging:

1. **Order by logical progress, not wall-clock time.** Log the training step, the micro-batch index and a per-rank collective sequence number. "Rank 12 is at step 4,001, every other rank is at step 4,002" tells you far more than any timestamp. PyTorch's NCCL flight recorder records exactly this kind of per-collective sequence number. See [the flight recorder section](../debug/pytorch.md#the-flight-recorder-trace).
2. **Compare timestamps only within one node,** or across nodes only when the gaps are much larger than plausible clock skew (seconds, not milliseconds).
3. **Use durations, not absolute times,** to find stragglers. "Rank 40's forward took 900 ms, everyone else took 600 ms" is robust to clock skew.

The theoretical answer to "what happened before what" without synchronized clocks is Lamport's logical clocks (1978): every process keeps a counter, increments it on each event and attaches it to every message, and on receiving a message sets its counter to `max(own, received) + 1`. You will not implement this for training, but step counters and collective sequence numbers are the same idea.


## Coordination

Before a distributed job can compute anything, its processes have to find each other and agree on who is who. While it runs, they have to notice when one of them has gone. Both are coordination problems.

### Rendezvous

When you launch 64 processes on 8 nodes, each process starts out knowing almost nothing: its own hostname, the environment variables the launcher gave it, and one well-known address. **Rendezvous** is the protocol by which they:

1. find each other (every process registers its address),
2. agree on the membership (exactly these N processes are in the job),
3. agree on ranks (who is rank 0, ..., N-1),
4. exchange the information the communication backend needs to set up direct connections. For NCCL this is a unique id that rank 0 generates and everyone else must receive.

```
  rank 3 --\
  rank 1 ---+--> [ key-value store at MASTER_ADDR:MASTER_PORT ] <--+-- rank 2
  rank 0 --/       "rank0 addr = 10.0.0.1:41233"                  \-- rank 7
                   "nccl_unique_id = 0x4f..."
                   "arrived = 8 / 8"
         after this, ranks connect to each other directly; the store is
         used only for occasional coordination
```

The [networking chapter](./networking.md#how-pytorch-distributed-bootstraps) walks through this bootstrap at the TCP level.

### Key-value stores: TCPStore and etcd

The meeting place is a small **key-value store**: a service that supports `set(key, value)`, `get(key)` (blocking until the key exists), `add(key, n)` (an atomic counter) and similar operations. With those primitives you can build rendezvous, barriers and "everyone has finished" detection.

- **TCPStore** is PyTorch's built-in store. One process (rank 0 with `init_process_group(init_method="env://")`, or the torchrun agent on the rendezvous host) runs a small TCP server, and everyone else connects as a client. It is simple and has no dependencies. It is also a single point of failure: if the process hosting it dies, the store goes with it.
- **etcd** is a separate, replicated key-value store that runs on several machines and uses the Raft consensus algorithm (below) to stay consistent when some of them fail. Kubernetes keeps *all* of its cluster state in etcd. torchrun can use etcd as its rendezvous backend (`--rdzv-backend=etcd` / `etcd-v2`). Its default `c10d` backend uses a TCPStore instead, because most users do not want to run an etcd cluster.

### Heartbeats, timeouts and "slow vs dead"

To notice that a peer has died, processes send **heartbeats**, periodic "I'm alive" messages, and declare a peer dead if they hear nothing for some **timeout**. torchrun's c10d rendezvous, for example, by default has each agent send a keep-alive every 5 seconds and treats a node as gone after 3 missed attempts (defaults read from the PyTorch 2.15 source; they may differ in other versions).

Here is the fundamental problem. **In an asynchronous network you cannot distinguish a dead process from a slow one.** Both are silent. A timeout does not *detect* death. It *decides* that you will treat silence longer than T as death. Every value of T is wrong in some case:

| Timeout | What goes wrong |
|---|---|
| too short | A slow-but-alive rank (long checkpoint write, GC pause, first-step compilation, congested filesystem) is declared dead. The job is torn down and restarted for nothing, maybe again and again. |
| too long | A rank that really is dead holds the job hostage. Thousands of GPUs sit idle until the timeout fires. |

This is why PyTorch's distributed timeouts look the way they do. The default process-group timeout is long: 30 minutes for gloo (verified in PyTorch 2.15), and 10 minutes for NCCL in recent releases (from memory; check `torch.distributed.constants` for your version). It has to be long enough for legitimate slow operations like a rank-0-only checkpoint save, and short enough that a hang does not burn a whole night. If you have one known-slow operation, the right fix is usually to raise the timeout for it specifically (pass `timeout=` to `init_process_group` or `new_group`), not to shorten it globally.

When an NCCL collective times out, PyTorch's watchdog thread aborts the communicator and kills the process. That turns a silent hang into a crash, which a launcher can then restart. That is the point: **a crash is much better than a hang,** because a crash can be detected and acted on automatically.

### Leader election and consensus

Many coordination tasks reduce to "a group of machines must agree on one value even though some may fail". Examples: who is the leader, what is the current membership, has this key been written. This is the **consensus** problem. It is harder than it sounds. The FLP result (Fischer, Lynch and Paterson, 1985) proves that no deterministic algorithm can guarantee consensus in a fully asynchronous system if even one process may crash. Practical algorithms get around this by using timeouts, which guarantee progress only when the network behaves reasonably, while never violating safety (never deciding two different values).

**Raft** (Ongaro and Ousterhout, 2014) is the consensus algorithm most people learn now, because it was designed to be understandable. In outline:

1. **Roles.** Every server is a *follower*, a *candidate* or the *leader*. There is at most one leader per *term*, a numbered epoch.
2. **Leader election.** The leader sends heartbeats. A follower that hears nothing for a randomized timeout becomes a candidate, increments the term and asks the others for votes. A candidate that gets votes from a **majority** becomes leader. Since two majorities always overlap, two leaders cannot be elected in the same term.
3. **Log replication.** Clients send writes to the leader. The leader appends each write to its log and sends it to followers. Once a majority has stored the entry, it is **committed**, and it will survive any minority of failures.
4. **Fault tolerance.** With `2f + 1` servers, Raft tolerates `f` failures. That is why etcd clusters have 3 or 5 members: 3 tolerate 1 failure, 5 tolerate 2.

```
       term 7                         leader fails          term 8
  [L] --heartbeat--> [F] [F] [F] [F]      X        [C] --vote?--> [F] [F] [F]
   |                                                 3 of 5 vote yes -> new leader
   +-- write x=1 -> replicated to 3 of 5 -> committed
```

Where consensus shows up in ML stacks:

- **Kubernetes:** etcd (Raft) holds all cluster state. Every pod, job and node object lives there. See [orchestration](../orchestration/README.md) and [orchestration internals](../orchestration/internals.md).
- **torchrun with the etcd backend:** rendezvous membership is agreed through etcd, so the rendezvous survives the loss of any single host.
- **torchrun with the c10d backend:** there is no consensus. A single TCPStore acts as a de facto leader. That is simpler, but if the host running the store dies, the rendezvous dies too.
- **Slurm:** the controller (`slurmctld`) can have a backup controller that takes over. That is primary/backup failover, not majority consensus.

Your training loop itself does *not* run consensus. All ranks deterministically compute the same thing, so there is nothing to vote on. That is the next section's point.


## Consistency, replication and CAP

### Replication and consistency

**Replication** means keeping copies of the same data on several machines, for fault tolerance or to spread out read load. Once you have copies, you need a **consistency model**, a statement of what a reader may observe:

- **Strong consistency (linearizability):** the system behaves as if there were a single copy. Once a write completes, every later read sees it.
- **Eventual consistency:** if writes stop, replicas eventually converge. Until then, different readers may see different values.

Strong consistency costs coordination (consensus or similar) on every write. Eventual consistency is cheaper and stays available, but the application has to tolerate stale reads.

### CAP, briefly

The CAP theorem (conjectured by Brewer in 2000, proved by Gilbert and Lynch in 2002) says that when a **network partition** happens, some replicas cannot talk to others, and a replicated store must choose between:

- **Consistency:** refuse requests that might return stale or conflicting data, or
- **Availability:** answer every request, accepting that answers may be stale or conflicting.

Partitions are not optional in a real network, so the real choice is "C or A during a partition". etcd chooses C: a minority partition refuses writes. Many caches and some databases choose A.

### Why training mostly sidesteps this

Data-parallel replicas of a model are replicas in the distributed-systems sense: N copies of the same weights. But they do not need a consistency protocol, because they are **deterministic state machines that receive identical inputs**:

```
all replicas start with identical weights      (DDP broadcasts rank 0's at init)
every step, all receive the identical gradient  (all-reduce output is the same on every rank)
every replica applies the identical update      (same optimizer, same hyperparameters)
=> all replicas hold identical weights after every step, with no voting
```

This is state-machine replication with the "agree on the inputs" part done by the all-reduce. And when a partition or failure happens, training simply chooses **consistency over availability**: the collective blocks or times out, and the job stops. It never continues with replicas that disagree. Stopping is acceptable because a training job has no users waiting on it from moment to moment. You restart from a checkpoint and lose minutes or hours.

(One caveat: the replicas are identical only if the all-reduce gives bit-identical results everywhere and every rank runs the same code. NCCL's all-reduce gives the same result on every rank. A rank-specific bug, such as a different dropout seed path that also changes weights, or one rank skipping a step, makes replicas diverge silently. Some teams periodically checksum the weights across ranks to catch this.)

### Where you don't get to sidestep it

Once you leave the training loop, the usual distributed-systems trade-offs come back:

- **Model serving:** many replicas behind a load balancer. During a rollout, some replicas serve model v1 and some v2. That is eventual consistency, and users may see both. Choosing between "keep serving" and "refuse until consistent" is a real product decision.
- **Feature stores and embedding tables:** online features are written by streaming pipelines and read at inference time. Staleness, read-your-writes and training/serving skew are consistency problems.
- **Asynchronous or parameter-server training** (recommendation models with huge sparse embeddings still use it): workers read possibly-stale parameters on purpose, trading consistency for throughput.
- **Checkpoint storage:** an object store may be eventually consistent for listings, or a checkpoint may be half-written when a node dies. Write to a temporary name, then rename or write a "complete" marker last.


## Collectives: the workhorse of distributed training

Point-to-point messages ("rank 3 sends this tensor to rank 7") are the primitive, but training code mostly uses **collective operations**: a pattern of communication that every rank in a process group takes part in together. The [networking chapter's introduction](./networking.md#collective-communication-an-intuitive-introduction) and the [communication patterns chapter](../network/comms.md) have diagrams for each. Here is the short list and what each is used for:

| Collective | What happens | Where you meet it |
|---|---|---|
| broadcast | one rank's tensor is copied to all | initial weights in DDP |
| all-reduce | every rank contributes a tensor, every rank gets the elementwise sum (or mean, max...) | gradient sync in DDP, TP layer outputs |
| reduce-scatter | sum, but each rank keeps only its 1/N slice of the result | gradient sync in ZeRO/FSDP |
| all-gather | every rank contributes a slice, every rank gets the concatenation | gathering sharded weights in ZeRO/FSDP |
| all-to-all | every rank sends a distinct chunk to every other rank | Mixture-of-Experts token routing, sequence parallelism |
| barrier | nobody proceeds until all arrive | coordination |

All-reduce - every rank ends with the sum of all ranks' inputs:

![all-reduce](../network/images/collective-all-reduce-1.png)

([source](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html))

A useful identity: **all-reduce = reduce-scatter + all-gather.** ZeRO and FSDP exploit this by splitting the two halves apart and doing useful work in between.

### Parameter servers vs all-reduce

There have historically been two architectures for synchronizing gradients.

**Parameter server (PS).** Some machines are *servers* that hold the parameters. The others are *workers* that compute gradients, push them to the servers, and pull back updated parameters. Google's DistBelief (2012) and Mu Li et al.'s Parameter Server (OSDI 2014) are the landmark systems.

```
  parameter server                      ring all-reduce
     [ PS ]                         r0 --> r1 --> r2 --> r3
    / |  | \                         ^                    |
  w0 w1 w2 w3                        +--------------------+
  every worker pushes to            each rank sends and receives
  and pulls from the server         to/from its neighbours only
```

**All-reduce.** There is no server. Workers run a collective algorithm, classically a *ring*, in which each rank exchanges chunks only with its neighbours. Baidu popularized ring all-reduce for deep learning in 2017, and Uber's Horovod (2018) made it mainstream.

Arithmetic, for a 7B model with bf16 gradients (S = 14 GB) and 64 workers, each with 50 GB/s (400 Gb/s) of network bandwidth:

```
Single parameter server:
   the server must receive 64 x 14 GB = 896 GB per step
   at 50 GB/s:  ~18 s per step, and it gets worse linearly with more workers

Ring all-reduce:
   each rank sends (and receives) 2 x (N-1)/N x S = 2 x 63/64 x 14 GB ~= 27.6 GB
   at 50 GB/s:  ~0.55 s per step, almost independent of N
```

The ring's per-rank traffic stays at about 2S no matter how many workers you add. That is why it is called bandwidth-optimal. The PS bottleneck can be reduced by sharding parameters across many servers, but then you are running and provisioning a second fleet of machines. [Collective communication internals](../network/collectives-internals.md) has the full cost models, including latency terms and tree algorithms.

Why all-reduce won for LLMs:

1. **Dense gradients.** Every LLM parameter gets a gradient every step, which is exactly the case all-reduce handles best. PS designs do well with *sparse* updates, which is why they survive in recommender systems with billion-row embedding tables.
2. **Synchronous SGD became the norm.** The PS architecture's selling point was asynchronous updates. Async SGD's stale gradients hurt convergence and reproducibility at large scale, and once everyone trained synchronously, a central server was just a bottleneck.
3. **Hardware.** NVLink, NVSwitch and InfiniBand with RDMA, plus NCCL's topology-aware ring and tree algorithms, made peer-to-peer collectives between GPUs extremely fast, with no CPU server in the data path.
4. **Simplicity.** Every rank is identical (SPMD). There is no separate server role to deploy, scale and keep alive.


## Data parallel vs model parallel

### Data parallelism

Every rank holds a **full copy** of the model. Each rank processes a **different slice of the batch**. Gradients are averaged with an all-reduce, so every replica applies the same update.

```
                global batch of 32
   +--------+--------+--------+--------+
   | 8 ex.  | 8 ex.  | 8 ex.  | 8 ex.  |
   +---+----+---+----+---+----+---+----+
       |        |        |        |
   [model]  [model]  [model]  [model]     <- 4 full replicas
   rank 0   rank 1   rank 2   rank 3
       |        |        |        |
      g0       g1       g2       g3       <- local gradients
       \________ all-reduce (mean) _______/
                       |
           g = (g0+g1+g2+g3)/4 on every rank
                       |
       identical optimizer.step() on every rank
```

Because the loss is a mean over examples, the averaged gradient equals the full-batch gradient, so 4-way DP with 8 examples per rank is mathematically the same as one GPU processing 32 examples. (The runnable example below checks this numerically.)

- **Scales:** compute (more throughput).
- **Does not help:** memory. Every rank holds everything.
- **Costs:** one gradient all-reduce per step, about 2x the model size in traffic per rank.

### Model parallelism

When the model does not fit on one device, you split *the model* across ranks.

Naive model parallelism leaves most GPUs idle; pipelining with micro-batches shrinks the idle "bubble":

![naive model parallelism vs pipeline parallelism](../training/model-parallelism/images/parallelism-gpipe-bubble.png)

([source](https://arxiv.org/abs/1811.06965))

```
Pipeline parallel (split by layers)          Tensor parallel (split within a layer)

 rank0: layers  1-20  --act-->               X @ W  where W = [W0 | W1 | W2 | W3]
 rank1: layers 21-40  --act-->                 rank0: X @ W0  -\
 rank2: layers 41-60  --act-->                 rank1: X @ W1  --+-- all-gather / all-reduce
 rank3: layers 61-80                           rank2: X @ W2  --+   to combine
       <--grad-- in backward                   rank3: X @ W3  -/
```

- **Pipeline parallelism (PP)** puts different layers on different ranks and passes activations forward and gradients backward point-to-point. Communication is light, but ranks sit idle while the pipeline fills and drains (the "bubble") unless you split the batch into many micro-batches.
- **Tensor parallelism (TP)** splits individual weight matrices, so every layer needs a collective. Communication is heavy and latency-sensitive, so TP is usually kept within a node, on NVLink.
- **ZeRO / FSDP (sharded data parallelism)** is a hybrid. It is data parallel in the computation, but parameters, gradients and optimizer state are *sharded* across ranks and all-gathered just in time for each layer's computation. Memory per rank drops roughly by a factor of N, at the cost of more communication than plain DP.

Real large-scale training combines these. A typical arrangement is TP inside a node, PP across a few nodes, and DP or FSDP across everything else, often called 3D parallelism.

### The three resources

Every parallelism strategy trades among three resources:

```
                    compute
                   (FLOPs/s, utilization)
                      /\
                     /  \
                    /    \
                   /      \
       memory  ---+--------+---  communication
   (bytes/device)          (bytes moved, latency, bandwidth)
```

| Strategy | Memory per device | Communication | Compute efficiency risk |
|---|---|---|---|
| DP | full model (high) | gradient all-reduce once per step | low, overlaps well |
| ZeRO-3 / FSDP | about 1/N of state | about 1.5x DP: all-gather params in fwd and bwd, reduce-scatter grads | moderate, needs prefetch overlap |
| TP | about 1/N of each layer | collectives in every layer, latency-bound | high if run over slow links |
| PP | about 1/N of the layers | small activations, point-to-point | pipeline bubble |
| activation recomputation | lower activations | none | about 30% more compute |

You can almost always trade one resource for another. You cannot get all three for free. The job of a distributed training configuration is to pick the cheapest trade for your hardware: how much memory per GPU, how fast the links inside and between nodes, how many GPUs. The [model parallelism chapter](../training/model-parallelism/README.md) covers each strategy in detail, and [Parallelism From First Principles](../training/model-parallelism/derivations.md) derives the memory and communication cost of each so you can do this arithmetic yourself.


## Fault tolerance patterns

Given that failures are routine at scale, these are the standard ways to survive them. Most of them predate ML. They come from decades of HPC and web infrastructure.

### Checkpoint / restart

The workhorse. Periodically save everything needed to resume (model weights, optimizer state, LR scheduler state, data loader position, RNG states, step counter). On failure, kill everything and restart from the last checkpoint.

Why "kill everything" rather than repairing just the failed rank? Because the surviving ranks are often stuck partway through a collective, with communicators in an undefined state. Restarting from a known-consistent snapshot is simpler and far more reliable than trying to repair live state. Some newer systems do attempt in-place recovery, but checkpoint/restart is still the baseline everyone has.

Rules that keep checkpoints trustworthy:

- **Atomic completion:** write to a temporary location, then rename, or write a "complete" marker last. A job killed mid-save must never leave a checkpoint that *looks* complete.
- **Keep more than one:** the newest checkpoint may be corrupt, or may capture a state just before a loss spike you want to roll back past.
- **Verify resume once:** actually kill a job and resume it, and check that the loss curve continues smoothly. A checkpoint that has never been restored is a hope, not a backup.

See the [checkpoints chapter](../training/checkpoints/README.md).

### Idempotency

An operation is **idempotent** if doing it twice has the same effect as doing it once. In a distributed system with retries, you cannot always know whether an operation that "failed" actually happened. The network may have lost the reply, not the request. If the operation is idempotent, you can just retry it.

In ML jobs:

- "Save checkpoint for step 4000 to `ckpt/step_4000/`" is idempotent. Doing it twice overwrites with the same data.
- "Append this checkpoint to the list" or "increment the epoch counter" is not. A retry duplicates it.
- Data loading should be a deterministic function of (seed, step) so that resuming at step 4000 replays exactly the batches that would have come next. Otherwise a restart silently skips or repeats data.

### Retries with exponential backoff and jitter

Transient failures (an object-store request failing with a 503, a DNS lookup timing out, the rendezvous store not up yet) should be retried. Retry naively and you can make things worse: 4,000 ranks retrying a struggling storage server every 100 ms is a self-inflicted denial-of-service attack.

The standard pattern:

```python
import random, time

def retry(fn, attempts=6, base=0.5, cap=30.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            # exponential backoff with "full jitter": spreads retries out in time
            time.sleep(random.uniform(0, min(cap, base * 2**i)))
```

The exponential part backs off from a struggling service. The jitter (randomness) stops all ranks from retrying in synchronized waves. Only retry operations that are idempotent.

### Elastic training

**Elastic** training lets the job keep going when the set of workers changes. With torchrun:

```bash
torchrun --nnodes=4:8 --nproc_per_node=8 --max-restarts=3 \
         --rdzv-backend=c10d --rdzv-endpoint=$HOST:29400 train.py
```

- `--nnodes=4:8` means run with anywhere from 4 to 8 nodes.
- When a worker fails, the torchrun agents stop *all* workers, run a new rendezvous with whoever is still alive (plus any new nodes), assign new ranks and restart the script, up to `--max-restarts` times.

This is not magic. Your script still restarts from scratch and must load its last checkpoint. A different world size also changes things: the global batch size (unless you adjust gradient accumulation), how data is sharded, and how sharded checkpoints map onto ranks. Elasticity saves you the round trip through the cluster scheduler. It does not save you from writing a resumable training loop.

### Hot spares

Keep a few extra healthy nodes allocated and idle. When a node fails, swap a spare in, restart from checkpoint, and send the bad node off for repair. This turns "wait hours for the scheduler and a replacement" into "restart in minutes". From the failure arithmetic above, a job that loses a node every few hours needs a pool of spares, not just one. The [fault tolerance chapter](../training/fault-tolerance/README.md) suggests 5-10% headroom and describes how to automate the swap with Slurm.

### Determinism and reproducibility across ranks

Determinism matters more in distributed training than on one GPU, for two reasons:

1. **Correctness of replicas.** Data-parallel ranks must make identical updates. Anything that should be identical across ranks (initialization, the dropout pattern *if* it affects shared state, the order in which parameters are bucketed for all-reduce) must be computed identically, or synchronized. DDP broadcasts rank 0's parameters at construction for exactly this reason.
2. **Debuggability of restarts.** If a loss spike shows up at step 51,234, you want to resume from the step-51,000 checkpoint and see the *same* spike. That needs the data order, RNG states and, ideally, the numerics to be reproducible.

What usually breaks it:

- **Floating-point non-associativity.** `(a + b) + c != a + (b + c)` in floating point. An all-reduce over 8 ranks and one over 16 ranks add in different orders, so results differ in the last bits. Change the world size and you change the numerics. NCCL's algorithm choice (ring vs tree) can also change the reduction order.
- **Non-deterministic kernels.** Some GPU kernels (for example atomic-add-based scatter operations) are non-deterministic run to run.
- **Data order** that depends on worker timing, on filesystem listing order, or on the world size.
- **Per-rank seeding mistakes.** You usually want the *same* seed for anything that must match across ranks (model init) and *different* seeds where ranks should differ (data augmentation, dropout on different data). Getting this backwards is a classic bug.

Bitwise reproducibility at scale is expensive and often not achievable. A realistic target is: reproducible data order, reproducible RNG streams, and loss curves that match closely when resumed on the same world size. See [reproducibility](../training/reproducibility/README.md).


## The distributed debugging mindset

Debugging one process means reading one stack trace. Debugging 1,024 processes means finding *which* of them is wrong, and the evidence is spread across 128 machines.

### Per-rank logs and rank-prefixed printing

When N processes print to one terminal or one file, their lines interleave. Lines from different ranks get mixed up, and sometimes two ranks' output ends up on the *same* line. (This happened in testing this chapter's example. See the expected output below.) Without a rank prefix you cannot tell which process said what.

Minimum hygiene:

- **Prefix every line with the rank**, for example `[rank17]`. Launchers can do it for you: `torchrun --tee 3 --log-dir logs/` writes per-rank log files on Linux, and `srun --label` prefixes each line with the task id.
- **Print one whole line per call** and flush (`print(..., flush=True)`). If lines still collide, use a file lock around prints. [`printflock.py`](../training/tools/printflock.py) and [`multi-gpu-non-interleaved-print.py`](../training/tools/multi-gpu-non-interleaved-print.py) in `training/tools` show how.
- **Log once for global facts** (`if rank == 0`), but **log on every rank** when hunting a rank-specific problem. The bad rank is never rank 0 when you need it to be.
- **Log progress counters** (step, micro-batch, collective sequence number), so that you can order events without trusting cross-node clocks.

### Hangs vs crashes

| | Crash | Hang |
|---|---|---|
| What you see | a traceback on some rank, nonzero exit code | nothing. GPU utilization flat (or stuck at 100% spinning in a collective), logs stop |
| Where to look | the *first* rank that failed. Other ranks often die afterwards with secondary "connection reset" or "NCCL error" messages, which are symptoms | the rank that is *not* where the others are |
| Typical causes | OOM, NaN assertion, code bug, hardware Xid error | mismatched collectives (SPMD violation), a dead peer before a timeout fires, deadlock in data loading, network misconfiguration |

The hard case is the hang. 1,023 ranks are waiting in an all-reduce, and one rank is somewhere else: stuck reading a file, in a different collective, or in an `if rank == ...` branch that skipped a collective. Finding that one rank is the whole job.

When reading crash logs from many ranks, **find the earliest real error.** When one rank dies, every rank talking to it reports a communication error a moment later. Hundreds of tracebacks that say "connection closed by peer" are symptoms. The single one that says "CUDA out of memory" is the cause.

### py-spy on all ranks

For hangs, the most useful tool is a stack dump of every rank. [`py-spy`](https://github.com/benfred/py-spy) attaches to a running Python process and prints its current stack without stopping it:

```bash
# on each node, dump all python processes' stacks
pgrep -f train.py | xargs -I {} py-spy dump --pid {}
```

Run it on all nodes (through `srun`, `pdsh` or similar), then look for the odd one out: 63 stacks in `all_reduce` and one in `torch.load` tells you where to look. The [PyTorch debugging chapter](../debug/pytorch.md#py-spy) shows how to run py-spy [across processes](../debug/pytorch.md#multi-process-py-spy) and across nodes, and covers [NCCL hang diagnosis](../debug/pytorch.md#diagnosing-nccl-collective-hangs) with the flight recorder.

Other debugging habits that pay off:

- **Reproduce small.** A bug that needs 512 GPUs is often reproducible with 2 processes on one machine, or with gloo on CPU. Try that first. See [emulating multi-node setups](../training/emulate-multi-node.md).
- **Test the plumbing separately from the model.** [`torch-distributed-gpu-test.py`](../debug/torch-distributed-gpu-test.py) checks that every rank can do collectives with every other rank, before you blame your code.
- **Bisect the cluster.** If a job fails on 64 nodes but not on 32, split the nodes in half and test each half. Bad hardware tends to show up as a specific node, not as a scale threshold.
- **Turn on verbose diagnostics when needed.** `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` are noisy but show which interface, transport and collective each rank is using. See [network debug](../network/debug/README.md).


## A runnable minimal example

This example runs on a laptop CPU with no GPU, using the `gloo` backend and `torchrun` to start 2-4 processes on one machine. It does three things:

1. a raw `all_reduce`: every rank contributes `rank + 1`, and every rank should end up with the sum
2. one DDP training step on a toy linear model, with each rank taking a different shard of the batch
3. a check that all replicas end up with identical weights, and that those weights match a single-process step on the full batch

Save as `dist_hello.py`:

```python
# dist_hello.py -- run with: torchrun --nnodes=1 --nproc_per_node=4 \
#                              --master_addr=127.0.0.1 --master_port=29500 dist_hello.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def log(msg):
    # rank-prefixed, one print call per line: without the prefix, output from
    # N processes sharing one terminal is unattributable
    print(f"[rank{dist.get_rank()}] {msg}", flush=True)

def main():
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    dist.init_process_group(backend="gloo")  # gloo = CPU backend; use "nccl" on GPUs
    rank, world = dist.get_rank(), dist.get_world_size()
    log(f"local_rank={os.environ['LOCAL_RANK']} world_size={world}")

    # --- 1. a raw collective: every rank contributes rank+1, everyone gets the sum
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    log(f"all_reduce sum = {t.item()}  (expected {world * (world + 1) / 2})")

    # --- 2. a toy DDP step
    torch.manual_seed(0)                      # same init on every rank...
    model = nn.Linear(4, 1, bias=False)
    ddp = DDP(model)                          # ...and DDP broadcasts rank 0's weights anyway
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1)

    # the "global batch" is 8 samples; each rank takes its own shard of 8/world
    g = torch.Generator().manual_seed(42)
    X = torch.randn(8, 4, generator=g)       # (B, 4) identical on all ranks
    Y = torch.randn(8, 1, generator=g)       # (B, 1)
    shard = slice(rank * 8 // world, (rank + 1) * 8 // world)
    x, y = X[shard], Y[shard]                 # (B/world, 4), (B/world, 1)

    loss = ((ddp(x) - y) ** 2).mean()         # local loss differs per rank
    loss.backward()                           # DDP all-reduces (averages) grads here
    log(f"local loss = {loss.item():.4f}  grad[:2] = {[round(v, 4) for v in model.weight.grad.flatten().tolist()[:2]]}")
    opt.step()

    # --- 3. check: replicas identical, and equal to single-process full-batch SGD
    w = model.weight.detach().clone()
    gathered = [torch.zeros_like(w) for _ in range(world)]
    dist.all_gather(gathered, w)
    same = all(torch.equal(gathered[0], gi) for gi in gathered)

    torch.manual_seed(0)
    ref = nn.Linear(4, 1, bias=False)
    ((ref(X) - Y) ** 2).mean().backward()
    with torch.no_grad():
        ref.weight -= 0.1 * ref.weight.grad
    match = torch.allclose(w, ref.weight, atol=1e-6)
    if rank == 0:
        log(f"replicas identical: {same}; matches 1-process full batch: {match}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run it:

```bash
pip install torch   # CPU build is fine
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 dist_hello.py
```

Expected output (tested with PyTorch 2.15, CPU, macOS; torchrun's own warning lines omitted). Line order differs from run to run, since the ranks are independent processes:

```
[rank0] local_rank=0 world_size=4
[rank1] local_rank=1 world_size=4
[rank3] local_rank=3 world_size=4
[rank2] local_rank=2 world_size=4
[rank0] all_reduce sum = 10.0  (expected 10.0)
[rank3] all_reduce sum = 10.0  (expected 10.0)
[rank2] all_reduce sum = 10.0  (expected 10.0)
[rank1] all_reduce sum = 10.0  (expected 10.0)
[rank0] local loss = 0.2858  grad[:2] = [-1.0857, 0.3197]
[rank1] local loss = 1.9882  grad[:2] = [-1.0857, 0.3197][rank3] local loss = 0.4984  grad[:2] = [-1.0857, 0.3197]
[rank2] local loss = 1.8236  grad[:2] = [-1.0857, 0.3197]
[rank0] replicas identical: True; matches 1-process full batch: True
```

With `--nproc_per_node=2` the sum is `3.0` and the local losses are `1.1370` and `1.1610`. The gradient is the same `[-1.0857, 0.3197]`, and both checks print `True`.

What to notice:

- **The local losses differ, the gradients do not.** Each rank computed its loss on its own 2 examples. After `backward()`, DDP's all-reduce replaced each rank's local gradient with the average, so every rank holds the identical, full-batch gradient. That is data parallelism in one line of output.
- **The DP step equals the single-process step.** `matches 1-process full batch: True` is the BSP guarantee from earlier: 4 ranks with 2 examples each give exactly the update of one process with 8 examples. (Exactly here, because the shards are equal in size and the loss is a mean.)
- **Interleaved output is real.** Look at the `[rank1]` line. `[rank3]`'s output landed on the same line even though each rank makes a single `print` call with `flush=True`. Processes writing to a shared terminal give no line-atomicity guarantees. At 4 ranks this is a curiosity. At 1,024 it makes logs unreadable, which is why the tools in the previous section exist.

Things to try, each of which teaches one of this chapter's lessons:

1. **Break SPMD.** Wrap the `all_reduce` in `if rank != 1:`. The job hangs. Ranks 0, 2 and 3 wait forever for rank 1. Run `py-spy dump` on each PID and you will see three stacks in `all_reduce` and one somewhere else.
2. **Kill a rank.** Add `if rank == 2: os._exit(1)` before the all-reduce. torchrun notices the dead worker and tears down the rest. Add `--max-restarts=2` and watch it re-rendezvous and retry. It fails the same way each time and then gives up, because restarts fix transient faults, not deterministic bugs.
3. **Change the world size,** compare the printed gradient, and confirm the `match` check still passes. Then make the shards uneven (for example 8 samples on 3 ranks) and see the check fail, because the average of per-shard means is no longer the global mean.

**A gotcha we hit while testing this, which is itself a distributed-systems lesson.** The first attempt used `torchrun --standalone`, which hung on macOS. c10d kept retrying the rendezvous with warnings like `The IPv6 network addresses of (...ip6.arpa, ...) cannot be retrieved (gai error: 8 - nodename nor servname provided, or not known)`. The machine's hostname did not resolve to an address the processes could reach. Binding the rendezvous explicitly to the loopback address (`--master_addr=127.0.0.1`) fixed it. Most "my distributed job hangs at startup" reports are a naming or addressing problem like this one. If you see it on Linux, check that `hostname` resolves to a reachable address. Also see [Firewalls and why MASTER_PORT gets blocked](./networking.md#firewalls-and-why-master_port-gets-blocked).


## You can now read

- [Model parallelism](../training/model-parallelism/README.md). DP, ZeRO/FSDP, TP, PP, sequence and expert parallelism, and how they combine. You know why each exists (the memory/compute/communication trade) and which collectives each uses.
- [Parallelism From First Principles](../training/model-parallelism/derivations.md). The memory and communication cost derivations. The "bytes per parameter" and "2(N-1)/N x S" arithmetic above is where they start.
- [Fault tolerance](../training/fault-tolerance/README.md). Spare nodes, watchdogs and the automation around them. You can now compute why they are necessary from a per-node failure rate.
- [Orchestration](../orchestration/README.md), [Slurm](../orchestration/slurm/README.md) and [Kubernetes](../orchestration/kubernetes/README.md). How launchers allocate nodes and start one process per GPU, then rendezvous.
- [Orchestration internals](../orchestration/internals.md). Schedulers, controllers and their state stores. You know what etcd and Raft are doing underneath.
- [Checkpoints](../training/checkpoints/README.md) and [reproducibility](../training/reproducibility/README.md). Checkpoint/restart, atomic saves and the limits of determinism across ranks.
- [Debugging PyTorch programs](../debug/pytorch.md). py-spy across nodes, NCCL flight recorder, hang diagnosis.
- [Communication patterns](../network/comms.md) and [collective internals](../network/collectives-internals.md). The collective algorithms and their cost models.


## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications* (O'Reilly). The best single book on replication, consistency, consensus, clocks and "the trouble with distributed systems" (chapters 5, 8 and 9 are the relevant ones).
- MIT 6.824 / 6.5840 *Distributed Systems* (Robert Morris), lecture videos and notes at https://pdos.csail.mit.edu/6.824/. The labs build Raft and MapReduce from scratch.
- PyTorch distributed documentation: `torch.distributed` (https://pytorch.org/docs/stable/distributed.html), `torchrun` and elastic rendezvous (https://pytorch.org/docs/stable/elastic/run.html), and the DDP design notes (https://pytorch.org/docs/stable/notes/ddp.html).
- Diego Ongaro and John Ousterhout, *In Search of an Understandable Consensus Algorithm* (USENIX ATC 2014). The Raft paper, and the visualization at https://raft.github.io/.
- Leslie Lamport, *Time, Clocks, and the Ordering of Events in a Distributed System* (CACM 1978).
- Leslie Valiant, *A Bridging Model for Parallel Computation* (CACM 1990). The BSP paper.
- Jeffrey Dean and Luiz André Barroso, *The Tail at Scale* (CACM 2013).
- Mu Li et al., *Scaling Distributed Machine Learning with the Parameter Server* (OSDI 2014), and Sergeev and Del Balso, *Horovod: fast and easy distributed deep learning in TensorFlow* (2018). The two sides of the PS vs all-reduce history.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (SC 2020). The 16-bytes-per-parameter accounting.
- Llama Team, *The Llama 3 Herd of Models* (2024). Section on infrastructure and reliability for real failure rates at 16K-GPU scale.
- Shen Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (VLDB 2020). How DDP's gradient bucketing and overlap work.
