# Foundations: Distributed Systems for ML Engineers

## Prerequisites

- [Concurrency and Parallelism](concurrency.md) - locks, deadlocks and SPMD carry over directly to many machines.
- [Computer Networking](networking.md) - TCP, rendezvous and collectives are introduced there.

This chapter is for readers who know the ML math and use PyTorch every day, but who have never had to think about programs that run on many machines at once. You can get a long way on one GPU without that knowledge. You cannot train a frontier model without it. The first time a 512-GPU job hangs with no error message, or one slow node drags the whole cluster down, or two ranks quietly end up with different weights, you are debugging a distributed system, whether you meant to build one or not.

The chapter starts from zero and works up to what you need to read the [model parallelism](../training/model-parallelism/README.md), [fault tolerance](../training/fault-tolerance/README.md) and [orchestration](../orchestration/README.md) chapters, and after that the more advanced [parallelism derivations](../training/model-parallelism/derivations.md) and [orchestration internals](../orchestration/internals.md). The emphasis is on *why* things are built the way they are. Every odd-looking design in a distributed training stack (rendezvous, barriers, watchdog timeouts, the restart-the-whole-job reflex) exists because of one or two facts about distributed systems. Once you know those facts, the designs stop looking odd.

All arithmetic in this chapter was checked with `python3`. The runnable example at the end was run with PyTorch 2.15 on macOS with 2 and 4 CPU processes. Where a number depends on hardware, software version or an assumption, the text says so.

Conventions for the code in this chapter:

- Almost every section has a small, self-contained script. Save it under the file name in its first comment and run it. Pure-Python scripts need only `python3` (run with CPython 3.14 on an Apple Silicon Mac). Scripts marked **[torch]** also need `pip install torch` (the CPU build is fine; run with PyTorch 2.15).
- Most of the distributed-systems ideas (elections, gossip, clocks, quorums) are shown as **simulations**: a whole cluster in one Python process, with the network replaced by a queue that can delay, drop or partition messages. That is how these algorithms are usually taught and tested, because a simulation is deterministic: the same seed gives the same "failure", so you can replay it.
- Output shown under a script is real output from running it, unless the block is labelled **untested** (Linux-only commands, or things that need a real cluster or extra services).
- Many simulations end with `assert`s. They are the self-check: if you change the code and break the property the section is about, the script fails loudly.

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

### Try this

Turn the arithmetic above into a calculator you can point at your own model:

```python
# why.py -- memory and time for your own model; edit the numbers
N, D, gpu_mem_gb = 8e9, 2e12, 80          # params, training tokens, memory per GPU
peak, mfu, gpus = 989e12, 0.40, 256       # bf16 dense peak (H100 SXM), utilization, GPU count

state_gb = N * 16 / 1e9                   # 16 bytes/param mixed-precision Adam
days = 6 * N * D / (peak * mfu * gpus) / 86400
print(f"training state {state_gb:,.0f} GB -> at least {state_gb / gpu_mem_gb:.1f} GPUs just to hold it")
print(f"6ND = {6 * N * D:.2e} FLOPs -> {days:.1f} days on {gpus} GPUs at {mfu:.0%} MFU")
```

```
$ python3 why.py
training state 128 GB -> at least 1.6 GPUs just to hold it
6ND = 9.60e+22 FLOPs -> 11.0 days on 256 GPUs at 40% MFU
```

1. An 8B model does not fit its training state on one 80 GB GPU either. How many bytes per parameter would you need to get it onto one card? Which items in the 16-byte table could you drop or shrink (see [the dtype chapter](../training/dtype.md))?
2. Set `mfu=0.25` (the Llama 3.1 cross-check above). How many days does the 70B, 15T-token run take on 16,384 GPUs?
3. Hold the GPU count fixed and double `D`. Then hold `D` fixed and double `N`. Which one also changes the number of GPUs you need just to hold the state? That asymmetry is why memory and time are separate reasons to distribute.


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

### Try this

See the vocabulary as environment variables. The launcher is what turns "run this script" into "run N copies, each told who it is" **[torch]**:

```python
# env_dump.py -- what the launcher tells each process
import os
keys = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "MASTER_ADDR", "MASTER_PORT"]
print(" ".join(f"{k}={os.environ.get(k)}" for k in keys), flush=True)
```

Pretend to have two nodes with two GPUs each by starting two `torchrun` agents on one laptop, one per "node" (`GROUP_RANK` is torchrun's name for the node rank):

```
$ for n in 0 1; do
    torchrun --nnodes=2 --node_rank=$n --nproc_per_node=2 \
             --master_addr=127.0.0.1 --master_port=29514 env_dump.py > node$n.out 2>&1 &
  done; wait
$ cat node0.out node1.out | grep RANK= | sort
RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 GROUP_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29514
RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 GROUP_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29514
RANK=2 LOCAL_RANK=0 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 GROUP_RANK=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29514
RANK=3 LOCAL_RANK=1 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 GROUP_RANK=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29514
```

1. Check `rank = node_rank x nproc_per_node + local_rank` against every line.
2. Start only the `node_rank=0` agent and leave out the other. What does it do while it waits, and for how long before it gives up? (That waiting is rendezvous, covered under Coordination below.)
3. Under Slurm the same information arrives as `SLURM_PROCID`, `SLURM_LOCALID` and `SLURM_NTASKS`. Write the three-line mapping from those to `RANK`, `LOCAL_RANK` and `WORLD_SIZE` (see [Slurm](../orchestration/slurm/README.md)).


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

The first two fallacies are easy to put numbers on. Below is a toy network: each message has a 0.1% chance of being lost, usually takes about 10 microseconds, and once in a thousand messages takes a 200 ms detour (a TCP retransmit, a full switch queue, a GC pause on the receiver). A synchronous training step needs one message from *every* rank, so it waits for the slowest:

```python
# fallacies.py -- what "the network is reliable" and "latency is zero" cost you
import random, statistics

random.seed(0)
N_MSG, LOSS = 10_000, 0.001          # 0.1% packet loss per message

def send():
    """One message over a toy network: returns latency in ms, or None if lost."""
    if random.random() < LOSS:
        return None
    base = 0.010                      # 10 us base latency (fast datacenter link)
    jitter = random.expovariate(1 / 0.005)   # mostly small, occasionally large
    if random.random() < 0.001:       # rare slow path: retransmit, queueing, GC
        jitter += 200.0
    return base + jitter

lat = [send() for _ in range(N_MSG)]
arrived = [x for x in lat if x is not None]
print(f"lost:   {N_MSG - len(arrived)} of {N_MSG}")
q = statistics.quantiles(arrived, n=1000)
print(f"median: {statistics.median(arrived):.3f} ms   p99: {q[989]:.3f} ms   p99.9: {q[998]:.3f} ms   max: {max(arrived):.1f} ms")

# a "step" that needs one message from each of R ranks waits for the slowest one
for R in (8, 1024):
    steps = [max(send() or float("inf") for _ in range(R)) for _ in range(1000)]
    hung = sum(s == float("inf") for s in steps)
    done = [s for s in steps if s != float("inf")]
    print(f"{R:5d} ranks: median step comm {statistics.median(done):.3f} ms, "
          f"steps with a lost message: {hung}/1000")
```

```
$ python3 fallacies.py
lost:   7 of 10000
median: 0.013 ms   p99: 0.033 ms   p99.9: 0.050 ms   max: 200.0 ms
    8 ranks: median step comm 0.023 ms, steps with a lost message: 7/1000
 1024 ranks: median step comm 200.012 ms, steps with a lost message: 671/1000
```

Measured message by message, this network looks excellent: a 13-microsecond median and a p99.9 of 50 microseconds. At 1,024 ranks the *median* step pays the rare 200 ms path, and two out of three steps include at least one lost message. A real stack recovers lost messages by retransmitting, which turns each loss into another slow path. This is the same arithmetic as the stragglers section below.

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

You can see this with four tiny TCP servers on your laptop: one healthy, one that accepts a connection and never answers (a hung process), one that answers after 2 seconds (a slow process), and a port where the process has exited (a crash). The client uses a 1-second timeout, as any real client must:

```python
# partial.py -- a crashed, a hung and a slow peer look the same from outside
import socket, threading, time

def serve(sock, behaviour):
    while True:
        conn, _ = sock.accept()
        if behaviour == "hung":
            continue                  # accept, never answer (stuck in a syscall, GC...)
        if behaviour == "slow":
            time.sleep(2.0)           # answers, but after our timeout
        conn.sendall(b"pong")

def start(behaviour):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen()
    threading.Thread(target=serve, args=(s, behaviour), daemon=True).start()
    return s.getsockname()[1]

ports = {b: start(b) for b in ("healthy", "hung", "slow")}
# a crashed process: its port has no listener any more
tmp = socket.socket(); tmp.bind(("127.0.0.1", 0)); ports["crashed"] = tmp.getsockname()[1]; tmp.close()

for name, port in ports.items():
    t0 = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as c:
            c.sendall(b"ping")
            reply = c.recv(4)
            result = f"reply {reply!r}"
    except ConnectionRefusedError:
        result = "ConnectionRefusedError"
    except socket.timeout:
        result = "timeout"
    print(f"{name:8s} -> {result:24s} after {time.monotonic() - t0:.2f}s")
```

```
$ python3 partial.py
healthy  -> reply b'pong'            after 0.00s
hung     -> timeout                  after 1.00s
slow     -> timeout                  after 1.00s
crashed  -> ConnectionRefusedError   after 0.00s
```

"Hung" and "slow" are indistinguishable: both are a timeout. The crashed process is the *lucky* case. Its machine is still up, so the operating system answers on its behalf and refuses the connection. If the whole node had lost power, or the network cable had been pulled, nothing would answer at all and the crash would also look like a timeout. A clean, immediate error only happens when part of the failed system is still healthy enough to report it.

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

### Try this

Formulas like Young's rest on assumptions (exponential failures, a fixed checkpoint cost). Check them with a simulation before trusting them:

```python
# young.py -- simulate checkpointing under random failures; compare with Young's formula
import math, random
random.seed(0)
MTBF, C, WORK = 1440.0, 5.0, 200_000.0      # minutes: job MTBF, checkpoint cost, useful work

def wall_time(T):
    t, done = 0.0, 0.0
    next_fail = random.expovariate(1 / MTBF)
    while done < WORK:
        seg = min(T, WORK - done)
        if t + seg + C <= next_fail:          # finish segment and its checkpoint
            t += seg + C; done += seg
        else:                                 # failure: lose the partial segment
            t = next_fail
            next_fail = t + random.expovariate(1 / MTBF)
    return t

print(f"Young's T = sqrt(2*C*MTBF) = {math.sqrt(2 * C * MTBF):.0f} min")
for T in (15, 60, 120, 240, 960):
    over = sum(wall_time(T) for _ in range(20)) / 20 / WORK - 1
    print(f"T = {T:4d} min -> overhead {over:6.1%}")
```

```
$ python3 young.py
Young's T = sqrt(2*C*MTBF) = 120 min
T =   15 min -> overhead  34.3%
T =   60 min -> overhead  10.8%
T =  120 min -> overhead   9.0%
T =  240 min -> overhead  11.5%
T =  960 min -> overhead  41.5%
```

The minimum is at Young's interval, and the simulated 9.0% is close to the first-order 8.3% estimate. The curve is flat near the optimum: anywhere from 1 to 4 hours costs within a few percent. Getting the order of magnitude right matters much more than the exact value.

1. Add a restart cost (say 15 minutes of scheduling and loading after each failure) to `wall_time`. How much does the overhead go up, and does the best `T` move?
2. Cut `C` to 0.5 minutes (an async checkpoint) and find the new best `T`. Compare with `sqrt(2 x 0.5 x 1440)`.
3. In `partial.py`, change the client timeout to 3 seconds. Which of the four peers changes category? Now make the "slow" server sleep 5 seconds. There is no timeout value that is right for every peer, and that is the heartbeat section's point.


## Communication models

### Shared memory vs message passing

There are two basic ways for parallel workers to cooperate:

- **Shared memory:** workers read and write the same memory. Threads in one process, or GPUs reading each other's memory over NVLink, work this way. It is fast and convenient, and it needs locks or other synchronization to avoid races. [Concurrency and Parallelism](./concurrency.md) covers this.
- **Message passing:** workers have private memory and cooperate only by sending each other messages. Separate processes on separate machines have no other option.

Distributed training is message passing. Even 8 GPUs inside one node are usually driven by 8 separate processes, and they exchange data through explicit operations (NCCL collectives over NVLink), not through shared Python objects. One practical consequence: **there is no global state.** A Python variable set on rank 0 does not exist on rank 3. If rank 3 needs it, someone has to send it.

```python
# no_global_state.py -- separate processes share nothing unless you send it
import multiprocessing as mp

config = {"lr": 0.1}

def worker(rank, q):
    if rank == 0:
        config["lr"] = 0.01            # changes rank 0's private copy only
    q.put((rank, config["lr"]))        # message passing: the only way out

if __name__ == "__main__":
    mp.set_start_method("spawn")       # each worker re-imports this file
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(r, q)) for r in range(3)]
    for p in ps: p.start()
    for p in ps: p.join()
    print(sorted(q.get() for _ in ps))
    print("parent sees lr =", config["lr"])
```

```
$ python3 no_global_state.py
[(0, 0.01), (1, 0.1), (2, 0.1)]
parent sees lr = 0.1
```

Rank 0's change is invisible to ranks 1 and 2 and to the parent. The only thing that crossed a process boundary is what went into the queue. In a training job the "queue" is a collective. If rank 0 decides something (say, an early-stopping flag or a learning-rate change computed from an eval), it has to `broadcast` the value. Otherwise ranks silently disagree, and the next mismatched collective hangs.

### Synchronous vs asynchronous

- **Synchronous:** a sender waits until the communication completes, and everyone proceeds in lockstep.
- **Asynchronous:** a sender fires off a message and continues. Results arrive whenever they arrive.

There are two separate meanings here. Keep them apart:

1. **Algorithmic synchrony.** Do all workers apply the same gradient at the same step (synchronous SGD), or does each worker apply whatever possibly-stale gradient it has (asynchronous SGD, Hogwild-style)? Modern LLM training is almost entirely synchronous.
2. **API-level asynchrony.** Does the *function call* block? In PyTorch, `dist.all_reduce(t, async_op=True)` returns a handle immediately, and you call `.wait()` later. With NCCL, even "blocking" collectives are only enqueued on a CUDA stream, so the CPU runs ahead of the GPU. This is how frameworks overlap communication with computation. DDP starts all-reducing the gradients of the last layers while backward is still computing the gradients of earlier ones.

So a training job is algorithmically synchronous (everyone agrees on every step) and implemented asynchronously underneath (communication overlapped with compute).

Here is the API-level difference with two CPU processes and the gloo backend **[torch]**:

```python
# async_op.py -- blocking vs async_op=True collectives, 2 CPU ranks with gloo
import os, time, torch, torch.distributed as dist, torch.multiprocessing as mp

def run(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    t = torch.ones(32 * 1024 * 1024)                 # (32M,) fp32 = 128 MB

    t0 = time.perf_counter()
    dist.all_reduce(t)                               # blocks until done (gloo on CPU)
    blocking = time.perf_counter() - t0

    t0 = time.perf_counter()
    work = dist.all_reduce(t, async_op=True)         # returns a handle immediately
    issued = time.perf_counter() - t0
    x = torch.randn(1024, 1024); x @ x               # "compute" overlapped with comm
    work.wait()                                      # now the result is safe to read
    total = time.perf_counter() - t0
    if rank == 0:
        print(f"blocking all_reduce: {blocking*1e3:6.1f} ms")
        print(f"async issue returned in {issued*1e3:6.3f} ms; issue+matmul+wait {total*1e3:6.1f} ms")
        print("value after two sums:", t[0].item(), "(expected", world**2, ")")
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)
```

```
$ python async_op.py
blocking all_reduce:   95.0 ms
async issue returned in  0.166 ms; issue+matmul+wait  225.2 ms
value after two sums: 4.0 (expected 4 )
```

The async call returned in 0.17 ms, and the work happened in the background until `wait()`. Notice that on a laptop CPU the overlapped version gained nothing. Across four runs the issue+matmul+wait total was 225, 86, 77 and 102 ms, against blocking times of 95, 86, 71 and 112 ms, so it was sometimes slower and never meaningfully faster, and the noise is larger than any difference. The "communication" and the "compute" competed for the same CPU cores and memory bandwidth, so there was nothing to gain by overlapping them. On GPUs, NCCL runs on its own CUDA stream and the bytes move over NVLink or the NIC, not through the cores doing the matmul. That is why overlap pays off there. The lesson generalizes: overlap helps only when the two activities use *different* resources. Reading `t` before `work.wait()` is a race: you may see a partially reduced tensor.

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

### Try this

Simulate BSP step time directly. Each rank takes about 300 ms with a little jitter, plus a 1-in-1000 chance of a 500 ms hiccup:

```python
# straggler.py -- BSP step time is the max over ranks
import random, statistics
random.seed(0)

def rank_time():
    t = random.gauss(300, 5)                  # ms of fwd+bwd, small jitter
    if random.random() < 0.001:
        t += 500                              # rare hiccup: GC, page fault, slow read
    return t

for R in (1, 8, 64, 1024, 8192):
    steps = [max(rank_time() for _ in range(R)) for _ in range(300)]
    print(f"{R:5d} ranks: mean step {statistics.mean(steps):6.1f} ms  "
          f"(single-rank mean is ~300.5 ms)")
```

```
$ python3 straggler.py
    1 ranks: mean step  301.4 ms  (single-rank mean is ~300.5 ms)
    8 ranks: mean step  308.7 ms  (single-rank mean is ~300.5 ms)
   64 ranks: mean step  329.8 ms  (single-rank mean is ~300.5 ms)
 1024 ranks: mean step  641.3 ms  (single-rank mean is ~300.5 ms)
 8192 ranks: mean step  806.9 ms  (single-rank mean is ~300.5 ms)
```

Two effects show up. At 8 ranks the step is already 8 ms slower with no hiccups at all, because the maximum of 8 Gaussian samples sits about 1.4 standard deviations above the mean. At 1,024 ranks most steps include a hiccup somewhere, so the job runs at roughly half speed even though each rank, on average, is only 0.5 ms slower than ideal.

1. Make one rank persistently 10% slow (`if r == 0: t *= 1.1`, passing the rank index in). What is the job slowdown at 8 ranks and at 1,024 ranks?
2. Make the hiccup probability 1e-4 instead of 1e-3. At what rank count does the mean step reach 400 ms now?
3. Asynchronous SGD removes the max: each worker applies its gradient whenever it is ready. Write down what you give up (hint: the "identical replicas" argument in the CAP section).


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

```python
# clocks.py -- wall clock vs monotonic clock, and how much a clock can drift
import time
print("time.time()      :", time.time(), " (seconds since 1970, can jump when NTP steps it)")
print("time.monotonic() :", time.monotonic(), " (arbitrary origin, never goes backwards)")
print("resolution (s)   : time", time.get_clock_info("time").resolution,
      " monotonic", time.get_clock_info("monotonic").resolution)
for ppm in (1, 20, 100):
    print(f"{ppm:4d} ppm drift -> {ppm * 1e-6 * 86400 * 1e3:8.1f} ms/day")
```

```
$ python3 clocks.py
time.time()      : 1793913302.648659  (seconds since 1970, can jump when NTP steps it)
time.monotonic() : 106497.647634  (arbitrary origin, never goes backwards)
resolution (s)   : time 1.0000000000000002e-06  monotonic 4.166666666666666e-08
   1 ppm drift ->     86.4 ms/day
  20 ppm drift ->   1728.0 ms/day
 100 ppm drift ->   8640.0 ms/day
```

The monotonic value is roughly the time since boot. It is meaningless on another machine, which is exactly why you must not compare it across nodes.

To see how far your own clock is from a time server, ask one. On macOS, `sntp` prints the offset and its uncertainty (real output from a laptop on home Wi-Fi):

```
$ sntp time.apple.com
+0.088563 +/- 0.033896 time.apple.com 2620:149:a33:3000::21
```

This laptop was 89 ms ahead, give or take 34 ms. The uncertainty comes from not knowing how the round trip splits between the outbound and return paths. On a Linux cluster node, ask the local NTP daemon instead (**untested** here, Linux with chrony):

```
$ chronyc tracking
Reference ID    : 0A000001 (ntp1.cluster.local)
System time     : 0.000012345 seconds fast of NTP time
Last offset     : +0.000003210 seconds
RMS offset      : 0.000021000 seconds
...
$ timedatectl          # systemd's summary: "System clock synchronized: yes"
```

The values in that block are illustrative. What to look for is `System time` within a few hundred microseconds on a healthy data-center network. A node that says "not synchronized", or is seconds off, will make your merged logs lie and will break anything that relies on wall-clock leases or certificate validity.

### Why you cannot trust timestamps across nodes

A GPU step takes a few hundred milliseconds. A collective can take microseconds to milliseconds. Cross-node clock error can be milliseconds. So when you merge logs from 64 nodes and sort by timestamp, the order of events that happened within a few milliseconds of each other on different nodes is essentially random. You can easily see "rank 12 received the tensor" logged before "rank 3 sent it".

```python
# skew.py -- merging logs from two nodes whose clocks disagree by 3 ms
true_t = 1000.000                  # real time the send happened (s)
skew = {"node0": +0.003, "node1": 0.000}   # node0's clock runs 3 ms ahead
net_delay = 0.0005                 # 0.5 ms one-way

logs = [
    ("node0", true_t + skew["node0"], "rank3 send tensor seq=17"),
    ("node1", true_t + net_delay + skew["node1"], "rank12 recv tensor seq=17"),
]
for node, ts, msg in sorted(logs, key=lambda r: r[1]):
    print(f"{ts:.4f} {node} {msg}")
recv_first = sorted(logs, key=lambda r: r[1])[0][2].startswith("rank12")
print("merged-by-timestamp says the receive happened first:", recv_first)
```

```
$ python3 skew.py
1000.0005 node1 rank12 recv tensor seq=17
1000.0030 node0 rank3 send tensor seq=17
merged-by-timestamp says the receive happened first: True
```

Three milliseconds of skew, which is well within what NTP gives you, reversed cause and effect. The `seq=17` field is what saves you: it says these two lines are the same message, whatever the clocks say.

Practical rules for debugging:

1. **Order by logical progress, not wall-clock time.** Log the training step, the micro-batch index and a per-rank collective sequence number. "Rank 12 is at step 4,001, every other rank is at step 4,002" tells you far more than any timestamp. PyTorch's NCCL flight recorder records exactly this kind of per-collective sequence number. See [the flight recorder section](../debug/pytorch.md#the-flight-recorder-trace).
2. **Compare timestamps only within one node,** or across nodes only when the gaps are much larger than plausible clock skew (seconds, not milliseconds).
3. **Use durations, not absolute times,** to find stragglers. "Rank 40's forward took 900 ms, everyone else took 600 ms" is robust to clock skew.

The theoretical answer to "what happened before what" without synchronized clocks is Lamport's logical clocks (1978): every process keeps a counter, increments it on each event and attaches it to every message, and on receiving a message sets its counter to `max(own, received) + 1`. You will not implement this for training, but step counters and collective sequence numbers are the same idea.

### Logical clocks: Lamport and vector clocks

Two definitions first:

- Event `a` **happened before** event `b` (written `a -> b`) if `a` came before `b` on the same process, or `a` was the sending of a message that `b` received, or there is a chain of such steps from `a` to `b`. This is the only ordering a distributed system can actually observe.
- Two events are **concurrent** if neither happened before the other. Nothing that one process did could have influenced the other, so "which came first" has no meaningful answer.

A **Lamport clock** guarantees one direction: if `a -> b` then `L(a) < L(b)`. The converse does not hold. `L(a) < L(b)` does *not* tell you that `a` caused `b`. A **vector clock** (Fidge and Mattern, 1988) fixes that by keeping one counter *per process*. Each process increments its own entry on every event, and on receive takes the elementwise max with the incoming vector, then increments its own entry. Then `a -> b` exactly when `V(a) <= V(b)` elementwise and `V(a) != V(b)`. If neither vector is `<=` the other, the events are concurrent.

```python
# vclock.py -- Lamport clocks and vector clocks on a 3-process message trace
# trace: (process, kind, peer)  kind = "local" | "send" | "recv"; messages matched FIFO
trace = [
    ("A", "local", None), ("A", "send", "B"),
    ("C", "local", None), ("C", "send", "B"),
    ("B", "recv", "A"), ("B", "recv", "C"), ("B", "send", "A"),
    ("C", "local", None),
    ("A", "recv", "B"),
]
procs = "ABC"
lamport = {p: 0 for p in procs}
vc = {p: {q: 0 for q in procs} for p in procs}
inflight = {}                          # (src, dst) -> list of (lamport, vector) stamps
events = []                            # (name, lamport, vector)

for i, (p, kind, peer) in enumerate(trace):
    if kind == "recv":
        lt, v = inflight[(peer, p)].pop(0)
        lamport[p] = max(lamport[p], lt) + 1
        vc[p] = {q: max(vc[p][q], v[q]) for q in procs}
        vc[p][p] += 1
    else:
        lamport[p] += 1
        vc[p][p] += 1
        if kind == "send":
            inflight.setdefault((p, peer), []).append((lamport[p], dict(vc[p])))
    name = f"e{i}:{p}.{kind}" + (f"({peer})" if peer else "")
    events.append((name, lamport[p], dict(vc[p])))
    print(f"{name:14s} lamport={lamport[p]}  vector={[vc[p][q] for q in procs]}")

def happened_before(u, v):             # u -> v iff u <= v elementwise and u != v
    return all(u[q] <= v[q] for q in procs) and u != v

a_local, c_last = events[0], events[7]
print("\nA's first local event vs C's last local event:")
print("  lamport:", a_local[1], "<", c_last[1], "=", a_local[1] < c_last[1], " (looks ordered)")
print("  vector : before?", happened_before(a_local[2], c_last[2]),
      " after?", happened_before(c_last[2], a_local[2]), " -> concurrent")
assert not happened_before(a_local[2], c_last[2]) and not happened_before(c_last[2], a_local[2])
assert happened_before(events[1][2], events[8][2])   # A's send -> ... -> A's final recv
```

```
$ python3 vclock.py
e0:A.local     lamport=1  vector=[1, 0, 0]
e1:A.send(B)   lamport=2  vector=[2, 0, 0]
e2:C.local     lamport=1  vector=[0, 0, 1]
e3:C.send(B)   lamport=2  vector=[0, 0, 2]
e4:B.recv(A)   lamport=3  vector=[2, 1, 0]
e5:B.recv(C)   lamport=4  vector=[2, 2, 2]
e6:B.send(A)   lamport=5  vector=[2, 3, 2]
e7:C.local     lamport=3  vector=[0, 0, 3]
e8:A.recv(B)   lamport=6  vector=[3, 3, 2]

A's first local event vs C's last local event:
  lamport: 1 < 3 = True  (looks ordered)
  vector : before? False  after? False  -> concurrent
```

The Lamport clock puts A's first event "before" C's last one, but no message chain connects them. The vector clock correctly reports them as concurrent.

Where you meet these in practice: replicated stores such as Dynamo and Riak used vector clocks (or the related "version vectors") to detect conflicting concurrent writes instead of silently picking one. Vector clocks cost O(number of processes) per message, which is why training stacks use the cheaper version: a single per-communicator sequence number. Every rank issues collectives in the same order (SPMD), so "collective #81,234" names the same logical event on every rank, and a rank stuck at #81,233 is the one behind.

### Try this

1. In `vclock.py`, add a message from C to A after `e7` and check that A's final event is now *after* C's last local event.
2. Change `skew.py` so node0 is 3 ms *behind* instead of ahead. Is the merged order now correct? Is it correct for the right reason?
3. Run `python3 -c "import time; t=time.time(); m=time.monotonic(); time.sleep(5); print(time.time()-t, time.monotonic()-m)"` and, while it sleeps, change your system clock by a minute (macOS: System Settings > General > Date & Time). Only one of the two durations is still about 5 s.


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

Here is rendezvous by hand with a real TCPStore: four processes register their addresses, rank 0 publishes a "unique id" (the role NCCL's id plays), and an atomic counter implements a barrier **[torch]**:

```python
# store.py -- rendezvous and a barrier built from TCPStore primitives
import time, multiprocessing as mp
from datetime import timedelta
import torch.distributed as dist

PORT, WORLD = 29512, 4

def proc(rank):
    # rank 0 hosts the server; everyone (including rank 0) is a client
    store = dist.TCPStore("127.0.0.1", PORT, WORLD, is_master=(rank == 0),
                          timeout=timedelta(seconds=10))
    store.set(f"addr/{rank}", f"10.0.0.{rank}:4{rank}000")   # 1. register
    if rank == 0:
        store.set("unique_id", "0x4f2a")                      # 4. rank 0 publishes an id
    uid = store.get("unique_id").decode()                     # blocks until it exists
    time.sleep(0.2 * rank)                                    # ranks arrive at different times
    arrived = store.add("barrier/0", 1)                       # atomic counter
    if arrived == WORLD:
        store.set("barrier/0/done", "1")                      # last one in opens the gate
    store.wait(["barrier/0/done"])
    peers = [store.get(f"addr/{r}").decode() for r in range(WORLD)]
    print(f"[rank{rank}] arrived #{arrived} uid={uid} peers={peers}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    ps = [mp.Process(target=proc, args=(r,)) for r in range(WORLD)]
    for p in ps: p.start()
    for p in ps: p.join()
```

```
$ python store.py
[rank1] arrived #2 uid=0x4f2a peers=['10.0.0.0:40000', '10.0.0.1:41000', '10.0.0.2:42000', '10.0.0.3:43000']
[rank2] arrived #3 uid=0x4f2a peers=['10.0.0.0:40000', '10.0.0.1:41000', '10.0.0.2:42000', '10.0.0.3:43000']
[rank0] arrived #1 uid=0x4f2a peers=['10.0.0.0:40000', '10.0.0.1:41000', '10.0.0.2:42000', '10.0.0.3:43000']
[rank3] arrived #4 uid=0x4f2a peers=['10.0.0.0:40000', '10.0.0.1:41000', '10.0.0.2:42000', '10.0.0.3:43000']
```

Nobody printed until the last rank arrived. That is the barrier: `add` is atomic on the server, so exactly one process sees `arrived == 4` and opens the gate. The addresses are fake strings, but this is the same sequence `init_process_group` performs with real ones. Kill rank 0 (the store host) halfway through and every other rank fails with a store timeout, which is the single point of failure described above.

The same primitives in etcd, from its command-line client (**untested** here; needs a running etcd, for example the official binary from https://etcd.io). The output is sketched from the etcd v3 docs:

```
$ etcdctl put /job42/addr/0 10.0.0.1:41233
OK
$ etcdctl get /job42/addr/0
/job42/addr/0
10.0.0.1:41233
$ etcdctl lease grant 10                   # a lease: keys attached to it vanish after 10 s
lease 694d77aa9e38260f granted with TTL(10s)
$ etcdctl put --lease=694d77aa9e38260f /job42/alive/0 1
OK
$ etcdctl lease keep-alive 694d77aa9e38260f   # a heartbeat: renews until you Ctrl-C
$ etcdctl watch /job42/alive --prefix      # blocks; prints DELETE when a member's lease expires
```

Lease plus keep-alive plus watch is how etcd-based systems do membership: a member that stops heartbeating loses its lease, its key disappears, and everyone watching is told. Kubernetes node heartbeats work on the same principle (they are `Lease` objects).

### Heartbeats, timeouts and "slow vs dead"

To notice that a peer has died, processes send **heartbeats**, periodic "I'm alive" messages, and declare a peer dead if they hear nothing for some **timeout**. torchrun's c10d rendezvous, for example, by default has each agent send a keep-alive every 5 seconds and treats a node as gone after 3 missed attempts (defaults read from the PyTorch 2.15 source; they may differ in other versions).

Here is the fundamental problem. **In an asynchronous network you cannot distinguish a dead process from a slow one.** Both are silent. A timeout does not *detect* death. It *decides* that you will treat silence longer than T as death. Every value of T is wrong in some case:

| Timeout | What goes wrong |
|---|---|
| too short | A slow-but-alive rank (long checkpoint write, GC pause, first-step compilation, congested filesystem) is declared dead. The job is torn down and restarted for nothing, maybe again and again. |
| too long | A rank that really is dead holds the job hostage. Thousands of GPUs sit idle until the timeout fires. |

This is why PyTorch's distributed timeouts look the way they do. The default process-group timeout is long: 30 minutes for gloo (verified in PyTorch 2.15), and 10 minutes for NCCL in recent releases (from memory; check `torch.distributed.constants` for your version). It has to be long enough for legitimate slow operations like a rank-0-only checkpoint save, and short enough that a hang does not burn a whole night. If you have one known-slow operation, the right fix is usually to raise the timeout for it specifically (pass `timeout=` to `init_process_group` or `new_group`), not to shorten it globally.

Fixed timeouts are not the only option. The **phi-accrual failure detector** (Hayashibara et al., 2004; used by Cassandra and Akka) learns the normal distribution of heartbeat gaps and outputs a suspicion level `phi = -log10(P(a gap this long))` instead of yes/no. `phi = 8` means "a gap this long happens by chance 1 time in 10^8". When heartbeats are very regular it notices silence quickly, and when they are noisy it waits longer. Here are three detectors on the same heartbeat trace: a peer that pauses for 6 seconds at t of about 30 s (alive, maybe writing a checkpoint), then really dies at about t = 61 s:

```python
# heartbeat.py -- fixed-timeout vs phi-accrual failure detection on the same trace
import math, random, statistics
random.seed(1)

# heartbeat arrival times (s) from one peer: every ~1 s with jitter,
# then a 6 s GC/checkpoint pause at t~30, then death at t~60
t, arrivals = 0.0, []
while t < 60:
    t += random.gauss(1.0, 0.1)
    if 30 < t < 31.2:
        t += 6.0                      # alive but silent for 6 s
    arrivals.append(t)

def fixed_timeout(now, last, T):
    return now - last > T

def phi(now, last, intervals):
    # phi = -log10(P(a heartbeat this late)) assuming normal inter-arrival times
    mu, sd = statistics.mean(intervals), max(statistics.stdev(intervals), 0.1)
    p_later = 0.5 * math.erfc((now - last - mu) / (sd * math.sqrt(2)))
    return -math.log10(max(p_later, 1e-300))

for name, check in [("timeout 3s", lambda now, last, iv: fixed_timeout(now, last, 3.0)),
                    ("timeout 10s", lambda now, last, iv: fixed_timeout(now, last, 10.0)),
                    ("phi > 8", lambda now, last, iv: phi(now, last, iv[-20:]) > 8)]:
    episodes, i, now, was = [], 0, 0.0, False
    while now < 75:
        now = round(now + 0.1, 1)
        while i < len(arrivals) and arrivals[i] <= now:
            i += 1
        if i < 3:
            continue
        iv = [b - a for a, b in zip(arrivals[:i], arrivals[1:i])]
        sus = check(now, arrivals[i - 1], iv)
        if sus and not was:
            episodes.append(now)        # trusted -> suspected transition
        was = sus
    print(f"{name:12s} suspected peer at t = {episodes}")
print(f"(truth: 6 s pause near t=30, real death after t={arrivals[-1]:.1f})")
```

```
$ python3 heartbeat.py
timeout 3s   suspected peer at t = [32.4, 63.7]
timeout 10s  suspected peer at t = [70.7]
phi > 8      suspected peer at t = [31.0, 62.3]
(truth: 6 s pause near t=30, real death after t=60.7)
```

Read it as a trade-off table. The 3-second timeout detects the death 3 s after it happens, but it also raised a false alarm during the pause. The 10-second timeout never raised a false alarm, but it left the job hostage for 10 s. Phi-accrual was fastest on the real death (1.6 s), *and* it still fired on the pause, because a 6-second gap really is astronomically unlikely for a peer that has been beating every 1.0 +/- 0.1 s. No detector can see inside the peer. A smarter detector only lets you pick the trade-off more precisely.

When an NCCL collective times out, PyTorch's watchdog thread aborts the communicator and kills the process. That turns a silent hang into a crash, which a launcher can then restart. That is the point: **a crash is much better than a hang,** because a crash can be detected and acted on automatically.

### Gossip: membership without a central store

A TCPStore or etcd is a *central* place where everyone reports. That works for a few thousand processes in one data center. It does not work for a decentralized training swarm with peers joining and leaving over the internet, and it is a single place to overload even in a data center. The alternative is **gossip** (also called epidemic protocols): every node periodically picks a few random peers and exchanges what it knows. News spreads like an infection, reaching everyone in O(log N) rounds, and no node does more than a constant amount of work per round.

The classic gossip failure detector (van Renesse et al., 1998): each node keeps a table of `peer -> heartbeat counter`. Every round it increments its own counter and sends its table to a couple of random peers. A receiver keeps the larger counter for each entry and notes *when it last saw that counter grow*. A peer whose counter has not grown for `T_FAIL` rounds is suspected dead:

```python
# gossip.py -- heartbeat-counter gossip membership (the scheme in Cassandra/Consul-style systems)
import math, random
random.seed(0)
N, FANOUT, T_FAIL = 64, 2, 8          # nodes, peers contacted per round, rounds before "dead"

# each node keeps: peer -> (heartbeat counter, local round when that counter last increased)
table = [{j: (0, 0) for j in range(N)} for _ in range(N)]
alive = set(range(N))

def gossip_round(r):
    for i in alive:
        hb, _ = table[i][i]
        table[i][i] = (hb + 1, r)                    # bump my own heartbeat
    for i in list(alive):
        for j in random.sample([k for k in range(N) if k != i], FANOUT):
            if j not in alive:
                continue                             # message to a dead node is lost
            for peer, (hb, _) in table[i].items():   # j merges i's table: keep the max counter
                if hb > table[j][peer][0]:
                    table[j][peer] = (hb, r)

def suspects(i, r):
    return {p for p, (_, seen) in table[i].items() if r - seen > T_FAIL}

for r in range(1, 41):
    if r == 10:
        alive.discard(13)                            # node 13 crashes silently
        print("round 10: node 13 crashes")
    gossip_round(r)
    if r >= 10:
        knowing = sum(13 in suspects(i, r) for i in alive)
        if knowing:
            print(f"round {r}: {knowing}/{len(alive)} live nodes suspect node 13; "
                  f"false suspicions: {sum(len(suspects(i, r) - {13}) for i in alive)}")
        if knowing == len(alive):
            break
print(f"log2(N) = {math.log2(N):.0f}: gossip spreads news in O(log N) rounds, "
      f"each node sends only {FANOUT} msgs/round")
```

```
$ python3 gossip.py
round 10: node 13 crashes
round 18: 10/63 live nodes suspect node 13; false suspicions: 0
round 19: 35/63 live nodes suspect node 13; false suspicions: 0
round 20: 57/63 live nodes suspect node 13; false suspicions: 0
round 21: 62/63 live nodes suspect node 13; false suspicions: 0
round 22: 63/63 live nodes suspect node 13; false suspicions: 0
log2(N) = 6: gossip spreads news in O(log N) rounds, each node sends only 2 msgs/round
```

`T_FAIL = 8` is the same timeout trade-off as before, now measured in rounds: too small and the randomness of who-talks-to-whom produces false suspicions, too large and detection is slow. **SWIM** (Das et al., 2002), used in HashiCorp's memberlist (and so in Consul and Serf), improves on this. A node *pings* a random peer directly, and if there is no answer, asks k other peers to ping it *indirectly* before suspecting it. That separates "the peer is dead" from "my own link to it is bad", which matters a lot on the internet. Decentralized training systems that run over the public internet (for example the Hivemind library, which uses a Kademlia distributed hash table for peer discovery) live in this world rather than the TCPStore one. See [decentralized training](../training/decentralized-training.md).

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

### A Raft-style leader election you can run

Reading about Raft is less convincing than watching it. Below is the election half of Raft as a discrete-event simulation: five nodes, each with a term, a vote and a role. Messages take 5-15 ms and are dropped across a partition. The script crashes the leader, restarts it, partitions the cluster with the current leader on the minority side, and then heals the partition. Two things to watch for:

- **Randomized election timeouts** (150-300 ms, the range the Raft paper suggests). If every follower timed out at the same moment, they would all become candidates, split the vote, and nobody would win. Randomness makes one node usually time out first and win before the others wake up.
- **Terms as a fencing mechanism.** Every message carries the sender's term. A node that sees a higher term immediately becomes a follower. That is how an old leader learns it has been replaced.

```python
# raft_election.py -- Raft leader election (no log replication) as a discrete-event simulation
# Note: election + heartbeats only; log replication and the "up-to-date log" vote rule are omitted
import heapq, random
random.seed(4)
N = 5
MAJORITY = N // 2 + 1
HEARTBEAT, ELECT_MIN, ELECT_MAX = 50, 150, 300      # ms, the Raft paper's suggested ranges

class Node:
    def __init__(self, i):
        self.i, self.term, self.voted_for, self.role = i, 0, None, "follower"
        self.votes, self.acks, self.up = set(), set(), True
        self.deadline = 0

nodes = [Node(i) for i in range(N)]
events, now, seq = [], 0, 0          # heap of (time, seq, kind, payload)
partition = None                     # None, or a set of node ids cut off from the rest
leaders_by_term = {}                 # term -> set of nodes that became leader in it

def push(t, kind, payload):
    global seq; seq += 1
    heapq.heappush(events, (t, seq, kind, payload))

def reset_timer(n):
    n.deadline = now + random.randint(ELECT_MIN, ELECT_MAX)   # randomized: avoids split votes
    push(n.deadline, "timeout", n.i)

def connected(a, b):
    return partition is None or ((a in partition) == (b in partition))

def send(src, dst, msg):
    if nodes[src].up and connected(src, dst):
        push(now + random.randint(5, 15), "msg", (dst, msg))   # 5-15 ms network delay

def step_down(n, term):
    n.term, n.voted_for, n.role = term, None, "follower"
    reset_timer(n)

def log(s): print(f"t={now:5d}ms  {s}")

def handle(n, m):
    if m["term"] > n.term:
        if n.role == "leader":
            log(f"node {n.i} sees term {m['term']} > {n.term}, steps down")
        step_down(n, m["term"])
    kind = m["kind"]
    if kind == "RequestVote":
        grant = m["term"] == n.term and n.voted_for in (None, m["src"])
        if grant:
            n.voted_for = m["src"]; reset_timer(n)
        send(n.i, m["src"], dict(kind="Vote", term=n.term, src=n.i, granted=grant))
    elif kind == "Vote" and n.role == "candidate" and m["term"] == n.term and m["granted"]:
        n.votes.add(m["src"])
        if len(n.votes) >= MAJORITY:
            n.role = "leader"
            leaders_by_term.setdefault(n.term, set()).add(n.i)
            log(f"node {n.i} becomes LEADER for term {n.term} with votes {sorted(n.votes)}")
            push(now, "beat", n.i)
    elif kind == "AppendEntries" and m["term"] == n.term:
        n.role = "follower"; reset_timer(n)          # a valid leader exists: stay follower
        send(n.i, m["src"], dict(kind="Ack", term=n.term, src=n.i))
    elif kind == "Ack" and n.role == "leader" and m["term"] == n.term:
        n.acks.add(m["src"])

def run_until(t_end):
    global now
    while events and events[0][0] <= t_end:
        now, _, kind, p = heapq.heappop(events)
        if kind == "msg":
            dst, m = p
            if nodes[dst].up: handle(nodes[dst], m)
            continue
        n = nodes[p]
        if not n.up: continue
        if kind == "timeout" and n.role != "leader" and now == n.deadline:
            n.term += 1; n.role = "candidate"; n.voted_for = n.i; n.votes = {n.i}
            log(f"node {n.i} times out, starts election for term {n.term}")
            reset_timer(n)
            for j in range(N):
                if j != n.i: send(n.i, j, dict(kind="RequestVote", term=n.term, src=n.i))
        elif kind == "beat" and n.role == "leader":
            if len(n.acks | {n.i}) < MAJORITY and now % 500 < HEARTBEAT:
                log(f"node {n.i} is leader of term {n.term} but only {len(n.acks)+1}/{N} reachable: cannot commit")
            n.acks = set()
            for j in range(N):
                if j != n.i: send(n.i, j, dict(kind="AppendEntries", term=n.term, src=n.i))
            push(now + HEARTBEAT, "beat", n.i)
    now = t_end

def leader():
    return [n.i for n in nodes if n.up and n.role == "leader"]

for n in nodes: reset_timer(n)
run_until(1000)
old = leader()[0]
log(f"--- crash leader node {old}");   nodes[old].up = False
run_until(2000)
log(f"--- restart node {old}");        nodes[old].up = True; step_down(nodes[old], nodes[old].term)
run_until(2500)
cur = leader()[0]
partition = {cur, (cur + 1) % N}
log(f"--- partition {sorted(partition)} | {sorted(set(range(N)) - partition)}")
run_until(3600)
log(f"--- heal partition; leaders now: {leader()}"); partition = None
run_until(4500)
log(f"final: leader(s) {leader()}, terms {[n.term for n in nodes]}")

assert all(len(v) == 1 for v in leaders_by_term.values()), "two leaders in one term!"
assert len(leader()) == 1
print("safety check passed: at most one leader per term", dict(sorted(leaders_by_term.items())))
```

```
$ python3 raft_election.py
t=  176ms  node 2 times out, starts election for term 1
t=  189ms  node 2 becomes LEADER for term 1 with votes [1, 2, 3]
t= 1000ms  --- crash leader node 2
t= 1169ms  node 4 times out, starts election for term 2
t= 1194ms  node 4 becomes LEADER for term 2 with votes [0, 3, 4]
t= 2000ms  --- restart node 2
t= 2500ms  --- partition [0, 4] | [1, 2, 3]
t= 2665ms  node 3 times out, starts election for term 3
t= 2685ms  node 3 becomes LEADER for term 3 with votes [1, 2, 3]
t= 3044ms  node 4 is leader of term 2 but only 2/5 reachable: cannot commit
t= 3544ms  node 4 is leader of term 2 but only 2/5 reachable: cannot commit
t= 3600ms  --- heal partition; leaders now: [3, 4]
t= 3645ms  node 4 sees term 3 > 2, steps down
t= 4500ms  final: leader(s) [3], terms [3, 3, 3, 3, 3]
safety check passed: at most one leader per term {1: {2}, 2: {4}, 3: {3}}
```

Walk through it:

1. **Startup.** Node 2's timer fired first, it asked for votes in term 1, and two votes plus its own made a majority of 5.
2. **Leader crash.** Heartbeats stopped, node 4 timed out about 170 ms later and won term 2. Failover time is roughly one election timeout plus one round trip. When node 2 came back it saw term 2 heartbeats and became a follower without fuss.
3. **Partition.** Node 4 (the leader) ended up on the 2-node side. The 3-node side elected node 3 for term 3. **Node 4 still believes it is the leader of term 2.** For about a second there were two "leaders" at once (see the `leaders now: [3, 4]` line), but in *different terms*, and the old one cannot commit anything because it cannot reach a majority. That is the safety argument in action: two majorities of 5 must share a node, so two leaders can never both commit.
4. **Heal.** Node 4 got a message carrying term 3 and stepped down.

What the simulation leaves out, on purpose: the log. In full Raft a node also refuses its vote to a candidate whose log is less up-to-date than its own, which is what guarantees that a new leader holds every committed entry. The MIT 6.5840 labs (in Further reading) build the whole thing in Go.

### Leases and fencing tokens

The partition above shows a hazard that has nothing to do with Raft specifically: **a node can believe it is in charge after it has been replaced.** Any "only one worker may do X" scheme (one writer of the checkpoint, one job controller, one rank allowed to publish new weights to the inference fleet) has to deal with it.

A **lease** is a lock with an expiry time: "you are the leader until t = 10 s, unless you renew". If the holder dies, the lease simply runs out and someone else can take it, with no need to detect the death. The flaw is that the *holder* decides whether its lease is still valid by reading its own clock, and a process can be paused for an arbitrary time (a GC pause, a VM migration, a laptop lid, a `SIGSTOP`) between checking "is my lease still valid?" and acting on the answer.

The fix (described clearly in Martin Kleppmann's 2016 post "How to do distributed locking") is a **fencing token**: every time the lock service grants the lease it hands out a number that only goes up, and the *resource being protected* rejects any request carrying a token lower than one it has already seen. Raft's term number is exactly this kind of token.

```python
# lease.py -- a lease with a paused holder, with and without fencing tokens
# One "lock service" hands out a 10 s lease. Worker A gets it, then freezes for 15 s
# (GC pause, swapped out, VM migrated). The lease expires, B gets it. A wakes up still
# believing it holds the lease and writes. The storage decides who wins.

class LockService:
    def __init__(self, ttl):
        self.ttl, self.holder, self.expires, self.token = ttl, None, 0.0, 0
    def acquire(self, who, now):
        if self.holder is None or now >= self.expires:
            self.token += 1                     # monotonically increasing fencing token
            self.holder, self.expires = who, now + self.ttl
            return self.token
        return None

class Storage:
    def __init__(self, fencing):
        self.fencing, self.max_token, self.data = fencing, 0, []
    def write(self, who, token, value):
        if self.fencing and token < self.max_token:
            return f"REJECTED {who}'s write (token {token} < {self.max_token})"
        self.max_token = max(self.max_token, token)
        self.data.append((who, value))
        return f"accepted {who}'s write (token {token})"

for fencing in (False, True):
    lock, store = LockService(ttl=10.0), Storage(fencing)
    tok_a = lock.acquire("A", now=0.0)          # A holds the lease until t=10
    # ... A pauses from t=1 to t=16; its own clock says only "a moment" has passed
    tok_b = lock.acquire("B", now=11.0)         # lease expired, B acquires
    r1 = store.write("B", tok_b, "ckpt step 5000")
    r2 = store.write("A", tok_a, "ckpt step 4000")   # A wakes at t=16, writes stale data
    print(f"fencing={fencing}:  {r1};  {r2}  -> final {store.data[-1]}")
    assert (store.data[-1][0] == "B") == fencing

# The clock-skew caveat: a lease only works if holder and service agree on elapsed time.
ttl, drift_ppm, margin = 10.0, 100, 0.5
print(f"holder should stop using the lease {margin}s early; {drift_ppm} ppm drift over "
      f"{ttl}s is only {ttl * drift_ppm * 1e-6 * 1e3:.1f} ms, but a 15 s pause beats any margin")
```

```
$ python3 lease.py
fencing=False:  accepted B's write (token 2);  accepted A's write (token 1)  -> final ('A', 'ckpt step 4000')
fencing=True:  accepted B's write (token 2);  REJECTED A's write (token 1 < 2)  -> final ('B', 'ckpt step 5000')
holder should stop using the lease 0.5s early; 100 ppm drift over 10.0s is only 1.0 ms, but a 15 s pause beats any margin
```

Without fencing, the paused worker overwrote a newer checkpoint with an older one, and nothing reported an error. The last line is the clock-skew caveat. Leases assume holder and lock service roughly agree on how much time has passed. Clock *drift* is small enough to cover with a safety margin (stop using the lease a little before it expires). Process *pauses* are unbounded, so no margin is enough, and only the fencing check at the resource is safe. In ML infrastructure the "resource" is often an object store, which cannot check tokens. Then the usual substitute is to make writes idempotent and versioned (write to `step_5000/`, never to `latest/`), so a late writer cannot destroy newer data.

### Try this

1. In `raft_election.py`, make the timeouts fixed (`ELECT_MIN, ELECT_MAX = 150, 150`) and rerun. Every node times out at the same instant, votes for itself and refuses everyone else, and the script dies at `leader()[0]` because no one ever wins. (Over 20 seeds we counted 83 elections and 60 leaders with randomized timeouts, and 600 elections and 0 leaders with fixed ones.) This is why the timeouts are randomized.
2. Partition the cluster 2/2/1 instead of 2/3 (`partition` currently supports two sides; add a third set). Verify that *no* side elects a leader, and that terms keep climbing on every side. That is Raft choosing consistency over availability.
3. Change `MAJORITY` to `N // 2` (a "majority" of 2 out of 5) and add `partition = {0, 1}` before the first `run_until`, so the cluster starts out split. Both sides elect a leader *in term 1*, and the final `assert` catches it. (With the original partition the bug stays hidden, because the old leader never runs a new election. Broken quorum logic can pass a test that does not exercise it.)
4. In `store.py`, make rank 2 sleep 15 s before `add` while the store timeout is 10 s. Which ranks fail, and with what error?
5. In `gossip.py`, set `FANOUT = 1` and then `T_FAIL = 3`. Find the smallest `T_FAIL` with zero false suspicions over 40 rounds at each fanout.


## Consistency, replication and CAP

### Replication and consistency

**Replication** means keeping copies of the same data on several machines, for fault tolerance or to spread out read load. Once you have copies, you need a **consistency model**, a statement of what a reader may observe:

- **Strong consistency (linearizability):** the system behaves as if there were a single copy. Once a write completes, every later read sees it.
- **Eventual consistency:** if writes stop, replicas eventually converge. Until then, different readers may see different values.

Strong consistency costs coordination (consensus or similar) on every write. Eventual consistency is cheaper and stays available, but the application has to tolerate stale reads.

A common middle ground is **quorum replication** (as in Dynamo and Cassandra). With N replicas, a write waits for acknowledgements from W of them and a read asks R of them, keeping the newest version it sees. If `R + W > N`, every read set overlaps every write set in at least one replica, so a read is guaranteed to see the latest acknowledged write. The simulation below reads immediately after a write, before the remaining replicas have caught up:

```python
# quorum.py -- async replication (eventual) vs quorum reads/writes (R + W > N)
import random
random.seed(0)
N = 3

def trial(W, R):
    replicas = [(0, "v0")] * N                     # (version, value) on each replica
    version = 1
    order = random.sample(range(N), N)
    for k in order[:W]:                            # W replicas ack synchronously
        replicas[k] = (version, "v1")
    # order[W:] get the write "later" -- we read before that happens
    read_from = random.sample(range(N), R)         # read R replicas, keep the newest
    return max(replicas[k] for k in read_from)[1]

for W, R in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)]:
    stale = sum(trial(W, R) == "v0" for _ in range(10_000))
    tag = "R+W>N" if R + W > N else "R+W<=N"
    print(f"N={N} W={W} R={R} ({tag:6s}): stale reads right after write = {stale / 100:5.1f}%")
```

```
$ python3 quorum.py
N=3 W=1 R=1 (R+W<=N): stale reads right after write =  67.4%
N=3 W=2 R=1 (R+W<=N): stale reads right after write =  33.8%
N=3 W=1 R=2 (R+W<=N): stale reads right after write =  32.3%
N=3 W=2 R=2 (R+W>N ): stale reads right after write =   0.0%
N=3 W=3 R=1 (R+W>N ): stale reads right after write =   0.0%
```

The stale fractions match the counting argument. With W=1 and R=1, you read the one fresh replica only 1 time in 3. As soon as `R + W > N`, staleness drops to zero. The knobs trade read latency against write latency: W=3, R=1 makes writes wait for the slowest replica and reads fast, and W=1, R=3 does the reverse. (Real quorum systems have further corner cases, such as concurrent writes and "sloppy" quorums during failures, so `R + W > N` alone is not full linearizability.)

### CAP, briefly

The CAP theorem (conjectured by Brewer in 2000, proved by Gilbert and Lynch in 2002) says that when a **network partition** happens, some replicas cannot talk to others, and a replicated store must choose between:

- **Consistency:** refuse requests that might return stale or conflicting data, or
- **Availability:** answer every request, accepting that answers may be stale or conflicting.

Partitions are not optional in a real network, so the real choice is "C or A during a partition". etcd chooses C: a minority partition refuses writes. Many caches and some databases choose A.

The same 5-replica register under both policies, partitioned 2 | 3:

```python
# cap.py -- a 5-replica register during a partition: CP (majority quorum) vs AP (accept locally)
N, MAJ = 5, 3
side = {"left": [0, 1], "right": [2, 3, 4]}

def write(mode, reachable, value, store):
    if mode == "CP" and len(reachable) < MAJ:
        return "refused (no majority)"
    for r in reachable: store[r] = value
    return "ok"

for mode in ("CP", "AP"):
    store = ["x=0"] * N
    a = write(mode, side["left"],  "x=1", store)
    b = write(mode, side["right"], "x=2", store)
    print(f"{mode}: left write {a:22s} right write {b:4s} -> replicas {store}")
    print(f"    after heal: {'consistent' if len(set(store)) == 1 else 'CONFLICT: needs merge/last-writer-wins'}"
          if mode == "AP" else
          f"    after heal: minority replicas catch up from majority -> all x=2 (left clients saw errors)")
```

```
$ python3 cap.py
CP: left write refused (no majority)  right write ok   -> replicas ['x=0', 'x=0', 'x=2', 'x=2', 'x=2']
    after heal: minority replicas catch up from majority -> all x=2 (left clients saw errors)
AP: left write ok                     right write ok   -> replicas ['x=1', 'x=1', 'x=2', 'x=2', 'x=2']
    after heal: CONFLICT: needs merge/last-writer-wins
```

Neither outcome is free. The CP store gave the left-hand clients errors. The AP store gave everyone a success and then held two different values, which some policy has to reconcile: last-writer-wins (which silently drops one write and depends on clocks), a merge function, or vector clocks that surface the conflict to the application. The PACELC formulation (Abadi, 2012) adds the other half of the story: *Else*, when there is no partition, you still trade **L**atency against **C**onsistency, because waiting for more replicas is slower.

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

The checkpoint case is the one you will write yourself, so here it is in code. `os.replace` is atomic on a local POSIX filesystem: a reader sees either the old file or the new one, never a mix:

```python
# atomic_save.py -- write-temp-then-rename, so a crash never leaves a "complete-looking" file
import json, os, tempfile

def atomic_write(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")     # same filesystem as the target
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
        f.flush(); os.fsync(f.fileno())                   # bytes on disk before the rename
    os.replace(tmp, path)                                 # atomic on POSIX local filesystems
    dfd = os.open(d, os.O_RDONLY); os.fsync(dfd); os.close(dfd)   # persist the rename itself

def crashing_write(path, obj):
    with open(path, "w") as f:                            # truncates the old good file first!
        f.write(json.dumps(obj)[:20])                     # ...then "crash" mid-write

os.makedirs("ck", exist_ok=True)
atomic_write("ck/latest.json", {"step": 4000, "w": [0.1] * 5})
try:
    crashing_write("ck/naive.json", {"step": 5000, "w": [0.2] * 5})
    json.load(open("ck/naive.json"))
except json.JSONDecodeError as e:
    print("naive write after crash:", type(e).__name__, "-", e)
print("atomic file still loads:", json.load(open("ck/latest.json"))["step"])
```

```
$ python3 atomic_save.py
naive write after crash: JSONDecodeError - Expecting value: line 1 column 21 (char 20)
atomic file still loads: 4000
```

The naive version destroyed the good checkpoint *before* writing the new one: `open(path, "w")` truncates first. Caveats (as of 2026; check your storage): rename atomicity is a local-filesystem guarantee. NFS and parallel filesystems such as Lustre also perform a same-filesystem rename as one server-side operation, but client-side caching can make *other* nodes see the change late, so a reader on another node should not assume it sees the new file immediately. Object stores (S3, GCS) have no rename at all. There, "atomic" means uploading the data objects first and writing a small manifest or `COMPLETE` marker object last, and having readers trust only the marker. See [storage and data](./storage-and-data.md).

### Consistent hashing: sharding that survives membership changes

Many ML services must map keys to machines: which cache node holds this dataset shard, which inference replica holds the KV cache for this prompt prefix (prefix-aware routing), which parameter-server shard owns this embedding row. The obvious scheme, `node = hash(key) % num_nodes`, breaks badly when `num_nodes` changes, because almost every key moves. **Consistent hashing** (Karger et al., 1997) places nodes and keys on the same circle of hash values and assigns each key to the first node clockwise from it. Adding a node only takes keys from its clockwise neighbour. **Virtual nodes** (each physical node placed at many points on the ring) even out the load:

```python
# chash.py -- modulo hashing vs a consistent-hash ring with virtual nodes
import bisect, hashlib, statistics

def h(s):  # stable 64-bit hash (Python's hash() is randomized per process)
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

class Ring:
    def __init__(self, nodes, vnodes):
        self.points = sorted((h(f"{n}#{v}"), n) for n in nodes for v in range(vnodes))
        self.keys = [p for p, _ in self.points]
    def lookup(self, key):
        i = bisect.bisect(self.keys, h(key)) % len(self.keys)   # first point clockwise
        return self.points[i][1]

keys = [f"shard-{i:06d}" for i in range(100_000)]   # e.g. dataset shards or KV-cache prefixes
old_nodes = [f"node{i}" for i in range(10)]
new_nodes = old_nodes + ["node10"]

mod = lambda nodes: {k: nodes[h(k) % len(nodes)] for k in keys}
a, b = mod(old_nodes), mod(new_nodes)
print(f"modulo hashing:        {sum(a[k] != b[k] for k in keys) / len(keys):6.1%} of keys move when adding 1 of 11 nodes")

for v in (1, 10, 200):
    r_old, r_new = Ring(old_nodes, v), Ring(new_nodes, v)
    a = {k: r_old.lookup(k) for k in keys}
    b = {k: r_new.lookup(k) for k in keys}
    moved = sum(a[k] != b[k] for k in keys) / len(keys)
    load = [list(b.values()).count(n) for n in new_nodes]
    print(f"ring, {v:3d} vnodes/node: {moved:6.1%} move (ideal {1/11:.1%}); "
          f"max/mean load = {max(load) / statistics.mean(load):.2f}")
    assert all(a[k] == b[k] or b[k] == "node10" for k in keys)   # keys only move TO the new node
```

```
$ python3 chash.py
modulo hashing:         90.8% of keys move when adding 1 of 11 nodes
ring,   1 vnodes/node:  13.5% move (ideal 9.1%); max/mean load = 2.55
ring,  10 vnodes/node:  11.7% move (ideal 9.1%); max/mean load = 1.33
ring, 200 vnodes/node:   9.1% move (ideal 9.1%); max/mean load = 1.18
```

Modulo hashing moved 91% of the keys, which for a cache means a near-total miss storm right after scaling up. The ring moves only the keys the new node must take (1/11, about 9.1%), and the `assert` confirms that no key moves between two *old* nodes. With one point per node the busiest node carries 2.55x the average load, and with 200 virtual nodes that drops to 1.18x. Note the use of `hashlib` instead of Python's built-in `hash()`. `hash()` of a string is randomized per process (`PYTHONHASHSEED`), so two processes would disagree about where every key lives, which is a distributed bug in one line.

### Try this

1. In `quorum.py`, add N=5 and find all (W, R) pairs with zero stale reads. Which pair minimizes `max(W, R)`?
2. In `cap.py`, implement last-writer-wins on heal using a timestamp per write, then give the left side's clock a +5 s skew. Whose write survives, and is that the one that actually happened last?
3. In `chash.py`, *remove* a node instead of adding one. Check that only that node's keys move, and where they go.
4. Print `hash("shard-000001")` in two separate `python3 -c` invocations. Then do it again with `PYTHONHASHSEED=0` set for both.


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

### Try this

Build the ring all-reduce yourself and check the bandwidth claim. Each rank's vector is cut into n chunks. In the **reduce-scatter** phase, every rank passes one chunk to its right-hand neighbour, which adds it to its own copy. After n-1 steps each rank holds one chunk of the full sum. In the **all-gather** phase the finished chunks travel around the ring once more, n-1 steps, until everyone has all of them:

```python
# ring.py -- ring all-reduce = reduce-scatter + all-gather, simulated with numpy
import numpy as np

def ring_allreduce(xs):
    n = len(xs)
    chunks = [np.array_split(x.astype(np.float64), n) for x in xs]   # chunks[rank][c]
    sent = 0
    for s in range(n - 1):                       # reduce-scatter: n-1 steps
        msgs = [(r, (r - s) % n, chunks[r][(r - s) % n].copy()) for r in range(n)]
        for r, c, data in msgs:                  # rank r sends chunk c to rank r+1
            chunks[(r + 1) % n][c] += data; sent += data.size
    for s in range(n - 1):                       # all-gather: n-1 steps
        msgs = [(r, (r + 1 - s) % n, chunks[r][(r + 1 - s) % n].copy()) for r in range(n)]
        for r, c, data in msgs:
            chunks[(r + 1) % n][c] = data; sent += data.size
    return [np.concatenate(c) for c in chunks], sent / n   # per-rank elements sent

rng = np.random.default_rng(0)
for n in (2, 4, 8, 16):
    S = 1_000_000
    xs = [rng.standard_normal(S) for _ in range(n)]       # (S,) per-rank gradient
    out, per_rank = ring_allreduce(xs)
    assert all(np.allclose(o, sum(xs)) for o in out)
    print(f"n={n:2d}: per-rank sent {per_rank / S:.3f} x S   (2(n-1)/n = {2 * (n - 1) / n:.3f})")
```

```
$ python ring.py
n= 2: per-rank sent 1.000 x S   (2(n-1)/n = 1.000)
n= 4: per-rank sent 1.500 x S   (2(n-1)/n = 1.500)
n= 8: per-rank sent 1.750 x S   (2(n-1)/n = 1.750)
n=16: per-rank sent 1.875 x S   (2(n-1)/n = 1.875)
```

Per-rank traffic approaches 2S and never exceeds it, however many ranks there are. Latency is a different story: the ring takes `2(n-1)` sequential steps, which is why NCCL also has tree algorithms for small messages and large n (see [collective internals](../network/collectives-internals.md)).

1. Count steps instead of elements. At n = 1,024 and 5 microseconds per step, what is the latency floor of a ring all-reduce, independent of message size?
2. Turn `ring_allreduce` into a pure `reduce_scatter` (stop after the first loop) and confirm that rank r ends up holding the fully summed chunk `(r + 1) % n`. That is the output ZeRO/FSDP keeps.
3. Redo the parameter-server arithmetic with the parameters sharded over 8 server machines. How does per-step time compare with the ring now, and how many extra machines did it take?


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

### Try this

Put numbers on the memory column of the table for the 70B model on 64 GPUs (training state only, 16 bytes per parameter, activations ignored):

```python
# shard_mem.py -- per-GPU training-state memory (no activations) for a 70B model on 64 GPUs
N, G = 70e9, 64
GB = 1e9
plans = {
    "DP (full replica)":    (2 + 2 + 12) * N,             # weights, grads, fp32 master+m+v
    "ZeRO-1 (shard optim)": (2 + 2) * N + 12 * N / G,
    "ZeRO-2 (+grads)":      2 * N + (2 + 12) * N / G,
    "ZeRO-3 / FSDP (all)":  16 * N / G,
    "TP=8 x DP=8":          16 * N / 8,                   # each TP rank holds 1/8 of every layer
    "TP=8 x FSDP=8":        16 * N / 64,
}
for name, b in plans.items():
    print(f"{name:22s} {b / GB:8.1f} GB/GPU  {'fits' if b < 80 * GB else 'does not fit'} in 80 GB")
```

```
$ python3 shard_mem.py
DP (full replica)        1120.0 GB/GPU  does not fit in 80 GB
ZeRO-1 (shard optim)      293.1 GB/GPU  does not fit in 80 GB
ZeRO-2 (+grads)           155.3 GB/GPU  does not fit in 80 GB
ZeRO-3 / FSDP (all)        17.5 GB/GPU  fits in 80 GB
TP=8 x DP=8               140.0 GB/GPU  does not fit in 80 GB
TP=8 x FSDP=8              17.5 GB/GPU  fits in 80 GB
```

The two configurations that fit hold the same 17.5 GB of state but communicate very differently. Pure FSDP all-gathers each layer's weights over the inter-node network. TP keeps per-layer collectives on NVLink and lets FSDP span only 8 replicas.

1. Add activation memory for your sequence length using [the training arithmetic chapter](../training/arithmetic.md). Which plan stops fitting first as sequence length grows?
2. Add `PP=4 x TP=8 x DP=2` to the dictionary. What fraction of the layers does each GPU hold, and what does it cost (hint: the bubble)?
3. Run `dist_hello.py` (at the end of this chapter) with 8 samples on 3 ranks and explain why `match` fails. Then fix it by weighting each rank's loss by its shard size.


## Fault tolerance patterns

Given that failures are routine at scale, these are the standard ways to survive them. Most of them predate ML. They come from decades of HPC and web infrastructure.

### Checkpoint / restart

The workhorse. Periodically save everything needed to resume (model weights, optimizer state, LR scheduler state, data loader position, RNG states, step counter). On failure, kill everything and restart from the last checkpoint.

Why "kill everything" rather than repairing just the failed rank? Because the surviving ranks are often stuck partway through a collective, with communicators in an undefined state. Restarting from a known-consistent snapshot is simpler and far more reliable than trying to repair live state. Some newer systems do attempt in-place recovery, but checkpoint/restart is still the baseline everyone has.

Rules that keep checkpoints trustworthy:

- **Atomic completion:** write to a temporary location, then rename, or write a "complete" marker last. A job killed mid-save must never leave a checkpoint that *looks* complete.
- **Keep more than one:** the newest checkpoint may be corrupt, or may capture a state just before a loss spike you want to roll back past.
- **Verify resume once:** actually kill a job and resume it, and check that the loss curve continues smoothly. A checkpoint that has never been restored is a hope, not a backup.

The "verify resume" rule as a test you can run. The toy loop below checkpoints the weights, the step **and the RNG state** every 5 steps, crashes at step 13, and resumes. The resumed run must match an uninterrupted run *bit for bit*:

```python
# resume.py -- a training loop whose crash+resume is bit-identical to an uninterrupted run
import json, os, random, sys

def train(steps, ckpt="state.json", crash_at=None):
    if os.path.exists(ckpt):
        s = json.load(open(ckpt))
        w, step = s["w"], s["step"]
        rng = random.Random(); rng.setstate((s["rng"][0], tuple(s["rng"][1]), s["rng"][2]))
    else:
        w, step, rng = 0.0, 0, random.Random(1234)
    while step < steps:
        x = rng.gauss(0, 1)                         # "batch" drawn from a checkpointed RNG
        w -= 0.1 * 2 * (w - 3.0 - 0.1 * x)          # SGD on (w - target)^2
        step += 1
        if step % 5 == 0:                           # checkpoint: weights, step AND rng state
            tmp = ckpt + ".tmp"
            json.dump({"w": w, "step": step, "rng": rng.getstate()}, open(tmp, "w"))
            os.replace(tmp, ckpt)
        if step == crash_at:
            raise SystemExit(f"crashed at step {step}")
    return w

for f in ("a.json", "b.json"):
    if os.path.exists(f): os.remove(f)
ref = train(20, "a.json")
try:
    train(20, "b.json", crash_at=13)                # dies after the step-10 checkpoint
except SystemExit as e:
    print(e)
resumed = train(20, "b.json")                       # restarts from step 10, redoes 11..13
print(f"uninterrupted w = {ref!r}\nresumed       w = {resumed!r}\nbit-identical: {ref == resumed}")
assert ref == resumed
```

```
$ python3 resume.py
crashed at step 13
uninterrupted w = 2.995910981057316
resumed       w = 2.995910981057316
bit-identical: True
```

Delete `"rng": rng.getstate()` from the checkpoint (and reseed on resume) and the `assert` fails: the resumed run draws different "batches" for steps 11-20. That is the toy version of a resumed job that silently sees different data. Steps 11-13 were computed twice, and that is fine because they are a deterministic function of the checkpoint. The same test works on a real training loop: run N steps, then run N/2, kill, resume, and compare the final losses or a weight checksum.

See the [checkpoints chapter](../training/checkpoints/README.md).

### Idempotency

An operation is **idempotent** if doing it twice has the same effect as doing it once. In a distributed system with retries, you cannot always know whether an operation that "failed" actually happened. The network may have lost the reply, not the request. If the operation is idempotent, you can just retry it.

In ML jobs:

- "Save checkpoint for step 4000 to `ckpt/step_4000/`" is idempotent. Doing it twice overwrites with the same data.
- "Append this checkpoint to the list" or "increment the epoch counter" is not. A retry duplicates it.
- Data loading should be a deterministic function of (seed, step) so that resuming at step 4000 replays exactly the batches that would have come next. Otherwise a restart silently skips or repeats data.

When an operation is *not* naturally idempotent, make it so with an **idempotency key**: the client generates one unique id per logical operation, sends it with every retry, and the server remembers which keys it has already applied and replays the stored reply instead of redoing the work. Combined with retries this gives "at-least-once delivery, exactly-once effect". Below, an RL trainer reports finished rollouts to a counting service over a link that loses 20% of requests *and* 20% of replies:

```python
# idempotent.py -- at-least-once delivery over a lossy link, with and without idempotency keys
import random, uuid
random.seed(7)

class Server:
    def __init__(self, dedup):
        self.dedup, self.seen, self.samples_done = dedup, {}, 0
    def handle(self, key, n):
        if self.dedup and key in self.seen:
            return self.seen[key]              # replay the stored reply, don't redo the work
        self.samples_done += n                 # the side effect: "count these rollouts"
        self.seen[key] = "ok"
        return "ok"

def call_with_retry(server, n, attempts=10):
    key = str(uuid.uuid4())                    # ONE key per logical operation, reused on retry
    for _ in range(attempts):
        if random.random() < 0.2:              # request lost: server never saw it
            continue
        reply = server.handle(key, n)
        if random.random() < 0.2:              # reply lost: server DID the work, client can't tell
            continue
        return reply
    raise TimeoutError

for dedup in (False, True):
    s = Server(dedup)
    for _ in range(1000):
        call_with_retry(s, n=1)
    print(f"dedup={dedup!s:5s}: client submitted 1000 rollouts, server counted {s.samples_done}")
```

```
$ python3 idempotent.py
dedup=False: client submitted 1000 rollouts, server counted 1254
dedup=True : client submitted 1000 rollouts, server counted 1000
```

A lost reply is the dangerous case. The server did the work, the client cannot tell, and it retries. Without dedup, a quarter of the rollouts were counted twice. In an RL pipeline that is duplicated training data and a wrong sample count. Two details matter in real systems. The key must be generated *once per logical operation*, before the first attempt (a new key per retry defeats the purpose). And the server's "seen" table must be stored durably together with the side effect, in the same transaction or file write, or a server crash between the two reintroduces duplicates. See [RL infrastructure](../platform/rl-infrastructure.md) and [services and control planes](../platform/control-planes.md).

### Retries with exponential backoff and jitter

Transient failures (an object-store request failing with a 503, a DNS lookup timing out, the rendezvous store not up yet) should be retried. Retry naively and you can make things worse: 4,000 ranks retrying a struggling storage server every 100 ms is a self-inflicted denial-of-service attack.

The standard pattern:

```python
# retry.py
import random, time

def retry(fn, attempts=6, base=0.5, cap=30.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            # exponential backoff with "full jitter": spreads retries out in time
            delay = random.uniform(0, min(cap, base * 2**i))
            print(f"attempt {i} failed ({e}); sleeping {delay:.2f}s")
            time.sleep(delay)

# a flaky "object store": fails the first 3 calls with a 503
calls = 0
def flaky_get():
    global calls; calls += 1
    if calls <= 3:
        raise ConnectionError("503 Slow Down")
    return b"shard bytes"

random.seed(0)
print(retry(flaky_get, base=0.1), "after", calls, "calls")
```

```
$ python3 retry.py
attempt 0 failed (503 Slow Down); sleeping 0.08s
attempt 1 failed (503 Slow Down); sleeping 0.15s
attempt 2 failed (503 Slow Down); sleeping 0.17s
b'shard bytes' after 4 calls
```

The exponential part backs off from a struggling service. The jitter (randomness) stops all ranks from retrying in synchronized waves. Only retry operations that are idempotent.

How much the jitter matters: 1,000 clients all fail at t = 0 (the service went down), and each retries 6 times. Count the peak number of retries landing in any 100 ms window:

```python
# backoff.py -- 1000 clients hit a service that comes back at t=0; how bunched are the retries?
import random
random.seed(0)
C, BASE, CAP, ATTEMPTS = 1000, 1.0, 30.0, 6

def schedule(policy):
    times = []
    for _ in range(C):
        t = 0.0
        for i in range(ATTEMPTS):
            d = min(CAP, BASE * 2 ** i)
            t += {"fixed": BASE, "exponential": d, "full jitter": random.uniform(0, d)}[policy]
            times.append(t)
    return times

for policy in ("fixed", "exponential", "full jitter"):
    ts = schedule(policy)
    buckets = {}
    for t in ts:
        buckets[int(t / 0.1)] = buckets.get(int(t / 0.1), 0) + 1   # 100 ms buckets
    print(f"{policy:12s} peak load: {max(buckets.values()):5d} requests in one 100 ms window "
          f"(spread over {max(ts):5.1f} s)")
```

```
$ python3 backoff.py
fixed        peak load:  1000 requests in one 100 ms window (spread over   6.0 s)
exponential  peak load:  1000 requests in one 100 ms window (spread over  61.0 s)
full jitter  peak load:   162 requests in one 100 ms window (spread over  55.9 s)
```

Exponential backoff *without* jitter spreads the retries over a minute, but all 1,000 clients still arrive in the same instant every time. It delays the thundering herd without breaking it up. Full jitter cuts the peak about 6x. (The "full jitter" name and the comparison come from the AWS Architecture Blog post "Exponential Backoff And Jitter", 2015.) Two more rules from production: put a cap on the *total* retry time, so a permanently failed dependency surfaces as an error instead of an invisible stall, and log every retry, because a job that is quietly retrying is often the explanation for a slow step.

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

How many spares is "a pool"? Treat failures as random arrivals (a Poisson process) and ask how many nodes are typically out for repair at once:

```python
# spares.py -- how many hot spares does a 1,000-node job need?
import math
gpu_mtbf_days = 2112                   # per-GPU, from the Llama 3 numbers above
node_mtbf = gpu_mtbf_days / 8          # 8 GPUs per node, any GPU failure takes the node out
nodes, repair_days = 1000, 3.0         # a failed node is gone ~3 days (RMA, reboot, burn-in)

lam = nodes * repair_days / node_mtbf  # expected nodes broken at any moment (Poisson mean)
def poisson_cdf(k, lam):
    return sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k + 1))

print(f"node MTBF {node_mtbf:.0f} days; expected nodes in repair at once: {lam:.1f}")
for target in (0.9, 0.99, 0.999):
    k = next(k for k in range(100) if poisson_cdf(k, lam) >= target)
    print(f"spares for P(enough spares) >= {target}: {k:3d}  ({k / nodes:.1%} of the job)")
```

```
$ python3 spares.py
node MTBF 264 days; expected nodes in repair at once: 11.4
spares for P(enough spares) >= 0.9:  16  (1.6% of the job)
spares for P(enough spares) >= 0.99:  20  (2.0% of the job)
spares for P(enough spares) >= 0.999:  23  (2.3% of the job)
```

At the Llama 3 failure rate, a 1,000-node job averages about 11 nodes in the repair shop at any moment, and 20 spares covers it 99% of the time. The 5-10% guidance in the fault tolerance chapter is more conservative: it also covers correlated failures (a whole rack or switch), slow-but-alive nodes you choose to pull, and repair times longer than 3 days. The Poisson model assumes independent failures, and correlated ones are exactly what breaks it.

### Determinism and reproducibility across ranks

Determinism matters more in distributed training than on one GPU, for two reasons:

1. **Correctness of replicas.** Data-parallel ranks must make identical updates. Anything that should be identical across ranks (initialization, the dropout pattern *if* it affects shared state, the order in which parameters are bucketed for all-reduce) must be computed identically, or synchronized. DDP broadcasts rank 0's parameters at construction for exactly this reason.
2. **Debuggability of restarts.** If a loss spike shows up at step 51,234, you want to resume from the step-51,000 checkpoint and see the *same* spike. That needs the data order, RNG states and, ideally, the numerics to be reproducible.

What usually breaks it:

- **Floating-point non-associativity.** `(a + b) + c != a + (b + c)` in floating point. An all-reduce over 8 ranks and one over 16 ranks add in different orders, so results differ in the last bits. Change the world size and you change the numerics. NCCL's algorithm choice (ring vs tree) can also change the reduction order.
- **Non-deterministic kernels.** Some GPU kernels (for example atomic-add-based scatter operations) are non-deterministic run to run.
- **Data order** that depends on worker timing, on filesystem listing order, or on the world size.
- **Per-rank seeding mistakes.** You usually want the *same* seed for anything that must match across ranks (model init) and *different* seeds where ranks should differ (data augmentation, dropout on different data). Getting this backwards is a classic bug.

Floating-point non-associativity in action **[torch]**. The global batch and the per-example gradients are identical, only the number of data-parallel ranks changes:

```python
# nonassoc.py -- the same global-batch gradient, all-reduced over 8 vs 16 ranks, differs in bits
import torch
torch.manual_seed(0)
B, P = 64, 100_000
per_sample = torch.randn(B, P, dtype=torch.float32)        # (B, P) per-example gradients

def dp_grad(world):
    local = per_sample.view(world, B // world, P).sum(1)     # (world, P) each rank's local sum
    total = torch.zeros(P)
    for r in range(world):                                   # ring order: rank 0, 1, 2, ...
        total = total + local[r]                             # (P,)
    return total / B

g8, g16 = dp_grad(8), dp_grad(16)
diff = (g8 != g16).sum().item()
print(f"elements that differ between world=8 and world=16: {diff} of {P} ({diff / P:.0%})")
print(f"max abs difference: {(g8 - g16).abs().max().item():.2e}   (values ~ {g8.abs().mean().item():.2e})")
print("same world size, same order -> bit-identical:", torch.equal(dp_grad(8), g8))
```

```
$ python nonassoc.py
elements that differ between world=8 and world=16: 70378 of 100000 (70%)
max abs difference: 1.19e-07   (values ~ 9.91e-02)
same world size, same order -> bit-identical: True
```

70% of the gradient entries differ in their last bits, by about one float32 ulp (unit in the last place). That is harmless for convergence, but it means a run resumed on a different world size will *not* reproduce the original loss curve bit for bit, and it slowly diverges after that. The seeding rule as code:

```python
# seeding.py -- same seed where ranks must agree, rank-offset seed where they must differ
import torch

def setup(rank, base_seed=1234):
    torch.manual_seed(base_seed)                     # identical init on every rank
    w = torch.randn(3)                               # (3,) "model weights"
    data_gen = torch.Generator().manual_seed(base_seed + 1 + rank)   # per-rank augmentation
    aug = torch.rand(2, generator=data_gen)          # (2,) e.g. crop offsets
    return w, aug

(w0, a0), (w1, a1) = setup(0), setup(1)
print("weights equal across ranks:", torch.equal(w0, w1))
print("augmentation differs across ranks:", not torch.equal(a0, a1))
print("rank 1 reproducible on restart:", torch.equal(setup(1)[1], a1))
```

```
$ python seeding.py
weights equal across ranks: True
augmentation differs across ranks: True
rank 1 reproducible on restart: True
```

Bitwise reproducibility at scale is expensive and often not achievable. A realistic target is: reproducible data order, reproducible RNG streams, and loss curves that match closely when resumed on the same world size. See [reproducibility](../training/reproducibility/README.md).

### When peers are untrusted

Everything above assumes ranks fail by *stopping* or *slowing down*. These are **crash faults**. In decentralized training, where volunteers or untrusted providers on the internet contribute compute, a peer can also send *wrong* data, because of broken hardware, a bug, or on purpose. A peer that can behave arbitrarily is said to have a **Byzantine fault** (after Lamport, Shostak and Pease's 1982 "Byzantine Generals" paper). Byzantine-tolerant consensus needs `3f + 1` nodes to tolerate `f` bad ones, against `2f + 1` for crash faults. Training systems usually do not run consensus. Instead they defend the aggregation step. A plain mean is destroyed by a single bad gradient. Robust aggregators such as the coordinate-wise median or the trimmed mean bound the damage **[torch]**:

```python
# robust.py -- averaging gradients from untrusted peers: mean vs coordinate-wise median vs trimmed mean
import torch
torch.manual_seed(0)
true_g = torch.randn(1000)                                   # (P,) the gradient honest peers estimate
honest = true_g + 0.5 * torch.randn(15, 1000)                # (15, P) noisy honest contributions
bad = -100 * true_g.expand(1, -1)                            # (1, P) one peer sends a scaled, flipped gradient
peers = torch.cat([honest, bad])                             # (16, P)

def trimmed_mean(x, k):                                      # drop k largest and k smallest per coordinate
    s, _ = x.sort(dim=0)
    return s[k:-k].mean(0)

cos = lambda a: torch.nn.functional.cosine_similarity(a, true_g, dim=0).item()
for name, agg in [("mean", peers.mean(0)), ("median", peers.median(0).values),
                  ("trimmed mean k=2", trimmed_mean(peers, 2)), ("honest-only mean", honest.mean(0))]:
    print(f"{name:17s} cosine to true gradient = {cos(agg):+.3f}")
```

```
$ python robust.py
mean              cosine to true gradient = -1.000
median            cosine to true gradient = +0.986
trimmed mean k=2  cosine to true gradient = +0.991
honest-only mean  cosine to true gradient = +0.992
```

One peer out of 16 sending `-100 x` the gradient turned the averaged update into exact gradient *ascent*. The robust aggregators stay close to the honest answer. They cost something even with no attacker, since they throw information away and tolerate fewer bad peers than they have honest ones. They also cannot be computed with a plain all-reduce, because a median is not a sum. Real systems add verification (recomputing a sample of a peer's work, comparing against a trusted replica) and reputation on top. [Decentralized training](../training/decentralized-training.md) covers the systems built on these ideas.

### Try this

1. In `resume.py`, crash at step 10, exactly on a checkpoint boundary, and then at step 11. Check that both still resume bit-identically. Then crash *inside* `json.dump` (raise between the two lines that write and rename) and confirm the `.tmp` file is ignored.
2. In `idempotent.py`, generate a fresh `uuid4()` inside the retry loop instead of before it. What does the server count now?
3. In `backoff.py`, implement "decorrelated jitter" (`d = min(CAP, random.uniform(BASE, prev * 3))`) and compare its peak with full jitter.
4. In `robust.py`, increase the number of bad peers from 1 to 7, then to 8 of 16. At what point does the median break, and why exactly there?
5. In `spares.py`, set `repair_days = 14`. How many spares do you need now for 99%?


## The distributed debugging mindset

Debugging one process means reading one stack trace. Debugging 1,024 processes means finding *which* of them is wrong, and the evidence is spread across 128 machines.

### Per-rank logs and rank-prefixed printing

When N processes print to one terminal or one file, their lines interleave. Lines from different ranks get mixed up, and sometimes two ranks' output ends up on the *same* line. (This happened in testing this chapter's example. See the expected output below.) Without a rank prefix you cannot tell which process said what.

Minimum hygiene:

- **Prefix every line with the rank**, for example `[rank17]`. Launchers can do it for you: `torchrun --tee 3 --log-dir logs/` writes per-rank log files on Linux, and `srun --label` prefixes each line with the task id.
- **Print one whole line per call** and flush (`print(..., flush=True)`). If lines still collide, use a file lock around prints. [`printflock.py`](../training/tools/printflock.py) and [`multi-gpu-non-interleaved-print.py`](../training/tools/multi-gpu-non-interleaved-print.py) in `training/tools` show how.
- **Log once for global facts** (`if rank == 0`), but **log on every rank** when hunting a rank-specific problem. The bad rank is never rank 0 when you need it to be.
- **Log progress counters** (step, micro-batch, collective sequence number), so that you can order events without trusting cross-node clocks.

A minimal version with only the standard library: every line carries rank, host and PID, and every rank also writes its own file, so you can `grep` one rank's history without untangling the shared stream:

```python
# ranklog.py -- rank-prefixed logging to the terminal AND one file per rank
import logging, multiprocessing as mp, os, socket

def get_logger(rank, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(f"[rank{rank:02d} {socket.gethostname().split('.')[0]} pid=%(process)d] "
                            "%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log = logging.getLogger(f"rank{rank}")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(f"{log_dir}/rank{rank:02d}.log", "w")):
        h.setFormatter(fmt); log.addHandler(h)
    return log

def worker(rank):
    log = get_logger(rank)
    for step in range(2):
        log.info(f"step={step} loss={1.0 / (step + rank + 1):.3f}")   # log progress counters

if __name__ == "__main__":
    mp.set_start_method("spawn")
    ps = [mp.Process(target=worker, args=(r,)) for r in range(3)]
    for p in ps: p.start()
    for p in ps: p.join()
```

```
$ python3 ranklog.py
[rank00 Ibrahims-MacBook-Pro pid=49132] 16:18:05 INFO step=0 loss=1.000
[rank00 Ibrahims-MacBook-Pro pid=49132] 16:18:05 INFO step=1 loss=0.500
[rank01 Ibrahims-MacBook-Pro pid=49133] 16:18:05 INFO step=0 loss=0.500
[rank01 Ibrahims-MacBook-Pro pid=49133] 16:18:05 INFO step=1 loss=0.333
[rank02 Ibrahims-MacBook-Pro pid=49134] 16:18:05 INFO step=0 loss=0.333
[rank02 Ibrahims-MacBook-Pro pid=49134] 16:18:05 INFO step=1 loss=0.250
$ ls logs; cat logs/rank01.log
rank00.log
rank01.log
rank02.log
[rank01 Ibrahims-MacBook-Pro pid=49133] 16:18:05 INFO step=0 loss=0.500
[rank01 Ibrahims-MacBook-Pro pid=49133] 16:18:05 INFO step=1 loss=0.333
```

In a real job, take the rank from `os.environ["RANK"]`. The hostname is what lets you map "rank 417 is slow" to "node gpu-st-p5-0052 is bad", and the PID is what you pass to `py-spy`. Zero-padding the rank (`rank01`) makes the log files sort correctly.

### Hangs vs crashes

| | Crash | Hang |
|---|---|---|
| What you see | a traceback on some rank, nonzero exit code | nothing. GPU utilization flat (or stuck at 100% spinning in a collective), logs stop |
| Where to look | the *first* rank that failed. Other ranks often die afterwards with secondary "connection reset" or "NCCL error" messages, which are symptoms | the rank that is *not* where the others are |
| Typical causes | OOM, NaN assertion, code bug, hardware Xid error | mismatched collectives (SPMD violation), a dead peer before a timeout fires, deadlock in data loading, network misconfiguration |

The hard case is the hang. 1,023 ranks are waiting in an all-reduce, and one rank is somewhere else: stuck reading a file, in a different collective, or in an `if rank == ...` branch that skipped a collective. Finding that one rank is the whole job.

When reading crash logs from many ranks, **find the earliest real error.** When one rank dies, every rank talking to it reports a communication error a moment later. Hundreds of tracebacks that say "connection closed by peer" are symptoms. The single one that says "CUDA out of memory" is the cause.

Since a crash beats a hang, build hang-to-crash conversion into your own code, not just into NCCL. Python's built-in `faulthandler` can dump every thread's stack after a timeout and exit, and can dump stacks on a signal without stopping the process:

```python
# watchdog.py -- turn a silent hang into a crash with a stack trace (what NCCL's watchdog does)
import faulthandler, signal, sys, threading

faulthandler.register(signal.SIGUSR1)           # `kill -USR1 <pid>` dumps all stacks, keeps running

def fake_all_reduce():
    threading.Event().wait()                    # waits for a peer that never arrives

def train_step(step):
    # re-arm the watchdog each step: if this step takes > 2 s, dump stacks and exit(1)
    faulthandler.dump_traceback_later(2.0, exit=True)
    if step == 3:
        fake_all_reduce()
    faulthandler.cancel_dump_traceback_later()

for step in range(5):
    print("step", step, flush=True)
    train_step(step)
```

```
$ python3 watchdog.py; echo "exit code: $?"
step 0
step 1
step 2
step 3
Timeout (0:00:02)!
Thread 0x00000001f7942240 (most recent call first):
  File ".../lib/python3.14/threading.py", line 369 in wait
  File ".../lib/python3.14/threading.py", line 670 in wait
  File "/private/tmp/ds/watchdog.py", line 7 in fake_all_reduce
  File "/private/tmp/ds/watchdog.py", line 13 in train_step
  File "/private/tmp/ds/watchdog.py", line 18 in <module>
exit code: 1
```

The silent hang became a nonzero exit code (which a launcher can restart) plus a stack trace that names the function it hung in. Set the timeout to several times your slowest normal step, including checkpoint steps, or you have built the "timeout too short" failure from the heartbeat section. The `SIGUSR1` handler gives you a py-spy-like dump on demand, with no extra tools: `kill -USR1 <pid>` on every rank, then compare stacks. (`faulthandler` prints Python frames only. To see where a rank is stuck *inside* C++ or CUDA code, use `py-spy dump --native`.)

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

### Try this

1. Make a 4-process hang and find the odd rank out. Run `dist_hello.py` (next section) with the `if rank != 1:` change from its first exercise, then in another terminal run `pgrep -f dist_hello.py | xargs -I {} py-spy dump --pid {}` (`pip install py-spy`; on macOS it needs `sudo`). Three stacks should end in `all_reduce` and one somewhere else.
2. Add `faulthandler.dump_traceback_later(30, exit=True)` around the training step of `dist_hello.py`, re-arming it every step, and repeat exercise 1. Now the hang turns into a crash after 30 s, with the stack in the log.
3. Take `ranklog.py`, add `time.sleep(random.random())` per rank, and write a 10-line script that reads all `logs/rank*.log` files and prints, for each step, which rank logged it last. That is a straggler report.
4. Run `TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ... dist_hello.py` with mismatched tensor shapes on one rank (for example `torch.tensor([1.0, 2.0])` on rank 0 only). Compare the error with and without the debug variable.


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

### Try this

Each of these teaches one of this chapter's lessons:

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
- Colin Fidge, *Timestamps in Message-Passing Systems That Preserve the Partial Ordering* (1988), and Friedemann Mattern, *Virtual Time and Global States of Distributed Systems* (1988). The two independent introductions of vector clocks.
- Naohiro Hayashibara et al., *The phi Accrual Failure Detector* (SRDS 2004).
- Abhinandan Das, Indranil Gupta and Ashish Motivala, *SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol* (DSN 2002), and Robbert van Renesse, Yaron Minsky and Mark Hayden, *A Gossip-Style Failure Detection Service* (Middleware 1998).
- David Karger et al., *Consistent Hashing and Random Trees* (STOC 1997), and DeCandia et al., *Dynamo: Amazon's Highly Available Key-value Store* (SOSP 2007), for consistent hashing, quorums and vector clocks in one production system.
- Martin Kleppmann, *How to do distributed locking* (2016), https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html. Leases, process pauses and fencing tokens.
- Marc Brooker, *Exponential Backoff And Jitter* (AWS Architecture Blog, 2015), https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/.
- Leslie Lamport, Robert Shostak and Marshall Pease, *The Byzantine Generals Problem* (ACM TOPLAS 1982), and Blanchard et al., *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (NeurIPS 2017), for robust aggregation.
