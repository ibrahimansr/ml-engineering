# Orchestration Internals for ML Workloads

## Prerequisites

This chapter assumes no systems background beyond what these foundation chapters cover. Read them first:

- [Computer architecture](../foundations/computer-architecture.md): explains PCIe, NUMA, HBM and NVLink, which come up in every placement decision below.
- [Linux and the OS](../foundations/linux-and-os.md): explains processes, signals, cgroups, tmpfs and the OOM killer, which the scheduler and container sections rely on.
- [Concurrency](../foundations/concurrency.md): explains threads, background workers and blocking versus asynchronous work, which you need for async checkpointing and launcher monitor loops.
- [Networking](../foundations/networking.md): explains NICs, switches, fat trees, InfiniBand/RoCE and RDMA, which you need for the topology and pod-networking sections.
- [Distributed systems](../foundations/distributed-systems.md): explains failure detection, heartbeats, rendezvous, key-value stores and MTBF reasoning, which the launcher and fault-tolerance sections build on.
- [Containers and virtualization](../foundations/containers-and-virtualization.md): explains images, namespaces and container runtimes, which every Kubernetes section assumes.
- [Kubernetes and Helm from zero](kubernetes-from-zero.md): explains pods, Jobs, Services, controllers and `kubectl`. This chapter does not re-teach Kubernetes basics; it starts where that chapter stops.

This chapter covers what happens inside the scheduler, the launcher and the health-check loop when you run a multi-node ML job, and why those internals decide how much of your GPU time turns into training steps or served tokens. The day-to-day SLURM commands are covered in [SLURM for users](slurm/users.md) and [SLURM administration](slurm/admin.md), and the recovery recipes (job arrays, signals, kill switches) are in [Fault Tolerance](../training/fault-tolerance/README.md). None of that is repeated here.

A warning about how current this is: the Kubernetes side of this chapter describes projects that ship a new minor release every few weeks, such as Kueue, JobSet, LeaderWorkerSet, KAI Scheduler, Slinky, the NVIDIA DRA driver and the GPU Operator. Where a claim depends on a specific version it is marked as such. Check the upstream docs before you build on any API field named here.

Conventions for the code in this chapter:

- Blocks marked **[ran]** were executed on an Apple M4 laptop (macOS, 10 cores) with Python 3.14 and PyTorch 2.15 on CPU, using the `gloo` backend (PyTorch's CPU collective library). The output shown is the real output, trimmed where noted. Timings are one run on one laptop, not benchmarks.
- Blocks marked **untested** need Slurm, Kubernetes, GPUs or Linux and were not executed. They use only flags and API fields I'm confident exist, but treat the "expected output" as a sketch of the shape, not a capture.
- The YAML manifests were checked for syntax with a YAML parser, which says nothing about whether a cluster accepts them.
- On macOS, `torch.distributed` with gloo needed `GLOO_SOCKET_IFNAME=lo0` (bind to the loopback interface) and `--local-addr=127.0.0.1`. On Linux neither is normally needed.


## What an ML scheduler has to solve

A **scheduler** is the program that decides which job runs on which machines and when. A web-service scheduler packs many small, independent, restartable units onto machines. A distributed training job is the opposite: a single process group of N ranks (a rank is one worker process, usually one per GPU, and the process group is the set of ranks that communicate with each other) that is useless until all N are running, that runs at the speed of its slowest rank, and whose collectives (communication operations in which every rank takes part, such as summing gradients across all GPUs) slow down sharply when ranks are split across the wrong switches. This leads to five requirements, and each one conflicts with at least one of the others.

| Requirement | What it means | What it conflicts with |
|---|---|---|
| Gang scheduling | All N workers start together or none start | Utilization: you have to hold idle nodes while you collect a gang |
| Topology-aware placement | Ranks go on the same leaf switch, rail group, or NVLink domain | Utilization: fragmentation, since strict placement leaves holes |
| Fairness | Every team or user gets its entitled share over time | Utilization and throughput for large jobs |
| Preemption | High-priority work can evict low-priority work | Lost work for the job that gets evicted (checkpoint cost) |
| Backfill | Small jobs fill the holes in front of a reserved large job | Accuracy of runtime estimates, and the large job's start time if those estimates are wrong |

**Utilization** here means the fraction of GPU-hours in the cluster that are allocated to some job rather than sitting idle. It is what the people paying for the cluster look at first, and it is why every one of these requirements is a trade-off rather than a rule.

### Why gang scheduling cannot be skipped

**Gang scheduling** means placing all the workers of a job at once, or none of them. Take a cluster of 96 free nodes and two queued jobs that each need 64 nodes. A scheduler that places pods one at a time, as the default Kubernetes scheduler does, can give 48 nodes to each job. Both jobs then sit at `torch.distributed` rendezvous until they time out, and 96 nodes do nothing. A gang scheduler sees that only one of the two jobs fits, admits that one, and leaves the other queued. The cost is that 32 nodes stay idle unless something smaller can be backfilled into them.

The whole argument fits in a few lines of Python **[ran]**:

```python
# gang.py -- pod-at-a-time vs all-or-nothing on 96 free nodes, two 64-node jobs
from itertools import zip_longest

FREE = 96
jobs = {"A": 64, "B": 64}

# 1) pod-at-a-time (default kube-scheduler): pods from both jobs arrive interleaved
placed, free = {"A": 0, "B": 0}, FREE
for pods in zip_longest(*[[j] * n for j, n in jobs.items()]):
    for j in pods:
        if j and free > 0:
            placed[j] += 1; free -= 1
running = [j for j in jobs if placed[j] == jobs[j]]
print(f"pod-at-a-time: placed={placed} free={free} jobs able to start={running}")

# 2) gang: admit a job only if ALL of its pods fit right now
placed, free = {}, FREE
for j, n in jobs.items():
    if n <= free:
        placed[j] = n; free -= n
print(f"gang:          placed={placed} free={free} (left for backfill)")
```

```
pod-at-a-time: placed={'A': 48, 'B': 48} free=0 jobs able to start=[]
gang:          placed={'A': 64} free=32 (left for backfill)
```

Why does this deadlock rather than just run slowly? Because a training rank that has been placed does not give its node back. It blocks in `init_process_group`, waiting for peers that will never arrive, while holding the GPUs those peers need. That is the classic **hold-and-wait** condition for deadlock (see [concurrency](../foundations/concurrency.md#deadlock)), just with nodes instead of locks.

### Why topology matters

An NVLink domain is a set of GPUs connected by NVLink, NVIDIA's direct GPU-to-GPU link, which is much faster than going through the network. An all-reduce is the collective that sums a tensor across all ranks and gives every rank the result. Inside an NVLink domain, an all-reduce runs at NVLink bandwidth. Across nodes it runs at NIC (network interface card) bandwidth, which is roughly an order of magnitude lower per GPU (see [networking](../network/README.md)). In a typical cluster each node plugs into a leaf switch, and leaf switches are connected to each other through higher-level spine switches. Across spine switches you also share the fabric with other jobs and are exposed to oversubscription (the spine has less total bandwidth than all the leaves below it, so not every node can use full bandwidth at the same time). Placement matters most for:

- tensor and expert parallelism (splitting individual layers, or MoE experts, across GPUs, which makes those GPUs exchange data on every layer), which should stay inside the NVLink domain
- pipeline stages and data-parallel groups, which should stay under as few leaf or spine switches as possible
- on rail-optimized fabrics (networks where each GPU in a node has its own NIC and the same-numbered NICs of all nodes share a switch), where GPU `i` on every node connects to leaf `i`, keeping the job within one scalable unit (a fixed block of nodes cabled to the same set of leaf switches) so that same-rail traffic stays one hop

The cost of placement is **fragmentation**: free GPUs scattered in pieces that are too small or in the wrong place for the jobs waiting in the queue. A GB200 NVL72 rack has 18 compute trays (a tray is one server board) with 4 GPUs each. If you place a 16-tray job (64 GPUs) inside one rack, 2 trays (8 GPUs) are left over, and only jobs that need 2 trays or fewer can use them. That is `2/18 = 11%` of the rack stranded unless backfill finds small jobs to fill it. The stranded fraction depends entirely on whether the job size divides the domain size **[ran]**:

```python
# frag.py -- NVL72 rack = 18 trays x 4 GPUs; what a topology-pinned job strands
TRAYS, GPUS_PER_TRAY = 18, 4
for job_trays in (16, 12, 9, 8):
    per_rack = TRAYS // job_trays                     # jobs that fit in one rack
    left = TRAYS - per_rack * job_trays
    print(f"{job_trays:>2}-tray job ({job_trays*GPUS_PER_TRAY:>2} GPUs): {per_rack} per rack, "
          f"{left} trays stranded = {left/TRAYS:.0%} unless backfilled")
```

```
16-tray job (64 GPUs): 1 per rack, 2 trays stranded = 11% unless backfilled
12-tray job (48 GPUs): 1 per rack, 6 trays stranded = 33% unless backfilled
 9-tray job (36 GPUs): 2 per rack, 0 trays stranded = 0% unless backfilled
 8-tray job (32 GPUs): 2 per rack, 2 trays stranded = 11% unless backfilled
```

This is why teams on NVL72 hardware pick parallelism layouts whose tensor/expert-parallel group sizes divide 72 GPUs (or 18 trays), and why sites keep a queue of small eval and debug jobs to soak up the leftovers.

### Fairness and utilization pull in opposite directions

**Fair share** means each team gets, averaged over time, the fraction of the cluster it is entitled to. Strict fair share leaves nodes idle when the team that is entitled to them has nothing queued. Pure utilization lets one team with a deep queue take the whole cluster. Production schedulers resolve this with quotas that can be borrowed, so idle capacity can be used by anyone, combined with reclaim through preemption (stopping a running job to give its resources to another) when the owner comes back. Borrowing is only safe if a borrowed job can be preempted cheaply, and for training that means "can be preempted with a recent checkpoint" (a checkpoint is a saved copy of model and optimizer state that training can resume from). So the checkpoint interval (see [below](#optimal-checkpoint-interval-youngdaly)) is a scheduling parameter as well as a training parameter.

A toy model of two teams with 50 GPUs of quota each, on a 100-GPU cluster **[ran]**:

```python
# borrow.py -- strict quotas vs borrowable quotas, 100 GPUs split 50/50 between two teams
demand = [(80, 10), (80, 60), (30, 70)]           # (team A wants, team B wants) per hour
QUOTA = {"A": 50, "B": 50}
for a, b in demand:
    strict = min(a, 50) + min(b, 50)
    # borrowing: each team first gets min(demand, quota); spare quota is lent to the other
    ga, gb = min(a, 50), min(b, 50)
    spare = 100 - ga - gb
    ga += min(a - ga, spare); spare = 100 - ga - gb
    gb += min(b - gb, spare)
    print(f"demand A={a:>2} B={b:>2} | strict uses {strict:>3}/100 | borrowing: A={ga} B={gb} "
          f"uses {ga+gb}/100, borrowed A={max(ga-50,0)} B={max(gb-50,0)}")
```

```
demand A=80 B=10 | strict uses  60/100 | borrowing: A=80 B=10 uses 90/100, borrowed A=30 B=0
demand A=80 B=60 | strict uses 100/100 | borrowing: A=50 B=50 uses 100/100, borrowed A=0 B=0
demand A=30 B=70 | strict uses  80/100 | borrowing: A=30 B=70 uses 100/100, borrowed A=0 B=20
```

Going from the first hour to the second, B's demand rises from 10 to 60, and the 30 GPUs that A borrowed have to come back. The toy hides the part that costs money: the scheduler has to preempt 30 GPUs' worth of A's jobs, and whatever A computed since its last checkpoint is lost. Kueue's `borrowingLimit`/`lendingLimit` and Slurm's QoS preemption (both below) are the real versions of this.


## How SLURM implements each of these

[Slurm](https://slurm.schedmd.com/) is the batch scheduler most HPC and many ML clusters run. Its central controller daemon is `slurmctld`; every compute node runs `slurmd`, which starts and tracks job processes; and `slurmdbd` stores accounting data. A **job** is a resource allocation, and a **job step** is a set of processes launched inside it with `srun`. The [users chapter](slurm/users.md) covers the commands; this section covers what the controller does with your request.

### The life of a multi-node training job

Before looking at the plugins one by one, here is a complete job script for multi-node `torchrun` training, so you can see where each internal mechanism shows up. The [launchers directory](slurm/launchers/) has more variants (accelerate, Lightning, plain `srun`). This one adds requeue-on-signal, which the fault-tolerance sections below rely on. **Untested** (no Slurm cluster available while writing):

```bash
#!/bin/bash
#SBATCH --job-name=pretrain-8b-ctx8k     # name runs after the hypothesis, not the date
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1              # ONE task per node: torchrun, which spawns 8 workers
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive                      # whole nodes; no other job shares them
#SBATCH --time=24:00:00                  # an accurate limit improves your backfill chances
#SBATCH --switches=1@01:00:00            # prefer one leaf switch, accept any after 1 h
#SBATCH --requeue                        # allow this job to be put back in the queue
#SBATCH --signal=B:USR1@300              # SIGUSR1 to the batch shell 300 s before the time limit
#SBATCH --output=%x-%j.out

set -euo pipefail

# The first host in the allocation runs the rendezvous store (see "c10d rendezvous").
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_PORT=29400

# On SIGUSR1: ask Slurm to requeue us (same job ID), so the next attempt resumes
# from the latest checkpoint. train.py must save on SIGTERM; see Fault Tolerance.
trap 'echo "$(date) got USR1, requeueing"; scontrol requeue "$SLURM_JOB_ID"' USR1

# srun starts one torchrun per node. Single quotes: $SLURM_* expand on each node.
srun --kill-on-bad-exit=1 --wait=60 bash -c '
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$RDZV_PORT" \
    --rdzv-id="$SLURM_JOB_ID" \
    --max-restarts=0 \
    train.py --resume=latest
' &
wait   # srun runs in the background so bash can run the trap when USR1 arrives
```

Two details here are easy to get wrong. First, `srun` must run in the background followed by `wait`, because bash only runs a trap handler between foreground commands, so a foreground `srun` would delay the handler until training ends. Second, `--max-restarts=0` is deliberate: on a big job most failures are hardware, and in-place restarts on the same broken node just fail again (explained [below](#what-happens-on-a-worker-failure)), so recovery is left to the scheduler.

What you would expect to see, sketched rather than captured (**untested**; exact columns depend on your `squeue`/`sacct` format settings):

```
$ sbatch train.slurm
Submitted batch job 81234

$ squeue -u $USER
  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  81234     train pretrain    alice PD       0:00      4 (Priority)

# ... later, once backfill or the main scheduler has started it:
  81234     train pretrain    alice  R    3:02:11      4 gpu[017-020]

$ sacct -j 81234 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
JobID           JobName      State ExitCode    Elapsed        NodeList
------------ ---------- ---------- -------- ---------- ---------------
81234          pretrain    RUNNING      0:0   03:02:11    gpu[017-020]
81234.batch       batch    RUNNING      0:0   03:02:11          gpu017
81234.0            bash    RUNNING      0:0   03:02:09    gpu[017-020]
```

The `(REASON)` column is the scheduler talking to you: `Priority` means higher-priority jobs are ahead of you, `Resources` means you're next but the nodes aren't free yet, and `ReqNodeNotAvail` or `PartitionTimeLimit` mean something in your request can't be satisfied. The `.batch` step is the shell running your script and `.0` is the first `srun`.

### Node selection: `select/cons_tres`

`SelectType=select/cons_tres` is the plugin that decides which nodes and which resources within a node (CPU cores, memory, and GRES, Slurm's name for generic resources such as GPUs) a job gets. It is the current consumable-resources plugin ("consumable" means each core, byte of memory or GPU can be handed to one job at a time and is subtracted from what is free), and the older `select/cons_res` has been removed in recent releases. I'm fairly but not fully sure that the removal happened in 23.11, so check your version's release notes.

Here is how it works internally, simplified:

1. The scheduler builds a bitmap (one bit per node, set if the node is eligible) of candidate nodes from the job's constraints: partition, features, `--exclude`, reservations, and node state.
2. For each node it computes how many of the job's tasks fit, given the free cores, memory and GRES. GPU GRES is tracked per device file, and with `AutoDetect=nvml` in `gres.conf` each GPU also gets a list of the cores it is close to. On a multi-socket server, each CPU socket with its directly attached memory and PCIe devices forms a NUMA node (non-uniform memory access), and accessing devices attached to the other socket is slower.
3. It then selects nodes. Without a topology plugin it walks the nodes in order and does best-fit. With a topology plugin, the topology plugin's selection routine decides which subset of nodes to evaluate (see the next sections).
4. Within each node it allocates specific cores and GPUs. `--gres-flags=enforce-binding` makes it a hard requirement that the job's CPUs are the ones close to its GPUs. Without the flag, closeness is only a preference.

The configuration that feeds all of this is a few lines in `slurm.conf` and `gres.conf` (**untested** sketch; the node and partition values are placeholders for an 8-GPU node):

```ini
# slurm.conf (excerpt)
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory          # cores and memory are consumable
GresTypes=gpu
NodeName=gpu[001-096] CPUs=192 RealMemory=2000000 Gres=gpu:h100:8 State=UNKNOWN
PartitionName=train Nodes=gpu[001-096] OverSubscribe=EXCLUSIVE MaxTime=7-00:00:00 State=UP

# gres.conf
AutoDetect=nvml                              # ask the NVIDIA driver which GPUs exist and their core affinity
```

In practice, for full-node training jobs, `cons_tres` has little to decide. Most ML partitions use `OverSubscribe=EXCLUSIVE` or jobs request `--exclusive`, so the interesting decisions are which nodes the job gets (topology) and when it gets them (priority plus backfill). You can see what it decided for a running job (**untested**, output trimmed and sketched):

```
$ scontrol show job 81234 -d | grep -E 'Nodes=|GRES'
   Nodes=gpu017 CPU_IDs=0-191 Mem=2000000 GRES=gpu:h100:8(IDX:0-7)
   ...
```

### Topology: `topology/tree`

`TopologyPlugin=topology/tree` reads a switch hierarchy from `topology.conf`:

```
# topology.conf -- leaf switches with their nodes, spine with its leaves
SwitchName=leaf01 Nodes=gpu[001-032]
SwitchName=leaf02 Nodes=gpu[033-064]
SwitchName=leaf03 Nodes=gpu[065-096]
SwitchName=spine01 Switches=leaf[01-03]
```

Given a job of N nodes, the tree plugin looks for the lowest switch in the hierarchy whose subtree has N eligible nodes. It then picks leaves under that switch so that the allocation touches as few leaves as possible, preferring leaves that are already partly used, which is a best-fit approach to limit fragmentation. The idea behind best-fit is to leave the big empty holes intact for the big jobs that will need them later. Here is the idea in miniature (a simplification of what Slurm does, not a port of its code) **[ran]**:

```python
# tree_select.py -- what topology/tree does, simplified: lowest switch that fits, fewest leaves
leaves = {"leaf01": 32, "leaf02": 32, "leaf03": 32}          # switch -> nodes under it
free   = {"leaf01": 10, "leaf02": 30, "leaf03": 24}          # currently idle nodes
spine  = {"spine01": ["leaf01", "leaf02", "leaf03"]}

def select(n, max_leaves=None):
    # 1) a single leaf that fits? pick the one with the FEWEST free nodes that still
    #    fits (best fit: keeps big holes intact for big jobs)
    fits = sorted((free[l], l) for l in leaves if free[l] >= n)
    if fits:
        return {fits[0][1]: n}
    if max_leaves == 1:
        return None                                   # --switches=1: wait instead
    # 2) go up a level; take the fullest leaves first so the job touches few leaves
    for sw, ls in spine.items():
        if sum(free[l] for l in ls) >= n:
            alloc, need = {}, n
            for l in sorted(ls, key=lambda l: -free[l]):
                take = min(free[l], need)
                if take: alloc[l] = take; need -= take
                if need == 0: return alloc
    return None

print("8 nodes  ->", select(8))
print("20 nodes ->", select(20))
print("40 nodes ->", select(40))
print("40 nodes, --switches=1 ->", select(40, max_leaves=1))
```

```
8 nodes  -> {'leaf01': 8}
20 nodes -> {'leaf03': 20}
40 nodes -> {'leaf02': 30, 'leaf03': 10}
40 nodes, --switches=1 -> None
```

The 8-node job goes to the most-used leaf (`leaf01`, 10 free) rather than the emptiest, and the 40-node job spans two leaves rather than three. Users can make this stricter:

```bash
# require at most 1 leaf switch, wait up to 2h for that, then accept any placement
sbatch --nodes=32 --switches=1@2:00:00 train.slurm
```

The limit on the wait (`@time`) matters. Without it a 32-node job on a busy cluster might never find a completely empty leaf. `max_switch_wait` in `SchedulerParameters` sets the upper bound that a user can request.

There are two limitations for ML. First, the tree has no concept of NVLink domains. Second, on rail-optimized fabrics a node connects to many leaves (one per rail), which a strict tree can't represent. Sites usually model a scalable unit as one "switch" for this reason.

### Topology: `topology/block`

`topology/block` was added in 23.11 and redesigned in 24.05, according to SchedMD's SLUG'24 material. It models flat groups of nodes with fast connectivity between them. The motivating case is the NVL72 rack, where 18 nodes share one NVLink domain:

```
# topology.conf for block topology
BlockName=nvl72_r01 Nodes=gb[001-018]
BlockName=nvl72_r02 Nodes=gb[019-036]
BlockName=nvl72_r03 Nodes=gb[037-054]
BlockSizes=18,36,54
```

Blocks are ranges of nodes that don't overlap. `BlockSizes` lists the aggregation levels the scheduler is allowed to treat as one unit: a single block, two neighbouring blocks, and so on. Jobs can ask for allocations to be split into segments that each stay within a block, using the `--segment` option. For example, a 64-node job with `--segment=16` asks for 4 segments of 16 nodes, each inside one NVLink domain:

```bash
# untested; check `man sbatch` for --segment on your release before relying on it
sbatch --nodes=64 --segment=16 train.slurm
```

I'm uncertain about the exact semantics and the version in which `--segment` appeared. Read `man sbatch` for your release, and NVIDIA's Mission Control docs, which say NVLink-aware scheduling on GB200/GB300 needs `topology/block`. Why segment sizes like 16 rather than 18? Because it leaves 2 trays per rack free for hot spares or small jobs, and because 16 divides typical tensor/expert-parallel group sizes (see the fragmentation table above).

Since roughly 25.05, Slurm can also configure several topologies per cluster (`topology.yaml`, with different partitions using different plugins). I'm uncertain about this too, so check the `topology.yaml` man page for your version before you rely on it.

### Priority: `priority/multifactor`

The order in which jobs are considered is set by `PriorityType=priority/multifactor`:

```
priority = site_factor
         + PriorityWeightAge       * age_factor        # 0..1, saturates at PriorityMaxAge
         + PriorityWeightAssoc     * assoc_factor
         + PriorityWeightFairshare * fairshare_factor  # 0..1
         + PriorityWeightJobSize   * jobsize_factor    # 0..1
         + PriorityWeightPartition * partition_factor
         + PriorityWeightQOS       * qos_factor
         + sum(PriorityWeightTRES[i] * tres_factor[i])
         - nice
```

An **association** in Slurm is the (cluster, account, user, partition) tuple that usage and limits are attached to; a **QoS** (quality of service) is a named bundle of priority, limits and preemption rules that jobs can request. Every factor is normalized to [0, 1], so the weights are what actually set the policy. If `PriorityWeightFairshare=10000` and `PriorityWeightAge=1000`, a full week of queue time is worth only a tenth of the fair-share range. In code **[ran]**:

```python
# priority.py -- Slurm multifactor priority: the weights are the policy
W = dict(age=1000, fairshare=10000, jobsize=2000, partition=0, qos=0)   # example slurm.conf weights
PRIORITY_MAX_AGE_DAYS = 7                                              # PriorityMaxAge=7-0

def priority(age_days, fairshare, size_frac, nice=0):
    f = dict(age=min(age_days / PRIORITY_MAX_AGE_DAYS, 1.0),   # each factor in [0, 1]
             fairshare=fairshare, jobsize=size_frac, partition=0, qos=0)
    return sum(W[k] * f[k] for k in W) - nice

# a heavy user's job that has waited a full week vs a light user's job submitted now
print("heavy user, waited 7d, F=0.25, 512/2048 nodes:", priority(7, 0.25, 512 / 2048))
print("light user, waited 0d, F=0.71, 512/2048 nodes:", priority(0, 0.707, 512 / 2048))
```

```
heavy user, waited 7d, F=0.25, 512/2048 nodes: 4000.0
light user, waited 0d, F=0.71, 512/2048 nodes: 7570.0
```

The light user's brand-new job jumps the heavy user's week-old one. Whether that is right is a policy question, and the weights are where you answer it. (The job-size factor is modelled here as the fraction of the cluster requested; Slurm's exact normalization has options, so treat it as illustrative.) The matching `slurm.conf` lines (**untested**):

```ini
PriorityType=priority/multifactor
PriorityDecayHalfLife=7-0          # usage half-life: 7 days
PriorityMaxAge=7-0                 # age factor reaches 1.0 after 7 days in queue
PriorityWeightAge=1000
PriorityWeightFairshare=10000
PriorityWeightJobSize=2000
PriorityFavorSmall=NO              # bigger jobs get a bigger job-size factor
```

The fair-share factor in its classic form is `F = 2^(-U/S)`, where `U` is the association's normalized usage, decayed with `PriorityDecayHalfLife` (default 7 days), and `S` is its normalized share. Here is a worked example for a team entitled to 25% of the cluster:

| Recent usage U | U/S | F = 2^(-U/S) |
|---|---|---|
| 12.5% | 0.5 | 0.707 |
| 25% (exactly its share) | 1.0 | 0.5 |
| 50% (double its share) | 2.0 | 0.25 |

And how a burst of usage fades with the half-life, assuming the team then stops using the cluster entirely **[ran]**:

```python
# fairshare.py -- classic Slurm fair-share factor F = 2^(-U/S), with usage decay
S = 0.25                                         # team is entitled to 25% of the cluster
for U in (0.125, 0.25, 0.5):
    print(f"U={U:<6} U/S={U/S:<4} F={2 ** (-U / S):.3f}")

# usage decays with PriorityDecayHalfLife (default 7 days): a burst fades away
half_life_days, burst = 7, 0.5                   # team used 50% of the cluster last week
for day in (0, 7, 14, 21):
    U = burst * 0.5 ** (day / half_life_days)
    print(f"day {day:>2}: decayed U={U:.3f}  F={2 ** (-U / S):.3f}")
```

```
U=0.125  U/S=0.5  F=0.707
U=0.25   U/S=1.0  F=0.500
U=0.5    U/S=2.0  F=0.250
day  0: decayed U=0.500  F=0.250
day  7: decayed U=0.250  F=0.500
day 14: decayed U=0.125  F=0.707
day 21: decayed U=0.062  F=0.841
```

The default algorithm has been Fair Tree since 19.05. Fair Tree ranks associations by walking the account hierarchy, and it does not apply the formula above at every level, but the intuition is the same: using more than your share lowers your rank, and the effect decays with the half-life. If you use the cluster heavily in one week, your jobs are deprioritized for about the next week. Before you blame the scheduler, look at the factor breakdown with `sprio -l` and the usage with `sshare -l` (**untested**; columns abbreviated and values invented to show the shape):

```
$ sprio -l -u alice
  JOBID PARTITION     USER  PRIORITY   SITE    AGE  FAIRSHARE  JOBSIZE  PARTITION  QOS
  81234     train    alice      4000      0   1000       2500      500          0    0

$ sshare -l -A team-llm
Account     User  RawShares  NormShares  RawUsage  EffectvUsage  FairShare
--------- ------ ---------- ----------- --------- ------------- ----------
team-llm               25        0.25   9.1e+08         0.50       0.25
```

`PriorityWeightJobSize` with `PriorityFavorSmall=NO` favours large jobs. Without it, large jobs can starve, because small jobs keep getting scheduled and the large job never finds N free nodes at once. Backfill (next section) is the other half of the fix.

### Backfill: `sched/backfill`

The main scheduler is event-driven and strict: it walks jobs in priority order and stops at the first one that doesn't fit in a partition. That keeps priority order but leaves holes. `SchedulerType=sched/backfill` runs every `bf_interval` seconds (default 30) and does the following:

1. Builds a timeline of node availability up to `bf_window` minutes ahead (default 1440, i.e. one day), at `bf_resolution` granularity (default 60 s), using the **time limits** of the running jobs.
2. Walks the pending jobs in priority order. For each job it finds the earliest time the job could start and records a reservation for it in the timeline.
3. Starts a lower-priority job now if it fits in the gaps without delaying any reservation made before it.

Here is the core rule on a 16-node cluster, where the only thing that differs between two eval jobs is the `--time` they were submitted with **[ran]**:

```python
# backfill.py -- conservative backfill in miniature: reserve for the top job, fill holes
# Cluster of 16 nodes. Times in hours. The scheduler only knows TIME LIMITS, not true runtimes.
NODES = 16
running = [(12, 3.0)]                  # (nodes, hours until its time limit) -> 4 nodes idle now
pending = [                            # priority order
    ("big-train", 16, 48.0),           # needs the whole cluster
    ("eval-honest", 4, 2.0),           # 2 h job, --time=2:00:00
    ("eval-lazy", 4, 168.0),           # same 2 h job, --time=7-0 (the default limit)
]

def free_at(t, extra):
    used = sum(n for n, end in running + extra if end > t)
    return NODES - used

# earliest start for the top job = when enough running jobs reach their limits
t_res = min(t for t in [0.0] + [end for _, end in running] if free_at(t, []) >= pending[0][1])
print(f"reservation: {pending[0][0]} starts at t={t_res}h on {pending[0][1]} nodes")

for name, n, limit in pending[1:]:
    fits_now = free_at(0.0, []) >= n
    ends_before_reservation = limit <= t_res
    ok = fits_now and ends_before_reservation
    print(f"{name:12s} n={n} limit={limit:>5}h -> {'BACKFILL now' if ok else 'wait'}"
          f"{'' if ok else ' (would delay the reservation)'}")
```

```
reservation: big-train starts at t=3.0h on 16 nodes
eval-honest  n=4 limit=  2.0h -> BACKFILL now
eval-lazy    n=4 limit=168.0h -> wait (would delay the reservation)
```

There are three consequences for ML clusters:

- **Time limits are the scheduler's input.** A 2-hour eval job submitted with a 7-day time limit can't backfill in front of a reserved 512-node training job, because the scheduler assumes it will run for 7 days. Accurate `--time` values give you a better queue position.
- **`bf_window` must cover your longest time limit.** If jobs can run for 7 days but `bf_window=1440`, then the reservation for a large job whose nodes free up on day 3 lies outside the window. Backfill can then place jobs that end up delaying it. SchedMD's guidance is to set `bf_window` to at least the largest partition time limit. The cost is CPU time in `slurmctld` (Slurm's central controller daemon; `slurmd` is the per-node daemon), which you can reduce by raising `bf_resolution`.
- **Large clusters cap the backfill depth.** `bf_max_job_test`, `bf_max_job_user` and `bf_continue` limit how many jobs each backfill cycle looks at. With 50k pending array tasks, the job you care about may never be evaluated. `sdiag` shows backfill cycle times and depth.

A configuration for a cluster with 7-day limits, and where to look when backfill misbehaves (**untested**; the `sdiag` output is sketched and trimmed):

```ini
SchedulerType=sched/backfill
# bf_window is in minutes: 7 days = 10080. bf_resolution in seconds.
SchedulerParameters=bf_window=10080,bf_resolution=300,bf_max_job_test=1000,bf_continue,max_switch_wait=86400
```

```
$ sdiag | sed -n '/Backfilling stats/,/^$/p'
Backfilling stats
        Total backfilled jobs (since last slurm start): 18211
        Last cycle: 1432112          # microseconds: a 1.4 s cycle
        Max cycle:  9876543
        Last depth cycle: 1000       # hit bf_max_job_test: jobs past 1000 were not evaluated
```

If "Last depth cycle" sits at your `bf_max_job_test` value, backfill is not seeing the whole queue.

### Preemption

There are two preemption types. `PreemptType=preempt/qos` makes preemption follow QoS relationships (`sacctmgr modify qos high set preempt=low`). `preempt/partition_prio` makes it follow partition `PriorityTier`. `PreemptMode` sets what happens to the victim:

| PreemptMode | Effect on the victim | Fits training? |
|---|---|---|
| `CANCEL` | Killed after `GraceTime` | Only with a requeue chain (job array or dependency) |
| `REQUEUE` | Killed after `GraceTime`, put back in the queue with the same job ID | Yes, if the job resumes from a checkpoint |
| `SUSPEND` | SIGSTOP; the processes stay in GPU memory | Rarely: GPU memory isn't freed and NCCL timeouts fire on resume |
| `GANG` | Time-slicing between jobs | Not for GPU jobs |

`GraceTime` is how long the victim has between SIGTERM and SIGKILL. A signal is an asynchronous notification the OS delivers to a process: SIGTERM asks it to exit and can be caught, SIGKILL kills it and can't be caught, and SIGSTOP freezes it. Set it to at least your emergency-checkpoint time. The [preemption section of Fault Tolerance](../training/fault-tolerance/README.md#dealing-with-forced-job-preemption) has the signal-handling side. There are a few more preemption knobs (`PreemptExemptTime`, which protects a job just after it starts, and `preempt_youngest_first` in `PreemptParameters`). Their exact names have changed between releases, so check `slurm.conf(5)`.

A QoS-based setup where a `low` QoS soaks up idle nodes and gets requeued when `high` work arrives (**untested**):

```bash
# slurm.conf:
#   PreemptType=preempt/qos
#   PreemptMode=REQUEUE
sacctmgr add qos low
sacctmgr modify qos low set GraceTime=300          # 5 min from SIGTERM to SIGKILL
sacctmgr modify qos high set preempt=low           # high may preempt low

# opportunistic jobs:
sbatch --qos=low --requeue train.slurm
```

The training side has to turn SIGTERM into "save and exit" within the grace time. The minimum handler (runnable anywhere; it just sets a flag the training loop checks between steps) **[ran]**:

```python
# sigterm_ckpt.py -- turn SIGTERM into "finish this step, save, exit"
import os, signal, time

stop = False
def on_term(signum, frame):
    global stop
    stop = True                              # never do the save inside the handler itself

signal.signal(signal.SIGTERM, on_term)
os.kill(os.getpid(), signal.SIGTERM)         # simulate Slurm preempting us

for step in range(1000):
    time.sleep(0.001)                        # one training step
    if stop:
        print(f"SIGTERM seen at step {step}: saving checkpoint and exiting cleanly")
        break
```

```
SIGTERM seen at step 0: saving checkpoint and exiting cleanly
```

The handler only sets a flag because a signal can arrive in the middle of anything, including the middle of an all-reduce. If every rank saved from inside the handler, some ranks would be mid-collective and the save would deadlock. Checking the flag at step boundaries means all ranks stop at the same consistent point (they should all-reduce the flag first so they agree on which step that is).


## Kubernetes for ML

The [Kubernetes chapter](kubernetes/README.md) gives the practical case against k8s for interactive training, and [Kubernetes and Helm from zero](kubernetes-from-zero.md) teaches the objects themselves. This section explains why the case against exists and which add-ons close each gap.

Kubernetes is built around **controllers**: programs that watch the desired state you wrote into the API server (a "spec") and keep acting until the actual state matches. That design is great for services ("keep 5 replicas running") and awkward for batch training ("start all 64 at once, and if one dies, kill the other 63"). Almost every add-on below exists to teach the controllers and the scheduler that second sentence.

The pieces of a Kubernetes cluster (control plane and nodes) that everything below plugs into:

![Kubernetes cluster components](images/kubernetes-components.svg)

([source](https://kubernetes.io/docs/concepts/overview/components/))

### Exposing GPUs: device plugins

The original mechanism is the [device plugin API](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/). In Kubernetes, a node is a machine and a pod is a group of containers that is scheduled onto one node as a unit. The kubelet is the agent on each node that starts and monitors pods. A device plugin is a gRPC (a remote procedure call protocol) service on each node that registers with the kubelet and implements three calls:

- `ListAndWatch`, which streams a list of device IDs and their health
- `Allocate`, which, given a list of device IDs, returns the env vars, mounts and device nodes to inject into the container
- `GetPreferredAllocation` (optional), which lets the plugin choose which N devices to hand out, for example NVLink-connected ones

The scheduler only sees an integer extended resource such as `nvidia.com/gpu: 8`. It has no idea which GPU, NUMA node, NVLink island or NIC a pod gets. The kubelet sees which devices sit on which PCIe bus (the motherboard bus that connects GPUs and NICs to the CPU) and NUMA node, but the scheduler doesn't. Topology inside the node is handled later by the kubelet's Topology Manager (`--topology-manager-policy=single-numa-node` and similar), which can only reject a pod after the scheduler has placed it. The device plugin model also can't express "a GPU plus the NIC on the same PCIe switch", sharing a device between pods, or devices that span nodes.

What this looks like from the outside: the node advertises a count, and a pod asks for a count (**untested**; output sketched):

```
$ kubectl get node gpu0712-03 -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'
8
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: one-gpu
spec:
  restartPolicy: Never
  containers:
  - name: smi
    image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04   # any CUDA base image
    command: ["nvidia-smi", "-L"]
    resources:
      limits:
        nvidia.com/gpu: 1          # an integer, nothing more; which GPU is the plugin's choice
```

### Exposing GPUs: Dynamic Resource Allocation (DRA)

DRA reached GA in Kubernetes v1.34, with the `resource.k8s.io/v1` API on by default. It was alpha with structured parameters in v1.30 and beta in v1.32 ([KEP-4381](https://www.kubernetes.dev/resources/keps/4381/)). Here is how it works:

- A DRA driver on each node publishes **ResourceSlices**: lists of devices with typed attributes such as product name, memory, PCIe root and NVLink clique.
- Admins define **DeviceClasses**, for example "any H100".
- Workloads create **ResourceClaims**, or **ResourceClaimTemplates** for a per-pod claim, that select devices with CEL (Common Expression Language, a small filter-expression language) expressions. Claims can also add constraints such as `matchAttribute`, which puts all devices in the claim on the same PCIe root.
- The **scheduler itself** allocates specific devices to claims while it places the pod. Allocation is part of the scheduling decision, not something the kubelet does afterwards.

Why the redesign? Because the device-plugin split (scheduler counts, kubelet picks) means the component that knows the topology runs *after* the component that decides placement. DRA moves the device attributes into the API server, where the scheduler can see them.

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: eight-gpus
spec:
  spec:
    devices:
      requests:
      - name: gpus
        exactly:
          deviceClassName: gpu.nvidia.com
          count: 8
```

The pod refers to the template by name, and each container lists which claims it uses (**untested**; field names as of `resource.k8s.io/v1` in Kubernetes 1.34):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: trainer-0
spec:
  resourceClaims:
  - name: gpus
    resourceClaimTemplateName: eight-gpus     # a fresh claim is created per pod
  containers:
  - name: trainer
    image: my-trainer:2026.11
    resources:
      claims:
      - name: gpus                            # this container gets the 8 allocated GPUs
```

A DeviceClass is a named CEL filter. The class named `gpu.nvidia.com` above is normally installed by the NVIDIA DRA driver; a hand-written one looks like this (**untested**):

```yaml
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: any-nvidia-gpu
spec:
  selectors:
  - cel:
      expression: "device.driver == 'gpu.nvidia.com'"
```

For ML clusters the important part is [NVIDIA's DRA driver for GPUs](https://github.com/NVIDIA/k8s-dra-driver-gpu), and specifically its **ComputeDomain** abstraction for multi-node NVLink (GB200 NVL72 and later). A ComputeDomain is a claim that spans pods on several nodes. It sets up the IMEX (Internode Memory Exchange) channels that cross-node NVLink memory sharing needs, which a per-node device plugin can't do. I'm moderately confident of this description; the driver has been moving fast, and I haven't included a ComputeDomain manifest because I'm not confident of its current field names. The core of DRA is GA, but parts of the ecosystem (autoscaler support, admin access, partitionable devices) were still beta or alpha at the time of writing.

### NVIDIA GPU Operator components

An **operator** is a controller plus CRDs that automates running some piece of software on Kubernetes. The [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html) is a set of DaemonSets (a DaemonSet runs one copy of a pod on every matching node) that turn a plain node into a GPU node:

| Component | Role |
|---|---|
| Node Feature Discovery (NFD) | Labels nodes with PCI vendor IDs so the rest of the stack knows where to run |
| Driver container | Builds and loads the kernel driver (the OS-level code that talks to the GPU hardware) from a container (optional; many sites install drivers on the host) |
| NVIDIA Container Toolkit | Configures containerd/CRI-O so that containers see the GPUs (CDI or the runtime hook) |
| Device plugin | Advertises `nvidia.com/gpu` (or the DRA driver, on newer setups) |
| GPU Feature Discovery (GFD) | Labels nodes with GPU model, memory, MIG capability and driver/CUDA version, e.g. `nvidia.com/gpu.product` |
| DCGM / DCGM exporter | Health and telemetry, with Prometheus metrics (see [Observability](#observability)) |
| MIG manager | Applies MIG (Multi-Instance GPU, which splits one GPU into isolated smaller GPUs) partition layouts from a node label |
| Validator | Runs CUDA sample workloads to gate the node as ready |

Installing it is a Helm chart (Helm is the Kubernetes package manager; see [Kubernetes and Helm from zero](kubernetes-from-zero.md)). The commands below follow NVIDIA's install docs; **untested**, and the pod list is a sketch:

```
$ helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
$ helm install --wait --generate-name -n gpu-operator --create-namespace nvidia/gpu-operator

$ kubectl get pods -n gpu-operator
NAME                                          READY   STATUS      RESTARTS
gpu-feature-discovery-xxxxx                   1/1     Running     0
nvidia-container-toolkit-daemonset-xxxxx      1/1     Running     0
nvidia-cuda-validator-xxxxx                   0/1     Completed   0
nvidia-dcgm-exporter-xxxxx                    1/1     Running     0
nvidia-device-plugin-daemonset-xxxxx          1/1     Running     0
nvidia-operator-validator-xxxxx               1/1     Running     0
...

$ kubectl get node gpu0712-03 -L nvidia.com/gpu.product
NAME         STATUS   ROLES    AGE   VERSION   GPU.PRODUCT
gpu0712-03   Ready    <none>   41d   v1.34.1   NVIDIA-H100-80GB-HBM3
```

RDMA is handled by the separate [NVIDIA Network Operator](https://docs.nvidia.com/networking/display/kubernetes2410/index.html). It deploys the MOFED/DOCA driver (NVIDIA's Mellanox NIC driver stack), Multus, the SR-IOV and RDMA shared device plugins, and the IPAM pieces covered [below](#rdma-networking-for-pods).

A common failure mode: a driver container upgrade or validator failure takes `nvidia.com/gpu` capacity to 0 on a node, and running pods keep running while new pods stay `Pending`. Alert on allocatable GPU count per node, not just on node `Ready`. A quick check across the fleet (**untested**):

```bash
# nodes whose allocatable GPU count is not 8 (prints "<none>" where the resource vanished)
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu' \
  | awk 'NR==1 || $2 != "8"'
```

### Why the default kube-scheduler fails at gang scheduling

`kube-scheduler` takes **one pod at a time** from its queue and runs filter and score plugins over the nodes, then binds the pod (records the chosen node, after which the kubelet starts it). It has no concept that 64 pods belong together. A Job or StatefulSet of 64 pods produces 64 independent decisions, and you get the 48+48 deadlock from [earlier](#why-gang-scheduling-cannot-be-skipped). Pod anti-affinity and topology spread constraints don't help, because they constrain where each pod goes, not whether the set should start at all.

Why was it built this way? Because for services, pod-at-a-time is exactly right: each replica is independently useful, and scheduling them one by one is simple, fast and easy to reason about. Gang semantics need the scheduler to hold state about groups and to sometimes decline to place a pod that fits, which is a different design.

The fixes come in three layers:

1. **A coscheduling plugin inside kube-scheduler** ([scheduler-plugins](https://github.com/kubernetes-sigs/scheduler-plugins) `Coscheduling` with a `PodGroup` CRD and `minMember`). Pods wait at the `Permit` stage (a hook between choosing a node and binding the pod) until `minMember` of them have been assigned. This works, but it holds partial reservations while it waits.
2. **A replacement batch scheduler** that schedules whole groups: Volcano or KAI.
3. **Admission control in front of the default scheduler**: Kueue. It doesn't place pods. It only decides when a whole job may be unsuspended, based on quota. A CRD (Custom Resource Definition) adds a new object type to the Kubernetes API, and many of the projects below are CRDs plus a controller that acts on them.

Layer 1 with scheduler-plugins (**untested**; the API group matches the project's current CRD as far as I know, but the pod label key and the scheduler name depend on the version and how you installed it, so check its README):

```yaml
apiVersion: scheduling.x-k8s.io/v1alpha1
kind: PodGroup
metadata:
  name: pretrain
spec:
  minMember: 32                  # bind nothing until 32 pods can all be placed
  scheduleTimeoutSeconds: 600    # give up and release partial reservations after 10 min
---
# in each worker pod template:
#   metadata:
#     labels:
#       scheduling.x-k8s.io/pod-group: pretrain
#   spec:
#     schedulerName: <the scheduler-plugins scheduler's name in your install>
```

Layer 2 with Volcano's own job type, where the gang size is `minAvailable` (**untested**):

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: pretrain
spec:
  schedulerName: volcano
  queue: research
  minAvailable: 32               # the gang: all 32 or none
  policies:
  - event: PodEvicted            # any pod evicted -> restart the whole job
    action: RestartJob
  tasks:
  - name: worker
    replicas: 32
    template:
      spec:
        restartPolicy: Never
        containers:
        - name: trainer
          image: my-trainer:2026.11
          resources:
            limits:
              nvidia.com/gpu: 8
```

Upstream Kubernetes has been working on native gang scheduling: a `Workload` API plus gang-scheduling support in kube-scheduler (KEP-4671), targeted at alpha around v1.35. I haven't verified its current status; treat it as experimental.

### Kueue vs Volcano vs KAI

| | [Kueue](https://kueue.sigs.k8s.io/) | [Volcano](https://volcano.sh/) | [KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler) |
|---|---|---|---|
| What it is | Job queueing and quota admission controller | Full batch scheduler that replaces kube-scheduler for its pods | Full batch scheduler (Run:ai's engine, open-sourced April 2025, Apache 2.0) |
| Places pods? | No: the default scheduler (or another one) does | Yes | Yes |
| Gang semantics | All-or-nothing **admission** (quota for the whole job); `waitForPodsReady` evicts or requeues jobs whose pods don't all become ready | `PodGroup` with `minMember`; `gang` plugin | `PodGroup`s, including hierarchical ones (for disaggregated inference) |
| Quota model | `ClusterQueue`/`LocalQueue`, `ResourceFlavor`s, cohorts with borrowing and lending limits | Queues with `proportion` / DRF plugins, `reclaim` action | Hierarchical queues with quotas, over-quota weights, fair share |
| Topology | Topology Aware Scheduling (TAS): `podset-required-topology` / `podset-preferred-topology` against node-label levels | Network-topology-aware scheduling via `HyperNode` CRDs (newer releases; not verified) | TAS from v0.10 (Oct 2025), including hierarchical; NVLink clique placement |
| Preemption | Within a ClusterQueue and across a cohort to reclaim lent quota | `preempt` and `reclaim` actions | Priority and reclaim, plus consolidation to reduce fragmentation |
| Ecosystem fit | Kubernetes SIG project; integrates with Job, JobSet, LWS, RayJob, Kubeflow trainer, plain pods | CNCF project; long history in HPC and AI shops; its own `vcjob` CRD plus integrations | NVIDIA stack (DRA ComputeDomains, Dynamo/Grove); GPU sharing (fractions) |

(DRF, dominant resource fairness, is a fair-share policy that equalizes each tenant's share of whichever resource it uses most.)

How to choose:

- **Kueue** if you want to keep kube-scheduler and add multi-tenant quota and job-level admission. Kueue suspends a Job (`spec.suspend: true`) until it fits, then injects node selectors for the chosen `ResourceFlavor` (and TAS domain) and unsuspends it. Pod placement within those selectors is still done pod by pod, so TAS works by narrowing the node set to a domain that Kueue has already confirmed has capacity, and by gating pods with a scheduling gate until it has computed the assignment.
- **Volcano or KAI** if you need the scheduler itself to reason about groups: bin-packing GPUs to limit fragmentation, consolidation, hierarchical gangs, or GPU sharing. The cost is running a second scheduler, and the pods it owns don't benefit from default-scheduler features until the replacement implements them.

Combinations exist, such as Kueue for quota on top of KAI or Volcano for placement. Check the compatibility matrix for your versions. I haven't validated any particular combination.

The Kueue objects that implement the borrowing model from the [fairness section](#fairness-and-utilization-pull-in-opposite-directions): a `ResourceFlavor` names a kind of node, a `ClusterQueue` holds quota for flavors and belongs to a cohort (a set of ClusterQueues that lend to each other), and a namespaced `LocalQueue` is what users submit to. **Untested**; field names as of Kueue's `v1beta1` API (in the `v1beta2` API, which newer releases introduce, I believe `cohort` is renamed `cohortName`, so check your version):

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: h100
spec:
  nodeLabels:
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
  topologyName: dc-topology          # enables TAS for this flavor (Topology object below)
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: team-llm
spec:
  cohort: research                   # lends to / borrows from other queues in "research"
  namespaceSelector: {}              # any namespace may point a LocalQueue at it
  preemption:
    reclaimWithinCohort: Any         # take back lent quota by preempting borrowers
    withinClusterQueue: LowerPriority
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: h100
      resources:
      - name: cpu
        nominalQuota: 3072
      - name: memory
        nominalQuota: 24Ti
      - name: nvidia.com/gpu
        nominalQuota: 128            # this team's share
        borrowingLimit: 128          # may use up to 128 more from the cohort when idle
        lendingLimit: 64             # never lend out more than 64 of its own
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: training
  namespace: team-llm
spec:
  clusterQueue: team-llm
---
apiVersion: kueue.x-k8s.io/v1alpha1   # Topology was alpha in the releases I know; check yours
kind: Topology
metadata:
  name: dc-topology
spec:
  levels:                            # outermost first; each is a node-label key
  - nodeLabel: example.com/topology-block
  - nodeLabel: example.com/topology-rack
  - nodeLabel: kubernetes.io/hostname
```

`waitForPodsReady`, which turns "admitted" into "all pods actually running, or requeue", lives in Kueue's controller configuration rather than in a CRD (**untested**; other sub-fields such as the requeue backoff vary by version):

```yaml
apiVersion: config.kueue.x-k8s.io/v1beta1
kind: Configuration
waitForPodsReady:
  enable: true
  timeout: 10m          # if not all pods are Ready within 10 min, evict and requeue the workload
```

Here is what Kueue TAS looks like on a training job: all pods inside one rack-level domain, or wait.

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: llm-pretrain
  labels:
    kueue.x-k8s.io/queue-name: training
spec:
  replicatedJobs:
  - name: workers
    replicas: 1
    template:
      spec:
        parallelism: 32
        completions: 32
        completionMode: Indexed
        template:
          metadata:
            annotations:
              # the topology level is a node-label key defined in a Kueue Topology object
              kueue.x-k8s.io/podset-required-topology: "example.com/topology-block"
          spec:
            containers:
            - name: trainer
              image: my-trainer:2026.11
              resources:
                limits:
                  nvidia.com/gpu: 8
```

What a user sees (**untested**, sketched): the JobSet's Workload object (Kueue's internal record of a job's resource request) moves from pending to admitted, and only then do pods appear.

```
$ kubectl -n team-llm get workloads
NAME                         QUEUE      RESERVED IN   ADMITTED   AGE
jobset-llm-pretrain-1a2b3   training   team-llm      True       2m
```

### JobSet and LeaderWorkerSet

A plain `Job` describes a set of identical pods and has no stable network identity. Multi-node training needs stable DNS (the system that maps hostnames to IP addresses) names for rendezvous (the step where all workers find each other before training starts), one failure policy for the whole group, and sometimes several pod roles (for example trainer plus parameter server or data loader).

The minimum that works without any add-on is an **Indexed Job** plus a **headless Service**. An Indexed Job gives each pod a stable index `0..N-1` (in the `JOB_COMPLETION_INDEX` env var) and a hostname `<job>-<index>`. A headless Service (`clusterIP: None`) gives no load balancer, just per-pod DNS records, and setting the pod's `subdomain` to the Service name makes `<job>-<index>.<service>` resolvable. That is enough for a static torchrun launch (**untested**):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: train
spec:
  clusterIP: None                 # headless: DNS records only, no virtual IP
  selector:
    job-name: train               # label the Job controller puts on its pods
---
apiVersion: batch/v1
kind: Job
metadata:
  name: train
spec:
  completionMode: Indexed         # pods get JOB_COMPLETION_INDEX=0..3 and hostnames train-0..train-3
  completions: 4
  parallelism: 4
  backoffLimit: 0                 # any pod failure fails the Job; recovery is a resubmit
  template:
    spec:
      subdomain: train            # => train-0.train resolves to pod 0
      restartPolicy: Never
      containers:
      - name: trainer
        image: my-trainer:2026.11
        command: ["bash", "-c"]
        args:
        - >
          torchrun --nnodes=4 --nproc-per-node=8
          --node-rank=$JOB_COMPLETION_INDEX
          --master-addr=train-0.train --master-port=29500
          train.py
        resources:
          limits:
            nvidia.com/gpu: 8
```

What is missing: if pod 2 dies, the Job just fails (or, with a larger `backoffLimit`, restarts pod 2 alone, which is useless for synchronous training because the other three are stuck in a collective). There is also no gang admission and no notion of roles. Two SIG-apps/SIG-scheduling-sponsored CRDs provide these:

**[JobSet](https://jobset.sigs.k8s.io/)** is aimed at training. It holds a list of `replicatedJobs`, each an Indexed Job template. It creates a headless Service (a Kubernetes DNS entry per pod with no load balancer) so that pods get names like `<jobset>-<replicatedJob>-<jobIndex>-<podIndex>.<jobset>`, which can be used directly as `MASTER_ADDR`. Its `failurePolicy` (`maxRestarts`, plus rules matched on failure reason) recreates **all** child Jobs when any one fails. That matches what a synchronous training job needs: one dead rank means the whole job restarts from the checkpoint. JobSet also supports exclusive placement per topology domain, one child Job per domain, through the `alpha.jobset.sigs.k8s.io/exclusive-topology` annotation, and it integrates with Kueue.

The same 4-node job as a JobSet (**untested**; I've left out `failurePolicy.rules` because I'm not confident of the current rule schema):

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: pretrain
spec:
  failurePolicy:
    maxRestarts: 3                 # any child Job fails -> recreate ALL of them, up to 3 times
  network:
    enableDNSHostnames: true       # headless Service; pod DNS = pretrain-workers-0-<i>.pretrain
  replicatedJobs:
  - name: workers
    replicas: 1                    # one child Job...
    template:
      spec:
        parallelism: 4             # ...of 4 pods, one per node
        completions: 4
        completionMode: Indexed
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            containers:
            - name: trainer
              image: my-trainer:2026.11
              command: ["bash", "-c"]
              args:
              - >
                torchrun --nnodes=4 --nproc-per-node=8
                --node-rank=$JOB_COMPLETION_INDEX
                --master-addr=pretrain-workers-0-0.pretrain --master-port=29500
                train.py --resume=latest
              resources:
                limits:
                  nvidia.com/gpu: 8
```

Note the `--resume=latest`: JobSet's restart recreates pods, which start `main()` from scratch, exactly like a torchrun restart. Resuming from the checkpoint is your code's job in both cases.

**[LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/)** is aimed at multi-node **inference** replicas. Its unit is a group: 1 leader pod plus `size-1` workers, scheduled and restarted together. `replicas` scales the number of groups. It injects `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE` and `LWS_WORKER_INDEX`, and `RestartPolicy: RecreateGroupOnPodRestart` restarts the whole group when any member fails. That's correct for TP/PP sharded serving, where a replica missing one shard can't serve. vLLM and SGLang multi-node deployments use it (vLLM through a Ray head on the leader). Rolling updates and HPA work per group. The manifest is shown in [Multi-node inference replicas](#multi-node-inference-replicas).

Rule of thumb: a job that runs to completion is a JobSet; a replicated service whose replica spans nodes is an LWS.

### Topology-aware scheduling on Kubernetes

Topology has to be expressed as **node labels**, because that's the only thing the k8s scheduling layers can match on. A typical hierarchy:

```
example.com/topology-zone      = dc1-hall2
example.com/topology-block     = spine07        # or NVL72 rack / NVLink clique ID
example.com/topology-rack      = r0712
kubernetes.io/hostname         = gpu0712-03
```

Someone has to populate these labels. On clouds the provider may set them (GKE, for example, exposes `cloud.google.com/gce-topology-*` labels). I'm not certain of the exact names on every provider, so check your provider's docs. On bare metal, tools such as NVIDIA's [Topograph](https://github.com/NVIDIA/topograph) read the fabric (UFM, `ibnetdiscover`, cloud topology APIs) and write either k8s node labels or a Slurm `topology.conf`. GFD's `nvidia.com/gpu.clique` label identifies the NVLink domain on GB200-class systems. I believe that label name is correct, but haven't verified it.

By hand, from a rack map, it's a loop over `kubectl label` (**untested**; `racks.tsv` is a file you'd generate from your inventory system):

```bash
# racks.tsv: <node> <rack> <spine>
#   gpu0712-03  r0712  spine07
while read -r node rack spine; do
  kubectl label node "$node" --overwrite \
    example.com/topology-rack="$rack" \
    example.com/topology-block="$spine"
done < racks.tsv

kubectl get nodes -L example.com/topology-block,example.com/topology-rack
```

Once the labels exist, Kueue TAS, KAI TAS, Volcano HyperNodes and JobSet exclusive placement can all use them. Plain `podAffinity` with `topologyKey` can also express "same rack as each other", but only as a preference, without gang semantics, and it's slow at scale. Don't use it for training.

JobSet's exclusive placement is a single annotation: each child Job gets a topology domain to itself (**untested** fragment):

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: pretrain
  annotations:
    alpha.jobset.sigs.k8s.io/exclusive-topology: example.com/topology-rack   # one child Job per rack
spec:
  replicatedJobs:
  - name: workers
    replicas: 4                    # 4 child Jobs -> 4 different racks, none shared with other JobSets
    template:
      spec:
        parallelism: 16
        completions: 16
        completionMode: Indexed
        template:
          spec:
            restartPolicy: Never
            containers:
            - name: trainer
              image: my-trainer:2026.11
```

### RDMA networking for pods

A pod by default has one veth (virtual Ethernet) interface on the cluster network, set up by a CNI (Container Network Interface) plugin, routed through the host kernel. NCCL (NVIDIA's collective communication library, which PyTorch uses for GPU collectives) over that interface means TCP sockets, roughly 10-25% of the line rate of an IB/RoCE NIC (an order-of-magnitude estimate, not a measurement), plus CPU overhead. RDMA (remote direct memory access) lets a NIC read and write another machine's memory without involving either CPU or kernel. InfiniBand (IB) and RoCE (RDMA over Converged Ethernet) are the two common RDMA networks. There are three ways to give pods the RDMA fabric:

| Approach | How | Tradeoffs |
|---|---|---|
| `hostNetwork: true` | The pod shares the node's network namespace and sees every `mlx5_*` device and IP | Simplest, and near bare-metal NCCL. No network isolation; port collisions between pods; usually needs privileged or broad capabilities; one training pod per node in practice |
| Multus + SR-IOV | [Multus](https://github.com/k8snetworkplumbingwg/multus-cni) attaches extra interfaces; SR-IOV lets one physical NIC present itself as several virtual NICs; the SR-IOV device plugin advertises these virtual functions (VFs) (e.g. `nvidia.com/rdma_vf`) and the SR-IOV CNI moves one VF per rail into the pod | Proper isolation and multi-tenancy. Need one VF per NIC per pod, matching GPU-NIC affinity, IPAM (IP address management) per rail; more moving parts |
| RDMA shared device plugin | Exposes the host's RDMA devices to pods without SR-IOV (`rdma/hca_shared_devices_a`) | Simpler than SR-IOV; isolation is weaker than with VFs |

Each Multus network is a `NetworkAttachmentDefinition`: a name plus a CNI config in JSON. One per rail, each tied to that rail's VF pool (**untested**; the resource-name annotation and the IPAM plugin, here `whereabouts`, depend on your SR-IOV and Network Operator setup):

```yaml
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: rail0
  annotations:
    k8s.v1.cni.cncf.io/resourceName: nvidia.com/rdma_vf_rail0
spec:
  config: |
    {
      "cniVersion": "0.3.1",
      "name": "rail0",
      "type": "sriov",
      "ipam": { "type": "whereabouts", "range": "192.168.100.0/24" }
    }
```

A Multus attachment request looks like this:

```yaml
metadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: rail0,rail1,rail2,rail3,rail4,rail5,rail6,rail7
spec:
  containers:
  - name: trainer
    resources:
      limits:
        nvidia.com/gpu: 8
        nvidia.com/rdma_vf: 8      # resource name depends on your SR-IOV config
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]          # RDMA needs to pin memory
```

There are four things to verify inside the pod before you trust it:

- `ibv_devices` lists every NIC you expect, and `ulimit -l` is `unlimited`. RDMA registers memory (pins pages, the OS's 4 KB units of memory, so they can't be moved or swapped while the NIC accesses them), and without `IPC_LOCK` or a memlock limit you get `ibv_reg_mr` failures, or NCCL quietly falls back to sockets.
- NCCL's init log (`NCCL_DEBUG=INFO`) shows `NET/IB` and GPUDirect RDMA (`GDRDMA`, the NIC reading GPU memory directly with no copy through host RAM), not `NET/Socket`.
- GPU-to-NIC affinity is right. With DRA you can request a GPU and NIC pair that share a PCIe root. With device plugins you rely on the Topology Manager and luck.
- `nccl-tests` across 2 pods on 2 nodes matches bare-metal bus bandwidth (the per-GPU throughput metric nccl-tests reports, comparable across GPU counts; see [network benchmarks](../network/benchmarks/)).

As a shell session (**untested**; device names and the exact log text vary by NCCL version, so grep for the transport name rather than copying these lines):

```
$ kubectl exec -it pretrain-workers-0-0 -- bash
# ibv_devices
    device                 node GUID
    ------              ----------------
    mlx5_0              b8cef60300a1b2c3
    ...                 (one line per rail; expect 8)
# ulimit -l
unlimited
# NCCL_DEBUG=INFO torchrun ... 2>&1 | grep -E 'NET/(IB|Socket)|GDRDMA' | head -3
... NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB ... 
... NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA
```

If you see `NET/Socket`, stop: the job will run, just several times slower, and nothing will error.

On clouds, EFA (AWS) and GPUDirect-TCPX/TCPXO (GCP) have their own device plugins and NCCL plugins. The same checks apply.

### Shared memory (`/dev/shm`) pitfalls

`/dev/shm` is a tmpfs (a filesystem held in RAM) that processes on the same machine use to share memory. Container runtimes default `/dev/shm` to **64 MB**. Both PyTorch DataLoader workers, which pass batches through shared memory, and NCCL, which uses shm segments for intra-node transfers when P2P isn't available, fail with errors such as `Bus error`, `unable to write to file </torch_...>`, or NCCL `Error while creating shared memory segment` once a batch or a set of NCCL buffers is larger than that. You can see the default with any container runtime (**untested** here; Docker's documented default is 64 MB, and `--shm-size` changes it):

```
$ docker run --rm ubuntu df -h /dev/shm
Filesystem      Size  Used Avail Use% Mounted on
shm              64M     0   64M   0% /dev/shm

$ docker run --rm --shm-size=16g ubuntu df -h /dev/shm
shm              16G     0   16G   0% /dev/shm
```

The fix on k8s is a memory-backed `emptyDir`:

```yaml
spec:
  containers:
  - name: trainer
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
```

There are three catches:

1. Pages in a memory-backed `emptyDir` **count against the container's memory limit**. If you set `memory: 256Gi` and the DataLoader puts 64 GiB in shm, the model process has only 192 GiB. With a hard limit, going over it triggers the cgroup OOM killer (cgroups are the kernel feature that enforces a container's resource limits, and the OOM killer kills processes when a memory limit is exceeded), and with `memory.oom.group=1` that kills the whole pod (see the [k8s chapter](kubernetes/README.md#overcoming-job-reset-on-cpu-oom-event)).
2. `sizeLimit` is enforced by the kubelet's eviction loop, not as a hard tmpfs size, on some versions. I'm unsure of the exact behaviour across versions. Overshooting can mean eviction rather than `ENOSPC`.
3. shm segments left behind by a crashed process stay until the pod exits. Under in-place restarts by torchrun (`--max-restarts`), leaked `/dev/shm/torch_*` files build up across restarts.

For catch 3, a pre-start cleanup in the container entrypoint is cheap insurance (**untested**; Linux only, since macOS has no `/dev/shm`):

```bash
# entrypoint.sh: show and remove leftovers from a previous attempt in this pod
du -sh /dev/shm 2>/dev/null
find /dev/shm -maxdepth 1 -name 'torch_*' -mmin +1 -print -delete
exec torchrun "$@"
```

With `hostIPC: true`, the pod uses the host's `/dev/shm`. That avoids the size problem but gives up isolation between pods.


## SLURM on K8s and K8s alongside SLURM

Research teams want `sbatch`, `srun` and Unix users. Platform teams want k8s for everything else: inference, CI, data pipelines, and operators for storage and networking. Several projects try to give both.

| Project | Approach | Status (as best I can tell) |
|---|---|---|
| [Slinky](https://github.com/SlinkyProject) (SchedMD, now NVIDIA: NVIDIA announced it was acquiring SchedMD in Dec 2025 and said Slurm stays open source) | `slurm-operator`: runs slurmctld, slurmd, slurmdbd and slurmrestd as pods, managed through CRDs. `slurm-bridge`: a k8s scheduler plugin that lets **Slurm schedule k8s pods**, so both kinds of workload share one view of the nodes | Open source; actively developed. I haven't verified the exact CRD names (e.g. `NodeSet`) for the current version |
| [Soperator](https://github.com/nebius/soperator) (Nebius) | A `SlurmCluster` CRD; controller, worker and login nodes are all pods. The main idea is the "jail": a shared network filesystem mounted as `/` on every Slurm node, so installing a package on a login node makes it appear on all workers | Open source; claims to run on several clouds and on bare metal. I haven't verified how well it works outside Nebius |
| [SUNK](https://www.coreweave.com/products/sunk) (CoreWeave) | Slurm on k8s with k8s-side scheduling integration | Proprietary; CoreWeave customers only |

I haven't included install manifests for these because their CRDs change between releases; follow each project's quick start.

The tradeoffs that actually matter:

- **Who owns the node?** If the Slurm daemons run as pods on k8s nodes, k8s owns the node lifecycle: drain, cordon, upgrades, GPU Operator. Slurm's `drain` and k8s's `cordon` have to be kept consistent, or a node Slurm considers healthy gets drained underneath a job. Each project solves this differently. Read its docs on node health before you deploy.
- **Two schedulers, one set of GPUs.** Unless something like `slurm-bridge` makes one scheduler the authority, you're statically partitioning nodes between Slurm and k8s, which gives up the utilization argument for running both.
- **Container overhead on the data path.** slurmd running inside a pod still needs host-level access to GPUs, RDMA devices, cgroups and `/dev/shm`. Expect privileged pods and host networking. The performance should be close to bare metal if that access is set up properly, but I haven't seen independent benchmarks, so measure NCCL bus bandwidth and step time yourself.
- **Persistent node state.** Slurm assumes nodes keep their state (installed software, `/home`, `/tmp`). Pods are ephemeral. Soperator's jail is one answer. With Slinky you bake the state into images.

To make "keep drain and cordon consistent" concrete, here is the naive one-way reconciler a site might start with, making Slurm the source of truth for health (**untested**; the projects above have their own, better mechanisms, and Slurm node names are assumed to equal k8s node names):

```bash
#!/bin/bash
# drain-sync.sh -- every node Slurm has drained gets cordoned in k8s (one direction only)
set -euo pipefail
for node in $(sinfo -h -N -t drain -o '%N' | sort -u); do
  if [ "$(kubectl get node "$node" -o jsonpath='{.spec.unschedulable}')" != "true" ]; then
    echo "cordon $node (drained in slurm)"
    kubectl cordon "$node"
  fi
done
```

The hard part this script skips is the reverse direction (undraining when k8s says the node is healthy again) and the race where both systems act at once. That is why the "who owns the node" question has to be answered by design, not by a cron job.

The reverse direction, running k8s workloads inside Slurm allocations, exists as research and site-specific tooling (e.g. starting a kubelet inside a job, or tools that launch Kubernetes workloads on Slurm-allocated nodes). I'm not aware of a widely used open source project for it, and I'd treat it as experimental.


## Distributed launch internals: torchrun and torch.distributed.elastic

`torchrun` is a thin CLI over `torch.distributed.elastic`. A **launcher** is the program that starts your training processes on each node and gives each one its identity (rank, world size, where to find the others). The [launchers directory](slurm/launchers/) has working SLURM recipes. This section covers what happens under the hood. Everything here was checked against the PyTorch 2.15 source and, where possible, run on CPU with gloo.

### Process model

```
node k:  torchrun (elastic agent, 1 per node)
           ├── worker local_rank 0  (your train.py)
           ├── worker local_rank 1
           └── ... nproc_per_node workers
```

The agent is not a rank. It runs the rendezvous, spawns workers, monitors them, and restarts them. Workers never talk to the rendezvous directly. They get env vars and connect to the `MASTER_ADDR:MASTER_PORT` store.

Why a separate agent process rather than having the workers coordinate among themselves? Because a worker that segfaults, hits a CUDA error or gets OOM-killed can't report its own death. The agent is a small, boring process with no GPU state that survives its workers' crashes and can act on them.

You can see the tree with `ps` while a job runs **[ran]** (a 3-worker job running `sleeper.py`, which just sleeps; Python paths shortened):

```bash
torchrun --nnodes=1 --nproc-per-node=3 --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:29405 --local-addr=127.0.0.1 sleeper.py &
ps -ax -o pid,ppid,command | grep -E 'torchrun|sleeper'
```

```
57496     1 .../Python /tmp/mle-venv/bin/torchrun --nnodes=1 --nproc-per-node=3 ...
57515 57496 .../Python -u sleeper.py
57516 57496 .../Python -u sleeper.py
57517 57496 .../Python -u sleeper.py
```

The three workers' parent PID (57496) is the agent. This is also why `kill -9` on the agent alone can leave its workers running, and why Slurm's `--kill-on-bad-exit=1` and cgroup cleanup matter: they kill the whole process tree, not just the agent.

### c10d rendezvous

With `--rdzv-backend=c10d --rdzv-endpoint=host:29400`, all agents meet at a `TCPStore` (a small key-value server that PyTorch uses for coordination) hosted on `host:29400`. The store is hosted by the agent whose hostname matches the endpoint (so the endpoint should be one of the job's nodes; `SLURM_JOB_NODELIST`'s first host is the usual choice). The rendezvous is a small state machine kept as one key in the store, updated with compare-and-set (an atomic write that succeeds only if the value hasn't changed since you read it, so concurrent agents can't overwrite each other):

1. **Join.** Each agent adds itself to `participants`. Once `min_nodes` have joined, a `last_call_timeout` timer starts (default 30 s), during which more agents can still join, up to `max_nodes`.
2. **Close.** When `max_nodes` have joined, or `last_call_timeout` expires with at least `min_nodes`, the round closes. Agents arriving later go into `wait_list` for the next round.
3. **Assign ranks.** Each participant gets a **group rank** (node rank) from its position in the **sorted** participant list. In PyTorch 2.15 each participant is identified by `(address, pid, local_id)`, where address is the node's FQDN (fully qualified hostname) or `--local-addr`, and ranks are assigned by sorting those tuples (`dynamic_rendezvous.py`, `_mark_rendezvous_complete`). So the order is lexicographic by hostname, not arrival order, and not `SLURM_NODEID`: Slurm orders `gpu[9-10]` as `gpu9, gpu10`, but string sorting puts `gpu10` first. Don't assume `node_rank == SLURM_NODEID` under c10d. (`--node-rank` is only used by the static rendezvous; `torchrun --help` says so.)
4. **Heartbeats.** Each agent writes a keep-alive (`keep_alive_interval` default 5 s; after `keep_alive_max_attempt` default 3 missed beats, the node is considered dead).

If `min_nodes` never join within `join_timeout` (default 600 s), the rendezvous fails. That's the error you see when one node of an allocation never starts. These timeouts can be set with `--rdzv-conf join_timeout=900,last_call_timeout=60`. Here is that failure for real, a 2-node job where the second node never shows up, with the timeout shortened to 10 s **[ran]**:

```
$ torchrun --nnodes=2 --nproc-per-node=1 --rdzv-backend=c10d \
    --rdzv-endpoint=127.0.0.1:29406 --local-addr=127.0.0.1 --rdzv-id=lonely \
    --rdzv-conf join_timeout=10 show_env.py
...
  File ".../torch/distributed/elastic/rendezvous/dynamic_rendezvous.py", line 686, in run
    raise RendezvousTimeoutError
torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError
# exited with status 1 after 10 s (16:25:50 -> 16:26:00 in the log timestamps)
```

#### Building the same protocol yourself

The fastest way to understand the rendezvous is to write a small one. The script below implements join, last call, close, the wait list, sorted rank assignment, publishing a master address, and a barrier, all on PyTorch's `TCPStore` using only `set`, `get` (which blocks until a key exists), `check`, `add` (an atomic counter) and `compare_set`. Four "nodes" are separate OS processes on one machine that arrive at different times **[ran]**:

```python
# mini_rdzv.py -- a c10d-style rendezvous in ~60 lines on top of torch's TCPStore
import json, os, sys, time, multiprocessing as mp
from datetime import timedelta
import torch.distributed as dist

HOST, PORT = "127.0.0.1", 29510
MIN_NODES, MAX_NODES, LAST_CALL_S = 2, 4, 1.0

def cas_update(store, fn):
    """Read-modify-write the state key with compare-and-set until we win."""
    while True:
        cur = store.get("state") if store.check(["state"]) else b""
        state = json.loads(cur) if cur else {"round": 0, "participants": [],
                                             "closed": False, "wait_list": []}
        new = json.dumps(fn(state))
        # compare_set returns the value now stored; equal to `new` means we won
        if store.compare_set("state", cur.decode() if cur else "", new) == new.encode():
            return json.loads(new)

def agent(name, delay):
    time.sleep(delay)
    store = dist.TCPStore(HOST, PORT, is_master=False, timeout=timedelta(seconds=30))

    def join(s):
        if s["closed"]:
            s["wait_list"].append(name)        # too late: wait for the next round
        else:
            s["participants"].append(name)
            if len(s["participants"]) == MAX_NODES:
                s["closed"] = True
        return s
    s = cas_update(store, join)
    if name in s["wait_list"]:
        print(f"{name}: round closed, parked in wait_list", flush=True); return

    # last call: once MIN_NODES are in, wait LAST_CALL_S for stragglers, then close
    while True:
        s = json.loads(store.get("state"))
        if s["closed"]: break
        if len(s["participants"]) >= MIN_NODES:
            time.sleep(LAST_CALL_S)
            s = cas_update(store, lambda s: {**s, "closed": True})
            break
        time.sleep(0.05)

    ranks = {n: i for i, n in enumerate(sorted(s["participants"]))}  # sorted, not arrival order
    rank, world = ranks[name], len(ranks)
    if rank == 0:
        store.set("master", f"{name}:{29600}")   # rank 0 publishes MASTER_ADDR:PORT
    master = store.get("master").decode()        # get() blocks until the key exists

    # barrier: atomic counter, then wait for everyone
    store.add("arrived", 1)
    while store.add("arrived", 0) < world:       # add(k, 0) is an atomic read
        time.sleep(0.01)
    print(f"{name}: rank={rank} world={world} master={master} (passed barrier)", flush=True)

if __name__ == "__main__":
    server = dist.TCPStore(HOST, PORT, is_master=True, wait_for_workers=False)
    # arrival order: node-c first, node-a second, node-b third, node-d far too late
    plan = [("node-c", 0.0), ("node-a", 0.3), ("node-b", 0.6), ("node-d", 3.0)]
    procs = [mp.get_context("spawn").Process(target=agent, args=p) for p in plan]
    for p in procs: p.start()
    for p in procs: p.join()
    print("final state:", server.get("state").decode())
```

```
node-b: rank=1 world=3 master=node-a:29600 (passed barrier)
node-c: rank=2 world=3 master=node-a:29600 (passed barrier)
node-a: rank=0 world=3 master=node-a:29600 (passed barrier)
node-d: round closed, parked in wait_list
final state: {"round": 0, "participants": ["node-c", "node-a", "node-b"], "closed": true, "wait_list": ["node-d"]}
```

Reading the output against the protocol: `node-c` arrived first but got rank 2, because ranks come from sorting names. `MIN_NODES=2` was reached when `node-a` joined at 0.3 s, which started the 1 s last call. `node-b` joined inside that window at 0.6 s. The round closed with 3 of a possible 4, and `node-d`, arriving at 3 s, was parked for the next round. All three participants agreed on the same world size and master without any of them being special, because every decision went through compare-and-set on one key.

Why compare-and-set rather than "read, modify, write"? Two agents joining at the same moment would both read `participants=[a]`, both append themselves, and the second write would erase the first. `compare_set` makes the write conditional on the value being what you read, so the loser sees a mismatch and retries with the fresh state. This is the same pattern as an atomic CAS instruction or optimistic concurrency control in a database (see [concurrency](../foundations/concurrency.md#atomicity)).

What the toy leaves out compared with the real one: heartbeats and dead-node detection, multiple rounds (the real state has a round number and the wait list triggers a new round), a per-round barrier key (`arrived` here would be wrong in round 2), and the races around two agents both closing the round (harmless here because closing is idempotent).

### How the env vars are derived

After the rendezvous, each agent computes the following for its workers:

| Variable | Source |
|---|---|
| `GROUP_RANK` | The node's index from the rendezvous |
| `LOCAL_RANK` | 0..`nproc_per_node-1` |
| `LOCAL_WORLD_SIZE` | `nproc_per_node` |
| `RANK` | `sum(local_world_size of all nodes with lower group rank) + LOCAL_RANK`; with a uniform `nproc_per_node` this is `GROUP_RANK * nproc_per_node + LOCAL_RANK` |
| `WORLD_SIZE` | The sum of `local_world_size` across all participants in **this round** |
| `ROLE_RANK`, `ROLE_WORLD_SIZE` | The same, within a role (normally all workers share one role) |
| `MASTER_ADDR`, `MASTER_PORT` | Chosen by the group-rank-0 agent (its hostname plus a free port), published through the rendezvous store |
| `TORCHELASTIC_RESTART_COUNT` | Number of restarts so far; useful for logging and for skipping warm-up work on restart |
| `TORCHELASTIC_RUN_ID` | `--rdzv-id`; must be the same on all nodes of one job and different across jobs |

So `MASTER_ADDR` under c10d is **not** the rendezvous endpoint. It's whichever node won group rank 0. In recent versions, workers can reuse the agent's store instead of creating a second one (`TORCHELASTIC_USE_AGENT_STORE`). That detail has changed across versions and isn't something to rely on.

All of this can be seen on one laptop by faking two nodes: two torchrun agents with different `--local-addr` values that both resolve to this machine (`localhost` and `127.0.0.1`), 2 workers each. The `localhost` agent is started 2 seconds **first** **[ran]**:

```python
# show_env.py -- print what the torchrun agent handed this worker
import os, torch.distributed as dist
keys = ["GROUP_RANK", "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID"]
dist.init_process_group("gloo")             # reads RANK/WORLD_SIZE/MASTER_* from env
print(" ".join(f"{k}={os.environ.get(k)}" for k in keys), flush=True)
dist.barrier()
dist.destroy_process_group()
```

```bash
#!/bin/bash
# two_nodes.sh -- fake a 2-node job on one Mac: two torchrun agents, 2 workers each
export GLOO_SOCKET_IFNAME=lo0
R="torchrun --nnodes=2 --nproc-per-node=2 --rdzv-backend=c10d \
   --rdzv-endpoint=127.0.0.1:29400 --rdzv-id=job42"
$R --local-addr=localhost show_env.py > nodeB.log 2>&1 &   # starts FIRST
sleep 2
$R --local-addr=127.0.0.1 show_env.py > nodeA.log 2>&1     # starts second
wait
```

```
== nodeA.log (--local-addr=127.0.0.1, started second)
GROUP_RANK=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=49660 TORCHELASTIC_RESTART_COUNT=0 TORCHELASTIC_RUN_ID=job42
GROUP_RANK=0 LOCAL_RANK=1 RANK=1 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=49660 TORCHELASTIC_RESTART_COUNT=0 TORCHELASTIC_RUN_ID=job42
== nodeB.log (--local-addr=localhost, started first)
GROUP_RANK=1 LOCAL_RANK=0 RANK=2 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=49660 TORCHELASTIC_RESTART_COUNT=0 TORCHELASTIC_RUN_ID=job42
GROUP_RANK=1 LOCAL_RANK=1 RANK=3 WORLD_SIZE=4 LOCAL_WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=49660 TORCHELASTIC_RESTART_COUNT=0 TORCHELASTIC_RUN_ID=job42
```

Three things to read off this, each confirmed over three runs:

- The agent that started first got `GROUP_RANK=1`, because `"127.0.0.1" < "localhost"` as strings. Rank order is sort order.
- `RANK = GROUP_RANK * 2 + LOCAL_RANK`, as in the table.
- `MASTER_PORT=49660`, not the rendezvous port 29400: the rank-0 agent picked a free port and published it.

`WORLD_SIZE` can change between rounds when `min_nodes < max_nodes`. Code that hard-codes the global batch from `WORLD_SIZE` at launch time will silently change its effective batch size after an elastic resize. If you allow elastic sizes, derive gradient accumulation from the world size you actually got, every time you start (**[ran]** as plain arithmetic):

```python
# keep the global batch fixed across elastic resizes
GLOBAL_BATCH, MICRO_BATCH = 1024, 8
for world in (64, 56, 48):                       # e.g. after losing 1 and then 2 nodes of 8 GPUs
    accum, rem = divmod(GLOBAL_BATCH, MICRO_BATCH * world)
    print(world, "ranks -> grad accum", accum, "OK" if rem == 0 else f"(remainder {rem}: global batch changes!)")
```

```
64 ranks -> grad accum 2 OK
56 ranks -> grad accum 2 (remainder 128: global batch changes!)
48 ranks -> grad accum 2 (remainder 256: global batch changes!)
```

Only world sizes that divide the global batch keep it exact, which is one more reason most large runs use a fixed `--nnodes=N`.

### What happens on a worker failure

The agent's monitor loop polls the worker processes every `monitor_interval` seconds (default 0.1 s in recent versions, confirmed in the 2.15 `run.py`; it used to be 5 s):

1. A worker exits non-zero, for example after an NCCL watchdog timeout (a background thread that aborts the process when a collective takes longer than the timeout), a CUDA error or an OOM. The agent marks its local group as `FAILED`.
2. If `remaining_restarts > 0`, the agent **kills all its local workers** (SIGTERM, then SIGKILL after a grace period), decrements `remaining_restarts` and **re-enters the rendezvous** as a new round.
3. The other nodes' agents notice that `num_nodes_waiting > 0` (someone is in the wait list for a new round). They stop their healthy workers and rejoin **without** decrementing their own restart count, because a membership change is not counted as a failure.
4. A new round closes, ranks are reassigned, and all workers start again from `main()`. **Your code has to load the latest checkpoint.** torchrun restarts processes; it doesn't restore state.
5. When `remaining_restarts` reaches 0 on any agent, that agent exits with the failure, and its error file (if you used `@record`) contains the traceback of the first failing worker.

Here is steps 1-4 for real: a toy job that checkpoints every 2 steps and has rank 1 die at step 3 on the first attempt **[ran]**:

```python
# crashy_train.py -- a "training job" that checkpoints, crashes once, and resumes
import os, torch, torch.distributed as dist

CKPT = "ckpt.pt"
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
restart = int(os.environ["TORCHELASTIC_RESTART_COUNT"])
# Connect to the agent's TCPStore ourselves and namespace keys by attempt, so
# this attempt never reads keys left over by the crashed one (see text).
store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                      world, is_master=False)
store = dist.PrefixStore(f"attempt{restart}", store)
dist.init_process_group("gloo", store=store, rank=rank, world_size=world)

torch.manual_seed(0)
w = torch.zeros(4)                                   # (D,) toy "model"
step = 0
if os.path.exists(CKPT):                             # torchrun restarts processes, not state:
    s = torch.load(CKPT)                             # resuming is YOUR job
    w, step = s["w"], s["step"]
if rank == 0:
    print(f"[attempt {restart}] world={world} resuming at step {step}", flush=True)

while step < 6:
    g = torch.full((4,), float(rank + 1))            # (D,) fake gradient
    dist.all_reduce(g)                               # sum over ranks -> (D,)
    w -= 0.1 * g / world
    step += 1
    if rank == 0 and step % 2 == 0:                  # checkpoint every 2 steps
        torch.save({"w": w, "step": step}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)              # atomic rename: never a half-written file
        print(f"[attempt {restart}] saved step {step} w[0]={w[0]:.3f}", flush=True)
    dist.barrier()
    if restart == 0 and step == 3 and rank == 1:
        print(f"[attempt {restart}] rank 1 crashing at step {step}", flush=True)
        os._exit(1)                                  # simulate a CUDA error / OOM
if rank == 0:
    print(f"[attempt {restart}] done, step {step} w[0]={w[0]:.3f}", flush=True)
dist.destroy_process_group()
```

```
$ GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --rdzv-backend=c10d \
    --rdzv-endpoint=127.0.0.1:29404 --local-addr=127.0.0.1 \
    --nproc-per-node=2 --max-restarts=2 crashy_train.py
[attempt 0] world=2 resuming at step 0
[attempt 0] saved step 2 w[0]=-0.300
[attempt 0] rank 1 crashing at step 3
W1105 16:22:38.348000 55221 .../multiprocessing/api.py:1028] Sending process 55242 closing signal SIGTERM
E1105 16:22:38.357000 55221 .../multiprocessing/api.py:1002] failed (exitcode: 1) local_rank: 1 (pid: 55244) of binary: .../python3.14
[attempt 1] world=2 resuming at step 2
[attempt 1] saved step 4 w[0]=-0.600
[attempt 1] saved step 6 w[0]=-0.900
[attempt 1] done, step 6 w[0]=-0.900
$ echo $?
0
```

The agent saw rank 1 exit with code 1, sent SIGTERM to the healthy rank 0, started attempt 1, and both new workers resumed from the step-2 checkpoint. Step 3's work was lost and redone, which is the "rework" term in the [checkpoint arithmetic](#optimal-checkpoint-interval-youngdaly). The final `w[0] = -0.9` equals 6 steps of `0.1 * (1 + 2) / 2 = 0.15`, the same as a run that never crashed, so the resume is correct.

A finding from writing this, which I have not root-caused: with a plain `dist.init_process_group("gloo")` (letting it read the env vars), attempt 1 **hung** in `init_process_group` on this Mac with PyTorch 2.15. One rank got through init and the other never did, with both `--standalone` and an explicit c10d endpoint, three runs out of three. Namespacing the store keys per attempt with `PrefixStore`, as above, fixed it every time, which suggests the new attempt was reading keys left in the agent's store by the crashed attempt. It may be macOS- or version-specific. The lesson that does generalize: **test your restart path**, not just your training loop, before you depend on `--max-restarts`. On macOS, `--standalone` without `--local-addr=127.0.0.1` also failed to rendezvous at all (repeated IPv6 reverse-lookup warnings), which is a laptop quirk, not something you'll see on Linux clusters.

Practical consequences:

- **In-place restarts only help with software or transient failures.** If the failure is a dead GPU, the restarted worker fails again on the same GPU, and you burn through `--max-restarts`. For hardware failures the node has to go away, which means a scheduler-level requeue on a new node set (SLURM job array or `--requeue`, JobSet `failurePolicy`).
- **A node that dies outright** (kernel panic, which is a fatal OS crash, or a network partition, where the node is alive but unreachable) takes down its agent, so no error is reported. The other agents only find out through missed heartbeats in the rendezvous or, more commonly, through an NCCL timeout in their own workers. With `NCCL_TIMEOUT` / `TORCH_NCCL_*` timeouts at the default of 10 minutes, that's 10 minutes of lost work per failure before recovery even begins. `TORCH_NCCL_ASYNC_ERROR_HANDLING` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` control this. See [hanging and deadlocks](../debug/torch-distributed-hanging-solutions.md).
- **Scale-up.** A new node joining with `--nnodes=MIN:MAX` triggers a restart of everyone, at full restart cost. Elastic scale-up is seldom worth it for LLM training, because the restart costs as much as a failure. Most large runs use a fixed `--nnodes=N` and a scheduler-level requeue.
- PyTorch 2.14 adds experimental in-place process group reconfiguration. See [Fault Tolerance](../training/fault-tolerance/README.md#rebuilding-the-process-group-in-place).

The timeout is set per process group. Shortening it trades false positives (a slow checkpoint save declared a hang) for faster recovery; the [distributed systems chapter](../foundations/distributed-systems.md#heartbeats-timeouts-and-slow-vs-dead) explains why no value is right in every case. **[GPU]**, untested:

```python
import datetime, torch.distributed as dist
# 3 minutes instead of the default; any collective stuck longer aborts the process,
# turning a silent hang into a crash the launcher and scheduler can act on
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=3))
```


## Fault tolerance at scale

### Failure-rate arithmetic

**MTBF** (mean time between failures) is the average time a component runs before failing. For a synchronous job, any single component failure is a job failure. With `N` independent components that each have a mean time between failures of `MTBF_1`, the job's MTBF is roughly `MTBF_1 / N`. ([Distributed systems: failure arithmetic](../foundations/distributed-systems.md#failure-arithmetic) derives this from the exponential model; here we use it.)

The Llama 3 paper ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783), section 3.3.4) gives the best public numbers for a run at this scale:

- 16,384 H100 GPUs, over a 54-day snapshot of Llama 3 405B pre-training
- 466 job interruptions: 47 planned (firmware and kernel upgrades, config and dataset updates) and 419 unexpected
- About 78% of the unexpected interruptions were attributed to confirmed or suspected hardware issues; GPU issues were the largest category at 58.7%
- Top root causes in Table 5: faulty GPU 148, GPU HBM3 memory 72 (HBM is the high-bandwidth memory stacked next to the GPU die), software bug 54, network switch/cable 35, host maintenance 32, GPU SRAM 19, GPU system processor 17
- Effective training time above 90%; only 3 cases needed significant manual intervention

footnote: Table 5's counts sum to 419, but the percentage given for the first row (30.1%) doesn't match `148/419 = 35.3%`. I use the counts here.

Here is what that implies **[ran]**:

```python
hours = 54 * 24                    # 1296 h
unexpected = 419
print(unexpected / 54)             # 7.76 unexpected interruptions per day
mtbi_h = hours / unexpected        # 3.09 h between unexpected interruptions
print(mtbi_h)
# if we pretend every interruption is attributable to one of 16,384 GPUs (upper bound on per-GPU failure rate)
print(mtbi_h * 16384 / 8760)       # ~5.8 years implied MTBF per GPU "slot" (GPU + its share of host/network)
```

The same arithmetic run forwards: if each GPU together with its share of host and network has an MTBF of 50,000 hours, then

- a 16,384-GPU job has a job-level MTBF of `50000 / 16384 = 3.05 h`, which means `24 * 16384 / 50000 = 7.9` interruptions per day
- a 1,024-GPU job has a job-level MTBF of `50000 / 1024 = 48.8 h`, about one interruption every 2 days

The per-GPU number looks very reliable, and the job-level number is still a failure every few hours. At 16k GPUs, **restart cost** and **checkpoint cost** matter as much as model FLOPs utilization.

The same arithmetic applies to starting a job: if a node passes your pre-flight checks but still has a 0.5% chance of being bad, the probability that a 256-node job starts on all-good nodes is `0.995^256 = 0.28`. At 0.1% it is `0.999^256 = 0.77`, and for 2048 nodes `0.999^2048 = 0.13`. This is why health checks need to run **before** a large job starts, and why they have to find the bad node rather than just report that the job failed. **[ran]**:

```python
for p_bad, n in ((0.005, 256), (0.001, 256), (0.001, 2048)):
    print(f"P(all {n} nodes good | p_bad={p_bad}) = {(1 - p_bad) ** n:.2f}")
```

```
P(all 256 nodes good | p_bad=0.005) = 0.28
P(all 256 nodes good | p_bad=0.001) = 0.77
P(all 2048 nodes good | p_bad=0.001) = 0.13
```

### Optimal checkpoint interval (Young/Daly)

With a blocking checkpoint cost `C` (the time training is stalled per checkpoint) and job-level MTBF `M`, Young's first-order optimum for the interval between checkpoints is:

```
tau_opt = sqrt(2 * C * M)
```

Where it comes from: the time lost to checkpointing plus rework is about `C/tau + tau/(2M)` per unit time. The first term is the checkpoint stall paid once per interval. The second is rework: failures arrive at rate `1/M`, and on average each one lands halfway through an interval, so half an interval is redone. The first term falls with `tau` and the second rises, so setting the derivative `-C/tau^2 + 1/(2M)` to zero gives `tau = sqrt(2CM)`. At the optimum the two terms are equal, so the total is about `sqrt(2C/M)`. You also pay a fixed restart cost `R/M` per unit time (rescheduling, process startup, loading the checkpoint, NCCL init, warm-up), which no checkpoint interval reduces.

Worked example using the Llama 3 rate, `M = 3.09 h = 11,135 s`:

| Blocking checkpoint cost C | tau_opt = sqrt(2CM) | Checkpoint overhead C/tau | Expected rework tau/2M | Total |
|---|---|---|---|---|
| 300 s (sync write to slow storage) | 2585 s (43 min) | 11.6% | 11.6% | 23.2% |
| 60 s (sync write to a fast parallel filesystem, i.e. shared storage striped across many servers, such as GPFS or Lustre) | 1156 s (19 min) | 5.2% | 5.2% | 10.4% |
| 5 s (async: only the device-to-host copy blocks) | 334 s (5.6 min) | 1.5% | 1.5% | 3.0% |

Add the restart cost. With `R = 10 min`, `R/M = 600/11135 = 5.4%`. So even with async checkpointing, a 16k-GPU job at this failure rate loses about `3.0 + 5.4 = 8.4%` before counting stragglers or planned maintenance. That's consistent with Llama 3's ">90% effective training time", and it shows why Meta spent effort on reducing job startup time as well as checkpoint time.

Young's formula assumes `C << M`. Daly's higher-order correction, `tau = sqrt(2CM) * [1 + (1/3)sqrt(C/2M) + (1/9)(C/2M)] - C`, gives 2389 s instead of 2585 s for `C = 300`, and 1116 s instead of 1156 s for `C = 60`. The first-order formula is good enough when C is a minute or less. The total-loss curve is flat around the optimum, so being off by a factor of 1.5 costs little. The overhead depends on `sqrt(C)`, so the way to cut it is to reduce C.

A calculator that reproduces the table and extends it to other cluster sizes. `goodput` here means the fraction of wall-clock time spent on useful (not redone) training. `goodput_exact` is Daly's closed form for exponentially distributed failures, which also charges for failures that land during checkpoints and restarts **[ran]**:

```python
# ckpt_interval.py -- Young/Daly optimal checkpoint interval + goodput
import math

def young(C, M):
    return math.sqrt(2 * C * M)

def daly(C, M):
    # Daly 2006 higher-order estimate, valid for C < 2M
    x = C / (2 * M)
    return math.sqrt(2 * C * M) * (1 + math.sqrt(x) / 3 + x / 9) - C

def goodput_first_order(tau, C, M, R):
    # fraction of wall time that is useful work, first-order (C, tau, R << M)
    return 1 - C / tau - tau / (2 * M) - R / M

def goodput_exact(tau, C, M, R):
    # Daly's exact model: exponential failures, which can also hit during
    # checkpoints and restarts. wall = M e^{R/M} (e^{(tau+C)/M} - 1) per tau of work
    return tau / (M * math.exp(R / M) * (math.exp((tau + C) / M) - 1))

if __name__ == "__main__":
    M = 54 * 24 * 3600 / 419                     # Llama 3 405B: s between unexpected interrupts
    print(f"Llama-3 job MTBF M = {M:.0f} s = {M/3600:.2f} h")
    print(f"{'C (s)':>6} {'young':>7} {'daly':>7} {'C/tau':>6} {'tau/2M':>7} {'total':>6}")
    for C in (300, 60, 5):
        t = young(C, M)
        print(f"{C:>6} {t:>7.0f} {daly(C, M):>7.0f} {C/t:>6.1%} {t/(2*M):>7.1%} {C/t + t/(2*M):>6.1%}")

    print("\nper-GPU MTBF 50,000 h, R = 600 s restart, tau = Daly optimum")
    print(f"{'GPUs':>7} {'M (h)':>7} {'C (s)':>6} {'tau (min)':>9} {'goodput':>8} {'(first-order)':>13}")
    for gpus in (512, 1024, 4096, 16384, 65536, 131072):
        M = 50_000 * 3600 / gpus
        for C in (60, 5):
            t = daly(C, M)
            print(f"{gpus:>7} {M/3600:>7.2f} {C:>6} {t/60:>9.1f} "
                  f"{goodput_exact(t, C, M, 600):>8.1%} {goodput_first_order(t, C, M, 600):>13.1%}")
```

```
Llama-3 job MTBF M = 11135 s = 3.09 h
 C (s)   young    daly  C/tau  tau/2M  total
   300    2585    2389  11.6%   11.6%  23.2%
    60    1156    1116   5.2%    5.2%  10.4%
     5     334     330   1.5%    1.5%   3.0%

per-GPU MTBF 50,000 h, R = 600 s restart, tau = Daly optimum
   GPUs   M (h)  C (s) tau (min)  goodput (first-order)
    512   97.66     60     107.6    98.0%         98.0%
    512   97.66      5      31.2    99.3%         99.3%
   1024   48.83     60      75.9    97.1%         97.0%
   1024   48.83      5      22.0    98.9%         98.9%
   4096   12.21     60      37.6    93.6%         93.4%
   4096   12.21      5      11.0    97.2%         97.1%
  16384    3.05     60      18.5    85.1%         84.1%
  16384    3.05      5       5.5    91.9%         91.5%
  65536    0.76     60       8.9    64.7%         57.2%
  65536    0.76      5       2.7    75.6%         72.1%
 131072    0.38     60       6.1    47.3%         26.6%
 131072    0.38      5       1.9    59.2%         47.8%
```

Three things the bigger table shows:

- Up to a few thousand GPUs, the first-order and exact formulas agree to within a fraction of a percent, and goodput is above 93% even with a 60 s checkpoint.
- At 64k-128k GPUs with the same per-GPU reliability, the job MTBF drops below an hour, the first-order formula breaks down (it assumes `R, C << M`), and a 10-minute restart costs more than everything else combined: at 131k GPUs, even a 5 s checkpoint only gets you to 59%. At that scale the levers are `R` (fast restart, hot spares, in-memory checkpoints) and per-component MTBF (better burn-in), not the checkpoint interval.
- The 50,000 h per-GPU figure is an assumption chosen to match Llama 3's rate. If your fleet is better or worse, the table shifts left or right; rerun it with your own number.

### Simulating goodput

Formulas like Daly's rest on assumptions (exponential failures, failures independent across nodes, failures possible during restart). A Monte Carlo simulation, which estimates an average by running a random process many times, lets you check the formula and later change the assumptions it can't handle (correlated rack failures, a restart time that depends on spare availability). This one draws each node's failure time separately, so you can see that the job-level MTBF really is `node_MTBF / N`, and asserts that it agrees with the closed form **[ran]**:

```python
# goodput_sim.py -- Monte Carlo of a synchronous job on N nodes, checked against Daly's formula
import math, random
from ckpt_interval import daly, goodput_exact, goodput_first_order

def simulate(n_nodes, node_mtbf, tau, C, R, work, rng):
    """Return useful_work / wall_time for one simulated run.
    Any node failing kills the job (synchronous training). Each node fails as a
    Poisson process; the min of N exponentials is exponential with rate N/node_mtbf,
    but we draw per node to show where that comes from."""
    wall, done = 0.0, 0.0
    def next_failure():                       # time until the FIRST of N nodes fails
        return min(rng.expovariate(1 / node_mtbf) for _ in range(n_nodes))
    t_fail = next_failure()
    while done < work:
        seg = min(tau, work - done) + C       # compute, then a blocking checkpoint
        if t_fail >= seg:                     # segment survives
            wall += seg; t_fail -= seg; done += seg - C
            continue
        wall += t_fail                        # lose the partial segment
        while True:                           # restart; failures can hit restarts too
            t_fail = next_failure()
            if t_fail >= R:
                wall += R; t_fail -= R; break
            wall += t_fail
    return work / wall

if __name__ == "__main__":
    rng = random.Random(0)
    node_mtbf = 50_000 * 3600 / 8             # 8 GPUs/node, 50k h per GPU slot -> 6250 h/node
    C, R = 60, 600
    print(f"{'nodes':>6} {'M (h)':>6} {'tau (s)':>7} {'sim':>7} {'+-2se':>6} {'exact':>7} {'1st-ord':>7}")
    for n in (128, 512, 2048):
        M = node_mtbf / n
        tau = daly(C, M)
        work = 2000 * 3600                    # 2000 h of useful work per run
        runs = [simulate(n, node_mtbf, tau, C, R, work, rng) for _ in range(40)]
        mean = sum(runs) / len(runs)
        se = (sum((r - mean) ** 2 for r in runs) / (len(runs) - 1) / len(runs)) ** 0.5
        ex = goodput_exact(tau, C, M, R)
        print(f"{n:>6} {M/3600:>6.2f} {tau:>7.0f} {mean:>7.2%} {2*se:>6.2%} {ex:>7.2%} "
              f"{goodput_first_order(tau, C, M, R):>7.2%}")
        assert abs(mean - ex) < max(4 * se, 0.002), "simulation disagrees with Daly's model"

    # sweep tau at 2048 nodes: the optimum is flat
    n = 2048; M = node_mtbf / n
    print("\n2048 nodes, goodput vs interval (exact formula):")
    for mult in (0.25, 0.5, 1, 1.5, 2, 4):
        t = mult * daly(C, M)
        print(f"  tau = {mult:>4} x opt = {t:>5.0f} s -> {goodput_exact(t, C, M, R):.2%}")
```

```
 nodes  M (h) tau (s)     sim  +-2se   exact 1st-ord
   128  48.83    4553  97.06%  0.09%  97.08%  97.05%
   512  12.21    2257  93.54%  0.11%  93.58%  93.41%
  2048   3.05    1109  85.10%  0.10%  85.13%  84.08%

2048 nodes, goodput vs interval (exact formula):
  tau = 0.25 x opt =   277 s -> 76.65%
  tau =  0.5 x opt =   554 s -> 83.07%
  tau =    1 x opt =  1109 s -> 85.13%
  tau =  1.5 x opt =  1663 s -> 84.41%
  tau =    2 x opt =  2217 s -> 82.97%
  tau =    4 x opt =  4434 s -> 75.61%
```

The simulation (seed 0, 40 runs per row, 10 s on the laptop) matches the exact formula within its standard error at every size, and the first-order formula is about 1 point pessimistic at 2048 nodes. The sweep confirms the flatness claim: 1.5x the optimum costs 0.7 points of goodput, while 4x costs nearly 10. The point of having the simulator is the next step, which the formula can't do: make `next_failure()` correlated (for example, one rack PDU taking out 18 nodes at once) or make `R` depend on whether a hot spare was available, and see how much the answer moves.

### Async and in-memory checkpointing

What is `C` for a large model? Take a 405B model with bf16 weights, fp32 master weights and two fp32 Adam moments, which is `2 + 4 + 4 + 4 = 14` bytes per parameter **[ran]**:

```python
B = 405e9 * 14          # 5.67e12 bytes = 5.67 TB of checkpoint
print(B / 1e12)         # 5.67 s at an aggregate 1 TB/s write bandwidth
print(B / 100e9)        # 56.7 s at 100 GB/s
print(B / 16384 / 1e6)  # 346 MB per GPU when sharded across 16,384 GPUs
print(B / 16384 / 25e9 * 1e3)  # ~14 ms device-to-host per GPU at an assumed 25 GB/s effective PCIe
```

The storage write is seconds to minutes. The per-GPU device-to-host copy (GPU memory to CPU RAM over PCIe, into pinned host memory, which is memory locked in place so the GPU can copy into it directly) is milliseconds. **Async checkpointing** makes `C` equal to the copy time. The training step continues while a background thread or process writes the host copy to storage. PyTorch's `torch.distributed.checkpoint.async_save` (DCP) does this. The constraints are:

- Host memory: `346 MB * 8 = 2.8 GB` per 8-GPU node in this example, which is fine. For small-DP-degree setups where each GPU holds tens of GB of optimizer state, pinned host buffers can become the limit.
- The next checkpoint can't start until the previous upload has finished, so `tau` has to be longer than the time to write one checkpoint to storage.
- The write is not durable until the upload finishes. A failure during the upload means you fall back to the checkpoint before it. Keep at least 2 checkpoints (see [Frequent checkpoint saving](../training/fault-tolerance/README.md#frequent-checkpoint-saving)).

The difference between "blocked" and "total" is visible even on a laptop, with a single CPU process and 1.07 GB of tensors. The script also mutates the weights right after `async_save` returns, then checks that the checkpoint still holds the old values, which is the property that makes async saving safe: the snapshot is taken before control returns **[ran]**:

```python
# async_ckpt.py -- blocking vs async DCP save, CPU + gloo, single process
import os, time, shutil, torch, torch.distributed as dist
import torch.distributed.checkpoint as dcp

os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29520")
dist.init_process_group("gloo", rank=0, world_size=1)

# ~1.1 GB of fp32 "model + optimizer" state: 16 tensors of (4096, 4096)
state = {f"w{i}": torch.randn(4096, 4096) for i in range(16)}   # each (4096, 4096) fp32 = 64 MiB
gb = sum(t.numel() * t.element_size() for t in state.values()) / 1e9

shutil.rmtree("ck_sync", ignore_errors=True); shutil.rmtree("ck_async", ignore_errors=True)
t0 = time.perf_counter()
dcp.save(state, checkpoint_id="ck_sync")
t_sync = time.perf_counter() - t0

t0 = time.perf_counter()
fut = dcp.async_save(state, checkpoint_id="ck_async")   # returns after staging the copy
t_block = time.perf_counter() - t0
for t in state.values(): t.mul_(0.0)                    # "training continues" and mutates weights
fut.result()                                            # wait for the background write
t_total = time.perf_counter() - t0

# the async checkpoint must hold the values from BEFORE we mutated them
loaded = {k: torch.empty_like(v) for k, v in state.items()}
dcp.load(loaded, checkpoint_id="ck_async")
print(f"state {gb:.2f} GB | sync save {t_sync:.2f}s | async: blocked {t_block:.2f}s, "
      f"total {t_total:.2f}s | snapshot intact: {bool(loaded['w0'].abs().sum() > 0)}")
dist.destroy_process_group()
```

Three runs:

```
state 1.07 GB | sync save 2.39s | async: blocked 0.07s, total 2.34s | snapshot intact: True
state 1.07 GB | sync save 1.08s | async: blocked 0.10s, total 1.45s | snapshot intact: True
state 1.07 GB | sync save 0.84s | async: blocked 0.11s, total 1.95s | snapshot intact: True
```

The blocking part is 0.07-0.11 s (a memory copy at roughly 10-15 GB/s), against 0.8-2.4 s for the synchronous save to the local SSD. The run-to-run spread in the sync numbers is filesystem caching, which is exactly why you measure `C` on the real cluster storage rather than trusting a laptop. On GPUs the staging copy goes over PCIe instead of RAM-to-RAM, which is the 14 ms per GPU estimated above.

**In-memory / peer checkpointing** goes a step further: each rank also stores its shard in the host RAM of a **different** node (a buddy in another failure domain, i.e. a set of components that can fail together, such as one rack or one power feed). If a node dies, its replacement fetches the shard from the buddy over the network, rather than everyone reading from the parallel filesystem. The restart then loads at network speed. Gemini (SOSP'23) and several in-house systems describe this approach. Persistent checkpoints still get written to storage, less often. This only helps if your restart path can use a peer copy, and if a replacement node joins quickly, which means hot spares.

The buddy assignment itself is simple, and the one rule that matters is "never in the same failure domain" (**[ran]** as plain Python):

```python
# buddy.py -- pair each node with a buddy in a DIFFERENT rack
nodes = [f"r{r}n{i}" for r in range(4) for i in range(4)]   # 4 racks x 4 nodes
rack = lambda n: n.split("n")[0]
# shift by one rack's worth of nodes: node i backs up to node i+4 (mod N)
buddy = {n: nodes[(i + 4) % len(nodes)] for i, n in enumerate(nodes)}
assert all(rack(n) != rack(b) for n, b in buddy.items())
print({k: buddy[k] for k in list(buddy)[:5]})
```

```
{'r0n0': 'r1n0', 'r0n1': 'r1n1', 'r0n2': 'r1n2', 'r0n3': 'r1n3', 'r1n0': 'r2n0'}
```

### Hot spares

A spare is only useful if it is **already healthy, imaged and inside the allocation**. If you have to wait for the scheduler to get one, it isn't hot. How many do you need? Suppose a node failure takes `T_repair` to fix (reboot, diagnose, RMA, i.e. return to the vendor), and nodes fail independently, each at rate `1/MTBF_node`. Then the number of nodes out of service at any time is roughly Poisson-distributed (the distribution of the number of independent random events in a fixed time) with mean `N * T_repair / MTBF_node`.

Example: 2048 nodes, an 8-GPU node MTBF of `50000/8 = 6250 h`, and a 24-hour turnaround from failure to returned to service **[ran]**:

```python
import math
lam = 2048 * 24 / 6250       # 7.86 nodes out of service on average
p, k = 0.0, 0
while p < 0.99:              # p99 of Poisson(lam)
    p += math.exp(-lam) * lam**k / math.factorial(k); k += 1
print(lam, k - 1)            # 7.86, 15
```

So 16 spares (0.8%) cover the 99th percentile of concurrent failures, under the independence assumption. Correlated failures, such as a rack PDU (power distribution unit), a leaf switch or a bad firmware rollout, break that assumption, which is why the [5-10% advice in Fault Tolerance](../training/fault-tolerance/README.md#always-plan-to-have-more-nodes-than-needed) is reasonable for new hardware even though the Poisson calculation gives a lower number. For topology-constrained jobs, spares have to be **inside the same block**. One spare tray per NVL72 rack is a very different cost from 16 spare nodes anywhere in the cluster.

In Slurm, the simplest form of a hot spare is to allocate more nodes than the job uses and pick the healthy subset at launch. A sketch (**untested**; `node_ok.sh` stands for your own fast health check, as in the next section):

```bash
#SBATCH --nodes=66          # 64 needed + 2 spares inside the same allocation
NEED=64
# run the quick check on every node; keep the first 64 that pass
GOOD=$(srun --ntasks-per-node=1 bash -c './node_ok.sh && hostname -s' 2>/dev/null | sort | head -n $NEED)
[ "$(echo "$GOOD" | wc -l)" -eq "$NEED" ] || { echo "not enough healthy nodes"; exit 1; }
NODELIST=$(echo "$GOOD" | paste -sd,)
srun --nodes=$NEED --nodelist="$NODELIST" --ntasks-per-node=1 bash -c '... torchrun ...'
```

The spares cost 2/66 = 3% of the allocation, which buys a restart that doesn't go back through the queue.

### Health checks before the job starts (DCGM diag levels)

DCGM (NVIDIA Data Center GPU Manager) is NVIDIA's daemon and tool set for GPU health and telemetry, and `dcgmi` is its CLI. `dcgmi diag -r N` runs increasingly long test suites. The [NVIDIA debug chapter](../compute/accelerator/nvidia/debug.md#running-diagnostics) has the commands and sample output. Durations vary by GPU and DCGM version. These are rough figures:

| Level | Runtime (approx.) | Covers | Where to run it |
|---|---|---|---|
| `-r 1` | Seconds | Software and deployment checks: driver, persistence mode, NVML, permissions | Every job prolog |
| `-r 2` | A few minutes | Plus a short PCIe/NVLink bandwidth test, memory and short stress tests | Epilog, or a prolog for large jobs; the repo author found it catches most faulty GPUs |
| `-r 3` | 10-30 min | Plus longer stress tests: targeted power and stress, memory bandwidth | After a reboot or repair, before the node rejoins the pool |
| `-r 4` | 1+ h | Plus extended memory tests and more | Suspected silent data corruption; before RMA |

A practical layout:

- **Prolog (fast, every job):** GPU count (`nvidia-smi -L | wc -l`), `-r 1`, no pending row remaps, no new Xids (error codes the NVIDIA driver writes to the kernel log) since boot, IB ports `Active` at the expected rate, required filesystems mounted. Keep it under about 30 s, because it runs on every node of every job.
- **Epilog:** `-r 2`, a check for leftover GPU processes, and checks that GPU memory is free and that `/dev/shm` has been cleaned up. A failing epilog drains the node before the next job lands on it.
- **Pre-flight for big jobs:** a 2-node NCCL all-reduce between neighbours, run in a tree across the whole allocation (pairs, then quads), plus a short GEMM benchmark per GPU compared against the fleet median. This catches the slow-but-working GPU that DCGM passes.

A prolog along those lines (**untested**; Linux with NVIDIA drivers and `infiniband-diags` installed; the expected counts and paths are placeholders). In Slurm a non-zero exit from the prolog drains the node and requeues the job:

```bash
#!/bin/bash
# prolog.sh -- fast node check, run by slurmd before every job. Exit non-zero = drain.
fail() { echo "prolog: $*" >&2; exit 1; }

EXPECT_GPUS=8
EXPECT_IB_PORTS=8

n=$(nvidia-smi -L | wc -l)
[ "$n" -eq "$EXPECT_GPUS" ] || fail "gpu_count:$n"

dcgmi diag -r 1 > /tmp/dcgm_r1.log 2>&1 || fail "dcgm_r1"

# rows waiting to be remapped mean the GPU needs a reset before it is trustworthy
pending=$(nvidia-smi --query-remapped-rows=remapped_rows.pending --format=csv,noheader | grep -c -i yes)
[ "$pending" -eq 0 ] || fail "row_remap_pending:$pending"

# new Xid errors since boot
dmesg | grep -q 'NVRM: Xid' && fail "xid_in_dmesg"

active=$(ibstat | grep -c 'State: Active')
[ "$active" -ge "$EXPECT_IB_PORTS" ] || fail "ib_active:$active"

mountpoint -q /shared || fail "fs_missing:/shared"
exit 0
```

In Slurm these are `Prolog=`, `Epilog=` and `HealthCheckProgram=` (run every `HealthCheckInterval` seconds by slurmd; [LBNL NHC](https://github.com/mej/nhc) is the common framework). On k8s the equivalents are a GPU Operator validator, the DCGM health check, node problem detectors, and init containers in the training pod. Cloud providers increasingly run their own checks. Ask yours what they run between tenants.

```ini
# slurm.conf (untested excerpt)
Prolog=/etc/slurm/prolog.sh
Epilog=/etc/slurm/epilog.sh
HealthCheckProgram=/usr/sbin/nhc
HealthCheckInterval=300        # seconds
```

### Automatic node draining

The loop that ties this together:

```
health signal (prolog fail / epilog fail / NHC / Xid / DCGM health watch / NCCL timeout attributed to host)
    -> drain the node with a machine-parseable reason:
         scontrol update nodename=gpu123 state=drain reason="xid79:2026-11-05T03:12"
         kubectl cordon gpu123 && kubectl label node gpu123 health=failed
    -> requeue the job onto the remaining nodes plus spares
    -> repair workflow: reboot -> dcgmi diag -r 3 -> pass: resume / fail: RMA
```

Some design points:

- **Drain, don't down.** A drained node lets running jobs finish. The node is just no longer scheduled.
- **The reason string is data.** Encode the failure class and a timestamp so you can count recurrences per GPU serial (see [detecting the same broken node](../compute/accelerator/nvidia/debug.md#how-to-detect-if-you-get-the-same-broken-node-again-and-again)). Undraining automatically after a reboot without a diag pass brings flaky nodes back into the pool. [undrain-good-nodes.sh](slurm/undrain-good-nodes.sh) is the narrow, safe version of this.
- **Attributing an NCCL timeout is hard.** Every rank times out, and only one node is at fault. The NCCL flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`, with dumps on timeout) shows which rank never entered the collective. Llama 3 used it extensively. Automate the analysis that picks the suspect, or you'll drain the wrong node or all of them.
- **Rate-limit draining.** A bug in a health check or a shared-storage hiccup can drain 30% of the cluster in a minute. Cap automatic drains per time window and page a human above that cap.

The rate limit is a sliding-window counter. Here a shared-filesystem hiccup makes five nodes fail their health check within 15 seconds, and the budget stops the automation after three **[ran]**:

```python
# drain_limiter.py -- auto-drain with a budget: above the cap, stop and page a human
import time
from collections import deque

class DrainBudget:
    def __init__(self, max_drains, window_s):
        self.max, self.window, self.events = max_drains, window_s, deque()
    def allow(self, now):
        while self.events and now - self.events[0] > self.window:
            self.events.popleft()                       # forget drains outside the window
        if len(self.events) >= self.max:
            return False
        self.events.append(now); return True

def drain(node, reason, budget, now):
    if budget.allow(now):
        # real version: subprocess.run(["scontrol", "update", f"nodename={node}",
        #                              "state=drain", f"reason={reason}"], check=True)
        print(f"t={now:>4}s drain {node} reason={reason}")
    else:
        print(f"t={now:>4}s REFUSED {node}: budget exhausted -> page on-call")

b = DrainBudget(max_drains=3, window_s=600)            # at most 3 auto-drains per 10 min
# a storage hiccup makes the health check fail on many nodes at once
for i, t in enumerate([0, 5, 9, 12, 15, 700]):
    drain(f"gpu{100 + i}", f"nhc_fs_missing:{t}", b, t)
```

```
t=   0s drain gpu100 reason=nhc_fs_missing:0
t=   5s drain gpu101 reason=nhc_fs_missing:5
t=   9s drain gpu102 reason=nhc_fs_missing:9
t=  12s REFUSED gpu103: budget exhausted -> page on-call
t=  15s REFUSED gpu104: budget exhausted -> page on-call
t= 700s drain gpu105 reason=nhc_fs_missing:700
```

A real version would also keep the budget's state somewhere that survives the drainer restarting, and would treat "many nodes failing the same check at once" as evidence that the check, not the nodes, is broken.


## Inference serving orchestration

Training is scheduled as one huge long-lived gang. Inference is many replicas, each possibly multi-node, whose number tracks demand. Orchestration problems that don't matter for training become the main problems here. ([Serving engine internals](../inference/serving-engine-internals.md) covers what happens inside one replica; [serving hands-on](../inference/serving-hands-on.md) runs one.)

### Autoscale on queue depth and KV-cache use, not CPU

An **autoscaler** changes the number of replicas in response to a metric. An LLM server's CPU is mostly idle. GPU "utilization" from NVML (NVIDIA's management library) is the fraction of time any kernel (a function launched on the GPU) ran, which reads close to 100% at batch size 1 and at batch size 256 alike, so it says nothing about headroom (see [real GPU utilization](../compute/accelerator/nvidia/debug.md#how-to-get-the-real-gpu-utilization-metrics)). The signals that do track saturation in a continuous-batching server (one that adds and removes requests from the running batch at every decoding step):

| Signal | vLLM metric name (changes across versions; check `/metrics`) | Meaning |
|---|---|---|
| Waiting queue length | `vllm:num_requests_waiting` | Requests admitted but not yet scheduled; above 0 for long means you're at capacity |
| KV-cache use | `vllm:gpu_cache_usage_perc` (renamed `vllm:kv_cache_usage_perc` in newer versions, I believe) | Near 1.0 leads to preemption and recompute of sequences, and TTFT spikes |
| Running batch size | `vllm:num_requests_running` | Concurrency actually achieved |
| TTFT / TPOT histograms (time to first token, time per output token) | `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds` | The SLO (service-level objective, the latency target you've committed to) itself |

Check what your server version actually exports before writing any rule against it (**untested**; sketched output):

```
$ curl -s localhost:8000/metrics | grep -E '^vllm:(num_requests_(waiting|running)|(gpu|kv)_cache_usage_perc)'
vllm:num_requests_running{model_name="..."} 37.0
vllm:num_requests_waiting{model_name="..."} 4.0
vllm:kv_cache_usage_perc{model_name="..."} 0.83
```

Use a scaler that can read Prometheus (the standard time-series metrics system on k8s) metrics, such as [KEDA](https://keda.sh/) or the HPA with a custom metrics adapter, targeting something like "average waiting requests per replica < 2" or "KV-cache use < 0.8". The exact thresholds depend on your traffic. Load-test to find the knee of the latency curve; don't copy numbers from elsewhere. Scale-down needs a long stabilization window, because a cold start costs far more than a few minutes of an idle replica.

A KEDA `ScaledObject` for that rule (**untested**; the Prometheus address, the Deployment name and the threshold are placeholders):

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-scaler
spec:
  scaleTargetRef:
    name: llm-server                 # the Deployment (or other scalable object) to scale
  minReplicaCount: 2                 # never scale to zero: a cold start costs minutes
  maxReplicaCount: 40
  cooldownPeriod: 900                # seconds; slow, deliberate scale-down
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: sum(vllm:num_requests_waiting{service="llm-server"})
      threshold: "2"                 # target ~2 waiting requests per replica
```

Routing matters as much as replica count. Round-robin load balancing ignores prefix-cache locality and each replica's KV load. The Kubernetes [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) (an endpoint picker that routes on queue depth, KV use and LoRA adapter residency), [llm-d](https://github.com/llm-d/llm-d) and NVIDIA Dynamo implement KV- and prefix-aware routing. All three were early-stage projects at the time of writing, so check their maturity yourself.

### Cold-start budget arithmetic

A **cold start** is the time from "the autoscaler asked for a replica" to "the replica serves its first request". Scale-up latency is a sum of stages that run one after another. Here is an example for a 70B model in bf16 (`70e9 * 2 = 140 GB` of weights), TP=8 on one node:

| Stage | Assumption | Time |
|---|---|---|
| Node provisioning (if the cluster autoscaler has to add a node) | Cloud VM plus GPU driver and operator readiness | Minutes. Highly variable, and the main reason to keep warm capacity |
| Image pull | 10 GB image at 1 GB/s (registry or local mirror) | 10 s; avoid this with pre-pulled or lazily loaded images |
| Weight load from object storage | 1 GB/s per node | `140 / 1 = 140 s` |
| Weight load from a shared parallel filesystem | 2 GB/s per client | `140 / 2 = 70 s` |
| Weight load from local NVMe | ~25 GB/s (a 4-drive Gen4 array, assumed) | `140 / 25 = 5.6 s` |
| Engine init: CUDA graph capture (recording a sequence of kernel launches so it can be replayed cheaply), `torch.compile`, KV allocation | Depends on engine and flags | Seconds to minutes; measure it |

The arithmetic is just `bytes / min(storage bandwidth, NIC bandwidth, PCIe bandwidth)`, done per node, and you have to measure the real bandwidth. To turn it into a capacity requirement: if load can grow by `g` fraction within one cold-start time `T_cold`, keep at least `g` spare capacity warm. If traffic can rise 10% in 5 minutes and cold start is 5 minutes, you need about 10% headroom. If cold start is 30 s, you need much less. Cutting cold start lets you run at higher utilization.

As a script you can rerun with your own measured bandwidths **[ran]**:

```python
# coldstart.py -- scale-up latency is a sum of serial stages; weights dominate
params, bytes_per = 70e9, 2                     # 70B in bf16
weights_gb = params * bytes_per / 1e9           # 140 GB
image_gb, registry_gbps = 10, 1.0
for src, gbps in (("object store", 1.0), ("parallel FS", 2.0), ("local NVMe", 25.0)):
    t = image_gb / registry_gbps + weights_gb / gbps
    print(f"{src:12s}: image {image_gb/registry_gbps:.0f}s + weights {weights_gb/gbps:6.1f}s = {t:6.1f}s (+ engine init)")

# fleet-wide: N replicas pulling at once share the storage egress
replicas, egress_gbps = 200, 50
print(f"{replicas} replicas x {weights_gb:.0f} GB = {replicas*weights_gb/1e3:.0f} TB "
      f"-> {replicas*weights_gb/egress_gbps:.0f}s at {egress_gbps} GB/s aggregate")

# peer-to-peer doubling: every node that has the weights sends a full copy to one new node
import math
nic_gbps = 25                                   # ~200 Gb/s NIC per node, assumed
rounds = math.ceil(math.log2(replicas + 1))     # 1 -> 2 -> 4 -> ... seeded from storage once
print(f"P2P doubling: {rounds} rounds x {weights_gb/nic_gbps:.1f}s = {rounds*weights_gb/nic_gbps:.0f}s "
      f"(+ {weights_gb/min(nic_gbps, egress_gbps):.1f}s for the first copy from storage)")
```

```
object store: image 10s + weights  140.0s =  150.0s (+ engine init)
parallel FS : image 10s + weights   70.0s =   80.0s (+ engine init)
local NVMe  : image 10s + weights    5.6s =   15.6s (+ engine init)
200 replicas x 140 GB = 28 TB -> 560s at 50 GB/s aggregate
P2P doubling: 8 rounds x 5.6s = 45s (+ 5.6s for the first copy from storage)
```

The [model loading section of the inference chapter](../inference/README.md#speeding-up-model-loading-time) covers engine-side techniques: pre-sharded weights, tensorizer, and loading directly to GPU.

### Model weight distribution

Scaling out by many replicas at once multiplies the storage load: `200 replicas * 140 GB = 28 TB`. At 50 GB/s of aggregate storage egress that takes `28000 / 50 = 560 s`, no matter how fast each node is (the last block above). Some options:

- **Node-local cache.** A DaemonSet or init container that pre-stages the weights for the current model version onto local NVMe (hostPath or a local PV). New replicas on a warm node load at NVMe speed.
- **Peer-to-peer fan-out.** Each node that already has the weights serves them to others, BitTorrent-style or as a tree. Total time then grows with `log(N)` rather than `N`: 8 doubling rounds, about 45 s, for the 200 replicas above. Container-registry P2P tools (e.g. Dragonfly, Spegel) and OCI-packaged weights (OCI is the standard container image format) make this usable on k8s. I haven't benchmarked them.
- **Image volumes / OCI artifacts.** Weights shipped as OCI artifacts and mounted with k8s image volumes (alpha in 1.31, beta later; I haven't verified the current status). This gets registry caching and P2P for free.
- **Don't bake weights into the application image.** A 140 GB layer makes every code change a 140 GB pull.

The node-local cache as an init container that copies weights once per node and skips the copy if the version is already there (**untested**; the copy tool, paths and image are placeholders):

```yaml
spec:
  initContainers:
  - name: stage-weights
    image: my-tools:2026.11
    command: ["bash", "-c"]
    args:
    - |
      set -e
      V=llama-70b-v3
      if [ -f /cache/$V/.complete ]; then echo "cache hit"; exit 0; fi
      mkdir -p /cache/$V.tmp
      my-copy-tool s3://models/$V/ /cache/$V.tmp/      # your object-store client here
      mv /cache/$V.tmp /cache/$V && touch /cache/$V/.complete   # rename = atomic "done"
    volumeMounts:
    - name: model-cache
      mountPath: /cache
  containers:
  - name: server
    image: my-server:2026.11
    volumeMounts:
    - name: model-cache
      mountPath: /models
      readOnly: true
  volumes:
  - name: model-cache
    hostPath:
      path: /mnt/nvme/model-cache
      type: DirectoryOrCreate
```

The `.complete` marker and the rename matter: two replicas on the same node can start at once, and a server must never load a half-copied directory.

### Multi-node inference replicas

When one replica spans nodes (for example a 600B+ MoE with wide expert parallelism, or disaggregated prefill and decode pools), the replica has the same gang and topology requirements as a training job, inside a service that autoscales:

- Use **LeaderWorkerSet** (or KAI/Grove hierarchical PodGroups for disaggregated setups) so the group is scheduled, restarted and rolled together.
- Keep a replica inside one NVLink domain if TP or EP spans the nodes. Express that with TAS required topology on the group.
- Any member failing makes the replica unusable. Whole-group restart plus enough replicas for N+1 redundancy is the simple, correct approach.
- Autoscaling works in units of whole groups. A 4-node replica means you scale in 32-GPU steps, and cold start includes every member's weight load plus NCCL init across the group.

An LWS with 2 replicas of 2 nodes each (**untested**; the server command is a placeholder, since each engine has its own multi-node launch flags):

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: big-moe
spec:
  replicas: 2                              # 2 independent serving replicas...
  leaderWorkerTemplate:
    size: 2                                # ...each 1 leader + 1 worker = 2 nodes, 16 GPUs
    restartPolicy: RecreateGroupOnPodRestart   # one pod dies -> the whole group restarts
    leaderTemplate:
      spec:
        containers:
        - name: server
          image: my-server:2026.11
          command: ["bash", "-c", "serve --role=leader --group-size=$LWS_GROUP_SIZE"]
          resources:
            limits:
              nvidia.com/gpu: 8
    workerTemplate:
      spec:
        containers:
        - name: server
          image: my-server:2026.11
          command: ["bash", "-c", "serve --role=worker --leader=$LWS_LEADER_ADDRESS"]
          resources:
            limits:
              nvidia.com/gpu: 8
```


## Observability

**Observability** means being able to answer questions about a running system from the data it emits (metrics, logs, traces) without adding new code first. The aim is to tell, within minutes, whether a job is **dead**, **hung**, **slow**, or **healthy but slow because of the code**, and if it's slow, which node is responsible.

### GPU signals (DCGM exporter)

The [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) field list is configurable through a CSV. These fields are worth enabling and alerting on:

| Field | Why |
|---|---|
| `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Real compute use, unlike `DCGM_FI_DEV_GPU_UTIL`. A per-rank drop shows a straggler or a rank stuck in a collective |
| `DCGM_FI_DEV_XID_ERRORS` | The most recent Xid; any change should page or drain (see [Xid errors](../compute/accelerator/nvidia/debug.md#xid-errors)) |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` (ECC double-bit errors: memory errors that error-correcting codes detect but can't fix), `DCGM_FI_DEV_ROW_REMAP_PENDING`, `DCGM_FI_DEV_ROW_REMAP_FAILURE` | Memory health: drain on pending remaps, RMA on a remap failure |
| `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` (`DCGM_FI_DEV_CLOCKS_EVENT_REASONS` in newer DCGM) | Thermal or power throttling: the most common cause of a "working but slow" GPU |
| `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_MEMORY_TEMP` | Context for throttling; the Llama 3 paper reports a 1-2% diurnal throughput swing from temperature affecting DVFS (dynamic voltage and frequency scaling, where the GPU lowers its clock when hot or power-limited) |
| `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` (and the replay/recovery counters) | NVLink link health |
| `DCGM_FI_PROF_NVLINK_TX_BYTES`, `DCGM_FI_PROF_PCIE_TX_BYTES`/`RX_BYTES` | Traffic patterns; a rank with clearly different traffic is suspicious |
| `DCGM_FI_DEV_FB_USED` | Memory leaks across steps; leftover allocations after a job ends |

Field names drift between DCGM versions, so check `dcgmi dmon -l` against the exporter's CSV after every upgrade. The CSV has one line per field: the DCGM field name, the Prometheus metric type, and a help string (**untested** excerpt):

```
# custom-counters.csv for dcgm-exporter
# Format: DCGM FIELD, Prometheus metric type, help message
DCGM_FI_PROF_SM_ACTIVE,            gauge,   Ratio of cycles an SM has at least one warp assigned.
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,   gauge,   Ratio of cycles the tensor pipe is active.
DCGM_FI_DEV_XID_ERRORS,            gauge,   Value of the last XID error encountered.
DCGM_FI_DEV_ROW_REMAP_PENDING,     gauge,   Pending row remaps.
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, gauge,  Throttling reasons bitmask.
DCGM_FI_DEV_SM_CLOCK,              gauge,   SM clock frequency (in MHz).
DCGM_FI_DEV_FB_USED,               gauge,   Framebuffer memory used (in MiB).
```

### Network signals

For InfiniBand and RoCE, per-port counters in sysfs (the kernel's virtual filesystem of device information; `/sys/class/infiniband/<dev>/ports/<port>/counters/` and `.../hw_counters/`):

| Counter | Meaning |
|---|---|
| `port_xmit_data`, `port_rcv_data` | Throughput (in units of 4 bytes, so multiply by 4) |
| `symbol_error`, `link_error_recovery`, `link_downed` | Physical layer: bad cable or transceiver; any increase during a job is suspicious |
| `port_rcv_errors`, `port_xmit_discards` | Corrupt or dropped packets |
| `port_xmit_wait` | Time spent waiting for credits, a sign of congestion |
| RoCE `hw_counters` (ECN marks congested packets and DCQCN is the RoCE congestion-control protocol that reacts to them by sending CNP packets): `np_cnp_sent`, `rp_cnp_handled`, `out_of_sequence`, `packet_seq_err` | ECN/DCQCN activity and loss; a high out-of-sequence count means drops |

Counters are cumulative, so what you alert on is the **delta** between two samples. A small reader that samples twice and reports rates and error increments. It was run against a fake sysfs tree (two NICs, one with traffic and one with 7 new symbol errors) because macOS has no InfiniBand; on Linux, run it with no arguments **[ran on a fake tree]**:

```python
# ib_counters.py -- sample IB port counters twice, print rates and any error increments
import glob, os, sys, time
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/sys/class/infiniband"
ERRS = ["symbol_error", "link_error_recovery", "link_downed", "port_rcv_errors", "port_xmit_discards"]

def snap():
    out = {}
    for d in glob.glob(f"{ROOT}/*/ports/*/counters"):
        port = d.split("/")[-4] + "/" + d.split("/")[-2]           # e.g. mlx5_0/1
        out[port] = {f: int(open(os.path.join(d, f)).read()) for f in os.listdir(d)}
    return out

INTERVAL = float(os.environ.get("INTERVAL", 1.0))
a = snap(); time.sleep(INTERVAL); b = snap()
for port in sorted(a):
    d = {k: b[port][k] - a[port][k] for k in a[port]}
    # port_xmit_data / port_rcv_data count 4-byte words
    tx, rx = d["port_xmit_data"] * 4 / INTERVAL / 1e9, d["port_rcv_data"] * 4 / INTERVAL / 1e9
    bad = {k: d[k] for k in ERRS if d.get(k)}
    print(f"{port}: tx {tx:6.2f} GB/s rx {rx:6.2f} GB/s {'ERRORS ' + str(bad) if bad else 'ok'}")
```

```
mlx5_0/1: tx  20.00 GB/s rx  19.60 GB/s ok
mlx5_1/1: tx   0.00 GB/s rx   0.00 GB/s ERRORS {'symbol_error': 7}
```

One caveat for real hardware: on some NICs the classic 32-bit counters saturate, and the 64-bit values live in `hw_counters` or need `perfquery -x`; check which your driver exposes.

Export these with node-exporter's infiniband collector or a vendor tool (UFM, or the Network Operator's exporters), and correlate with the switch side: a port's error counters at the switch point to the cable, not the host. The [network debug chapter](../network/debug/README.md) has the debugging workflow. NCCL itself can emit per-collective timings through its profiler plugin interface and through the PyTorch flight recorder. The flight recorder is cheap enough to leave on for large runs.

### Job-level signals

Per rank, from the training loop (not only on rank 0):

- step time, split into data wait, forward/backward compute, and time blocked in collectives (CUDA events, which are GPU-side timestamps, placed around the comms, or the profiler for sampled steps)
- tokens/s and MFU, computed rather than estimated
- checkpoint save duration (the blocking part and the total), and time from job start to first step, which is your restart cost `R`
- heartbeat: a per-rank "last step completed" timestamp written to the metrics system. Alert on "no progress for k × median step time", which catches hangs that never produce an error (see the [is-job-hanging watchdog](../training/fault-tolerance/README.md#is-job-hanging-watchdog))

Per job, from the scheduler: queue wait, number of requeues, the nodes used in each attempt (so failures can be attributed), and preemptions.

A per-rank step breakdown in its simplest form, on 2 CPU ranks where rank 1 has a slower data loader. On CPU, wall-clock timers are accurate because operations are synchronous; on GPU you'd need CUDA events or a synchronize, because kernel launches return before the work is done (see [concurrency](../foundations/concurrency.md#the-host-launches-the-device-executes-asynchronously)) **[ran]**:

```python
# step_timer.py -- per-rank step breakdown: data wait / compute / collective, plus heartbeat
import json, os, time, torch, torch.distributed as dist

dist.init_process_group("gloo")
rank = dist.get_rank()
model = torch.nn.Linear(1024, 1024)
x = torch.randn(256, 1024)                           # (B, D)

def fake_loader():
    while True:
        time.sleep(0.002 * (rank + 1))               # rank 1's "disk" is slower
        yield x

it = fake_loader()
for step in range(5):
    t0 = time.perf_counter()
    batch = next(it)                                 # (B, D)
    t1 = time.perf_counter()
    loss = model(batch).square().mean()              # (B, D) -> scalar
    loss.backward()
    t2 = time.perf_counter()
    for p in model.parameters():                     # naive DDP: all-reduce every grad
        dist.all_reduce(p.grad); p.grad /= dist.get_world_size()
    t3 = time.perf_counter()
    model.zero_grad()
    # heartbeat: what a metrics exporter would scrape; on GPU use CUDA events instead
    rec = dict(rank=rank, step=step, ts=time.time(), data_ms=round((t1-t0)*1e3, 1),
               compute_ms=round((t2-t1)*1e3, 1), comm_ms=round((t3-t2)*1e3, 1))
    if step == 4: print(json.dumps(rec), flush=True)
dist.destroy_process_group()
```

```
$ GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29407 \
    --local-addr=127.0.0.1 --nproc-per-node=2 step_timer.py
{"rank": 0, "step": 4, "ts": 1793914022.1247108, "data_ms": 2.4, "compute_ms": 3.0, "comm_ms": 6.9}
{"rank": 1, "step": 4, "ts": 1793914022.1247182, "data_ms": 4.2, "compute_ms": 4.0, "comm_ms": 3.9}
```

Rank 1 is the slow one (4.2 ms of data wait against 2.4), yet rank 0 shows **more** time in the collective (6.9 ms against 3.9), because it arrives at the all-reduce first and waits for rank 1. That inversion is the subject of the next section.

The "no progress" alert as a Prometheus rule, assuming your exporter publishes the heartbeat as a gauge called `train_last_step_timestamp_seconds` (a name made up for this example) **untested**:

```yaml
groups:
- name: training
  rules:
  - alert: TrainingNoProgress
    # newest heartbeat across all ranks of a job is more than 10 minutes old
    expr: time() - max by (job_name) (train_last_step_timestamp_seconds) > 600
    for: 2m
    labels:
      severity: page
    annotations:
      summary: "{{ $labels.job_name }}: no training step completed in 10+ minutes"
```

Use `min by` instead of `max by` to catch a single stuck rank while the others still report, which happens when a rank is hung outside the collective (for example in data loading) and the rest are blocked waiting for it.

### Alerting on stragglers

A **straggler** is a rank that is alive and working but slower than the others. In synchronous training the step time is the **maximum** over ranks. One GPU running at `1/1.1` of normal speed, for example because of thermal throttling, takes 10% longer per step, so the whole job's step time goes up 10% and its throughput drops to `1/1.1 = 91%`. At 16,384 GPUs that 9% throughput loss is like losing `0.091 * 16384 = 1489` GPUs. The Llama 3 paper notes stragglers "often appearing as functioning but slow communications". The slow rank finishes compute late, and everyone else's time shows up as waiting in the collective. Two consequences:

- **Collective wait time points at the wrong ranks.** The healthy ranks wait the longest. Look for the rank with the **least** time waiting in collectives, or the longest compute time per step.
- A practical detector: for each step, compute each rank's compute time (excluding collective wait), compare it with the median across ranks, and flag a rank that is more than X% above the median for M consecutive steps. Cross-check against that node's DCGM clocks and throttle reasons and against its NIC error counters. X and M need to be tuned per job. Start from the run-to-run noise you observe on a healthy run, not from a fixed number.

The detector on simulated data: 64 ranks with 1% step-time noise, and rank 37 throttling to 10% slower from step 10 **[ran]**:

```python
# straggler.py -- flag a rank whose COMPUTE time (not collective wait) is persistently high
import random
random.seed(0)
R, STEPS, X, M = 64, 30, 0.05, 5            # ranks, steps, 5% over median, 5 consecutive steps
SLOW_RANK, SLOW_FROM = 37, 10               # rank 37 starts throttling at step 10

streak = [0] * R
for step in range(STEPS):
    compute = [1.0 + random.gauss(0, 0.01) for _ in range(R)]       # seconds, ~1% noise
    if step >= SLOW_FROM:
        compute[SLOW_RANK] *= 1.10                                  # 10% slower GPU
    step_time = max(compute)                                        # sync step = slowest rank
    wait = [step_time - c for c in compute]                         # time blocked in all-reduce
    med = sorted(compute)[R // 2]
    for r in range(R):
        streak[r] = streak[r] + 1 if compute[r] > med * (1 + X) else 0
        if streak[r] == M:
            print(f"step {step}: flag rank {r} (compute {compute[r]:.3f}s vs median {med:.3f}s, "
                  f"its wait {wait[r]:.3f}s = the LEAST of any rank: {wait[r] == min(wait)})")
```

```
step 14: flag rank 37 (compute 1.104s vs median 0.999s, its wait 0.000s = the LEAST of any rank: True)
```

It flags exactly the throttled rank, 5 steps after it slowed down (the `M` setting), and no false positives over 30 steps with X = 5% against 1% noise. With noisier steps (variable sequence lengths, MoE load imbalance), X has to rise or the median has to be taken over a window of steps.

When a straggler is confirmed, it's a drain-and-requeue event just like a crash: the arithmetic above says that a 10% straggler costs more per hour at scale than a restart does, as long as the restart is fast.


## You can now read

- [Fault Tolerance](../training/fault-tolerance/README.md): the recipes (signal handling, job arrays, watchdogs) that sit on top of the mechanisms here.
- [SLURM for users](slurm/users.md) and [SLURM administration](slurm/admin.md): the day-to-day commands, now with the scheduler internals behind them.
- [Kubernetes gotchas](kubernetes/README.md): the practical case against k8s for interactive training.
- [Decentralized training](../training/decentralized-training.md): what changes when nodes are not in one cluster, the failure rate is far higher, and there is no shared scheduler.
- [RL infrastructure](../platform/rl-infrastructure.md) and [services and control planes](../platform/control-planes.md): RL systems mix a training gang with an autoscaled fleet of inference replicas, so both halves of this chapter apply at once.
- [Serving hands-on](../inference/serving-hands-on.md): running one of the replicas that the inference section scales.


## Further reading

- Llama 3 herd of models, section 3.3.4 Reliability and Operational Challenges: [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- J. W. Young, "A first order approximation to the optimum checkpoint interval", CACM 1974; J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart dumps", FGCS 2006
- Slurm docs: [select/cons_tres](https://slurm.schedmd.com/cons_tres.html), [topology](https://slurm.schedmd.com/topology.html), [topology.conf](https://slurm.schedmd.com/topology.conf.html), [multifactor priority](https://slurm.schedmd.com/priority_multifactor.html), [Fair Tree](https://slurm.schedmd.com/fair_tree.html), [scheduling configuration and backfill](https://slurm.schedmd.com/sched_config.html), [preemption](https://slurm.schedmd.com/preempt.html), and SchedMD's SLUG'24 talks on topology/block
- Kubernetes: [device plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/), [Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), [KEP-4381](https://www.kubernetes.dev/resources/keps/4381/), [Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/), [Indexed Jobs with pod-to-pod communication](https://kubernetes.io/docs/tasks/job/job-with-pod-to-pod-communication/)
- NVIDIA: [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html), [DRA driver for GPUs](https://github.com/NVIDIA/k8s-dra-driver-gpu), [KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler), [Topograph](https://github.com/NVIDIA/topograph), [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter), [DCGM diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)
- Batch on k8s: [Kueue](https://kueue.sigs.k8s.io/) (including [Topology Aware Scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling)), [Volcano](https://volcano.sh/), [JobSet](https://jobset.sigs.k8s.io/), [LeaderWorkerSet](https://lws.sigs.k8s.io/), [scheduler-plugins coscheduling](https://github.com/kubernetes-sigs/scheduler-plugins)
- Networking for pods: [Multus](https://github.com/k8snetworkplumbingwg/multus-cni), [SR-IOV network device plugin](https://github.com/k8snetworkplumbingwg/sriov-network-device-plugin), [NVIDIA Network Operator](https://docs.nvidia.com/networking/display/kubernetes2410/index.html)
- Slurm and k8s together: [Slinky](https://github.com/SlinkyProject), [Soperator](https://github.com/nebius/soperator) and its [architecture explainer](https://nebius.com/blog/posts/soperator-in-open-source-explained)
- PyTorch: [torchrun / elastic](https://pytorch.org/docs/stable/elastic/run.html), [rendezvous](https://pytorch.org/docs/stable/elastic/rendezvous.html), [Distributed Checkpoint async_save](https://pytorch.org/docs/stable/distributed.checkpoint.html), [flight recorder](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html)
- Inference: [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension), [llm-d](https://github.com/llm-d/llm-d), [KEDA](https://keda.sh/)
- Gemini: "Fast Failure Recovery in Distributed Training with In-Memory Checkpoints", SOSP 2023
- In this book: [SLURM for users](slurm/users.md), [SLURM administration](slurm/admin.md), [launchers](slurm/launchers/), [Kubernetes gotchas](kubernetes/README.md), [Kubernetes and Helm from zero](kubernetes-from-zero.md), [Fault Tolerance](../training/fault-tolerance/README.md), [NVIDIA GPU troubleshooting](../compute/accelerator/nvidia/debug.md), [hanging and deadlocks](../debug/torch-distributed-hanging-solutions.md), [Inference](../inference/README.md), [Networking](../network/README.md), [Distributed systems](../foundations/distributed-systems.md)
