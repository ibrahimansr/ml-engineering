# Fault Tolerance, Checkpoints and Reproducibility

A large training run is a long-running distributed program on hardware that fails. GPUs overheat, HBM goes bad, a switch port flaps, a teammate fills the shared disk, the scheduler's time limit arrives, a higher-priority job preempts you. Whether you own the hardware or rent it by the hour, every one of those events costs you the training time since your last checkpoint, plus the time it takes a human to notice. If a GPU dies at 2am and nobody looks until 9am, you lose seven hours on hundreds of GPUs.

Redundancy in the web-services sense, two of everything with automatic fail-over, is mostly off the table: the hardware is far too expensive to duplicate. What you can do instead is a handful of cheap recipes. Keep spare nodes, queue the next job before the current one dies, checkpoint at the right frequency, give the program ways to stop and save cleanly, run watchdogs that warn you before the disk fills up, and catch the scheduler's warning signals. That is the first half of this chapter.

The second half is about being able to run the same thing twice: pinning the random number generators and GPU kernels so that a bug reproduces, recording the software environment so that a performance difference can be explained, and a few practical tools for testing multi-node code without a multi-node cluster.

Conventions:

- Examples marked **[ran]** were executed while writing this chapter on a MacBook (CPU only, `torch==2.15.0`, Python 3.14, `transformers==5.22.2`), and the output shown is from that run. They need `pip install torch numpy psutil` and, where noted, `safetensors transformers`.
- Examples marked **[untested]** need a GPU, a Slurm cluster or `sudo` on a GPU node, and were not run here. Outputs shown in those sections are from the original author's runs (dates and versions are given where known).
- Most examples use Slurm, because most large training runs happen under Slurm, but almost every idea carries over to Kubernetes or a bare `torchrun` on rented machines.

## Prerequisites

- [Linux and the OS](../foundations/linux-and-os.md): [signals](../foundations/linux-and-os.md#signals), [cron](../foundations/linux-and-os.md#cron), [inodes](../foundations/linux-and-os.md#inodes-names-and-links), [`df` and `du`](../foundations/linux-and-os.md#disk-usage-du-and-df), [shared libraries and `LD_LIBRARY_PATH`](../foundations/linux-and-os.md#shared-libraries-and-ld_library_path), [environment modules](../foundations/linux-and-os.md#environment-modules-lmod), [local vs network filesystems](../foundations/linux-and-os.md#local-vs-network-filesystems), [`fsync` and why checkpoints get lost](../foundations/linux-and-os.md#fsync-and-durability-why-checkpoints-get-lost) and [how `torchrun` spawns workers](../foundations/linux-and-os.md#how-torchrun-spawns-workers).
- [Distributed systems](../foundations/distributed-systems.md): [partial failure](../foundations/distributed-systems.md#partial-failure), [barriers](../foundations/distributed-systems.md#barriers), [rendezvous](../foundations/distributed-systems.md#rendezvous), [checkpoint/restart](../foundations/distributed-systems.md#checkpoint--restart), [hot spares](../foundations/distributed-systems.md#hot-spares), [elastic training](../foundations/distributed-systems.md#elastic-training) and [determinism across ranks](../foundations/distributed-systems.md#determinism-and-reproducibility-across-ranks).
- [Storage and data](../foundations/storage-and-data.md): [what is in a training checkpoint](../foundations/storage-and-data.md#whats-in-a-training-checkpoint), [how often to checkpoint](../foundations/storage-and-data.md#how-often-to-checkpoint), [async checkpointing](../foundations/storage-and-data.md#async-checkpointing), [safetensors](../foundations/storage-and-data.md#safetensors-a-header-plus-raw-bytes), [the Hugging Face sharded layout](../foundations/storage-and-data.md#sharded-checkpoints-the-hugging-face-layout) and [deterministic, resumable data loading](../foundations/storage-and-data.md#deterministic-resumable-data-loading).
- [Slurm from zero](../orchestration/slurm/from-zero.md): [job arrays](../orchestration/slurm/from-zero.md#job-arrays-many-similar-jobs), [dependencies](../orchestration/slurm/from-zero.md#dependencies-and-pipelines), [`squeue`](../orchestration/slurm/from-zero.md#squeue-what-is-queued-and-running), [how jobs get stopped](../orchestration/slurm/from-zero.md#how-jobs-get-stopped), [QoS](../orchestration/slurm/from-zero.md#accounts-associations-qos-and-fair-share), [drained nodes](../orchestration/slurm/from-zero.md#drained-nodes) and [cron emulation](../orchestration/slurm/from-zero.md#cron-and-self-perpetuating-jobs).
- [Orchestration internals](../orchestration/internals.md#fault-tolerance-at-scale) for failure-rate arithmetic and the [Young/Daly checkpoint interval](../orchestration/internals.md#optimal-checkpoint-interval-youngdaly).
- [Parallelism derivations](model-parallelism/derivations.md) for DP, [HSDP](model-parallelism/derivations.md#hsdp-hybrid-sharding), TP, PP and [how they compose](model-parallelism/derivations.md#composing-parallelisms-device-mesh-ordering); [the caching allocator](arithmetic.md#pytorch-caching-allocator-internals) for GPU memory fragmentation.
- [The GPU software stack](../compute/gpu-software-stack.md) for NCCL versions and [MIG](../compute/gpu-software-stack.md#sharing-one-gpu-mig-and-mps).

## Why fault tolerance is a training problem

Two numbers from earlier chapters frame everything here. From [failure-rate arithmetic](../orchestration/internals.md#failure-rate-arithmetic): a synchronous job fails when any one of its components fails, so a 16,384-GPU job with a per-GPU MTBF (mean time between failures) of 50,000 hours fails about every 3 hours. And from [Young/Daly](../orchestration/internals.md#optimal-checkpoint-interval-youngdaly): the time you lose per failure is the restart cost plus, on average, half a checkpoint interval of redone work.

So the levers are:

1. **Restart fast.** Have healthy nodes ready and the next job already queued ([spare capacity](#spare-capacity), [queued jobs](#queue-up-multiple-training-jobs), or [rebuild the communicator without restarting](#rebuilding-the-process-group-in-place)).
2. **Lose little work.** Checkpoint at the right frequency ([frequent checkpoint saving](#frequent-checkpoint-saving)), or avoid filesystem checkpoints entirely by recovering from a peer replica ([multi-replica fault tolerance](#multi-replica-based-fault-tolerance)).
3. **Stop cleanly when you know a stop is coming.** Save first, then exit ([kill and save switches](#kill-switch), [memory watchdogs](#dealing-with-slow-memory-leaks), [preemption signals](#dealing-with-forced-job-preemption)).
4. **Prevent the failures you can prevent.** You can't stop a GPU dying (other than by cooling it properly), but you can stop the disk from filling up ([watchdogs](#prevention)).

## Spare capacity

### Always plan to have more nodes than needed

GPUs fail. Sometimes they overheat and shut down, then recover. Sometimes they break and need replacing. The situation improves when you keep the same nodes for weeks or months, because the bad ones get weeded out. But a new shipment, especially of a GPU generation that has just come out, can have a sizeable fraction of early failures.

So if you need 64 nodes for your training, make sure you have a few spare nodes, and find out how quickly failed nodes can be replaced if the spares run out. The exact margin is hard to predict, but **5-10% spare is reasonable**, and the tighter your deadline, the bigger the margin should be. [Hot spares](../orchestration/internals.md#hot-spares) shows how to size the pool from the failure rate and repair time with a Poisson calculation; 5-10% is what that calculation gives for typical numbers.

Spares only help if replacement is automatic:

- On Slurm, check that a node that fails a health check is **drained** (taken out of the pool of nodes jobs can be scheduled on) and that the next job gets a healthy node instead. See [automatic node draining](../orchestration/internals.md#automatic-node-draining) and [finding faulty nodes and excluding them](../orchestration/slurm/from-zero.md#finding-faulty-nodes-and-excluding-them).
- On any other scheduler, verify that it too can replace bad nodes without a human.

You also want **at least one extra node** outside the training allocation for the watchdogs described [below](#prevention), and possibly for offloading checkpoints to slower storage and running cleanup jobs.

### Preferring fixed accelerator allocations to dynamic ones

When you get a new set of accelerator nodes, especially of a recently released accelerator type, many of them fail early. Failure rates as high as **10%** early in a new accelerator's life are possible, and they stay fairly high for a while after that. Remember that if a node has 8 accelerators, one failing accelerator takes the whole node out from the training program's point of view, since the job can't run with a hole in it.

With a **fixed allocation** (you keep the same nodes for months), the bad accelerators get weeded out over the first few months and failures become rare. They still happen, just not often.

Make sure your provider gives you **new** accelerators when one fails, rather than returning the same one after it has cooled down (literally). Transient failures tend to repeat under heavy load. Record the GPU UUIDs (the unique ID `nvidia-smi -L` prints for each GPU) of every node you run on, so that you can tell when a "new" node is an old broken one coming back; [the GPU software stack chapter](../compute/gpu-software-stack.md) shows how.

```bash
# log this at the start of every job, for every node, to persistent storage
srun --ntasks-per-node=1 bash -c 'echo "$(date -Is) $(hostname) $(nvidia-smi -L | tr "\n" " ")"' >> $HOME/gpu-uuid-history.log
```

**[untested]** How to read it: one line per node per job, with a timestamp, the hostname and the UUID of every GPU on it. `grep` for a UUID that appeared in an earlier failure to see whether it's back.

With a **dynamic allocation** (you get whatever nodes are free each time), expect lots of failing accelerators even a year after a new type was released, because you are getting the nodes other users gave back. Some clouds are better than others at replacing bad hardware, but many accelerators don't fail outright. A technician testing a returned node may not see anything wrong with it, and if the user who released it never reported it as broken, and the provider doesn't re-check nodes before handing them to the next customer, the chance of getting a bad node is very high.

## Recovering from a crash

### Queue up multiple training jobs

Make sure that when the training crashes, there is already a new job lined up to take its place. So instead of starting training with:

```bash
sbatch train.slurm
```

start it with:

```bash
sbatch --array=1-10%1 train.slurm
```

This books a [job array](../orchestration/slurm/from-zero.md#job-arrays-many-similar-jobs) of 10 jobs. `%1` throttles the array to one running job at a time, so the jobs run in series: whether a job finishes normally or crashes, Slurm immediately schedules the next one, which resumes from the latest checkpoint. [Job array trains](../orchestration/slurm/from-zero.md#job-array-trains) shows how to keep editing what the next job runs, and how to hold and release the train without losing its queue position.

If you already started training without this, you can add it without stopping the running job, using `--dependency`. Find the running job's ID first. The `-o` format string picks the columns: `%i` job id, `%P` partition, `%j` job name, `%T` state, `%M` time used so far, `%l` time limit, `%D` number of nodes, `%S` start time, `%R` node list or the reason the job is pending, and the numbers set column widths (see [`squeue`](../orchestration/slurm/from-zero.md#squeue-what-is-queued-and-running)):

```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-64]
```

The running job is `JOBID=87`, so:

```bash
sbatch --array=1-10%1 --dependency=87 train.slurm
```

and now the queue shows:

```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-64]
 88_[1-10%1]   prod    my-training-10b  PENDING       0:00 1-16:00:00   64                    N/A (Dependency)
```

How to read it: `88_[1-10%1]` is the new array of 10 tasks, one at a time, and `(Dependency)` says it is waiting for job 87 to end (a bare `--dependency=ID` means `afterany`, see [dependencies](../orchestration/slurm/from-zero.md#dependencies-and-pipelines)). As soon as 87 completes or fails, task 1 of 88 starts.

If the condition that caused the crash is still there, the next job fails too. If the storage is full, no number of restarts will help; [watchdogs](#prevention) are how you avoid that. But the most common cause of crashes is a failing GPU, and if faulty nodes are removed automatically and the next job starts on healthy ones, recovery is smooth.

In Slurm terms, a removed node is **drained**. On a hypothetical cluster:

```bash
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
prod*       up   infinite       4  drain node-[0-3]
prod*       up   infinite      47  alloc node-[4-51]
prod*       up   infinite      23   idle node-[52-73]
```

How to read it: 47 nodes are in use (`alloc`), 23 are available (`idle`) and 4 are out of service (`drain`). `sinfo -R` shows [why each was drained and by whom](../orchestration/slurm/from-zero.md#drained-nodes). A sysadmin is expected to check the drained nodes periodically, fix or replace them, and put them back to `idle`.

The other approach is to daisy-chain individual jobs with `--dependency` (each job submits or is submitted behind the previous one, see [dependencies and pipelines](../orchestration/slurm/from-zero.md#dependencies-and-pipelines)). The two can be combined.

How does a job array know when to stop? Normally the training loop exits immediately on start if it sees that training is already done (for example, the last checkpoint is at the final step). A [kill switch](#kill-switch) is an even easier way to burn through the rest of an array.

### Rebuilding the process group in place

Everything above recovers from a dead rank by letting the whole job exit and starting the next queued job on a fresh allocation. That throws away every surviving process's state: the CUDA context, `torch.compile` caches, the data loader state, the model in GPU memory. It all has to be rebuilt from the checkpoint on disk.

Starting with PyTorch 2.14, the communicator itself (the process group all ranks use for collectives, created by `torch.distributed.init_process_group`) can be rebuilt in place, so surviving ranks keep their warm state and continue with a smaller (or re-filled) world. The path is **experimental**: you opt in with `init_process_group(..., enable_reconfigure=True)`, and the calls (`dist._get_reconfigure_handle`, `dist._reconfigure`, `dist._supports_reconfigure`) have leading underscores and are documented as subject to change or removal. Until they stabilize, keep spare nodes and a job array as your main recovery path.

Here is what it looks like with three Gloo ranks on one machine, where rank 2 "dies" and ranks 0 and 1 carry on **[ran]**:

```python
# reconfig.py -- rebuild a gloo process group in place after one rank dies (experimental, underscored API; torch 2.15)
import os, datetime, torch, torch.distributed as dist
import torch.multiprocessing as mp
T = datetime.timedelta(seconds=30)

def form_group(store, rank, members, uuid):
    # out-of-band exchange of handles; in real life membership comes from heartbeats (torchft's Lighthouse etc.)
    store.set(f"{uuid}/h{rank}", dist._get_reconfigure_handle())
    handles = [store.get(f"{uuid}/h{r}").decode() for r in members]   # a list: new rank = position
    dist._reconfigure(uuid=uuid, handles=handles, timeout=T).wait()

def worker(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
    dist.init_process_group("gloo", rank=rank, world_size=world, enable_reconfigure=True)
    store = dist.TCPStore("127.0.0.1", 29534, world, is_master=(rank == 0), timeout=T, wait_for_workers=False)
    form_group(store, rank, list(range(world)), uuid=0)               # even the first group is formed this way
    t = torch.ones(1); dist.all_reduce(t)
    print(f"rank {rank}: world={dist.get_world_size()} all_reduce(1)={t.item():.0f}  pid={os.getpid()}", flush=True)
    if rank == world - 1:
        print(f"rank {rank}: dying", flush=True); os._exit(1)        # simulated GPU failure
    form_group(store, rank, list(range(world - 1)), uuid=1)          # survivors only, same processes
    t = torch.ones(1); dist.all_reduce(t)
    print(f"rank {rank}: world={dist.get_world_size()} all_reduce(1)={t.item():.0f}  pid={os.getpid()} (rebuilt)", flush=True)
    store.set(f"done{rank}", "1"); store.wait([f"done{r}" for r in range(world - 1)])

if __name__ == "__main__":
    ps = [mp.get_context("spawn").Process(target=worker, args=(r, 3)) for r in range(3)]
    for p in ps: p.start()
    for p in ps: p.join(60)
    print("exit codes:", [p.exitcode for p in ps])
```

```
rank 0: world=3 all_reduce(1)=3  pid=2877
rank 2: world=3 all_reduce(1)=3  pid=2879
rank 2: dying
rank 1: world=3 all_reduce(1)=3  pid=2878
rank 0: world=2 all_reduce(1)=2  pid=2877 (rebuilt)
rank 1: world=2 all_reduce(1)=2  pid=2878 (rebuilt)
exit codes: [0, 0, 1]
```

How to read it: the all-reduce of a tensor of ones returns the number of ranks, so `3` and then `2` shows the group really shrank. The pids of ranks 0 and 1 are the same before and after, so these are the same processes, not restarted ones. Two things I found while running it that the summary above doesn't tell you: with `enable_reconfigure=True`, even the *first* group must be formed through `_reconfigure` (a collective before that raises `ProcessGroupGloo has not been initialized. Call reconfigure() before issuing collectives`), and the handle exchange is your job: here a second `TCPStore`, in a real system the coordinator that tracks heartbeats. This demo kills a rank *between* collectives. A rank dying *inside* a collective additionally needs the survivors to detect the failure and abort the stuck collective first; I did not test that. The NCCL backend is the case that matters on GPUs, and I did not test it either. Libraries like torchft ([below](#multi-replica-based-fault-tolerance)) wrap exactly this kind of machinery.

## Checkpoints

### Frequent checkpoint saving

When a training job fails, all the work since the last checkpoint is lost. If the failure happens 12 hours after the last save, 12 hours of training have to be redone, which is very expensive on hundreds of GPUs.

In principle you could save every 10 minutes and never lose more than 10 minutes. But large models can't be saved quickly: depending on your checkpointing method and the speed of your storage, saving takes anywhere from dozens of seconds to several minutes, and if saving becomes a bottleneck, the training takes longer overall. The optimum is somewhere in the middle.

The basic arithmetic: measure how long one save takes, multiply by the number of saves you plan, and see what fraction of the total training time that is. Here is the BLOOM-176B case, which had an unusually fast GPFS-over-NVMe filesystem: a 2.3TB checkpoint, written concurrently by 384 processes, took only 40 seconds, and it was saved about every 3 hours over roughly 3 months of training **[ran]**:

```python
# ckpt_cost.py -- what checkpointing cost BLOOM-176B, and what it would cost elsewhere
import math
days, every_h, save_s, ckpt_tb = 90, 3, 40, 2.3
n = days * 24 / every_h                          # checkpoints written
hours = n * save_s / 3600                        # time spent blocked on saving
print(f"{n:.0f} checkpoints, {hours:.1f} h saving = {hours / (days * 24):.2%} of the run")
for slow in (5, 15):
    print(f"  IO {slow:>2}x slower: {slow * hours / (days * 24):.2%}")
print(f"disk written per day: {24 / every_h * ckpt_tb:.1f} TB")
# Young's optimum for the same C, under an ASSUMED job-level MTBF M (not a BLOOM number)
for M_h in (12, 24, 72):
    tau = math.sqrt(2 * save_s * M_h * 3600)
    print(f"  M = {M_h:>2} h -> tau_opt = {tau/60:5.1f} min; at tau = 3 h the expected rework is {every_h / (2 * M_h):.1%}")
```

```
720 checkpoints, 8.0 h saving = 0.37% of the run
  IO  5x slower: 1.85%
  IO 15x slower: 5.56%
disk written per day: 18.4 TB
  M = 12 h -> tau_opt =  31.0 min; at tau = 3 h the expected rework is 12.5%
  M = 24 h -> tau_opt =  43.8 min; at tau = 3 h the expected rework is 6.2%
  M = 72 h -> tau_opt =  75.9 min; at tau = 3 h the expected rework is 2.1%
```

How to read it: 720 saves (`90 days * 24h / 3h`) at 40 s each is 8 hours (`720 * 40 s / 3600`), or 0.37% of the run (`8h / (90 days * 24h)`). If the storage were 5 times slower, which is common in the cloud unless you pay for premium storage, that becomes about 2%, which is significant. The last three lines connect this to the [Young/Daly optimum](../orchestration/internals.md#optimal-checkpoint-interval-youngdaly): $\tau_{\text{opt}} = \sqrt{2CM}$, with save cost $C$ and job-level MTBF $M$. I don't know BLOOM's actual failure rate, so those lines use assumed values of $M$. They show that with a 40-second save, the optimal interval is well under an hour unless failures are rare, and that a 3-hour interval costs an expected 3h/2 of rework per failure. Fast storage makes frequent saves cheap, so use it. With [async checkpointing](../foundations/storage-and-data.md#async-checkpointing), $C$ shrinks to the device-to-host copy and the optimum moves to minutes.

If you don't have much local storage and have to offload checkpoints to the cloud, **keep the 2 most recent checkpoints local** for a quick resume. Two, not one, because a crash during a save can leave the newest checkpoint incomplete or corrupt. Writing to a temporary name and renaming it at the end (as the loop [below](#a-training-loop-with-every-switch) does) is the cheap protection against half-written files; [making checkpoints trustworthy](../foundations/storage-and-data.md#making-checkpoints-trustworthy) covers the rest.

Checkpoints cost time, but they are useful beyond crash recovery:

- You can **roll back** many steps when the loss diverges (see [Training stability](stability.md)).
- You can **analyze** what happened around an event.
- Many trainings now replace the in-training eval (a single held-out loss, which says little) with a full **benchmark evaluation of each checkpoint**, run on separate nodes so it doesn't slow the training down.

### Multi-replica-based fault tolerance

There is another way to deal with accelerator failures that doesn't involve filesystem checkpoints at all. It works only when the training has at least two **model replicas** (complete copies of the model state):

- With some form of 3D parallelism ([TP, PP and DP](model-parallelism/derivations.md#composing-parallelisms-device-mesh-ordering)), the number of replicas equals the DP degree.
- With hybrid ZeRO-DP ([HSDP](model-parallelism/derivations.md#hsdp-hybrid-sharding): FSDP/ZeRO sharding within a group of GPUs, plain data-parallel replication across groups), the number of replicas equals the number of replication groups.

Example: TP=4, PP=2, DP=2. Each replica uses $4 \times 2 = 8$ GPUs, so in practice each replica is a whole 8-GPU node: replica A on `node0`, replica B on `node1`. There is also an idle standby node `node2`.

Now `node0.gpu0` fails. Replica B still has the full, current model and optimizer state. So you switch to the standby node, copy the state from replica B's GPUs to `node2`'s GPUs with RDMA (a direct GPU-to-GPU transfer over the network that bypasses the CPUs, see [GPUDirect](../foundations/computer-architecture.md#gpudirect-cutting-the-cpu-and-its-memory-out-of-the-path)), and continue where you left off. This is a simplified picture. Recovering correctly depends on where in the iteration the failure happened (before or after the gradient all-reduce, in the middle of the optimizer step), so the real algorithm is complex. At scale you'd have about a hundred active nodes and a small handful of standbys.

This is better than filesystem checkpoints because you lose at most one iteration, instead of up to a full checkpoint interval.

The open-source library closest to this idea is [`torchft`](https://github.com/meta-pytorch/torchft) (Meta / PyTorch). It treats each training step as a fault boundary across replica groups, coordinates health with a lighthouse/quorum service, and heals a failed group by live state transfer from a healthy peer instead of rolling back to a filesystem checkpoint. Out of the box it supports fault-tolerant DDP and HSDP, and [`torchtitan`](https://github.com/pytorch/torchtitan) wires it into an end-to-end HSDP training loop: see its [`torchft` experiment docs](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/torchft) and the [PyTorch blog post on a checkpoint-free run under synthetic failures](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/). Two differences from the sketch above: the failure domain is a whole replica group rather than a single GPU, and healing is peer recovery through torchft's checkpoint transports rather than an RDMA copy onto a dedicated standby. Same goal (lose at most a step, keep training), different machinery. A runnable torchft example, where one replica is killed and heals from the survivor, is in [Training across datacenters](decentralized-training.md#torchft-as-of-031), the next chapter.

Related research systems attack the same problem from other angles:

- [Oobleck](https://github.com/SymbioticLab/Oobleck) (SOSP'23) keeps logically equivalent pipeline replicas and recovers by re-instantiating precomputed pipeline templates after failures, again without a full checkpoint restart.
- [Bamboo](https://www.usenix.org/system/files/nsdi23-thorpe.pdf) (NSDI'23) schedules redundant computation into pipeline bubbles, for training on spot instances.
- [Varuna](https://github.com/microsoft/varuna) reconfigures PP+DP elastically.

Bamboo and Varuna are largely dormant now and are mostly useful as design references. There are also commercial infrastructure layers that migrate a failed GPU's work onto a spare without changing the training script. That is a different trade-off (a per-GPU failure domain, handled outside the framework), not another torchft.

### Checkpoint conversion and repair tools

Two small scripts live next to this chapter's sources, in [`training/checkpoints/`](#checkpoint-conversion-and-repair-tools).

**[torch-checkpoint-convert-to-bf16](checkpoints/torch-checkpoint-convert-to-bf16)** converts an existing fp32 checkpoint to bf16, which halves its size (for inference or for re-training in bf16). It expects a Hugging Face-style checkpoint directory (`*.bin` weight shards, `*json` config files and a `*model` tokenizer file, see [the Hugging Face layout](../foundations/storage-and-data.md#sharded-checkpoints-the-hugging-face-layout)). You `cd` into that directory and run it, and the bf16 copy is written to a `bf16` sub-directory. If [safetensors](https://github.com/safetensors/safetensors) files are present, they are converted too: they are regenerated from the converted `*.bin` files rather than converted in place, and `pytorch_model.bin.index.json` is rewritten into `model.safetensors.index.json`. It is easy to adapt to similar cases (fp16, other file patterns).

Here it is on a toy checkpoint with two 1000x256 fp32 tensors **[ran]** (needs `safetensors`):

```bash
$ ls
config.json  model.safetensors  pytorch_model.bin  tokenizer.model
$ bash /path/to/training/checkpoints/torch-checkpoint-convert-to-bf16
creating a new checkpoint under dir bf16
converting *bin torch files
converting *safetensors files
the dir bf16 now contains a copy of the original checkpoint with bf16 weights
$ wc -c pytorch_model.bin model.safetensors bf16/pytorch_model.bin bf16/model.safetensors
 2049941 pytorch_model.bin
 2048216 model.safetensors
 1025941 bf16/pytorch_model.bin
 1024216 bf16/model.safetensors
 6148314 total
$ python -c "import torch; print({k: v.dtype for k,v in torch.load('bf16/pytorch_model.bin').items()})"
{'model.embed.weight': torch.bfloat16, 'model.lm_head.weight': torch.bfloat16}
```

How to read it: each file is half its original size, and the tensors in the copy are `bfloat16`. The script uses `torch.load` on the `.bin` files, which [can execute code from the file](../foundations/storage-and-data.md#pickle-and-why-loading-a-checkpoint-can-run-code), so only run it on checkpoints you trust.

**[torch-checkpoint-shrink.py](checkpoints/torch-checkpoint-shrink.py)** fixes checkpoints that saved tensors with more storage than they needed. A PyTorch tensor is a **view** (shape, strides and offset) into an underlying storage buffer, and `torch.save` writes the whole storage, not only the part the view covers. So saving a 10-element slice of a 64MiB tensor writes 64MiB **[ran]**:

```python
# view_bloat.py -- torch.save writes the whole storage behind a view, not just the view
import os, torch
big = torch.zeros(64 * 2**20 // 4)          # 64MiB of fp32
small = big[:10]                            # (10,) view into the same 64MiB storage
os.makedirs("ckpt", exist_ok=True)
torch.save({"w": small}, "ckpt/view.pt")
torch.save({"w": small.clone()}, "ckpt/clone.pt")
for f in ("ckpt/view.pt", "ckpt/clone.pt"):
    print(f"{f:15} {os.path.getsize(f):>10,} bytes")
```

```
ckpt/view.pt    67,110,356 bytes
ckpt/clone.pt        1,563 bytes
```

The script walks every tensor in each `*.pt` file (recursing into nested dicts), replaces it with `.clone()` (a copy with storage exactly the size of the view), and saves the file again in place. Usage is `./torch-checkpoint-shrink.py --checkpoint_dir ./checkpoints/global_step10`, optionally with `--patterns 'layer*pt' 'zero*pt'` to process only matching files (quote the patterns so the shell doesn't expand them), and `-d` for per-file sizes. On a copy of the bloated file above **[ran]**:

```bash
$ cp ckpt/view.pt ckpt/layer_01.pt
$ ./torch-checkpoint-shrink.py --checkpoint_dir ckpt --patterns 'layer*pt'
Processing zero checkpoint 'ckpt'
-> ckpt/layer_01.pt
Done. Before 64.00MiB, after 0.00MiB, saved 64.00MiB
```

It overwrites the files, so run it on a copy first if the checkpoint is precious.

## Graceful stops: kill and save switches

### Kill switch

In many Slurm environments users have no `sudo`, and only the user who submitted a job can cancel it. If one person starts a training, goes to sleep, and then a problem is found, nobody else can stop it and restart it.

This was the situation during BLOOM-176B training, and the team implemented a **kill switch**. The mechanism is simple: before starting each iteration, the training loop checks whether a specific file exists. If it does, the program saves a checkpoint and exits, so that someone other than the submitter can change things and start a new run. A second check was added at the very beginning of `main`, so that if the sleeping user had a long job array queued, each queued job would exit within seconds of starting, and the array could be "burned through" quickly. [Stopping a teammate's job](../orchestration/slurm/from-zero.md#stopping-a-teammates-job-the-kill-switch) covers the Slurm side of this, including Megatron-DeepSpeed's `--kill-switch-path` flag.

A kill switch minimizes the training time wasted while waiting for the person who can cancel the job.

### Save switch

The kill switch's cousin is a **save switch**: if the training loop sees the save-switch file, it saves a checkpoint and **continues** training. It also deletes the file, so it doesn't save again on every following iteration.

This is useful for whoever is watching the training charts. If you see something interesting or worrying in the loss or another metric, you can ask the program to save a checkpoint right now, and later reproduce that exact situation at will. The main use is around loss spikes and divergences (see [Training stability](stability.md)).

### A training loop with every switch

Here is a loop with all the graceful-stop hooks in this chapter: the kill switch (checked at startup and every step), the save switch, a [SIGUSR1 preemption handler](#dealing-with-forced-job-preemption), an [exit-duration time limit](#approach-a-tell-the-program-at-launch-time-when-it-should-start-the-exiting-process) and a [CPU-memory watchdog](#dealing-with-slow-memory-leaks). The detail that matters most in a distributed job is `any_rank()`: the decision to stop must be **the same on every rank**, or some ranks exit while others wait forever in the next collective. So every flag is all-reduced with `MAX` before acting on it **[ran]**:

```python
# ft_loop.py -- a training loop with every "graceful stop" hook from this chapter.
# Run: torchrun --nproc-per-node 2 ft_loop.py [--exit-duration-in-mins M] [--max-cpu-mem-pct P]
import argparse, os, signal, sys, time
import psutil, torch, torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt_ft")
p.add_argument("--kill-switch-path", default="kill-switch")
p.add_argument("--save-switch-path", default="save-switch")
p.add_argument("--exit-duration-in-mins", type=float, default=None)
p.add_argument("--max-cpu-mem-pct", type=float, default=95.0)
p.add_argument("--steps", type=int, default=1000)
args = p.parse_args()

dist.init_process_group("gloo")
rank = dist.get_rank()
start = time.time()

def log(*a):
    if rank == 0: print(f"[{time.time()-start:5.1f}s]", *a, flush=True)

def any_rank(flag: bool) -> bool:
    # every rank must take the same branch, or the next collective hangs
    t = torch.tensor([int(flag)])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())

# kill switch checked before anything expensive: queued jobs "burn through" in seconds
if any_rank(os.path.exists(args.kill_switch_path)):
    log("kill switch present at startup, exiting"); sys.exit(0)

preempt = False
def on_sigusr1(signum, frame):
    global preempt
    preempt = True        # only set a flag: the loop saves at a step boundary
signal.signal(signal.SIGUSR1, on_sigusr1)

torch.manual_seed(0)
model = torch.nn.Linear(16, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
step = 0
path = f"{args.ckpt}/last.pt"
if os.path.exists(path):
    sd = torch.load(path)
    model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); step = sd["step"]
    log(f"resumed from step {step}")

def save(reason):
    if rank == 0:     # DDP replicas are identical: one writer is enough
        os.makedirs(args.ckpt, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path + ".tmp")
        os.replace(path + ".tmp", path)   # atomic: a crash mid-save leaves the old file intact
    dist.barrier()
    log(f"saved step {step} ({reason})")

while step < args.steps:
    # ---- the stop/save checks, once per step, synchronized across ranks ----
    reasons = {
        "kill switch": os.path.exists(args.kill_switch_path),
        "SIGUSR1": preempt,
        "time limit": args.exit_duration_in_mins is not None
                      and (time.time() - start) / 60 > args.exit_duration_in_mins,
        "cpu memory": psutil.virtual_memory().percent > args.max_cpu_mem_pct,
    }
    stop = [k for k, v in reasons.items() if any_rank(v)]
    if stop:
        save("exit: " + ", ".join(stop)); sys.exit(0)
    if any_rank(os.path.exists(args.save_switch_path)):
        save("save switch")
        if rank == 0: os.remove(args.save_switch_path)   # or it would save every step
    # ---- the actual step ----
    x = torch.randn(32, 16)                        # (B, in)
    loss = (model(x) - x.sum(1, keepdim=True)).pow(2).mean()
    opt.zero_grad(); loss.backward()
    for prm in model.parameters(): dist.all_reduce(prm.grad); prm.grad /= dist.get_world_size()
    opt.step(); step += 1
    time.sleep(0.05)                               # pretend a step takes a while
log("finished")
```

A driver that exercises every path. Note how SIGUSR1 is delivered: to the **workers**, found as children of the `torchrun` process, not to `torchrun` itself. That is the same trick as [Approach B.2](#approach-b2-choosing-which-process-to-send-the-signal-to) below:

```bash
# ft_demo.sh -- drive ft_loop.py through the save switch, SIGUSR1, resume, kill switch and memory watchdog
set -u
rm -rf ckpt_ft kill-switch save-switch
run() { torchrun --nproc-per-node 2 --master-addr 127.0.0.1 --master-port 29511 ft_loop.py "$@" 2>&1 | grep -v Warning; }

echo "== 1. save switch, then SIGUSR1 to the workers (not to torchrun)"
run & sleep 4
touch save-switch; sleep 2
launcher=$(pgrep -f "torchrun --nproc-per-node 2")
pgrep -P $launcher | xargs kill -USR1
wait
echo "== 2. resume, then kill switch"
run & sleep 4
touch kill-switch; wait
echo "== 3. a queued job starting while the kill switch is still there"
run; rm kill-switch
echo "== 4. time limit (0.05 min) and memory watchdog (threshold 1%)"
run --exit-duration-in-mins 0.05
run --max-cpu-mem-pct 1
```

Output, with torchrun's startup warnings removed:

```
== 1. save switch, then SIGUSR1 to the workers (not to torchrun)
[  3.1s] saved step 45 (save switch)
[  5.2s] saved step 81 (exit: SIGUSR1)
== 2. resume, then kill switch
[  0.3s] resumed from step 81
[  3.1s] saved step 129 (exit: kill switch)
== 3. a queued job starting while the kill switch is still there
[  0.0s] kill switch present at startup, exiting
== 4. time limit (0.05 min) and memory watchdog (threshold 1%)
[  0.3s] resumed from step 129
[  3.0s] saved step 175 (exit: time limit)
[  0.3s] resumed from step 175
[  0.4s] saved step 175 (exit: cpu memory)
```

How to read it: the save switch produced a checkpoint at step 45 and training continued; SIGUSR1 produced a final checkpoint at 81 and a clean exit; the next run resumed at exactly 81. Run 3 is the "burn through a job array" case: it exited before loading anything. Every exit wrote a checkpoint first, so no step was lost. The time is checked once per step, so the real grace period you need is one step plus one save (see [preemption](#dealing-with-forced-job-preemption)).

Two costs to be aware of. Every check is a tiny all-reduce, which on GPUs is a host-device sync each step; for a fast step, check every N steps instead. And `os.path.exists` on a network filesystem is a metadata request from every rank on every step; on a big job, let rank 0 check and broadcast the result, or check less often (see [metadata storms](../foundations/storage-and-data.md#metadata-storms-and-how-ml-jobs-cause-them)).

## Prevention

The cheapest lost training time is the time you never lose. You can't stop a GPU from failing, other than by making sure it is cooled properly, but you can certainly make sure there is enough disk space for the next few days of training. This is done with scheduled **watchdogs** that monitor resources and alert someone long before a problem crashes the job.

### Scheduled watchdogs

First you need a way to run jobs on a schedule. On Unix this is [cron](../foundations/linux-and-os.md#cron). This crontab line runs `~/bin/watch-fs.sh` every hour (the five fields are minute, hour, day of month, month and day of week, so `0 * * * *` means "at minute 0 of every hour"):

```
0 * * * * ~/bin/watch-fs.sh
```

`crontab -e` edits your crontab and `crontab -l` lists the scheduled jobs. The cron section linked above covers other frequencies.

Many Slurm clusters don't give users access to cron, though. Then you emulate it with jobs that resubmit themselves: see [cron and self-perpetuating jobs](../orchestration/slurm/from-zero.md#cron-and-self-perpetuating-jobs), which sets up a `cron.hourly/` and `cron.daily/` directory where any `*.slurm` file you drop in gets run on that schedule, and [self-perpetuating standalone jobs](../orchestration/slurm/from-zero.md#self-perpetuating-standalone-jobs).

### Notification facility

Then you need at least one way to tell a human.

The simplest is email. Make sure a Slurm job can actually send email: if it can't, ask your sysadmin to enable it, or use an external SMTP server provider. Both watchdog scripts below use Python's `smtplib` against an SMTP server on `localhost`.

You can also send SMS alerts, or post to a Slack channel (Slack's "incoming webhooks" accept a JSON POST, so `curl` is enough).

With scheduling and notification working, here are the watchdogs that matter.

### Is-job-running watchdog

The most obvious one checks that a training job is either running or scheduled to run.

[slurm-status.py](fault-tolerance/slurm-status.py) was used during BLOOM-176B training. It sends an email if the job is neither running nor scheduled, and it appends its result to the main training log, so the log shows a `PULSE:` line every hour. Since BLOOM used [cron emulation](../orchestration/slurm/from-zero.md#cron-and-self-perpetuating-jobs), it was enough to drop [slurm-status.slurm](fault-tolerance/slurm-status.slurm) into the `cron/cron.hourly/` folder, and the already-running scheduler job launched the check about once an hour.

The key line of the Slurm job is:

```bash
tools/slurm-status.py --job-name $WATCH_SLURM_NAME 2>&1 | tee -a $MAIN_LOG_FILE
```

which tells the script which job name to watch, and appends its output to the log file. How the script works:

- It looks up all members of a Unix group (`getent group six`; `six` was the BLOOM project's group), because the training job could have been submitted by any of them.
- It runs `squeue --user=<all members>` with a fixed format and looks for a job with exactly the given name.
- For a `RUNNING` job it prints how long it has been running. For a `PENDING` job it explains why: waiting on the previous task of the job array (`JobArrayTaskLimit`), waiting on a `--dependency`, or scheduled to start in a given time.
- If no job with that name is found, it prints an `***ALERT` line and sends the email (unless `--no-email`).

So if you launch:

```
tools/slurm-status.py --job-name my-training-10b
```

and `squeue` shows:

```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
  JOBID    PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
    87     prod      my-training-10b  RUNNING 2-15:52:19 1-16:00:00  64    2023-10-07T01:26:28 node-[1-64]
```

all is well. If `my-training-10b` isn't there, the alert is sent.

The script is specific to the Jean Zay cluster where BLOOM was trained: it refuses to run unless the hostname contains `idris.fr` or `idrsrv`, and it hardcodes the group name and email addresses. To adapt it, remove `check_running_on_jean_zay()`, and edit `SLURM_GROUP_NAME`, `FROM_ADDR` and `TO_ADDRS`. If you only care about your own jobs, `squeue -u $(whoami)` is enough; `whoami` only works if you launched the job yourself.

### Is-job-hanging watchdog

If the program uses `torch.distributed` and hangs in a collective, the collective eventually times out (10 minutes by default for NCCL) and throws an exception. The job then crashes, the next queued job starts, and you can alert on the restart.

But if the hang is in a call with no timeout, for example reading from a disk that has stopped responding, the program can sit there for hours with nobody noticing.

Most training programs log regularly, for example statistics for the last N steps every few minutes. So the watchdog checks whether the log file has been written to within the expected interval, and alerts if it hasn't. You can write your own (the generic watchdog [below](#low-disk-space-alerts) includes this check) or use [io-watchdog](https://github.com/grondo/io-watchdog). (io-watchdog has had no commits since 2019, so treat it as working but unmaintained.) Finding out *where* a hung job is stuck is a debugging task, covered in [debugging distributed training](../development/debugging-distributed.md).

### Low disk space alerts

The next biggest problem is running out of disk space. If checkpoints are large, saved often and not offloaded elsewhere, a disk fills up fast. Typically several teams share a cluster, and your colleagues may be consuming space quickly too. Ideally your training would have its own dedicated storage partition, but that is often hard to arrange. Either way, you need to know when space is getting low so that someone can make room.

At what threshold should the alert fire? Not too early: if alerts start at 50% usage, people learn to ignore them. And percentages are misleading: on a huge shared disk, 5% free can be many TBs, while on a small partition 25% free might be only a few TBs. What you really need to know is **how fast you write** (how often you checkpoint and how many TBs per day that is) and how much space is available.

During BLOOM training, a 2.3TB checkpoint was written every 3 hours, so the training consumed **18.4TB a day** (computed in [`ckpt_cost.py`](#frequent-checkpoint-saving) above).

Things that make this harder:

- **Several partitions.** Typically there is a fast partition for checkpoints, a slower one for code and libraries, and maybe others. Every partition the training needs to keep running must be monitored.
- **Distributed filesystems can't always use 100% of their space.** Some only reliably give you about 80% of the allocated capacity. They spread data over many physical disks and rebalance periodically or on certain events, so an individual disk can reach 100% and fail a write, crashing the training, while `df` still reports only 80% usage for the partition. This didn't happen during BLOOM-176B, but it did during IDEFICS-80B training, where "80% was the new 100%". Usually you discover this while preparing for the training, so test it then.
- **Inodes.** An inode is the filesystem's per-file metadata record, so the inode quota is effectively a cap on the number of files (see [inodes](../foundations/linux-and-os.md#inodes-names-and-links)). Some storage partitions have small inode quotas. Python packages consist of thousands of small files, which take little space in total but add up to tens of thousands of files in a virtual environment, so you can run out of inodes while still having TBs of free space, and the training crashes. `df -ih` shows the inode quota and current usage.
- **Delayed statistics.** Many distributed filesystems don't report usage in real time; it can take up to 30 minutes to update.
- **Compression.** Some filesystems compress internally, so the usage they report can be smaller than what the same data takes when copied elsewhere, which can be confusing.

[fs-watchdog.py](fault-tolerance/fs-watchdog.py) was used during BLOOM-176B training, driven by [fs-watchdog.slurm](fault-tolerance/fs-watchdog.slurm). It emails an alert if any threshold is crossed. Reading the code, for each partition it checks both bytes and inodes, with different tools depending on the partition:

- For `WORK` and `STORE` it used the cluster's own quota tool (`idrquota`), which returns bytes and inode usage as percentages instantly, and alerted at 85%.
- For the checkpoint partition `SCRATCH`, which filled fastest, it ran `du -bs` against a 400TB quota and alerted at a lower threshold, 75%. Since `SCRATCH` was shared with everyone, it also parsed `df` for the whole filesystem and alerted if less than 100TiB remained.
- For `WORKSF` (code and environments) it checked bytes with `du -bs` (2TB quota) and inodes with `du --inodes` (3M quota), alerting at 85%.

`du` walks every file and can be very slow on a big directory tree, which is why the quota tools were preferred where they worked. Like slurm-status.py, the script checks it is running on Jean Zay and hardcodes paths, quotas and addresses, so adapting it means editing those.

A generic version with no cluster-specific tools, covering free bytes, free inodes and log staleness (the [hang watchdog](#is-job-hanging-watchdog)). `os.statvfs` is the system call `df` uses **[ran]**:

```python
# watchdog.py -- generic disk/inode/log-staleness check; run it from cron or a cron-emulating Slurm job
import os, sys, time
# path -> (min free TB, min free inode fraction); size these from your write rate, not a round %
CHECKS = {"/tmp": (0.05, 0.10), "/": (0.01, 0.05)}
LOG, MAX_SILENCE_MIN = "train.log", 30             # training logs every few minutes
alerts = []
for path, (min_tb, min_inode_frac) in CHECKS.items():
    st = os.statvfs(path)                           # what `df` and `df -i` read
    free_tb = st.f_bavail * st.f_frsize / 1e12      # f_bavail: blocks available to non-root users
    inode_frac = st.f_favail / st.f_files if st.f_files else 1.0
    print(f"{path:5} free {free_tb:6.2f} TB, free inodes {inode_frac:6.1%}")
    if free_tb < min_tb: alerts.append(f"{path}: only {free_tb:.2f} TB free")
    if inode_frac < min_inode_frac: alerts.append(f"{path}: only {inode_frac:.1%} inodes free")
if os.path.exists(LOG):
    silent = (time.time() - os.path.getmtime(LOG)) / 60
    print(f"{LOG} last written {silent:.1f} min ago")
    if silent > MAX_SILENCE_MIN: alerts.append(f"{LOG} silent for {silent:.0f} min: hung?")
if alerts:
    print("ALERT:", *alerts, sep="\n  ")             # replace with smtplib / a Slack webhook
    sys.exit(1)
```

On the laptop, with a `train.log` last touched 78 minutes earlier:

```
/tmp  free   0.03 TB, free inodes  98.7%
/     free   0.03 TB, free inodes  99.8%
train.log last written 77.8 min ago
ALERT:
  /tmp: only 0.03 TB free
  train.log silent for 78 min: hung?
```

How to read it: the laptop really did have only about 30GB free, below the 0.05TB threshold set for `/tmp`, and the log check fired because the file's modification time was older than 30 minutes. With the BLOOM numbers, a sensible byte threshold for the checkpoint partition would be at least a couple of days of writes, about 40TB. On a parallel filesystem, `statvfs` may report stale numbers (the 30-minute delay above), and per-user or per-project quotas are not visible to it at all, which is why fs-watchdog.py used the cluster's quota tool.

### Dealing with slow memory leaks

Some programs have tiny memory leaks that are very hard to track down. Don't confuse them with [memory mapping](../foundations/storage-and-data.md#safetensors-a-header-plus-raw-bytes) (mmap: mapping a file into memory so that its pages are read into RAM on first access). A program that reads its data through mmap can appear to use more and more CPU memory over time, but that memory is page cache that the kernel frees when it's needed elsewhere, so it isn't a leak. [A Deep Investigation into MMAP Not Leaking Memory](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/) explains why, and [RSS vs VSZ](../foundations/linux-and-os.md#rss-vs-vsz-and-pss) explains which numbers to look at.

Ideally you'd find and fix the leak. But sometimes it is in a third-party package or very hard to diagnose, and there isn't time.

On the GPU side, there is also **memory fragmentation** (see [the caching allocator](arithmetic.md#pytorch-caching-allocator-internals)). Over time, many small unused memory segments accumulate. The GPU appears to have plenty of free memory, but when the program tries to allocate a large tensor it fails with an OOM like:

```
RuntimeError: CUDA out of memory. Tried to allocate 304.00 MiB (GPU 0; 8.00 GiB total capacity;
142.76 MiB already allocated; 6.32 GiB free; 158.00 MiB reserved in total by PyTorch)
```

How can 304MiB fail to allocate when 6.32GiB are free? Because the free memory is scattered across many small non-contiguous blocks, none of which is 304MiB. [OOM triage](profiling-and-optimization.md#oom-triage) shows how to confirm this from a memory snapshot.

During IDEFICS-80B training, the team had a small CPU memory leak that took several days to exhaust the CPU memory. Their workaround was a **watchdog inside the training loop**: check memory usage, and if it passes a threshold, save and exit voluntarily. The next queued job then resumes with all the CPU memory reclaimed. This is the same idea as the kill and save switches, but it watches memory usage instead of a file. To exit when the OS reports that less than 5% of virtual memory is left:

```python
import sys
import psutil
for batch in iterator:
    total_used_percent = psutil.virtual_memory().percent # 0-100
    if total_used_percent > 95:
        print(f"Exiting early since the cpu memory is almost full: ({total_used_percent}%)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

This is the `cpu memory` branch of [`ft_loop.py`](#a-training-loop-with-every-switch), which was triggered there with an artificially low threshold.

The same approach works for GPU memory, but you have to account for the CUDA caching allocator and Python's garbage collection schedule. To see the real usage, first run the garbage collector, then return the cached blocks: `torch.cuda.empty_cache()` gives the blocks PyTorch's caching allocator holds but isn't using back to the driver, and `torch.cuda.mem_get_info()` returns the device's `(free, total)` bytes as the driver sees them. Only then do you get the real numbers, and you can exit gracefully if the GPU is too close to full **[untested]**:

```python
import gc
import sys
import torch

for batch in iterator:
    gc.collect()
    torch.cuda.empty_cache()

    # get mem usage in GiBs and exit if less than 2GiB of free GPU memory remain
    free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info())
    if free < 2:
        print(f"Exiting early since the GPU memory is almost full: ({free}GB remain)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

Caveats:

- **Don't do this unless you have to.** The cache is what makes allocation fast, and `gc.collect()` plus `empty_cache()` every step is slow. Try to fix the fragmentation instead: look up `max_split_size_mb` (and `expandable_segments`) in the docs for [`PYTORCH_ALLOC_CONF`](https://docs.pytorch.org/docs/stable/notes/cuda.html#environment-variables), which controls how the allocator splits and grows memory. Some frameworks, like [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), avoid fragmentation entirely by pre-allocating their tensors at startup and reusing them.
- **Multi-node.** These examples decide per process. In a distributed job, gather the stats from all ranks, act on the one with the least memory left, and make every rank exit at the same step: that is the `any_rank()` all-reduce in `ft_loop.py`.

A note on workarounds. In real training, not every problem can be fixed with the time and people you have, and a solid workaround often gets you to the finish line sooner than "stopping the presses" and possibly delaying the training by weeks while you hunt for the cause. BLOOM-176B was trained with `CUDA_LAUNCH_BLOCKING=1` (which makes every CUDA kernel launch synchronous, so the CPU waits for each kernel to finish before continuing), because the training hung without it, and after several failed attempts to diagnose the hang the team couldn't afford to wait any longer. Luckily, this variable, normally used only for debugging and which in theory should slow some CUDA operations down, made no measurable difference to throughput. The root cause was never found, and it no longer matters, since the team moved on to other projects that aren't affected.

## Dealing with forced job preemption

The [kill switch](#kill-switch) stops training on demand. This section is about stops you don't choose.

On HPC clusters, Slurm jobs have a maximum run time, typically something like 20 hours. Resources are shared between many users and groups, so each gets a time slice, and then the job is forcibly stopped so others can use the nodes. (This also means you can't plan how long the training will take unless your jobs have the highest priority on the cluster: otherwise it is common to wait hours, sometimes days, for the next job to start.)

You could let the job be killed and resume from [the last checkpoint](#frequent-checkpoint-saving), losing everything since then. That is wasteful. The efficient solution is to **exit gracefully before the hard limit**: finish the current step, save, exit.

First, work out how much time a graceful exit needs. Two durations:

1. how long one iteration takes, in case one has just started, and
2. how long a checkpoint save takes.

If an iteration takes at most 2 minutes and a save another 2 minutes, you need at least 4 minutes. Double it to be safe. Exiting a little early wastes nothing, since the next job resumes from the checkpoint.

Say your cluster allows 100-hour jobs, so the Slurm script says:

```
#SBATCH --time=100:00:00
```

[How jobs get stopped](../orchestration/slurm/from-zero.md#how-jobs-get-stopped) lists what Slurm sends in each case and how much warning you get; [the self-requeuing batch script](../orchestration/slurm/from-zero.md#the-self-requeuing-batch-script) is a complete script that checkpoints on a signal and requeues itself. Below are the variants in more detail.

### Approach A. Tell the program at launch time when it should start the exiting process

```bash
srun ... torchrun ... --exit-duration-in-mins 5990
```

100 hours is 6000 minutes, so this gives the program 10 minutes to exit gracefully. At startup the program records the time, and before every iteration it checks whether the limit has been reached. If it has, it saves and exits.

Case study: this is how it was set [in the BLOOM training job](https://github.com/bigscience-workshop/bigscience/blob/58d99c67f643d27b5765a73a2ee2d1ce0a4b2c6b/train/tr11-176B-ml/tr11-176B-ml.slurm#L97-L100), and acted upon [in Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L985-L998):

```python
        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()
```

Because training is distributed, the exit has to be synchronized across ranks. Each rank's clock started at a slightly different moment, so one rank may cross the limit an iteration earlier than another. The `all_reduce` with `ReduceOp.MAX` makes every rank see `done=1` as soon as any rank has passed the limit, so they all save and exit at the same iteration, instead of some ranks exiting while the rest hang in the next collective. `ft_loop.py`'s `time limit` branch [above](#a-training-loop-with-every-switch) is the same logic, and its run 4 shows it working.

You can also derive the limit automatically, from the running job's `EndTime`:

```bash
$ scontrol show -d job $SLURM_JOB_ID | grep Time
   RunTime=00:00:42 TimeLimit=00:11:00 TimeMin=N/A
   SubmitTime=2023-10-26T15:18:01 EligibleTime=2023-10-26T15:18:01
   AccrueTime=2023-10-26T15:18:01
   StartTime=2023-10-26T15:18:01 EndTime=2023-10-26T15:18:43 Deadline=N/A
```

and compare it with the current time inside the program, leaving your grace period. The output has other timestamps and durations you could use too (see [`scontrol`](../orchestration/slurm/from-zero.md#scontrol-the-full-record-and-changing-things)).

### Approach B.1. Sending a custom signal X minutes before the end

In the sbatch script, set:

```
#SBATCH --signal=USR1@600
```

and Slurm sends a `SIGUSR1` signal to your program 10 minutes (600 seconds) before the job's end time.

By default, Slurm sends `SIGCONT`+`SIGTERM` about 30-60 seconds before the time is up, and `SIGCONT`+`SIGTERM`+`SIGKILL` when it is up if the job is still running. `SIGTERM` can be caught, but 30 seconds isn't enough to exit a large training program gracefully. Hence a custom, earlier signal. (See [signals](../foundations/linux-and-os.md#signals) for the basics.)

How signal sending and trapping works. In terminal A **[ran]**:

```bash
python -c "
import time, os, signal

def sighandler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```

It prints its pid, e.g. `4034989`, and goes to sleep (pretending to work). In terminal B, send it `SIGUSR1`:

```bash
kill -s USR1 4034989
```

The program catches the signal, runs `sighandler`, prints and exits:

```
Signal handler called with signal 10
```

`10` is the number of `SIGUSR1` on Linux. On macOS, where I ran this, it prints `Signal handler called with signal 30`, because signal numbers differ between operating systems; `kill -l USR1` prints the local number. Use names (`signal.SIGUSR1`, `kill -s USR1`) rather than numbers.

The same thing under Slurm **[untested]**:

```bash
$ cat sigusr1.slurm
#SBATCH --job-name=sigusr1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:03:00
#SBATCH --partition=mypartition
#SBATCH --output=%x-%j.out
#SBATCH --signal=USR1@170

srun python -c "
import time, os, signal

def sighandler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```

The job runs for 180 seconds (3 minutes), and Slurm is asked to send the signal 170 seconds before the end. After:

```bash
sbatch sigusr1.slurm
```

the job exits 10 seconds (`180-170`) after it starts, with the log:

```
58307
Signal handler called with signal 10
```

meaning the process had pid `58307`, caught `SIGUSR1` (`10`) and exited. (Slurm documents the signal time as accurate to within 60 seconds, because of how often it checks, so don't cut it finer than that.)

In a real program, the handler should not call `exit(0)` directly. Instead it sets an exit-as-soon-as-possible flag, the loop finishes the current iteration, sees the flag, saves and exits, just like [Approach A](#approach-a-tell-the-program-at-launch-time-when-it-should-start-the-exiting-process). The `SIGUSR1` branch of [`ft_loop.py`](#a-training-loop-with-every-switch) does exactly that.

### Approach B.2. Choosing which process to send the signal to

If your training program isn't the process `srun` starts, because you use a launcher like `torchrun` or `accelerate`, the recipe above doesn't work: the signal goes to the launcher, and launchers generally don't forward `SIGUSR1` to their children. The Slurm script needs to do a little more.

Replace:

```
#SBATCH --signal=USR1@600
```

with:

```
#SBATCH --signal=B:USR1@600
```

The `B:` tells Slurm to send the signal not to the `srun` processes but to the batch shell (the shell running the sbatch script).

Then change the end of the script from the usual launcher form:

```bash
CMD="python -u -m torch.distributed.run ... train.py ..." # real command here
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE
```

to:

```bash
trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|deepspeed|torchrun|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; \
wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # real command here
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

The batch shell now receives the signal, and `trap` (a bash builtin that runs the given command string when the shell receives the named signal) handles it. The handler finds every process that is an immediate child of one of the launchers (`accelerate`, `deepspeed`, `torchrun`, `torch.distributed.run`) and sends it `SIGUSR1`. Step by step:

- `pgrep -f` matches the regex against each process's full command line and returns the launcher's pid,
- `pgrep -P $pid` lists that pid's children, which are the actual training processes,
- `xargs -r kill -USR1` sends each of them `SIGUSR1` (`-r` skips running `kill` if the list is empty).

The last change: for `trap` to run, the shell must not be blocked in a foreground command, so `srun` is run in the background with `&`, followed by `wait` so the batch shell doesn't exit before `srun` finishes. (Bash only runs a trap handler once the current foreground command returns; `wait` is interruptible, a foreground `srun` is not.)

Note that `pgrep -f` here runs on the node where the batch shell runs, so it only finds the launcher on that node. With one node that is fine. With several nodes, the launchers on the other nodes never get the signal; either drop `B:` and make sure each node's launcher forwards it, or rely on the `any_rank()` all-reduce: once the ranks on the first node have the flag, the `MAX` all-reduce spreads the decision to all ranks at the same step. `ft_loop.py` does exactly that, which is why sending the signal to any subset of ranks is enough.

The Python side is the same as in Approach B.1. All the important parts of the Slurm script together **[untested]**:

```bash
$ cat launch.slurm
#!/bin/bash
[...]
#SBATCH --partition=dev
#SBATCH --signal=B:USR1@600 # custom preemption signal, 10 min before the end
[...]

trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|torchrun|deepspeed|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # real command here
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

A training loop that looked like:

```python
for batch in dl:
    train_iteration(batch)
```

becomes:

```python
import signal
import sys

pre_emption_activated = False
def activate_pre_emption(sig, frame):
    global pre_emption_activated
    print("SIGUSR1 received, saving checkpoint")
    pre_emption_activated = True

signal.signal(signal.SIGUSR1, activate_pre_emption)

for batch in dl:
    train_iteration(batch)

    if pre_emption_activated:
        save_checkpoint()
        sys.exit()
```

In real code you'd keep the flag on the trainer object rather than in a `global`, but this keeps the demo short. In a distributed job, combine it with the all-reduce as above.

To test it, change the Slurm header to:

```
#SBATCH --time=0:05:00
#SBATCH --signal=B:USR1@60
```

This runs the job for 5 minutes (`--time=0:05:00`) and asks Slurm to send `SIGUSR1` to the batch script 60 seconds before the end, i.e. 4 minutes after the job starts.

### QoS-based SLURM preemption

A job can also be preempted by Slurm's **QoS** (quality of service) mechanism (see [accounts, associations, QoS and fair share](../orchestration/slurm/from-zero.md#accounts-associations-qos-and-fair-share) and [preemption internals](../orchestration/internals.md#preemption)). It works like time-limit preemption, except it can happen at **any time**, not at a known time before the end of the job.

Consider a setup where `--qos=high` can preempt `--qos=low` jobs, and a low-priority job gets 10 minutes to shut down:

```bash
$ sacctmgr show qos format=name,priority,preempt,MaxTRESPerUser,GraceTime,Preempt,Flags
      Name   Priority     MaxTRESPU  GraceTime    Preempt                Flags
---------- ---------- ------------- ---------- ---------- --------------------
       low          0                 00:10:00
      high          0                 00:00:00        low
```

How to read it: the `Preempt` column of `high` lists `low`, so a high job may preempt low jobs, and `low`'s `GraceTime` of 10 minutes is the time between the preemption signal (`SIGCONT`+`SIGTERM`) and `SIGKILL`. Unlike `--signal`, the grace time is set by the admin and you can't change it.

If a job is submitted with `--qos=high` and there aren't enough free nodes, Slurm kicks out some low-priority jobs to make room.

The default `GraceTime` can be too short for your program to wind down safely. If it is, ask your sysadmin to raise it to what you need (one iteration plus one save, doubled). Otherwise, the same solutions as in Approaches B.1 and B.2 work, except that the signal to handle is `SIGTERM` instead of `SIGUSR1`: register the same handler for both.

## Reproducibility

### Achieve determinism in randomness based software

When debugging, always set a fixed seed for every random number generator (RNG) your code uses, so that each re-run sees the same data and takes the same code path.

On GPU, seeding alone isn't enough for bitwise-identical results. cuDNN and cuBLAS are free to pick different (but individually valid) algorithms from run to run for the same op, and some algorithms are themselves non-deterministic (for example, ones that add with atomics in whatever order threads finish). A truly deterministic re-run also requires forcing them to always pick the same, deterministic algorithms. That is what the `cudnn.benchmark`, `use_deterministic_algorithms()` and `CUBLAS_WORKSPACE_CONFIG` settings below do.

Many different systems are involved, so it's hard to cover them all, but here are the most common knobs in one function:

```python
import os, random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    if use_seed is not None: # slower speed! https://docs.pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        # CUBLAS_WORKSPACE_CONFIG must be set before any CUDA context gets created, i.e. before
        # the torch.cuda.* calls below - if CUDA is initialized earlier elsewhere it's too late,
        # and silently so, hence the explicit check instead of letting it fail quietly
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ and torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA was already initialized before enforce_reproducibility() got to set "
                "CUBLAS_WORKSPACE_CONFIG - move this call earlier, or export "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 before the process starts"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # multi-gpu - can be called without gpus

    return seed
```

Notes:

- **Leave all the seeding calls in.** You could remove the ones you think you don't need, e.g. `np.random.seed` if you don't use numpy directly. But a library you depend on may use it internally without your knowledge, so it is safer to seed everything.

- **`CUBLAS_WORKSPACE_CONFIG`** closes a gap that `use_deterministic_algorithms(True)` doesn't close by itself. cuBLAS, the library behind `torch.mm`, `mv`, `bmm` and other matmul-based ops, can be non-deterministic on CUDA 10.2 and later, because it dynamically picks internal workspace memory for routines running in parallel CUDA streams on the same handle, so results can differ from run to run even with the same seed. Without this variable, `use_deterministic_algorithms(True)` raises a `RuntimeError` as soon as it hits a matmul, which is your signal to set it. It **must be set before the CUDA context is created**, i.e. before the first CUDA call. If it is set after CUDA has already been initialized elsewhere in the process, it is silently ignored, which is why the function sets it before any of its own `torch.cuda.*` calls.

  Because a silently ignored variable is easy to miss, the function uses [`torch.cuda.is_initialized()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html) to raise an error instead if `CUBLAS_WORKSPACE_CONFIG` isn't set yet and CUDA has already been initialized by an earlier call (a `.cuda()` tensor, `torch.cuda.set_device()`, another library touching CUDA first). This only catches initialization that went through PyTorch's own lazy-init path: if a non-PyTorch library created a raw CUDA context first, `is_initialized()` won't know. So call `enforce_reproducibility()` as early in your program as possible, or simply export the variable in the launch script.

  NVIDIA's cuBLAS docs sanction exactly two values for deterministic behavior. Both trade memory against speed: a smaller workspace leaves cuBLAS fewer algorithms to choose from, which can cost performance, while a larger one gives it more room to pick a fast algorithm at the cost of some GPU memory:

  - `:16:8` is deterministic and uses minimal extra memory, but the smaller workspace may limit performance.
  - `:4096:8` is deterministic and costs about 24MiB of extra GPU memory, but leaves more performance headroom. It is the more commonly used of the two, and the one used above.

  There is no third, free option. Determinism is fundamentally at odds with letting cuBLAS freely pick whichever algorithm and workspace layout is fastest across concurrent streams, so pinning it down always gives up some of that freedom. `:4096:8` usually has the smaller penalty of the two (more workspace means fewer algorithms are ruled out), but smaller isn't zero: the actual slowdown depends on the workload and the matmul shapes, and can range from negligible to significant (some reports show throughput roughly halved on specific workloads).

  The general format is `:[SIZE]:[COUNT]` (SIZE in KiB), and you'll see it chained into several `SIZE:COUNT` pairs. For example, the default behavior when the variable is unset is equivalent to `CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8`. The chained form comes from cuBLAS's internal workspace pool configuration, where each pair originally described a separate bucket size and count, so the pool could serve both large and small requests efficiently. PyTorch, however, just sums every `SIZE * COUNT` pair into one total and hands that single fixed-size buffer to cuBLAS. So `:4096:2:16:8` simply means `2*4096 + 8*16` KiB in total, and chaining more pairs only changes how many KiB you add up, not the determinism. In practice you don't need to chain anything: pick one of NVIDIA's two values above, or set `:0:0` to forbid cuBLAS from using any workspace at all (not recommended: it is the most restrictive and slowest option).

- **`torch.use_deterministic_algorithms(True)`** makes cuDNN convolutions deterministic, and also covers hundreds of other ops throughout PyTorch (`index_add_`, `scatter_`, interpolation and so on). Instead of silently doing its best, it raises a `RuntimeError` the moment it hits an op that has no deterministic implementation, so non-determinism can't sneak in unnoticed. (`warn_only=True` turns the error into a warning; see the docs linked below.)

- **`torch.backends.cudnn.benchmark = False`** is a separate, complementary knob. With benchmarking on, cuDNN times several convolution algorithms the first time it sees each new input shape and caches the fastest. That timing step is itself a source of run-to-run variation, because which algorithm "wins" can depend on incidental system noise, independently of whether the winner is deterministic.

> **PyTorch version note**: `torch.use_deterministic_algorithms()` requires **PyTorch >= 1.8**. On older versions, fall back to the narrower `torch.backends.cudnn.deterministic = True`, which only makes cuDNN convolutions deterministic and leaves everything else (matmuls, `index_add_`, `scatter_`, etc.) untouched. `CUBLAS_WORKSPACE_CONFIG` is independent of the PyTorch version and still applies:
> ```python
> if use_seed:
>     os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
>     torch.backends.cudnn.benchmark     = False
>     torch.backends.cudnn.deterministic = True # PyTorch < 1.8 fallback - cuDNN convolutions only
> ```

Resources:

- [NVIDIA cuBLAS docs - Reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility): the canonical explanation of cuBLAS non-determinism and the accepted `CUBLAS_WORKSPACE_CONFIG` values.
- [`torch.use_deterministic_algorithms`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html): the full list of affected ops and how `warn_only` works.
- [PyTorch CUDA environment variables reference](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html): the exact `:[SIZE]:[COUNT]` format of `CUBLAS_WORKSPACE_CONFIG`.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html): ties together seeding, `cudnn.deterministic` and `cudnn.benchmark`.

If your code also uses other accelerator backends or frameworks besides CPU and CUDA, for example NPU, XPU or TensorFlow, seed their RNGs too:

```python
    torch.npu.manual_seed_all(seed)
    torch.xpu.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

**How to use it.** When you re-run the same code over and over to chase a problem, set a specific seed at the start of your program:

```python
enforce_reproducibility(42)
```

This is **for debugging only**: it turns on `torch` determinism flags that can slow things down, so you don't want it in production.

In production, call it **without** a seed:

```python
enforce_reproducibility()
```

It then picks a random seed and logs it. If something goes wrong in production, you can reproduce the RNG state the problem happened under, and there is no performance penalty, because the determinism settings are only applied when you pass a seed explicitly. Say it logged:

```
Using seed: 1234
```

then you change the call to:

```python
enforce_reproducibility(1234)
```

and you get the same RNG setup.

There may be many other RNGs in a system. For example, if you want a `DataLoader` to feed data in the same order, you need to [give it a seeded generator too](https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader), and seed numpy and `random` inside each worker process. A complete demo, run on CPU twice with seed 42, twice without, and then replaying one of the logged seeds **[ran]**:

```python
# repro.py -- seed everything, log the seed, and make the DataLoader order reproducible too
import os, random, sys, torch, numpy as np
from torch.utils.data import DataLoader
# ... enforce_reproducibility() exactly as above ...

def seed_worker(worker_id):
    # each worker's torch seed is already base_seed + worker_id; derive numpy/random from it
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

if __name__ == "__main__":   # DataLoader workers are spawned processes that re-import this file
    seed = enforce_reproducibility(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    w = torch.nn.Linear(4, 1).weight                     # (1, 4) init depends on the torch RNG
    g = torch.Generator().manual_seed(seed)              # the DataLoader's shuffle RNG
    dl = DataLoader(range(10), batch_size=5, shuffle=True, generator=g, num_workers=2,
                    worker_init_fn=seed_worker)
    print("init :", w.detach().numpy().round(4))
    print("order:", [b.tolist() for b in dl])
    print("np   :", np.random.rand(2).round(4), " deterministic:", torch.are_deterministic_algorithms_enabled())
```

```
$ python repro.py 42
Using seed: 42
init : [[ 0.3823  0.415  -0.1171  0.4593]]
order: [[6, 5, 4, 0, 8], [9, 2, 1, 3, 7]]
np   : [0.3745 0.9507]  deterministic: True
$ python repro.py 42
Using seed: 42
init : [[ 0.3823  0.415  -0.1171  0.4593]]
order: [[6, 5, 4, 0, 8], [9, 2, 1, 3, 7]]
np   : [0.3745 0.9507]  deterministic: True
$ python repro.py
Using seed: 288199
init : [[ 0.021   0.128  -0.3217  0.1833]]
order: [[9, 2, 8, 5, 3], [7, 1, 4, 6, 0]]
np   : [0.3692 0.5515]  deterministic: False
$ python repro.py 288199
Using seed: 288199
init : [[ 0.021   0.128  -0.3217  0.1833]]
order: [[9, 2, 8, 5, 3], [7, 1, 4, 6, 0]]
np   : [0.3692 0.5515]  deterministic: True
```

How to read it: the two seed-42 runs agree on the initial weights, the batch order and numpy's draws. The unseeded run picks and logs `288199`, without the determinism flags (`deterministic: False`, so no slowdown). Replaying `288199` reproduces that run exactly, now with the flags on. On a CPU the deterministic-kernels part is mostly moot; on a GPU it is what makes the forward and backward bitwise repeatable. The first time I wrote this demo, the worker init function was a `lambda` and the script had no `__main__` guard. On Python 3.14 that fails with `PicklingError: Can't pickle <function <lambda>>`, because DataLoader workers are started with spawn/forkserver and everything passed to them must be importable and picklable. Hence the named function and the guard.

Seeding gets you the same run on the same setup. It won't give you the same numbers across a different number of GPUs, a different NCCL algorithm or a different GPU model, because floating-point addition isn't associative. See [determinism across ranks](../foundations/distributed-systems.md#determinism-and-reproducibility-across-ranks) for the multi-GPU side and [debugging PyTorch](../development/debugging-pytorch.md) for floating-point discrepancies between devices.

### Reproduce the software and system environment

This is useful when you find a discrepancy in outcomes, quality or throughput, between two runs that "should" be the same. The idea is to record the key parts of the environment every training (or inference) job was launched with, so that it can be reproduced exactly later, or compared with another run.

There is a huge variety of systems, so no single recipe works everywhere. Here is one that you can adapt. It goes into the Slurm launcher script (or whatever you launch training with):

```bash
SAVE_DIR=/tmp # edit to a real persistent path
export REPRO_DIR=$SAVE_DIR/repro/$SLURM_JOB_ID
mkdir -p $REPRO_DIR
# 1. modules (writes to stderr) (remove if you don't use lmod or similar modules implementation)
module list 2> $REPRO_DIR/modules.txt
# 2. shell env vars
/usr/bin/printenv | sort > $REPRO_DIR/env.txt
# 3. pip (this includes devel installs SHA)
pip freeze > $REPRO_DIR/requirements.txt
# 4. uncommitted diff in git clones installed into conda
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; if [ ! -z "\$(git diff)" ]; then git diff > \$REPRO_DIR/$2.diff; fi]' $CONDA_PREFIX/lib/python*/site-packages/*.dist-info/direct_url.json
```

Since it runs inside a Slurm job, every new training writes its environment to a directory named after the job ID. Step by step:

1. **Modules.** Saves which [environment modules](../foundations/linux-and-os.md#environment-modules-lmod) were loaded. On HPC clusters, CUDA and cuDNN are often loaded this way. If you don't use modules, remove this line.

2. **Environment variables.** A single variable can make a huge difference to performance in some environments, for example `LD_PRELOAD` (a list of shared libraries the dynamic loader loads before all others, overriding symbols from the regular ones) or `LD_LIBRARY_PATH` (see [shared libraries](../foundations/linux-and-os.md#shared-libraries-and-ld_library_path)). Note that `env.txt` may contain secrets (tokens, keys), so store it where only your team can read it, or filter those variables out.

3. **Python packages.** `pip freeze` lists every installed package and its version, and works with conda, `venv`, or no virtual environment at all. If you use `uv`, use `uv pip freeze`, which is much faster.

4. **Uncommitted changes to editable installs.** With an editable install (`pip install -e .`, where site-packages points at your git clone instead of a copy of the code), pip only knows the clone's git SHA. But you have probably modified files locally, and `pip freeze` misses those changes. This step goes through every package installed from a local path (found via `site-packages/*.dist-info/direct_url.json`) and, for each one with uncommitted changes, saves its `git diff` to `$REPRO_DIR/<package>.diff`.

   How to read the perl one-liner: `perl -nle '...' files` runs the quoted code on every line of every `direct_url.json` file (`-n` loops over input lines, `-l` strips newlines, `-e` gives the code inline). The regex captures the local path of a package installed from a `file://` URL (`$1`) and its directory name (`$2`), and `qx[...]` runs the bracketed shell command in that directory, writing `git diff` to `$REPRO_DIR/<dirname>.diff` if the diff is non-empty. `$CONDA_PREFIX` is set by conda to the active environment's root; with a `venv`, use `$VIRTUAL_ENV`.

To save the system (`apt`) packages too, add:

```bash
apt list --installed > $REPRO_DIR/apt-packages.txt
```

I ran the recipe outside Slurm, in a fresh `venv` with one editable package (`mypkg`, a git repo with one commit) whose source I then edited **[ran]**. The changes from the version above: the directory falls back to a timestamp when `SLURM_JOB_ID` isn't set, `module list` is allowed to fail, the env root falls back to `$VIRTUAL_ENV`, and two lines record the training repo's own commit and the command line:

```bash
# capture_env.sh -- the environment-capture recipe, adapted to a venv and runnable outside Slurm
SAVE_DIR=$PWD
export REPRO_DIR=$SAVE_DIR/repro/${SLURM_JOB_ID:-local-$(date +%s)}
mkdir -p $REPRO_DIR
module list 2> $REPRO_DIR/modules.txt || true          # 1. Lmod modules (no-op without Lmod)
/usr/bin/printenv | sort > $REPRO_DIR/env.txt          # 2. env vars
pip freeze > $REPRO_DIR/requirements.txt               # 3. packages (+ editable installs' git SHA)
ENV_ROOT=${CONDA_PREFIX:-$VIRTUAL_ENV}                 # 4. uncommitted diffs of editable installs
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; if [ ! -z "\$(git diff)" ]; then git diff > \$REPRO_DIR/$2.diff; fi]' $ENV_ROOT/lib/python*/site-packages/*.dist-info/direct_url.json
git -C ${TRAIN_REPO:-.} rev-parse HEAD > $REPRO_DIR/git-sha.txt 2>/dev/null || true
echo "$0 $*" > $REPRO_DIR/cmd.txt
ls $REPRO_DIR
```

```
cmd.txt
env.txt
git-sha.txt
modules.txt
mypkg.diff
requirements.txt
$ cat repro/local-*/requirements.txt
# Editable Git install with no remote (mypkg==0.1)
-e /private/tmp/ft-work/mypkg
$ cat repro/local-*/mypkg.diff
diff --git a/mypkg/__init__.py b/mypkg/__init__.py
index 9676680..55b86d2 100644
--- a/mypkg/__init__.py
+++ b/mypkg/__init__.py
@@ -1 +1 @@
-X = 1
+X = 2  # local hack
```

How to read it: `pip freeze` recorded only the path of the editable package (with a git remote configured it would record `-e git+<url>@<sha>`, but never the local edits), while step 4 captured the local edit that `pip freeze` missed.

If you use conda, [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md) shows the exact differences between two conda environments. For two `requirements.txt` files, `diff <(sort a.txt) <(sort b.txt)` does the same job.

An anecdote from the original author (Stas Bekman) shows why step 2 matters. He and a colleague were getting very different training TFLOPS on a cloud cluster running exactly the same code, literally launching the same Slurm script from the same shared directory. They first compared their conda environments with conda-env-compare.pl and found some differences; he installed exactly her packages, and the performance gap remained. Then they compared their `printenv` output, and found that he had `LD_PRELOAD` set and she didn't. That made a huge difference, because this cloud provider required several environment variables to point at custom paths to get the most out of their hardware.

## Preprocessing datasets on the main process

When several processes start at once and each one downloads and preprocesses the same dataset, two things go wrong. They all repeat the same work in parallel, and they are likely to write to the same cache location at the same time, interleaving their writes and leaving a broken result. It is also much faster in terms of I/O to do this work once. So you want one process to download and preprocess the data, and the rest to wait and then load the cached result.

[HF Accelerate](https://huggingface.co/docs/accelerate) (Hugging Face's thin wrapper library for running the same PyTorch training loop on one or many GPUs) provides a context manager (a `with` block) for this, [`main_process_first`](https://huggingface.co/docs/accelerate/v0.4.0/accelerator.html#accelerate.Accelerator.main_process_first):

```python
with accelerator.main_process_first():
    # load and pre-process datasets
    dataset = datasets.load_dataset(...)
    # optionally cache it and have the rest of the processes load the cache
```

instead of the less intuitive version that repeats the code (here `rank` is this process's global rank and `dist.barrier()` blocks until every rank reaches it, see [barriers](../foundations/distributed-systems.md#barriers)):

```python
if rank == 0:
    dataset = datasets.load_dataset(...)
dist.barrier()
if not rank == 0:
    dataset = datasets.load_dataset(...)
```

Accelerate has two variants, and which one you need depends on where the data lives (see [local vs network filesystems](../foundations/linux-and-os.md#local-vs-network-filesystems)):

- `main_process_first` is for data on a **shared filesystem** that all nodes see. Only global rank 0 does the work, and every other process on every node waits.
- `local_main_process_first` is for data on storage **local to each node**. Local rank 0 (the first process on each node) does the work on each node, all nodes in parallel.

If you aren't using Accelerate, [training/tools/main_process_first.py](tools/main_process_first.py) has equivalent context managers:

- `global_main_process_first()` for a shared filesystem,
- `local_main_process_first()` for node-local storage,
- `main_process_by_path_first(path)`, which detects what kind of filesystem `path` is on and picks one of the two:

```python
path = "/path/to/data"
with main_process_by_path_first(path):
    # load and pre-process datasets
    dataset = datasets.load_dataset(...)
    # optionally cache it and have the rest of the processes load the cache
```

and matching predicates to check which kind of main process you are: `is_main_process_by_path(path)`, `is_local_main_process()` and `is_global_main_process()`.

How it works, from the code. All three context managers share one helper: non-main processes call `dist.barrier()` **before** entering the block, and the main process calls `dist.barrier()` **after** leaving it. The barrier only completes when every rank has called it, so the main process runs the block alone, and when it finishes, its barrier releases everyone else. Filesystem detection uses `psutil.disk_partitions()` to map each mount point to its filesystem type, walks up from `path` to the nearest mount point, and treats a fixed list of types as local (`ext*`, `xfs`, `btrfs`, `zfs`, `overlay`, and so on); anything else (NFS, Lustre, GPFS, ...) counts as shared. Because the walk goes up to the nearest *existing* mount point, you can ask about a cache directory before it has been created. One warning from the code's docstrings: since these use a barrier, you can't put code that itself requires all ranks to participate (a gather, another barrier) inside the block, or it deadlocks.

The script's own demo runs with `torch.distributed.run` (the module behind `torchrun`), starting 2 processes on this machine (`--nproc_per_node=2`), with `--rdzv_endpoint`/`--rdzv_backend` telling them where and how to meet (see [rendezvous](../foundations/distributed-systems.md#rendezvous)), and `python -u` disabling output buffering so both processes' prints show up immediately:

```bash
python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d tools/main_process_first.py
```

That command is **[untested]** as written: on my Mac, the c10d rendezvous kept retrying an IPv6 reverse lookup of the hostname and never started, and once worked around with `--rdzv_endpoint 127.0.0.1:6000 --local-addr 127.0.0.1`, the script's own `dist.init_process_group()` (with no backend) picked the Apple MPS device for its barrier, which isn't implemented there. Both are macOS issues; on a Linux GPU box the default backend is NCCL. So I ran the same demo with an explicit `gloo` backend **[ran]**:

```python
# mpf_demo.py -- the __main__ demo of training/tools/main_process_first.py, with an explicit gloo backend
import time, torch.distributed as dist
from main_process_first import (global_main_process_first, local_main_process_first,
    main_process_by_path_first, is_main_process_by_path, get_global_rank, get_local_rank)
dist.init_process_group("gloo")
g, l = get_global_rank(), get_local_rank()
def show(tag): print(f"{time.strftime('%S')}s {tag} global={g} local={l}", flush=True); time.sleep(2)
with global_main_process_first(): show("global_main_process_first")
dist.barrier()
with local_main_process_first(): show("local_main_process_first ")
dist.barrier()
kind = "main" if is_main_process_by_path("./") else "non-main"
with main_process_by_path_first("./"): show(f"by_path('./') {kind:8}")
dist.destroy_process_group()
```

```bash
$ PYTHONPATH=training/tools torchrun --nproc-per-node 2 --master-addr 127.0.0.1 --master-port 29512 mpf_demo.py
40s global_main_process_first global=0 local=0
42s global_main_process_first global=1 local=1
44s local_main_process_first  global=0 local=0
46s local_main_process_first  global=1 local=1
48s by_path('./') main     global=0 local=0
50s by_path('./') non-main global=1 local=1
```

How to read it: the first column is the wall-clock second. In every block, rank 1 enters exactly 2 seconds (the length of the fake "work") after rank 0, so it really did wait for rank 0 to finish. On one node, global and local rank coincide, so the two variants look the same here; on 2 nodes, the local variant would let both nodes' local rank 0 work at the same time. One thing I noticed: `is_local_fs()` classified my Mac's `apfs` disk as *shared*, because `apfs` isn't in the script's list of local filesystem types. The result is still correct (shared is the safe assumption, just slower), but if you rely on the by-path variant, check that your filesystem types are in that list.

## Emulating a multi-node setup on one node

Multi-node bugs (wrong ranks, a launcher that assumes one node, hostfile mistakes, code that only works when `LOCAL_RANK == RANK`) are expensive to debug on a real multi-node allocation, where you might wait hours for nodes. This section emulates a 2-node environment on a single node with 2 GPUs, for testing. It extends to [larger setups](#larger-set-ups).

### With the deepspeed launcher

We use the `deepspeed` launcher here: the multi-node launcher that ships with [DeepSpeed](https://github.com/deepspeedai/DeepSpeed). It `ssh`es into every host listed in a hostfile (via `pdsh`, which runs a command on many hosts in parallel) and starts the right number of processes on each, with `RANK`, `LOCAL_RANK`, `WORLD_SIZE` etc. set. Your program doesn't need to use any DeepSpeed code; the launcher's multi-node features are just convenient. Install it with `pip install deepspeed`. (See also [hostfiles for DeepSpeed and MPI launchers](../orchestration/slurm/from-zero.md#hostfiles-for-deepspeed-and-mpi-launchers).)

Everything in this section is **[untested]** here (it needs a machine with 2+ GPUs and an ssh server); the outputs are from the original author's run in September 2022 (`torch 1.12.1+cu116`, NCCL 2.10.3, A100).

1. Create a `hostfile`: one line per node, with the hostname and, in `slots=`, how many processes (normally one per GPU) to start on it. Here both "nodes" get 1 slot:

```bash
$ cat hostfile
worker-0 slots=1
worker-1 slots=1
```

2. Add matching entries to your ssh client config, so both fake hostnames resolve to this machine (see [ssh: keys, config, tunnels](../foundations/networking.md#ssh-keys-config-tunnels)):

```bash
$ cat ~/.ssh/config
[...]

Host worker-0
    HostName localhost
    Port 22
Host worker-1
    HostName localhost
    Port 22
```

Adapt the port if your ssh server isn't on 22, and the hostname if `localhost` isn't right.

3. Your local ssh is probably password protected, so add your public key to `~/.ssh/authorized_keys`. The `deepspeed` launcher explicitly uses password-less connections; for example on worker-0 it runs `ssh -o PasswordAuthentication=no worker-0 hostname`. So you can always debug the ssh setup with:

```bash
$ ssh -vvv -o PasswordAuthentication=no worker-0 hostname
```

4. Create a test script to check that both GPUs are used:

```python
$ cat test1.py
import os
import time
import torch
import deepspeed
import torch.distributed as dist

# critical hack to use the 2nd gpu (otherwise both processes will use gpu0)
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK"))
print(f'{dist.get_rank()=}, {local_rank=}')

x = torch.ones(2**30, device=f"cuda:{local_rank}")
print(x)
time.sleep(100)
```

Run it (`-H` points the launcher at the hostfile):

```bash
$ deepspeed -H hostfile test1.py
[2022-09-08 12:02:15,192] [INFO] [runner.py:415:main] Using IP address of 192.168.0.17 for node worker-0
[2022-09-08 12:02:15,192] [INFO] [multinode_runner.py:65:get_cmd] Running on the following workers: worker-0,worker-1
[2022-09-08 12:02:15,192] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test1.py
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
worker-0: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
```

How to read this log: the launcher builds one `pdsh` command that runs `deepspeed.launcher.launch` on both workers, passing the hostfile contents base64-encoded as `--world_info` and each host's index as `--node_rank` (`pdsh` fills in `%n`). Lines prefixed `worker-0:`/`worker-1:` come from the process on that "node". Each reports `nnodes=2` and one local process, and the launcher sets `CUDA_VISIBLE_DEVICES=0` on both, because as far as it knows each node has just 1 GPU. That is why the script needs the hack at the top.

If ssh works, run `nvidia-smi` while the script sleeps: both GPUs should show about 4GiB allocated by the `torch.ones(2**30)` call (2^30 fp32 elements is 4GiB).

The hack sets `CUDA_VISIBLE_DEVICES` so that the second process uses GPU 1, but both processes see their GPU as `local_rank==0`. (`CUDA_VISIBLE_DEVICES` restricts which physical GPUs a process can see and renumbers them from 0. It must be set before CUDA is initialized in the process, which is why the hack sits at the very top of the script.)

5. Finally, test that NCCL collectives work. This script is adapted from [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py), with only the `CUDA_VISIBLE_DEVICES` tweak added (the original is covered in [debugging distributed training](../development/debugging-distributed.md)):

```python
$ cat test2.py
import deepspeed
import fcntl
import os
import socket
import time
import torch
import torch.distributed as dist

# a critical hack to use the 2nd GPU by the 2nd process (otherwise both processes will use gpu0)
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def printflock(*msgs):
    """ solves multi-process interleaved print problem """
    # fcntl.flock takes an exclusive OS-level lock on a file (here the script itself), so only one process prints at a time
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}-{local_rank}]"

try:
    # test distributed
    dist.init_process_group("nccl")
    dist.all_reduce(torch.ones(1).to(device), op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f'{dist.get_rank()=}, {local_rank=}')

    # test cuda is available and can allocate memory
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    # global rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    printflock(f"{gpu} is OK (global rank: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        printflock(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        printflock(f"device compute capabilities={torch.cuda.get_device_capability()}")
        printflock(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")

except Exception:
    printflock(f"{gpu} is broken")
    raise
```

Run:

```bash
$ deepspeed -H hostfile test2.py
[2022-09-08 12:07:09,336] [INFO] [runner.py:415:main] Using IP address of 192.168.0.17 for node worker-0
[2022-09-08 12:07:09,337] [INFO] [multinode_runner.py:65:get_cmd] Running on the following workers: worker-0,worker-1
[2022-09-08 12:07:09,337] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test2.py
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: [hope-0] is OK (global rank: 0/2)
worker-1: [hope-0] is OK (global rank: 1/2)
worker-0: pt=1.12.1+cu116, cuda=11.6, nccl=(2, 10, 3)
worker-0: device compute capabilities=(8, 0)
worker-0: pytorch compute capabilities=['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
worker-1: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] Process 576485 exits successfully.
worker-0: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] Process 576484 exits successfully.
```

Mission accomplished. In this output, `[hope-0]` is `[hostname-local_rank]`: both processes are on the same real host (`hope`), and both see their GPU as local rank 0. `nccl=(2, 10, 3)` is the NCCL version, and the two compute-capability lines compare the GPU's own architecture (`(8, 0)`, i.e. A100) with the list of architectures this PyTorch build was compiled for.

This tested that NCCL collectives work, but they ran over the node's local NVLink/PCIe, not over InfiniBand or Ethernet as on real multiple nodes. Whether that is good enough depends on what you are testing: rank logic, launchers and hostfiles, yes; network configuration and NCCL's inter-node transports, no.

### Larger set ups

Say you have 4 GPUs and want to emulate 2 nodes of 2 GPUs. Change the `hostfile` to:

```bash
$ cat hostfile
worker-0 slots=2
worker-1 slots=2
```

and the `CUDA_VISIBLE_DEVICES` hack to the following. Global ranks 0 and 1 are on `worker-0` and keep the default GPUs 0,1; ranks 2 and 3 are on `worker-1` and get GPUs 2,3, which they see as their local GPUs 0 and 1:

```python
if os.environ["RANK"] in ["2", "3"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
```

Everything else stays the same.

### Automating the process

For any topology, compute the hack automatically:

```python
def set_cuda_visible_devices():
    """
    automatically assign the correct groups of gpus for each emulated node by tweaking the
    CUDA_VISIBLE_DEVICES env var
    """

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    emulated_node_size = int(os.environ["LOCAL_SIZE"]) # set by the deepspeed launcher: processes per node
    emulated_node_rank = int(global_rank // emulated_node_size)
    gpus = list(map(str, range(world_size)))
    emulated_node_gpus = ",".join(gpus[emulated_node_rank*emulated_node_size:(emulated_node_rank+1)*emulated_node_size])
    print(f"Setting CUDA_VISIBLE_DEVICES={emulated_node_gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = emulated_node_gpus

set_cuda_visible_devices()
```

To check the arithmetic without GPUs, here is the same mapping as a pure function, for a few topologies **[ran]**:

```python
# cvd.py -- what set_cuda_visible_devices() assigns to every rank, for a few emulated topologies (pure Python)
def emulated_cvd(global_rank, world_size, emulated_node_size):
    emulated_node_rank = global_rank // emulated_node_size
    gpus = list(map(str, range(world_size)))
    return ",".join(gpus[emulated_node_rank * emulated_node_size:(emulated_node_rank + 1) * emulated_node_size])

for nodes, per_node in ((2, 1), (2, 2), (4, 2)):
    world = nodes * per_node
    print(f"{nodes} nodes x {per_node} GPUs:", {r: emulated_cvd(r, world, per_node) for r in range(world)})
```

```
2 nodes x 1 GPUs: {0: '0', 1: '1'}
2 nodes x 2 GPUs: {0: '0,1', 1: '0,1', 2: '2,3', 3: '2,3'}
4 nodes x 2 GPUs: {0: '0,1', 1: '0,1', 2: '2,3', 3: '2,3', 4: '4,5', 5: '4,5', 6: '6,7', 7: '6,7'}
```

How to read it: every rank on an emulated node sees exactly that node's GPUs, and within the node its `LOCAL_RANK` (0 or 1) indexes into them, just as on a real node. It assumes one GPU per process (world size = number of physical GPUs).

### Emulating multiple GPUs with a single GPU

A related but separate need: emulating several GPUs with **one** physical GPU.

With NVIDIA A100 and later, [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) (Multi-Instance GPU, introduced in [the GPU software stack](../compute/gpu-software-stack.md#sharing-one-gpu-mig-and-mps)) partitions one GPU into up to 7 hardware-isolated instances, each of which looks like its own GPU. But for years those instances could only be used standalone: no DDP, no NCCL communication between them. The original author hoped to carve an A100 into 7 instances, add one more real GPU and have 8 GPUs for development, but it didn't work, and NVIDIA engineers said at the time that there were no plans to support it.

That has changed. [NCCL 2.31 (2026) added experimental support for communicators that span MIG instances](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-mig-instances), so one physical GPU can now stand in for a small multi-GPU node, and you can run DDP, FSDP, ZeRO and any NCCL collective across its instances. What follows is the original author's working recipe on an H200, and his measurements of how fast those emulated GPUs talk to each other. Everything in this section is **[untested]** here; I had no GPU.

#### Getting a new enough NCCL

You need an NVIDIA driver supporting CUDA 13.0 or newer, and NCCL 2.31 or newer.

As of August 2026, the latest PyTorch, 2.13.0, bundled an older NCCL. Until a PyTorch build ships with NCCL 2.31 or newer, install NCCL yourself and load it ahead of the bundled one:

```bash
pip install -U --target $HOME/nccl-new 'nvidia-nccl-cu13>=2.31'
LD_PRELOAD=$HOME/nccl-new/nvidia/nccl/lib/libnccl.so.2 python train.py
```

It goes into its own directory rather than into the environment because PyTorch pins an exact NCCL version (`nvidia-nccl-cu13==2.29.7` for 2.13.0), and installing over it would put pip's dependency resolver at odds with torch's metadata. `LD_PRELOAD` then makes the dynamic linker resolve NCCL's symbols from the new library before it reaches the one inside the torch wheel.

To be sure, run the program once with `NCCL_DEBUG=INFO` and check that the banner says `NCCL version 2.31.2+cuda13.3` or newer. Don't use `torch.cuda.nccl.version()` for this check: it reports the version PyTorch was built against, not the preloaded one that is actually doing the work (see [which NCCL you are running](../compute/gpu-software-stack.md#which-nccl-you-are-running)).

#### The NCCL settings MIG needs

```bash
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_NVLS_ENABLE=0
```

The NCCL docs say `NCCL_MULTI_RANK_GPU_ENABLE` isn't required, since each MIG instance is a distinct CUDA device. But on an H200 with NCCL 2.31.2 the communicator won't come up without it, so set it anyway. NCCL identifies a rank's device by the parent GPU's NVML index and PCI bus id (NVML, the NVIDIA Management Library, is the driver API `nvidia-smi` is built on), not by the MIG instance. So four ranks on four instances of one GPU look to NCCL like four ranks sharing one device, and without the flag initialization stops with `Multiple Ranks are using the same GPU/Partition`.

`NCCL_NVLS_ENABLE=0` is a guard rather than a requirement. NCCL's default already leaves NVLink SHARP ([NVLS](../network/collectives-internals.md#nvls-nvlink-sharp): doing the reduction inside the NVSwitch instead of on the GPUs) off in this situation, so if nothing in your environment sets the variable, you can leave it alone. But many cluster images and pod specs export `NCCL_NVLS_ENABLE=1`, and when NVLS is explicitly demanded, NCCL refuses to start with `NCCL_NVLS_ENABLE has been set to "1" and communicator has multiple ranks using the same NVML device`. Setting it to 0 costs nothing and makes the recipe work regardless of what the environment exported. `NCCL_NVLS_ENABLE=2` also works: it uses NVLS automatically when possible.

The docs also tell you to set `NCCL_MNNVL_ENABLE=0` (multi-node NVLink). On a node with no NVLink fabric, the flag changes nothing: the same benchmark returned 142.28, 142.42 and 142.23 GBps with it unset, set to `0` and set to `1`. It only matters on fabric-attached systems like GB200 NVL72.

#### Carving up a GPU

List the profiles the GPU offers, enable MIG on it, then create the instances. An H200 allows up to 7 instances. Here we create 4 instances of 35GB each on GPU 4, which is enough for, say, SP=2 × EP=2 model parallelism ([sequence parallelism](model-parallelism/derivations.md#sequence-parallelism-and-context-parallelism), [expert parallelism](model-parallelism/derivations.md#expert-parallelism)) on a single GPU.

In the commands below, `-i 4` selects GPU 4, `-cgi` creates GPU instances from a comma-separated list of profile names (a profile name like `1g.35gb` means 1 compute slice and 35GB of memory, see the table further down), and `-C` also creates the compute instance inside each, which is what makes it usable by CUDA. To undo, `-dci` and `-dgi` destroy the compute and GPU instances respectively.

```bash
nvidia-smi mig -lgip             # show profiles available on this GPU
sudo nvidia-smi -i 4 -mig 1      # enable MIG on GPU 4
sudo nvidia-smi mig -i 4 -cgi 1g.35gb,1g.35gb,1g.35gb,1g.35gb -C
```

To undo MIG and get the full GPU back:

```bash
sudo nvidia-smi mig -i 4 -dci && sudo nvidia-smi mig -i 4 -dgi
sudo nvidia-smi -i 4 -mig 0
```

#### Finding the MIG UUIDs

Each rank must get exactly one instance through `CUDA_VISIBLE_DEVICES`, and MIG instances are addressed by UUID, not by index. `nvidia-smi -L` prints them indented under their parent GPU:

```bash
$ nvidia-smi -L
[...]
GPU 4: NVIDIA H200 (UUID: GPU-31f70a1b-dee6-617e-13c5-5c2a2fbe0013)
  MIG 1g.35gb  Device 0: (UUID: MIG-e26b1e11-8424-5ef0-97a8-d87e519f7824)
  MIG 1g.35gb  Device 1: (UUID: MIG-1a7d7808-5dcc-50fa-842c-2ceaa9f6e8ee)
  MIG 1g.35gb  Device 2: (UUID: MIG-46c30315-dfa5-56d4-82f6-345f84b4d52d)
  MIG 1g.35gb  Device 3: (UUID: MIG-6b4ff05d-689c-5bf3-8a51-3ff519c617ac)
```

Collect every rank-sized device on the node into one comma-separated variable: a GPU in MIG mode contributes its instances, a GPU not in MIG mode contributes itself. The perl one-liner pushes each `GPU-` UUID onto a list, and when a `MIG-` UUID follows, it pops the parent GPU's UUID and pushes the MIG UUID instead, then prints the list joined by commas:

```bash
export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')
```

The one-liner itself doesn't need a GPU, so I checked it on a saved `nvidia-smi -L` listing with a 2-instance GPU, a whole GPU and another 2-instance GPU **[ran]**:

```bash
$ cat smi-L.txt
GPU 0: NVIDIA H200 (UUID: GPU-aaaa)
  MIG 1g.18gb  Device 0: (UUID: MIG-a0)
  MIG 1g.18gb  Device 1: (UUID: MIG-a1)
GPU 1: NVIDIA H200 (UUID: GPU-bbbb)
GPU 4: NVIDIA H200 (UUID: GPU-31f70a1b-dee6-617e-13c5-5c2a2fbe0013)
  MIG 1g.35gb  Device 0: (UUID: MIG-e26b1e11-8424-5ef0-97a8-d87e519f7824)
  MIG 1g.35gb  Device 1: (UUID: MIG-1a7d7808-5dcc-50fa-842c-2ceaa9f6e8ee)
$ MIG_UUIDS=$(cat smi-L.txt | perl -ne '...same one-liner...')
$ for LOCAL_RANK in 0 2 4; do echo "LOCAL_RANK=$LOCAL_RANK -> $(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))"; done
LOCAL_RANK=0 -> MIG-a0
LOCAL_RANK=2 -> GPU-bbbb
LOCAL_RANK=4 -> MIG-1a7d7808-5dcc-50fa-842c-2ceaa9f6e8ee
```

How to read it: the list has 5 entries, MIG-a0, MIG-a1, GPU-bbbb and the two MIG instances of GPU 4. The partitioned GPUs' own UUIDs were replaced by their instances, and the whole GPU stayed. The `cut` expression is the one the launcher below uses to give each rank its device.

#### Launching the ranks

`torchrun` gives each process a `LOCAL_RANK`, which is all that's needed to give each one its own instance:

```bash
torchrun --nproc_per_node 4 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python train.py'
```

`--no-python` makes `torchrun` run `bash` instead of `python`, so each worker first runs the small shell script. `cut -d, -f$((LOCAL_RANK+1))` picks the `LOCAL_RANK`-th UUID from the list (`cut` counts from 1) into `CUDA_VISIBLE_DEVICES`, and `exec` replaces the shell with the training process, so signals reach it directly. Resetting `LOCAL_RANK` to `0` is required, see [caveats](#caveats).

`all_reduce`, `all_gather`, `reduce_scatter`, `broadcast` and a DDP training step all work across 4 such instances.

#### How fast do the emulated GPUs talk to each other?

The emulation is only useful if the collectives don't crawl. So here is [torch-dist-bench.py](../network/benchmarks/torch-dist-bench.py) run on MIG instances of one H200, compared with the same number of whole H200s: 4 instances against 4 real GPUs, 2 instances against 2 real GPUs.

The profiles used, all on one H200 141GB:

| MIG profile | compute slices | memory | instances per GPU |
| :---------- | -------------: | -----: | ----------------: |
| `1g.18gb`   |         1 of 7 |   18GB |                 7 |
| `1g.35gb`   |         1 of 7 |   35GB |                 4 |
| `2g.35gb`   |         2 of 7 |   35GB |                 3 |
| `3g.71gb`   |         3 of 7 |   71GB |                 2 |

The compute-slice count is the digit in the profile name (`2g.35gb` gets 2 of the GPU's 7 compute slices), and the memory figure says how many of its 8 memory slices the instance owns: 18GB is 1 slice, 35GB is 2, 71GB is 4.

Each results table starts with the whole-GPU row it is compared against, then the MIG rows in descending `busbw` (the all-reduce bus bandwidth, normalized so it can be compared with the link's peak bandwidth, see [algbw vs busbw](../network/collectives-internals.md#algbw-vs-busbw)).

4 ranks, 512MiB payload:

| what the 4 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 4x whole H200 (NVLink)  | 341.1GBps |          100% |
| 4x `1g.35gb` of one GPU | 273.8GBps |           80% |
| 4x `1g.18gb` of one GPU | 142.3GBps |           42% |

4 ranks, 2GiB payload:

| what the 4 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 4x whole H200 (NVLink)  | 354.7GBps |          100% |
| 4x `1g.35gb` of one GPU | 281.3GBps |           79% |
| 4x `1g.18gb` of one GPU | 144.7GBps |           41% |

2 ranks, 512MiB payload:

| what the 2 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 2x whole H200 (NVLink)  | 308.2GBps |          100% |
| 2x `3g.71gb` of one GPU | 413.8GBps |          134% |
| 2x `2g.35gb` of one GPU | 245.3GBps |           80% |
| 2x `1g.35gb` of one GPU | 230.9GBps |           75% |
| 2x `1g.18gb` of one GPU | 122.0GBps |           40% |

2 ranks, 2GiB payload:

| what the 2 ranks run on |     busbw | vs whole GPUs |
| :---------------------- | --------: | ------------: |
| 2x whole H200 (NVLink)  | 331.7GBps |          100% |
| 2x `3g.71gb` of one GPU | 429.5GBps |          129% |
| 2x `2g.35gb` of one GPU | 250.4GBps |           75% |
| 2x `1g.35gb` of one GPU | 235.0GBps |           71% |
| 2x `1g.18gb` of one GPU | 123.6GBps |           37% |

Setup: H200 141GB SXM, `torch==2.9.1+cu130`, NCCL 2.31.2 preloaded as above, CUDA driver 13.0.

The big surprise is how much the choice of profile matters. Traffic between MIG instances never leaves the GPU: with `NCCL_DEBUG=INFO`, NCCL logs which transport each ring hop uses, and every channel shows `P2P/CUMEM` (a direct GPU memory-to-memory copy), with no socket or PCIe fallback anywhere in the run. So the limit is the memory bandwidth the instance owns, which is set by its memory slices and hardly at all by its SMs. The two 512MiB tables isolate this:

- In the 4-rank table, going from `1g.18gb` to `1g.35gb` keeps compute identical (one slice) and doubles the memory, and bandwidth nearly doubles (142 to 274GBps).
- In the 2-rank table, going from `1g.35gb` to `2g.35gb` doubles the compute at the same memory, and buys only 6% (231 to 245GBps).

So 4x `1g.35gb` instances reproduce about 79% of the interconnect bandwidth of 4 real H200s, while 4x `1g.18gb` instances (same rank count, smaller slices) give only 41%. **Pick the profile with the most memory per instance that still gives you the rank count you need.**

And with only 2 ranks, `3g.71gb` instances are actually *faster* than 2 whole GPUs (429 against 332GBps in the 2-rank 2GiB table), because a copy within one chip's HBM beats a hop over NVLink.

#### Spreading the instances over 2 GPUs

Until NVIDIA lifts the [7-instances-per-GPU limit](#only-7-instances-per-gpu), getting to 8 ranks, for example for a small [3D parallelism](model-parallelism/derivations.md#composing-parallelisms-device-mesh-ordering) layout, needs at least 2 GPUs.

Everything above keeps the whole emulated world inside one physical GPU. Partitioning 2 GPUs into 4 instances each gives 8 ranks. Setup and benchmark:

```bash
for i in 0 1; do
    sudo nvidia-smi -i $i -mig 1
    sudo nvidia-smi mig -i $i -cgi 1g.35gb,1g.35gb,1g.35gb,1g.35gb -C
done

export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')

torchrun --nproc_per_node 8 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python torch-dist-bench.py --payload_size_in_gib 2'
```

8 ranks, 512MiB payload:

| what the 8 ranks run on        |     busbw | vs whole GPUs |
| :----------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)         | 450.2GBps |          100% |
| 4x `1g.35gb` on each of 2 GPUs |  26.8GBps |            6% |

8 ranks, 2GiB payload:

| what the 8 ranks run on        |     busbw | vs whole GPUs |
| :----------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)         | 465.9GBps |          100% |
| 4x `1g.35gb` on each of 2 GPUs |  25.6GBps |            5% |

That is a collapse, and it isn't a small-message effect: the 2GiB payload is no better than the 512MiB one. NCCL's own log says why: of the ring's hops, 24 went `via P2P/CUMEM` and 6 went `via SHM/direct`. The hops that stay inside a GPU keep the fast path, but the ones that cross to the other GPU fall back to host shared memory, because the CUDA driver doesn't support NVLink P2P for MIG instances (the same limitation that rules out NVLS here). A ring runs at the speed of its slowest link ([a slow link dragging a ring](../network/collectives-internals.md#a-slow-link-dragging-a-ring)), so those few host-memory hops set the pace for all 8 ranks.

#### Mixing MIG instances and full GPUs

Not every GPU has to be partitioned. Since 7 instances is the most one GPU can give, the cheapest route to an 8-rank world is to carve one H200 (or another high-end GPU) into 7 instances and leave a second GPU whole. The `MIG_UUIDS` command already produces the right list for that mix: the partitioned GPU contributes its 7 instances, the whole one contributes itself:

```bash
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -i 0 -cgi 1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb -C

export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')

torchrun --nproc_per_node 8 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python torch-dist-bench.py --payload_size_in_gib 2'
```

8 ranks, 512MiB payload:

| what the 8 ranks run on                |     busbw | vs whole GPUs |
| :------------------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)                 | 450.2GBps |          100% |
| 7x `1g.18gb` of one GPU + 1 whole H200 |  28.5GBps |            6% |

8 ranks, 2GiB payload:

| what the 8 ranks run on                |     busbw | vs whole GPUs |
| :------------------------------------- | --------: | ------------: |
| 8x whole H200 (NVLink)                 | 465.9GBps |          100% |
| 7x `1g.18gb` of one GPU + 1 whole H200 |  27.5GBps |            6% |

The whole GPU rescues nothing: 24 hops ran `via P2P/CUMEM` inside the partitioned GPU and 6 `via SHM/direct` to the whole one, and the ring again runs at the speed of the host-memory hops. The useful corollary is that the second device's own bandwidth doesn't matter: a whole H200 in the eighth slot gave 28.5GBps, no better than the 26.8GBps of the run where the eighth rank was a MIG instance on a second partitioned H200. So if the goal is only to get past the 7-instances-per-GPU ceiling, the second device can be any GPU CUDA will talk to, and it needs no MIG support of its own since it joins whole. One H200 plus one modest GPU is an 8-rank world.

The practical rule: **keep an emulated multi-GPU job inside a single physical GPU.** Crossing to a second GPU costs a factor of 10 compared with 4 instances on one GPU (26.8 against 273.8GBps at 512MiB), and leaves you at 5% of what 8 real GPUs do. With a tiny model and small payloads, though, it probably doesn't matter much for development.

Keep in mind that this emulates the **topology** of a multi-GPU job, not its performance: each rank also gets only a fraction of the SMs and of the memory. It is a development and testing tool, not a way to measure throughput.

#### Caveats

##### `LOCAL_RANK` is `0` in every rank

Once `CUDA_VISIBLE_DEVICES` names a single instance, the process sees exactly one device, and CUDA numbers it `0`. Code doing `torch.cuda.set_device(local_rank)` with the value torchrun handed out would fail with `invalid device ordinal`, which is why the launcher resets `LOCAL_RANK` to 0. There is no way around this: CUDA enumerates only a single compute instance per process, even when several MIG devices are visible. It stays true when the ranks span 2 partitioned GPUs: which physical GPU a rank landed on is carried by the UUID, not by the ordinal, so every rank says `cuda:0` and every rank is on different hardware.

So every rank reports `LOCAL_RANK=0`, and it is worth being precise about what that breaks.

Nothing in `torch.distributed` is affected. `dist.get_rank()`, `dist.get_world_size()` and the process group are built from `RANK` and `WORLD_SIZE`, which the launch wrapper doesn't touch. The usual device-selection idiom also keeps working unchanged:

```python
local_rank = int(os.environ["LOCAL_RANK"])   # 0 in every process
torch.cuda.set_device(local_rank)            # binds to this rank's own instance
model = DDP(model.to(local_rank), device_ids=[local_rank])
```

Each process asks for device 0 and each one gets a different slice, because `CUDA_VISIBLE_DEVICES` differs per rank.

What breaks is code that treats `LOCAL_RANK` as an **identity** rather than a device ordinal. The node-leader guard is the common case:

```python
if local_rank == 0:
    download_dataset()   # meant to run once per node
```

Under MIG that condition is true in every process, so all ranks download the dataset, write the same log file, or race to fill the same cache directory: exactly the problem [main-process-first](#preprocessing-datasets-on-the-main-process) exists to prevent, and `local_main_process_first()` would be fooled the same way.

In a single-node emulation, the fix is to use `RANK`, which is still unique across the job:

```python
if dist.get_rank() == 0:
    download_dataset()
```

If the code must keep working unmodified on real multi-node runs, keep the original slot number under a different name. The launch wrapper stashes it before overwriting:

```bash
torchrun --nproc_per_node 8 --no-python bash -c '
export MIG_SLOT=$LOCAL_RANK
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((MIG_SLOT+1)))
export LOCAL_RANK=0
exec python train.py'
```

and the guard reads whichever variable is present:

```python
node_slot = int(os.environ.get("MIG_SLOT", os.environ["LOCAL_RANK"]))
if node_slot == 0:
    download_dataset()
```

##### Only 7 instances per GPU

MIG cuts along boundaries the silicon already has. A compute slice comes from a **GPC** (Graphics Processing Cluster), the top-level block that bundles a set of SMs with its own work distribution, and a memory slice comes from the memory system attached to the HBM stacks. Ampere, Hopper, Blackwell and, according to NVIDIA sources, Rubin each contain exactly 8 GPCs in the full GPU (a shipping SKU may have fewer enabled).

MIG allows at most 7 instances, not 8, because the eighth compute unit is reserved for managing the instances. It is held back as soon as MIG mode is enabled, whether you then create 7 instances or one. The memory system has no such tenant, so all 8 memory slices stay available, and the MIG documentation states the resulting ratio outright: each memory slice is 1/8 of the memory, each SM slice is 1/7 of the SMs.

That asymmetry is visible in the profile table above. Seven `1g` instances use all 7 compute slices but only 7 of the 8 memory slices, so one slice's worth of HBM is stranded on a fully partitioned GPU: 18GB of the H200's 141GB. It is also why a `4g` profile allows only one instance: a second one would need 8 compute slices, and there are only 7.

##### `torch.cuda.device_count()` disagrees with `LOCAL_WORLD_SIZE`

Each process can see only its own instance, so `torch.cuda.device_count()` returns 1 in every rank. On whole GPUs, that call happens to equal the number of ranks on the node, and plenty of code relies on it: to size a shard count, to build a list of devices to iterate over, or to check that the requested world fits on the node. Take the number from the environment instead:

```python
ranks_on_this_node = torch.cuda.device_count()               # 1 in every rank under MIG
ranks_on_this_node = int(os.environ["LOCAL_WORLD_SIZE"])     # what torchrun was given as --nproc_per_node
```

`LOCAL_WORLD_SIZE` is set by torchrun in every worker, alongside `RANK`, `LOCAL_RANK` and `WORLD_SIZE`, and it keeps reporting the full local rank count under MIG.

## Re-training Hub models from scratch

Hugging Face Transformers has [finetuning example scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch) that cover pretty much every modality and work out of the box. But what if you want to train a model **from scratch** rather than finetune it, for example to reproduce a published model, or to test your training setup on a known architecture?

A simple trick does it. `AutoConfig.from_pretrained` downloads only the model's architecture config (layer count, hidden size, vocab size and so on), and `AutoModelForCausalLM.from_config` builds that architecture with **freshly initialized random weights**, unlike `AutoModelForCausalLM.from_pretrained`, which would also load the trained weights. The tokenizer is reused as is. Save the result, and you have a directory that looks exactly like a pretrained model, except that it has only been initialized and has seen no training. Any finetuning script will then happily "finetune" it, which is training from scratch.

The original recipe uses `facebook/opt-1.3b` in bf16. (The `cat << EOT > prep-bf16.py ... EOT` wrapper is a shell heredoc that writes everything between the two `EOT` markers into `prep-bf16.py`.)

```bash
cat << EOT > prep-bf16.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

mname = "facebook/opt-1.3b"

config = AutoConfig.from_pretrained(mname)
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(mname)

path = "opt-1.3b-bf16"

model.save_pretrained(path)
tokenizer.save_pretrained(path)
EOT
```

then run:

```bash
python prep-bf16.py
```

This creates a folder `opt-1.3b-bf16` with everything you need to train the model from scratch. Change `torch.bfloat16` to `torch.float16` or `torch.float32` if that is the regime you plan to train in.

I ran the same script with the smaller `facebook/opt-125m` and checked that the weights really are untrained **[ran]**. Two things to know: recent `transformers` versions renamed the argument, printing `` `torch_dtype` is deprecated! Use `dtype` instead! `` (it still works), and the check is that the loss of an untrained model is close to $\ln(\text{vocab size})$, the cross-entropy of a uniform guess over the vocabulary:

```python
import math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("opt-125m-bf16"); t = AutoTokenizer.from_pretrained("opt-125m-bf16")
x = t("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
with torch.no_grad(): loss = m(**x, labels=x.input_ids).loss
print("dtype", m.dtype, "| params", sum(p.numel() for p in m.parameters())/1e6, "M")
print(f"loss {loss.item():.2f}  vs ln(vocab) = {math.log(m.config.vocab_size):.2f}")
```

```
dtype torch.bfloat16 | params 125.239296 M
loss 11.13  vs ln(vocab) = 10.83
```

How to read it: 125M parameters in bf16, and the loss is about $\ln(50272) = 10.83$, i.e. the model is guessing uniformly: it has no trained knowledge. The saved `model.safetensors` is 250MB, which is 125M parameters × 2 bytes.

Now train the saved model with the normal finetuning script. `run_clm.py` (causal language modeling) is one of the example scripts in the Transformers repo linked above, so run this from a clone of that repo, and point `--model_name_or_path` at the directory you created. Its flags are the HF `Trainer`'s `TrainingArguments`: `--bf16` enables bf16 mixed precision, `--dataset_name`/`--dataset_config_name` pick a dataset from the HF Hub, `--report_to tensorboard` logs to TensorBoard, and the rest are the usual batch size, optimizer and LR-schedule hyperparameters **[untested]**:

```bash
torchrun \
--nproc_per_node=1 --nnode=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=9901 \
examples/pytorch/language-modeling/run_clm.py --bf16 \
--seed 42 --model_name_or_path opt-1.3b-bf16 \
--dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--gradient_accumulation_steps 1 --do_train --do_eval --logging_steps 10 \
--save_steps 1000 --eval_steps 100 --weight_decay 0.1 --num_train_epochs 1 \
--adam_beta1 0.9 --adam_beta2 0.95 --learning_rate 0.0002 --lr_scheduler_type \
linear --warmup_steps 500 --report_to tensorboard --output_dir save_dir
```

The key entry is:

```
--model_name_or_path opt-1.3b-bf16
```

where `opt-1.3b-bf16` is the local directory you just generated.

Sometimes you can find the dataset the original model was trained on; sometimes you have to use an alternative. The rest of the hyperparameters can often be found in the paper or the documentation that came with the model (see [Training stability](stability.md) for choosing them yourself).

To summarize: this recipe lets you use the finetuning examples to re-train any model you can find on [the HF Hub](https://huggingface.co/models).

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).

## You can now read

- [Training across datacenters](decentralized-training.md), where failures are the normal case: [elastic membership](decentralized-training.md#elastic-membership-and-fault-tolerance), heartbeats, and a runnable torchft example in which a killed replica heals from a peer.
- [Debugging distributed training](../development/debugging-distributed.md) for what to do once a watchdog tells you a job is hung, and [debugging PyTorch](../development/debugging-pytorch.md) for memory and numerics problems behind OOM exits and irreproducible results.
- [Testing](../development/testing.md), where fixed seeds and tiny models make tests repeatable.

## Further reading

- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) and [NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility).
- [torchft](https://github.com/meta-pytorch/torchft) and its [torchtitan integration](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/torchft); [Fault-tolerant Llama training with 2000 synthetic failures every 15 seconds and no checkpoints](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/).
- [Oobleck](https://github.com/SymbioticLab/Oobleck), [Bamboo](https://www.usenix.org/system/files/nsdi23-thorpe.pdf) and [Varuna](https://github.com/microsoft/varuna).
- The [Llama 3 paper](https://arxiv.org/abs/2407.21783), section 3.3.4, for interruption statistics of a 16k-GPU run.
- [NCCL: using MIG instances](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-mig-instances) and [NVIDIA MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/).
- The BigScience [BLOOM-176B training scripts](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml) and [Megatron-DeepSpeed fork](https://github.com/bigscience-workshop/Megatron-DeepSpeed), where the kill switch, exit-duration and watchdog scripts above were used in production.
- [A Deep Investigation into MMAP Not Leaking Memory](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/).
- [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md) and [io-watchdog](https://github.com/grondo/io-watchdog).
