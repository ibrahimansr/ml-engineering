# Foundations: Storage, Checkpoints and Data Pipelines

A training job touches storage in three ways: it reads the training data (a steady stream for days), it writes checkpoints (a big burst every few minutes to hours, and the training stalls if the burst is slow), and it loads code and environments at start-up (lots of small files). Inference adds a fourth: loading the weights quickly when a server starts. Each of these has a different access pattern, and storage systems are good at some patterns and terrible at others. Most "the cluster is slow" complaints that aren't network or GPU problems come from a mismatch between the pattern and the storage.

This chapter starts from "what is a disk" and works up to the arithmetic a senior engineer does before choosing where a 5.7 TB checkpoint goes, how a 15-trillion-token dataset is laid out, and how a data loader resumes bit-exactly after a crash. The book's [Storage chapter](../storage/README.md) is the hands-on companion: it has the `fio-scan` benchmark, mmap-vs-sequential measurements on Lustre, filesystem recommendations and disk-cleanup scripts. This chapter doesn't repeat that material. It explains the mechanisms underneath it.

Conventions:

- Examples marked **[torch]** need `pip install torch numpy safetensors` (tested with torch 2.15.0, safetensors 0.8.0, numpy 2.5.4, Python 3.14). Pure-Python examples need only `python3`. **[boto3, moto]** needs `pip install boto3 "moto[s3]"` (boto3 is AWS's Python client library; moto fakes AWS services in-process for testing). **[tiktoken]** needs `pip install tiktoken`.
- Unless marked otherwise, outputs come from one run on an Apple M4 laptop (macOS, 16 GB RAM, internal SSD). The `fio` and `dd` outputs come from a Linux 6.10 VM (Docker Desktop's linuxkit kernel) on the same laptop. Its virtual disk is a file on the macOS SSD, so its absolute numbers mean nothing for real servers. They show what the tools print and how the numbers relate to each other.
- Commands that need a real cluster (Lustre `lfs`, `mdadm`, `nvme`, real S3) are marked **untested**. Their flags are taken from the tools' documentation.

## Prerequisites

- [Computer Architecture](computer-architecture.md): especially [Storage devices](computer-architecture.md#storage-devices) and [Bandwidth vs latency](computer-architecture.md#bandwidth-vs-latency).
- [Linux and Operating Systems](linux-and-os.md): especially section 3, [Files and filesystems](linux-and-os.md#3-files-and-filesystems), covering inodes, mount points, the page cache, and `fsync` with atomic rename. This chapter assumes you know why `write()` returning doesn't mean the data is on disk.
- [Concurrency and Parallelism](concurrency.md): [DataLoader internals](concurrency.md#part-4-how-the-pytorch-dataloader-works-inside) and [Async I/O for checkpoints and data](concurrency.md#part-8-async-io-for-checkpoints-and-data).
- [Computer Networking](networking.md): enough to know what bandwidth, latency, RTT and TCP are, plus HTTP (object stores are HTTP APIs) and the [bandwidth-delay product](networking.md#bandwidth-delay-product).
- [The toolkit](toolkit.md): `pip` and virtual environments for the **[torch]**-style examples, JSON (the safetensors header and checkpoint manifests are JSON), and git (a run's git SHA belongs in its checkpoint).
- [Distributed Systems](distributed-systems.md): ranks, barriers and process groups (the gloo backend), [data vs model parallelism](distributed-systems.md#model-parallelism) (FSDP, tensor and pipeline parallelism, which decide how a checkpoint is sharded), and [MTBF and how often to checkpoint](distributed-systems.md#how-often-to-checkpoint).
- [Containers and Virtualization](containers-and-virtualization.md): volumes and bind mounts, and why environments belong in images rather than on a shared filesystem.
- Forward links: [Training arithmetic](../training/arithmetic.md) (bytes per parameter, FLOPs per token) and [Model Parallelism](../training/model-parallelism/README.md).


## 1. Three ways to store bytes: block, file, object

Every storage product you'll meet sells one of three interfaces. Most confusion about storage comes from not knowing which one you're looking at.

```
 application:   open("/data/x.bin")  read(fd, 1 MiB)          GET s3://bucket/x.bin  Range: bytes=0-1048575
                        |                                                |
                 +-------------+   POSIX file API                        |  HTTP API
                 | filesystem  |   (names, dirs, permissions,            |
                 | (ext4, xfs, |    byte offsets, locks)          +-------------+
                 |  Lustre...) |                                  | object store|
                 +-------------+                                  | (S3, GCS,   |
                        |  "read blocks 81920..82175"             |  MinIO...)  |
                 +-------------+                                  +-------------+
                 | block device|   numbered fixed-size blocks            |
                 | (NVMe, EBS, |   nothing else                   internally: disks, block
                 |  SAN LUN)   |                                  devices, its own metadata DB
                 +-------------+
```

### Block storage

A **block device** is a numbered array of fixed-size blocks (usually 512 bytes or 4 KiB), and you read or write whole blocks by number. It has no names, directories or permissions. A local NVMe drive is a block device. So is a cloud "volume" such as AWS EBS or a GCP Persistent Disk: it is attached over the network, but the VM sees it as a disk. A SAN LUN (in the diagram) is the on-premises version: a virtual disk carved out of a dedicated storage array and served to one server over a storage network. On Linux they appear under `/dev`:

```bash
$ lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL        # untested; sketched output from a GPU node
NAME        SIZE TYPE MOUNTPOINT MODEL
nvme0n1     1.7T disk            SAMSUNG MZ1L21T9HCLS
└─nvme0n1p1 1.7T part /
nvme1n1     3.5T disk            SAMSUNG MZQL23T8HCLS
nvme2n1     3.5T disk            SAMSUNG MZQL23T8HCLS
md0          14T raid0 /local     # 4 data drives striped together
```

Only one machine at a time can safely mount a block device read-write. Two kernels each running their own ext4 on the same blocks would corrupt each other's metadata within seconds. That is why block storage is fast (there is nothing in the way) and also why it is not shared.

### File storage

A **filesystem** turns blocks into named files in a tree of directories. It keeps **metadata** (data about the data: names, sizes, owners, timestamps, and which blocks hold which file) in structures called inodes, and it gives you the **POSIX** API (the Unix standard for file operations): `open`, `read`, `write` at any byte offset, `rename`, `fsync`, locks. A local filesystem (ext4, XFS) lives on one block device. A **network filesystem** (NFS, Lustre, GPFS, Weka) serves the same API over the network to many clients at once, so every node in the cluster sees the same `/data`.

The POSIX API is very convenient, and it is expensive to provide across many machines. `rename` must be atomic. A write must be visible to a reader on another node at the right time. A `stat()` must return the current size. Each of these needs coordination between the client and the servers. That is why shared filesystems are much slower per operation than local disks, and why their metadata servers fall over when a job runs `os.listdir` on a million files from a thousand ranks at once.

```python
# posix_ops.py - the operations a filesystem must make atomic and coherent
import os, tempfile
d = tempfile.mkdtemp()
p = os.path.join(d, "ckpt.bin")
with open(p + ".tmp", "wb") as f:
    f.write(b"x" * 4096); f.flush(); os.fsync(f.fileno())   # durable data
os.replace(p + ".tmp", p)                                    # atomic rename: the POSIX feature checkpoints rely on
with open(p, "r+b") as f:
    f.seek(1000); f.write(b"Y")                              # overwrite 1 byte in the middle: trivial here, impossible in S3
st = os.stat(p)
print(st.st_size, st.st_ino, oct(st.st_mode))
```

```text
4096 109344057 0o100644
```

### Object storage

An **object store** (Amazon S3, Google Cloud Storage, Azure Blob, Cloudflare R2, MinIO, Ceph RGW) keeps **objects**: an immutable blob of bytes, a **key** (a string such as `runs/a/step_100/shard_0.bin`) and a little metadata. Objects live in a flat namespace called a **bucket**, and you use them through an HTTP API: `PUT` a whole object, `GET` it (or a byte range of it), `DELETE` it, `LIST` keys by prefix. That's nearly the whole interface.

What's missing is deliberate. There is no appending, no writing at an offset, no rename (a "rename" is copy plus delete), no locks, and no real directories (the `/` in a key is just a character). Giving these up lets the provider spread objects over thousands of servers with no cross-server coordination. That is how S3 offers effectively unlimited capacity and very high aggregate throughput at a few cents per GB-month. The cost is latency: every request is an HTTP round trip, typically tens of milliseconds to the first byte.

Below, `aws` is Amazon's command-line client for its services, and `model.safetensors` is a weights file in the safetensors format, a JSON header followed by raw tensor bytes (taught in [section 6](#safetensors-a-header-plus-raw-bytes)).

```bash
# untested here (needs AWS credentials); standard aws-cli commands
$ aws s3 cp model.safetensors s3://my-bucket/models/llama-8b/model.safetensors
$ aws s3 ls s3://my-bucket/models/llama-8b/
2026-11-05 10:12:01 16060522624 model.safetensors
$ aws s3api get-object --bucket my-bucket --key models/llama-8b/model.safetensors \
      --range bytes=0-7 header.bin        # range read: first 8 bytes only
```

### Which one for what

| | Block (local NVMe) | Shared filesystem (NFS / parallel FS) | Object store |
|---|---|---|---|
| Interface | blocks; you put a local FS on it | POSIX files | HTTP: PUT / GET / LIST / DELETE |
| Shared between nodes | no | yes | yes, also across regions and clouds |
| Latency per op | ~10-100 µs | ~0.1-few ms (metadata can be far worse) | ~10-100 ms first byte |
| Throughput | a few GB/s per drive, ~10-25 GB/s per node with several drives | 10s-1000s of GB/s aggregate (parallel FS) | very high aggregate, if you issue many requests in parallel |
| Small files | excellent | the classic weak spot | per-request overhead and per-request price |
| Survives node loss | no | yes | yes, and designed for very high durability |
| Typical ML use | staging data, local checkpoint tier, `/tmp` | code, envs, datasets, checkpoints | the dataset and checkpoint archive, cross-site transfer, model distribution |

The usual cluster layout described in [Linux: Local vs network filesystems](linux-and-os.md#local-vs-network-filesystems) follows from this table: code on NFS `/home`, hot data and checkpoints on a parallel FS, staging on node-local NVMe, and everything durable in an object store.


## 2. Local NVMe

### What an NVMe drive does

An SSD stores bits in NAND flash: many dies (individual flash chips) that can be read in parallel. **NVMe** (Non-Volatile Memory Express) is the protocol for talking to an SSD directly over PCIe. Its key design choice is many deep queues. The spec allows up to 65,535 I/O queues of up to 65,536 commands each, so every CPU core can have its own queue pair and submit requests without taking a lock. The older SATA/AHCI interface had one queue of 32 commands. The NVMe design exists because flash is internally parallel: a single request at a time uses one die and leaves the rest idle. The [Queue Depth](../storage/README.md#queue-depth) section of the Storage chapter shows the effect.

Two flash facts that matter for ML:

- **Writes are slower and less steady than reads.** Flash is erased in big blocks and written in pages. Sustained writes eventually trigger internal garbage collection, and throughput drops once the drive's fast cache region fills. A short benchmark can overstate sustained write speed.
- **Endurance is finite.** Drives are rated in DWPD (drive writes per day over the warranty). Writing a multi-TB checkpoint to the same local drive every ten minutes for months uses up real endurance.

```bash
# untested: needs nvme-cli on a Linux node with NVMe drives
$ sudo nvme list
Node          Model                  Namespace  Usage                    Format
/dev/nvme1n1  SAMSUNG MZQL23T8HCLS   1          3.84 TB / 3.84 TB        512 B + 0 B
$ sudo nvme smart-log /dev/nvme1n1 | grep -E 'percentage_used|data_units_written|temperature'
temperature          : 41 C
percentage_used      : 3%
data_units_written   : 812,345,678     # units of 512,000 bytes -> ~416 TB written
$ cat /sys/block/nvme1n1/queue/nr_requests       # the kernel's per-queue request limit
1023
```

### Striping local drives together

GPU nodes usually ship with 4-8 data NVMe drives. One drive gives a few GB/s. **RAID 0** (striping) spreads consecutive chunks of every file across all drives, so a big sequential read uses all of them at once. RAID 0 has no redundancy: one dead drive loses everything. That is acceptable for scratch space, which is always reproducible from the shared tier.

```bash
# untested: creates a 4-drive RAID0 and an XFS filesystem (destroys data on those drives)
$ sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 --chunk=512K \
      /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
$ sudo mkfs.xfs /dev/md0 && sudo mkdir -p /local && sudo mount -o noatime /dev/md0 /local
$ cat /proc/mdstat
md0 : active raid0 nvme4n1[3] nvme3n1[2] nvme2n1[1] nvme1n1[0]
      15002963968 blocks super 1.2 512k chunks
```

`noatime` stops the filesystem from writing an access timestamp on every read, which is pure overhead for training data. Cloud providers often set this up for you. Look for something like `/opt/dlami/nvme`, `/mnt/local_disk` or `/scratch` before building your own.

### GPUDirect Storage, briefly

Normally data travels disk → kernel page cache → user buffer → pinned buffer → GPU. **GPUDirect Storage (GDS)** lets the NVMe drive or the network storage client DMA straight into GPU memory and skip host RAM (DMA, direct memory access, means the device copies into memory without the CPU doing the copying). The API is NVIDIA's `cuFile`, and some parallel filesystems support it. GDS removes CPU and bounce-buffer copies, but it doesn't make a disk faster than its own bandwidth. It matters most when host-memory bandwidth or CPU is the bottleneck, for example many GPUs loading weights at once. See [NVIDIA's GDS docs](https://docs.nvidia.com/gpudirect-storage/). This is untested here.


## 3. Network and parallel filesystems

### NFS

**NFS** (Network File System) is the oldest and simplest shared filesystem: one server (or an HA pair: two servers where one takes over if the other dies) exports a directory, and clients mount it. Every operation becomes an RPC to that server, so it has one server's bandwidth and one server's metadata throughput. NFS is fine for home directories and code, and poor for datasets and checkpoints at scale. Its consistency model is **close-to-open**: another client is only guaranteed to see your writes after you `close()` the file and it `open()`s it. It is not guaranteed to see them while you're still writing.

```bash
# untested: typical NFS mount and a look at its options
$ sudo mount -t nfs -o vers=4.1,nconnect=8,rsize=1048576,wsize=1048576 nfs-server:/export/home /home
$ nfsstat -m
/home from nfs-server:/export/home
 Flags: rw,vers=4.1,rsize=1048576,wsize=1048576,nconnect=8,hard,proto=tcp,...
```

`vers=4.1` picks the NFS protocol version, and `rsize`/`wsize` are the largest single read and write request the client sends (1 MiB here). `nconnect=8` opens 8 TCP connections to the server instead of 1 (Linux 5.3+). A single TCP stream is often the limit for one NFS client, for the reasons in [section 9](#transfer-time-and-the-bandwidth-delay-product).

### How a parallel filesystem works

A **parallel filesystem** splits the two jobs a filesystem does across many servers:

```
                     compute nodes (clients, each runs the FS client in its kernel)
           +--------+ +--------+ +--------+         +--------+
           | node 1 | | node 2 | | node 3 |  ...    | node N |
           +--------+ +--------+ +--------+         +--------+
              |  1. open("/lustre/ckpt/shard_7.bin")  -> "which servers hold it?"
              v
    +--------------------+        metadata: names, dirs, permissions, sizes,
    | metadata server(s) |        and the file's LAYOUT: "stripe 4 MiB chunks
    |  (MDS / MDT)       |        round-robin over OST 3, 9, 14, 22"
    +--------------------+
              |  2. read/write the bytes directly, in parallel, from many data servers
              v
    +-------+ +-------+ +-------+ +-------+ +-------+ +-------+      data servers (OSS),
    | OST 0 | | OST 1 | | ... 3 | | ... 9 | | ..14  | | ..22  | ...  each with several
    +-------+ +-------+ +-------+ +-------+ +-------+ +-------+      targets (OSTs) = disk arrays
```

- A client asks the **metadata server** once per `open`. The answer includes the file's **layout**, the map of which data servers hold which byte ranges.
- After that, the client reads and writes the data servers directly and in parallel. The metadata server is not on the data path.
- **Striping** means a file is cut into fixed-size chunks (the *stripe size*, e.g. 1-4 MiB) placed round-robin across several data targets (the *stripe count*). A large file striped over 16 targets can be read at roughly 16 targets' combined bandwidth. Bandwidth grows with the number of servers, which is the whole point.

The design has a consequence you will hit: **data scales, metadata scales much less.** `ls -l` on a directory of a million files, `import torch` (thousands of `stat` and `open` calls), or a DataLoader that opens one small JPEG per sample all go to the metadata servers. Large-file bandwidth can be hundreds of GB/s while small-file workloads crawl. That is why ML data is packed into big shard files (see [section 8](#8-datasets-at-scale)).

### Lustre: MDS, OSS and `lfs setstripe`

**Lustre** is the open-source parallel FS behind many supercomputers. In Lustre terms, the MDS (metadata server) stores metadata on MDTs (metadata targets), and the OSS (object storage server) stores file data on OSTs (object storage targets). Recent versions can spread metadata over several MDTs (the feature is called DNE, Distributed Namespace). Striping is set per file or per directory, and new files inherit their directory's setting:

```bash
# untested: Lustre client commands (lfs ships with the client)
$ lfs df -h /lustre                               # capacity per MDT/OST
UUID                 bytes   Used  Available Use% Mounted on
lustre-MDT0000_UUID   2.8T  91.2G      2.7T   4% /lustre[MDT:0]
lustre-OST0000_UUID  58.2T  41.0T     17.2T  71% /lustre[OST:0]
...
$ lfs getstripe /lustre/datasets/shard_000.bin    # how is this file laid out?
lmm_stripe_count:  1
lmm_stripe_size:   1048576
        obdidx   objid   objid   group
             7  123456  0x1e240   0        # the whole file lives on OST 7

$ mkdir /lustre/ckpts/run1
$ lfs setstripe -c 8 -S 4M /lustre/ckpts/run1     # new files here: 4 MiB stripes over 8 OSTs
$ lfs setstripe -c -1 /lustre/models/big          # -1 = stripe over all OSTs
```

How to choose:

- Many small and medium files (a dataset of 100k 256 MB shards, read by many ranks): stripe count 1. Each file sits on one OST and parallelism comes from different ranks reading different files. Striping small files only adds round trips.
- A few huge files read or written by many clients at once (a consolidated checkpoint, a model every node loads): stripe wide, so the load spreads over many OSTs instead of hammering one.
- An OST that fills up makes writes fail even if the filesystem as a whole has space. The Storage chapter covers this ["80%" trap](../storage/README.md#beware-that-youre-often-being-sold-only-80-of-the-storage-you-pay-for).

Many sites set a default layout called PFL (Progressive File Layout): the first MiBs on one OST and wider striping as the file grows. Check `lfs getstripe -d` on your directories before changing anything.

### GPFS, Weka, VAST: other answers to the metadata problem

| | Lustre | IBM Storage Scale (GPFS) | WekaFS | VAST |
|---|---|---|---|---|
| Metadata | dedicated MDS/MDT servers (can be several) | distributed across the cluster's nodes | distributed across all backend nodes | in the shared storage itself, accessed by stateless servers |
| Client | kernel module | native NSD client (GPFS's own protocol), or NFS/SMB (SMB is the Windows file-sharing protocol) | native client (POSIX), or NFS/SMB/S3 | mostly NFS (with multipath / RDMA extensions) |
| Notable | open source, needs careful tuning (see the [NASA best practices](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html)) | very mature; tiering to object storage; sub-blocks for small files | flash-native; tiers to an object store | flash-native; global data reduction |

These descriptions are from vendor documentation and the Storage chapter's [Which file system to choose](../storage/README.md#which-file-system-to-choose). They are not from benchmarks run for this chapter. For ML, what to evaluate is: aggregate large-file read and write bandwidth at your node count, small-file metadata ops/s, and behaviour when hundreds of clients write checkpoints at once. Use the tools in [section 5](#5-measuring-storage) and the Storage chapter's [`fio-scan`](../storage/README.md#fio).

### Metadata storms, and how ML jobs cause them

A **metadata storm** is many clients issuing metadata operations (`stat`, `open`, `listdir`, `create`) at once and saturating the metadata servers. Every user of the filesystem then sees `ls` hang. ML jobs cause them in a few standard ways:

```python
# DON'T: every one of 1024 ranks lists the same directory of 2M files at start-up
files = sorted(os.listdir("/lustre/data/images"))           # 1024 x 2M dirent reads on the MDS

# DO: list once offline, ship the list as one file, every rank reads one file
#   python -c "import os,json; json.dump(sorted(os.listdir(D)), open('index.json','w'))"
files = json.load(open("/lustre/data/index.json"))           # 1 open per rank, sequential read
```

Other common causes: Python environments with 100k files on a shared FS (each `import` is dozens of `stat` calls, multiplied by every rank), one-file-per-sample datasets, and checkpoints that write one file per tensor per rank (a 405B model with about 1,000 tensors on 1,024 ranks makes up to a million `create`s every save). The fixes are always the same: pack small things into big files, list once, and put environments in a container image or on local disk (see [Containers and Virtualization](containers-and-virtualization.md)).


## 4. Object stores

### S3 semantics

S3's API has become the standard that GCS, R2, MinIO, Ceph and others imitate, so learn its rules once:

- **Whole-object writes.** `PUT` replaces the entire object. To change one byte you rewrite the object.
- **Consistency.** As of December 2020, AWS S3 gives strong read-after-write consistency: after a successful `PUT`, every `GET` and `LIST` sees the new data. Older S3 and some S3-compatible stores were only *eventually consistent*: a `LIST` right after a `PUT` might not show the new object. Check this for any non-AWS store you rely on.
- **No atomic multi-object operations.** A checkpoint made of 1,000 objects is never written atomically. A reader can see 600 new shards and 400 old ones. You build atomicity yourself by writing a marker object last (see [section 7](#making-checkpoints-trustworthy)).
- **Conditional writes.** As of late 2024, S3 supports `If-None-Match: *` on `PUT` ("only create if absent") and `If-Match` ("only overwrite if the ETag is still X"; an **ETag** is a fingerprint string S3 returns for each version of an object). These are building blocks for locks and leader election on top of S3. They are version-sensitive, so check your provider.
- **Rate limits per prefix.** AWS documents at least 3,500 `PUT`/`COPY`/`POST`/`DELETE` and 5,500 `GET`/`HEAD` requests per second per partitioned prefix. Throughput comes from many parallel requests, often spread across prefixes. One request at a time gets you one connection's worth of bandwidth.

```python
# s3_demo.py  [boto3, moto]  - S3 semantics against moto's in-process S3 emulator
import hashlib, os, boto3
from moto import mock_aws

@mock_aws
def main():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="ckpts")

    # 1) PUT is whole-object: there is no "append" and no "write at offset"
    s3.put_object(Bucket="ckpts", Key="runs/a/step_100/meta.json", Body=b'{"step": 100}')

    # 2) "directories" are just key prefixes; LIST by prefix + delimiter fakes them
    r = s3.list_objects_v2(Bucket="ckpts", Prefix="runs/a/", Delimiter="/")
    print("common prefixes:", [p["Prefix"] for p in r.get("CommonPrefixes", [])])

    # 3) multipart upload: parts are uploaded independently (in parallel in real life), then committed
    part = 5 * 2**20                                   # S3 minimum part size is 5 MiB (except the last)
    data = os.urandom(3 * part + 123)
    mpu = s3.create_multipart_upload(Bucket="ckpts", Key="runs/a/step_100/shard_0.bin")
    parts = []
    for n, off in enumerate(range(0, len(data), part), start=1):
        r = s3.upload_part(Bucket="ckpts", Key="runs/a/step_100/shard_0.bin", UploadId=mpu["UploadId"],
                           PartNumber=n, Body=data[off:off + part])
        parts.append({"PartNumber": n, "ETag": r["ETag"]})
    print("before complete, object visible?",
          "Contents" in s3.list_objects_v2(Bucket="ckpts", Prefix="runs/a/step_100/shard_0"))
    s3.complete_multipart_upload(Bucket="ckpts", Key="runs/a/step_100/shard_0.bin",
                                 UploadId=mpu["UploadId"], MultipartUpload={"Parts": parts})
    etag = s3.head_object(Bucket="ckpts", Key="runs/a/step_100/shard_0.bin")["ETag"].strip('"')

    # 4) a multipart ETag is NOT the md5 of the file: it is md5(concat(md5(part_i))) + "-<nparts>"
    md5s = b"".join(hashlib.md5(data[o:o + part]).digest() for o in range(0, len(data), part))
    print("ETag from S3      :", etag)
    print("md5 of whole file :", hashlib.md5(data).hexdigest())
    print("md5-of-part-md5s  :", hashlib.md5(md5s).hexdigest() + f"-{len(parts)}")

    # 5) range read: fetch bytes [10 MiB, 10 MiB + 16) without downloading 15 MiB
    r = s3.get_object(Bucket="ckpts", Key="runs/a/step_100/shard_0.bin", Range="bytes=10485760-10485775")
    print("range read:", r["ContentRange"], "ok:", r["Body"].read() == data[10485760:10485776])

main()
```

```text
common prefixes: ['runs/a/step_100/']
before complete, object visible? False
ETag from S3      : 4c012e54e12c8b83236d9849d4198210-4
md5 of whole file : bb574a7ec45d1d47f2a363106531e7d6
md5-of-part-md5s  : 4c012e54e12c8b83236d9849d4198210-4
range read: bytes 10485760-10485775/15728763 ok: True
```

`moto` is an emulator, not S3. It reproduces the API, not the performance or the rate limits.

### Multipart uploads and range reads

**Multipart upload** splits a large object into parts that are uploaded independently, retried independently and committed together. As of the S3 docs at the time of writing, parts are 5 MiB to 5 GiB, with at most 10,000 parts per object. Object size has long been capped at 5 TiB; check the current docs, because AWS has announced larger limits. The object doesn't exist until `CompleteMultipartUpload`, which is a single atomic commit for one object. Abandoned uploads keep their parts, and you keep paying for them, until you abort them or a lifecycle rule (a bucket setting that automatically deletes or archives objects or incomplete uploads by age) cleans them up. Set an `AbortIncompleteMultipartUpload` lifecycle rule on every bucket you checkpoint to.

The demo shows the trap in point 4: a multipart object's ETag is not the file's MD5, so `md5sum file == ETag` "verification" fails, or worse, gets skipped. To verify, either recompute the ETag the same way with the same part size, or store your own checksum as object metadata. S3 also supports additional checksums (CRC32C, SHA-256 and others) at upload time.

A **range read** (`Range: bytes=a-b`) fetches part of an object. Range reads are what make object stores usable for more than whole-file download:

- safetensors on S3: one range read for the 8-byte length and the header, then one range read per tensor you need.
- Parallel download of one big object: N concurrent range reads of 64 MiB each. Tools such as `s5cmd`, `rclone` and `aws s3 cp` do this internally.
- Streaming datasets read whole shards sequentially, but can start at an offset when resuming.

### Getting bandwidth out of an object store

A single HTTPS stream to S3 typically gets tens to low hundreds of MB/s. The service is not slow. One TCP connection, one request at a time, with tens of milliseconds of latency per request just can't fill a fat pipe (the same [Little's law](#bandwidth-iops-latency-and-littles-law) arithmetic as for disks). Aggregate bandwidth comes from concurrency:

```python
# s3_parallel_get.py  [boto3]  - untested against real S3; the pattern every fast S3 client uses
import boto3
from concurrent.futures import ThreadPoolExecutor

s3 = boto3.client("s3")
bucket, key, part = "my-bucket", "models/llama-70b/model-00001-of-00030.safetensors", 64 << 20
size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]

def get(off):
    end = min(off + part, size) - 1
    return off, s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={off}-{end}")["Body"].read()

buf = bytearray(size)
with ThreadPoolExecutor(32) as pool:                        # 32 requests in flight
    for off, chunk in pool.map(get, range(0, size, part)):
        buf[off:off + len(chunk)] = chunk
```

### FUSE mounts of object stores

A **FUSE** (Filesystem in Userspace) client such as `s3fs`, `gcsfuse`, `rclone mount` or AWS `mountpoint-s3` makes a bucket look like a directory. It is handy and a common source of surprises. A FUSE mount can't give you POSIX semantics the store doesn't have. Depending on the client, "rename" is a full copy or unsupported, random writes are unsupported (mountpoint-s3 supports only sequential writes to new files, per its docs), `fsync` may not mean "durable in the store", and `ls` on a big prefix is slow and paginated. `torch.save` to a FUSE mount followed by the atomic-rename pattern from the [Linux chapter](linux-and-os.md#fsync-and-durability-why-checkpoints-get-lost) may silently not be atomic. Prefer the object API (via DCP storage plugins, `fsspec`, or your own uploader) for checkpoints, and keep FUSE for read-mostly convenience. (DCP, `torch.distributed.checkpoint`, is PyTorch's sharded checkpoint library, taught in [section 6](#torchdistributedcheckpoint-every-rank-writes-its-own-pieces); `fsspec` is a Python library that gives local files, S3, GCS and others one file-like interface.)

```bash
# untested: AWS mountpoint-s3, read-only mount for loading a dataset or model
$ mount-s3 --read-only my-bucket /mnt/s3
$ python -c "from safetensors import safe_open; f = safe_open('/mnt/s3/models/x.safetensors', 'pt'); print(list(f.keys())[:3])"
```


## 5. Measuring storage

### Bandwidth, IOPS, latency and Little's law

The three numbers (defined in the Storage chapter's [Metrics](../storage/README.md#metrics) section) are bandwidth (bytes/s), IOPS (operations/s) and latency (time per operation). Two equations connect them:

$$
\text{bandwidth} = \text{IOPS} \times \text{block size}
$$

$$
\text{outstanding} = \text{IOPS} \times \text{latency}
$$

The second is Little's law: requests in flight = rate $\times$ time each spends inside.

The second equation explains almost every storage benchmark. If a device has 100 µs latency and you keep 1 request in flight, you get 10k IOPS. To get 1M IOPS from the same device you need 100 requests in flight. "The storage is slow" very often means "the application keeps too few requests in flight". The fix then is more DataLoader workers, a thread pool, `iodepth`, or multipart concurrency, not new hardware. The fio run below checks the law: $4\ \text{jobs} \times \text{iodepth } 32 = 128$ outstanding requests, $128 / 3.143\ \text{ms} = 40.7\text{k}$ IOPS, and fio measured 40.6k.

### fio, flag by flag

`fio` generates a synthetic I/O load that you describe precisely. This is the run quoted above, executed in the Linux VM (`apt install fio`, fio-3.36):

```bash
fio --name=randread-4k \          # job name (also names the data files)
    --directory=/root/fio \       # where to put data files: the filesystem under test
    --size=1G \                   # per-job file size; make total >> RAM, or use --direct=1
    --rw=randread \               # read | write | randread | randwrite | randrw
    --bs=4k \                     # block size of each I/O
    --iodepth=32 \                # requests kept in flight per job (only meaningful with an async engine)
    --numjobs=4 \                 # parallel jobs (processes); total in flight = numjobs x iodepth
    --ioengine=libaio \           # Linux native async I/O; io_uring (newer kernel async-I/O interface) is the modern alternative
    --direct=1 \                  # O_DIRECT: bypass the page cache, so we measure storage, not RAM
    --time_based --runtime=10s \  # run for a fixed time instead of until size is done
    --ramp_time=2s \              # ignore the first 2 s (warm-up)
    --group_reporting             # one summary for all jobs
```

(The comments are for reading only. Bash doesn't allow comments after a trailing `\`, so remove them before pasting.)

```text
randread-4k: (g=0): rw=randread, bs=(R) 4096B-4096B, (W) 4096B-4096B, (T) 4096B-4096B, ioengine=libaio, iodepth=32
...
fio-3.36
Starting 4 processes
randread-4k: Laying out IO file (1 file / 1024MiB)
...
randread-4k: (groupid=0, jobs=4): err= 0: pid=2285: Thu Nov  5 21:20:04 2026
  read: IOPS=40.6k, BW=159MiB/s (166MB/s)(1588MiB/10004msec)
    slat (nsec): min=541, max=6393.9k, avg=5906.56, stdev=22499.28
    clat (usec): min=247, max=22920, avg=3143.27, stdev=1047.28
     lat (usec): min=252, max=22928, avg=3149.17, stdev=1047.58
    clat percentiles (usec):
     |  1.00th=[ 2212],  5.00th=[ 2376], 10.00th=[ 2474], 20.00th=[ 2573],
     | 30.00th=[ 2671], 40.00th=[ 2802], 50.00th=[ 2900], 60.00th=[ 3032],
     | 70.00th=[ 3195], 80.00th=[ 3392], 90.00th=[ 3851], 95.00th=[ 4686],
     | 99.00th=[ 7504], 99.50th=[ 8979], 99.90th=[14484], 99.95th=[15795],
     | 99.99th=[20579]
  ...
  IO depths    : 1=0.0%, 2=0.0%, 4=0.0%, 8=0.0%, 16=0.0%, 32=100.0%, >=64=0.0%
Run status group 0 (all jobs):
   READ: bw=159MiB/s (166MB/s), 159MiB/s-159MiB/s (166MB/s-166MB/s), io=1588MiB (1665MB), run=10004-10004msec
```

How to read it:

- `slat` is submission latency (time to hand the request to the kernel). `clat` is completion latency (time from submission to completion, the storage's real latency). `lat` is the two added together.
- Look at the percentiles, not the average. A data loader or a checkpoint barrier waits for the *slowest* request. Here p50 is 2.9 ms and p99.9 is 14.5 ms, a 5x tail.
- `IO depths: 32=100%` confirms the requested depth was actually reached.

Two flag traps:

```bash
# Same job with --direct=0 (buffered). Measured in the same VM, one run each:
#   direct=1  randread 4k qd32 x4 jobs : 62,075 IOPS   mean clat  2.06 ms
#   direct=0  randread 4k qd32 x4 jobs :  5,442 IOPS   mean clat 22.8 ms
```

1. **libaio is only asynchronous with `O_DIRECT`.** The fio docs say so: with buffered I/O, libaio submissions block, so `iodepth=32` quietly becomes about 1. That is why the buffered run above is 11x slower, not faster. (These two rows come from a separate, JSON-output pass in the same VM. The 62k vs 40.6k difference between the two `direct=1` runs is run-to-run noise in a VM. Take the ratio as the finding, not the absolute numbers.)
2. **fio reuses its data files** if they exist with the same name, even when the parameters changed. Use `--unlink=1` or delete them, as the [Storage chapter warns](../storage/README.md#fio).

ML-shaped fio jobs:

```bash
# untested on a real cluster; patterns to adapt
# checkpoint-like: each of 8 ranks per node writes one big file with 16 MiB writes, fsync at end
fio --name=ckpt --directory=/lustre/bench --rw=write --bs=16M --size=8G --numjobs=8 \
    --ioengine=libaio --direct=1 --iodepth=4 --end_fsync=1 --group_reporting --unlink=1
# dataloader-like: many readers, 1 MiB sequential chunks out of large shards
fio --name=shards --directory=/lustre/bench --rw=read --bs=1M --size=4G --numjobs=32 \
    --ioengine=libaio --direct=1 --iodepth=8 --time_based --runtime=60s --group_reporting --unlink=1
```

A parallel filesystem has to be measured from many nodes at once. One client can't saturate it. Run the same fio job on N nodes (via Slurm's `srun`, or `pdsh`, which runs one command over ssh on many hosts at once) and add the results, or use fio's client/server mode (`fio --server` on each node, `fio --client=hosts.txt job.fio` from one). For metadata, IOR/mdtest (from the IO500 benchmark) is the standard tool.

### dd and its pitfalls

`dd` copies bytes. People use it as a benchmark because it's always installed, and it is easy to get wrong. This was run in the Linux VM, 1 GiB per command, twice:

```bash
dd if=/dev/zero of=dd.bin bs=1M count=1024                  # write, no sync
dd if=/dev/zero of=dd.bin bs=1M count=1024 conv=fdatasync   # write, flush to disk before reporting
dd if=dd.bin of=/dev/null bs=1M                             # read right after writing
dd if=dd.bin of=/dev/null bs=1M                             # read again
sync; echo 3 > /proc/sys/vm/drop_caches                     # root only: drop the page cache
dd if=dd.bin of=/dev/null bs=1M                             # cold read
dd if=dd.bin of=/dev/null bs=1M iflag=direct                # O_DIRECT read
```

```text
             no-sync write   fdatasync write   read #1    read #2    cold read   direct read
run 1        185 MB/s        140 MB/s          2.7 GB/s   6.9 GB/s   633 MB/s    1.4 GB/s
run 2        232 MB/s        336 MB/s          5.4 GB/s   11.1 GB/s  1.1 GB/s    2.4 GB/s
(an earlier session: no-sync write 2.3 GB/s, fdatasync write 772 MB/s)
```

What the numbers teach, given that this is a noisy VM:

- **Reads right after writing are page-cache reads.** 5-11 GB/s here is memory bandwidth, not the disk. The cold read, after `drop_caches`, is 5-10x slower. A benchmark that writes a file and reads it back measures RAM.
- **A write without `conv=fdatasync` measures how fast dirty pages pile up in RAM** when there is room, as in the 2.3 GB/s vs 772 MB/s session. When the kernel's dirty-page limits are already hit, as in runs 1 and 2, the no-sync write is throttled to disk speed anyway. Either way the number isn't "disk write speed" unless the data is forced to disk.
- **The same command varied 2x between runs.** One run is an anecdote. Repeat it, and report the spread.
- **`/dev/zero` is a stream of zeros.** Storage that compresses or deduplicates (ZFS with compression, some appliances) writes almost nothing. Use random data, or fio, which fills buffers with random data by default.
- `bs=1M count=1` with a single process and no queue depth measures one stream. It can't show what 1,000 ranks do.

On macOS, `dd` has no `conv=fdatasync` or `iflag=direct` (BSD `dd`). Use the Python version below.

### The page cache in your own measurements

```python
# pagecache.py - what a naive write/read benchmark actually measures
import os, time
P, N = "/tmp/pc_test.bin", 1 << 30                  # 1 GiB
buf = os.urandom(64 << 20)                          # random, so compression/dedup can't cheat

def write(fsync):
    t0 = time.perf_counter()
    with open(P, "wb") as f:
        for _ in range(N // len(buf)): f.write(buf)
        f.flush()
        if fsync: os.fsync(f.fileno())
    return N / (time.perf_counter() - t0) / 1e9

def read():
    t0 = time.perf_counter()
    with open(P, "rb", buffering=0) as f:
        while f.read(16 << 20): pass
    return N / (time.perf_counter() - t0) / 1e9

print(f"write, no fsync : {write(False):5.1f} GB/s")
print(f"write + fsync   : {write(True):5.1f} GB/s")
print(f"read (cached)   : {read():5.1f} GB/s")
print(f"read again      : {read():5.1f} GB/s")
os.remove(P)
```

```text
# macOS, M4 laptop, internal SSD ~94% full, two runs
write, no fsync :   0.4 GB/s        write, no fsync :   0.6 GB/s
write + fsync   :   0.6 GB/s        write + fsync   :   0.6 GB/s
read (cached)   :   2.1 GB/s        read (cached)   :   3.0 GB/s
read again      :   6.1 GB/s        read again      :   3.2 GB/s
# Linux VM, one run
write, no fsync :   0.3 GB/s
write + fsync   :   0.8 GB/s
read (cached)   :   4.5 GB/s
read again      :   7.5 GB/s
```

I expected the no-fsync write to be much faster than the fsync write, and on both machines it wasn't. Both are writing through to a nearly full SSD (or the VM's disk image on it) under memory pressure, so the kernel throttles writers to about disk speed. Your server, with 1-2 TB of RAM and an empty NVMe, will probably show the gap that `dd` showed in its first session. The read side behaved as expected: cached reads run at RAM speed. The lesson is that the page cache makes some measurements meaningless, and you have to find out *which* ones on *your* machine, rather than assume.

### What to measure for ML

Benchmark the access pattern your job actually produces. Generic numbers don't transfer:

| Workload | Pattern | Measure |
|---|---|---|
| Checkpoint save | N ranks each write one or a few large files at once, then a barrier | aggregate write GB/s from all nodes at once, with fsync, and the *slowest* rank's time |
| Checkpoint load / model load | every rank reads its shard (or everyone reads the same file) | aggregate read GB/s, and the "thundering herd" case of all nodes reading one file |
| Pretokenized data | sequential reads of big shards from many workers | read GB/s at your worker count; is it more than you need (often trivially, see [section 8](#how-much-bandwidth-does-training-need))? |
| Image/video data | many medium reads, decode-heavy | samples/s *including decode*; the CPU is often the bottleneck |
| Environments, imports | thousands of small `stat`/`open` | `time python -c "import torch"` cold, from 64 nodes at once |


## 6. Tensor file formats

### Pickle, and why loading a checkpoint can run code

`torch.save` writes a zip file containing a **pickle** (Python's object serialization format) plus raw tensor bytes. A pickle isn't data. It is a small program for a stack machine, and loading it runs that program, which can call any importable function. Any object can control what happens on load through `__reduce__`:

```python
# pickle_risk.py  [torch]
import pickle, torch, os

class Evil:
    def __reduce__(self):
        # pickle stores "call os.system with this arg" - and load() obeys
        return (os.system, ("echo '  >>> arbitrary code ran during load'",))

torch.save({"weight": torch.zeros(2), "x": Evil()}, "/tmp/evil.pt")

print("torch.load(weights_only=False):", flush=True)
torch.load("/tmp/evil.pt", weights_only=False)

print("torch.load() default (weights_only=True since torch 2.6):")
try:
    torch.load("/tmp/evil.pt")
except pickle.UnpicklingError as e:
    print("  refused:", str(e).splitlines()[0][:100])
```

```text
torch.load(weights_only=False):
  >>> arbitrary code ran during load
torch.load() default (weights_only=True since torch 2.6):
  refused: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argumen
```

Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`, which allows only tensors, primitive types and an allow-list of classes. Code that "fixes" the error by passing `weights_only=False` to a file downloaded from the internet is running a stranger's program. Pickle also ties a checkpoint to your code: if the pickle references `mymodule.MyConfig`, renaming that class breaks loading.

### safetensors: a header plus raw bytes

[safetensors](https://github.com/huggingface/safetensors) is the format that replaced pickle for distributing weights. The whole format is:

```
+----------------+------------------------------------+---------------------------------------+
| N: u64 little- | JSON header (N bytes):             | raw tensor bytes, back to back,       |
| endian (8 B)   | {"name": {"dtype","shape",         | row-major, no padding inside tensors  |
|                |   "data_offsets":[begin,end]}, ... |                                       |
|                |  "__metadata__": {str: str}}       |                                       |
+----------------+------------------------------------+---------------------------------------+
```

There is no code and nothing to execute, so loading an untrusted file is safe (modulo parser bugs). Because the bytes are already laid out exactly as they sit in memory, a loader can **memory-map** the file. `mmap` asks the OS to make the file appear as memory, and pages are read from disk only when touched (see [Linux: page faults](linux-and-os.md#page-faults-and-lazy-allocation)). Opening the file costs only reading the header. One tensor, or a slice of one, costs reading just those bytes. That is what makes lazy loading of a single tensor-parallel shard out of a full checkpoint possible.

```python
# st_demo.py  [torch]
import json, struct, time, os, torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

torch.manual_seed(0)
# a fake "model": 8 layers of 2048x2048 bf16 = 8 * 8 MiB = 64 MiB, plus a 32000x512 embedding
state = {f"layers.{i}.weight": torch.randn(2048, 2048).to(torch.bfloat16) for i in range(8)}
state["embed.weight"] = torch.randn(32000, 512).to(torch.bfloat16)          # (V, C)
save_file(state, "/tmp/model.safetensors", metadata={"step": "1200"})
print(f"file size: {os.path.getsize('/tmp/model.safetensors') / 2**20:.1f} MiB")

# 1) the format, by hand: 8-byte little-endian header length, JSON header, raw bytes
with open("/tmp/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
print("header bytes:", n)
print("metadata:", header.pop("__metadata__"))
k = "layers.3.weight"
print(k, "->", header[k])

# 2) read ONE tensor's bytes ourselves using the offsets (offsets are relative to end of header)
start, end = header[k]["data_offsets"]
with open("/tmp/model.safetensors", "rb") as f:
    f.seek(8 + n + start)
    raw = bytearray(f.read(end - start))
mine = torch.frombuffer(raw, dtype=torch.bfloat16).view(2048, 2048)
print("hand-decoded equals original:", torch.equal(mine, state[k]))

# 3) lazy, mmap-backed access: open is O(header), a tensor is read only when asked for
t0 = time.perf_counter()
with safe_open("/tmp/model.safetensors", framework="pt") as f:
    t_open = time.perf_counter() - t0
    names = list(f.keys())
    row = f.get_slice("embed.weight")[100:104]     # (4, 512): reads ~4 KiB, not 32 MiB
print(f"safe_open + keys: {t_open*1e3:.2f} ms for {len(names)} tensors; slice shape {tuple(row.shape)}")
print("slice correct:", torch.equal(row, state["embed.weight"][100:104]))

# 4) "load" time vs torch.save/torch.load of the same dict (warm page cache)
torch.save(state, "/tmp/model.pt")
for name, fn in [("safetensors load_file", lambda: load_file("/tmp/model.safetensors")),
                 ("torch.load mmap=True ", lambda: torch.load("/tmp/model.pt", mmap=True)),
                 ("torch.load          ", lambda: torch.load("/tmp/model.pt"))]:
    fn(); t0 = time.perf_counter(); sd = fn(); dt = time.perf_counter() - t0
    print(f"{name}: {dt*1e3:6.1f} ms")

# 5) why load_file looks free: tensors are views of the mmap; pages are read on first touch
def touch(sd): return sum(t.clone().numel() for t in sd.values())   # forces every page in
for name, fn in [("safetensors load_file + touch", lambda: touch(load_file("/tmp/model.safetensors"))),
                 ("torch.load mmap=True + touch ", lambda: touch(torch.load("/tmp/model.pt", mmap=True))),
                 ("torch.load + touch          ", lambda: touch(torch.load("/tmp/model.pt")))]:
    fn(); t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
    print(f"{name}: {dt*1e3:6.1f} ms")
```

```text
file size: 95.3 MiB
header bytes: 832
metadata: {'step': '1200'}
layers.3.weight -> {'dtype': 'BF16', 'shape': [2048, 2048], 'data_offsets': [57933824, 66322432]}
hand-decoded equals original: True
safe_open + keys: 3.07 ms for 9 tensors; slice shape (4, 512)
slice correct: True
safetensors load_file:    0.2 ms
torch.load mmap=True :    1.3 ms
torch.load          :   22.2 ms
safetensors load_file + touch:   32.4 ms
torch.load mmap=True + touch :   35.8 ms
torch.load + touch          :   52.2 ms
```

Read the timings carefully, because this is a common benchmarking mistake. `load_file` in 0.2 ms didn't read 95 MiB. It mapped the file and returned tensors whose pages hadn't been touched yet. Once every byte is touched, the formats are within 2x of each other (on a warm page cache, with one run). The real advantages of safetensors are safety, partial reads, and no Python object reconstruction. "Faster load" claims need an honest touch-every-byte measurement. `torch.load(..., mmap=True)` (PyTorch 2.1+) gives pickle checkpoints the same lazy behaviour for the tensor bytes, but not the safety.

Also note that `layers.3.weight` sits at offset 57.9 MB, after the 32.8 MB embedding. The writer orders tensors itself, so never assume file order matches dict order. Always go through the header.

### Sharded checkpoints: the Hugging Face layout

Large models are stored as several files plus an index. The Hugging Face convention is `model-0000k-of-0000n.safetensors` files and a `model.safetensors.index.json` whose `weight_map` maps each tensor name to its file. Loaders read the index, then open only the files they need:

```python
# hf_shards.py  [torch, safetensors]  - the Hugging Face sharded-checkpoint layout, by hand
import json, os, torch
from safetensors.torch import save_file
from safetensors import safe_open

D = "/tmp/hf_ckpt"; os.makedirs(D, exist_ok=True)
state = {f"model.layers.{i}.mlp.weight": torch.randn(1024, 1024).to(torch.bfloat16) for i in range(6)}   # 2 MiB each
max_shard, shards, cur, size = 5 * 2**20, [], {}, 0
for k, t in state.items():                                   # greedy packing into <= 5 MiB files
    b = t.numel() * t.element_size()
    if cur and size + b > max_shard: shards.append(cur); cur, size = {}, 0
    cur[k] = t; size += b
shards.append(cur)

weight_map = {}
for i, sh in enumerate(shards, 1):
    fn = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
    save_file(sh, f"{D}/{fn}", metadata={"format": "pt"})
    weight_map |= {k: fn for k in sh}
json.dump({"metadata": {"total_size": sum(t.numel() * 2 for t in state.values())}, "weight_map": weight_map},
          open(f"{D}/model.safetensors.index.json", "w"), indent=1)

print(sorted(os.listdir(D)))
idx = json.load(open(f"{D}/model.safetensors.index.json"))
k = "model.layers.4.mlp.weight"
with safe_open(f"{D}/{idx['weight_map'][k]}", framework="pt") as f:   # open only the file that has it
    print(k, "lives in", idx["weight_map"][k], "->", tuple(f.get_tensor(k).shape))
```

```text
['model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors', 'model.safetensors.index.json']
model.layers.4.mlp.weight lives in model-00003-of-00003.safetensors -> (1024, 1024)
```

In practice `model.save_pretrained(dir, max_shard_size="5GB")` and `huggingface_hub.save_torch_state_dict` do this for you. The layout is the *distribution* format: full, unsharded tensors grouped into files by size. It is not how a running FSDP/TP job wants to save, because there each rank holds *pieces* of tensors.

### torch.distributed.checkpoint: every rank writes its own pieces

In a sharded training job (FSDP, tensor parallel, pipeline parallel; introduced in [Distributed Systems](distributed-systems.md#model-parallelism), in depth in [Model Parallelism](../training/model-parallelism/README.md)), no rank has the full model. The naive save, gathering everything to rank 0 and calling `torch.save`, sends the whole model through one GPU's memory and one node's network link and disk. At 405B scale that takes hours, or runs out of memory.

**torch.distributed.checkpoint (DCP)** has each rank write only the shards it owns, in parallel, plus one small global `.metadata` file describing how the pieces fit together. Loading can use a *different* parallel layout than saving: DCP reads the metadata and gives each rank the byte ranges that make up the shard it now owns. This is **resharding on load**, and it is what lets you resume a 256-GPU run on 192 GPUs.

New PyTorch pieces in the demo below: `mp.spawn(run, nprocs=2)` starts two processes that each call `run(rank, ...)`, and they form a process group over the CPU-only `gloo` backend (both as in [Distributed Systems](distributed-systems.md#communication-models)). A **device mesh** (`init_device_mesh`) is a named grid of ranks, here just a line of 2. A **DTensor** (`distribute_tensor(full, mesh, [Shard(0)])`) is a tensor that knows it is split across that mesh, here by rows (dim 0), so each rank physically holds half of it; `w.to_local()` is that rank's half. FSDP2 (the current version of PyTorch's FSDP) stores parameters as DTensors, which is why DCP understands them.

```python
# dcp_demo.py  [torch]  - torch.distributed.checkpoint on 2 CPU ranks (gloo)
import os, shutil, torch, torch.distributed as dist, torch.multiprocessing as mp
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import distribute_tensor, Shard, init_device_mesh

CKPT = "/tmp/dcp_ckpt"

def run(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    full = torch.randn(1024, 256)                                  # (out, in), same on all ranks
    w = distribute_tensor(full, mesh, [Shard(0)])                  # each rank owns 512 rows
    state = {"model": {"w": w}, "step": 1200}
    print(f"rank {rank}: local shard {tuple(w.to_local().shape)}", flush=True)

    dcp.save(state, checkpoint_id=CKPT)                            # every rank writes only its shard
    fut = dcp.async_save(state, checkpoint_id=CKPT + "_async")     # stage to CPU, write in background
    fut.result()                                                   # (a real loop keeps training here)

    # reload into a plain, unsharded tensor: a different layout than was saved -> DCP reshards
    target = {"model": {"w": torch.zeros(1024, 256)}, "step": 0}
    dcp.load(target, checkpoint_id=CKPT)
    if rank == 0:
        print("reloaded full tensor equals original:", torch.equal(target["model"]["w"], full), "step:", target["step"])
    dist.destroy_process_group()

if __name__ == "__main__":
    shutil.rmtree(CKPT, ignore_errors=True); shutil.rmtree(CKPT + "_async", ignore_errors=True)
    mp.spawn(run, args=(2,), nprocs=2)
    for f in sorted(os.listdir(CKPT)):
        print(f"{f:20s} {os.path.getsize(os.path.join(CKPT, f)):>9,} bytes")
```

```text
rank 1: local shard (512, 256)
rank 0: local shard (512, 256)
reloaded full tensor equals original: True step: 1200
.metadata                1,094 bytes
__0_0.distcp           525,865 bytes
__1_0.distcp           527,126 bytes
```

Each `.distcp` file is one rank's $512 \times 256 \times 4\ \text{B} = 524{,}288$ bytes of tensor data plus a little framing. With FSDP2 you'd use `torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizer)` to get sharded model and optimizer state in this form, and `set_state_dict` after loading. DCP's `StorageWriter`/`StorageReader` interfaces are how you plug in an object-store backend (an `fsspec`-based writer ships with PyTorch). To export a final model for inference, convert DCP to a single `torch.save` file with `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`, and from there to safetensors. API surface as of torch 2.15. DCP has moved quickly, so check the [docs](https://pytorch.org/docs/stable/distributed.checkpoint.html) for your version.

| Format | Code on load | Partial / lazy reads | Sharded save from many ranks | Reshard on load | Use for |
|---|---|---|---|---|---|
| `torch.save` (pickle) | yes, unless `weights_only=True` | with `mmap=True` | no (one file per writer, you manage it) | no | small experiments |
| safetensors | no | yes (header + offsets) | by hand, or HF sharding | no (full tensors) | distribution, inference loading |
| DCP | no for tensors (metadata is pickled internally; only load checkpoints you produced) | yes | yes, the point of it | yes | training checkpoints |


## 7. Checkpoint I/O arithmetic

### What's in a training checkpoint

For mixed-precision training with Adam, each parameter carries (see [Training arithmetic: static memory per parameter](../training/arithmetic.md#static-per-parameter)):

| Item | Bytes/param |
|---|---|
| bf16 working weights | 2 |
| fp32 master weights | 4 |
| Adam first moment `m` (fp32) | 4 |
| Adam second moment `v` (fp32) | 4 |
| **Total for a resumable checkpoint** | **14** (16 if fp32 grads or fp32 weights are also saved; 12 if bf16 weights are rebuilt from master) |
| Inference export (bf16 weights only) | 2 |

Plus the small but essential extras: the step count, LR scheduler state, RNG states for every rank, the data loader position ([section 8](#deterministic-resumable-data-loading)), and the config and git SHA. These are kilobytes, and without them the checkpoint can't reproduce the run.

### Size and time for 8B, 70B and 405B

```python
# ckpt_math.py - checkpoint size and write time
import math
models = {"8B": 8.03e9, "70B": 70.6e9, "405B": 405.9e9}     # Llama-3.1 parameter counts
# bytes/param: bf16 weights only (inference/export) vs full mixed-precision Adam training state
#   training: fp32 master weights 4 + Adam m 4 + Adam v 4 + bf16 weights 2 = 14
bpp = {"weights bf16 (2 B/p)": 2, "train state (14 B/p)": 14}
bws = {"1 GB/s": 1e9, "10 GB/s": 10e9, "100 GB/s": 100e9, "1 TB/s": 1e12}

print(f"{'model':5} {'what':22} {'size':>9} " + " ".join(f"{k:>10}" for k in bws))
for m, n in models.items():
    for what, b in bpp.items():
        size = n * b
        ts = [size / bw for bw in bws.values()]
        fmt = lambda t: f"{t/60:8.1f}m" if t >= 120 else f"{t:8.1f}s"
        print(f"{m:5} {what:22} {size/1e9:7.0f}GB " + " ".join(f"{fmt(t):>10}" for t in ts))

# GPU -> host snapshot, "D2H" = device-to-host copy (the only part async checkpointing cannot hide)
n_gpu, pcie = 1024, 25e9         # 405B train state over 1024 GPUs; ~25 GB/s achieved D2H per GPU (assumption)
per_gpu = models["405B"] * 14 / n_gpu
print(f"\n405B train state over {n_gpu} GPUs: {per_gpu/1e9:.1f} GB/GPU -> D2H snapshot {per_gpu/pcie:.2f} s at 25 GB/s")
print(f"host RAM for one staged copy per 8-GPU node: {per_gpu*8/1e9:.0f} GB")

# Young/Daly optimal interval: tau = sqrt(2 * C * MTBF)
mtbf_h = 54 * 24 / 466           # Llama 3 405B: 466 interruptions in 54 days
for C in (5, 60, 600):           # seconds of training blocked per checkpoint
    tau = math.sqrt(2 * C * mtbf_h * 3600)
    waste = C / tau + tau / (2 * mtbf_h * 3600)        # checkpoint overhead + expected lost work
    print(f"MTBF {mtbf_h:.2f} h, blocking cost {C:4d} s -> checkpoint every {tau/60:5.1f} min, ~{100*waste:.1f}% time lost")
```

```text
model what                        size     1 GB/s    10 GB/s   100 GB/s     1 TB/s
8B    weights bf16 (2 B/p)        16GB      16.1s       1.6s       0.2s       0.0s
8B    train state (14 B/p)       112GB     112.4s      11.2s       1.1s       0.1s
70B   weights bf16 (2 B/p)       141GB       2.4m      14.1s       1.4s       0.1s
70B   train state (14 B/p)       988GB      16.5m      98.8s       9.9s       1.0s
405B  weights bf16 (2 B/p)       812GB      13.5m      81.2s       8.1s       0.8s
405B  train state (14 B/p)      5683GB      94.7m       9.5m      56.8s       5.7s

405B train state over 1024 GPUs: 5.5 GB/GPU -> D2H snapshot 0.22 s at 25 GB/s
host RAM for one staged copy per 8-GPU node: 44 GB
MTBF 2.78 h, blocking cost    5 s -> checkpoint every   5.3 min, ~3.2% time lost
MTBF 2.78 h, blocking cost   60 s -> checkpoint every  18.3 min, ~10.9% time lost
MTBF 2.78 h, blocking cost  600 s -> checkpoint every  57.8 min, ~34.6% time lost
```

What the bandwidth columns correspond to (orders of magnitude, not quotes):

- **1 GB/s**: one NFS client, a single-stream object-store upload, or a slow WAN link (wide-area network: a link between sites or regions).
- **10 GB/s**: one node writing to a few local NVMe drives, or one node's share of a good parallel FS.
- **100 GB/s**: the aggregate of tens of nodes writing to a strong parallel FS at once. The Storage chapter's [JeanZay case study](../storage/README.md#which-file-system-to-choose), 2.3 TB in 40 s from 384 processes, works out to about 58 GB/s.
- **1 TB/s**: hundreds of nodes each writing to *their own local NVMe* at a few GB/s. This is why fast checkpointing systems write locally first.

The bottleneck is usually the smallest of: per-GPU device-to-host copy, per-node NIC to storage (for example $2 \times 200\ \text{Gb/s} = 50\ \text{GB/s}$), storage-side aggregate bandwidth, and metadata (file creates). Do the division for each and take the minimum.

### How often to checkpoint

The Young/Daly formula (Young 1974; Daly 2006; first met in [Distributed Systems](distributed-systems.md#how-often-to-checkpoint)) balances two costs. Checkpointing often wastes the blocking time $C$ on every save. Checkpointing rarely loses on average half an interval of work per failure. With mean time between failures $M$, the optimum interval is about

$$
\tau \approx \sqrt{2 C M}
$$

The script plugs in the failure rate reported for Llama 3 405B pre-training on 16k H100s: 466 job interruptions in a 54-day snapshot ([The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), arXiv 2407.21783). (Distributed Systems used 419 from the same table: that is the *unexpected* interruptions only; the other 47 were planned, such as maintenance. For checkpoint planning every interruption costs you work, so this chapter uses the total.) That gives an MTBF of 2.8 hours. The numbers are first-order: they ignore restart time, which at this scale is often minutes and adds to every failure. The conclusion still holds. Cutting the blocking cost from 10 minutes to 5 seconds cuts the overhead from about 35% to about 3%, and that is what async checkpointing buys.

### Async checkpointing

The mechanism and a runnable minimal version are in [Concurrency: Async checkpointing](concurrency.md#async-checkpointing): snapshot to host memory (blocking, fast), persist from a background thread or process (overlapped), and never have more than one save in flight. The arithmetic above shows why the split works. The 405B snapshot is 0.22 s of D2H copy per GPU, under the assumed 25 GB/s. The write, minutes at realistic bandwidth, is hidden behind training steps, as long as the write finishes before the next checkpoint is due.

Production systems layer storage tiers so that the blocking part only touches the fastest tier:

```
 step N  ──► GPU HBM ──(D2H, pinned buffers, ~0.2 s)──► host RAM ──► training resumes
                                                         │
                         background, overlapped with steps N+1 ...
                                                         ├─► local NVMe          (seconds; survives process crash)
                                                         ├─► parallel FS         (minutes; survives node loss)
                                                         └─► object store, async (minutes-hours; survives the cluster)
                                     keep: last 2-3 on fast tiers, every k-th forever in object store
```

```python
# untested at scale; DCP async save in a training loop (the pattern, torch >= 2.3 API)
import torch.distributed.checkpoint as dcp
pending = None
for step in range(start, total_steps):
    train_step()
    if step % ckpt_every == 0:
        if pending is not None:
            pending.result()                  # backpressure: previous save must have finished
        state = {"model": model_sd(), "optim": optim_sd(), "step": step, "data": loader.state_dict()}
        pending = dcp.async_save(state, checkpoint_id=f"/lustre/ckpts/run1/step_{step:08d}")
if pending is not None:
    pending.result()
```

Host RAM is the hidden constraint. One staged copy for 405B on 1,024 GPUs is 44 GB per 8-GPU node, which is affordable. For 405B on 128 GPUs it would be 355 GB per node, and double-buffering doubles it. Do this arithmetic before turning on pinned staging buffers.

Replication is an alternative to storage for fast recovery. Some systems keep each rank's checkpoint shard in the host RAM of a *peer* node (sent over the training network), so a single node failure is recovered from memory without touching storage. GEMINI (Wang et al., SOSP 2023) and [Fault tolerance](../training/fault-tolerance/README.md) discuss this. For decentralized training, where each "site" has slow links to the others, the checkpoint and transfer arithmetic dominates the design. See [Decentralized Training](../training/decentralized-training.md).

### Making checkpoints trustworthy

A checkpoint you can't trust is worse than none, because you find out at resume time, days later. Rules:

1. **Write, fsync, then mark complete.** The atomic-rename recipe is in [Linux: fsync and durability](linux-and-os.md#fsync-and-durability-why-checkpoints-get-lost). For a multi-rank checkpoint, every rank fsyncs its files, then a barrier, then rank 0 writes a `COMPLETE` marker. On object storage the marker is the last object written.
2. **Resume only from complete checkpoints.**

```python
# latest.py - resume only from checkpoints that finished writing
import os, glob, json, shutil
R = "/tmp/run_ckpts"; shutil.rmtree(R, ignore_errors=True)
for step, done in [(1000, True), (2000, True), (3000, False)]:       # 3000 died mid-write
    d = f"{R}/step_{step:08d}"; os.makedirs(d)
    open(f"{d}/shard_0.bin", "wb").write(b"\0" * 10)
    if done:                                                         # written LAST, after every rank fsynced
        json.dump({"step": step, "world_size": 64}, open(f"{d}/COMPLETE.json", "w"))

def latest_complete(root):
    good = [d for d in sorted(glob.glob(f"{root}/step_*")) if os.path.exists(f"{d}/COMPLETE.json")]
    return good[-1] if good else None

print("newest dir:     ", sorted(glob.glob(f"{R}/step_*"))[-1])
print("resume from:    ", latest_complete(R))
```

```text
newest dir:      /tmp/run_ckpts/step_00003000
resume from:     /tmp/run_ckpts/step_00002000
```

3. **Zero-pad step numbers** (`step_00002000`) so sorting by name sorts by step.
4. **Record checksums** of every shard in the marker, or alongside it. Verify them when copying between tiers or sites ([section 9](#checksums)).
5. **Test resume, once, early.** Save at step 100, kill the job, resume, and check that the loss at step 101 matches an uninterrupted run to the last bit (or within the run's known nondeterminism). This catches a missing optimizer state, LR scheduler or data position before it costs you a week. See [Reproducibility](../training/reproducibility/README.md).
6. **Delete on a policy, not by hand.** See the Storage chapter's [automatic deletion of old checkpoints](../storage/README.md#how-to-automatically-delete-old-checkpoints).


## 8. Datasets at scale

### How much bandwidth does training need?

Do this arithmetic before optimizing a text data loader:

```python
toks = 16384 * 989e12 * 0.40 / (6 * 405.9e9)   # GPUs x peak bf16 FLOP/s x MFU / (6N FLOPs per token)
print(f"405B on 16384 H100 @40% MFU: {toks/1e6:.2f} M tok/s -> {toks*4/1e6:.1f} MB/s of uint32 tokens")
toks8 = 8 * 989e12 * 0.40 / (6 * 8.03e9)
print(f"8B on 8 H100 @40%: {toks8/1e3:.0f} k tok/s -> {toks8*4/1e6:.2f} MB/s")
print(f"15T tokens: uint16 {15e12*2/1e12:.0f} TB, uint32 {15e12*4/1e12:.0f} TB")
print(f"vision: 64 GPUs x 2000 img/s x 110 KB = {64*2000*110e3/1e9:.1f} GB/s")   # 2000 img/s/GPU is an assumption
```

```text
405B on 16384 H100 @40% MFU: 2.66 M tok/s -> 10.6 MB/s of uint32 tokens
8B on 8 H100 @40%: 66 k tok/s -> 0.26 MB/s
15T tokens: uint16 30 TB, uint32 60 TB
vision: 64 GPUs x 2000 img/s x 110 KB = 14.1 GB/s
```

(`6N` FLOPs per token and MFU are derived in [Training arithmetic](../training/arithmetic.md#flops-of-a-decoder-transformer).) **Pretokenized LLM pretraining needs about 10 MB/s for the whole cluster.** Any storage can deliver that. Text data problems are almost never bandwidth problems. They are latency spikes (one slow read stalls a synchronous step on all 16k GPUs), metadata storms, CPU-bound tokenization on the fly, and non-resumable loaders. Image, video and audio are a different story: 14 GB/s of compressed media plus CPU or GPU decode is a real pipeline engineering problem.

### Tokenization pipelines

Tokenize once, offline, and train from token IDs. On-the-fly tokenization in DataLoader workers costs CPU on GPU nodes, makes throughput depend on document length, and makes exact resumption harder. A typical offline pipeline:

```
raw text (jsonl.zst = one JSON doc per line, zstd-compressed / parquet = a columnar table file format; on object store)
  -> filter + dedup (separate stage, separate chapter of your life)
  -> tokenize in parallel (CPU nodes, many processes)
  -> concatenate documents with an end-of-text (EOT) token between them
  -> cut into fixed-size shards of token ids (uint16 if vocab < 65536, else uint32)
  -> write shards + a manifest (shard list, token counts, tokenizer hash) to object store
```

```python
# pretok.py  [numpy, tiktoken]  - parallel tokenization into fixed-size uint16 shards
import os, time, numpy as np, tiktoken
from multiprocessing import Pool

enc = tiktoken.get_encoding("gpt2")                 # vocab 50257 < 2**16, so uint16 is enough
EOT = enc.eot_token

def tok(doc):                                        # runs in a worker process
    return np.array([EOT] + enc.encode_ordinary(doc), dtype=np.uint16)

if __name__ == "__main__":
    docs = [f"Document {i}. " + "Storage is the slowest part of the system, so overlap it. " * (i % 50 + 1)
            for i in range(20_000)]
    out, shard_tokens = "/tmp/pretok", 1_000_000
    os.makedirs(out, exist_ok=True)
    t0, buf, n, k = time.perf_counter(), [], 0, 0
    with Pool(os.cpu_count()) as pool:
        for ids in pool.imap(tok, docs, chunksize=256):   # imap keeps document order -> reproducible shards
            buf.append(ids); n += len(ids)
            if n >= shard_tokens:
                flat = np.concatenate(buf)
                flat[:shard_tokens].tofile(f"{out}/train_{k:06d}.bin")
                buf, n, k = [flat[shard_tokens:]], n - shard_tokens, k + 1
    if n: np.concatenate(buf).tofile(f"{out}/train_{k:06d}.bin"); k += 1
    dt = time.perf_counter() - t0
    total = sum(os.path.getsize(f"{out}/{f}") for f in os.listdir(out)) // 2
    print(f"{len(docs):,} docs -> {total:,} tokens in {k} shards, {total/dt/1e6:.2f} M tok/s on {os.cpu_count()} procs")
    m = np.memmap(f"{out}/train_000000.bin", dtype=np.uint16, mode="r")
    print("decoded:", repr(enc.decode(m[:24].tolist())))
    print("max id in shard:", int(m.max()), "< 65536")
```

```text
20,000 docs -> 7,259,830 tokens in 8 shards, 12.96 M tok/s on 10 procs
decoded: '<|endoftext|>Document 0. Storage is the slowest part of the system, so overlap it. <|endoftext|>Document 1. Storage'
max id in shard: 50256 < 65536
```

Notes:

- **Decode a sample and look at it** every time you build shards. The `decoded:` line catches wrong tokenizers, missing EOT tokens and byte-order bugs in seconds.
- **dtype matters.** Llama 3's 128,256-token vocabulary doesn't fit in uint16. Writing it as uint16 silently wraps IDs above 65,535 into other tokens. Assert `max_id < 2**16` or use uint32.
- The 13M tok/s is optimistic. The text is very repetitive, which BPE handles fast. Real web text is slower; measure on your corpus. At 15T tokens, even 10M tok/s means 17 days on one machine, which is why this runs on a CPU cluster (Ray and Spark are Python/JVM frameworks for spreading tasks over many machines; a Slurm array job submits many copies of one script, each with its own index) with one task per input file.
- **Store the tokenizer's identity** (name, version, hash of its vocab file) in the manifest. A model trained on shards from tokenizer A and evaluated with tokenizer B is a silent disaster.

### Pretokenized shards and memory-mapped token arrays

The simplest high-performance training format is a flat binary array of token IDs per shard, as in nanoGPT and llm.c. A training sample is a window of `T+1` consecutive tokens: the input is the first `T`, the target is the last `T`. With `np.memmap`, the shard appears as an array, and only the pages you slice are read from disk:

```python
# tokshards.py  [numpy, torch]  - pretokenize -> uint16 shards -> memmap loader -> resumable sampler
import glob, os, numpy as np, torch

DIR, T = "/tmp/tokshards", 64                       # T = context length

# ---- 1) pretokenize: text -> token ids -> flat uint16 shard files (nanoGPT-style .bin)
def tokenize(s):                                    # byte-level "tokenizer", vocab 256
    return list(s.encode("utf-8"))

def write_shards(docs, shard_tokens=50_000, eot=0):
    os.makedirs(DIR, exist_ok=True)
    buf, k = [], 0
    for d in docs:
        buf += tokenize(d) + [eot]                  # documents separated by an end-of-text id
        if len(buf) >= shard_tokens:
            np.array(buf[:shard_tokens], dtype=np.uint16).tofile(f"{DIR}/shard_{k:05d}.bin")
            buf, k = buf[shard_tokens:], k + 1
    if buf:
        np.array(buf, dtype=np.uint16).tofile(f"{DIR}/shard_{k:05d}.bin")

docs = [f"document {i}: the quick brown fox number {i} jumps over {i % 7} lazy dogs." for i in range(20_000)]
write_shards(docs)

# ---- 2) memmap loader: no file is read until a window is sliced
class TokenWindows:
    def __init__(self, pattern, T):
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in sorted(glob.glob(pattern))]
        self.T = T
        # window w lives in shard s at offset o; windows never straddle shards (simple, loses < T tokens/shard)
        self.index = [(s, o) for s, m in enumerate(self.shards) for o in range(0, len(m) - T - 1, T)]

    def __len__(self): return len(self.index)

    def __getitem__(self, w):
        s, o = self.index[w]
        chunk = torch.from_numpy(self.shards[s][o:o + self.T + 1].astype(np.int64))   # (T+1,)
        return chunk[:-1], chunk[1:]                                                  # x, y: (T,), (T,)

ds = TokenWindows(f"{DIR}/shard_*.bin", T)
n_tok = sum(len(m) for m in ds.shards)
print(f"{len(ds.shards)} shards, {n_tok:,} tokens, {len(ds):,} windows of T={T}")
x, y = ds[1234]
print("look at the data  x:", repr(bytes(x.tolist()).decode()))
print("                  y:", repr(bytes(y.tolist()).decode()))
```

```text
29 shards, 1,437,780 tokens, 22,458 windows of T=64
look at the data  x: ' lazy dogs.\x00document 1144: the quick brown fox number 1144 jumps'
                  y: 'lazy dogs.\x00document 1144: the quick brown fox number 1144 jumps '
```

`y` is `x` shifted left by one token, which is the next-token objective made visible. The window crosses a document boundary (`\x00` is our EOT). Whether to allow that or to mask attention across documents is a modelling choice, and you can only see it by looking.

Why this design is so robust:

- **No parsing, no per-sample files.** A sample is one slice of a mapped array. 29 files instead of 20,000.
- **mmap shares memory across DataLoader workers.** All workers map the same file and the OS keeps one copy in the page cache. That avoids the copy-on-write blow-up of Python lists described in [Linux: Copy-on-write after fork](linux-and-os.md#copy-on-write-after-fork-and-the-dataloader-memory-leak). The `index` list above *is* a Python list of tuples, so at a billion windows you'd compute `(s, o)` arithmetically from cumulative shard lengths instead of storing it.
- **Random access is cheap on local NVMe and expensive on a network FS.** The Storage chapter [measured shuffled mmap access on Lustre at 50x slower](../storage/README.md#mmap-vs-sequential-dataset-reads) than a sequential read. Either stage shards to local NVMe first (at the 10 MB/s needed for text, staging the next few shards while training is trivial), or read whole shards sequentially and shuffle in memory.

### Deterministic, resumable data loading

Resuming after a crash must continue the data stream exactly where it stopped. Otherwise you silently train on some data twice and some never, and runs aren't reproducible. The trick is to make the order a pure function of `(seed, epoch)` and store only a position:

```python
# (continues tokshards.py)
# ---- 3) deterministic, resumable, rank-sharded sampler
class ResumableSampler:
    """Order is a pure function of (seed, epoch); position is the only mutable state."""
    def __init__(self, n, rank, world, seed=1234):
        self.n, self.rank, self.world, self.seed = n, rank, world, seed
        self.epoch, self.pos = 0, 0                 # pos = samples already yielded by this rank this epoch

    def _order(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g)
        perm = perm[: self.n - self.n % self.world]            # drop tail so every rank gets the same count
        return perm[self.rank :: self.world]                    # rank's strided slice

    def __iter__(self):
        while True:
            order = self._order()
            while self.pos < len(order):
                i = int(order[self.pos]); self.pos += 1
                yield i
            self.epoch, self.pos = self.epoch + 1, 0

    def state_dict(self): return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed}
    def load_state_dict(self, sd): self.epoch, self.pos, self.seed = sd["epoch"], sd["pos"], sd["seed"]

def batches(sampler, B):
    it = iter(sampler)
    while True:
        idx = [next(it) for _ in range(B)]
        xs, ys = zip(*(ds[i] for i in idx))
        yield idx, torch.stack(xs), torch.stack(ys)             # (B, T), (B, T)

B, world = 8, 4
# uninterrupted run on rank 1: 30 steps
ref = ResumableSampler(len(ds), rank=1, world=world)
g = batches(ref, B); ref_idx = [next(g)[0] for _ in range(30)]

# interrupted run: 17 steps, "checkpoint", die, new process restores, 13 more steps
a = ResumableSampler(len(ds), rank=1, world=world)
g = batches(a, B); got = [next(g)[0] for _ in range(17)]
ckpt = a.state_dict(); print("sampler state at step 17:", ckpt)
b = ResumableSampler(len(ds), rank=1, world=world); b.load_state_dict(ckpt)
g = batches(b, B); got += [next(g)[0] for _ in range(13)]
print("resumed stream identical to uninterrupted:", got == ref_idx)

# ranks are disjoint and cover the epoch
per_rank = [set(ResumableSampler(len(ds), r, world)._order().tolist()) for r in range(world)]
print("ranks disjoint:", all(per_rank[i].isdisjoint(per_rank[j]) for i in range(world) for j in range(i + 1, world)),
      "| covered:", sum(map(len, per_rank)), "of", len(ds))
x, y = next(batches(b, B))[1:]
print("batch shapes:", tuple(x.shape), tuple(y.shape), x.dtype)
```

```text
sampler state at step 17: {'epoch': 0, 'pos': 136, 'seed': 1234}
resumed stream identical to uninterrupted: True
ranks disjoint: True | covered: 22456 of 22458
batch shapes: (8, 64) (8, 64) torch.int64
```

The whole data state is three integers, and the uninterrupted and interrupted streams match exactly. Every rank computes the same permutation from the same seed and takes a disjoint strided slice, so no communication is needed. The 2 dropped windows are the tail that doesn't divide by the world size. This sampler re-derives the order for each rank, so it does **not** support resuming with a different world size: `pos` counts samples *per rank*. To support elastic resumption, store a *global* sample counter and derive each rank's position from it.

The trap that breaks this in practice is prefetching. A PyTorch `DataLoader` with workers pulls indices from the sampler ahead of what training has consumed:

```python
# overshoot.py  [torch]  - the sampler runs ahead of training when DataLoader prefetches
import torch
from torch.utils.data import DataLoader

class Counting(torch.utils.data.Sampler):
    def __init__(self, n): self.n, self.pos = n, 0
    def __iter__(self):
        for i in range(self.n):
            self.pos += 1
            yield i
    def __len__(self): return self.n

if __name__ == "__main__":
    data = torch.arange(10_000).float()
    s = Counting(len(data))
    dl = DataLoader(data, batch_size=8, sampler=s, num_workers=2, prefetch_factor=4)
    it = iter(dl)
    for step in range(17):
        next(it)
    print("samples consumed by training:", 17 * 8)
    print("samples drawn from sampler:  ", s.pos)
```

```text
samples consumed by training: 136
samples drawn from sampler:   200
```

The difference is 64 = `num_workers × prefetch_factor × batch_size` = $2 \times 4 \times 8$, the in-flight batches. Saving `sampler.state_dict()` here would skip 64 samples on resume. The fixes: count consumed samples in the training loop (`samples_seen += B` after each step) and checkpoint that, or use `torchdata`'s `StatefulDataLoader` (`from torchdata.stateful_dataloader import StatefulDataLoader`, torchdata 0.8+), which tracks what was actually delivered and has `state_dict()`/`load_state_dict()`. The StatefulDataLoader is untested here.

### Shuffling at scale

A full random permutation is the gold standard, but it doesn't always fit:

- **Materializing the permutation.** $10^{10}$ windows $\times$ 8 bytes (int64) = 80 GB per rank. Alternatives: a permutation computed on demand (a seeded bijection on `[0, N)` such as a Feistel network, which gives `perm(i)` in $O(1)$ memory), or a two-level shuffle.
- **Random access on remote storage** is slow (see above), so streaming formats read sequentially and shuffle approximately.

The standard approximate scheme is **two-level shuffling**: shuffle the order of shards, then pass samples through a **shuffle buffer** that holds `k` samples and emits a random one each time a new one arrives. Here it is over WebDataset-style tar shards (WebDataset, [described below](#streaming-from-object-storage-webdataset-and-mosaicml-streaming), stores each sample as a few files sharing a name inside an ordinary tar archive):

```python
# tarstream.py  - WebDataset-style tar shards: sequential reads, shard shuffle + shuffle buffer
import io, json, random, tarfile, glob, os

D = "/tmp/wds"; os.makedirs(D, exist_ok=True)
# write: each sample = files sharing a basename ("key"); one tar = one shard
for s in range(4):
    with tarfile.open(f"{D}/shard-{s:06d}.tar", "w") as tar:
        for i in range(250):
            key = f"{s:02d}{i:04d}"
            for ext, payload in [("txt", f"caption for sample {key}".encode()),
                                 ("json", json.dumps({"id": key, "len": i}).encode())]:
                ti = tarfile.TarInfo(f"{key}.{ext}"); ti.size = len(payload)
                tar.addfile(ti, io.BytesIO(payload))

def samples(url):                                   # url could be a local path or an HTTP/S3 stream
    with tarfile.open(url, "r|") as tar:           # "r|" = pure streaming, no seeking
        cur, key = {}, None
        for m in tar:
            k, ext = m.name.rsplit(".", 1)
            if key is not None and k != key: yield cur; cur = {}
            key, cur[ext] = k, tar.extractfile(m).read()
        if cur: yield cur

def stream(urls, seed, epoch, buf=200):
    rng = random.Random(seed * 1000 + epoch)
    urls = urls[:]; rng.shuffle(urls)               # level 1: shuffle shard order
    pool = []
    for u in urls:
        for s in samples(u):
            pool.append(s)
            if len(pool) >= buf:                    # level 2: shuffle buffer
                yield pool.pop(rng.randrange(len(pool)))
    rng.shuffle(pool); yield from pool

urls = sorted(glob.glob(f"{D}/shard-*.tar"))
out = [json.loads(s["json"])["id"] for s in stream(urls, seed=0, epoch=0)]
print("first 8 ids:", out[:8])
print("all 1000 seen once:", sorted(out) == sorted({f"{s:02d}{i:04d}" for s in range(4) for i in range(250)}))
# how "shuffled" is it? fraction of consecutive pairs coming from the same shard
same = sum(a[:2] == b[:2] for a, b in zip(out, out[1:])) / (len(out) - 1)
print(f"consecutive pairs from same shard: {same:.0%} (fully random would be ~25%)")
print("deterministic:", out == [json.loads(s["json"])["id"] for s in stream(urls, 0, 0)])
```

```text
first 8 ids: ['020066', '020131', '020125', '020104', '020078', '020126', '020093', '020156']
all 1000 seen once: True
consecutive pairs from same shard: 54% (fully random would be ~25%)
deterministic: True
```

```bash
$ tar tvf /tmp/wds/shard-000000.tar | head -4
-rw-r--r--  0 0      0          25 Dec 31  1969 000000.txt
-rw-r--r--  0 0      0          26 Dec 31  1969 000000.json
-rw-r--r--  0 0      0          25 Dec 31  1969 000001.txt
-rw-r--r--  0 0      0          26 Dec 31  1969 000001.json
```

(The `tar tvf` listing shows permissions, owner, size, date and name per member; the 1969 dates are just Unix time 0, because the script never set a modification time.)

The metric shows the weakness. With a buffer of 200 and shards of 250 samples, 54% of neighbouring samples come from the same shard, against 25% for a true shuffle. Approximate shuffles are biased toward shard locality, and the bias matters when shards are *sorted*, for example by source, by language, or by crawl date: a batch drawn mostly from one shard has a skewed distribution, which can show up as periodic loss spikes. Three remedies: make the buffer much larger than one shard, read several shards at once and interleave them (WebDataset and MosaicML Streaming both do this through workers and their own sampling schemes), and, most important, **randomize sample order when you write the shards**, so that each shard is already a random sample of the corpus. Pre-shuffling is a one-time cost at write time and fixes the problem at the source.

### Streaming from object storage: WebDataset and MosaicML Streaming

When the dataset doesn't fit on local disk, or lives in another region, you stream shards from object storage during training. Two widely used libraries take this approach. Both are untested here; check their docs for current APIs.

- **[WebDataset](https://github.com/webdataset/webdataset)** stores samples as plain POSIX tar files (exactly the format above) and opens shards by URL, including `pipe:` URLs that stream from any command. Sequential reads of 100 MB-1 GB tars suit both object stores and parallel filesystems.

```python
# untested; WebDataset pattern
import webdataset as wds
urls = "pipe:aws s3 cp s3://my-bucket/train/shard-{000000..009999}.tar -"
ds = (wds.WebDataset(urls, shardshuffle=True)      # shuffle shard order
        .shuffle(5000)                            # shuffle buffer of 5000 samples
        .decode("pil")
        .to_tuple("jpg", "json"))
```

- **[MosaicML Streaming](https://github.com/mosaicml/streaming)** uses its own MDS shard format (Mosaic Data Shard; unrelated to Lustre's metadata server, also called MDS) with an index file. It caches shards on local disk (the `local=` directory), downloads ahead of use, and is built for deterministic resumption across changes in the number of nodes. Its documentation calls this "elastic determinism".

```python
# untested; MosaicML Streaming pattern
from streaming import MDSWriter, StreamingDataset
with MDSWriter(out="s3://my-bucket/mds/train", columns={"tokens": "ndarray:uint32"}) as w:
    for ids in token_arrays: w.write({"tokens": ids})
ds = StreamingDataset(remote="s3://my-bucket/mds/train", local="/local/mds-cache",
                      shuffle=True, batch_size=8)
```

Whatever library you use, the design questions are the same: shard size (100 MB-1 GB: big enough to amortize request latency, small enough to spread over workers and resume cheaply), how many shards are in flight per worker, how much local cache, how the state is checkpointed (shard ID + sample offset + RNG state), and what happens when a GET fails (retry with backoff; one stuck request must not stall 1,000 GPUs).


## 9. Moving data between sites

### Transfer time and the bandwidth-delay product

```python
for tb, gbps, eff in [(10,10,0.7), (100,10,0.7), (100,100,0.7), (1000,100,0.7)]:
    s = tb * 1e12 * 8 / (gbps * 1e9 * eff)
    print(f"{tb:5d} TB over {gbps:3d} Gb/s at {eff:.0%} efficiency: {s/3600:6.1f} h")
rtt, bw = 0.08, 10e9
print(f"BDP 10 Gb/s x 80 ms = {bw*rtt/8/1e6:.0f} MB in flight; one stream with 4 MiB window maxes at {4*2**20*8/rtt/1e6:.0f} Mb/s")
```

```text
   10 TB over  10 Gb/s at 70% efficiency:    3.2 h
  100 TB over  10 Gb/s at 70% efficiency:   31.7 h
  100 TB over 100 Gb/s at 70% efficiency:    3.2 h
 1000 TB over 100 Gb/s at 70% efficiency:   31.7 h
BDP 10 Gb/s x 80 ms = 100 MB in flight; one stream with 4 MiB window maxes at 419 Mb/s
```

Note the bits vs bytes: network links are quoted in bits per second, storage in bytes (see [Networking: Units](networking.md#units-reading-spec-sheets-without-getting-fooled)). The second line is the reason single-stream transfers across continents are slow. TCP can only have one window of unacknowledged data in flight, so one stream's throughput is at most $\text{window} / \text{RTT}$. Filling a 10 Gb/s link at 80 ms RTT needs 100 MB in flight (the **bandwidth-delay product**), far more than a typical stream's window. This is the same Little's law as for disks. The fix is many parallel streams, plus tuned TCP buffers on both ends. For a petabyte, also price shipping physical drives (AWS Snowball and similar). Seriously.

### rsync

`rsync` copies files and, on re-runs, skips what's already there. By default it decides "already there" by **size and modification time**, not content. Run on this laptop (macOS's `openrsync`, rsync 2.6.9 compatible):

```bash
mkdir -p /tmp/rs/src /tmp/rs/dst && cd /tmp/rs
for i in 0 1 2; do head -c 50000000 /dev/urandom > src/shard_$i.bin; done
echo "== first sync";              rsync -a --stats src/ dst/ | grep -E 'Number of files transferred|Total transferred file size'
echo "== again, nothing changed";  rsync -a --stats src/ dst/ | grep -E 'Number of files transferred'
printf 'X' | dd of=src/shard_1.bin bs=1 seek=1000 conv=notrunc 2>/dev/null   # corrupt 1 byte in place
touch -r dst/shard_1.bin src/shard_1.bin                                       # ...keeping size and mtime equal
echo "== 1 byte changed, same size+mtime: default quick check"; rsync -a --stats src/ dst/ | grep 'Number of files transferred'
echo "== with --checksum";         rsync -a --checksum --stats src/ dst/ | grep 'Number of files transferred'
shasum -a 256 src/shard_1.bin dst/shard_1.bin | cut -c1-16
```

```text
== first sync
Number of files transferred: 3
Total transferred file size: 150000000 B
== again, nothing changed
Number of files transferred: 0
== 1 byte changed, same size+mtime: default quick check
Number of files transferred: 0
== with --checksum
Number of files transferred: 1
2f1a3bdd6f26f64a
2f1a3bdd6f26f64a
```

The default quick check missed a real content difference. Fine for repeated syncs of files you control, but useless as *verification*. `--checksum` reads both sides fully. Useful flags for big ML transfers (GNU rsync 3.x; see [Linux: rsync basics](linux-and-os.md#working-remotely-ssh-scp-rsync)):

```bash
# untested between real sites
rsync -a --partial --info=progress2 --no-inc-recursive src/ user@remote:/dst/   # resume partial files, overall progress
rsync -a -e "ssh -c aes128-gcm@openssh.com" ...       # cheaper cipher if ssh CPU is the bottleneck
rsync -a --whole-file ...                             # skip delta algorithm: faster on fast links for new data
```

A single rsync is a single stream and a single process. To parallelize, split the file list:

```bash
# untested: 16 parallel rsyncs, one per top-level shard directory
ls src/ | xargs -P 16 -I{} rsync -a --partial src/{} user@remote:/dst/   # -P 16: run up to 16 at once
```

### rclone, s5cmd and other parallel tools

For anything involving object stores, use a tool built for many concurrent requests. These commands are untested here; the flags are from each tool's docs.

```bash
# rclone: one tool, many backends (S3, GCS, Azure, SFTP, local); configure remotes with `rclone config`
rclone copy /local/ckpts/step_00020000 s3remote:my-bucket/ckpts/step_00020000 \
    --transfers 32 --checkers 64 --s3-chunk-size 64M --s3-upload-concurrency 8 --progress
rclone check /local/ckpts/step_00020000 s3remote:my-bucket/ckpts/step_00020000   # compare hashes/sizes
# rclone copy also works bucket-to-bucket across providers (streams through the machine running it)

# s5cmd: very fast S3 client, many parallel workers
s5cmd --numworkers 256 cp 's3://my-bucket/datasets/tok/*' /local/tok/

# aws cli: raise its concurrency
aws configure set default.s3.max_concurrent_requests 64
aws s3 sync /local/ckpts s3://my-bucket/ckpts
```

The parallelism knobs (`--transfers`, `--numworkers`, multipart concurrency) are the `outstanding requests` term of Little's law. Raise them until bandwidth stops growing or the provider starts returning `503 SlowDown`, then back off.

### Checksums

Every hop between storage systems is a chance for silent corruption: a flaky NIC, a buggy FUSE mount, a truncated upload that "succeeded". The Storage chapter's [Don't forget the checksums](../storage/README.md#dont-forget-the-checksums) section makes the case. Here is the cost of the hashing itself, on one core:

```python
# hashspeed.py - how fast can you checksum a checkpoint shard? (one core)
import hashlib, os, time, zlib
data = os.urandom(512 << 20)                        # 512 MiB in RAM, so we time the hash, not the disk
for name, fn in [("md5",     lambda b: hashlib.md5(b).digest()),
                 ("sha256",  lambda b: hashlib.sha256(b).digest()),
                 ("blake2b", lambda b: hashlib.blake2b(b).digest()),
                 ("crc32",   lambda b: zlib.crc32(b))]:
    t0 = time.perf_counter(); fn(data); dt = time.perf_counter() - t0
    print(f"{name:8s} {len(data) / dt / 1e9:5.2f} GB/s  -> 1 TB in {1e12 / (len(data) / dt) / 60:5.1f} min")
```

```text
md5       0.75 GB/s  -> 1 TB in  22.1 min
sha256    2.95 GB/s  -> 1 TB in   5.7 min
blake2b   1.32 GB/s  -> 1 TB in  12.6 min
crc32    38.87 GB/s  -> 1 TB in   0.4 min
```

On this Apple M4, SHA-256 beats MD5 because the CPU has SHA-256 instructions. On an x86 server without SHA extensions the ranking can flip, so measure on your hardware. CRC32 detects accidental corruption, not tampering, and that is the threat model for internal transfers. Practical recipe:

```bash
# at the source: one manifest per checkpoint/dataset, hashed in parallel (one process per file)
# (-print0 / -0 separate names with NUL bytes so spaces in names are safe; -P 16 = 16 processes at once)
find step_00020000 -type f -print0 | xargs -0 -P 16 sha256sum > step_00020000.sha256
# at the destination, after transfer:
sha256sum -c --quiet step_00020000.sha256 && echo "all shards verified"
```

Hashing 5.7 TB with 16 parallel processes at ~3 GB/s each takes about 2 minutes if the disks can feed 48 GB/s. Otherwise disk read speed is the limit. Remember that S3 multipart ETags are not file MD5s ([section 4](#multipart-uploads-and-range-reads)): store your own manifest next to the data.


## Checklist

```text
[ ] Know which tier each path is: local NVMe / NFS / parallel FS / object store, and its bandwidth from N nodes
[ ] Datasets packed into 100 MB-1 GB shards; no per-sample files on shared storage; file lists precomputed
[ ] Token dtype checked (uint16 only if vocab < 65536); decoded samples looked at; tokenizer hash in manifest
[ ] Loader state = (seed, epoch, samples consumed by *training*), checkpointed with the model
[ ] Resume tested once: step N+1 loss matches the uninterrupted run
[ ] Checkpoint size and write time computed; blocking time measured; interval from sqrt(2 C MTBF)
[ ] Async/staged checkpoints with backpressure; host RAM for staging budgeted
[ ] COMPLETE marker written last; resume picks latest complete; checksums in manifest
[ ] Never torch.load(weights_only=False) on files you didn't produce; distribute weights as safetensors
[ ] Benchmarks: fio --direct=1 with real queue depth, from many nodes; dd only with fdatasync/direct
[ ] Lifecycle rules: abort incomplete multipart uploads; delete old checkpoints by policy
```


## You can now read

- [Storage: File Systems and IO](../storage/README.md): the `fio-scan` benchmark, mmap measurements on Lustre, filesystem choices, and disk-cleanup scripts, with the mechanisms now clear.
- [Checkpoints](../training/checkpoints/README.md): scripts to convert fp32 checkpoints to bf16 and to shrink tensors with oversized storage.
- [Fault Tolerance](../training/fault-tolerance/README.md): frequent checkpoint saving, preemption signals, and planning for node failures.
- [Dealing with Datasets](../training/datasets.md): preprocessing on the main process and caching.
- [Training arithmetic](../training/arithmetic.md): the per-parameter memory and FLOPs numbers used above.
- [Decentralized Training](../training/decentralized-training.md): when the "storage" between sites is a slow WAN and checkpoint and transfer arithmetic dominates.
- [RL Infrastructure](../platform/rl-infrastructure.md): moving fresh weights from trainers to inference workers is the same safetensors / range-read / parallel-transfer problem on a loop.
- [Serving hands-on](../inference/serving-hands-on.md): cold-start weight loading is section 6 and 7 arithmetic with a latency budget.
- [Profiling and Optimization](../training/profiling-and-optimization.md): how to tell whether your step is waiting on the data loader.


## Further reading

- Remzi and Andrea Arpaci-Dusseau, [Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/): the "Persistence" part (I/O devices, hard drives, RAID, files and directories, FFS, journaling, flash SSDs, NFS). It is the best from-zero explanation of everything under section 1.
- [fio documentation](https://fio.readthedocs.io/en/latest/), especially the `ioengine`, `direct` and `iodepth` entries.
- [Lustre Operations Manual](https://doc.lustre.org/lustre_manual.xhtml): striping, `lfs setstripe`, PFL and DNE.
- AWS S3 docs: [consistency model](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel), [multipart upload limits](https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html), [performance guidelines](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html), [conditional requests](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-requests.html).
- [safetensors](https://github.com/huggingface/safetensors) README: the format specification and the security audit.
- PyTorch: [torch.distributed.checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html), and [`torch.load` and `weights_only`](https://pytorch.org/docs/stable/notes/serialization.html).
- [WebDataset](https://github.com/webdataset/webdataset), [MosaicML Streaming](https://docs.mosaicml.com/projects/streaming/), and [torchdata StatefulDataLoader](https://pytorch.org/data/beta/torchdata.stateful_dataloader.html).
- J. W. Young, "A first order approximation to the optimum checkpoint interval", CACM 1974; J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart dumps", FGCS 2006.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (arXiv 2407.21783): section 3.3.4 on reliability and the interruption statistics used above.
- Wang et al., "GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints", SOSP 2023 (citation from memory; check the venue).
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) `data/*/prepare.py` and [llm.c](https://github.com/karpathy/llm.c) `dev/data/`: the uint16 token-shard pattern in its minimal form.
- [rclone docs](https://rclone.org/docs/), [s5cmd](https://github.com/peak/s5cmd).
