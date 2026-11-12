# CPU Memory

This is a tiny chapter, since usually there are very few nuances one needs to know about CPU memory - which is a good thing!

Most of the ML workload compute happens on GPUs, but typically there should be at least as much CPU memory on each node as there is on the GPUs. So, for example, if you're on a H100 node with 8x 80GiB GPUs, you have 640GiB of GPU memory. Thus you want at least as much of CPU memory. Most recent high end cloud packages usually come with 1-2TiBs of CPU memory.

## What CPU memory is needed for in ML workloads

- Loading the model weights, unless they are loaded directly onto the GPUs - this is usually a transitory memory usage that goes back to zero once the model has been moved to GPUs.
- Saving the model weights. In some situations each GPU writes its own checkpoint directly to the disk, in other cases the model is recomposed on the CPU before it's written to disk - this too is a transitory memory usage.
- Possible parameter and optimizer state offloading when using frameworks like [DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/) (ZeRO-Offload keeps the parameters and/or optimizer state in CPU memory and copies them to the accelerator only when needed). In which case quite a lot of CPU memory might be needed.
- Activations calculated in the `forward` pass, and which need to be available for the `backward` path can also be offloaded to CPU, rather than discarded and then recomputed during the backward pass to save the unnecessary overhead
- `DataLoader` is usually one of the main users of CPU memory and at times it may consume very large amounts of memory. Typically there are at least 2x 8 DL workers running on each node (2 `DataLoader` workers for each of the 8 accelerators), so you need enough memory to support at least 16 processes each holding some data. For example, in the case of streaming data from the cloud, if the data shards are large, these processes could easily eat up hundreds of GBs of CPU memory.
- The software itself and its dependent libraries uses a bit of CPU memory, but this amount is usually negligible.

## Things to know

- If the `DataLoader` uses HF `datasets` in `mmap` mode (the dataset file is memory-mapped, so its pages are read into memory from disk on first access, see [Page faults and lazy allocation](../../foundations/linux-and-os.md#page-faults-and-lazy-allocation)) the Resident memory usage (RSS, see [RSS vs VSZ (and PSS)](../../foundations/linux-and-os.md#rss-vs-vsz-and-pss)) may appear to be using a huge amount of CPU memory as it'll try to map out the whole datasets to the memory. Except this is misleading, since if the memory is needed elsewhere the OS will page out any unneeded mmap'ed pages back to the system (pages backed by a file can simply be dropped and re-read from disk later, so they count in RSS without being memory the process holds onto). You can read more about it [here](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/). This awareness, of course, applies to any dataset using `mmap`, I was using HF `datasets` as an example since it's very widely used.
