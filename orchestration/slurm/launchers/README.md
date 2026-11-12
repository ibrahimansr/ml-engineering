# Single and Multi-Node Launchers with SLURM

The following are complete SLURM scripts that demonstrate how to integrate various launchers with software that uses `torch.distributed` (but should be easily adaptable to other distributed environments).

- [torchrun](torchrun-launcher.slurm) - to be used with [PyTorch distributed](https://github.com/pytorch/pytorch).
- [accelerate](accelerate-launcher.slurm) - to be used with [HF Accelerate](https://github.com/huggingface/accelerate) (Hugging Face's library that runs a plain PyTorch training loop on multiple GPUs/nodes with few code changes, and has its own `accelerate launch` launcher).
- [lightning](lightning-launcher.slurm) - to be used with [Lightning](https://lightning.ai/) (“PyTorch Lightning” and “Lightning Fabric”, frameworks that structure PyTorch training code and handle the distributed setup for you).
- [srun](srun-launcher.slurm) - to be used with the native SLURM launcher - here we have to manually preset env vars that `torch.distributed` expects (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, etc., derived from Slurm's own variables as shown in [Mapping to torch.distributed](../from-zero.md#mapping-to-torchdistributed)).

All of these scripts use [torch-distributed-gpu-test.py](../../../debug/torch-distributed-gpu-test.py) as the demo script, which you can copy here with just:
```bash
cp ../../../debug/torch-distributed-gpu-test.py .
```
assuming you cloned this repo. But you can replace it with anything else you need.
