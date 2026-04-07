"""
Distributed training with DDP and FSDP.

DDP (DistributedDataParallel):
==============================
- Every GPU holds a COMPLETE copy of the model (all parameters, gradients)
- Each GPU processes a different mini-batch independently
- After backward pass: gradient all-reduce across all GPUs (NCCL)
- Each GPU then applies the same gradient update -> all models stay in sync
- Memory cost: full model on every GPU
- Use when: model fits on a single GPU, you want speed from data parallelism

FSDP (FullyShardedDataParallel):
=================================
- Parameters, gradients, and optimizer states are SHARDED across GPUs
- Each GPU holds only 1/N of everything (N = number of GPUs)
- Before forward pass of each layer: All-Gather to reconstruct full layer params
- After backward pass of each layer: Reduce-Scatter to distribute gradients
- Memory cost: roughly 1/N of DDP memory per GPU
- Use when: model does NOT fit on a single GPU (large ViT, LLMs)

Adam optimizer memory:
======================
Adam stores: parameters (fp32) + first moment (fp32) + second moment (fp32)
= 3 x model_size in memory
With FSDP: each GPU only stores 3 x (model_size / N) for optimizer states
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


class DistributedTrainer:
    """
    Wraps DDP and FSDP for histopathology model training.

    Handles:
    - Process group initialization (NCCL for GPU, Gloo for CPU)
    - Model wrapping with DDP or FSDP
    - DistributedSampler for data loading
    - Gradient synchronization
    - Cleanup on exit

    Usage:
        trainer = DistributedTrainer(strategy='ddp', backend='nccl')
        trainer.setup(rank=0, world_size=4)
        model = trainer.wrap_model(model)
        sampler = trainer.get_sampler(dataset)
        # ... training loop ...
        trainer.cleanup()

    Args:
        strategy: 'ddp' for DistributedDataParallel,
                  'fsdp' for FullyShardedDataParallel
        backend: 'nccl' for GPU training (fastest),
                 'gloo' for CPU or fallback
    """

    def __init__(
        self,
        strategy: str = "ddp",
        backend: str = "nccl",
    ):
        if strategy not in ("ddp", "fsdp"):
            raise ValueError(f"strategy must be 'ddp' or 'fsdp', got '{strategy}'")
        if backend not in ("nccl", "gloo"):
            raise ValueError(f"backend must be 'nccl' or 'gloo', got '{backend}'")

        self.strategy = strategy
        self.backend = backend
        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.device: Optional[torch.device] = None

    def setup(self, rank: int, world_size: int):
        """
        Initialize the distributed process group.

        Each process calls this with its unique rank (0 to world_size-1).
        The process group enables collective communication operations
        (all-reduce, all-gather, etc.) between processes.

        Args:
            rank: unique identifier for this process (0-indexed)
            world_size: total number of processes (GPUs)
        """
        self.rank = rank
        self.world_size = world_size

        # Set environment variables for torch.distributed
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")

        # Initialize the process group
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
        )

        # Set the device for this rank
        if torch.cuda.is_available() and self.backend == "nccl":
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank == 0:
            print(f"Distributed training initialized: {world_size} processes, "
                  f"backend={self.backend}, strategy={self.strategy}")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap a model for distributed training.

        DDP: wraps with DistributedDataParallel. Each GPU holds a full copy
        of the model. Gradients are synchronized via all-reduce after backward.

        FSDP: wraps with FullyShardedDataParallel. Parameters, gradients,
        and optimizer states are sharded across GPUs. Each GPU holds only
        1/N of the total memory footprint.

        Args:
            model: the PyTorch model to wrap

        Returns:
            wrapped model (DDP or FSDP)
        """
        if self.rank is None:
            raise RuntimeError("Call setup() before wrap_model()")

        model = model.to(self.device)

        if self.strategy == "ddp":
            # DDP: each GPU holds a complete copy of the model
            # find_unused_parameters=False is faster but requires all params
            # to be used in every forward pass (true for ViT and ResNet)
            if self.backend == "nccl":
                model = DDP(
                    model,
                    device_ids=[self.rank],
                    output_device=self.rank,
                    find_unused_parameters=False,
                )
            else:
                # Gloo backend (CPU) doesn't use device_ids
                model = DDP(model, find_unused_parameters=False)

        elif self.strategy == "fsdp":
            # FSDP: shard parameters, gradients, and optimizer states
            # across all GPUs. Each GPU holds only 1/N of everything.
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import ShardingStrategy

                model = FSDP(
                    model,
                    # FULL_SHARD: shard params + gradients + optimizer states
                    # This gives maximum memory savings (1/N per GPU)
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    # Auto-wrap transformer blocks for efficient sharding
                    auto_wrap_policy=None,
                    device_id=self.rank if self.backend == "nccl" else None,
                )
            except ImportError:
                raise RuntimeError(
                    "FSDP requires PyTorch >= 1.12. "
                    "Please upgrade or use strategy='ddp'."
                )

        return model

    def get_sampler(self, dataset, shuffle: bool = True) -> DistributedSampler:
        """
        Create a DistributedSampler for the dataset.

        The sampler ensures each GPU processes a unique, non-overlapping
        subset of the data. Each GPU sees 1/N of the total dataset per epoch.

        Important: call sampler.set_epoch(epoch) at the start of each epoch
        to ensure proper shuffling across epochs.

        Args:
            dataset: the PyTorch Dataset
            shuffle: whether to shuffle data (True for training, False for val)

        Returns:
            DistributedSampler configured for this rank
        """
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

    def get_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader with DistributedSampler.

        Args:
            dataset: the PyTorch Dataset
            batch_size: per-GPU batch size (total = batch_size * world_size)
            shuffle: whether to shuffle data
            num_workers: number of data loading workers
            pin_memory: pin memory for faster GPU transfer

        Returns:
            DataLoader configured for distributed training
        """
        sampler = self.get_sampler(dataset, shuffle=shuffle)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=True,  # Ensures equal batch sizes across GPUs
        )

    @property
    def is_main_process(self) -> bool:
        """Return True if this is rank 0 (main process)."""
        return self.rank == 0

    def barrier(self):
        """
        Synchronize all processes.

        Blocks until all processes have reached this point. Use to ensure
        all processes are done with a step before proceeding (e.g., after
        saving a checkpoint on rank 0).
        """
        if dist.is_initialized():
            dist.barrier()

    def cleanup(self):
        """
        Clean up the distributed process group.

        Must be called when training is complete to release resources.
        """
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def compute_memory_footprint(model: nn.Module) -> dict:
        """
        Estimate memory footprint for DDP vs FSDP.

        DDP stores: full model + full gradients + full optimizer states
        FSDP stores: 1/N of (model + gradients + optimizer states)

        Adam optimizer memory = 3 x model_size (params + m1 + m2)

        Args:
            model: the PyTorch model

        Returns:
            dict with memory estimates in MB for different configurations
        """
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        param_mb = param_bytes / (1024 * 1024)

        # Gradients are same size as parameters
        grad_mb = param_mb

        # Adam optimizer: first moment (m1) + second moment (m2) + params copy
        # = 3x parameter size (all in fp32)
        optimizer_mb = 3 * param_mb

        total_ddp_mb = param_mb + grad_mb + optimizer_mb

        return {
            "parameters_mb": round(param_mb, 2),
            "gradients_mb": round(grad_mb, 2),
            "optimizer_states_mb": round(optimizer_mb, 2),
            "total_ddp_per_gpu_mb": round(total_ddp_mb, 2),
            "total_fsdp_per_gpu_2gpus_mb": round(total_ddp_mb / 2, 2),
            "total_fsdp_per_gpu_4gpus_mb": round(total_ddp_mb / 4, 2),
            "total_fsdp_per_gpu_8gpus_mb": round(total_ddp_mb / 8, 2),
        }
