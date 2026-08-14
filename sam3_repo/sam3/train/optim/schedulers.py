# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math


class InverseSquareRootParamScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        cooldown_steps: int,
        timescale: int,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.timescale = timescale

    def __call__(self, step: int, where: float):
        lr = self.base_lr

        if where > 0:
            total_steps = step / where
            progress = (step - self.warmup_steps) / float(
                total_steps - self.warmup_steps
            )
            progress = max(min(progress, 1), 0)
        else:
            progress = 0
            total_steps = 1

        shift = self.timescale - self.warmup_steps
        if self.warmup_steps < step:
            lr = lr / math.sqrt((step + shift) / self.timescale)

        if self.warmup_steps:
            lr = lr * min(1.0, step / self.warmup_steps)
        if self.cooldown_steps:
            lr = lr * min(1.0, (total_steps - step) / self.cooldown_steps)

        return lr


class CosineWithWarmupScheduler:
    """cosine decay with linear warmup, compatible with SAM3 optimizer API.

    Args:
        base_lr: peak learning rate after warmup
        warmup_epochs: number of warmup epochs
        total_epochs: total training epochs (used to compute warmup progress fraction)
        min_lr: minimum lr after cosine decay (relative to base_lr)
    """

    def __init__(
        self,
        base_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
    ):
        self.base_lr = base_lr
        self.warmup_frac = warmup_epochs / max(total_epochs, 1)
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def __call__(self, step: int, where: float):
        if where <= self.warmup_frac:
            # linear warmup
            lr = self.base_lr * (where / self.warmup_frac)
        else:
            # cosine decay
            progress = (where - self.warmup_frac) / (1.0 - self.warmup_frac)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        return lr
