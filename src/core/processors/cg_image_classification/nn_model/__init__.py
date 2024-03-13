from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.logger import Logger
from .bagnet import BagNet, bagnet9, bagnet17, bagnet33


def get_batch_size(model: nn.Module, device: torch.device, input_shape: Tuple[int, int, int], output_shape: Tuple[int],
                   dataset_size: int, max_batch_size: int = None, num_iterations: int = 5) -> int:
    """
    https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1
    """
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 2
    while True:
        Logger.info(f"Trying batch size {batch_size}")
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    return batch_size
