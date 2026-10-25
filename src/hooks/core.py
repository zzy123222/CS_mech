from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import torch


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"No tensor found in module output of type {type(output)}")


def replace_first_tensor(output: Any, new_tensor: torch.Tensor):
    if torch.is_tensor(output):
        return new_tensor
    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item):
                items[i] = new_tensor
                return tuple(items)
    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if torch.is_tensor(item):
                items[i] = new_tensor
                return items
    raise TypeError(f"No tensor found in module output of type {type(output)}")


@contextmanager
def temporary_hooks(hooks: list[tuple[torch.nn.Module, Callable]]):
    handles = [module.register_forward_hook(fn) for module, fn in hooks]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class ActivationCache:

    def __init__(self, to_cpu: bool = True):
        self.data: dict[str, torch.Tensor] = {}
        self.handles = []
        self.to_cpu = to_cpu

    def add(self, module: torch.nn.Module, name: str):
        def hook(_module, _inputs, output):
            x = first_tensor(output).detach()
            if self.to_cpu:
                x = x.cpu()
            self.data[name] = x
        self.handles.append(module.register_forward_hook(hook))

    def clear(self):
        self.data.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
