"""
Element-wise Addition Example
===========================

This example demonstrates how to implement an element-wise addition kernel using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# Addition Kernel
# --------------
# @helion.kernel()
# Tuner's best config:
@helion.kernel(config=helion.Config(block_sizes=[16, 128], flatten_loops=[True], indexing='block_ptr', l2_groupings=[2], loop_orders=[[0, 1]], num_stages=2, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[]))
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors element-wise with broadcasting support.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        A new tensor containing the element-wise sum of x and y
    """
    # match pytorch broadcasting rules
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        # match type promotion of torch.add
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    # tile will be a tuple of blocks
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


# %%
# Verification Function
# -------------------
def check(m: int, n: int) -> None:
    """
    Verify the add kernel implementation against PyTorch's native add function.

    Args:
        m: First dimension of the test tensors
        n: Second dimension of the test tensors
    """
    x = torch.randn([m, n], device="xpu", dtype=torch.float16)
    y = torch.randn([m, n], device="xpu", dtype=torch.float16)
    run_example(add, torch.add, (x, y))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the add kernel verification with 1024x1024 tensors.
    """
    check(1024, 1024)


if __name__ == "__main__":
    main()
