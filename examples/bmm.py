"""
Batch Matrix Multiplication Example
===============================

This example demonstrates how to implement a batch matrix multiplication kernel using Helion.
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
# Batch Matrix Multiplication Kernel
# -------------------------------
# static_shapes=True gives a performance boost for matmuls
# @helion.kernel(static_shapes=True)
# Bench 1 & 2
# 2D acc
@helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[1, 512, 64, 32], indexing='pointer', l2_groupings=[32], loop_orders=[[2, 1, 0]], num_stages=3, num_warps=16, pid_type='flat', range_flattens=[None, False, None], range_multi_buffers=[None, False, False], range_num_stages=[0, 3, 3], range_unroll_factors=[0, 3, 0], range_warp_specializes=[]))
# 3D acc
# @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[1, 512, 64, 32], indexing='pointer', l2_groupings=[32], loop_orders=[[2, 1, 0]], num_stages=3, num_warps=16, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, False], range_num_stages=[0, 3], range_unroll_factors=[0, 0], range_warp_specializes=[]))
# Bench 3
# 2D acc
# @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[1, 256, 128, 32], indexing='pointer', l2_groupings=[4], loop_orders=[[2, 1, 0]], num_stages=3, num_warps=16, pid_type='flat', range_flattens=[None, None, None], range_multi_buffers=[None, True, None], range_num_stages=[0, 0, 2], range_unroll_factors=[0, 3, 0], range_warp_specializes=[]))
# 3D acc
# @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[1, 256, 128, 32], indexing='pointer', l2_groupings=[4], loop_orders=[[2, 1, 0]], num_stages=3, num_warps=16, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 2], range_unroll_factors=[0, 0], range_warp_specializes=[]))
def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batch matrix multiplication.

    Args:
        A: Input tensor of shape [B, M, K]
        B: Input tensor of shape [B, K, N]

    Returns:
        Output tensor of shape [B, M, N] containing the result of batch matrix multiplication
    """
    # # Use below for 3D accumulation
    # # A: [B, M, K], B: [B, K, N], Out: [B, M, N]   # dense bmm
    # b, m, k = A.size()
    # b, k, n = B.size()
    # out = torch.empty(
    #     [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    # )
    # for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
    #     acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
    #     for tile_k in hl.tile(k):
    #         acc = torch.baddbmm(
    #             acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
    #         )
    #     out[tile_b, tile_m, tile_n] = acc

    # Use below for 2D accumulation
    # Batch loop - compute GEMM
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        for i in range(tile_b.begin, tile_b.end):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, A[i, tile_m, tile_k], B[i, tile_k, tile_n])
            out[i, tile_m, tile_n] = acc

    return out


# %%
# Verification Function
# -------------------
def check(b: int, m: int, k: int, n: int) -> None:
    """
    Verify the bmm kernel implementation against PyTorch's native bmm function.

    Args:
        b: Batch size
        m: First dimension of the first matrix
        k: Second dimension of the first matrix / First dimension of the second matrix
        n: Second dimension of the second matrix
    """
    x = torch.randn([b, m, k], device="xpu", dtype=torch.float16)
    y = torch.randn([b, k, n], device="xpu", dtype=torch.float16)
    run_example(bmm, torch.bmm, (x, y))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the bmm kernel verification with specific parameters.
    Tests with batch size 16, and matrices of dimensions 512x768 and 768x1024.
    Ensures torch version is at least 2.8 for 16-bit tensor support in baddbmm.
    """
    # torch.baddbmm support for 16-bit tensors requires torch 2.8+
    assert torch.__version__.split(".")[:2] >= ["2", "8"], "Requires torch 2.8+"
    # check(16, 512, 768, 1024)
    # check(4, 1024, 1024, 1024)
    check(8, 1024, 1024, 1024)
    # check(4, 512, 8192, 8192)


if __name__ == "__main__":
    main()
