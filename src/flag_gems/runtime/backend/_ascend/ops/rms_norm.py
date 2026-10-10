# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# kernel 体循环 over N, BLOCK_SIZE 可自由调小;
# 上限 12064(fwd)/6912(bwd) 是现网已验证的 UB 安全界, 候选绝不超过
_ASCEND_RMS_BLOCK_CANDIDATES = [8192, 4096, 2048, 1024]
_ASCEND_RMS_NW = 4  # 验证后端隐式默认后回填, 暂按 stock (4, 3) 假设
_ASCEND_RMS_NS = 3


def cfggen_ascend_rms_norm():
    # 静态占位列表(必须 >= 2 个, LibTuner.run 用 len(self.configs) > 1
    # 判断是否进搜索); 实际候选由 early_config_prune 按 N 重建,
    # 第一个恒为当前默认公式值, 保证 OFF 轮(取首候选)复现现网行为
    return [
        triton.Config(
            {"BLOCK_SIZE": bs}, num_warps=_ASCEND_RMS_NW, num_stages=_ASCEND_RMS_NS
        )
        for bs in [12064, 8192, 4096, 2048, 1024]
    ]


def make_rms_norm_config_prune(cap):
    def prune(configs, named_args, **kwargs):
        N = named_args["N"]
        default_bs = min(triton.next_power_of_2(N), cap)

        def mk(bs):
            return triton.Config(
                {"BLOCK_SIZE": bs}, num_warps=_ASCEND_RMS_NW, num_stages=_ASCEND_RMS_NS
            )

        new_configs = [mk(default_bs)]  # 第一个 = 当前默认
        for bs in _ASCEND_RMS_BLOCK_CANDIDATES:
            if bs < default_bs:  # 只追加更小者: 大 N 不退化, UB 绝对安全
                new_configs.append(mk(bs))
        return new_configs

    return prune


@libtuner(
    configs=cfggen_ascend_rms_norm(),
    key=["N"],
    prune_configs_by={"early_config_prune": make_rms_norm_config_prune(12064)},
    warmup=5,
    rep=10,
)
@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    var = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        var += tl.sum(x * x / N)

    rrms = 1 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask, other=0.0)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols * y_stride_c, y, mask=mask)

    tl.store(INV_RMS + pid, rrms)


@libtuner(
    configs=cfggen_ascend_rms_norm(),
    key=["N"],
    prune_configs_by={"early_config_prune": make_rms_norm_config_prune(6912)},
    warmup=5,
    rep=10,
)
@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid

    inv_rms = tl.load(INV_RMS).to(tl.float32)

    row_sum_stats = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        dy = dy * w
        normalized_buf = x * inv_rms
        row_sum_stats += tl.sum(normalized_buf * dy)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        dy = dy * w
        normalized_buf = x * inv_rms
        norm_val = normalized_buf / N
        dx = (dy - norm_val * row_sum_stats) * inv_rms
        tl.store(DX + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit
def rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE

    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start

    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    row_mask = (row_start + rows) < M
    col_mask = (col_start + cols) < N

    x = tl.load(
        X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(
        DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_ASCEND LAYERNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        # BLOCK_SIZE 由 libtuner 的 early_config_prune 按 N 重建候选
        # (第一个 = min(next_power_of_2(N), 12064), 即原默认公式)

        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)
        inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

        with torch_device_fn.device(x.device):
            rms_norm_kernel[M,](y, inv_rms, x, weight, N, 1, N, 1, N, eps)

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        logger.debug("GEMS_ASCEND LAYERNORM_BACKWARD")
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        # BLOCK_SIZE 由 libtuner 的 early_config_prune 按 N 重建候选
        # (第一个 = min(next_power_of_2(N), 6912), 即原默认公式)
        x = x.contiguous()
        weight = weight.contiguous()
        dx = torch.empty_like(x)

        with torch_device_fn.device(x.device):
            rms_norm_grad_dx_kernel[M,](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps
            )

        ROW_BLOCK_SIZE = 16
        COL_BLOCK_SIZE = 256
        row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
        col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

        partial_buffer = torch.empty(
            (row_block_num, N), dtype=torch.float32, device=x.device
        )

        with torch_device_fn.device(x.device):
            rms_norm_grad_dw_kernel[row_block_num, col_block_num](
                x,
                dy,
                inv_rms,
                partial_buffer,
                N,
                1,
                N,
                1,
                M,
                N,
                ROW_BLOCK_SIZE,
                COL_BLOCK_SIZE,
            )
            dw = torch.sum(partial_buffer, dim=0, dtype=x.dtype).reshape(-1)

        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
