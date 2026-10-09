import csv
import glob
import os
import time
from typing import Any, Dict, Tuple

import pytest
import torch
import triton
import triton.language as tl

current_device = "npu"

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1}),
        triton.Config({'BLOCK_S': 2}),
        triton.Config({'BLOCK_S': 4}),
        triton.Config({'BLOCK_S': 8}),
        triton.Config({'BLOCK_S': 16}),
        triton.Config({'BLOCK_S': 32}),
    ],
    key=['d'],
)
@triton.jit
def _rmsnorm_fwd_fused(
    input_ptr,  # pointer to the input
    output_ptr,  # pointer to the output
    weight_ptr,  # pointer to the weights
    rrms_ptr,  # pointer to the rrms
    stride,  # how much to increase the pointer when moving by 1 row
    s,  # number of rows in input
    d,  # number of columns in input
    eps,  # epsilon to avoid division by zero
    BLOCK_S: tl.constexpr,  # number of rows per program (autotuned)
    D_ROUNDED: tl.constexpr,
    NUM_CORES: tl.constexpr,  # number of physical cores for grid mapping
):
    # Step 1: 获取 program id，计算总任务数（行块数）
    pid = tl.program_id(0)
    num_tasks = tl.cdiv(s, BLOCK_S)

    # Step 2: 列方向循环不变量在循环外计算一次
    # col_offsets / col_mask / w 只依赖列方向，所有行块共用，
    # 外提避免每轮重复计算与访存
    col_offsets = tl.arange(0, D_ROUNDED)  # (D_ROUNDED,)
    col_mask = col_offsets < d
    w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.).to(tl.float32)

    # Step 3: 物理分核循环 —— 每个 program 通过 stride 循环处理多个行块
    # for 循环使得上一轮的 compute（Vector）可与下一轮的 load（MTE2）重叠执行，
    # 实现软件流水线，解决 Vector 与 MTE2 串行问题
    for row_block_idx in range(pid, num_tasks, NUM_CORES):
        # 行方向 offset 每次迭代独立计算（基地址 + 偏移量，无 RAW 依赖）
        row_offsets = row_block_idx * BLOCK_S + tl.arange(0, BLOCK_S)  # (BLOCK_S,)
        row_mask = row_offsets < s
        mask = row_mask[:, None] & col_mask[None, :]  # (BLOCK_S, D_ROUNDED)

        # Load input block（每轮唯一的 MTE2 load，w 已在循环外加载）
        x = tl.load(
            input_ptr + row_offsets[:, None] * stride + col_offsets[None, :],
            mask=mask, other=0.
        ).to(tl.float32)

        # Compute rms along the column dimension (axis=1)
        x_square = x * x
        ms = tl.sum(x_square, axis=1) / d + eps  # (BLOCK_S,)
        rrms = 1 / tl.sqrt(ms)  # (BLOCK_S,)

        # Write rrms
        tl.store(rrms_ptr + row_offsets, rrms, mask=row_mask)

        # Compute output（w 已在循环外提前加载，此处直接复用）
        x_norm = x * rrms[:, None]
        y = x_norm * w[None, :]

        # Store output
        tl.store(
            output_ptr + row_offsets[:, None] * stride + col_offsets[None, :],
            y, mask=mask
        )

# fy
@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.autotune(
    configs=[
        triton.Config({'SUB_BLOCK_S': 1}),
        triton.Config({'SUB_BLOCK_S': 2}),
        triton.Config({'SUB_BLOCK_S': 4}),
        triton.Config({'SUB_BLOCK_S': 8}),
        triton.Config({'SUB_BLOCK_S': 16}),
    ],
    key=['d'],
)
@triton.jit
def _rmsnorm_bwd_dxdw_fused(
    dy_ptr,  # pointer to the output grads
    dx_ptr,  # pointer to the input grads
    dw_ptr,  # pointer to the partial weights grads
    input_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    rrms_ptr,  # pointer to the rrms
    stride,  # how much to increase the pointer when moving by 1 row
    s,  # number of rows in input_ptr
    d,  # number of columns in input_ptr, hidden dimension
    BLOCK_S: tl.constexpr,  # number of rows per program
    SUB_BLOCK_S: tl.constexpr,  # number of rows per inner loop iteration (autotuned)
    D_ROUNDED: tl.constexpr,
):
    # Map the program id to the row block of all the data should compute
    row_block_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, D_ROUNDED)  # full d for each row
    col_mask = col_offsets < d
    w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.).to(tl.float32)

    partial_dw = tl.zeros((D_ROUNDED,), dtype=tl.float32)
    start = row_block_idx * BLOCK_S
    end = min(start + BLOCK_S, s)

    for sub_start in range(start, end, SUB_BLOCK_S):
        sub_end = min(sub_start + SUB_BLOCK_S, end)
        sub_row_offsets = sub_start + tl.arange(0, SUB_BLOCK_S)  # (SUB_BLOCK_S,)
        sub_row_mask = sub_row_offsets < sub_end

        mask = sub_row_mask[:, None] & col_mask[None, :]  # (SUB_BLOCK_S, D_ROUNDED)

        # Load data for SUB_BLOCK_S rows
        rrms = tl.load(rrms_ptr + sub_row_offsets, mask=sub_row_mask, other=0.)  # (SUB_BLOCK_S,)
        dy = tl.load(
            dy_ptr + sub_row_offsets[:, None] * stride + col_offsets[None, :],
            mask=mask, other=0.
        ).to(tl.float32)
        x = tl.load(
            input_ptr + sub_row_offsets[:, None] * stride + col_offsets[None, :],
            mask=mask, other=0.
        ).to(tl.float32)

        # Compute dx and accumulate dw
        x_norm = x * rrms[:, None]  # (SUB_BLOCK_S, D_ROUNDED)
        partial_dw += tl.sum(dy * x_norm, axis=0)  # accumulate over SUB_BLOCK_S rows

        dxnorm = w[None, :] * dy  # (SUB_BLOCK_S, D_ROUNDED)
        c1 = tl.sum(x_norm * dxnorm, axis=1) / d  # (SUB_BLOCK_S,)
        dx = (dxnorm - x_norm * c1[:, None]) * rrms[:, None]  # (SUB_BLOCK_S, D_ROUNDED)

        tl.store(
            dx_ptr + sub_row_offsets[:, None] * stride + col_offsets[None, :],
            dx, mask=mask
        )

    tl.store(dw_ptr + row_block_idx * d + col_offsets, partial_dw, mask=col_mask)


class TritonRMSNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.nn.Parameter, eps: float):
        assert len(x.shape) == 2 and x.shape[1] == w.shape[0]
        x = x.contiguous()
        y = torch.empty_like(x)
        s, d = x.shape
        rrms = torch.empty(s, dtype=torch.float32, device=x.device)

        # 纯 vector 算子（无 tl.dot/tl.matmul），使用 num_vectorcore 进行物理分核
        # grid 使用物理核数，kernel 内部 for 循环处理多个行块，
        # 实现 Vector 运算与 MTE2 访存的软件流水线并行
        device_properties = triton.runtime.driver.active.utils.get_device_properties(x.device)
        num_cores = device_properties.get("num_vectorcore", -1)

        # enqueue kernel
        _rmsnorm_fwd_fused[(num_cores,)](
            x, y, w, rrms, x.stride(0), s, d, eps,
            D_ROUNDED=d,
            NUM_CORES=num_cores,
        )

        ctx.save_for_backward(x, w, rrms)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w, rrms = ctx.saved_tensors
        s, d = x.shape
        dy = dy.contiguous()

        dx = torch.empty_like(dy)
        # heuristic to parallel the s and determine the number_group_s
        # num_group_s = get_device_properties(x.device)[1]
        num_group_s = torch.npu.get_device_properties(x.device).vector_core_num
        BLOCK_S = triton.cdiv(s, num_group_s)
        _dw = torch.empty((num_group_s, d), dtype=torch.float32, device=w.device)
        _rmsnorm_bwd_dxdw_fused[(num_group_s,)](
            dy, dx, _dw, x, w, rrms, x.stride(0), s, d,
            BLOCK_S=BLOCK_S, D_ROUNDED=d, multibuffer=False,
        )

        dw = _dw.sum(dim=0).to(w.dtype)
        return dx, dw, None


# ---------------------------------------------------------------------------
# SiTU-GLU v2 kernels + helpers + autograd wrapper
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1}),
        triton.Config({'BLOCK_M': 2}),
        triton.Config({'BLOCK_M': 4}),
        triton.Config({'BLOCK_M': 8}),
        triton.Config({'BLOCK_M': 16}),
        triton.Config({'BLOCK_M': 32}),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_situgluv2_fwd_triton_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    beta,
    inv_beta,
    linear_beta,
    inv_linear_beta,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Step 1: N 方向一次性处理完整列，offset 和 mask 在循环外计算一次
    pid = tl.program_id(0)
    num_m_tasks = tl.cdiv(M, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)
    col_mask = offs_n < N

    # Step 2: M 方向物理分核循环，每个 program 处理连续 BLOCK_M 行
    for m_task_id in range(pid, num_m_tasks, NUM_CORES):
        offs_m = m_task_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = (offs_m < M)[:, None] & col_mask[None, :]

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        b_ptrs = b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
        o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

        a = tl.load(a_ptrs, mask=mask, other=0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask, other=0).to(tl.float32)

        sig_a = tl.sigmoid(a)
        z_a = a * inv_beta
        tanh_a = _tanh(z_a)
        f = beta * tanh_a * sig_a

        z_b = b * inv_linear_beta
        tanh_b = _tanh(z_b)
        r = linear_beta * tanh_b

        out = f * r
        tl.store(o_ptrs, out, mask=mask)


def fused_situgluv2_forward(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    """SiTU-GLU v2 前向计算（M 方向切 BLOCK_M 行，一次处理完整 N，物理分核）。

        left = beta * tanh(a/beta) * sigmoid(a)
        right = linear_beta * tanh(b/linear_beta)
        output = left * right

    Args:
        input_a: 输入张量 a，形状 (M, N)。
        input_b: 输入张量 b，形状 (M, N)。
        beta: SiTU 左路缩放系数，必须 > 0。
        linear_beta: 右路缩放系数，必须 > 0。

    Returns:
        输出张量，形状与 input_a 相同。

    Raises:
        AssertionError: 当输入形状不匹配、维度不为 2D、或 beta/linear_beta <= 0 时。
    """
    assert input_a.shape == input_b.shape
    assert beta > 0 and linear_beta > 0
    assert input_a.ndim == 2, f"Expected 2D tensor, got {input_a.ndim}D"

    M, N = input_a.shape
    out = torch.empty_like(input_a)
    inv_beta = float(1.0 / beta)
    inv_linear_beta = float(1.0 / linear_beta)

    # 纯 vector 算子（无 tl.dot/tl.matmul），使用 num_vectorcore 进行物理分核
    device_properties = triton.runtime.driver.active.utils.get_device_properties(input_a.device)
    num_cores = device_properties.get("num_vectorcore", -1)

    _fused_situgluv2_fwd_triton_kernel[(num_cores,)](
        input_a, input_b, out,
        M, N,
        input_a.stride(0), input_a.stride(1),
        input_b.stride(0), input_b.stride(1),
        out.stride(0), out.stride(1),
        float(beta), inv_beta,
        float(linear_beta), inv_linear_beta,
        NUM_CORES=num_cores,
        BLOCK_N=N,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1}),
        triton.Config({'BLOCK_M': 2}),
        triton.Config({'BLOCK_M': 4}),
        triton.Config({'BLOCK_M': 8}),
        triton.Config({'BLOCK_M': 16}),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_situgluv2_bwd_triton_kernel(
    a_ptr, b_ptr, dout_ptr,
    da_ptr, db_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_dom, stride_don,
    stride_dam, stride_dan,
    stride_dbm, stride_dbn,
    beta, inv_beta,
    linear_beta, inv_linear_beta,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Step 1: N 方向一次性处理完整列，offset 和 mask 在循环外计算一次
    pid = tl.program_id(0)
    num_m_tasks = tl.cdiv(M, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)
    col_mask = offs_n < N

    # Step 2: M 方向物理分核循环，每个 program 处理连续 BLOCK_M 行
    for m_task_id in range(pid, num_m_tasks, NUM_CORES):
        offs_m = m_task_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = (offs_m < M)[:, None] & col_mask[None, :]

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        b_ptrs = b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
        do_ptrs = dout_ptr + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_don
        da_ptrs = da_ptr + offs_m[:, None] * stride_dam + offs_n[None, :] * stride_dan
        db_ptrs = db_ptr + offs_m[:, None] * stride_dbm + offs_n[None, :] * stride_dbn

        a = tl.load(a_ptrs, mask=mask, other=0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask, other=0).to(tl.float32)
        dout = tl.load(do_ptrs, mask=mask, other=0).to(tl.float32)

        s = tl.sigmoid(a)
        s_neg = tl.sigmoid(-a)
        z_a = a * inv_beta
        p_a = tl.sigmoid(2.0 * z_a)
        t_a = 2.0 * p_a - 1.0
        tanh_prime_a = 4.0 * p_a * (1.0 - p_a)
        f = beta * t_a * s
        dfdx = beta * t_a * s * s_neg + s * tanh_prime_a

        z_b = b * inv_linear_beta
        p_b = tl.sigmoid(2.0 * z_b)
        t_b = 2.0 * p_b - 1.0
        tanh_prime_b = 4.0 * p_b * (1.0 - p_b)
        r = linear_beta * t_b

        da = dout * r * dfdx
        db = dout * f * tanh_prime_b

        tl.store(da_ptrs, da, mask=mask)
        tl.store(db_ptrs, db, mask=mask)


def fused_situgluv2_backward(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    doutput: torch.Tensor,
    beta: float,
    linear_beta: float,
):
    """SiTU-GLU v2 反向计算（M 方向切 BLOCK_M 行，一次处理完整 N，物理分核）。

    Args:
        input_a: 前向输入张量 a，形状 (M, N)。
        input_b: 前向输入张量 b，形状 (M, N)。
        doutput: 上游梯度，形状 (M, N)。
        beta: SiTU 左路缩放系数，必须 > 0。
        linear_beta: 右路缩放系数，必须 > 0。

    Returns:
        (dA, dB) 元组：a 与 b 的梯度，形状与输入相同。

    Raises:
        AssertionError: 当输入形状不匹配、维度不为 2D、或 beta/linear_beta <= 0 时。
    """
    assert input_a.shape == input_b.shape == doutput.shape
    assert beta > 0 and linear_beta > 0
    assert input_a.ndim == 2, f"Expected 2D tensor, got {input_a.ndim}D"

    M, N = input_a.shape
    dA = torch.empty_like(input_a)
    dB = torch.empty_like(input_b)
    inv_beta = float(1.0 / beta)
    inv_linear_beta = float(1.0 / linear_beta)

    # 纯 vector 算子（无 tl.dot/tl.matmul），使用 num_vectorcore 进行物理分核
    device_properties = triton.runtime.driver.active.utils.get_device_properties(input_a.device)
    num_cores = device_properties.get("num_vectorcore", -1)

    _fused_situgluv2_bwd_triton_kernel[(num_cores,)](
        input_a, input_b, doutput,
        dA, dB,
        M, N,
        input_a.stride(0), input_a.stride(1),
        input_b.stride(0), input_b.stride(1),
        doutput.stride(0), doutput.stride(1),
        dA.stride(0), dA.stride(1),
        dB.stride(0), dB.stride(1),
        float(beta), inv_beta,
        float(linear_beta), inv_linear_beta,
        NUM_CORES=num_cores,
        BLOCK_N=N,
    )
    return dA, dB


class SiTUGLUv2(torch.autograd.Function):
    """Autograd wrapper for fused SiTU-GLU v2 using Triton kernel."""

    @staticmethod
    def forward(ctx, a, b, beta, linear_beta):
        orig_shape = a.shape
        ctx.orig_shape = orig_shape
        ctx.beta = beta
        ctx.linear_beta = linear_beta

        if a.ndim > 2:
            a = a.view(-1, a.shape[-1])
            b = b.view(-1, b.shape[-1])

        ctx.save_for_backward(a, b)

        out = fused_situgluv2_forward(a, b, beta, linear_beta)

        if len(orig_shape) > 2:
            out = out.view(orig_shape)

        return out

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        a, b = ctx.saved_tensors
        beta = ctx.beta
        linear_beta = ctx.linear_beta
        orig_shape = ctx.orig_shape

        if len(orig_shape) > 2:
            dout = dout.view(-1, dout.shape[-1])

        dinputa, dinputb = fused_situgluv2_backward(a, b, dout, beta, linear_beta)

        if len(orig_shape) > 2:
            dinputa = dinputa.view(orig_shape)
            dinputb = dinputb.view(orig_shape)

        return dinputa, dinputb, None, None


# ---------------------------------------------------------------------------
# Native references
# ---------------------------------------------------------------------------

def rmsnorm_native(x, w, eps):
    """Native RMSNorm forward reference."""
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    y = x_float * rrms * w.float()
    return y.to(x.dtype)


def rmsnorm_native_diff(x, w, eps):
    """Differentiable RMSNorm reference for backward test."""
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    y = x_float * rrms * w.float()
    return y


def situgluv2_native(a, b, beta, linear_beta):
    """Native SiTU-GLU v2 forward reference."""
    a_float = a.float()
    b_float = b.float()
    sig_a = torch.sigmoid(a_float)
    tanh_a = torch.tanh(a_float / beta)
    left = beta * tanh_a * sig_a
    tanh_b = torch.tanh(b_float / linear_beta)
    right = linear_beta * tanh_b
    return (left * right).to(a.dtype)


def situgluv2_native_diff(a, b, beta, linear_beta):
    """Differentiable SiTU-GLU v2 reference for backward test."""
    a_float = a.float()
    b_float = b.float()
    sig_a = torch.sigmoid(a_float)
    tanh_a = torch.tanh(a_float / beta)
    left = beta * tanh_a * sig_a
    tanh_b = torch.tanh(b_float / linear_beta)
    right = linear_beta * tanh_b
    return left * right


# ---------------------------------------------------------------------------
# Accuracy tests — RMSNorm
# ---------------------------------------------------------------------------

RMSNORM_DTYPE_CASES = [
    pytest.param(torch.float32, 5e-5, 5e-5, id="fp32"),
    pytest.param(torch.bfloat16, 1e-2, 1e-2, id="bf16"),
]

RMSNORM_SHAPE_CASES = [
    pytest.param(4096, 7168, id="S4096-D7168"),
    pytest.param(1, 7168, id="S1-D7168"),
    pytest.param(128, 4096, id="S128-D4096"),
]


@pytest.mark.parametrize("dtype,atol,rtol", RMSNORM_DTYPE_CASES)
@pytest.mark.parametrize("S,D", RMSNORM_SHAPE_CASES)
def test_rmsnorm_forward_acc(S, D, dtype, atol, rtol):
    torch.manual_seed(42)
    x = torch.randn((S, D), device=current_device, dtype=dtype)
    w = torch.randn((D,), device=current_device, dtype=torch.float32)
    eps = 1e-6

    expected = rmsnorm_native(x, w, eps)
    actual = TritonRMSNormFunc.apply(x, w, eps)

    assert actual.shape == expected.shape == (S, D)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,atol,rtol", RMSNORM_DTYPE_CASES)
@pytest.mark.parametrize("S,D", RMSNORM_SHAPE_CASES)
def test_rmsnorm_backward_acc(S, D, dtype, atol, rtol):
    torch.manual_seed(42)
    x = torch.randn((S, D), device=current_device, dtype=dtype)
    w = torch.randn((D,), device=current_device, dtype=torch.float32)
    eps = 1e-6
    dy = torch.randn((S, D), device=current_device, dtype=dtype)

    # Native reference
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    y_ref = rmsnorm_native_diff(x_ref, w_ref, eps)
    y_ref.backward(dy)

    # Triton
    x_tri = x.detach().clone().requires_grad_(True)
    w_tri = w.detach().clone().requires_grad_(True)
    y_tri = TritonRMSNormFunc.apply(x_tri, w_tri, eps)
    y_tri.backward(dy)

    torch.testing.assert_close(x_tri.grad, x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(w_tri.grad, w_ref.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Accuracy tests — SiTU-GLU v2
# ---------------------------------------------------------------------------

SITU_DTYPE_CASES = [
    pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    pytest.param(torch.bfloat16, 1e-2, 1e-2, id="bf16"),
]

SITU_SHAPE_CASES = [
    pytest.param(4096, 7168, id="M4096-N7168"),
    pytest.param(1, 7168, id="M1-N7168"),
    pytest.param(128, 4096, id="M128-N4096"),
]


@pytest.mark.parametrize("dtype,atol,rtol", SITU_DTYPE_CASES)
@pytest.mark.parametrize("M,N", SITU_SHAPE_CASES)
def test_situgluv2_forward_acc(M, N, dtype, atol, rtol):
    torch.manual_seed(42)
    a = torch.randn((M, N), device=current_device, dtype=dtype)
    b = torch.randn((M, N), device=current_device, dtype=dtype)
    beta = 0.5
    linear_beta = 0.5

    expected = situgluv2_native(a, b, beta, linear_beta)
    actual = SiTUGLUv2.apply(a, b, beta, linear_beta)

    assert actual.shape == expected.shape == (M, N)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,atol,rtol", SITU_DTYPE_CASES)
@pytest.mark.parametrize("M,N", SITU_SHAPE_CASES)
def test_situgluv2_backward_acc(M, N, dtype, atol, rtol):
    torch.manual_seed(42)
    a = torch.randn((M, N), device=current_device, dtype=dtype)
    b = torch.randn((M, N), device=current_device, dtype=dtype)
    beta = 0.5
    linear_beta = 0.5
    dout = torch.randn((M, N), device=current_device, dtype=dtype)

    # Native reference
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    y_ref = situgluv2_native_diff(a_ref, b_ref, beta, linear_beta)
    y_ref.backward(dout)

    # Triton
    a_tri = a.detach().clone().requires_grad_(True)
    b_tri = b.detach().clone().requires_grad_(True)
    y_tri = SiTUGLUv2.apply(a_tri, b_tri, beta, linear_beta)
    y_tri.backward(dout)

    torch.testing.assert_close(a_tri.grad, a_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(b_tri.grad, b_ref.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Performance tests — shared helpers
# ---------------------------------------------------------------------------

PERF_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rmsnorm_situ_performance.csv",
)

PERF_DTYPE_CASES = [
    pytest.param(torch.bfloat16, id="bf16"),
]


def dtype_name(dtype):
    name = str(dtype)
    return name[6:] if name.startswith("torch.") else name


def _matches_kernel(row, kernel_name):
    haystack = " ".join(str(value) for value in row.values()).lower()
    aliases = (kernel_name.lower(), kernel_name.lstrip("_").lower())
    return any(alias in haystack for alias in aliases)


def device_perf_npu(
    executor,
    profiling_dir,
    kernel_name,
    active=5,
    warmup=5,
):
    """Return one target Triton kernel's average NPU time."""
    import torch_npu

    os.makedirs(profiling_dir, exist_ok=True)

    executor()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=warmup,
            active=active,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            profiling_dir
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(warmup + active):
            torch.npu.synchronize()
            executor()
            torch.npu.synchronize()
            prof.step()

    summary_files = glob.glob(
        os.path.join(
            profiling_dir,
            "**",
            "mindstudio_profiler_output",
            "op_summary*.csv",
        ),
        recursive=True,
    )
    if summary_files:
        total_time_us = 0.0
        for summary_file in summary_files:
            with open(summary_file, newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                if "Task Duration(us)" not in (reader.fieldnames or []):
                    raise RuntimeError(
                        f"Task Duration(us) is missing from {summary_file}"
                    )
                rows = list(reader)
                matched_rows = [
                    row for row in rows if _matches_kernel(row, kernel_name)
                ]
                if not matched_rows:
                    matched_rows = [
                        row
                        for row in rows
                        if "triton" in " ".join(
                            str(value) for value in row.values()
                        ).lower()
                    ]
                total_time_us += sum(
                    float(row["Task Duration(us)"])
                    for row in matched_rows
                    if row.get("Task Duration(us)")
                )
        if total_time_us > 0:
            return total_time_us / active, profiling_dir

    statistic_files = glob.glob(
        os.path.join(profiling_dir, "**", "op_statistic.csv"),
        recursive=True,
    )
    if statistic_files:
        total_time_us = 0.0
        for statistic_file in statistic_files:
            with open(statistic_file, newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                if "Total Time(us)" not in (reader.fieldnames or []):
                    raise RuntimeError(
                        f"Total Time(us) is missing from {statistic_file}"
                    )
                rows = list(reader)
                matched_rows = [
                    row for row in rows if _matches_kernel(row, kernel_name)
                ]
                if not matched_rows:
                    matched_rows = [
                        row
                        for row in rows
                        if "triton" in " ".join(
                            str(value) for value in row.values()
                        ).lower()
                    ]
                total_time_us += sum(
                    float(row["Total Time(us)"])
                    for row in matched_rows
                    if row.get("Total Time(us)")
                )
        if total_time_us > 0:
            return total_time_us / active, profiling_dir

    raise RuntimeError(
        f"Kernel {kernel_name!r} was not found in profiler CSVs under "
        f"{profiling_dir}"
    )


def append_perf_result(
    test_function,
    operator_name,
    test_case,
    average_time_us,
    profiling_dir,
):
    """Append one test case row to the performance CSV."""
    fieldnames = [
        "test_function",
        "operator_name",
        "test_case",
        "average_kernel_time_us",
        "profiling_dir",
        "timestamp",
    ]
    needs_header = not os.path.exists(PERF_CSV_PATH) or os.path.getsize(
        PERF_CSV_PATH
    ) == 0
    with open(PERF_CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "test_function": test_function,
                "operator_name": operator_name,
                "test_case": test_case,
                "average_kernel_time_us": f"{average_time_us:.6f}",
                "profiling_dir": profiling_dir,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


def make_profiling_dir(operator_name, direction, dtype, shape_desc):
    dtype_label = dtype_name(dtype)
    run_id = f"{time.time_ns()}_{os.getpid()}"
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "npu_profiling",
        f"{operator_name}_{direction}_{dtype_label}_{shape_desc}_{run_id}",
    )


# ---------------------------------------------------------------------------
# Performance tests — RMSNorm
# ---------------------------------------------------------------------------

RMSNORM_PERF_SHAPE = [
    pytest.param(4096, 7168, id="S4096-D7168"),
]


@pytest.mark.parametrize("dtype", PERF_DTYPE_CASES)
@pytest.mark.parametrize("S,D", RMSNORM_PERF_SHAPE)
def test_rmsnorm_forward_performance(S, D, dtype):
    torch.manual_seed(42)
    x = torch.randn((S, D), device=current_device, dtype=dtype)
    w = torch.randn((D,), device=current_device, dtype=torch.float32)
    eps = 1e-6

    def executor():
        TritonRMSNormFunc.apply(x, w, eps)

    shape_desc = f"S{S}_D{D}"
    profiling_dir = make_profiling_dir("rmsnorm", "forward", dtype, shape_desc)
    average_time_us, _ = device_perf_npu(
        executor, profiling_dir, kernel_name="_rmsnorm_fwd_fused"
    )
    print(
        f"rmsnorm_fwd dtype={dtype_name(dtype)} shape=({S}, {D}) "
        f"average_kernel_time={average_time_us:.6f} us"
    )
    test_case = f"dtype={dtype_name(dtype)},S={S},D={D}"
    append_perf_result(
        "test_rmsnorm_forward_performance",
        "_rmsnorm_fwd_fused",
        test_case,
        average_time_us,
        profiling_dir,
    )


@pytest.mark.parametrize("dtype", PERF_DTYPE_CASES)
@pytest.mark.parametrize("S,D", RMSNORM_PERF_SHAPE)
def test_rmsnorm_backward_performance(S, D, dtype):
    torch.manual_seed(42)
    x = torch.randn((S, D), device=current_device, dtype=dtype)
    w = torch.randn((D,), device=current_device, dtype=torch.float32)
    eps = 1e-6
    dy = torch.randn((S, D), device=current_device, dtype=dtype)

    x_bench = x.clone().requires_grad_(True)
    w_bench = w.clone().requires_grad_(True)
    y = TritonRMSNormFunc.apply(x_bench, w_bench, eps)

    def executor():
        x_bench.grad = None
        w_bench.grad = None
        y.backward(dy, retain_graph=True)

    shape_desc = f"S{S}_D{D}"
    profiling_dir = make_profiling_dir("rmsnorm", "backward", dtype, shape_desc)
    average_time_us, _ = device_perf_npu(
        executor, profiling_dir, kernel_name="_rmsnorm_bwd_dxdw_fused"
    )
    print(
        f"rmsnorm_bwd dtype={dtype_name(dtype)} shape=({S}, {D}) "
        f"average_kernel_time={average_time_us:.6f} us"
    )
    test_case = f"dtype={dtype_name(dtype)},S={S},D={D}"
    append_perf_result(
        "test_rmsnorm_backward_performance",
        "_rmsnorm_bwd_dxdw_fused",
        test_case,
        average_time_us,
        profiling_dir,
    )


# ---------------------------------------------------------------------------
# Performance tests — SiTU-GLU v2
# ---------------------------------------------------------------------------

SITU_PERF_SHAPE = [
    pytest.param(4096, 7168, id="M4096-N7168"),
]


@pytest.mark.parametrize("dtype", PERF_DTYPE_CASES)
@pytest.mark.parametrize("M,N", SITU_PERF_SHAPE)
def test_situgluv2_forward_performance(M, N, dtype):
    torch.manual_seed(42)
    a = torch.randn((M, N), device=current_device, dtype=dtype)
    b = torch.randn((M, N), device=current_device, dtype=dtype)
    beta = 0.5
    linear_beta = 0.5

    def executor():
        SiTUGLUv2.apply(a, b, beta, linear_beta)

    shape_desc = f"M{M}_N{N}"
    profiling_dir = make_profiling_dir("situgluv2", "forward", dtype, shape_desc)
    average_time_us, _ = device_perf_npu(
        executor, profiling_dir, kernel_name="_fused_situgluv2_fwd_triton_kernel"
    )
    print(
        f"situgluv2_fwd dtype={dtype_name(dtype)} shape=({M}, {N}) "
        f"average_kernel_time={average_time_us:.6f} us"
    )
    test_case = f"dtype={dtype_name(dtype)},M={M},N={N}"
    append_perf_result(
        "test_situgluv2_forward_performance",
        "_fused_situgluv2_fwd_triton_kernel",
        test_case,
        average_time_us,
        profiling_dir,
    )


@pytest.mark.parametrize("dtype", PERF_DTYPE_CASES)
@pytest.mark.parametrize("M,N", SITU_PERF_SHAPE)
def test_situgluv2_backward_performance(M, N, dtype):
    torch.manual_seed(42)
    a = torch.randn((M, N), device=current_device, dtype=dtype)
    b = torch.randn((M, N), device=current_device, dtype=dtype)
    beta = 0.5
    linear_beta = 0.5
    dout = torch.randn((M, N), device=current_device, dtype=dtype)

    a_bench = a.clone().requires_grad_(True)
    b_bench = b.clone().requires_grad_(True)
    y = SiTUGLUv2.apply(a_bench, b_bench, beta, linear_beta)

    def executor():
        a_bench.grad = None
        b_bench.grad = None
        y.backward(dout, retain_graph=True)

    shape_desc = f"M{M}_N{N}"
    profiling_dir = make_profiling_dir("situgluv2", "backward", dtype, shape_desc)
    average_time_us, _ = device_perf_npu(
        executor, profiling_dir, kernel_name="_fused_situgluv2_bwd_triton_kernel"
    )
    print(
        f"situgluv2_bwd dtype={dtype_name(dtype)} shape=({M}, {N}) "
        f"average_kernel_time={average_time_us:.6f} us"
    )
    test_case = f"dtype={dtype_name(dtype)},M={M},N={N}"
    append_perf_result(
        "test_situgluv2_backward_performance",
        "_fused_situgluv2_bwd_triton_kernel",
        test_case,
        average_time_us,
        profiling_dir,
    )
