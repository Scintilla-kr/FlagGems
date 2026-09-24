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

"""fused attention (../06-fused-attention_modify.py) 的基准测试。

一次运行得到三列数据:
- AutoTune OFF: 前向 _attn_fwd 取作者默认固定配置(候选列表第一个);
- AutoTune ON: 前向为原生 @triton.autotune 多配置 kernel, 完整搜索择优;
- torch C++ 基线(latency_base): scaled_dot_product_attention, MLU 上 dispatch
  到 CNNL、NPU 上 dispatch 到厂商 fused attention C++ 算子
  (若落到 math fallback 则为非融合 matmul+softmax 路径, 对照时需注明)。

反向 _attn_bwd 为固定配置 Triton kernel, OFF/ON 两轮应基本一致(仅作对照)。

注意:
- 必须在 exec 06 模块前设置 FUSED_ATTN_FULL_CONFIGS=1, 否则该模块在 pytest
  环境下会把 autotune 候选折叠为单配置, AutoTune 失效、OFF/ON 数据失真。
- 运行 cwd 必须在 FlagGems-1 根目录: pytest benchmark/test_fused_attention.py
"""

import importlib.util
import os

import pytest
import torch

from . import autotune_compare, base

# 06 文件名以数字开头, 无法常规 import; 环境开关必须在 exec 前设置
os.environ.setdefault("FUSED_ATTN_FULL_CONFIGS", "1")
_SPEC = importlib.util.spec_from_file_location(
    "fused_attention_modify",
    os.path.join(os.path.dirname(__file__), "..", "06-fused-attention_modify.py"),
)
_fused_attn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fused_attn)


# warp_specialize=False: 非 CUDA 后端不支持 warp specialize(与 06 脚本 bench 一致)
def _gems_attn(q, k, v, causal, sm_scale):
    return _fused_attn.attention(q, k, v, causal, sm_scale, False)


def _torch_attn(q, k, v, causal, sm_scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm_scale
    )


class FusedAttentionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (BATCH, H, N_CTX, HEAD_DIM, causal), 与 06 脚本 perf_report 同源;
        # N_CTX 需为 128 的倍数(反向 PRE_BLOCK=128)
        self.shapes = [
            (4, 32, n_ctx, head_dim, causal)
            for n_ctx in [1024, 4096, 16384]
            for head_dim in [64, 128]
            for causal in [False, True]
        ]

    def get_input_iter(self, dtype):
        # 固定种子: OFF/ON 两轮(以及重跑)输入完全一致, 结果才可对比、可复现
        torch.manual_seed(2026)
        for batch, h, n_ctx, head_dim, causal in self.shapes:
            q = torch.randn((batch, h, n_ctx, head_dim), dtype=dtype, device=self.device)
            k = torch.randn((batch, h, n_ctx, head_dim), dtype=dtype, device=self.device)
            v = torch.randn((batch, h, n_ctx, head_dim), dtype=dtype, device=self.device)
            yield q, k, v, causal, 1.3


@pytest.mark.fused_attention
def test_fused_attention_forward():
    bench = FusedAttentionBenchmark(
        op_name="fused_attention",
        torch_op=_torch_attn,
        dtypes=[torch.float16],  # kernel 内硬编码 tl.float16
        shape_desc="BATCH, H, N_CTX, HEAD_DIM, causal",
    )
    bench.set_gems(_gems_attn)
    # OFF(默认配置)/ON(完整搜索) 两轮, 每轮均含 latency_base(torch C++ 基线)
    autotune_compare.run_autotune_comparison(bench)


@pytest.mark.fused_attention
def test_fused_attention_backward():
    bench = FusedAttentionBenchmark(
        op_name="fused_attention",
        torch_op=_torch_attn,
        dtypes=[torch.float16],
        shape_desc="BATCH, H, N_CTX, HEAD_DIM, causal",
        is_backward=True,
    )
    bench.set_gems(_gems_attn)
    # 反向为固定配置 Triton kernel, OFF/ON 两轮应基本一致(仅作对照);
    # latency_base 为 torch SDPA 反向(C++ 实现)
    autotune_compare.run_autotune_comparison(bench)
