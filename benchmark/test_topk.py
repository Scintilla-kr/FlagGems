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

import pytest
import torch

import flag_gems

from . import autotune_compare, base, consts

# 说明: shapes 与 _input_fn 同 FlagGems_new 仓的 test_topk.py 保持一致。
# NPU 上 topk 位于 _ascend 的 CUSTOMIZED_UNUSED_OPS 名单, Triton 版 topk
# 不会注册为 aten::topk 的 override, use_gems() 对 topk 空转(会静默回退
# CANN 原生实现); 而 flag_gems.topk 已被 SpecOpRegistrar 替换为 DSA topk
# (当前 NPU 上触发 UB overflow)。因此 Ascend 分支显式传入通用 Triton topk
# (flag_gems.ops.topk.topk) 作 gems_op, 保证 NPU 上真实执行 Triton kernel,
# 与 CANN 原生 topk 形成真实对比。注意: 通用 topk 的 kernel 均为固定配置
# (无 Triton autotune), AutoTune OFF/ON 两轮结果应一致(仅作对照)。


class TopKBenchmark(base.GenericBenchmark2DOnly):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (4096, 4096),
            (10000, 256),
            (10000, 65536),
            (4, 128),
            (8, 256),
            (64, 128, 8),
            (64, 1024, 32),
            (64, 8192, 128),
            (128, 32768, 256),
            ((4, 128, 64), 5),
            ((4, 128, 64), 64),
            ((8, 512, 32), 32),
            ((16, 1024, 256), 256),
        ]


def _input_fn(shape, dtype, device):
    if len(shape) == 2 and isinstance(shape[0], (tuple, list)):
        x_shape, k = shape
        x = torch.randn(x_shape, device=device, dtype=dtype)
        yield {"x": x, "k": k, "dim": -1},
    elif len(shape) == 3:
        m, n, k = shape
        x = torch.randn((m, n), device=device, dtype=dtype)
        yield {"x": x, "k": k, "dim": -1},
    else:
        x = torch.randn(shape, device=device, dtype=dtype)
        k = 5 if shape[-1] > 5 else shape[-1]
        yield {"x": x, "k": k, "dim": -1},
    # TODO:  Currently only support sorted == True and only support topk in last dimension
    # if Config.bench_level == BenchLevel.COMPREHENSIVE:
    #     k = 5 if shape[0] > 5 else shape[0]
    #     yield {"x": x, "k": k, "dim": 0},
    #     yield {"x": x, "k": k, "dim": -1, "sorted": False},


@pytest.mark.topk
def test_topk():
    if flag_gems.vendor_name == "ascend":
        from flag_gems.ops.topk import topk as generic_topk

        bench = TopKBenchmark(
            op_name="topk",
            input_fn=_input_fn,
            torch_op=torch.topk,
            gems_op=generic_topk,
            dtypes=consts.FLOAT_DTYPES,
        )
    else:
        bench = TopKBenchmark(
            op_name="topk",
            input_fn=_input_fn,
            torch_op=torch.topk,
            dtypes=consts.FLOAT_DTYPES,
        )

    # 依次执行 无AutoTune(默认配置)/有AutoTune(完整搜索) 两轮并输出对比
    autotune_compare.run_autotune_comparison(bench)
