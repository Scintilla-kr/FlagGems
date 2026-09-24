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


class TopKAscendBenchmark(base.GenericBenchmark2DOnly):
    # 与 FlagGems_new 仓的通用 shapes 保持一致: 原大 Shape(FlagTree 规格,
    # N 最大 524288、k 最大 8192)会在 NPU 上触发 DSA topk 的 UB overflow,
    # 已移除。DSA topk 仅支持 2D 输入, 故按 FlagGems_new _input_fn 的语义
    # 把通用 shapes 折算成 ((m, n), k): 2 元 shape 即 x=shape、k=5;
    # 3 元 (m, n, k) 即 x=(m, n)、k=k; 3D 输入展平为 2D
    # (对 last-dim topk 语义等价)。
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((64, 64), 5),
            ((4096, 4096), 5),
            ((10000, 256), 5),
            ((10000, 65536), 5),
            ((4, 128), 5),
            ((8, 256), 5),
            ((64, 128), 8),
            ((64, 1024), 32),
            ((64, 8192), 128),
            ((128, 32768), 256),
            ((512, 64), 5),  # 原 ((4, 128, 64), 5), 3D 展平为 2D
            ((512, 64), 64),  # 原 ((4, 128, 64), 64)
            ((4096, 32), 32),  # 原 ((8, 512, 32), 32)
            ((16384, 256), 256),  # 原 ((16, 1024, 256), 256)
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


def _ascend_input_fn(shape, dtype, device):
    if isinstance(shape[0], (tuple, list)):
        x_shape, k = shape
    else:
        x_shape, k = shape, (5 if shape[-1] > 5 else shape[-1])
    x = torch.randn(x_shape, device=device, dtype=dtype)
    yield {"x": x, "k": k},


@pytest.mark.topk
def test_topk():
    if flag_gems.vendor_name == "ascend":
        # DSA topk has no dim/largest args and is called directly (base.py
        # skips use_gems when gems_op is given); torch.topk(x, k) defaults to
        # last-dim largest=True on both sides. DSA 实现为固定配置, 无 Triton
        # AutoTune, OFF/ON 两轮结果应基本一致(仅作对照)。
        bench = TopKAscendBenchmark(
            op_name="topk",
            input_fn=_ascend_input_fn,
            torch_op=torch.topk,
            gems_op=flag_gems.topk,
            dtypes=[torch.float32],
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
