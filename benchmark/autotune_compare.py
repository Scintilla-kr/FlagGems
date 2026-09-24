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

"""在同一进程内对同一算子执行 有/无 AutoTune 两轮基准测试并对比结果。

语义定义
--------
- 无 AutoTune (OFF): 让 tuner 跳过真实搜索, 直接采用其(经 prune 的)候选列表中的
  第一个配置, 即算子作者给出的默认固定配置。未选中的候选不参与编译与运行。
- 有 AutoTune (ON): 恢复原生行为, 对每个 autotune key 完整搜索全部合法候选并择优。
  计时发生在 warmup 之后, 因此测得的是调优后的稳态性能。

为什么要清理三层缓存(否则第二轮永远测不到调优效果)
------------------------------------------------
1. LibEntry 的 kernel 缓存按“调用参数”缓存了第一轮选好配置的 kernel,
   当 @libentry 在 @libtuner 外层时, 同一输入第二轮直接命中缓存、完全绕过 tuner;
2. stock triton autotuner 的内存 best-config dict;
3. flag_gems 持久化调优库(libcache, 默认 ~/.flaggems/TunedConfig_*.db)。
   测试期间将其切换到临时隔离的 sqlite 文件: 既避免历史 tuned 配置污染 OFF 轮,
   也避免 OFF 轮的占位数据写坏真实调优库; 结束后切回原库(原库内容不受影响)。

注意
----
- 若当前平台/算子没有多配置的 Triton kernel(如 Ascend DSA topk、固定 BLOCK_SIZE
  的 rms_norm), OFF/ON 两轮结果应基本一致, 输出中会打印对照说明。
- 该对比应串行执行, 不要与 --parallel 组合。
"""

import math
import os
import tempfile
from contextlib import contextmanager

from triton.runtime import Autotuner as _StockAutotuner

import flag_gems
from flag_gems.utils import libentry as _le

_LibTuner = _le.LibTuner
_LibEntry = _le.LibEntry
_libcache = _le.libcache
_ORIG_DB_URL = _libcache.db_url

# 进程内出现过的 tuner / LibEntry 实例(首次调用 run 时注册)
_TUNERS = set()
_LIB_ENTRIES = set()
_state = {"off": False, "seen_off": set()}


def _fresh_isolated_db_url():
    dir_path = tempfile.mkdtemp(prefix="flaggems_autotune_cmp_")
    return "sqlite:///" + dir_path.replace("\\", "/").rstrip("/") + "/isolated.db"


def _swap_tuning_db(db_url):
    """把全局调优缓存(libcache 单例)切换到 db_url, 并重绑已注册 tuner 的 ConfigCache。

    LibCache 是单例, 再次实例化会在原对象上重跑 __init__, 更换底层 SQL model
    与缓存池; libentry 模块内的 ``libcache`` 全局名指向同一对象, 无需重新赋值。
    """
    _le.LibCache(db_url)
    for tuner in _TUNERS:
        table = getattr(tuner, "config_table_name", None)
        if table is not None:
            tuner.cache = _libcache[table]


def _clear_launch_caches():
    """清空进程内 launch 层缓存, 强制下一轮重新经过 tuner 选择配置。"""
    for entry in _LIB_ENTRIES:
        for cache in entry.kernel_cache:
            cache.clear()
        entry._cpu_cache.clear()
    for tuner in _TUNERS:
        if isinstance(tuner.cache, dict):
            # stock triton autotuner 的 self.cache 是普通 dict
            tuner.cache.clear()


def _wrap_tuner_run(orig_run):
    def run(self, *args, **kwargs):
        _TUNERS.add(self)
        if not _state["off"] or len(getattr(self, "configs", ())) <= 1:
            return orig_run(self, *args, **kwargs)
        _state["seen_off"].add(id(self))
        had_bench = "_bench" in self.__dict__
        saved_bench = self.__dict__.get("_bench")
        saved_cache = self.__dict__.get("cache")
        # OFF: 候选评估恒定返回同一耗时, tuner 由此选择(经 prune 的)第一个配置;
        # 评估函数被短路意味着未选中配置既不编译也不运行。
        self._bench = lambda *a, **k: 1.0
        off_cache = getattr(self, "_autotune_cmp_off_cache", None)
        if off_cache is None:
            off_cache = {}
            setattr(self, "_autotune_cmp_off_cache", off_cache)
        # OFF 轮的默认配置只落在进程内 dict, 不写入持久化调优库
        self.cache = off_cache
        try:
            return orig_run(self, *args, **kwargs)
        finally:
            if had_bench:
                self._bench = saved_bench
            else:
                self.__dict__.pop("_bench", None)
            if saved_cache is not None:
                self.cache = saved_cache
            else:
                self.__dict__.pop("cache", None)

    return run


def _wrap_libentry_run(orig_run):
    def run(self, *args, **kwargs):
        _LIB_ENTRIES.add(self)
        return orig_run(self, *args, **kwargs)

    return run


# 在 import 期打补丁: 同时覆盖 stock triton.autotune 与 FlagGems LibTuner
# (LibTuner 定义了自己的 run, 需单独包装; 依据只覆写 policy 的子类不受影响)。
_StockAutotuner.run = _wrap_tuner_run(_StockAutotuner.run)
if _LibTuner.run is not _StockAutotuner.run:
    _LibTuner.run = _wrap_tuner_run(_LibTuner.run)
_LibEntry.run = _wrap_libentry_run(_LibEntry.run)


@contextmanager
def _autotune_disabled():
    prev_print = os.environ.get("TRITON_PRINT_AUTOTUNING")
    # OFF 轮不打印伪调优信息
    os.environ.pop("TRITON_PRINT_AUTOTUNING", None)
    _state["off"] = True
    _state["seen_off"] = set()
    try:
        yield
    finally:
        _state["off"] = False
        if prev_print is not None:
            os.environ["TRITON_PRINT_AUTOTUNING"] = prev_print


@contextmanager
def _tuning_evidence():
    prev = os.environ.get("TRITON_PRINT_AUTOTUNING")
    # ON 轮打印每个 key 实际选中的配置, 作为调优发生的直接证据
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TRITON_PRINT_AUTOTUNING", None)
        else:
            os.environ["TRITON_PRINT_AUTOTUNING"] = prev


def _run_pass(bench, tag):
    """执行一轮基准测试, 并把 tag 写入每个 metric 的 case_id 以便在报告中区分。"""
    orig_measure = bench._measure_input

    def measure(inputs, case_id=None):
        metric = orig_measure(inputs, case_id)
        if metric is not None and getattr(metric, "case_id", None) is None:
            metric.case_id = tag
        return metric

    bench._measure_input = measure
    try:
        return bench.run()
    finally:
        bench.__dict__.pop("_measure_input", None)


def _fmt_ms(value):
    return "N/A" if value is None else f"{value:.4f}"


def _report(op, off_results, on_results):
    if not off_results or not on_results:
        print(f"[autotune-cmp] op={op}: 无可对比结果(query/list/profile 模式), 跳过对比")
        return
    if not _state["seen_off"]:
        print(
            f"[autotune-cmp] op={op}: 本轮未发现参与配置搜索的多配置 Triton kernel"
            f"(当前平台可能使用 DSA/固定配置实现), OFF/ON 两轮结果应基本一致(仅作对照)。"
        )
    print(f"\n===== [{op}] AutoTune OFF vs ON 对比 =====")
    print(f"{'dtype':<18}{'#':>3}  {'OFF(ms)':>10}{'ON(ms)':>10}{'gain':>8}  shape")
    gains = []
    for r_off, r_on in zip(off_results, on_results):
        if str(r_off.dtype) != str(r_on.dtype):
            print(f"[autotune-cmp] dtype 不匹配: {r_off.dtype} vs {r_on.dtype}, 跳过")
            continue
        for i, (m_off, m_on) in enumerate(zip(r_off.result, r_on.result)):
            lat_off, lat_on = m_off.latency, m_on.latency
            if lat_off is None or lat_on is None:
                gain_txt = "N/A"
            else:
                gain = (lat_off + 1e-9) / (lat_on + 1e-9)
                gains.append(gain)
                gain_txt = f"{gain:.2f}x"
            print(
                f"{str(r_off.dtype):<18}{i:>3}  "
                f"{_fmt_ms(lat_off):>10}{_fmt_ms(lat_on):>10}{gain_txt:>8}  "
                f"{m_off.shape_detail}"
            )
    if gains:
        geo = math.exp(sum(math.log(g) for g in gains) / len(gains))
        wins = sum(1 for g in gains if g > 1.0)
        print(
            f"[autotune-cmp] op={op}: 有效样本={len(gains)}, ON 更快样本={wins}, "
            f"几何平均提升={geo:.2f}x (>1 表示开启 AutoTune 后更快)"
        )
    else:
        print(f"[autotune-cmp] op={op}: 无有效 latency 样本可对比")


def run_autotune_comparison(bench):
    """对给定 benchmark 依次执行 无AutoTune/有AutoTune 两轮压测并输出对比。

    Returns:
        (off_results, on_results): 两轮 ``bench.run()`` 返回的 BenchmarkResult 列表。
    """
    op = bench.op_name
    vendor = flag_gems.vendor_name
    print(
        f"\n[autotune-cmp] op={op} vendor={vendor}: "
        f"OFF=默认固定配置(不搜索), ON=完整 AutoTune 搜索"
    )
    off_results, on_results = None, None
    try:
        # 隔离真实调优库: 防止历史 tuned 配置让 OFF 轮直接“免费”吃到调优结果;
        # 同时清掉本会话先前测试在 launch 层留下的同参数缓存
        _clear_launch_caches()
        _swap_tuning_db(_fresh_isolated_db_url())
        print(f"===== [{op}] Pass 1/2: AutoTune OFF (默认固定配置) =====")
        with _autotune_disabled():
            off_results = _run_pass(bench, "AutoTune_OFF")
        # 清 launch 层缓存 + 换全新隔离库, 确保第二轮真正重新调优
        _clear_launch_caches()
        _swap_tuning_db(_fresh_isolated_db_url())
        print(f"===== [{op}] Pass 2/2: AutoTune ON (完整配置搜索) =====")
        with _tuning_evidence():
            on_results = _run_pass(bench, "AutoTune_ON")
    finally:
        _swap_tuning_db(_ORIG_DB_URL)
    _report(op, off_results, on_results)
    return off_results, on_results
