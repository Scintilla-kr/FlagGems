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
  对非 LibTuner 的原生 triton Autotuner(如 FlagGems 之外脚本用的 @triton.autotune),
  额外把候选截断为第一个作为兜底: vendor fork 的 run 未必经由 self._bench 短路,
  截断可在机制上保证 OFF 轮不发生真实搜索。
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
- NPU 上参考 benchmark/rmsnorm_situ_optim.py 的 profiling 方式, 在每轮计时后
  追加一个 kernel 级 profiling pass(torch_npu profiler 采集, 解析落盘
  op_summary/op_statistic CSV), 优先汇总 Triton kernel 时间(无 Triton 实现时
  回退为全部 NPU kernel); 两轮的 kernel 级与端到端数据一并追加写入
  benchmark/autotune_compare_performance.csv(每 case 一行, OFF/ON 并列)。
"""

import csv
import glob
import math
import os
import tempfile
import time
from contextlib import contextmanager

import torch

import flag_gems

# 必须直接按模块路径导入: flag_gems.utils 包的 __init__ 里
# `from .libentry import libentry, libtuner` 把装饰器“函数” libentry
# 遮蔽了同名“子模块”, 写 `from flag_gems.utils import libentry` 拿到的是函数。
from flag_gems.utils.libentry import LibCache, LibEntry, LibTuner, libcache

_libcache = libcache
_ORIG_DB_URL = _libcache.db_url

# 进程内出现过的 tuner / LibEntry 实例(首次调用 run 时注册)
_TUNERS = set()
_LIB_ENTRIES = set()
_state = {"off": False, "seen_off": set()}


def _stock_autotuner_cls():
    """从 LibTuner 的继承链解析原生 triton Autotuner, 而不直接 import triton.runtime。

    部分 vendor 的 triton 分支(如 triton-ascend)在首次导入 triton.runtime 时
    会连带加载后端 C 扩展(triton._C.libtriton.*), 在尚未初始化的进程里可能
    ModuleNotFoundError; 走 flag_gems.libentry 已经成功加载过的路径最稳妥。
    """
    for cls in LibTuner.__mro__[1:]:
        if cls.__name__ == "Autotuner" and cls.__module__.startswith("triton"):
            return cls
    return None


_StockAutotuner = _stock_autotuner_cls()


def _fresh_isolated_db_url():
    dir_path = tempfile.mkdtemp(prefix="flaggems_autotune_cmp_")
    return "sqlite:///" + dir_path.replace("\\", "/").rstrip("/") + "/isolated.db"


def _swap_tuning_db(db_url):
    """把全局调优缓存(libcache 单例)切换到 db_url, 并重绑已注册 tuner 的 ConfigCache。

    LibCache 是单例, 再次实例化会在原对象上重跑 __init__, 更换底层 SQL model
    与缓存池; libentry 模块内的 ``libcache`` 全局名指向同一对象, 无需重新赋值。
    """
    LibCache(db_url)
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
        # 兜底: vendor fork 的原生 Autotuner.run 未必经由 self._bench 短路,
        # 对非 LibTuner 的原生 Autotuner 直接把候选截断为第一个(经 prune 后即
        # 算子作者默认配置), 从机制上强制 OFF 轮跳过真实搜索
        saved_configs = None
        if (
            _StockAutotuner is not None
            and isinstance(self, _StockAutotuner)
            and not isinstance(self, LibTuner)
        ):
            saved_configs = self.configs
            self.configs = list(self.configs[:1])
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
            if saved_configs is not None:
                self.configs = saved_configs
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
# (LibTuner 定义了自己的 run, 需单独包装; 只覆写 policy 的子类不受影响)。
# 若解析不到原生 Autotuner(理论上不会), 退化为只包装 LibTuner。
if _StockAutotuner is not None:
    _StockAutotuner.run = _wrap_tuner_run(_StockAutotuner.run)
    if LibTuner.run is not _StockAutotuner.run:
        LibTuner.run = _wrap_tuner_run(LibTuner.run)
else:
    LibTuner.run = _wrap_tuner_run(LibTuner.run)
LibEntry.run = _wrap_libentry_run(LibEntry.run)


# ---------------------------------------------------------------------------
# kernel 级 profiling(参考 benchmark/rmsnorm_situ_optim.py 的 device_perf_npu):
# torch_npu profiler 采集 + 解析落盘 CSV, 与框架的端到端计时互补。
# 仅 NPU 可用; 其他平台跳过, 性能数据文件只记录端到端时延。
# ---------------------------------------------------------------------------

PERF_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "autotune_compare_performance.csv"
)
_PROFILING_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "npu_profiling"
)
_PROF_WARMUP = 5
_PROF_ACTIVE = 5

_KERNEL_NAME_COLUMNS = ("Op Name", "Name", "Kernel Name")


def _npu_profiling_supported():
    if flag_gems.device != "npu":
        return False
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return True


def _row_mentions_triton(row):
    return "triton" in " ".join(str(value) for value in row.values()).lower()


def _collect_kernel_name(row, names):
    for column in _KERNEL_NAME_COLUMNS:
        value = row.get(column)
        if value:
            names.add(str(value))
            return


def _join_names(names, limit=256):
    text = "|".join(sorted(names))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _collect_profile_rows(profiling_dir):
    """解析 profiler 落盘 CSV, 返回 (行列表, 时长列名)。

    与 rmsnorm_situ_optim.device_perf_npu 的两级解析一致: 优先
    mindstudio_profiler_output/op_summary*.csv 的 "Task Duration(us)",
    无则回退 op_statistic.csv 的 "Total Time(us)"。
    """
    for pattern, duration_column in (
        (
            os.path.join("**", "mindstudio_profiler_output", "op_summary*.csv"),
            "Task Duration(us)",
        ),
        (os.path.join("**", "op_statistic.csv"), "Total Time(us)"),
    ):
        rows = []
        for path in glob.glob(os.path.join(profiling_dir, pattern), recursive=True):
            with open(path, newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                if duration_column not in (reader.fieldnames or []):
                    raise RuntimeError(
                        f"{duration_column} is missing from {path}"
                    )
                rows.extend(reader)
        if rows:
            return rows, duration_column
    return [], None


def _device_perf_npu(executor, profiling_dir):
    """torch_npu profiler 采集平均每迭代的 kernel 总时间(us)。

    返回 (平均us, kernel名集合, 汇总范围)。汇总范围 "triton" 表示只统计
    Triton kernel; 当前算子无 Triton 实现(如 DSA/原生库)时回退为全部
    NPU kernel, 范围记为 "all"。
    """
    import torch_npu

    os.makedirs(profiling_dir, exist_ok=True)

    # 预热一次, 确保编译/建图发生在 profiler 窗口之外
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
            warmup=_PROF_WARMUP,
            active=_PROF_ACTIVE,
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
        for _ in range(_PROF_WARMUP + _PROF_ACTIVE):
            torch.npu.synchronize()
            executor()
            torch.npu.synchronize()
            prof.step()

    rows, duration_column = _collect_profile_rows(profiling_dir)
    matched = [row for row in rows if _row_mentions_triton(row)]
    scope = "triton"
    if not matched:
        # 非 Triton 实现(如 Ascend DSA/原生库): 回退为统计全部 NPU kernel
        matched = rows
        scope = "all"
    if not matched:
        raise RuntimeError(
            f"profiler 未采集到任何 kernel 行: {profiling_dir}"
        )
    total_time_us = 0.0
    names = set()
    for row in matched:
        duration = row.get(duration_column)
        if duration:
            total_time_us += float(duration)
        _collect_kernel_name(row, names)
    return total_time_us / _PROF_ACTIVE, names, scope


def _make_case_executor(bench, args, kwargs):
    """按 _measure_input/get_latency 的语义构造单个 case 的可执行体。

    use_gems 是一次性上下文管理器(__exit__ 会 del 掉自身属性), 同一实例
    不能二次进入: 每次调用都通过 _candidate_call() 新建 dispatch 再进入,
    与框架 _measure_input 每次测量重建 dispatch 的语义一致。
    反向测试先在 dispatch 下建一次计算图, 之后反复 autograd.grad
    (autograd.grad 也需在 dispatch 窗口内, 反向 aten 算子才会路由到 gems)。"""
    if not bench.is_backward:

        def executor():
            op, dispatch, _ = bench._candidate_call()
            with dispatch:
                op(*args, **kwargs)

        return executor

    op, dispatch, _ = bench._candidate_call()
    with dispatch:
        out = op(*args, **kwargs)
    dout = torch.randn_like(out)
    xs = [a for a in args if torch.is_tensor(a) and a.requires_grad]

    def executor():
        _, dispatch, _ = bench._candidate_call()
        with dispatch:
            torch.autograd.grad(
                (out,), xs, grad_outputs=(dout,), retain_graph=True
            )

    return executor


def _make_profiling_dir(op, mode, dtype, idx):
    run_id = f"{time.time_ns()}_{os.getpid()}"
    dtype_label = str(dtype).removeprefix("torch.")
    return os.path.join(
        _PROFILING_ROOT, f"{op}_{mode}_{dtype_label}_case{idx}_{run_id}"
    )


def _profile_pass(bench, tag):
    """对每个 case 采集 kernel 级平均耗时, 返回 record 列表。

    record 含 (dtype, idx, avg_us, names, scope, dir, error); 单个 case 失败
    不影响主流程, 错误信息记入 record 并回填到性能数据文件。"""
    if not _npu_profiling_supported():
        print(
            f"[autotune-cmp] op={bench.op_name}: torch_npu profiler 不可用, "
            f"跳过 kernel 级 profiling(性能数据文件仅记录端到端时延)"
        )
        return None
    print(
        f"[autotune-cmp] op={bench.op_name}: kernel 级 profiling pass "
        f"(torch_npu profiler, warmup={_PROF_WARMUP}, active={_PROF_ACTIVE})"
    )
    records = []
    for dtype in bench.to_bench_dtypes:
        try:
            input_iter = bench.get_input_iter(dtype)
        except Exception as exc:
            print(
                f"[autotune-cmp] profile[{tag}] dtype={dtype} 无法枚举输入: {exc}"
            )
            continue
        idx = 0
        while True:
            try:
                inputs = next(input_iter)
            except StopIteration:
                break
            except Exception as exc:
                print(
                    f"[autotune-cmp] profile[{tag}] dtype={dtype} case#{idx} "
                    f"输入生成失败: {exc}"
                )
                break
            record = {
                "dtype": str(dtype),
                "idx": idx,
                "avg_us": None,
                "names": "",
                "scope": "",
                "dir": "",
                "error": "",
            }
            try:
                args, kwargs = bench.unpack_to_args_kwargs(inputs)
                executor = _make_case_executor(bench, args, kwargs)
                profiling_dir = _make_profiling_dir(
                    bench.op_name, tag, dtype, idx
                )
                avg_us, names, scope = _device_perf_npu(executor, profiling_dir)
                record.update(
                    avg_us=avg_us,
                    names=names,
                    scope=scope,
                    dir=profiling_dir,
                )
            except Exception as exc:  # noqa: BLE001 单 case 失败不影响主流程
                record["error"] = str(exc)
            records.append(record)
            if record["avg_us"] is not None:
                status = f"{record['avg_us']:.1f}us"
            elif record["error"]:
                status = f"失败({record['error']})"
            else:
                status = "跳过"
            print(
                f"[autotune-cmp] profile[{tag}] dtype={record['dtype']} "
                f"case#{idx} kernel_time={status}"
            )
            idx += 1
    return records


def _fmt_us(value):
    return "" if value is None else f"{value:.1f}"


def _ratio(num, den):
    if num is None or den is None or not den:
        return ""
    return f"{num / den:.3f}"


def _flatten_results(results):
    flat = []
    for r in results or []:
        for idx, m in enumerate(r.result):
            flat.append((str(r.dtype), idx, m))
    return flat


def _index_profile(records):
    if not records:
        return {}
    return {(r["dtype"], r["idx"]): r for r in records}


def _append_perf_csv(op, vendor, off_results, on_results, off_prof, on_prof):
    """把 OFF/ON 两轮的 kernel 级与端到端性能数据追加写入 CSV(每 case 一行)。"""
    off_flat = _flatten_results(off_results)
    on_flat = _flatten_results(on_results)
    if not off_flat or not on_flat:
        print(f"[autotune-cmp] op={op}: 无可写入性能数据文件的结果, 跳过")
        return
    prof_off = _index_profile(off_prof)
    prof_on = _index_profile(on_prof)
    fieldnames = [
        "op",
        "vendor",
        "dtype",
        "case",
        "shape_detail",
        "kernel_time_off_us",
        "kernel_time_on_us",
        "kernel_gain",
        "e2e_off_ms",
        "e2e_on_ms",
        "e2e_gain",
        "kernel_scope",
        "kernel_names",
        "profiling_dir_off",
        "profiling_dir_on",
        "timestamp",
    ]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for (dtype, idx, m_off), (_, _, m_on) in zip(off_flat, on_flat):
        key = (dtype, idx)
        p_off = prof_off.get(key, {})
        p_on = prof_on.get(key, {})
        k_off = p_off.get("avg_us")
        k_on = p_on.get("avg_us")
        names = p_on.get("names") or p_off.get("names")
        error = p_off.get("error") or p_on.get("error")
        rows.append(
            {
                "op": op,
                "vendor": vendor,
                "dtype": dtype,
                "case": idx,
                "shape_detail": "" if m_off.shape_detail is None else str(m_off.shape_detail),
                "kernel_time_off_us": _fmt_us(k_off),
                "kernel_time_on_us": _fmt_us(k_on),
                "kernel_gain": _ratio(k_off, k_on),
                "e2e_off_ms": _fmt_ms(m_off.latency),
                "e2e_on_ms": _fmt_ms(m_on.latency),
                "e2e_gain": _ratio(m_off.latency, m_on.latency),
                "kernel_scope": p_on.get("scope") or p_off.get("scope") or "",
                "kernel_names": _join_names(names) if names else error,
                "profiling_dir_off": p_off.get("dir", ""),
                "profiling_dir_on": p_on.get("dir", ""),
                "timestamp": timestamp,
            }
        )
    needs_header = not os.path.exists(PERF_CSV_PATH) or os.path.getsize(
        PERF_CSV_PATH
    ) == 0
    with open(PERF_CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
    print(
        f"[autotune-cmp] op={op}: 已追加 {len(rows)} 行性能数据到 {PERF_CSV_PATH}"
    )


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

    每轮计时结束后, 在相同的 AutoTune 状态下(OFF 轮仍在禁用上下文内)追加
    一个 kernel 级 profiling pass; 两轮数据最终追加写入性能数据文件
    (PERF_CSV_PATH, 每 case 一行)。

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
    off_prof, on_prof = None, None
    try:
        # 隔离真实调优库: 防止历史 tuned 配置让 OFF 轮直接“免费”吃到调优结果;
        # 同时清掉本会话先前测试在 launch 层留下的同参数缓存
        _clear_launch_caches()
        _swap_tuning_db(_fresh_isolated_db_url())
        print(f"===== [{op}] Pass 1/2: AutoTune OFF (默认固定配置) =====")
        with _autotune_disabled():
            off_results = _run_pass(bench, "AutoTune_OFF")
            # 计时轮已让 launch 缓存持有 OFF 默认配置的 kernel,
            # 在同一禁用状态下采集 kernel 级耗时, 口径与计时轮一致
            off_prof = _profile_pass(bench, "AutoTune_OFF")
        # 清 launch 层缓存 + 换全新隔离库, 确保第二轮真正重新调优
        _clear_launch_caches()
        _swap_tuning_db(_fresh_isolated_db_url())
        print(f"===== [{op}] Pass 2/2: AutoTune ON (完整配置搜索) =====")
        with _tuning_evidence():
            on_results = _run_pass(bench, "AutoTune_ON")
            # 调优后的稳态 kernel 级耗时
            on_prof = _profile_pass(bench, "AutoTune_ON")
    finally:
        _swap_tuning_db(_ORIG_DB_URL)
    _report(op, off_results, on_results)
    _append_perf_csv(op, vendor, off_results, on_results, off_prof, on_prof)
    return off_results, on_results
