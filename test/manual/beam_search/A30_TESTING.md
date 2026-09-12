# A30 Beam Search Attention 测试说明

## 1. 测什么、不测什么

本实验比较同一套 Q/K/V 和路径映射上的三种执行方式：

| 模式 | 行为 |
| --- | --- |
| `gather` | 当前 SGLang Triton decode attention，逐 beam 读取自己的路径 |
| `shared` | 按祖先树序分组，共同 decode 前缀共享加载，剩余路径 gather，最后合并 LSE |
| `auto` | 在分组后，只有共同前缀长度和预计节省量达到固定阈值的组才共享 |

不修改 beam selection、模型、生产 scheduler、KV allocator 或 VMM。输入是合成 Q/K/V 和合法的 parent-selection 轨迹，不是模型生成的真实输出。测试目标是通用 beam search，不是 SID 专用实现。

- `beam_width=1` 只运行原 gather，跳过共享 metadata。
- `auto` 的零前缀组仍会经过 metadata、空前缀和合并 kernel，并非零成本切换到完整原路径。
- 阈值尚未在 A30 上校准；`policy_status=loaded_candidate` 仅表示读取了配置文件，不表示该策略已经通过性能验收。
- 开发机已通过 CPU 参考、调用参数、资源预算及命令行检查；A30 CUDA 编译、数值和性能必须由下面的 GPU 测试确认。

入口：`test/manual/beam_search/test_beam_kv_attention.py`。

以下命令均在仓库根目录、同一个已激活的 Python 环境中执行；无需启动 SGLang server 或下载模型。

## 2. 环境准备与记录

主目标为完整的 **NVIDIA A30 24GB / SM80**。脚本允许 SM80 及更新架构，但不同 GPU 的结果应分开保存。若使用 MIG 实例，其显存、SM 数和性能不能按完整 A30 解读。

需要：

- Linux，建议 Python 3.11；CPU 检查仅依赖 Python 标准库。
- CUDA 版 PyTorch 和 Triton，优先复用已经可工作的环境并固定版本。
- 与实际 CUDA runtime 兼容的驱动，以及 Triton 所需的系统编译工具。
- FP16 为主矩阵的数据类型，BF16 单独补测。
- SASS 导出需要可用的反汇编工具；缺失时结果会记录原因，不会伪造 SASS。

`a30-case` 使用独立源码加载器，只加载 torch/Triton 和仓库内的数值 kernel，不导入完整 SGLang runtime。不要为了运行实验执行不加约束的 `pip install -e ./python`，否则可能升级整套服务依赖。

此前 V100 数据使用 PyTorch 2.7.1、CUDA runtime 12.6、Triton 3.4.0；这只是历史环境记录，不是 A30 已认证的版本组合，也不要求强行覆盖 PyTorch 声明的 Triton 依赖。新旧实现必须在同一环境比较。

```bash
nvidia-smi --query-gpu=name,driver_version,mig.mode.current --format=csv,noheader

python - <<'PY'
import sys
import torch
import triton

print("Python:", sys.version)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Triton:", triton.__version__)
assert torch.cuda.is_available()
assert torch.version.hip is None
print("GPU:", torch.cuda.get_device_name())
print("Capability:", torch.cuda.get_device_capability())
print("Properties:", torch.cuda.get_device_properties(0))
print("Free/total bytes:", torch.cuda.mem_get_info())
assert torch.cuda.get_device_capability()[0] >= 8
x = torch.randn(128, 128, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print("FP16 PyTorch smoke: OK")
PY
```

上面的检查不等于 Triton kernel 已通过。不要同时运行其他 GPU 负载；记录功率限制、MIG 和共享机器上的干扰，勿擅自修改公共机器驱动或时钟。

## 3. 推荐测试顺序

### 3.1 CPU 自检与预览

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py self-test

RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULTS="test/manual/beam_search/results/a30_${RUN_ID}"

python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-smoke --output-dir "$RESULTS/smoke" --dry-run
```

`--dry-run` 不导入 CUDA、不创建结果目录，只打印将执行的命令。

### 3.2 先运行 smoke

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-smoke --output-dir "$RESULTS/smoke"
```

4 个案例覆盖 k=1、非二次幂 MHA/GQA 和独立路径。除基本数值检查外，小宽度共享案例还检查 inactive query 与全共享边界。smoke 也有短性能采样，不是纯编译测试。

先确认所有案例正确，检查编译信息中共享前缀 kernel 是否生成 MMA，再运行后续矩阵。出现失败时先定位，不要放宽 tolerance 或跳过失败项后宣布通过。

### 3.3 校准 kernel 配置

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-calibrate --output-dir "$RESULTS/calibrate"
```

39 个案例，针对少量宽度与 MHA/GQA 比较 query tile、KV tile、prefix split、warps 和 pipeline stages；不是所有参数的完整笛卡尔积，也不是自动选择阈值的完整搜索。

先用 `total` 和跨层结果选择候选配置，再在低共享、多子树和未参与校准的宽度上验证。不要将每个覆盖测试点各自最优的结果拼接成一个“自动策略”。

### 3.4 运行覆盖矩阵和跨层/轨迹测试

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-main --output-dir "$RESULTS/main"

python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-trace --output-dir "$RESULTS/trace"

python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-stress --output-dir "$RESULTS/stress"
```

上面的命令使用默认候选策略。正式验证冻结策略时，按下一节对每个命令追加同一份 `--policy`。

| preset | 案例数 | 内容 |
| --- | ---: | --- |
| `a30-smoke` | 4 | 小规模正确性、编译与基本计时 |
| `a30-calibrate` | 39 | 有限 kernel 配置校准 |
| `a30-main` | 330 | 11 个宽度 × 5 个输出长度 × 3 种共享形态 × MHA/GQA |
| `a30-stress` | 150 | 5 个大宽度 × 同样的长度、共享形态和 head 配置 |
| `a30-trace` | 13 | 多步轨迹、随机种子、8/32 层、多请求、BF16、head_dim=64、prompt 和缓存压力 |

主宽度：`1,2,3,6,8,12,16,24,32,64,128`。
压力宽度：`256,512,1024,2048,4096`。
输出长度：`3,16,32,64,128`。
共享形态：`shared,grouped,independent`；head 配置为 `8/8` 和 `8/2`。

主/压力矩阵保留了最初提出的全部 55 个宽度×长度组合，并补齐常用宽度。矩阵默认 30 个计时样本、10 次 warmup；大矩阵耗时可能较长，先确认 smoke 和代表案例，再跑全量。

## 4. 单案例、重跑与冻结策略

### 单案例

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py a30-case \
  --beams 24 --output-tokens 64 --pattern grouped \
  --q-heads 8 --kv-heads 2 --head-dim 128 --dtype float16 \
  --query-tile 16 --tile 32 --prefix-splits 2 --splits 1 2 4 \
  --output "$RESULTS/manual/grouped_gqa.json" \
  --dump-dir "$RESULTS/manual/kernels"
```

`query-tile` 限制每个 KV head 对应的 query 行数，不是 beam 数。组大小为 `min(beams, floor(query_tile / GQA_ratio))`；尾组允许不足额。prefix split 与 suffix split 是两个独立参数。

### 冻结策略

将校准后选择的参数保存为 JSON，例如仓库外的 `/tmp/a30_policy.json`。以下只是格式示例，不是已验证的最优配置：

```json
{
  "schema": "beam-prefix-policy-v1",
  "min_prefix": 16,
  "min_saved": 128,
  "query_tile": 16,
  "tile": 32,
  "prefix_splits": 2,
  "suffix_splits": 2,
  "warps": 4,
  "stages": 2
}
```

- `min_prefix`：组内共同 **beam-KV** 前缀的最小 token 数。
- `min_saved`：`(组内 beam 数 - 1) × common_len` 的最小值；这是逻辑 KV 引用节省量，不是实测 HBM 字节数。
- query/tile/prefix split/warps/stages 会覆盖相应 CLI 配置，作用于共享计算路径。
- `suffix_splits` 固定 `auto` 的尾部 split；`gather/shared` 仍按 `--splits` 比较，便于诊断。
- 若未设置 `suffix_splits`，`auto` 也会枚举 `--splits`，不能把最优 split 汇总解释为已经实现了在线选优。
- 校准 kernel 时不要误传一份覆盖所有参数的 policy，否则会抵消校准矩阵的参数变化。

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-main --policy /tmp/a30_policy.json \
  --output-dir "$RESULTS/main_frozen"
```

策略由使用者确认后固定，脚本不会自动从测试结果写出生产策略。保留 policy 文件及其来源，正式报告必须包含未参与校准的案例。

### 只重跑指定案例

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-main --case k24_n64_grouped_gqa \
  --output-dir "$RESULTS/retry_grouped_gqa"
```

矩阵目录必须不存在；单案例 `--output` 也拒绝覆盖。失败后不自动续跑旧目录，用 `--case` 和新目录重跑。矩阵会在首个失败处停止，失败日志和之前成功的 JSON 留在原目录。

矩阵仅转发自身支持的选项。若需改变 `--memory-gib`、`--execution`、`--layers` 等单案例参数，使用 `a30-case`；不要把这些参数直接追加给 `matrix`。

## 5. 多层、prompt 与真实父节点轨迹

- `--output-tokens n` 对应 beam-KV 最终深度 `n-1`；首个输出按常规流程来自 prefill logits。
- 默认是最终深度的一步 attention 快照，不是从第 1 步到第 n 步的生成延迟。
- `--layers 8/32` 使用独立的每层 Q/K/V，`total` 中 metadata 只构建一次；没有 MLP、LM head 或层间隐藏状态传递。
- `--prompt-len` 将 prompt 留在逐 beam gather 路径中，只有 decode 前缀共享被优化，不能将 prompt 读取收益计入本算法。
- `--trace` 逐步更新 parent/path/tree order，验证输出与路径；KV 使用预生成的 append-only step slot，不模拟在线模型计算或 KV 回收。

真实轨迹文件格式为 `parents[request][step][beam]`。每个 parent 是上一轮 beam 的局部行号，范围 `[0, beam_width)`；step 数必须为 `output_tokens-1`。首步之前 beam history 为空，因此首步 parent 不影响历史内容。

例如一个请求、3 个 beam、3 个输出 token：

```json
{
  "parents": [
    [[0, 1, 2], [0, 0, 2]]
  ]
}
```

保存为 `/tmp/beam_parent_trace.json` 后执行：

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py a30-case \
  --beams 3 --output-tokens 3 --requests 1 \
  --trace --parent-trace /tmp/beam_parent_trace.json \
  --output "$RESULTS/manual/real_parents.json"
```

这里只替换 parent-selection 轨迹，Q/K/V 仍为合成数据。`trace_sum` 是各步重复测量中位数的和，GPU 区间不包含 parent 上传和上一步路径 snapshot；它不是单次完整请求墙钟时间。基准同一步重复测量使用固定的 prior-step 状态，不会重复追加 token。

## 6. 显存预算与大宽度测试

默认预算为 `min(16 GiB, 启动时空闲显存 × 0.7)`，分配前估算 KV、query/output、metadata、分段 workspace、参考计算和余量。

- 新方案不分配 `[beam, pool_slot]` dense mask，也不物化全池 attention score。
- 普通快照在计时前将存活节点紧凑放置；trace 使用 append-only slot。离线布局成本不算在线 compaction 收益。
- `--reference-rows 1` 表示参考计算一次处理一行，所有行仍会校验，不是只校验第一行。
- 预算不足返回 `status=resource_skipped` 并继续矩阵，不缩小 beam width；这不等于测试通过。
- 预算是估算，不保证实际一定不会 OOM；额外编译/图缓存、其他进程和碎片仍可能影响运行。

例如手动降低预算：

```bash
python -B test/manual/beam_search/test_beam_kv_attention.py a30-case \
  --beams 4096 --output-tokens 128 --pattern independent \
  --memory-gib 12 --memory-fraction 0.6 --reference-rows 1 \
  --output "$RESULTS/manual/large_budgeted.json"
```

不要通过增加 `reference-rows`、复制多层 KV 或同时保留多套大输入来强行加速参考检查；先检查估算和实际峰值。

## 7. 如何解读结果

每个矩阵目录包含：

- `manifest.json`：命令、退出码、案例状态与日志位置。
- `<case>.json`：环境、正确性、计时、分组和资源信息。
- `<case>.log`：该案例 stdout/stderr，失败时首先查看。
- `kernels/<case>/`：smoke/calibrate 自动导出的 PTX、TTGIR、cubin 和可用的 SASS；其他矩阵仅记录编译信息，单案例可用 `--dump-dir` 导出。

`manifest.complete=true` 只表示矩阵运行到了结尾；还要检查 `resource_skipped=0` 和 `all_cases_validated=true`。对选定子集的完成也不等于整套矩阵完成。

重点字段：

| 字段 | 解读 |
| --- | --- |
| `correctness` | 输出误差；共享路径还检查 LSE，trace 检查路径和树序 |
| `grouping` | 共同前缀长度、选中组数、逻辑 KV 引用量 |
| `global_common_prefix` / `ancestor_counts_by_depth` | 区分全局共享和子树共享 |
| `timings.scope=metadata` | 排序、共同前缀识别和路径准备 |
| `attention_all_layers` | 所有层 attention 与必要合并，不含 metadata |
| `attention_first_layer` | 多层实验的首层对照 |
| `total` | 一次 metadata + 全部层 attention |
| `comparisons` | 本案例测试过的最佳 suffix split 对照，不是生产自动选优 |
| `compiled_kernels` | 实际 grid、寄存器、spill、shared memory、静态 MMA/FMA 指令计数 |
| `memory_estimate` / `torch_peak_allocated_delta_bytes` | 分配前估算与 PyTorch 统计峰值；后者不涵盖所有驱动内存 |
| `source_hashes` / `loaded_gather_sha256` | 原文件和独立加载源码的可追溯信息 |

`speedup_vs_best_gather > 1` 才表示更快。只在相同 dtype、形状、层数、缓存条件、环境和计时方式下比较；不能把 V100 的绝对时间直接当作 A30 基线。

默认 graph 计时仅用于降低 Python 提交噪声，不是本轮优化成果：warm 模式将 64 次相同工作放进一个 replay 后平均；scrub 模式每次只有一个调用，仍可能含 replay 提交开销。eager 区间也可能含 Python 提交间隙。不要将不同计时口径的差值直接解释为 HBM 流量变化。

PTX MMA 计数是静态指令数，不是 Tensor Core 利用率；spill 不是实测 HBM 字节数。本工具不采集 L2/DRAM 硬件计数器，若要解释访存瓶颈，需要额外的 Nsight Compute 测量。

## 8. 失败处理与回传

1. 首先保留失败日志和 manifest，不覆盖旧目录。
2. 数值失败记录 dtype、case、配置、软件版本；不要直接放宽容差。
3. CUDA illegal access 后重启案例进程；矩阵本身已按案例隔离进程。
4. SASS 导出失败检查 JSON 中的 `sass_unavailable`，与数值测试失败分开处理。
5. `resource_skipped` 记录为未验证；OOM 也不是算法正确性通过。
6. 不以 `auto` 默认阈值声称达到“低共享退化小于 5%”或“加速超过 10%”，这些仍是后续实测验收目标。

回传至少保留 JSON、manifest、日志和使用的 policy；编译器异常或 spill/MMA 分析再附对应案例的 PTX/SASS。产物可能很多，不必重复提交所有 cubin；日志中可能含机器路径，上传前检查是否需要去除环境敏感信息。

旧 `gpu-check`、`benchmark`、`v100-tune` 和 `compiler` 入口继续保留，但它们测试的是前一版全池扫描实验，不应与新的 `a30-case` 前缀/尾部算法混为同一方案。
