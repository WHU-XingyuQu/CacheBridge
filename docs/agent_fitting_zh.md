# Agent 轨迹与拟合开发指南

CacheBridge 将模型路由、轨迹采样、离线拟合和在线 KV 转换拆分为独立接口。
`offline` 负责拟合数据与系数，`online` 负责转换、缓存生命周期和模型切换。

## 接入 agent 与 router

在 `cachebridge.agents` 中实现环境、router 和模型 backend 协议，
通过 `sample_trajectory` 收集完整事件序列。每次模型切换记录路由决策、
实际 cache mode、精确 token 前缀和模型身份。生成请求使用独立 request ID，
同一 episode 使用明确的缓存命名空间。

`cachebridge.offline.trajectory.handoff_samples` 将 native TRAIN 轨迹转换为
拟合输入。保存 source history 和 target history/query 的 KV 快照，保证
token、tokenizer、模型版本以及 H/Q 边界一致。

## 采样与续采

τ-bench 适配器提供现成的采样入口：

```bash
cachebridge serve --config /path/to/runtime.json -- \
  python benchmarks/tau_bench/run.py sample \
  --tasks recipes/tau_train.example.json --output /path/to/native-bank
```

采样目录保存全部尝试、可用 handoff、token 与 KV 文件引用。生成失败、
无 handoff 和预算停止均保留在尝试记录中；训练输入不按任务 reward 筛选。

```bash
cachebridge serve --config /path/to/runtime.json -- \
  python benchmarks/tau_bench/extend.py \
  --inputs /path/to/native-bank/inputs.jsonl \
  --output /path/to/extended-bank \
  --max-query-tokens 4096 --max-target-calls 24
```

续采保留原任务顺序与已结束记录，只延长达到预算的 native continuation。
执行前后检查原 token、动作、观察和 KV 前缀；原 assistant 预测位置全部保留。
工具观察作为上下文，assistant mask 对齐实际下一个 token 的预测输入位置。

## 离线拟合

```bash
cachebridge-fit prepare \
  --inputs /path/to/native-bank/inputs.jsonl \
  --target-model /path/to/Qwen3-32B \
  --initializer /path/to/initializer/weights \
  --freeze-keys --output /path/to/fit

CUDA_VISIBLE_DEVICES=0,1,2,3 cachebridge-fit run \
  --directory /path/to/fit --mapper-id my-agent-mapper

cachebridge-verify /path/to/fit/candidate.json
```

`--freeze-keys` 保留 initializer 的 K；省略该选项则用输入轨迹重新拟合 K。
V 按 target layer 顺序校准，每层使用当前候选状态计算 attention-output
目标并闭式求解，再传播到下一层。改变 K 或数据时，应重新执行完整 V 校准。
支持层、数值精度、数据顺序和采样预算写入拟合配置。

## 在线运行与评估

将 mapper descriptor 填入 runtime 配置，通过 SGLang 集成运行 Source TP4
和 Target TP4。在线路径包含分块转换、双缓冲、GPU peer copy 和 Target
缓存直接安装。新适配器应检查完整历史的一致性、Source 不变性和缓冲区生命周期。

```bash
python benchmarks/tau_bench/compare.py \
  --native /path/to/native-evaluation \
  --candidate /path/to/mapped-evaluation \
  --tasks recipes/tau_test.json --output /path/to/comparison.json
```

任务评估使用相同 router、任务列表、模型版本、用户模拟器、生成预算和 seed。
比较完整任务 reward，同时记录 handoff、fallback、工具调用和停止原因。
三组对照可使用 `benchmarks/tau_bench/analyze.py`。

接口契约见 [trajectory 文档](agent_trajectories.md)，数值步骤见
[fitting 文档](fitting.md)，部署参数见 [serving 文档](serving.md)。
