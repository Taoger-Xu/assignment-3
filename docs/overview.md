# Assignment 3 项目总览：scottyrl

> 本文是对 [ASSIGNMENT.md](../ASSIGNMENT.md)、[README.md](../README.md) 和源码的梳理，帮助快速理解框架结构与每个 Part 需要完成的代码和实验。具体规范以各 stub 的 docstring 为准。

---

## 1. 一句话概括

用一个极简的 RL 框架 `scottyrl`，在 **Qwen3.5-4B + LoRA(rank 32)** 上训练一个多轮工具调用 Agent：

- 先在便宜的 **Number Search**（二分查找猜数）上实现并验证 REINFORCE / GRPO / 异步 GRPO，做消融；
- 最后在昂贵的 **TextCraft-Synth**（合成配方 DAG）上做一次 "hero run"。

训练与采样的 GPU 计算都交给一个 **Tinker 兼容的后端服务**（本作业是自己在 Modal 上部署的 SkyRL server），本地框架只负责 "Agent ↔ Env 交互 → 构造训练数据 → 调用 `forward_backward` / `optim_step` / `sample`"。

---

## 2. 总体架构

### 2.1 分层视图

```
┌──────────────────────── 本地（CPU，可离线测试） ───────────────────────────┐
│                                                                            │
│  scripts/train_sync.py / train_async.py   ← 入口：加载 config、数据、session │
│              │                                                             │
│              ▼                                                             │
│  Trainer (scottyrl/train/)                                                 │
│   ├─ SyncTrainer       ★  采样 → 构 batch → 更新 → 刷新 policy（交替）      │
│   └─ AsyncGRPOTrainer  ★  rollout workers → 有界队列 → learner（重叠）      │
│              │ 调用                                                         │
│   common.py ★ collect_group / groups_to_batch                              │
│              │                                                             │
│   ┌──────────┴───────────┐                                                 │
│   ▼                      ▼                                                 │
│  rollout.run_episode    advantages.py ★ + datagen.py ★                     │
│  (Agent ↔ Env 一个回合)   (Trajectory → TokenSequence)                       │
│   │        │                                                               │
│   ▼        ▼                                                               │
│  agents/   envs/                                                           │
│  XMLToolCallAgent   NumberSearchEnv / TextCraftEnv                          │
│   │                                                                        │
│   ▼                                                                        │
│  policy.TinkerPolicy  ── sample ──┐                                        │
│  tinker_adapter.TinkerSession ★ ──┼── forward_backward / optim_step ──┐    │
└───────────────────────────────────┼───────────────────────────────────┼────┘
                                    ▼                                   ▼
                    ┌──────── SkyRL Tinker Server on Modal (GPU) ────────┐
                    │  vLLM 采样引擎            Megatron + LoRA 训练       │
                    └────────────────────────────────────────────────────┘
```

★ = 需要你实现的文件（共 6 个，见第 4 节）。其余全部为给定代码，评分时会被还原，**不要改**。

### 2.2 核心数据流（建议先读 [scottyrl/types.py](../scottyrl/types.py)）

```
Task ──run_episode──▶ Trajectory ──stitch + advantage──▶ TokenSequence ──to_tinker_datum──▶ tinker.Datum ──▶ 后端
```

| 类型 | 含义 |
|---|---|
| `Task` | 一个环境实例：`goal`、`id`、`max_steps`、`info`（目标数字 / 配方与库存） |
| `LLMCall` | 一次 LLM 调用，**在 token 空间记录**：`prompt_tokens`、`completion_tokens`、`completion_logprobs` |
| `Trajectory` | 一个完整回合：`calls: list[LLMCall]` + 最终 `reward`（0/1） |
| `TokenSequence` | 训练单元：`tokens`（长度 n）+ `mask` / `logprobs` / `advantages`（长度均为 n-1，第 j 位描述 target `tokens[j+1]`） |

> 用一个具体回合逐步演示这条数据流（含 token 对齐表）见 [dataflow.md](dataflow.md)。

### 2.3 关键设计：Sequence Extension（序列扩展）

一个回合包含多次 LLM 调用。Agent（[xml_tool_call.py](../scottyrl/agents/xml_tool_call.py)）在 token 空间中维护上下文：每轮把**模型自己生成的 completion token** 原样拼回去，只对新增的尾部（turn 标记、环境观测、generation prompt）做 tokenize。因此通常有：

```
calls[i+1].prompt_tokens == calls[i].prompt_tokens + calls[i].completion_tokens + <新尾部>
```

满足这个性质时，整个回合可以**拼成一条训练序列**。以下情况会断开，需要另起一条新序列：

- `model.preserve_thinking_history: false`：历史中的 `<think>` 被删掉，prompt 被重写；
- 模型以 `<|endoftext|>` 而不是 `<|im_end|>` 停止，模板回填的结尾不同。

`datagen.stitch_trajectory` 必须同时处理"能拼"与"断开"两种情况。模板、停止符、思维链等基础概念见 [template.md](template.md)。

### 2.4 损失函数与算法（由 config 两个字段独立控制）

- `train.algo`：优势估计
  - `reinforce`：`A = r`（无 baseline，group_size=1）
  - `grpo`：`A_i = r_i − mean(r)`（Dr.GRPO：**不除以 std，无 KL**）；组内奖励完全相同 → 退化组，返回 `None` 并跳过
- `train.loss_fn`：token 级损失（在 [tinker_adapter.to_tinker_datum](../scottyrl/tinker_adapter.py) 中实现，已给定）
  - `ppo`：clipped 重要性比率（GRPO 默认）
  - `importance_sampling`：不 clip 的 ρ·A（REINFORCE 用）
  - `cross_entropy`：A·log π，无重要性比率（仅 Part 4.3 "no IW" 消融）

`train_step` 会把优势统一乘 `1/T`（T = batch 中 action token 总数），得到 batch 级的平均 token loss。

### 2.5 两个环境

| | Number Search | TextCraft-Synth |
|---|---|---|
| 工具 | `guess(n)` | `craft`、`get_info`、`view_inventory`、`finish` |
| 反馈 | too low / too high / correct | 文本观测 |
| 步数预算 | ⌈log₂(high−low+2)⌉，正好是二分最坏情况，**零冗余** | 75 |
| 成功条件 | 猜中 | 合成全部目标物品 **并且** 调用 `finish()` |
| CoT | 关闭（`enable_thinking: false`） | 开启，每轮最多 512 token |
| 数据 | 50,000 train / 1,000 val | 852 train / 213 val（medium）/ 136 val_hard |
| 单次成本 | ~$10，~40 分钟 | ~$100，6–8 小时 |

奖励均为二值 0/1。Agent 每轮输出一个 `<tool>name(args)</tool>`。

### 2.6 目录速查

| 路径 | 作用 |
|---|---|
| [scottyrl/types.py](../scottyrl/types.py) | 核心数据结构 |
| [scottyrl/config.py](../scottyrl/config.py) | `ModelConfig` / `TrainConfig`，YAML 加载，`--set` 覆盖，run_name 防覆盖 |
| [scottyrl/policy.py](../scottyrl/policy.py) | `TinkerPolicy`（真实采样）、`MockPolicy`（测试用）、`CompletionsServerPolicy`（vLLM） |
| [scottyrl/agents/](../scottyrl/agents/) | `XMLToolCallAgent`：解析工具调用、维护 token 上下文 |
| [scottyrl/envs/](../scottyrl/envs/) | 环境基类与两个环境 |
| [scottyrl/rollout.py](../scottyrl/rollout.py) | `run_episode`：跑一个回合，记录 transcript 和时间 |
| [scottyrl/evaluate.py](../scottyrl/evaluate.py) | `evaluate_policy`：success_rate 与 pass@k |
| [scottyrl/metrics.py](../scottyrl/metrics.py) | `MetricsLogger`（jsonl / transcripts / wandb）、`sanity_warnings` |
| [scottyrl/tinker_adapter.py](../scottyrl/tinker_adapter.py) | 唯一接触 Tinker SDK 的地方：`TinkerSession`、`train_step`、checkpoint、resume |
| [configs/](../configs/) | 7 个必做实验各一份 YAML + `required_runs.yaml` |
| [scripts/](../scripts/) | 训练、评估、导出 HF、打包提交 |
| [skyrl_tinker_modal/](../skyrl_tinker_modal/) | 三个 Modal 部署脚本 |
| [tests/](../tests/) | 公开测试（25 个，起始时 12 个通过） |

---

## 3. 运行产物

每个 run 由 `train.run_name` 标识，输出到 `runs/`：

| 产物 | 路径 |
|---|---|
| 每步指标 | `runs/<run_name>.jsonl`（**评分依据**；step 0 是未训练模型的基线评估） |
| Transcript | `runs/<run_name>_trajectories/step_XXXXXX/`（每 5 步保存一次，务必阅读） |
| Checkpoint 清单 | `runs/<run_name>.checkpoints.json`（含 `final_sampler_path`） |
| Resume 标记 | `runs/<run_name>.resume.json` |

已有输出的 run_name 会被拒绝：新实验换名字，崩溃后用 `--set train.resume=true` 续跑。

**报告中每个实验都要画的标准诊断图**（横轴为 training step）：

- 奖励：`train/success_rate`、`eval/success_rate`、所有 `train/pass@*` 与 `eval/pass@*`
- 优化：`train/grad_norm`
- 策略漂移：`train/importance_weight_mean` / `_max` / `_std`
- 数据健康：`train/frac_degenerate_groups`、`train/entropy_proxy`

计时指标（`train/step_time_s`、`train/learner_wait_frac` 等）不用画进标准图，但 Part 3/4.2 的速度对比要用到。

---

## 4. 需要实现的 6 个文件（按顺序）

| # | 文件 | 实现内容 | Part |
|---|---|---|---|
| 1 | [scottyrl/advantages.py](../scottyrl/advantages.py) | `compute_group_advantages` | 1 |
| 2 | [scottyrl/datagen.py](../scottyrl/datagen.py) | `extension_suffix`、`stitch_trajectory`、`trajectory_to_training_sequences` | 1 |
| 3 | [scottyrl/train/common.py](../scottyrl/train/common.py) | `collect_group`、`groups_to_batch` | 2 |
| 4 | [scottyrl/tinker_adapter.py](../scottyrl/tinker_adapter.py) | 仅 `TinkerSession.submit_update` | 2 |
| 5 | [scottyrl/train/sync.py](../scottyrl/train/sync.py) | `SyncTrainer`：`collect_batch`、`build_batch`、`update`、`evaluate_now`、`train` | 2 |
| 6 | [scottyrl/train/async_grpo.py](../scottyrl/train/async_grpo.py) | `AsyncGRPOTrainer`：`generate_group`、`rollout_worker`、`collect_fresh_batch`、`update`、`evaluate_now`、`train` | 3 |

规则：函数签名不能改（隐藏测试会直接调用），可以加私有 helper；公开测试通过不等于完全正确。

---

## 5. 各 Part 详解

### Part 0：热身（0 分）

阅读 [agents/xml_tool_call.py](../scottyrl/agents/xml_tool_call.py)、[rollout.py](../scottyrl/rollout.py)、[envs/](../scottyrl/envs/)，理解一个回合如何产生 `Trajectory`。代码分析与 `async` / `await` 的解释见 [part0.md](part0.md)。

### Part 1：数据管线（20 分）——纯代码，无需 GPU

**实现**

- `compute_group_advantages(rewards, method)`
  - `reinforce` → 原样返回奖励；
  - `grpo` → 组大小 < 2 时 `ValueError`；奖励全部**精确相等**时返回 `None`；否则返回 `r − mean`；
  - 其它 method → `ValueError`。
- `extension_suffix(accumulated, prompt)`：若 `accumulated` 是 `prompt` 的前缀，返回剩余后缀（可为空），否则返回 `None`。
- `stitch_trajectory(traj)`：按顺序遍历 calls，能扩展就只追加后缀和 completion，断开就结束当前序列、从该 call 的完整 prompt 重新开始。completion 位置的 `mask=1`，`logprobs` 填采样 logprob；其余位置为 0；`advantages` 全 0。不输出长度 < 2 或没有 action token 的序列。
- `trajectory_to_training_sequences(traj, advantage)`：先 stitch，再把标量优势广播到所有 `mask==1` 的位置。

**最容易出错的地方：off-by-one。** 长度为 L 的上下文后接长度为 c 的 completion，对应的位置是 `L-1 : L-1+c`，不是 `L : L+c`。

**解答思路**：各函数的详细分析见 [part1.md](part1.md)。

**测试**：`uv run pytest tests/test_interfaces.py -k "advantages or datagen"`；`tests/test_envs.py` 中的 `ThinkStrippingPolicy` 可以构造"扩展断开"的回合。

**实验**：无。

### Part 2：同步 REINFORCE 与 GRPO（25 分）

**实现**

- `collect_group`：为同一个 task 用 `asyncio.gather` 并发跑 `group_size` 个回合，每个回合用全新的 env 和 agent。
- `groups_to_batch`：逐组算优势；退化组不产出序列但计数，且其奖励仍计入 `mean_reward`；返回 `(sequences, stats)`，`stats` 中的键见 docstring。
- `submit_update`：**先提交** `forward_backward_async` 和 `optim_step_async(AdamParams(learning_rate, grad_clip_norm))` 两个请求，**再分别** `await result_async()`（顺序 await 会让延迟翻倍，有专门测试检查）；不要捕获 SDK 异常。
- `SyncTrainer`
  - `collect_batch`：用 `self.rng` 抽 `tasks_per_batch` 个不同任务，各组并发采集；
  - `build_batch`：调用 `groups_to_batch(groups, cfg.algo)`；
  - `update`：调用 `session.train_step`，原样返回 metrics；
  - `evaluate_now`：前 `eval_tasks` 个验证任务，每个跑 `eval_k` 次；
  - `train`：开始前保存快照 `step_000000`，每步更新后保存 `step_{t:06d}` 并刷新 policy（保证 on-policy）；batch 为空时是 **skipped step**，不更新、不保存快照，但仍调用一次 `record_training(step, [], {})`；`eval_due(step)` 为真时评估当前 policy（step 0 的基线由入口脚本完成）。

**测试**：`uv run pytest tests/test_interfaces.py tests/test_contracts.py -k "not async"`

**实验**（部署 `serve_number_search_sync.py`，2×H100 共置）

| Run | 配置 | 说明 |
|---|---|---|
| `p2a_number_search_reinforce` | `algo=reinforce`，G=1，`loss=importance_sampling` | 8 tasks/step，50 步 |
| `p2b_number_search_grpo` | `algo=grpo`，G=8，`loss=ppo` | **参考配置**，其它实验都相对它描述 |

两者都每 10 步在 50 个验证任务上评估，每个任务 4 次 rollout。

**报告**：两组 run 的标准诊断图；用 3–5 句比较 REINFORCE 与 GRPO 的学习速度并解释原因。正常实现的奖励在 50 步内应明显高于 step 0；曲线是平的就先 debug 再写分析。

### Part 3：异步 GRPO（25 分）

> 异步的详细解释（group 与 batch 的关系、陈旧度模拟、与经典 off-policy 的区别）见 [async.md](async.md)。

**设计**：生产者-消费者。

```
N 个 rollout_worker ──put──▶ asyncio.Queue(maxsize=queue_capacity) ──get──▶ learner
   （用捕获的 (policy, version) 采样）                                （丢弃过旧的组 → 更新 → publish 新版本）
```

**实现要点**

- `generate_group(task_id, policy, version)`：按 id 查任务（未知 id 抛错）；调用模块级的 `collect_group`；用 GRPO 算优势；**总是**返回 `GroupResult`（退化组 `sequences=[]` 但保留 `rewards`）；version 必须是传入值，不能读当前值。
- `rollout_worker`：循环执行"取 task id → `active_policy()` 捕获一对 (policy, version) → `generate_group` → `await queue.put`"；不捕获异常。
- `collect_fresh_batch`：在**消费时**判断陈旧度：`self.version − group.policy_version > max_staleness` 就丢弃，并累加到 `stats["groups_dropped_stale"]`；凑够 `groups_per_batch` 个新鲜组（含退化组）后返回。
- `update`：按组顺序拼接所有序列，调用 `session.train_step`。
- `evaluate_now`：与同步版本相同，使用模块级的 `evaluate_policy`。
- `train`
  - `publish_policy(0)` → 启动 workers，共享一个循环的 task id 迭代器（`itertools.cycle`）；
  - step 从 1 到 `num_steps`：收集 batch → 有序列就 update 并 `publish_policy(version+1)`，否则跳过 → 每步调用一次 `record_training`；
  - 评估：`eval_due` 时捕获 `active_policy()`，以 **后台 task** 调度 `run_evaluation`，不阻塞训练；
  - 结束时 cancel 并 await workers，然后 await 所有评估 task；
  - **任一异常**（worker、评估、learner）都要 cancel 并 await 其余所有 task，然后重新抛出**原始异常对象**（不能包成 ExceptionGroup）。worker 挂掉时 learner 不能卡在 `queue.get()` 上，例如用 `asyncio.wait(..., FIRST_COMPLETED)` 同时监视 get 与 worker/eval tasks；
  - 返回恰好三个键：`train_steps`、`groups_consumed`、`groups_dropped_stale`。

**测试**：`uv run pytest`，全部公开测试都应通过。`test_async_*` 用 fake rollout 测并发逻辑，不花算力。

**实验**（部署 `serve_number_search_async.py`，1 卡训练 + 1 卡采样）

| Run | 与 p2b 的差异 |
|---|---|
| `p3_number_search_async` | 异步 trainer，8 workers，`groups_per_batch=8`，`max_staleness=2`，`queue_capacity=16` |

日志最后一条记录包含 `summary/train_steps`、`summary/groups_consumed`、`summary/groups_dropped_stale`（Part 4.2 要用）。

**报告**

1. p3 的标准诊断图；比较 importance weight 的 mean / min / max / std（mean ≈ 1 可能掩盖重尾）。
2. 用 `train/step_time_s` 比较 p3 与 p2b 每步的墙钟时间，量化加速比。
3. 从 transcript 中挑早、中、晚期各一条训练 rollout（注明 step、task id、文件头中的 `policy_version`），描述至少两种行为变化（如是否二分、如何利用反馈、如何从非法调用中恢复）。

### Part 4：Number Search 消融（20 分）——配置不要改

| 小节 | 对比的 runs | 新增 run | 部署 | 关注点 |
|---|---|---|---|---|
| 4.1 组大小 | p2a（G=1）、p41（G=2）、p2b（G=8） | `p41_number_search_grpo_g2` | sync 服务器 | 用 `eval/success_rate` 与 `train/frac_degenerate_groups` 解释小组对优势方差和每步有效序列数的影响；REINFORCE 的退化比例记为 N/A；注意 p2a 的优势规则和 loss 也不同，不是纯粹的组大小消融 |
| 4.2 同步 vs 异步 | p2b、p3 | 无 | — | 复用 Part 3 的时间分析，加上两者最终 `eval/success_rate` 的小表；解释加速来源，以及陈旧数据是否损害样本质量（用 IW 统计与 `groups_dropped_stale` 作证据） |
| 4.3 高陈旧度下的重要性加权 | iw vs noiw | `p43_..._stale4_iw`（ppo）、`p43_..._stale4_noiw`（cross_entropy） | async 服务器 | 两者都是 `max_staleness=4`，只有 loss 不同；画标准诊断图 |

每小节都要包含所述图表和一段分析。

### Part 5：TextCraft-Synth hero run（10 分）

**先决条件**：异步 trainer 已在 Number Search 上跑通。只跑一次，过程中盯紧指标和 transcript。

**实验**（部署 `serve_textcraft.py`，1 卡训练 + 3 卡采样，上下文 65k）

| Run | 配置 |
|---|---|
| `p5_textcraft_grpo` | 异步 GRPO，thinking 开启，max_tokens=512，每步 16 组×8 rollouts，32 workers，`max_staleness=4`，50 步 |

**训练后**

1. 评估：微调模型（`--model-path <final_sampler_path>`）与基座模型（不传 `--model-path`），分别在 `textcraft_val.jsonl`（medium）与 `textcraft_val_hard.jsonl` 上用 `--k 4` 评估，共 4 次；
2. 导出：`uv run hf auth login`，然后 `uv run python scripts/export_checkpoint.py --run p5_textcraft_grpo`，上传到公开仓库 `<hf-username>/scottyrl-a3-p5_textcraft_grpo`（需要 textcraft 服务处于运行状态）。

**报告**：p5 的标准诊断图；一张 4 行表（微调/基座 × medium/hard，列为 `success_rate` 和 `pass@4`）；几句结果分析。

---

## 6. 实验总表（7 个必做 runs）

| Run name | Part | 入口脚本 | Modal 部署 | algo / G | loss | 关键差异 |
|---|---|---|---|---|---|---|
| `p2a_number_search_reinforce` | 2, 4.1 | `train_sync.py` | `serve_number_search_sync.py` | reinforce / 1 | importance_sampling | 无 baseline |
| `p2b_number_search_grpo` | 2, 4.1, 4.2 | `train_sync.py` | `serve_number_search_sync.py` | grpo / 8 | ppo | 参考配置 |
| `p41_number_search_grpo_g2` | 4.1 | `train_sync.py` | `serve_number_search_sync.py` | grpo / 2 | ppo | 小组 |
| `p3_number_search_async` | 3, 4.2 | `train_async.py` | `serve_number_search_async.py` | grpo / 8 | ppo | 异步，staleness 2 |
| `p43_number_search_async_stale4_iw` | 4.3 | `train_async.py` | `serve_number_search_async.py` | grpo / 8 | ppo | staleness 4 |
| `p43_number_search_async_stale4_noiw` | 4.3 | `train_async.py` | `serve_number_search_async.py` | grpo / 8 | cross_entropy | staleness 4，无 IW |
| `p5_textcraft_grpo` | 5 | `train_async.py` | `serve_textcraft.py` | grpo / 8 | ppo | TextCraft，CoT 开启 |

公共设置：Qwen3.5-4B，LoRA r=32，lr=1e-5，grad_clip=1.0，50 步，每 10 步评估（50 tasks × k=4），seed=11768。

**预算估算**：6 个 Number Search 约 6×$10，加 1 个 TextCraft 约 $100，加上 probe 与镜像构建（首次构建约 20 分钟以上），总计约 $180–200。三个 Modal app 相互独立可以并行；同一个 launcher 上想同时跑两个 run，需要用 `SKYRL_APP_NAME` 部署多个 app。

---

## 7. 推荐工作流

1. `uv sync && uv run pytest`，确认起始状态是 12/25 通过。
2. 实现 Part 1，跑对应测试。
3. 实现 Part 2 代码，跑 `-k "not async"` 测试。
4. 部署 sync 服务器，先跑 **short probe**：
   ```bash
   uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
       --set train.num_steps=10 --set train.eval_every=2 --set train.eval_tasks=20 \
       --set train.run_name=probe_p2b
   ```
   检查：step 0 有基线评估；transcript 看起来像真实回合（`null` tool_call 表示解析失败）；`importance_weight_mean ≈ 1`；能解释所有 `[sanity]` 输出。
5. 正式跑 p2a、p2b、p41（可以用多个 app 并行）。
6. 实现 Part 3，`uv run pytest` 全部通过；部署 async 服务器，probe 后跑 p3、p43_iw、p43_noiw。
7. 确认异步训练表现正常后，部署 textcraft 服务器跑 p5，然后做 4 次评估并导出到 HF。
8. 写 `report.pdf` 与 `AI_USAGE.md`（放在仓库根目录），运行 `uv run python scripts/package_submission.py` 生成 `submission.zip`，上传 Canvas。

---

## 8. 调试信号与常见坑

| 现象 | 可能原因 |
|---|---|
| 同步训练中 `importance_weight_mean` 明显偏离 1 | logprobs 与 token 错位（Part 1 的 off-by-one） |
| 奖励曲线平坦 | mask 覆盖了观测 token、优势广播错误、policy 没有在更新后刷新 |
| `frac_degenerate_groups` 接近 1 | 任务太难或太易、组太小（p41 中属于预期现象） |
| `entropy_proxy < 0.005` | 熵坍缩 |
| `importance_weight_max > 10` 或 `std > 1` | 策略漂移重尾（异步 / 高陈旧度时值得关注） |
| `grad_norm > 50` | 梯度尖峰 |
| 异步训练卡住不动 | worker 异常后 learner 仍阻塞在 `queue.get()`；或结束时没有 cancel workers |
| 每步耗时翻倍 | `submit_update` 中先 await 了 forward_backward 才提交 optim_step |

`[sanity]` 警告的阈值定义在 [scottyrl/metrics.py](../scottyrl/metrics.py) 的 `sanity_warnings` 中。

---

## 9. 评分

| Part | 分值 | 内容 |
|---|---|---|
| 1 | 20 | 数据管线（主要看隐藏单测） |
| 2 | 25 | 同步 trainer + p2a/p2b 实验与分析 |
| 3 | 25 | 异步 trainer + p3 实验与分析 |
| 4 | 20 | 三组消融 |
| 5 | 10 | TextCraft hero run、评估、HF 导出 |
| | **100** | |
