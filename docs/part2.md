# Part 2：同步 RL 的逻辑链

> 配合 [overview.md](overview.md)、[part1.md](part1.md)（数据管线）和 [part0.md](part0.md)（Agent、Rollout、asyncio）阅读。本文把 Part 2 要实现的函数串成一条逻辑链，说明每一环的输入输出、契约要点，以及它们如何组合成同步训练循环。以各 stub 的 docstring 为准。

Part 2 要实现的函数：

| 文件 | 函数 |
|---|---|
| [scottyrl/train/common.py](../scottyrl/train/common.py) | `collect_group`、`groups_to_batch` |
| [scottyrl/tinker_adapter.py](../scottyrl/tinker_adapter.py) | `TinkerSession.submit_update` |
| [scottyrl/train/sync.py](../scottyrl/train/sync.py) | `SyncTrainer`：`collect_batch`、`build_batch`、`update`、`evaluate_now`、`train` |

一句话概括：**同步 trainer 的每一步 = 采数据 → 加工成训练样本 → 更新模型 → 刷新采样策略 → 记录 / 评估。**

---

## 1. 全景：一次训练运行的调用链

★ 为需要实现的函数，其余为给定代码。

```
scripts/train_sync.py  main()
│
├─ TinkerSession.start()                                     连接后端、创建 LoRA 训练客户端
│
├─ 基线评估（step 0，由入口脚本负责）
│     session.new_sampling_client("baseline") → make_policy → ★evaluate_now → record_evaluation(0, …)
│
└─ ★SyncTrainer.train()
      │
      ├─ policy = make_policy(await session.new_sampling_client("step_000000"))      初始快照 v0
      │
      └─ for step in 1 … num_steps:
            │
            ├─ ①  groups = ★collect_batch(policy)
            │        └─ asyncio.gather( ★collect_group(policy, task, …) × tasks_per_batch )
            │                └─ asyncio.gather( run_episode(agent, env) × group_size )
            │
            ├─ ②  seqs, stats = ★build_batch(groups)
            │        └─ ★groups_to_batch(groups, cfg.algo)
            │                ├─ compute_group_advantages        （Part 1）
            │                └─ trajectory_to_training_sequences （Part 1）
            │
            ├─ ③  if seqs:
            │        metrics = ★update(seqs)
            │           └─ session.train_step(seqs, loss_fn, lr, grad_clip_norm)
            │                 ├─ to_tinker_datum × N                 （乘 1/T 归一化）
            │                 └─ ★submit_update(datums, …)
            │                       ├─ forward_backward_async
            │                       └─ optim_step_async
            │     else: 跳过（skipped step），metrics = {}
            │
            ├─ ④  if 更新了: policy = make_policy(await session.new_sampling_client(f"step_{step:06d}"))
            │
            ├─ ⑤  record_training(step, groups, seqs, stats, metrics)
            │
            └─ ⑥  if eval_due(step): record_evaluation(step, await ★evaluate_now(policy))
```

---

## 2. 数据在链上的形态变化

```
list[Task]                                       train_tasks（50,000 个）
   │ ① 抽 8 个任务，每个跑 8 次
   ▼
list[list[Trajectory]]                           8 组 × 8 条 = 64 个回合
   │ ② 组内算优势 → 拼接成序列 → 平铺
   ▼
(list[TokenSequence], stats)                     约 64 条序列（退化组不贡献）+ 统计
   │ ③ 转成 Datum → 前向反向 → 优化器一步
   ▼
dict（loss、grad_norm、importance_weight_* …）    后端返回的训练指标
   │ ④ 保存新权重的快照
   ▼
新的 TinkerPolicy                                 下一步用它采样
```

（数字以 p2b 为例：`tasks_per_batch = 8`，`group_size = 8`。）

---

## 3. 逐环展开

### ① `collect_batch` → `collect_group`：采数据

| | `collect_group`（common.py） | `SyncTrainer.collect_batch`（sync.py） |
|---|---|---|
| 粒度 | 一个任务的一组 | 一步的整个 batch |
| 做什么 | 对**同一个任务**并发运行 `group_size` 个回合 | 抽 `tasks_per_batch` 个**不同的**任务，每个任务调用一次 `collect_group`，各组也并发 |
| 关键要求 | 每个回合使用**新的** env 和 agent：`run_episode(agent_cls(policy), env_factory(task))`；用 `asyncio.gather` 并发启动 | 用 `self.rng` 抽任务（保证可复现）；任务互不重复；使用 cfg 指定的 agent 类 |
| 返回 | `list[Trajectory]`，长度为 G，按 rollout 顺序 | `list[list[Trajectory]]`，每个任务一组 |

两层 `gather` 使 8 × 8 = 64 个回合同时进行，vLLM 可以批量生成（见 [part0.md 第 7 节](part0.md)）。

**为什么每个回合都要新建 env 和 agent？** env 有状态（TextCraft 的库存、已用步数），agent 也有状态（消息列表、token 上下文、`llm_calls`）。共用会让不同回合互相污染。

### ② `build_batch` → `groups_to_batch`：加工成训练样本

`build_batch` 只是一层薄包装：调用 `groups_to_batch(groups, method=cfg.algo)`。真正的逻辑在 `groups_to_batch`：

```
对每一组：
    rewards = 这一组每条轨迹的 reward（按 rollout 顺序）
    advs = compute_group_advantages(rewards, method)
    if advs is None:                          # 退化组（GRPO 下组内奖励全部相同）
        计数 +1，不产出序列（但奖励仍计入 mean_reward）
    else:
        对每条轨迹：seqs += trajectory_to_training_sequences(traj, 它的优势)   # 优势为 0.0 也要
```

返回 `(sequences, stats)`，`stats` 至少包含：

| 键 | 含义 |
|---|---|
| `num_groups` | 传入的总组数 |
| `num_degenerate_groups` | 被跳过的退化组数 |
| `frac_degenerate_groups` | 两者之比（无组时为 0.0） |
| `mean_reward` | **所有**轨迹的平均奖励，包括退化组（无轨迹时为 0.0） |
| `num_sequences` | 序列条数 |
| `num_action_tokens` | 所有 mask = 1 的位置总数 |

最后两个键会被给定的 `record_training` 与 `sequence_batch_stats` 的结果核对，不一致时打印 `[sanity]` 警告，可借此发现计数错误。

**`cfg.algo` 在这里起作用**：p2a 为 `reinforce`（A = r），p2b 和 p41 为 `grpo`（A = r − 组均值）。

### ③ `update` → `train_step` → `submit_update`：更新模型

```
SyncTrainer.update(seqs)                         ★ 只需调用下一层，原样返回 metrics
   └─ session.train_step(seqs, cfg.loss_fn, cfg.lr, cfg.grad_clip_norm)     给定
         ├─ T = batch 中动作 token 总数
         ├─ datums = [to_tinker_datum(seq, loss_fn, advantage_scale=1/T) …]
         ├─ fwd_result, opt_result = await submit_update(datums, …)         ★
         └─ 从两个结果中提取 loss、importance_weight_*、grad_norm 等指标
```

`update` 只负责调用并返回指标，**不要**在这里写日志，日志由 `train()` 交给 `record_training`。

`submit_update` 是 Part 2 中唯一直接使用 Tinker SDK 的地方：

| 要求 | 说明 |
|---|---|
| 两个调用 | `training_client.forward_backward_async(datums, loss_fn=loss_fn)` 与 `training_client.optim_step_async(AdamParams(learning_rate=lr, grad_clip_norm=grad_clip_norm))` |
| **顺序** | **先提交两个，再等待两个结果**（`await future.result_async()`）。Tinker 会把连续到达的请求放进同一个服务器周期；若等前一个结果回来再提交后一个，每一步的延迟大约翻倍 |
| 异常 | **不要捕获**，让 SDK 错误直接抛出 |
| 返回 | `(fwd_result, opt_result)` |

**`cfg.loss_fn` 在这里起作用**：p2a 为 `importance_sampling`，p2b 和 p41 为 `ppo`。

### ④ 刷新策略：保证 on-policy

更新完成后，必须**立即保存新权重的快照**，并用它创建新的采样策略：

```
policy = self.make_policy(await self.session.new_sampling_client(f"step_{step:06d}"))
```

| 规则 | 原因 |
|---|---|
| 训练开始前先建 `step_000000` 的快照 | 第 1 步要用初始权重采样 |
| 第 t 步更新后建 `step_{t:06d}` | 第 t+1 步的数据必须来自第 t 步更新后的权重，这就是"同步 = on-policy" |
| 跳过的步不建快照 | 没有更新，权重未变，下一步继续用同一个策略；快照名因此可能不连续 |

如果忘记刷新，模型虽然在训练，采样却一直使用旧权重：数据越来越 off-policy，奖励曲线也上不去。快照名使用六位数字，是因为给定的 `_snapshot_name` 在断点续训时依据这个格式换算绝对步数。

### ⑤ 记录：每一步恰好一次

调用给定的 `record_training(step, groups, seqs, stats, metrics)`，**每一步恰好调用一次**，跳过的步也要调用（传 `seqs=[]`、`metrics={}`）。它会合并统计、计算训练时的 success_rate / pass@k（k = `group_size`）、打印 sanity 警告，并加上 `train/` 前缀写入日志；跳过的步会额外记录 `skipped: 1`。

一条记录中各类指标的"时间点"不同：

| 指标 | 描述的是 |
|---|---|
| `success_rate`、`pass@k`、`mean_reward` | 第 t 步**更新之前**采到的数据 |
| `loss`、`grad_norm`、`importance_weight_*` | 第 t 步**这次更新**本身 |
| 序列统计（`num_sequences`、`entropy_proxy` 等） | 第 t 步更新所用的 batch |
| `eval/*`（如果第 t 步有评估） | 第 t 步**更新之后**的策略 |

因此同一步的 train 曲线与 eval 曲线相差一次更新（跳过的步除外）。

### ⑥ 评估：`evaluate_now`

```
evaluate_now(policy) = evaluate_policy(policy, 前 cfg.eval_tasks 个验证任务, env_factory,
                                       k=cfg.eval_k, agent_cls=cfg 指定的 agent 类)
```

- 返回原始指标字典，**不要**自己加 `eval/` 前缀或写日志，这些由 `record_evaluation` 负责；
- 在 `train()` 中，`eval_due(step)` 为真时评估**当前**策略：更新过的步是刚刷新的策略，跳过的步是这一步采样用的同一个策略；
- `eval_due` 按绝对步数计算，断点续训后仍保持每 `eval_every` 步评估一次；
- step 0 的基线评估由入口脚本完成，`train()` 只负责周期性评估。

---

## 4. 组合成 `train()`

```
policy = 初始快照 step_000000 的策略
for step in 1 … cfg.num_steps:
    groups = await collect_batch(policy)                 ①
    seqs, stats = build_batch(groups)                    ②
    if seqs:
        metrics = await update(seqs)                     ③
        policy = 新快照 step_{step:06d} 的策略            ④
    else:
        metrics = {}                                     （跳过：不更新、不建快照）
    record_training(step, groups, seqs, stats, metrics)  ⑤  每步恰好一次
    if eval_due(step):
        record_evaluation(step, await evaluate_now(policy))   ⑥
```

`train()` 的几条硬性要求汇总：

| 要求 | 对应环节 |
|---|---|
| 训练前建 `step_000000` 快照，每次更新后建 `step_{t:06d}` 快照 | ④ |
| batch 没有任何序列时跳过：不更新、不建快照，下一步沿用同一策略 | ③④ |
| 每一步（含跳过的步）恰好调用一次 `record_training` | ⑤ |
| `eval_due(step)` 为真时评估当前策略，跳过的步也要评估 | ⑥ |
| 不做 step 0 的基线评估（入口脚本已做） | ⑥ |

---

## 5. 配置项在链上的去向

| 配置 | 用在哪一环 | p2a | p2b |
|---|---|---|---|
| `tasks_per_batch` | ① `collect_batch` | 8 | 8 |
| `group_size` | ① `collect_group`，⑤ pass@k | 1 | 8 |
| `algo` | ② `groups_to_batch` | reinforce | grpo |
| `loss_fn` | ③ `train_step` | importance_sampling | ppo |
| `lr`、`grad_clip_norm` | ③ `submit_update` | 1e-5、1.0 | 1e-5、1.0 |
| `num_steps` | `train()` 循环 | 50 | 50 |
| `eval_every` | ⑥ `eval_due` | 10 | 10 |
| `eval_tasks`、`eval_k` | ⑥ `evaluate_now` | 50、4 | 50、4 |
| `seed` | ① `self.rng` | 11768 | 11768 |

p41 与 p2b 只差 `group_size = 2`。

---

## 6. 总结

Part 2 就是把 Part 1 的数据管线接进一个循环：

- **采样（①）和加工（②）在本地完成**：并发运行回合，计算优势，拼接成带 mask 的序列；
- **更新（③）和快照（④）交给后端**：一次前向反向加一次优化器步，然后保存新权重；
- **记录（⑤）和评估（⑥）由给定代码负责格式**，你只需在正确的时机、以正确的次数调用它们。

每一步都用最新的权重采样，这就是同步 RL，也是 Part 3 异步 RL 要改造的对象（见 [async.md](async.md)）。
