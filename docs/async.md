# 同步与异步训练详解

> 配合 [overview.md](overview.md) 阅读。本文解释 Part 2 的同步训练和 Part 3 的异步训练有什么区别、异步和 batch 是什么关系、off-policy 从哪里来。示例数字以 [p2b](../configs/p2b_number_search_grpo.yaml) 和 [p3](../configs/p3_number_search_async.yaml) 的配置为准，时间线中的秒数仅为示意，不是实测值。

---

## 1. 本质：采样和训练能不能同时进行

RL 训练的每一步都包含两件事：

1. **采样（rollout）**：用当前模型和环境交互，得到一批轨迹和奖励。这一步在 vLLM 推理引擎上完成。
2. **训练（update）**：用这批轨迹计算 loss、反向传播、更新 LoRA 权重。这一步在 Megatron 训练引擎上完成。

同步和异步的区别，就在于这两件事是**轮流做**还是**同时做**。

---

## 2. 同步（SyncTrainer，Part 2）

```
policy = v0
for step in 1..50:
    groups = 用 policy 采样 8 个任务 × 8 次 rollout   # 等全部跑完
    seqs   = 计算优势、拼接序列
    训练一步                                          # 等训练完
    policy = 新权重的快照 v_step                       # 刷新
```

```
时间 ──────────────────────────────────────────────────────▶
采样:  [用 v0 采样 batch1]                [用 v1 采样 batch2]
训练:                      [训练→得到 v1]                     [训练→得到 v2]
```

**优点：严格 on-policy。** 训练用的数据一定是刚刚那版权重采出来的，所以采样时记录的 logprob 和训练时重新计算的 logprob 基本相同，重要性比率 ρ = π_θ / π_old ≈ 1。

> ρ 不严格等于 1，是因为采样用 vLLM、训练用 Megatron，两个引擎的数值实现（kernel、精度）略有差异。作业说同步 run 的 `importance_weight_mean` 应该非常接近 1：如果偏离很大，通常说明 logprob 和 token 错位了（Part 1 的 off-by-one）。

**缺点：总有一方在闲着。**

- 采样时，训练引擎在等；训练时，采样引擎在等。
- 更严重的是**长尾**：一个 batch 的 64 条 rollout 必须全部完成才能开始训练。只要有一条特别慢（步数多、生成长），所有人都得等它。

---

## 3. 异步（AsyncGRPOTrainer，Part 3）

把采样和训练拆成两种并行运行的角色，中间用一个队列连接：

```
 worker 1 ─┐
 worker 2 ─┤
   ...     ├──put──▶ [ 有界队列 (容量 16) ] ──get──▶ learner
 worker 8 ─┘                                        │
     ▲                                              │ 每凑够 8 组：训练一步
     └──────────── 发布新版本 policy v+1 ◀───────────┘
```

- **Worker（生产者）**：不停循环："取一个任务 → 拿到当前最新的 policy 和版本号 → 跑一组 8 条 rollout → 放进队列"。
- **Learner（消费者）**：不停循环："从队列取 8 组 → 训练一步 → 发布新版本 → 继续取"。

```
时间 ──────────────────────────────────────────────────────▶
采样:  [v0 采样][v0 采样][v0/v1 采样][v1 采样][v1/v2 采样]…   ← 一直不停
训练:              [训练→v1]          [训练→v2]          …   ← 也一直不停
```

**优点：快。** 两边都不闲着，一条慢 rollout 也不会卡住整个训练（其它 worker 的组先到就先用）。Part 3 要求量化 p3 相对 p2b 的每步耗时加速比，就是为了验证这一点。

**代价：数据可能是旧的（off-policy，即陈旧度 staleness）。** learner 训练时，队列里的某一组可能是几个版本之前的模型采的。例如 learner 当前是 v5，而这组是 v3 采的，陈旧度就是 5 − 3 = 2。

由此带来两个问题，对应两种对策：

| 问题 | 对策 | 对应代码 |
|---|---|---|
| 太旧的数据和当前策略差别太大，没法用 | 丢弃：陈旧度 > `max_staleness` 的组直接扔掉 | `collect_fresh_batch`，计入 `groups_dropped_stale` |
| 不太旧的数据，分布仍有偏差 | **重要性加权**：用 ρ 校正，PPO 再把 ρ 裁剪到 [1−ε, 1+ε] 防止爆炸 | `loss_fn: ppo` |

因为有这个偏差，异步 run 的 ρ 不再近似恒等于 1，而是在 1 附近分布。所以 Part 3 要求同时比较 mean、min、max、std：mean 接近 1 并不代表没问题，个别 token 的 ρ 可能很大（重尾）。

---

## 4. 以两张卡为例

作业里 Number Search 的两个服务器正好都是 2 张 H100，但分工不同：

| | 同步 `serve_number_search_sync.py` | 异步 `serve_number_search_async.py` |
|---|---|---|
| GPU 0 | 采样 **和** 训练（轮流） | 只做训练（Megatron） |
| GPU 1 | 采样 **和** 训练（轮流） | 只做采样（vLLM） |
| 配置 | `colocate_all: True` | `colocate_all: False` |

### 4.1 同步：两张卡一起"换班"

每张卡上同时装着 vLLM 和 Megatron，但同一时刻只有一个在工作，另一个处于"睡眠"状态（权重和 KV cache 移出显存，腾出空间）。

```
时间(s)   0        20   25        45   50        70   75
GPU 0:    [ 采样  ][训练][  采样   ][训练][  采样   ][训练]
GPU 1:    [ 采样  ][训练][  采样   ][训练][  采样   ][训练]
          └ v0 采 ┘ →v1 └  v1 采  ┘ →v2 └  v2 采  ┘ →v3
```

一步的过程：

1. **采样阶段**：两张卡各运行一个 vLLM 引擎，64 条 rollout 分到两张卡上并行生成。
2. **等待最慢的那条**：大部分 rollout 可能 10 秒就结束了，但只要有一条需要 20 秒，所有人都得等。这期间卡上只剩零星几条在跑，**利用率很低**。
3. **切换**：vLLM 睡眠，Megatron 唤醒。
4. **训练阶段**：两张卡做数据并行，各计算一半 batch 的梯度，同步后更新 LoRA。
5. **切换回来**：新的 LoRA 权重交给 vLLM，vLLM 唤醒，下一步用 v1 采样。

采样时两张卡都在采，训练时两张卡都在训，卡本身不会"被浪费"，这正是共置的目的。真正的浪费在于**等长尾**。

> 同步必须共置。如果让 0 号卡只训练、1 号卡只采样，采样时 0 号卡闲着，训练时 1 号卡闲着，每张卡都有一半时间空转。

### 4.2 异步：两张卡各干各的

```
时间(s)   0     10    20    30    40    50    60    70
GPU 1     [v0 采][v0 采][v0/v1采][v1 采][v1/v2采][v2 采]…   ← 一直在采
(采样):          ↓ 组进队列 ↓        ↓              ↓
GPU 0             [训练→v1]     [训练→v2]     [训练→v3]…   ← 有数据就训
(训练):                 ↓             ↓
                  新 LoRA 推给 GPU 1   新 LoRA 推给 GPU 1
```

- **GPU 1（采样）一直在生成。** 8 个 worker 各自不停地取任务、跑一组、放进队列。某条 rollout 慢不影响其它 worker，空出的位置马上有新的 rollout 补上，vLLM 始终满载。
- **GPU 0（训练）凑够 8 组就训一步。** 训完把新的 LoRA 快照发布出去（版本号 +1），GPU 1 上的 worker 下次取 policy 时就会拿到新版本。

关键变化在于：GPU 0 在训练 v1→v2 的期间，GPU 1 **没有停**，仍在用 v1（甚至 v0）采样。这些组之后进入下一个 batch：

```
learner 当前版本 = 2
一组 policy_version = 0 的数据  → 陈旧度 2，≤ max_staleness(2)，保留，用 ρ 校正
一组 policy_version = 1 的数据  → 陈旧度 1，保留
如果有一组比 v0 还旧             → 陈旧度 3 > 2，丢弃
```

异步之所以必须分卡，是因为两件事要同时进行，不能共用一张卡轮流睡眠。

### 4.3 并排对比

| | 同步（2 卡共置） | 异步（1 训练 + 1 采样） |
|---|---|---|
| 某一时刻 GPU 0 在做什么 | 和 GPU 1 做同一件事 | 训练 |
| 某一时刻 GPU 1 在做什么 | 和 GPU 0 做同一件事 | 采样 |
| 采样用几张卡 | 2 张（只占部分时间） | 1 张（一直在跑） |
| 训练用几张卡 | 2 张（只占部分时间） | 1 张（有数据就训） |
| 闲置来源 | 等最慢的 rollout；切换开销 | 训练卡偶尔等队列凑够 8 组 |
| 数据新鲜度 | 永远是最新版本 | 可能落后 0～`max_staleness` 个版本 |
| ρ（重要性比率） | ≈ 1 | 围绕 1 分布，可能有重尾 |
| 对应 run | p2a、p2b、p41 | p3、p43_iw、p43_noiw |

**类比：** 同步像一家只有两个员工的小店，两人先一起进货、再一起卖货，轮流进行；异步像一人专门进货、一人专门卖货，各干各的，但卖出去的货可能是前几批进的，不一定是最新的。

Part 5 的 TextCraft 是异步的放大版：4 张卡，1 张训练、3 张采样。因为 TextCraft 开了思维链，episode 很长，绝大部分时间都花在生成上，所以把更多的卡分给采样。

---

## 5. 异步与 batch：生产单位是 group，消费单位是 batch

```
1 个 group = 1 个任务 × group_size(8) 条 rollout          ← worker 每次生产一个
1 个 batch = groups_per_batch(8) 个 group = 64 条 rollout ← learner 每步消费一个
```

- **同步**：这两个粒度绑在一起。每步开始时用同一个版本一次性采出 8 个 group，凑成 batch，采完就训。
- **异步**：两者被拆开。
  - worker 只管一个一个地产出 group，放进队列，不关心 batch。
  - learner 从队列里一个一个取 group，凑满 8 个就算一个 batch，然后训练一步。

所以**一个异步 batch 里的 8 个 group，可能来自不同 worker、不同时间、不同版本的模型**。这是异步和同步在数据上的根本区别。

### 为什么版本号记在 group 上，而不是单条 rollout 或整个 batch 上

GRPO 的优势 `A_i = r_i − 组内平均奖励` 以组内平均为 baseline，这要求一组 8 条 rollout 都来自**同一个策略**。因此：

- **组内必须同一版本**：worker 开始一组之前先用 `active_policy()` 一次性捕获 `(policy, version)`，整组都用这一对。一组 rollout 要跑一段时间，如果中途 learner 发布了新版本，而 worker 在结束时才去读"当前版本号"，就会把 v3 采的数据错标成 v4，陈旧度就算错了。
- **优势在组内算完**：在 `generate_group` 中完成。
- **组与组之间版本可以不同**：所以每个 `GroupResult` 都带一个 `policy_version`，learner 凑 batch 时按组判断是否新鲜。

---

## 6. 一步步模拟（训练是瓶颈的情况）

配置（p3）：8 个 worker，队列容量 16，每个 batch 8 组，`max_staleness=2`。

**开始**：发布 v0，8 个 worker 各自用 v0 开始采一组。

**Step 1**

- 8 个 worker 陆续把 8 个 v0 组放进队列，然后立即用 v0 开始采下一组。
- learner 取出 8 组，凑成 batch 1，开始训练。
- **learner 训练期间 worker 不停**：继续产出 v0 组，队列涨到 16 个，满了。
- 队列满后 worker 卡在 `await queue.put()`，每个 worker 手里拿着一个采好但放不进去的组。这就是**背压**（backpressure）：生产者被迫停下来等消费者。
- 训练完成，发布 v1。

此时等待训练的有：队列中 16 个 + worker 手里 8 个 = **24 个组，全是 v0 采的**。

**Step 2**（learner 版本 1）：取 8 个 v0 组，陈旧度 1 − 0 = 1 ≤ 2，保留。训练后发布 v2。队列腾出位置，worker 手里的组放进去，接着用 **v1** 采新组。

**Step 3**（learner 版本 2）：再取 8 个 v0 组，陈旧度 2 ≤ 2，保留。训练后发布 v3。

**Step 4**（learner 版本 3）：剩下的 v0 组陈旧度 3 > 2，**丢弃**并计入 `groups_dropped_stale`；继续取，直到凑够 8 个 v1 及以后的组。

这就是 `collect_fresh_batch` 的逻辑：**一边取一边检查，旧的扔掉，凑够 `groups_per_batch` 个新鲜组才返回**。陈旧度在消费时由 learner 判断，而不是由 worker 判断。

---

## 7. 数据最多能有多旧

最坏情况下，排在 learner 前面的组最多有：

```
worker 手里的组 + 队列里的组 = num_workers + queue_capacity
```

learner 每步消耗 `groups_per_batch` 个，所以最旧的组大约要等：

```
(num_workers + queue_capacity) / groups_per_batch  步
```

| | 计算 | 结果 | max_staleness | 是否丢组 |
|---|---|---|---|---|
| p3 | (8 + 16) / 8 | 3 | 2 | 训练慢时可能丢 |
| p5 | (32 + 16) / 16 | 3 | 4 | 基本不丢 |

[p5 配置](../configs/p5_textcraft_grpo.yaml)的注释说的就是这一点：让这个比值小于 `max_staleness`，就不会浪费采好的数据。p3 的比值大于 `max_staleness`，所以 Part 4.2 要看 `groups_dropped_stale`。

**队列为什么要有界**：如果 worker 比 learner 快很多，无界队列会堆满旧数据，learner 拿到的大多过期被丢弃，白白浪费算力。有界队列满了以后，`await queue.put()` 让 worker 暂停等待，通过背压天然限制了数据的最大陈旧度。实现时要用 `await put()` / `await get()` 提供背压，而不是轮询 `full()` / `empty()`。

### 反过来：采样是瓶颈的情况

Number Search 和 TextCraft 更常见的情况是采样慢、训练快：

- learner 训完一步，队列里常常不足 8 组，它就卡在 `await queue.get()` 上等待。
- 组一产出几乎马上被取走，队列长期接近空，陈旧度多为 0 或 1，很少丢组。
- 此时异步的收益主要来自**不再等长尾**：一个 batch 不再需要"64 条 rollout 同时开始、等最慢那条结束"，而是先完成的 8 组先凑成 batch。慢的组会进入后面的某个 batch，但不会拖住整一步。

---

## 8. Off-policy：数据重复使用 vs 异步延迟

### 8.1 经典 PPO：一批数据训练多次

```
用 π_old 采一大批数据
for epoch in 1..K:                 # 比如 K = 4
    for minibatch in 切分(数据):    # 比如切成 4 份
        训练一步，权重更新           # 一共更新 K × 4 = 16 次
丢掉这批数据，用新策略重新采
```

- 第 1 次更新时，模型就是 π_old，数据是 on-policy 的。
- 从第 2 次开始，模型已经变了，但用的还是 π_old 采的数据，于是**变成 off-policy**。
- 所以需要重要性比率 ρ 校正，再用 clip 限制 ρ 不能偏离 1 太远，防止在旧数据上走得太远。

这里 off-policy 的来源是**重复使用**：数据不变，模型在变。

### 8.2 本框架：每个 batch 只训练一次

[`submit_update`](../scottyrl/tinker_adapter.py) 中，一个 batch 只对应**一次** `forward_backward` 加**一次** `optim_step`，没有 epoch，也没有 minibatch 循环，训练完这批数据就丢弃。所以这里的 batch 就是"一次梯度更新所用的全部数据"：

- 同步：`tasks_per_batch` × `group_size` = 8 × 8 = 64 条 rollout。
- 异步：`groups_per_batch` × `group_size` = 8 × 8 = 64 条 rollout。

那 off-policy 从哪里来？

- **同步**：数据由 v_t 采出，用 v_t 计算 loss（`forward_backward` 在 `optim_step` 之前执行），然后更新成 v_{t+1}。计算 ρ 时用的就是采样时那份权重，近似 on-policy。
- **异步**：数据由 v_{t−s} 采出，但 learner 已经是 v_t，用 v_t 计算 loss，数据和当前模型差了 s 个版本，是 off-policy 的。来源是**异步延迟**。

| | 采数据的策略 | 训练时的策略 | off-policy 的来源 |
|---|---|---|---|
| 经典 PPO 第 k 次更新 | π_old | π_old 更新 k−1 次后 | 数据被重复使用 |
| 本框架同步 | v_t | v_t | 无（≈ on-policy） |
| 本框架异步 | v_{t−s} | v_t | 采样与训练并行，数据产生后要排队，期间模型更新了 |

### 8.3 共同点：同一套校正

不管成因是什么，问题都是"数据来自 π_old，要更新的是 π_θ"，处理方式也一样：

```
ρ = exp(log π_θ − log π_old)               # log π_old 就是采样时记录的 completion_logprobs
loss = −min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)      # PPO
```

这也是 Part 1 要把采样 logprob 精确存进 `TokenSequence.logprobs` 的原因：它就是公式中的 log π_old。

### 8.4 两种思路的本质：都是让每次采样换来更多训练进度

| | 经典 PPO 重复使用 | 本框架异步 |
|---|---|---|
| 想解决的问题 | 采样很贵，一批数据只更新一次太浪费 | 采样和训练轮流进行，GPU 有空闲 |
| 做法 | 同一批数据更新多次 | 采样和训练同时进行 |
| 每条数据被训练几次 | K 次 | 1 次（过旧则 0 次） |
| 代价 | 数据相对模型越来越旧 | 数据排队期间模型更新了 |
| off-policy 程度由谁控制 | epoch 数 K | `max_staleness` |
| 校正方法 | 重要性比率 + PPO clip | 重要性比率 + PPO clip |

**off-policy 是一种状态（训练用的数据不是当前模型采的），重复使用和异步是导致它的两种不同原因。本作业只涉及后者。**

---

## 9. Part 4.3 实验在验证什么

两个 run 都把 `max_staleness` 调到 4，让数据更旧：

- `p43_number_search_async_stale4_iw`：用 PPO，有 ρ 校正和裁剪。
- `p43_number_search_async_stale4_noiw`：用 cross_entropy，loss = −A·log π_θ，**完全忽略** π_old，把旧数据当作 on-policy 数据使用。

对比两者的曲线，就能看到在数据更陈旧时，重要性加权到底起了多大作用，以及不做校正会不会让训练变差或变得不稳定。

---

## 10. 对应到要实现的函数

| 函数 | 粒度 | 做什么 |
|---|---|---|
| `generate_group` | group | 用传入的 `(policy, version)` 跑一组 rollout，组内算 GRPO 优势，返回带版本号的 `GroupResult`（退化组 `sequences=[]` 但保留 `rewards`） |
| `rollout_worker` | group | 循环：取任务 id → 捕获 `active_policy()` → `generate_group` → `await queue.put` |
| `collect_fresh_batch` | group → batch | 循环 `await queue.get`，过旧的丢弃并计数，凑够 `groups_per_batch` 个返回 |
| `update` | batch | 按顺序拼接 batch 中所有组的序列，调用 `session.train_step` |
| `evaluate_now` | — | 用模块级 `evaluate_policy` 评估 |
| `train` | 整体 | 发布 v0 → 启动 workers → 循环"凑 batch → 更新 → 发布 v+1" → 后台评估 → 结束或出错时取消所有任务 |

### 实现中容易困惑的几点

1. **为什么要先捕获 `(policy, version)` 这一对**：见第 5 节。整组必须用同一个版本，且版本号必须是采样开始时的那个。
2. **为什么队列要有界**：见第 7 节。通过背压限制数据陈旧度，避免堆积大量会被丢弃的旧数据。
3. **为什么评估要放到后台**：评估要跑 50 个任务 × 4 次，比较慢。如果 learner 停下来等评估，异步的优势就没了。所以要捕获当时的 `(policy, version)`，用 `asyncio.create_task` 调度 `run_evaluation`，训练继续进行。
4. **为什么异常处理要求这么严格**：如果某个 worker 崩溃，learner 可能永远卡在 `await queue.get()`，等一个不会到来的结果，程序挂起而 GPU 还在计费。因此需要同时监视队列和 worker / 评估任务（例如 `asyncio.wait(..., return_when=FIRST_COMPLETED)`）。任何一方出错，都要取消并 await 所有 worker 和评估任务，然后重新抛出**原始异常对象**，不要包成 `ExceptionGroup` 或新的 `RuntimeError`。

---

## 11. 总结

- **同步** = 采完再训、训完再采：数据新鲜、实现简单，但 GPU 有空闲，尤其要等长尾 rollout。一个 batch 由同一个版本一次性采完。
- **异步** = 边采边训：速度快，但数据可能是旧的。worker 以 group 为单位、用各自当时的版本持续生产，learner 从队列中挑新鲜的 group 凑成 batch，所以同一个 batch 里混有不同版本的数据。需要按组检查陈旧度、丢弃过旧的组，并用重要性加权校正。
