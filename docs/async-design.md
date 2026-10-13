# Part 3：异步流水线的设计与实现

> 配合 [async.md](async.md)（同步与异步的概念、group 与 batch、陈旧度、off-policy）和 [part0.md 第 7 节](part0.md)（为什么框架到处使用 asyncio）阅读。本文分三部分：
>
> - **A. asyncio 基础**：Part 3 用到的 `async` / `await` 工具；
> - **B. 架构设计**：按文件顺序梳理 [async_grpo.py](../scottyrl/train/async_grpo.py)，分析关键设计，并用例子推演；
> - **C. 实现**：每个学生方法用什么工具、怎么写，附参考实现。
>
> C 部分的参考实现曾在 scratchpad 中以"在内存中替换类方法"的方式验证：5 个异步相关的公开测试全部通过，另外构造的"评估失败"场景能在第 1 次更新后立即停止并抛出原始异常。源码未被修改。若参考这些写法，请在 `AI_USAGE.md` 中说明。

---

# A. asyncio 基础

## 1. 三个基本概念

| 概念 | 是什么 |
|---|---|
| **协程函数** | 用 `async def` 定义的函数 |
| **协程对象** | 调用协程函数得到的东西。**调用时函数体并不执行**，只返回一个"待执行的任务描述" |
| **事件循环** | 由 `asyncio.run(...)` 启动的调度器，负责运行协程并在它们之间切换 |

```python
async def sample(name, sec):
    print(f"{name} 开始")
    await asyncio.sleep(sec)      # 模拟等待网络（比如等 vLLM 生成）
    print(f"{name} 结束")
    return name

c = sample("X", 1)
print(type(c).__name__)           # 输出: coroutine  —— 什么都没打印，函数体没执行
```

要让它真正执行，必须 `await` 它，或者把它包装成任务（第 2 节）。

**`await` 的含义**："在这里等结果；等待期间把控制权交还给事件循环，让别的协程先运行。"由此得出最重要的性质：

> **只有在 `await` 处才可能切换协程。两个 `await` 之间的代码会一口气执行完，不会被任何其它协程打断。**

补充：`await` 是"**可能**"的切换点而不是"一定"。例如 `await queue.put(x)` 在队列没满时直接放入并返回，不会暂停；只有真正需要等待（队列满、网络未返回）时才让出控制权。

## 2. 让多件事同时进行的三种方式

以下输出均为实际运行结果（计时从 0 开始）。

**依次 `await`：一件做完再做下一件**

```python
await sample("A", 1)
await sample("B", 1)
```
```
[0.0s] A 开始
[1.0s] A 结束
[1.0s] B 开始
[2.0s] B 结束          ← 共 2 秒
```

**`asyncio.gather`：同时运行一批，等全部完成**

```python
results = await asyncio.gather(sample("A", 1), sample("B", 1))
```
```
[0.0s] A 开始
[0.0s] B 开始
[1.0s] A 结束
[1.0s] B 结束
[1.0s] gather 返回 ['A', 'B']     ← 共 1 秒；结果顺序与传入顺序一致
```

**`asyncio.create_task`：放到后台运行，自己马上继续**

```python
t = asyncio.create_task(sample("后台", 1))    # 立即返回一个 Task，不等待
print("创建完立刻继续，task.done() =", t.done())
await asyncio.sleep(0.5)
print("主协程做别的事")
r = await t                                  # 需要结果时再等它
print("await task →", r)
```
```
[0.0s] 创建完立刻继续，task.done() = False
[0.0s] 后台 开始
[0.5s] 主协程做别的事
[1.0s] 后台 结束
[1.0s] await task → 后台
```

| | `gather` | `create_task` |
|---|---|---|
| 调用后 | 等待全部完成 | 立刻返回 |
| 得到 | 所有结果的列表 | 一个 Task 句柄 |
| 适合 | 一批事情一起开始、一起结束 | 长期在后台运行的工作 |
| 作业中 | `collect_group`、`collect_batch` | rollout worker、后台评估 |

**Task 的常用方法**

| 方法 | 作用 |
|---|---|
| `task.done()` | 是否已经结束（正常完成、出错、被取消都算） |
| `task.result()` | 取结果；任务出错时抛出那个异常 |
| `task.exception()` | 取异常（没有异常返回 `None`） |
| `task.cancel()` | 请求取消（第 4 节） |
| `await task` | 等它结束并拿到结果；出错时在这里抛出它的异常 |

## 3. `asyncio.Queue`：有界队列与背压

```python
q = asyncio.Queue(maxsize=2)

async def producer():
    for i in range(4):
        await q.put(i)            # 队列满时在这里等待

async def consumer():
    await asyncio.sleep(1)        # 消费者晚 1 秒才开始
    for _ in range(4):
        x = await q.get()         # 队列空时在这里等待
        await asyncio.sleep(0.5)

await asyncio.gather(producer(), consumer())
```
```
[0.0s] 生产者 put 0 完成
[0.0s] 生产者 put 1 完成
[0.0s] 生产者 put 2 …          ← 队列已满（容量 2），生产者在这里停住
[1.0s] 消费者 get 0
[1.0s] 生产者 put 2 完成       ← 消费者取走一个，生产者才能继续
[1.0s] 生产者 put 3 …
[1.5s] 消费者 get 1
[1.5s] 生产者 put 3 完成
```

这就是**背压**：生产者自动被消费者的速度"拖住"，不需要写任何轮询代码。所以 docstring 要求用 `await put()` / `await get()`，而不是检查 `full()` / `empty()`。

## 4. 取消任务

```python
async def worker():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("在 await 处收到 CancelledError，做清理")
        raise                                  # 清理完要重新抛出

t = asyncio.create_task(worker())
t.cancel()                                     # 只是"请求"取消
try:
    await t                                    # 等它真正结束
except asyncio.CancelledError:
    print("await 被取消的任务 → CancelledError")
print(t.cancelled())                           # True
```

1. `cancel()` 只是**发出请求**，任务在它**下一次 `await` 处**收到 `CancelledError` 后才真正结束；
2. 取消之后**一定要 `await`** 这个任务，确保它真正退出；
3. `await` 被取消的任务会抛出 `CancelledError`。一次等待多个被取消的任务，可用 `await asyncio.gather(*tasks, return_exceptions=True)`，它把异常当作结果收集而不抛出；
4. 任务内部捕获 `CancelledError` 做完清理后应**重新抛出**，不要吞掉。

## 5. `asyncio.wait`：谁先完成就处理谁

```python
q = asyncio.Queue()                            # 一直是空的

async def bad_worker():
    await asyncio.sleep(0.2)
    raise ValueError("worker 挂了")

w = asyncio.create_task(bad_worker())
get_task = asyncio.create_task(q.get())        # 把"从队列取数据"也包装成任务

done, pending = await asyncio.wait({get_task, w}, return_when=asyncio.FIRST_COMPLETED)
print(w in done, repr(w.exception()))          # True ValueError('worker 挂了')
get_task.cancel()                              # 收拾没完成的那个
```

如果消费者只写 `await q.get()`，生产者挂掉后队列永远是空的，消费者就会**永远卡住**。用 `asyncio.wait(..., FIRST_COMPLETED)` 同时等"队列有数据"和"生产者结束"，哪个先发生就处理哪个。这就是 Part 3 中"worker 失败不能让 learner 死锁"的实现思路。

## 6. 异常如何传播

| 场景 | 行为 |
|---|---|
| `await 协程` 或 `await task` | 内部异常在 `await` 处原样抛出 |
| 后台 Task 出错，但没人 `await` 它 | 异常**不会**自动传到别处，只会在结束时打印 "Task exception was never retrieved"，必须主动检查 |
| `gather` 中某一个出错 | 第一个异常被抛出，但**其它的不会被取消，仍在运行**（实测：抛出 `boom` 时，另一个任务 `slow` 仍在运行） |
| `asyncio.TaskGroup` 中某一个出错 | 取消其余任务，但把异常包成 **`ExceptionGroup`** 抛出 |

Part 3 要求重新抛出**原始的**异常对象，所以不宜直接使用 `TaskGroup`；更直接的做法是 `create_task` + `asyncio.wait`，自己管理取消。

## 7. 常见坑

| 坑 | 现象 | 正确做法 |
|---|---|---|
| 调用协程函数忘了 `await` | 函数没执行，警告 "coroutine was never awaited" | `await f()` 或 `create_task(f())` |
| 在协程里用 `time.sleep()` | 整个事件循环被卡住 | `await asyncio.sleep()` |
| `create_task` 后不保存 Task | 可能被垃圾回收；无法取消或检查异常 | 把 Task 存进列表 |
| `cancel()` 后不 `await` | 任务没真正退出，可能有警告 | 取消后 `await`（或 `gather(..., return_exceptions=True)`） |
| 只 `await q.get()` 等数据 | 生产者挂了就永远卡住 | 用 `asyncio.wait` 同时监视生产者 |
| 轮询队列（`while q.empty(): sleep`） | 浪费资源、反应慢 | `await q.get()` / `await q.put()` |
| 捕获 `CancelledError` 后不重新抛出 | 关闭流程出问题 | 清理后 `raise` |
| 在 `await` 前后读取共享状态并假设没变 | `await` 期间别的协程可能已修改它 | 在 `await` 之前一次性捕获需要的值 |

---

# B. 异步流水线的架构

## 8. 总体架构

### 8.1 系统的四个部分和五条数据流

```
                 ┌──────── 权威策略状态 ────────┐
                 │  self.policy = π2            │
                 │  self.version = 2            │
                 └──▲────────────────────────┬──┘
       ⑤ 发布新版本 │                        │ ② 捕获 (策略, 版本)
   publish_policy  │                        │ active_policy()
                   │                        ▼
┌──────────────────┴──┐            ┌──────────────────────────┐   ① 取任务 id
│      learner        │            │  worker W1、W2、……        │◀──────────── 共享任务流
│  （train 主循环）    │            │  跑一组 rollout，打包成    │              cycle([A,B,C,D,E])
│  凑 batch → 训练     │            │  GroupResult(版本号)      │
│  → 发布 → 记录       │            └────────────┬─────────────┘
└──┬───────────▲──────┘                         │ ③ put（队列满就等）
   │           │ ④ get（检查陈旧度）             ▼
   │           └──────────────── [ 有界队列 asyncio.Queue ]
   │
   │ 到评估步时：捕获 (策略, 版本)，创建后台任务
   ▼
 评估任务（后台运行，不阻塞 learner）
```

| 编号 | 谁 → 谁 | 传递什么 |
|---|---|---|
| ① | 任务流 → worker | 下一个任务 id |
| ② | 权威状态 → worker | 当前的 (策略, 版本) 这一对 |
| ③ | worker → 队列 | 一个 `GroupResult`（带版本号） |
| ④ | 队列 → learner | 一个 `GroupResult`，learner 检查是否过旧 |
| ⑤ | learner → 权威状态 | 训练后的新 (策略, 版本) |

注意：任务流只连到 **worker**。learner 从不取任务 id，只从队列取已经采好的组。

### 8.2 组件一览

| 组件 | 实现方式 | 职责 |
|---|---|---|
| 权威策略状态 | `self.policy`、`self.version` | 所有人读取"当前策略"的唯一来源 |
| 共享任务流 | 所有 worker 共用的 `itertools.cycle` 迭代器 | 把任务分配给各个 worker |
| rollout worker | `num_workers` 个常驻 asyncio 任务，运行 `rollout_worker` | 持续生产 `GroupResult` |
| 有界队列 | `asyncio.Queue(maxsize=queue_capacity)` | 缓冲 + 背压 |
| learner | `train()` 的主循环 | 消费、训练、发布新版本 |
| 评估任务 | 每次评估一个后台任务，运行 `run_evaluation` | 不阻塞训练地评估 |
| 统计 | `stats` 字典 | 已训练步数、已消费组数、因过旧丢弃的组数 |

所有任务都运行在同一个事件循环、同一个线程中，只在 `await` 处切换：

```
train()（learner，主协程）
 ├─ worker 任务 × num_workers      ← create_task，常驻，永远循环
 │     next(task_iter) → active_policy() → generate_group → await queue.put
 ├─ 评估任务 × 若干                 ← create_task，到评估步时创建，跑完就结束
 └─ 主循环本身
       await 等数据（同时监视 worker 和评估）→ update → publish → record → 调度评估
```

## 9. 按文件顺序梳理 `async_grpo.py`

文件的组织思路：

```
模块说明 → 导入 → 数据载体（GroupResult） → 类：
    初始化（权威状态）
    → 给定的基础设施（读写状态、发布、记录、评估）
    → 学生实现的部件（自底向上：一组 → worker → 取 batch → 更新 → 评估）
    → 总装配 train()
```

### 9.1 模块说明（L1–24）

docstring 是整个架构的文字版：`rollout workers -> bounded queue -> learner`。它强调：rollout 可能来自旧快照；每个结果记录生成它的版本，learner 据此拒绝过旧数据；设置了 `maxsize` 的 `asyncio.Queue` 是有界的，`await put` 让快的生产者暂停（背压），`await get` 让消费者在队列空时等待。

### 9.2 导入（L26–50）：各个部件的"原料"

| 导入 | 用在架构的哪里 |
|---|---|
| `asyncio` | 队列、后台任务、`wait`、`gather`：整个并发骨架 |
| `itertools` | `itertools.cycle`：共享任务流 |
| `Iterator`（类型） | `rollout_worker` 参数 `task_iter: Iterator[str]` 的类型标注 |
| `evaluate_policy` | 评估任务的内部实现（`evaluate_now` 调用） |
| `collect_group`、`groups_to_batch` | worker 内部：跑一组 rollout、加工成训练序列 |
| `rollout_phase` | 给评估回合打上 eval 标记（`run_evaluation` 使用） |
| `TinkerSession` | 与后端通信：训练（`train_step`）、建快照（`new_sampling_client`） |
| `pass_at_k`、`sanity_warnings`、`sequence_batch_stats` | learner 的记录环节（`record_training` 使用） |
| `TinkerPolicy`、`load_chat_template` | 构造采样策略（`make_policy` 使用） |

标着 `# noqa: F401 (student dependency)` 的是预先为你导入、给定代码本身没用到的名字。`evaluate_policy` 和 `collect_group` 要直接使用这里导入的名字，因为评分时会在模块层面替换它们。

### 9.3 `GroupResult`（L53–66）：队列里流动的数据

```python
@dataclass
class GroupResult:
    task_id: str
    policy_version: int                  # 采样时的版本 → learner 据此判断陈旧度
    sequences: list[TokenSequence]       # 训练数据；退化组为 []
    rewards: list[float]                 # 每条 rollout 的奖励（退化组也保留）
```

它就是图中箭头 ③④ 上传递的东西，也是 worker 与 learner 之间唯一的数据接口。

### 9.4 `__init__`（L72–102）：建立权威策略状态

```python
self.tasks_by_id = {task.id: task for task in train_tasks}   # 给 generate_group 按 id 查任务
self.policy = None                                          # ┐ 权威策略状态
self.version = 0                                            # ┘ （初始时还没有策略）
```

- 先检查 `queue_capacity`、`num_workers`、`groups_per_batch` 必须大于 0，`train_tasks` 不能为空，配置错误在启动任何 GPU 工作之前就报错；
- `policy` 在 `train()` 调用 `publish_policy(0)` 之后才有值；
- **队列、任务流、worker 都不在这里创建**，而是在 `train()` 中创建，生命周期只限于一次训练运行。

### 9.5 给定的基础设施（L105–263）

**`make_policy`（L105）、`agent_cls`（L118）**：把采样客户端包装成 `TinkerPolicy`（供 `publish_policy` 使用）；按配置返回 Agent 类（供 worker 使用）。

**`active_policy`（L124）：箭头 ② 读**

```python
def active_policy(self):
    if self.policy is None:
        raise RuntimeError("publish_policy must run before collecting rollouts")
    return self.policy, self.version
```

成对返回，且没有 `await`，不会读到一半的状态。在 `publish_policy(0)` 之前调用会报错，提醒 `train()` 必须先发布版本 0 再启动 worker。

**`absolute_version`（L130）**：`self.version` 是本进程从 0 开始的计数，断点续训时加上之前进程的更新次数才是绝对版本。**只用于日志和 transcript**，计算陈旧度用相对版本。

**`eval_due`（L140）**：判断第 `step` 步是否需要评估，按绝对步数计算。

**`publish_policy`（L155）：箭头 ⑤ 写**

```python
sampling_client = await self.session.new_sampling_client(f"v{version}")   # 先建好快照（会让出）
policy = self.make_policy(sampling_client)
policy.version = self.absolute_version(version)                         # 打上版本号（写进 transcript）
self.policy, self.version = policy, version                             # 最后一次性替换（不会让出）
```

这是**唯一**修改权威状态的地方，只有 learner 调用。快照名为 `v0`、`v1`、……

**`record_training`（L172）：learner 的记录环节**

```python
num_degenerate = sum(1 for group in batch if not group.sequences)   # 以 sequences 是否为空判断退化组
rewards = [r for group in batch for r in group.rewards]             # mean_reward 包括退化组
"policy_version": self.absolute_version(self.version)               # 调用时的版本
```

这解释了为什么 `generate_group` 必须**总是**返回 `GroupResult`，并且退化组要保留 `rewards`、`sequences` 为空。每轮迭代恰好调用一次，在更新和发布**之后**。

**`record_evaluation`（L232）与 `run_evaluation`（L247）：评估任务**

```python
async def run_evaluation(self, policy, version, step):
    with rollout_phase("eval", step=...):          # 评估回合标为 eval，transcript 写到 step 对应的目录
        metrics = await self.evaluate_now(policy)
    self.record_evaluation(step, metrics, policy_version=version)
```

这就是要用 `create_task` 放到后台的协程。它接收**捕获时**的 `(policy, version)` 和调度时的 `step`，所以即使评估结束时策略已经前进了几版，记录仍然准确。

### 9.6 学生实现的部件（L266–365）：自底向上

| 方法 | 在架构中的角色 |
|---|---|
| `generate_group`（L266） | worker 内部的"干活"部分：查任务 → 跑一组 → 算 GRPO 优势 → 打包成带传入版本号的 `GroupResult` |
| `rollout_worker`（L297） | worker 本体：循环 ①取 id → ②捕获 → 生成 → ③`put` |
| `collect_fresh_batch`（L313） | 箭头 ④：反复 `get`，丢弃过旧的，凑满一个 batch |
| `update`（L334） | learner 的训练：拼接序列，调用 `session.train_step` |
| `evaluate_now`（L350） | 评估任务的内部实现，调用模块级 `evaluate_policy` |

`rollout_worker` 的两个参数正好是 worker 与外界的两个连接：

| 参数 | 对应图中 | 来源 |
|---|---|---|
| `results` | 箭头 ③ 的终点：有界队列 | `train()` 创建，所有 worker 共用同一个 |
| `task_iter` | 箭头 ① 的起点：共享任务流 | `train()` 中用 `itertools.cycle(...)` 创建，**所有 worker 共用同一个迭代器** |

`task_iter` 是**迭代器对象**而不是列表：每个 worker 调用 `next(task_iter)` 都从同一个迭代器中取走下一个 id，任务因此被自然分配、不会重复。如果为每个 worker 各建一个迭代器，它们会按同样的顺序处理同样的任务。`cycle` 保证取完一轮后自动从头开始。

### 9.7 `train`（L369）：总装配

```
train()
 ├─ publish_policy(0)                              → 权威状态有了第一个值 (π0, 0)
 ├─ queue = asyncio.Queue(maxsize=queue_capacity)  → 创建有界队列
 ├─ task_iter = itertools.cycle(任务 id)            → 创建共享任务流
 ├─ workers = [create_task(rollout_worker(queue, task_iter)) × num_workers]
 ├─ for step in 1..num_steps:                      → learner 主循环
 │     collect_fresh_batch（同时监视故障）          → 箭头 ④
 │     有序列：update → publish_policy(v+1)        → 训练 + 箭头 ⑤
 │     record_training                             → 记录
 │     eval_due：active_policy() → create_task(run_evaluation(...))
 ├─ 正常结束：取消并等待 worker，等待评估
 ├─ 出错：取消并等待所有任务，重新抛出原始异常
 └─ return {train_steps, groups_consumed, groups_dropped_stale}
```

### 9.8 对照表：图中元素在文件中的位置

| 图中元素 | 定义 / 创建位置 | 由谁读写 |
|---|---|---|
| 权威策略状态 | [`__init__` L101–102](../scottyrl/train/async_grpo.py#L101-L102) | 写：`publish_policy`；读：`active_policy` |
| ① 共享任务流 | `train()` 中创建，作为参数传给 [`rollout_worker` L300](../scottyrl/train/async_grpo.py#L300) | worker 调用 `next()` |
| ② 捕获 (策略, 版本) | [`active_policy` L124](../scottyrl/train/async_grpo.py#L124) | worker |
| worker | [`rollout_worker` L297](../scottyrl/train/async_grpo.py#L297) + [`generate_group` L266](../scottyrl/train/async_grpo.py#L266) | 由 `train()` 用 `create_task` 启动 |
| ③④ 上的数据 | [`GroupResult` L53](../scottyrl/train/async_grpo.py#L53) | worker 生产，learner 消费 |
| 有界队列 | `train()` 中创建 | worker `put`，learner `get` |
| ④ get + 陈旧度检查 | [`collect_fresh_batch` L313](../scottyrl/train/async_grpo.py#L313) | learner |
| learner 的训练 | [`update` L334](../scottyrl/train/async_grpo.py#L334) | learner |
| ⑤ 发布新版本 | [`publish_policy` L155](../scottyrl/train/async_grpo.py#L155) | learner |
| learner 的记录 | [`record_training` L172](../scottyrl/train/async_grpo.py#L172) | learner |
| 评估任务 | [`run_evaluation` L247](../scottyrl/train/async_grpo.py#L247) → [`evaluate_now` L350](../scottyrl/train/async_grpo.py#L350) → [`record_evaluation` L232](../scottyrl/train/async_grpo.py#L232) | 由 `train()` 用 `create_task` 启动 |
| learner 本身 | [`train` L369](../scottyrl/train/async_grpo.py#L369) 的主循环 | — |

## 10. 关键设计决策

### 10.1 单一权威状态 + 原子发布

- **先建快照，再替换**：保存快照需要 `await`。若先改 `self.version` 再等快照，等待期间 worker 会读到"新版本号 + 旧策略"；
- **两者在同一条语句中替换**：这条赋值中间没有 `await`，其它协程不可能看到改了一半的状态；
- **`active_policy()` 成对读取**：同样没有 `await`，拿到的策略和版本一定匹配。

这就是 docstring 所说的 "one authoritative policy state"。

### 10.2 每组捕获一次 (策略, 版本)

一组 rollout 必须**全程**使用同一个策略（GRPO 的组内 baseline 才有意义），并记录**开始时**的版本号。若在 `await` 之后再读 `self.version`，得到的可能是更新后的版本，陈旧度就会算错。

### 10.3 陈旧度在消费端检查

| 在哪里检查 | 问题 |
|---|---|
| worker 生产时 | 刚生产出来陈旧度几乎总是 0，检查没有意义；数据真正"变旧"是在队列里排队时 |
| learner 消费时 ✅ | 这时才知道"这组数据被拿来训练时，策略已经前进了多少版" |

```
陈旧度 = self.version − group.policy_version
陈旧度 > max_staleness  →  丢弃
```

计数统一由 learner 负责：`groups_dropped_stale` 是取出但被丢弃的组，`groups_consumed` 是被接受进入 batch 的组（**包括退化组**），一个组只落入其中一类。

### 10.4 有界队列与背压

1. **限制数据最多有多旧**：最坏情况下，排在 learner 前面的组最多有 `num_workers + queue_capacity` 个（队列里的，加上每个 worker 手里各一个），learner 每步消耗 `groups_per_batch` 个，所以最旧的数据大约落后 `(num_workers + queue_capacity) / groups_per_batch` 个版本；
2. **避免浪费**：队列无界时，worker 会不停生产，大量数据还没用到就过期。

| 配置 | workers | 队列容量 | 每批组数 | 最大落后版本数 | max_staleness | 是否可能丢组 |
|---|---|---|---|---|---|---|
| p3 | 8 | 16 | 8 | 3 | 2 | 训练慢时会丢 |
| p43 | 8 | 16 | 8 | 3 | 4 | 基本不丢 |
| p5 | 32 | 16 | 16 | 3 | 4 | 基本不丢 |

### 10.5 共享的循环任务流

所有 worker 共用**同一个**迭代器：每个 worker 的 `next()` 拿到不同的任务；用完一轮自动从头开始；`next()` 中间没有 `await`，在单线程的 asyncio 中天然安全，不需要加锁。

### 10.6 退化组同样入队、同样计入 batch

```
batch = 恰好 groups_per_batch 个新鲜组（退化组也算）
if batch 中没有任何序列:   跳过（不更新、不发布）
```

这与同步 trainer 一致：同步时每步固定采 `tasks_per_batch` 个任务，退化组只是不贡献序列。若异步时只收集有信号的组，两边的 `frac_degenerate_groups`、`mean_reward`、pass@k 含义就不同，Part 4.2 的对比也就失去意义。

### 10.7 "步数"与"版本"是两个概念

| | step（迭代序号） | version（策略版本） |
|---|---|---|
| 含义 | learner 循环到了第几轮 | 模型实际更新了多少次 |
| 每轮是否 +1 | 是，包括跳过的轮 | 只有真正更新时才 +1 |
| 日志中 | 每条记录的 step | `train/policy_version` |
| 快照名 | — | `v0`、`v1`、…… |

例如前两步都跳过：step 为 1、2，version 一直是 0，只有 `v0` 一个快照。同步 trainer 的快照按步数命名（`step_000005`），异步按版本命名（`v5`）。

### 10.8 评估在后台进行

```
if self.eval_due(step):
    policy, version = self.active_policy()                  # 捕获此刻的一对
    启动后台任务 run_evaluation(policy, version, step)      # 不等待，训练继续
```

- 评估要跑 50 个任务 × 4 次，耗时较长，learner 等待就失去了异步的意义；
- 评估结束时策略可能已前进几版，记录必须描述**被评估的那个**版本，`run_evaluation` 会把结果记在**调度它的那一步**并附上 `eval/policy_version`；
- `rollout_phase` 是 ContextVar，后台任务创建时复制一份上下文，评估回合被标为 eval，同时运行的训练回合不受影响；
- 评估与训练 rollout **共用同一个 vLLM**，会争抢推理资源；`evaluate_policy` 默认最多 64 个并发。

### 10.9 故障传播与关闭

| 情况 | 要求 |
|---|---|
| 正常结束 | 最后一轮完成后，取消并等待所有 worker（此时被取消不算错误），再等待所有评估任务完成 |
| worker 出错 | learner 不能一直卡在 `queue.get()` 上，必须及时发现并失败 |
| 评估出错 | 同样让训练失败，而且要**立即**失败，不能等到最后一轮 |
| 任何异常 | 取消并等待所有 worker 和评估任务，然后重新抛出**原始异常对象** |

- **死锁 = 烧钱**：worker 全挂而 learner 还在等，程序不会退出，GPU 却一直计费；
- **要原始异常**：不能包成 `ExceptionGroup` 或新的 `RuntimeError`；
- **要等待被取消的任务**：只 `cancel()` 不 `await`，任务可能没真正结束。

### 10.10 相对版本与绝对版本

| | 相对版本 `self.version` | 绝对版本 `absolute_version(v)` |
|---|---|---|
| 起点 | 本进程从 0 开始 | 加上之前进程完成的更新次数 |
| 用途 | **陈旧度计算**、`GroupResult.policy_version` | 日志、transcript、快照名 |

计算陈旧度时两边必须同一口径，所以流水线内部一律用相对版本，对外记录时才由给定代码换算成绝对版本。

### 10.11 返回的统计

| 键 | 含义 |
|---|---|
| `train_steps` | learner 的迭代次数，**包括跳过的** |
| `groups_consumed` | 被接受进入 batch 的组数，包括退化组 |
| `groups_dropped_stale` | 因过旧被丢弃的组数 |

入口脚本把它们以 `summary/*` 写进日志最后一条记录，Part 4.2 要用到 `groups_dropped_stale`。

## 11. 一个 group 的生命周期：逐步看 `await`

规则回顾：**只有在 `await` 处才可能切换协程；两个 `await` 之间的代码一口气执行完。**

```
① worker: task_id = next(task_iter)
② worker: policy, version = self.active_policy()          ← 此刻捕获，比如 (π_v3, 3)
③ worker: result = await self.generate_group(task_id, policy, version)
             └─ await collect_group(...)                   ← 耗时最长：跑 8 条 rollout
             └─ groups_to_batch([group], "grpo")           ← 本地 CPU：算优势、拼序列
             └─ GroupResult(task_id, policy_version=3, sequences, rewards)
④ worker: await queue.put(result)                         ← 队列满就在这里等（背压）
          ……在队列里排队，期间 learner 可能发布了 v4、v5……
⑤ learner: result = await queue.get()
⑥ learner: 陈旧度 = self.version − 3
             > max_staleness → 丢弃，groups_dropped_stale += 1
             ≤ max_staleness → 接受，groups_consumed += 1，放进当前 batch
⑦ learner: 凑满 groups_per_batch 个 → update(batch) → publish_policy(version + 1)
```

### 11.1 逐步拆解

**① 取任务 id（没有 await）**：普通函数调用，不会让出控制权。所有 worker 共享同一个迭代器，`next()` 执行时其它协程不可能插进来，所以两个 worker 不会拿到同一个 id，**不需要加锁**（换成多线程就必须加锁）。

**② 捕获 (策略, 版本)（没有 await）**：一次性读出一致的一对。写的一方 `publish_policy` 也是在一条没有 `await` 的语句里同时替换两者，读写都不可能被打断，所以读到的要么全是旧的 `(π_v3, 3)`，要么全是新的 `(π_v4, 4)`。①② 连续执行，相当于一个"原子"的准备动作。

**③ `await self.generate_group(...)`（有 await，耗时最长）**：

```
generate_group
 └─ await collect_group(...)                   ← 让出控制权，等很久
      └─ await asyncio.gather(run_episode × 8)
           └─ await agent.step(obs)
                └─ await policy.complete(tokens)
                     └─ await sampling_client.sample_async(...)   ← 真正在等网络（vLLM 生成）
 └─ groups_to_batch([group], "grpo")           ← 没有 await，纯 CPU，一口气算完
 └─ GroupResult(task_id, policy_version=version, ...)
```

等待期间，其它 worker 也在采样（请求一起到达 vLLM，被批量生成）；learner 可能正在训练、**发布了 v4、v5**；评估任务可能在运行。这一组不受影响，因为采样一直使用局部变量 `policy`，记录的是局部变量 `version`，局部变量属于这个协程自己，别人改不了：

```python
GroupResult(..., policy_version=version)          # ✅ 用捕获的局部变量
GroupResult(..., policy_version=self.version)     # ❌ await 之后再读共享状态，可能已经是 5 了
```

`groups_to_batch` 是纯 CPU 计算，执行期间事件循环在等它；它只需毫秒级，所以不成问题。若采样出错，异常从 `sample_async` 一路向上抛出，worker 任务以"带着异常结束"的状态停止，等待 learner 发现。

**④ `await queue.put(result)`（有 await，可能让出）**：

| 队列状态 | 行为 |
|---|---|
| 没满 | 直接放入，**不让出**；同时唤醒在 `get()` 上等待的 learner（learner 要等这个 worker 下次让出时才真正运行） |
| 已满 | **让出**，worker 暂停，直到 learner 取走一个，即背压 |

排队期间 `result` 已经固定，`policy_version=3` 不会再变，但 learner 可能继续发布新版本，这组数据在"变旧"。放入后 worker 回到 ①，捕获**此刻**的 (策略, 版本)，可能已是 `(π_v5, 5)`。

**⑤ `await queue.get()`（有 await，可能让出）**：

| 队列状态 | 行为 |
|---|---|
| 有数据 | 直接取出，**不让出**；同时唤醒一个因队列满而卡在 `put()` 的 worker |
| 为空 | **让出**，learner 暂停，直到有 worker 放入数据 |

隐患：所有 worker 都出错停止后，队列永远为空，learner 会永远卡在这里。所以实际实现中 `get()` 要与"监视 worker"一起等待（第 14.4 节）。

**⑥ 判断陈旧度（没有 await）**：`self.version` 只会被 learner 自己在 ⑦ 中修改，而 learner 此刻正在执行这段代码，所以判断期间它一定不变。在这里判断而不是在 ④ 判断，是因为放入时数据还是新的，直到被取出才知道它被使用时已落后几版。

**⑦ `await update` → `await publish_policy`（有 await，这正是"异步"的意义）**：

```
同步 trainer：  learner 在 await update 时，没有任何 worker 存在 → 推理端空闲
异步 trainer：  learner 在 await update 时，worker 们在 await 采样 → 两边同时忙
```

`publish_policy` 先 `await` 建快照（等待期间 worker 读到的仍是旧的一对，完全一致），再一次性替换。顺序反过来，worker 就可能在等待期间读到"新版本号 + 旧策略"。

### 11.2 汇总表

| 步骤 | 有没有 await | 会不会让出 | 让出期间可能发生什么 | 为什么安全 |
|---|---|---|---|---|
| ① 取任务 id | 无 | 不会 | — | 不会被打断，无需加锁 |
| ② 捕获 (策略, 版本) | 无 | 不会 | — | 读写两侧都不会被打断，一定成对一致 |
| ③ `await collect_group` | 有 | **会，而且很久** | 其它 worker 采样；learner 训练、发布；评估运行 | 只使用局部变量 `policy`、`version` |
| ③ `groups_to_batch` | 无 | 不会 | — | 纯 CPU，很快 |
| ④ `await put` | 有 | 队列满时会 | learner 取数据、训练、发布 | `result` 已固定；等待即背压 |
| ⑤ `await get` | 有 | 队列空时会 | worker 生产数据 | 需配合故障监视，否则可能永远等待 |
| ⑥ 判断陈旧度 | 无 | 不会 | — | `self.version` 只由 learner 自己修改 |
| ⑦ `await update` | 有 | **会，而且很久** | worker 持续采样、入队 | 正是采样与训练重叠之处 |
| ⑦ `publish_policy` | 有 | 建快照时会 | worker 继续读到旧的一对 | 先建好快照，再一次性替换 |

### 11.3 两个协程如何交错（示意）

`▶` 表示正在执行，`…` 表示在 `await` 处等待：

```
时间 →
W1     ▶①②  …③ 采样（等 vLLM）……………  ▶③CPU ▶④put(没满，不让出) ▶①②(π_v4,4) …③ 采样 ……
learner …⑤ 等数据（队列空）……… ▶⑤⑥⑦  …⑦ update（等后端）………… ▶⑦publish→(π_v4,4)  ▶⑤ ……
                                     ↑                                  ↑
                         learner 在这段时间里一直在运行 / 等待      W1 在这之后才读到 v4
```

绝大部分时间两边都在 `…`（等网络），事件循环趁等待在两者之间切换，这就是并发；真正执行 Python 代码的 `▶` 片段都很短且互不重叠，所以共享状态不会被同时修改。

### 11.4 取消和出错时

| 情况 | 发生了什么 |
|---|---|
| 训练结束，learner 取消 worker | `CancelledError` 出现在 worker **当前所在的 `await`**：可能在 ③ 的采样中，也可能在 ④ 的 `put` 中。这组数据被放弃，worker 退出 |
| 采样出错 | 异常从 ③ 一路抛出，worker 任务以"带着异常结束"的状态停止 |
| learner 如何发现 worker 出错 | ⑤ 不只写 `await queue.get()`，而是用 `asyncio.wait({get 任务, *workers}, FIRST_COMPLETED)`：数据先到就正常处理，worker 先结束就抛出它的异常 |
| 评估出错 | 同样通过 `asyncio.wait` 监视，及时发现 |

**核心原则：在 `await` 之前把需要的共享状态捕获到局部变量；在 `await` 之后只使用局部变量。**

## 12. 完整例子：两个 worker 的推演

### 12.1 设定

| 参数 | 取值 |
|---|---|
| worker 数 | 2（W1、W2） |
| 队列容量 | 2 |
| 每个 batch 的组数 | 2 |
| `max_staleness` | 1 |
| `num_steps` | 3 |
| `eval_every` | 2（只在第 2 步评估） |
| 任务 | A、B、C、D、E，循环使用 |
| 采一组用时 | 2 个时间单位 |
| 训练一步 + 发布用时 | 5 个时间单位（训练比采样慢，learner 是瓶颈） |

记号：`C(v0)` 表示"任务 C 的一组，用 v0 版本采样"。时间单位为示意。

### 12.2 按时间推演

**t = 0：启动**

```
publish_policy(0)          → 权威状态 = (π0, 0)，保存快照 v0
创建队列、任务流，启动 W1、W2
W1: 取 A，捕获 (π0, 0)，开始采样
W2: 取 B，捕获 (π0, 0)，开始采样
learner 第 1 步: await queue.get() —— 队列为空，等待
```

**t = 2：第一批数据到达，开始第 1 步训练**

```
W1: A(v0) 采完 → put → learner 取走：陈旧度 0 − 0 = 0 ≤ 1，接受（consumed = 1）
W2: B(v0) 采完 → put → learner 取走：陈旧度 0，接受（consumed = 2）
learner: batch₁ = [A, B] → 开始训练（到 t = 7 结束）
W1: 取 C，捕获 (π0, 0)          ← 训练还没完成，当前仍是 v0
W2: 取 D，捕获 (π0, 0)
```

**t = 4 ~ 6：learner 在训练，队列被填满，出现背压**

```
t=4  W1: C(v0) → put → 队列 [C]         W1: 取 E，捕获 (π0, 0)
     W2: D(v0) → put → 队列 [C, D]（满） W2: 取 A，捕获 (π0, 0)    ← 任务流转完一圈
t=6  W1: E(v0) 采完 → 队列已满，W1 卡住，手里拿着 E
     W2: A(v0) 采完 → 队列已满，W2 卡住，手里拿着 A
```

此刻在途数据：队列里 2 组 + worker 手里 2 组 = 4 组，全是 v0。

**t = 7：第 1 步完成，开始第 2 步**

```
publish_policy(1)          → 权威状态 = (π1, 1)，保存快照 v1
record_training(step=1)    （policy_version = 1）

learner 第 2 步:
  get → C(v0)：陈旧度 1 ≤ 1，接受（consumed = 3）
        → W1 放入 E，队列 [D, E]；W1 取 B，捕获 (π1, 1)    ← 开始用新版本
  get → D(v0)：陈旧度 1，接受（consumed = 4）
        → W2 放入 A，队列 [E, A]；W2 取 C，捕获 (π1, 1)
  batch₂ = [C, D] → 开始训练（到 t = 12 结束）
```

**t = 9：worker 再次被拦住**

```
W1: B(v1) 采完 → 队列 [E, A] 已满 → 卡住
W2: C(v1) 采完 → 卡住
```

**t = 12：第 2 步完成，后台启动评估，开始第 3 步**

```
publish_policy(2)          → 权威状态 = (π2, 2)，保存快照 v2
record_training(step=2)    （policy_version = 2）
eval_due(2)? 是 → 捕获 (π2, 2) → 创建后台评估任务（假设运行到 t = 18），learner 不等待

learner 第 3 步:
  get → E(v0)：陈旧度 2 > 1，丢弃 ✗（dropped = 1）
        → W1 放入 B(v1)，队列 [A, B]；W1 取 D，捕获 (π2, 2)
  get → A(v0)：陈旧度 2 > 1，丢弃 ✗（dropped = 2）
        → W2 放入 C(v1)，队列 [B, C]；W2 取 E，捕获 (π2, 2)
  get → B(v1)：陈旧度 1，接受（consumed = 5）
  get → C(v1)：陈旧度 1，接受（consumed = 6）
  batch₃ = [B, C] → 开始训练（到 t = 17 结束）
```

E、A 在 t = 6 就采完了，但被背压挡在 worker 手里，被取出时策略已前进两版，只能丢弃。这期间评估任务（用 π2）、两个 worker（用 π2 采 D 和 E）、learner 的训练三者同时进行。

**t = 17：第 3 步完成，循环结束**

```
publish_policy(3)          → 权威状态 = (π3, 3)，保存快照 v3
record_training(step=3)    （policy_version = 3）
```

**收尾**

```
取消 W1、W2 → 正在用 π2 采的 D、E 被放弃 → 等待它们退出（被取消不算错误）
等待评估任务 → t = 18 完成 → record_evaluation(step=2, policy_version=2)
返回 {"train_steps": 3, "groups_consumed": 6, "groups_dropped_stale": 2}
```

最后一个快照 v3 就是这次运行的 `final_sampler_path`。

### 12.3 汇总表

| 时间 | 权威状态 | W1 | W2 | 队列 | learner | 接受 / 丢弃 |
|---|---|---|---|---|---|---|
| 0 | (π0, 0) | 采 A@v0 | 采 B@v0 | [] | 等待 get | 0 / 0 |
| 2 | (π0, 0) | 采 C@v0 | 采 D@v0 | [] | batch₁=[A,B]，训练 | 2 / 0 |
| 4 | (π0, 0) | 采 E@v0 | 采 A@v0 | [C,D] 满 | 训练中 | 2 / 0 |
| 6 | (π0, 0) | **卡住**，拿着 E | **卡住**，拿着 A | [C,D] 满 | 训练中 | 2 / 0 |
| 7 | **(π1, 1)** | 采 B@v1 | 采 C@v1 | [E,A] | batch₂=[C,D]，训练 | 4 / 0 |
| 9 | (π1, 1) | **卡住**，拿着 B | **卡住**，拿着 C | [E,A] 满 | 训练中 | 4 / 0 |
| 12 | **(π2, 2)** | 采 D@v2 | 采 E@v2 | [] | 丢 E、A；batch₃=[B,C]，训练；**后台评估 π2** | 6 / 2 |
| 17 | **(π3, 3)** | 被取消 | 被取消 | — | 结束 | 6 / 2 |

### 12.4 从例子看每条设计

| 设计 | 例子中的体现 |
|---|---|
| 权威状态成对更新 | 版本号只在 `publish_policy` 中与策略一起改变（t = 7、12、17） |
| worker 捕获 (策略, 版本) | t = 6 时 W1 拿着的 E 是 v0 采的；虽然 t = 7 发布了 v1，它仍标记为 v0 |
| 共享任务流 | W1、W2 交替取 A–E，再回到 A，从不重复领同一个任务 |
| 有界队列与背压 | t = 6、t = 9 两个 worker 都被挡住 |
| 消费时检查陈旧度 | E、A 生产时是新的，被取出时已过旧，在第 3 步被丢弃 |
| batch 可能混合版本 | 本例中每个 batch 恰好同版本；若采样速度不同，一个 batch 中可能同时有 v0 和 v1 的组 |
| 评估不阻塞 | t = 12 调度评估后 learner 立刻进入第 3 步；评估在 t = 18 完成，结果记在第 2 步、版本 2 |
| 步数与版本 | 本例无跳过的步，step 与 version 一致；若某步全是退化组，step 增加而 version 不变 |
| 干净的关闭 | 结束时取消正在采样的 worker，等待退出，再等待评估完成，最后返回统计 |

## 13. 与同步 trainer 的对比及参数权衡

| 方面 | 同步 `SyncTrainer` | 异步 `AsyncGRPOTrainer` |
|---|---|---|
| batch 从哪里来 | 每步现采 `tasks_per_batch` 组 | 从队列中取 `groups_per_batch` 个新鲜组 |
| 采样与训练 | 交替进行 | 同时进行 |
| batch 内的版本 | 全部相同 | 可能来自不同版本 |
| 每组带版本号 | 不需要 | 需要 |
| 过旧数据 | 不存在 | 消费时检查并丢弃 |
| 快照命名 | `step_000005`（按步数） | `v5`（按版本） |
| 跳过的步 | 不更新、不建快照 | 不更新、不发布，版本不变 |
| 评估 | 阻塞，在循环中等待 | 后台任务，捕获 (策略, 版本) |
| 日志额外字段 | — | `policy_version`、`loss_fn_is_iw` |
| `train()` 返回 | `None` | 三个统计值 |
| 失败处理 | 自然向上抛出 | 需主动监视、取消、重新抛出 |
| 支持的算法 | REINFORCE / GRPO | 只支持 GRPO |
| 服务器 | 训练与推理共置 | 训练卡与推理卡分开 |

| 参数 | 调大的好处 | 调大的代价 |
|---|---|---|
| `num_workers` | 推理引擎更满载，吞吐更高 | 在途数据更多，平均更旧 |
| `queue_capacity` | 能吸收生产与消费速度的波动 | 数据可能更旧 |
| `groups_per_batch` | 每步数据更多，梯度更稳 | 每步更慢 |
| `max_staleness` | 丢弃更少，不浪费采样 | 数据更 off-policy，依赖重要性加权校正（Part 4.3 正是检验这一点） |

p5 配置注释给出的经验规则：让 `(num_workers + queue_capacity) / groups_per_batch` 小于 `max_staleness`，就能在吞吐与新鲜度之间取得平衡。

---

# C. 实现

## 14. 需求与工具的对应

| Part 3 的需求 | 不用对应工具会怎样 | 用什么 |
|---|---|---|
| 采样和训练**同时**进行 | 只能交替进行，退化成同步 | `create_task` 启动常驻 worker |
| 传数据并**限制数据有多旧** | 无界堆积，大量数据过期 | `asyncio.Queue(maxsize=…)` + `await put/get` |
| 一组 rollout 内部并发 | 8 条依次跑，慢 8 倍 | `gather`（`collect_group` 已实现） |
| 评估**不阻塞**训练 | learner 停下来等评估 | `create_task` 启动后台评估 |
| worker 出错时 learner **不卡死** | 永远停在 `await queue.get()`，GPU 空转计费 | `asyncio.wait(FIRST_COMPLETED)` |
| 结束或出错时**干净关闭** | 遗留任务、警告、资源未释放 | `task.cancel()` + `await` |
| 重新抛出**原始**异常 | 调用方看到的是包装后的异常 | `try/except` + `raise`；不直接用 `TaskGroup` |
| 读取 (策略, 版本) 不会读到一半 | 版本号与策略不匹配 | 利用"两个 `await` 之间不会被打断" |

## 15. 逐个方法的参考实现

### 15.1 `generate_group`：查任务 → 跑一组 → 算优势 → 打包

**契约要点**：

| 要求 | 原因 |
|---|---|
| 先用 `self.tasks_by_id[task_id]` 查任务，未知 id 立即抛 `KeyError` | 有问题应立即失败，而不是先跑完一组 rollout（占用 GPU）再报错；不要用 `.get()` |
| 调用文件顶部导入的 `collect_group`，参数为 `(policy, task, self.env_factory, self.cfg.group_size, agent_cls=self.agent_cls())` | 评分时会替换模块中的 `collect_group`；在函数内重新导入或自己写 `gather` 都会让替换失效 |
| 用 `groups_to_batch([group], method="grpo")` 计算组内 GRPO 优势 | 复用 Part 2 的逻辑，退化组自然返回 `[]`；固定写 `"grpo"`，异步 trainer 只支持 GRPO；单组的 stats 用不上，`record_training` 会重新计算 |
| **总是**返回 `GroupResult`，退化组 `sequences=[]` 但保留 `rewards` | `record_training` 以 `sequences` 是否为空判断退化组，并用 `rewards` 计算 `mean_reward` 和 pass@k |
| `policy_version` 用传入的 `version`，采样用传入的 `policy` | 见 11.1 节 ③；也不要换算成绝对版本，陈旧度计算用相对版本 |

```python
async def generate_group(self, task_id, policy, version):
    # ① 先查任务：未知 id 立即抛 KeyError，此时还没有启动任何 rollout
    task = self.tasks_by_id[task_id]

    # ② 用传入的 policy 运行一组 rollout；必须调用模块级的 collect_group
    group = await collect_group(
        policy,
        task,
        self.env_factory,
        self.cfg.group_size,
        agent_cls=self.agent_cls(),
    )

    # ③ 组内计算 GRPO 优势并拼成训练序列；退化组得到 []
    sequences, _ = groups_to_batch([group], method="grpo")

    # ④ 总是返回 GroupResult；版本号用传入的 version，而不是 self.version
    return GroupResult(
        task_id=task_id,
        policy_version=version,
        sequences=sequences,
        rewards=[traj.reward for traj in group],
    )
```

另一种写法是直接调用 `compute_group_advantages` 和 `trajectory_to_training_sequences`，但需要额外导入；推荐上面的写法，与同步 trainer 共用同一份逻辑，统计口径不会不一致。

### 15.2 `rollout_worker`：常驻的生产者

```python
async def rollout_worker(self, results, task_iter):
    while True:                                          # 永远循环，直到被 cancel
        task_id = next(task_iter)                        # 共享迭代器；没有 await，不会被打断
        policy, version = self.active_policy()           # 成对捕获；同样不会被打断
        group = await self.generate_group(task_id, policy, version)
        await results.put(group)                         # 队列满时在这里等待 → 背压
```

| 要点 | 原因 |
|---|---|
| `while True`，没有退出条件 | 由 `train()` 在结束时取消，`CancelledError` 会在某个 `await` 处到达 |
| 不写 `try/except` | docstring 要求让错误向上传播，由 `train()` 发现并处理 |
| 通过 `self.generate_group` 调用 | 测试会替换实例上的这个方法 |

### 15.3 `collect_fresh_batch`：取数据并丢弃过旧的

```python
async def collect_fresh_batch(self, results, stats):
    stats.setdefault("groups_dropped_stale", 0)
    batch = []
    while len(batch) < self.cfg.groups_per_batch:
        group = await results.get()                      # 队列空时在这里等待
        if self.version - group.policy_version > self.cfg.max_staleness:
            stats["groups_dropped_stale"] += 1           # 过旧：丢弃，继续取
            continue
        batch.append(group)                              # 退化组也算，照样放进 batch
    return batch
```

### 15.4 `update` 与 `evaluate_now`：普通的 await

```python
async def update(self, batch):
    seqs = [s for g in batch for s in g.sequences]       # 按组、组内按顺序拼接
    return await self.session.train_step(
        seqs, loss_fn=self.cfg.loss_fn, lr=self.cfg.lr, grad_clip_norm=self.cfg.grad_clip_norm)

async def evaluate_now(self, policy):
    return await evaluate_policy(                         # 模块级名称，评分时会被替换
        policy, self.val_tasks[: self.cfg.eval_tasks], self.env_factory,
        k=self.cfg.eval_k, agent_cls=self.agent_cls())
```

`update` 只负责训练并返回指标，发布和记录属于 `train()`。

### 15.5 辅助函数：等待的同时监视故障

```python
async def _await_or_fail(coro, watched):
    main = asyncio.ensure_future(coro)                   # 把要等的协程（如取 batch）包装成任务
    try:
        while True:
            for t in watched:                            # 已经出错的任务 → 立即抛出它的原始异常
                if t.done() and not t.cancelled() and t.exception() is not None:
                    raise t.exception()
            pending = {t for t in watched if not t.done()}
            done, _ = await asyncio.wait({main, *pending}, return_when=asyncio.FIRST_COMPLETED)
            if main in done:
                return main.result()                     # 正常拿到结果
            # 否则：有被监视的任务结束了，回到循环开头检查它是否出错
    finally:
        if not main.done():                              # 因故障提前退出时，收拾没完成的取数任务
            main.cancel()
            await asyncio.gather(main, return_exceptions=True)
```

| 要点 | 原因 |
|---|---|
| `asyncio.wait(FIRST_COMPLETED)` | 只写 `await queue.get()` 的话，worker 全挂后会永远等待 |
| 每轮只监视**尚未结束**的任务 | 评估任务正常结束后仍是"已完成"状态，若继续放进 `wait`，`wait` 会立刻返回，造成空转 |
| `raise t.exception()` | 抛出的是原始异常对象本身 |
| `finally` 中取消并等待 `main` | 避免遗留悬空的取数任务 |

### 15.6 `train`：把所有工具组合起来

```python
async def train(self):
    stats = {"train_steps": 0, "groups_consumed": 0, "groups_dropped_stale": 0}
    await self.publish_policy(0)                                          # 版本 0
    queue = asyncio.Queue(maxsize=self.cfg.queue_capacity)                # 有界队列
    task_iter = itertools.cycle([t.id for t in self.train_tasks])         # 共享、循环的任务流
    workers = [asyncio.create_task(self.rollout_worker(queue, task_iter))  # 启动常驻生产者
               for _ in range(self.cfg.num_workers)]
    evals = []
    try:
        for step in range(1, self.cfg.num_steps + 1):
            batch = await _await_or_fail(                                 # 取数据，同时监视故障
                self.collect_fresh_batch(queue, stats), workers + evals)
            stats["train_steps"] += 1
            stats["groups_consumed"] += len(batch)
            if any(g.sequences for g in batch):                           # 有训练信号
                metrics = await self.update(batch)
                await self.publish_policy(self.version + 1)
            else:                                                         # 全是退化组：跳过
                metrics = {}
            self.record_training(step, batch, metrics)                    # 每步恰好一次
            if self.eval_due(step):
                policy, version = self.active_policy()                    # 捕获此刻的一对
                evals.append(asyncio.create_task(                         # 后台评估，不等待
                    self.run_evaluation(policy, version, step)))
        # —— 正常结束 ——
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)            # worker 被取消不算错误
        await asyncio.gather(*evals)                                      # 等评估完成；评估出错会抛出
    except BaseException:
        # —— 任何异常：收拾所有任务，再重新抛出原始异常 ——
        for t in workers + evals:
            t.cancel()
        await asyncio.gather(*workers, *evals, return_exceptions=True)
        raise
    return stats
```

| 位置 | 工具 | 作用 |
|---|---|---|
| 启动 worker | `create_task` | 常驻后台，与 learner 同时运行 |
| 有界队列 | `Queue(maxsize=…)` | 传数据，并限制在途数据量 |
| 取 batch | `_await_or_fail` → `asyncio.wait` | worker 或评估出错时及时发现，不会卡死 |
| 调度评估 | `create_task` | 不等待，训练继续进行 |
| 正常结束：关闭 worker | `cancel` + `gather(return_exceptions=True)` | 此时被取消属于正常情况 |
| 正常结束：等待评估 | `gather(*evals)` | 等最后的评估写完日志；评估出错会被抛出 |
| 异常处理 | `except BaseException` → 取消全部 → `raise` | 无论在哪里出错，都先清理再抛出原始异常 |

容易忽略的细节：

1. **用 `except BaseException`**：外部取消（例如 `wait_for` 超时）抛出的 `CancelledError` 属于 `BaseException`，用 `except Exception` 会漏掉，worker 就没人取消了；
2. **用不带参数的 `raise`**：重新抛出的就是捕获到的异常对象本身；
3. **不直接用 `asyncio.TaskGroup`**：它会把异常包成 `ExceptionGroup`；
4. **方法都通过 `self.` 调用**：`self.rollout_worker`、`self.generate_group`、`self.update`、`self.evaluate_now` 都会被测试替换；`collect_group` 和 `evaluate_policy` 则直接使用文件顶部导入的名字。

### 15.7 与 docstring 要求的对照

| docstring 要求 | 实现方式 |
|---|---|
| 先发布版本 0 | `await self.publish_policy(0)` |
| 有界队列 + 共享循环任务流 | `Queue(maxsize=…)` + 一个所有 worker 共用的 `itertools.cycle` |
| 恰好 `num_steps` 轮，编号从 1 开始 | `for step in range(1, num_steps + 1)` |
| 没有序列时跳过：不更新、不发布 | `if any(g.sequences for g in batch)` |
| 每轮恰好一次 `record_training` | 放在 if / else 之后 |
| 评估时捕获 (策略, 版本)，以任务形式调度 | `active_policy()` + `create_task(run_evaluation(...))` |
| worker 出错不能让 learner 卡死 | `_await_or_fail` 同时监视 worker |
| 评估出错要让训练**及时**失败 | `_await_or_fail` 同时监视评估任务（实测在第 1 次更新后即停止） |
| 结束时取消并等待 worker，再等待评估 | 正常结束分支 |
| 任何异常：取消全部任务并重新抛出原始异常 | `except BaseException` 分支 |
| 恰好返回三个键 | `stats` 初始化时就包含这三个键 |

---

## 16. 总结

**架构**：worker 不断"领任务、按当时的版本采一组、放进队列"；learner 不断"从队列取组、丢掉太旧的、凑满就训练、发布新版本"；权威状态保证双方对"当前版本"的认识一致，有界队列限制数据能有多旧，评估在后台旁观，结束时把所有任务干净地收回。

**五条设计原则**：

1. 一份权威状态，成对发布、成对读取；
2. 版本跟着数据走：每组在开始时捕获版本，learner 在消费时判断陈旧度；
3. 有界队列提供背压：限制陈旧度，也避免无用的生产；
4. 评估不阻塞训练：后台运行，结果归属于捕获时的版本和调度时的步数；
5. 出错时立即、干净地失败：任何一方出错，都取消所有任务并抛出原始异常。

**asyncio 层面**：`create_task` 让 worker 和评估在后台同时运行；`Queue` 传数据并自动限速；`asyncio.wait` 让 learner 在等数据时也能发现故障；`cancel` + `await` + `raise` 保证干净收尾；"两个 `await` 之间不会被打断"保证共享状态的读写无需加锁。核心原则：**在 `await` 之前捕获共享状态，在 `await` 之后只用局部变量。**
