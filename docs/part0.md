# Part 0：Agent、Rollout 与 Env 代码分析

> 配合 [overview.md](overview.md) 和 [dataflow.md](dataflow.md) 阅读。Part 0 要求阅读 [agents/xml_tool_call.py](../scottyrl/agents/xml_tool_call.py)、[rollout.py](../scottyrl/rollout.py) 和 [envs/](../scottyrl/envs/)，理解一个回合如何产生 `Trajectory`。第 7 节另外解释代码中随处可见的 `async` / `await`。

三者的分工可以概括为：

- **Agent 负责"想"**：构造 prompt、调用模型、解析动作；
- **Env 负责"判"**：执行工具、给出反馈和奖励、管理步数；
- **`run_episode` 负责"跑"**：在两者之间来回传递，最后打包成 Trajectory。

---

## 1. 三个角色的接口

```python
# envs/base.py
class Env:
    tool_docs                    # 工具说明，会写进 system prompt
    async reset() -> Observation # 重置状态，返回初始观测
    async step(ToolCall | None) -> Observation   # 执行一步（基类提供的通用逻辑）
    async apply(ToolCall) -> Observation         # 具体环境实现的工具逻辑

# agents/base.py
class Agent:
    llm_calls: list[LLMCall]     # 训练数据就在这里
    async reset(task, tool_docs)
    async step(Observation) -> AgentAction(tool_call, text)
```

[agents/base.py](../scottyrl/agents/base.py) 的 docstring 写明了一条硬性约定：**不论 Agent 内部如何实现，每一次调用模型都必须记入 `self.llm_calls`，因为那就是训练数据**。Env 则完全不知道模型和 token 的存在：它只接收解析好的 `ToolCall`，返回文本形式的 `Observation`。

两者都通过注册表按名字创建（[agents/__init__.py](../scottyrl/agents/__init__.py) 中的 `xml_toolcall`，[envs/__init__.py](../scottyrl/envs/__init__.py) 中的 `number_search` / `textcraft`）。`make_env_factory` 返回 `lambda task: cls(task)`，每次调用都**新建**一个环境实例。这是 `collect_group` 要求"每条 rollout 使用全新 env"的基础：环境有状态（例如 TextCraft 的库存），不能在 rollout 之间共用。

---

## 2. 主循环：`run_episode`

对应 [rollout.py](../scottyrl/rollout.py) 中的 `run_episode`，简化后如下：

```python
obs = await env.reset()
await agent.reset(env.task, env.tool_docs)
for _ in range(task.max_steps):
    try:
        action = await agent.step(obs)        # 调用模型
    except Exception:
        if num_steps == 0: raise              # 第一轮就失败：配置问题，直接报错
        del agent.llm_calls[calls_before:]    # 删除失败这一轮可能已记下的调用
        stop_reason = ...; break              # 放弃这个回合
    obs = await env.step(action.tool_call)    # 执行动作
    num_steps += 1
    if obs.done: break

reward = 0.0 if stop_reason else obs.reward   # 奖励只看最后一个观测
return Trajectory(task, agent.llm_calls, reward, num_steps, info)
```

要点：

- **奖励是稀疏的**：只取最后一个 Observation 的 reward，中间步骤都是 0。
- **异常的两种处理**：
  - 第一轮就失败，说明是配置或连接问题，直接抛出；
  - 中途失败（典型情况是上下文超过 vLLM 的 `max_model_len` 被拒绝），这个回合被**放弃**，记为失败（reward = 0），并删除失败那一轮的调用，保证 `calls` 中只有完整的轮次、仍然可以拼接。transcript 里最后一轮带 `stop_reason`、没有 completion 的，就是这种情况。
- **副作用（只用于记录，不影响训练）**：每个回合结束时向 `PENDING_TRAJECTORIES` 追加一条 transcript，向 `EPISODE_TIMES` 追加起止时间，由 `MetricsLogger` 取出后写盘和计时。`rollout_phase` 用 `ContextVar` 把回合标记为 train 或 eval；不用全局变量，是因为异步训练中评估任务与 rollout worker 同时运行，各自的标记不能互相干扰（Part 3 会用到）。

---

## 3. Agent：`XMLToolCallAgent`

### 3.1 prompt 结构

```
system:    You are Scotty ... 每轮只输出一个 <tool>...</tool> ... + 环境的 tool_docs
user:      Task: <goal>\n<初始观测>
assistant: <tool>guess(3)</tool>
user:      3 is too low.
assistant: ...
```

经 [qwen3_5.jinja](../scottyrl/chat_templates/qwen3_5.jinja) 渲染后，每个 assistant 轮次的格式为：

```
<|im_start|>assistant\n + prefill + 内容 + <|im_end|>\n
prefill = "<think>\n\n</think>\n\n"   （thinking 关闭：空思考块，Number Search）
        = "<think>\n"                 （thinking 开启：模型接着写推理，TextCraft）
```

**prefill 属于 prompt，不属于模型的动作**，其 token 的 mask 为 0。开启 thinking 时，模型采样出的内容是 `推理…</think>\n\n<tool>…</tool><|im_end|>`，推理部分也算动作，会被训练。

### 3.2 `step()` 的流程

1. 第一轮：构造 system + user 消息，渲染后**整段编码**；
2. 之后每一轮：调用 `_extend_context`，**只编码新增的尾部**，模型自己生成的 token 原样复用（详见 [dataflow.md 2.2 节](dataflow.md)）；
3. `policy.complete(tokens)` 采样，并立即记录 `LLMCall(prompt, completion, logprobs)`；
4. `parse_tool_call` 将文本解析为 `ToolCall`，与原始文本一起作为 `AgentAction` 返回。

### 3.3 `parse_tool_call` 的行为

- 用正则匹配**第一个** `<tool>name(args)</tool>`，之后的内容全部忽略。system prompt 要求"工具调用之后不要输出任何内容"，但多输出了也不会报错。
- 参数用 `ast.literal_eval` 解析，只接受 Python 字面量（数字、字符串、dict、tuple 等），不会执行任意代码。
- 没有匹配或解析失败时返回 `None`。transcript 中出现 `"tool_call": null`，就表示模型输出格式不对。

### 3.4 扩展何时成立、何时断开

结合模板分析，有三种情况：

| completion 的结尾 | `strip_trailing_stop` 后 | 模板写回的历史 | 扩展是否成立 |
|---|---|---|---|
| `…</tool><\|im_end\|>`（正常） | 去掉 `<\|im_end\|>` | 内容 + `<\|im_end\|>\n` | ✅ 与采样文本完全一致 |
| 达到 `max_tokens` 被截断，没有停止符 | 不变 | 内容 + `<\|im_end\|>\n` | ✅ `<\|im_end\|>` 落在新尾部，mask = 0 |
| `…<\|endoftext\|>` | 去掉 `<\|endoftext\|>` | 内容 + `<\|im_end\|>\n` | ❌ 文本对不上，重新编码 |

此外，`enable_thinking: true` 且 `preserve_thinking_history: false` 时，历史中 `</think>` 及之前的内容会被删除，**每一轮都会断开**。thinking 关闭时这个开关没有作用（模板中只有 `enable_thinking and not preserve_thinking_history` 这个分支会删除内容）。

第二行是个容易忽略的细节：被截断的回复本身是合法的，扩展也成立，只是补上的 `<|im_end|>` 不是模型生成的，不参与训练。

---

## 4. Env

### 4.1 基类 `Env.step`：所有环境共用的规则

对应 [envs/base.py](../scottyrl/envs/base.py)：

```python
self._steps_used += 1                         # 无论动作是否合法，都消耗一步
if call is None:  obs = "Could not parse a tool call..."
else:             obs = await self.apply(call)
if not obs.done and steps_used >= max_steps:
    obs = obs.text + "\n(Out of steps.)", done=True
```

- **步数预算由环境管理**，`run_episode` 中的 for 循环只是双保险。
- **格式错误同样消耗一步**，除了一条提示文本外没有其它反馈。

### 4.2 Number Search

对应 [number_search.py](../scottyrl/envs/number_search.py)：

- 唯一工具 `guess(n)`，参数必须是**一个 int**（`guess(3.0)`、`guess("3")` 都非法）。
- 反馈：`n is too low.` / `n is too high.` / `Correct!`（reward = 1）。
- 预算 ⌈log₂(high − low + 2)⌉，恰好等于二分查找最坏情况所需次数，**一步都不能浪费**。一次格式错误或一次非二分的猜测，基本就意味着失败（除非运气好）。

因此 Number Search 上的 RL 主要在学两件事：**格式稳定**和**严格二分**。Part 3 要求对比早、中、晚期 transcript 的行为变化时，可以从这两个角度观察。

### 4.3 TextCraft-Synth

对应 [textcraft/env.py](../scottyrl/envs/textcraft/env.py)：

| 工具 | 作用 | 要点 |
|---|---|---|
| `get_info(items)` | 查询配方 | 返回 `can_craft`、`is_base`、`in_inventory`、`crafting_depth`、`recipes` |
| `craft(ingredients, (item, count))` | 合成 | `count` 必须是 `result_count` 的整数倍；原料必须**恰好**等于配方 × 合成次数，不能缺、不能多、数量不能错；库存必须足够 |
| `view_inventory()` | 查看库存 | 不能带参数 |
| `finish(message)` | 结束 | **一旦调用回合即结束**；仅当每个目标物品的"最终库存 − 初始库存"都达到要求时 reward = 1 |

难点：

- **配方构成深度 5–6 的 DAG**（见数据中的 `max_depth`），中间产物需要逐层合成，一个任务约需 22–25 次 craft（`num_craft_steps`）。
- **原料只来自初始库存**，没有采集动作，合成顺序和用量规划不能出错。
- **一个物品可能有多个配方**：`craft` 会逐个尝试，用第一个能成功的；全部失败时列出每个配方的错误原因。
- **过早调用 `finish` 直接得 0 分**（`done=True`）；**完全不调用 `finish`** 也是 0 分，即使物品已全部合成。
- 75 步预算加上思维链，回合非常长，这也是 Part 5 的成本远高于 Number Search 的原因。

---

## 5. 采样：`TinkerPolicy.complete`

对应 [policy.py](../scottyrl/policy.py)：

- 显式设置 `top_p=1.0, top_k=-1`，从**完整、未截断**的分布中采样，这样返回的 logprob 才是 PPO / IS loss 所假设的 log π_old。若做了截断，logprob 与训练时算出的不一致，ρ 会出现系统性偏差。
- 解码时**保留特殊 token**（`skip_special_tokens=False`），Agent 才能看出模型是以 `<|im_end|>` 还是 `<|endoftext|>` 结束的。
- 测试用的 `MockPolicy` 把每个字节当作一个 token、按预设脚本输出，不需要 GPU 就能跑完整个回合。`tests/test_envs.py` 中的 `ThinkStrippingPolicy` 就是用这种方式构造"扩展断开"的回合，Part 1 测试时可以使用。

---

## 6. 对后续 Part 的影响

| 观察 | 影响到哪里 |
|---|---|
| `llm_calls` 是唯一的训练数据来源，prompt 逐轮累积 | Part 1：`stitch_trajectory` 的输入与拼接逻辑 |
| prefill、观测、补上的 `<\|im_end\|>` 都在 prompt 中 | Part 1：这些位置 mask 必须为 0 |
| 有三种原因会导致扩展断开 | Part 1：必须支持一个回合拆成多条序列 |
| 被放弃的回合 reward = 0，`calls` 仍是完整轮次 | Part 1/2：照常拼接和训练，作为失败样本 |
| 每次 `env_factory(task)` 都新建实例 | Part 2：`collect_group` 中每条 rollout 必须独立创建 env 和 agent |
| 奖励只有 0/1，Number Search 没有任何容错 | Part 2/4：同组奖励容易全部相同，形成退化组（G = 2 时尤其明显） |
| `rollout_phase` 使用 ContextVar | Part 3：异步评估不会把训练 rollout 误标为 eval |
| transcript 中的 `tool_call: null` 和 `stop_reason` | 所有 Part：判断格式错误和被放弃回合的直接信号 |
| `top_p=1, top_k=-1` 采样 | 同步训练时 ρ ≈ 1 才成立 |

---

## 7. 代码中的 `async` / `await` 是不是为了异步 RL？

不是。这里有两个都叫"异步"的概念：

| | Python 的 `async` / `await` | 异步 RL（Part 3） |
|---|---|---|
| 层次 | **编程语言 / 工程**层面 | **算法 / 系统设计**层面 |
| 解决的问题 | 等网络返回时不闲着，先处理别的任务 | 采样和训练同时进行，不让 GPU 空闲 |
| 使用范围 | **整个框架**，同步 trainer 也用 | 只有 `AsyncGRPOTrainer` |
| 是否影响训练数据 | 否 | 是（数据可能来自旧策略） |

代码中的 `async def` / `await` 属于前者，即使 Part 2 的同步 RL 也离不开它。

### 7.1 `async` / `await` 是什么

```python
async def run_episode(...):        # async def：定义一个"可以中途暂停"的函数（协程）
    action = await agent.step(obs) # await：在此等待结果；等待期间把控制权让给其它协程
```

- **`async def`**：声明这个函数执行中可能需要等待（例如等网络），可以在等待处暂停。
- **`await`**：真正暂停的位置，含义是"这个结果需要等一会儿，等的时候让其它任务先运行，结果回来后再从这里继续"。
- **事件循环（event loop）**：`asyncio` 的调度器，在**单线程**中负责切换：谁在等待就挂起，谁的结果到了就恢复谁。

**类比**：一个服务员同时照看 64 桌客人。给 1 号桌下单后，他不会站在厨房门口等菜，而是去给 2 号、3 号桌下单；哪桌的菜好了就回去端哪桌。服务员只有一个（单线程），但没有人在空等。

### 7.2 为什么需要它：瓶颈在等待网络

一个回合中最耗时的是这一行：

```python
# policy.py
result = await self.sampling_client.sample_async(...)   # 把 prompt 发给 Modal 上的 vLLM，等它生成完返回
```

模型不在本地，每次采样都是一次网络请求，本地 CPU 在此期间**无事可做，只是在等**。相比之下，环境逻辑（比较大小、查配方）几乎不花时间。

假设一次采样往返 1 秒、一个回合 3 轮（示意数字）：

```
不并发（逐条运行）:
回合1 [采样][采样][采样]
回合2                   [采样][采样][采样]
...
回合64                                              ...[采样]
总计 ≈ 64 × 3 = 192 秒

并发（asyncio.gather）:
回合1  [采样][采样][采样]
回合2  [采样][采样][采样]
...    ← 64 个请求几乎同时到达 vLLM，vLLM 将它们合成一个大 batch 一起生成
回合64 [采样][采样][采样]
总计 ≈ 3 秒多一点
```

并发还有第二个好处：vLLM 一次收到很多请求，可以**批量生成**，GPU 利用率高得多。逐个发送时 GPU 每次只处理一条序列，大部分算力被浪费。

### 7.3 在代码中的体现

**同步 trainer（Part 2）中的两层并发：**

```python
# SyncTrainer.collect_batch：8 个任务的组同时运行
groups = await asyncio.gather(*[collect_group(policy, task, ...) for task in tasks])

# collect_group：同一任务的 8 条 rollout 同时运行
trajs = await asyncio.gather(*[run_episode(agent_cls(policy), env_factory(task)) for _ in range(G)])
```

于是有 8 × 8 = **64 个回合同时进行**。docstring 中"groups for different tasks are collected concurrently"和"launch all with `asyncio.gather`"指的就是这件事。

调用链上的每一层都是 `async`：

```
collect_batch ─await─▶ collect_group ─await─▶ run_episode ─await─▶ agent.step ─await─▶ policy.complete ─await─▶ sample_async（真正等网络的地方）
```

这就是 async 的"传染性"：只要最底层需要 `await`，所有调用者都必须是 `async def`，否则无法 `await` 它。

**为什么 Env 也是 async？** Number Search 的 `apply` 中其实没有任何 `await`，是纯 CPU 计算。之所以也写成 `async`，[envs/base.py](../scottyrl/envs/base.py) 的 docstring 给出了原因：以后接入需要等待的环境（远程服务、沙箱、子 Agent 等）时，不必修改任何调用方。

**Part 2 的 `submit_update` 也是同一个道理（示意）：**

```python
fwd = await training_client.forward_backward_async(...)   # 提交请求，拿到 future
opt = await training_client.optim_step_async(...)         # 提交请求，拿到 future
fwd_result = await fwd.result_async()                     # 之后再等待结果
opt_result = await opt.result_async()
```

两个请求先连续发出，服务器可以把它们放进同一个处理周期；如果等第一个结果回来再发第二个，延迟会翻倍。这同样是"不要干等"的思路，与异步 RL 无关。

### 7.4 异步 RL 是在 asyncio 之上的另一层设计

```python
queue = asyncio.Queue(maxsize=16)                          # 有界队列
workers = [asyncio.create_task(self.rollout_worker(...))   # 8 个常驻后台的生产者
           for _ in range(8)]
batch = await self.collect_fresh_batch(queue, stats)       # learner 从队列取数据
...
for w in workers: w.cancel()                               # 结束时取消
```

区别在于：

- **同步 RL**：用 `gather` 一次性跑完 64 个回合，然后训练。asyncio 只负责让这 64 个回合并发，**采样和训练仍是轮流进行的**。
- **异步 RL**：worker 是常驻任务，**训练期间也在继续采样**。它用到 `create_task`、`Queue`、`cancel` 等更进一步的 asyncio 工具，目的是让采样和训练在时间上重叠（详见 [async.md](async.md)）。

### 7.5 一个容易误解的地方

asyncio 是**并发**而非**并行**：始终只有一个线程，同一时刻只执行一段 Python 代码。它能加速，完全是因为大部分时间都在等网络，可以趁等待时切换到别的任务。如果某段代码是纯 CPU 计算（例如大量 tokenize），它运行期间其它协程都会被阻塞。

在这个框架中，真正的重计算都在远端 GPU 服务器上，本地只负责发请求和收结果，所以 asyncio 正好适用。

### 7.6 小结

- **`async` / `await`** = "等网络时别闲着"，是让 64 个回合同时运行、让 vLLM 批量生成的工程手段。同步和异步 trainer **都依赖它**。
- **异步 RL** = "训练时也不停止采样"，是 Part 3 的算法设计。它**用 asyncio 实现**，但这是两个层次的问题。
