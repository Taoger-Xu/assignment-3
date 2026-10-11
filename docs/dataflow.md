# 核心数据流示例：从 Task 到 tinker.Datum

> 配合 [overview.md](overview.md) 和 [scottyrl/types.py](../scottyrl/types.py) 阅读。下面用一个虚构的 Number Search 回合，把 `types.py` 中的数据结构按出现顺序串起来。环境文本、prompt 格式和 `TokenSequence` 的对齐规则都与代码一致；token id 是为了便于阅读而编的，真实 id 会很大，一句话也会被切成很多个 token。

---

## 0. 总览：谁创建了什么

```
Task ──▶ Observation ──▶ LLMCall ──▶ ToolCall ──▶ Observation … ──▶ Trajectory ──▶ TokenSequence ──▶ tinker.Datum
数据文件    env.reset()     agent.step()  agent 解析     env.step()      run_episode()   datagen（你写）   tinker_adapter
```

| 类型 | 由谁创建 | 所在层 |
|---|---|---|
| `Task` | `load_tasks` 读取 jsonl | 数据 |
| `Observation` | `Env.reset()` / `Env.step()` | 环境 |
| `LLMCall` | `XMLToolCallAgent.step()` | Agent，**token 空间** |
| `ToolCall` | `parse_tool_call()` | Agent，文本空间 |
| `Trajectory` | `run_episode()` | rollout |
| `TokenSequence` | `stitch_trajectory` / `trajectory_to_training_sequences` | 数据管线（Part 1） |
| `tinker.Datum` | `to_tinker_datum` | 发送给后端 |

---

## 1. Task

`data/number_search_train.jsonl` 中的一行被加载成：

```python
Task(
    goal="Guess the correct number between 1 and 6.",
    id="number_search.train.42",
    max_steps=3,                         # ⌈log2(6 − 1 + 2)⌉ = ⌈log2 7⌉ = 3
    info={"low": 1, "high": 6, "target": 5},
)
```

`info` 是环境私有的数据（例如答案 `target`），模型看不到它。

---

## 2. 一个回合（run_episode）

### 2.1 第 1 轮

**① 环境给出初始观测**

```python
env.reset() → Observation(
    text="I'm thinking of a number between 1 and 6 (inclusive). You have 3 guesses.",
    reward=0.0, done=False)
```

**② Agent 构造消息，渲染后编码成 token**

```python
messages = [
  {"role": "system", "content": "You are Scotty, ... Tools available: guess(n: int) ..."},
  {"role": "user",   "content": "Task: Guess the correct number between 1 and 6.\nI'm thinking of ..."},
]
# 用 chat template 渲染成文本（末尾是 assistant 开头和空 think 块），再编码：
prompt_tokens = [1, 2, 3, 4, 5]          # 示意：system + user + "<|im_start|>assistant\n<think></think>"
```

**③ 模型采样**

```python
policy.complete([1, 2, 3, 4, 5]) → Completion(
    text="<tool>guess(3)</tool><|im_end|>",
    tokens=[50, 51],                      # 50 = "<tool>guess(3)</tool>"，51 = "<|im_end|>"
    logprobs=[-0.4, -0.1],                # 采样时每个 token 的 log π_old
)
```

Agent 立即把这次调用记录为一个 **LLMCall**：

```python
LLMCall(prompt_tokens=[1, 2, 3, 4, 5],
        completion_tokens=[50, 51],
        completion_logprobs=[-0.4, -0.1])
```

**④ 将文本解析为 ToolCall**

```python
parse_tool_call("<tool>guess(3)</tool><|im_end|>")
→ ToolCall(name="guess", args=(3,), raw="<tool>guess(3)</tool>")
```

解析失败时返回 `None`，环境会回复 "Could not parse a tool call..."，同时消耗一步。

**⑤ 环境执行**

```python
env.step(ToolCall("guess", (3,))) → Observation(text="3 is too low.", reward=0.0, done=False)
```

### 2.2 第 2 轮

**① Agent 扩展上下文**（`_extend_context`，sequence extension 的关键）

对应 [xml_tool_call.py](../scottyrl/agents/xml_tool_call.py) 中的 `_extend_context`：

```python
self.messages.append({"role": "assistant", "content": "<tool>guess(3)</tool>"})  # 去掉 <|im_end|>
self.messages.append({"role": "user", "content": "3 is too low."})
full_text = self.policy.render(self.messages)            # 重新渲染整段对话

seen = self._context_text + completion.text              # 旧 prompt 文本 + 模型刚生成的文本
if full_text.startswith(seen):
    self._context_tokens = (self._context_tokens         # [1, 2, 3, 4, 5]
                            + completion.tokens          # [50, 51]
                            + encode(full_text[len(seen):]))  # 只编码新尾部 → [6, 7, 8]
else:
    self._context_tokens = encode(full_text)             # 扩展断开：整段重新编码
```

渲染出的 `full_text` 可以切成三段，前两段合起来就是 `seen`：

```
┌───────────────────────────────────────────────────────────┐
│ <|im_start|>system ... <|im_start|>user Task: ...           │  ← _context_text（旧 prompt 文本）
│ <|im_start|>assistant\n<think></think>                      │     token [1, 2, 3, 4, 5]
├───────────────────────────────────────────────────────────┤
│ <tool>guess(3)</tool><|im_end|>                             │  ← completion.text（模型刚生成的）
│                                                             │     token [50, 51]
├───────────────────────────────────────────────────────────┤
│ \n<|im_start|>user\n3 is too low.<|im_end|>\n               │  ← 新尾部 full_text[len(seen):]
│ <|im_start|>assistant\n<think></think>                      │     编码为 [6, 7, 8]（示意）
└───────────────────────────────────────────────────────────┘
```

```python
prompt_tokens = 旧 prompt + 旧 completion + 新尾部
              = [1, 2, 3, 4, 5] + [50, 51] + [6, 7, 8]
              = [1, 2, 3, 4, 5, 50, 51, 6, 7, 8]
```

**`<tool>guess(3)</tool>` 出现在 `full_text` 中，但不在新尾部里**：它对应的 token `[50, 51]` 已经在上下文中，计算尾部时被 `seen` 减掉了。assistant 消息之所以仍要加入 messages，有两个原因：

1. **渲染需要完整对话**：缺少 assistant 这一轮，模板会把两个 user 消息连在一起，后面的观测和 generation prompt 就接不上。
2. **用于校验**：`full_text.startswith(seen)` 检查模板写回的历史是否与模型当初生成的文本一字不差。一致时直接复用模型自己的 token `[50, 51]`（重新编码可能因 BPE 合并得到不同的 token，前缀关系就会断开），只编码新尾部；不一致时（例如以 `<|endoftext|>` 结束，或删除了 `<think>`）整段重新编码，扩展断开，见 4.3 节。

**② 采样、记录、解析、执行**

```python
Completion(text="<tool>guess(5)</tool><|im_end|>", tokens=[60, 61], logprobs=[-0.3, -0.05])

LLMCall(prompt_tokens=[1, 2, 3, 4, 5, 50, 51, 6, 7, 8],
        completion_tokens=[60, 61],
        completion_logprobs=[-0.3, -0.05])

ToolCall(name="guess", args=(5,))
env.step(...) → Observation(text="Correct! The number was 5.", reward=1.0, done=True)
```

`done=True`，回合结束。

### 2.3 回合产物：Trajectory

```python
Trajectory(
    task=<上面的 Task>,
    calls=[LLMCall_0, LLMCall_1],
    reward=1.0,                 # 最后一个 Observation 的 reward
    num_steps=2,
    info={},                    # 回合中途被放弃时为 {"stop_reason": ...}，此时 reward 强制为 0
)
```

文本层面的 Observation / ToolCall 只用于交互和写 transcript。**训练只用 `calls`（token）和 `reward`。**

---

## 3. 计算优势（advantages.py）

GRPO 对同一个 Task 采一组 rollout。假设 group_size = 4（实际配置为 8），奖励为：

```python
rewards = [1.0, 0.0, 0.0, 1.0]       # 本例的回合是第 1 条
compute_group_advantages(rewards, "grpo")
→ mean = 0.5 → [+0.5, -0.5, -0.5, +0.5]
```

- 本例回合的优势 A = **+0.5**：比组内平均表现好，需要强化。
- 若为 `[1, 1, 1, 1]` 或 `[0, 0, 0, 0]`，则是退化组，返回 `None`，整组不参与训练。
- REINFORCE（group_size = 1）中 A = r = 1.0。

---

## 4. 拼接成 TokenSequence（datagen.py）

### 4.1 拼接过程

| 步骤 | 累积序列 acc | 操作 |
|---|---|---|
| call 0 | `[]` | `extension_suffix([], [1,2,3,4,5])` → `[1,2,3,4,5]`，追加 prompt，再追加 completion `[50,51]` |
| | `[1,2,3,4,5,50,51]` | |
| call 1 | | `extension_suffix(acc, [1,2,3,4,5,50,51,6,7,8])` → `[6,7,8]`，**只追加后缀**，再追加 completion `[60,61]` |
| | `[1,2,3,4,5,50,51,6,7,8,60,61]` | 结束 |

### 4.2 对齐：n 个 token 有 n−1 个预测目标

`tokens` 长度 n = 12；`mask` / `logprobs` / `advantages` 长度均为 11，**第 j 位描述的是 target `tokens[j+1]`**：

| j | 输入 `tokens[j]` | target `tokens[j+1]` | 来源 | mask | logprobs | advantages（广播后） |
|---|---|---|---|---|---|---|
| 0 | 1 | 2 | prompt | 0 | 0.0 | 0.0 |
| 1 | 2 | 3 | prompt | 0 | 0.0 | 0.0 |
| 2 | 3 | 4 | prompt | 0 | 0.0 | 0.0 |
| 3 | 4 | 5 | prompt | 0 | 0.0 | 0.0 |
| **4** | 5 | **50** | **completion 0** | **1** | **−0.4** | **+0.5** |
| **5** | 50 | **51** | **completion 0** | **1** | **−0.1** | **+0.5** |
| 6 | 51 | 6 | 观测尾部 | 0 | 0.0 | 0.0 |
| 7 | 6 | 7 | 观测尾部 | 0 | 0.0 | 0.0 |
| 8 | 7 | 8 | 观测尾部 | 0 | 0.0 | 0.0 |
| **9** | 8 | **60** | **completion 1** | **1** | **−0.3** | **+0.5** |
| **10** | 60 | **61** | **completion 1** | **1** | **−0.05** | **+0.5** |

**off-by-one 规则验证**：completion 0 之前的上下文长度 L = 5，completion 长度 c = 2，对应位置是 `L−1 : L−1+c = 4:6`，即 j = 4、5，而不是 5、6。completion 1 的 L = 10，对应位置 `9:11`。

最终得到：

```python
TokenSequence(
    tokens     = [1, 2, 3, 4, 5, 50, 51, 6, 7, 8, 60, 61],
    mask       = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
    logprobs   = [0, 0, 0, 0, -0.4, -0.1, 0, 0, 0, -0.3, -0.05],
    advantages = [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5],
)
# num_action_tokens = 4
```

- `stitch_trajectory` 产出的 advantages 全为 0；
- `trajectory_to_training_sequences(traj, 0.5)` 再把 0.5 填到所有 mask = 1 的位置。

**mask 的意义**：环境的话（"3 is too low."）mask = 0，不计入 loss。模型只为自己生成的内容负责，不学习预测环境。

### 4.3 对照：扩展断开的情况

假设第 1 轮模型以 `<|endoftext|>`（id 52）而不是 `<|im_end|>` 结束。模板回填历史时写的是 `<|im_end|>`，因此 call 1 的 prompt 被重新编码为 `[1,2,3,4,5,50,51,6,7,8]`，其中第 7 位是 51，而不是模型生成的 52：

```
acc = [1,2,3,4,5,50,52]  不是  [1,2,3,4,5,50,51,6,7,8] 的前缀  → extension_suffix 返回 None
```

结果输出**两条**序列：

```python
TokenSequence(tokens=[1,2,3,4,5,50,52],                mask=[0,0,0,0,1,1])
TokenSequence(tokens=[1,2,3,4,5,50,51,6,7,8,60,61],    mask=[0,0,0,0,0,0,0,0,0,1,1])
```

第二条中的 50、51 来自 prompt，不是本次调用采样的，所以 mask = 0。`preserve_thinking_history: false`（删除历史中的 `<think>`）也会导致同样的断开。

---

## 5. 组成 batch，转成 tinker.Datum

`groups_to_batch` 把所有组、所有回合的序列平铺成一个列表；`train_step` 再逐条调用 `to_tinker_datum`：

```python
model_input   = tokens[:-1]   = [1, 2, 3, 4, 5, 50, 51, 6, 7, 8, 60]   # 11 个输入
target_tokens = tokens[1:]    = [2, 3, 4, 5, 50, 51, 6, 7, 8, 60, 61]  # 11 个目标
logprobs      = seq.logprobs                                            # log π_old（ppo / importance_sampling 使用）
advantages    = seq.advantages × (1 / T)                                # T = 整个 batch 的 action token 总数
```

若整个 batch 只有这一条序列，T = 4，送出的优势就是 0.5 / 4 = 0.125。这样 loss 是 batch 内所有 action token 的平均值，长回合不会因为 token 多而占更大权重。`cross_entropy` 损失不需要 logprobs，优势以 `weights` 字段传入。

后端对每个 mask = 1 的位置计算：

```
ρ_j    = exp(log π_θ(target_j) − logprobs[j])
loss_j = −min(ρ_j·A_j, clip(ρ_j, 1−ε, 1+ε)·A_j)        # ppo
```

`logprobs[j]` 必须恰好对应 `target_tokens[j]`。同步训练时 ρ 应接近 1；若明显偏离，通常说明第 4 节的对齐出错了。

---

## 6. 一图总结

```
Task(goal, max_steps=3, info={target:5})
  │ env.reset()
  ▼
Observation("I'm thinking ... 3 guesses.")
  │ agent.step()  ── render + encode ──▶ prompt [1..5]
  ▼                                      │ policy.complete
LLMCall_0(prompt=[1..5], completion=[50,51], logprobs=[-.4,-.1])
  │ parse
  ▼
ToolCall(guess, 3) ── env.step ──▶ Observation("3 is too low.")
  │ agent.step()  ── 只编码新尾部 [6,7,8] ──▶ prompt [1..5,50,51,6,7,8]
  ▼
LLMCall_1(prompt=[...10 个], completion=[60,61], logprobs=[-.3,-.05])
  │
ToolCall(guess, 5) ── env.step ──▶ Observation("Correct!", reward=1, done)
  ▼
Trajectory(calls=[LLMCall_0, LLMCall_1], reward=1.0)
  │ 组内 GRPO：A = 1.0 − 0.5 = +0.5
  ▼
TokenSequence(tokens 12 个, mask/logprobs/advantages 各 11 个, 4 个 action token)
  │ to_tinker_datum（advantage × 1/T）
  ▼
tinker.Datum ──▶ forward_backward + optim_step
```

---

## 7. 常见疑问：已经有了 Trajectory，为什么还需要 TokenSequence？

两者描述的是同一个回合，但面向不同的使用者：

- **`Trajectory`** 面向 **Agent / 环境侧**，按交互过程记录"发生了什么"，相当于原始日志；
- **`TokenSequence`** 面向 **训练侧**，把日志加工成后端能直接计算 loss 的格式，相当于训练样本。

这类似监督学习中"原始数据"与"预处理后的张量"的区别。需要单独转换一步，原因如下。

### 7.1 Trajectory 中有大量重复的 token

`Trajectory.calls` 中每个 LLMCall 都保存了**完整的 prompt**，后一次调用的 prompt 包含前一次的全部内容：

```
LLMCall_0: prompt = [1,2,3,4,5]                    completion = [50,51]
LLMCall_1: prompt = [1,2,3,4,5,50,51,6,7,8]        completion = [60,61]
                     └── 与 call 0 完全重复 ──┘
```

若每个 LLMCall 单独训练，需要两次前向传播，共处理 7 + 12 = 19 个 token；拼接成一条 TokenSequence 后：

```
tokens = [1,2,3,4,5,50,51,6,7,8,60,61]    一次前向传播，处理 12 个 token
```

本例只有 2 轮，差别不大。TextCraft 一个回合最多 75 轮且开启思维链，prompt 越来越长：单独训练的计算量随轮数**二次增长**，拼接后只**线性增长**。这正是 sequence extension 的意义，`stitch_trajectory` 就是执行拼接的那一步。

### 7.2 优势无法存放在 Trajectory 中

```
Trajectory.reward        = 1.0     ← 只由这个回合决定
TokenSequence.advantages = +0.5    ← 由这个回合与同组其它回合共同决定
```

GRPO 的优势是 `r_i − 组内平均`，必须等同组 8 条 rollout 全部完成才能计算。`run_episode` 生成 Trajectory 时还不知道其它 rollout 的结果，所以只能存奖励。优势在 `groups_to_batch` 中按组计算，再写入 TokenSequence。

### 7.3 后端需要按"预测位置"对齐的数据

后端计算 loss 时逐位置考虑："给定 `tokens[0..j]` 预测 `tokens[j+1]`，这个位置是否计入 loss、用哪个旧 logprob、乘哪个优势"。

Trajectory 没有这种按位置组织的信息：

- logprob 只附在各次 completion 上，不知道它们落在拼接后序列的哪个位置；
- 没有 mask，不区分哪些 token 是模型生成的、哪些是环境给出的。

TokenSequence 已经把这些**按 target 位置对齐**（`mask` / `logprobs` / `advantages` 长度均为 n−1，第 j 位对应 `tokens[j+1]`），因此 `to_tinker_datum` 几乎只需原样转换（见第 5 节）。

### 7.4 两者不是一对一的关系

| 情况 | 一个 Trajectory 产出的 TokenSequence 数量 |
|---|---|
| 扩展一直成立（正常情况） | 1 条 |
| 扩展中途断开（`<|endoftext|>` 结束、删除 `<think>`） | 多条 |
| 所在组为退化组（GRPO 组内奖励全部相同） | 0 条 |
| 没有任何 action token 可训练 | 0 条 |

训练侧只关心"有哪些可训练的序列"，不关心它们来自哪个回合。因此 `groups_to_batch` 最终返回的是一个**平铺的 `list[TokenSequence]`**，回合边界在此已不存在。

### 7.5 解耦：两侧可以各自独立变化

```
Agent / Env 侧                              训练侧
──────────────                              ──────
可以更换 Agent、环境、                       tinker_adapter 只认识 TokenSequence，
prompt 格式                                  不需要知道 Agent 如何交互
        │                                            ▲
        └──── Trajectory ── datagen ── TokenSequence ─┘
                     （唯一需要同时了解两侧的地方）
```

以后如果换一种 Agent（例如带子 Agent 的、用 JSON 调用工具的），只要它能产出 Trajectory 或直接产出 TokenSequence，训练代码无需改动。`types.py` 将 TokenSequence 称为 "the unit of training data"，说的就是这种分工。

### 7.6 对比总结

| | Trajectory | TokenSequence |
|---|---|---|
| 回答的问题 | 这个回合发生了什么？ | 模型应该在哪些 token 上、朝哪个方向更新？ |
| 组织方式 | 按 LLM 调用（每轮一个 LLMCall） | 一条平铺的 token 序列 |
| 存储效率 | prompt 大量重复 | 每个 token 只出现一次 |
| 奖励信号 | `reward`（本回合自身的标量） | `advantages`（相对本组，按 token 展开） |
| 是否区分 token 的生成者 | 隐式（prompt 与 completion 分开存） | 显式（`mask`） |
| 是否按预测位置对齐 | 否 | 是（长度 n−1） |
| 创建者 | `run_episode`（给定代码） | `datagen`（Part 1，由你实现） |
| 使用者 | 奖励统计、pass@k、transcript；作为 datagen 的输入 | `train_step` → `to_tinker_datum` → 后端 |
| 与回合的对应关系 | 1 个回合 = 1 个 | 1 个回合 = 0、1 或多个 |

一句话：**Trajectory 是交互日志，TokenSequence 是训练样本**。Part 1 的任务就是实现"日志 → 样本"的转换。

---

## 8. 常见疑问：为什么不直接用最后一次 LLMCall 的 prompt 拼上 completion？

在扩展一直成立时，这样做确实能得到同样的 `tokens`，但还不够：

1. 只有 `tokens` 不够，`mask` 和 `logprobs` 仍然需要回头遍历每一个 LLMCall 才能填出来；
2. 扩展一旦中途断开，最后一次调用的 prompt 里就**不再包含**早先模型真正生成的 token，这时用捷径得到的训练数据是错的。

### 8.1 正常情况：tokens 确实一样

```
最后一次调用的 prompt + completion = [1,2,3,4,5,50,51,6,7,8] + [60,61]
                                   = [1,2,3,4,5,50,51,6,7,8,60,61]   ← 和拼接结果完全相同
```

这不是巧合：扩展成立时，最后一次调用的 prompt 本来就是一层层累积起来的。

### 8.2 问题 1：最后一次的 prompt 不会告诉你哪些 token 是模型生成的

```
[1,2,3,4,5, 50,51, 6,7,8, 60,61]
            ^^^^^          ^^^^^
            call 0 生成的，logprob = [-0.4, -0.1]   ← 只看最后一次调用无法得知
```

在最后一次调用的 prompt 里，`50, 51` 和环境观测 `6, 7, 8` 没有任何区别，都只是 prompt 的一部分。要知道 `50, 51` 是模型生成的、处在哪个位置、logprob 是多少，只能回去看 `calls[0]`：

```
calls[0] 的 completion 位置 = 从 len(calls[0].prompt_tokens) − 1 开始，共 len(calls[0].completion_tokens) 个
                           = 位置 4、5
```

所以无论如何都要遍历所有 LLMCall。"取最后一次的 tokens，再按每次调用的 prompt 长度算出 completion 位置"本身可以作为一种实现方式，但它成立的前提是**每一步的前缀关系都成立**，而这个前提必须逐步检查（见 8.3）。

### 8.3 问题 2：扩展断开时，捷径会产生错误的数据

**情况 A：模型以 `<|endoftext|>` 结束**

```
call 0 实际生成：      completion = [50, 52]        # 52 = <|endoftext|>，logprob = [-0.4, -0.1]
模板写回历史时写的是：              [50, 51]        # 51 = <|im_end|>

最后一次的 prompt = [1,2,3,4,5, 50,51, 6,7,8]        ← 里面是 51，不是模型真正生成的 52
```

若用捷径把位置 5 标为 action 并填入 logprob −0.1，就是拿 52 的旧 logprob 去配 51 这个 target。计算 ρ = π_θ(51) / π_old(52) 时，分子分母不是同一个 token，ρ 会离谱，梯度也是错的；而模型真正生成的 52 完全没有被训练到。

正确做法是检测到断开，拆成两条（同 4.3 节）：

```
序列 1：[1,2,3,4,5,50,52]               mask=[0,0,0,0,1,1]            ← 训练模型真正生成的 52
序列 2：[1,2,3,4,5,50,51,6,7,8,60,61]   mask=[0,0,0,0,0,0,0,0,0,1,1]  ← 50、51 当作 prompt，不训练
```

**情况 B：`preserve_thinking_history: false`（删除历史中的 `<think>`）**

TextCraft 开启了思维链。假设第 1 轮模型生成：

```
call 0 completion = "<think>目标需要 a1_i5，先查配方……</think><tool>get_info('a1_i5')</tool>"
```

到最后一次调用时，模板已把历史中的 `<think>...</think>` **整段删除**：

```
最后一次的 prompt = [... <tool>get_info('a1_i5')</tool> ... 观测 ...]   ← 推理过程已不存在
```

如果只看最后一次调用，第 1 轮到倒数第 2 轮的**全部推理 token 都会丢失**，永远得不到强化或抑制，而 TextCraft 的推理往往正是决定成败的关键。

### 8.4 结论：stitch_trajectory 就是"带校验的捷径"

```
for each call:
    if 前缀关系成立:   接着拼（即这个捷径，逐步执行）
    else:              结束当前序列，从这次调用的完整 prompt 重新开始
```

- 扩展一直成立：结果与"最后一次 prompt + completion"完全相同，只是额外填好了 mask 和 logprobs；
- 扩展中途断开：能检测出来并正确拆分，而捷径会悄悄产生错误的数据。

这就是 `extension_suffix` 需要返回 `None` 的原因：它是每一步的校验。`datagen.py` docstring 中的第二个例子考查的正是断开的情况，隐藏测试也一定会覆盖。
