# Part 1：数据管线解答思路

> 配合 [dataflow.md](dataflow.md)（完整示例）和 [template.md](template.md)（模板、停止符、思维链）阅读。本文整理 Part 1 各函数的契约、解答思路与常见坑。以各 stub 的 docstring 为准；本文只描述思路，不提供完整实现。

Part 1 实现两个文件：

| 文件 | 函数 |
|---|---|
| [scottyrl/advantages.py](../scottyrl/advantages.py) | `compute_group_advantages` |
| [scottyrl/datagen.py](../scottyrl/datagen.py) | `extension_suffix`、`stitch_trajectory`、`trajectory_to_training_sequences` |

---

## 1. 整体结构：datagen 中的三层调用

[datagen.py](../scottyrl/datagen.py) 负责把一个回合的交互记录（Trajectory）加工成训练样本（TokenSequence）。三个函数层层调用：

```
groups_to_batch（common.py，Part 2）
   └─▶ trajectory_to_training_sequences(traj, advantage)   第 3 层：填入优势
          └─▶ stitch_trajectory(traj)                      第 2 层：拼接 + mask + logprobs（核心）
                 └─▶ extension_suffix(acc, prompt)         第 1 层：每一步的前缀校验
```

另一个函数 `compute_group_advantages` 由 `groups_to_batch` 调用，算出每条轨迹的优势后，再传给第 3 层。

下面从底层往上讲。

---

## 2. 第 1 层：`extension_suffix(accumulated, prompt)`

> 契约：若 `accumulated` 是 `prompt` 的（非严格）前缀，返回剩余后缀（可为空）；否则返回 `None`。

### 2.1 它回答的问题

> 已经拼好的序列 `accumulated`，是不是下一次调用的 `prompt` 的开头？如果是，prompt 还多出哪些 token？

```
accumulated = [1, 2, 3, 4, 5, 50, 51]                ← 已拼好：call 0 的 prompt + completion
prompt      = [1, 2, 3, 4, 5, 50, 51, 6, 7, 8]       ← call 1 的 prompt
              └────────── 相同 ──────────┘ └─后缀─┘
返回 [6, 7, 8]
```

### 2.2 先看懂每个 token 从哪里来

（id 为示意，一段文字实际会被切成很多 token）

| token | 内容 | 谁写的 |
|---|---|---|
| 1–4 | system 消息、user 消息、`<\|im_start\|>assistant\n` | 模板 |
| 5 | prefill：空思考块 `<think>\n\n</think>\n\n` | 模板 |
| 50 | `<tool>guess(3)</tool>` | **模型** |
| 51 | `<\|im_end\|>`（停止符） | **模型** |
| 6 | `\n<\|im_start\|>user\n` | 模板 |
| 7 | `3 is too low.` | 环境（经模板包装） |
| 8 | `<\|im_end\|>\n<\|im_start\|>assistant\n` + prefill | 模板 |

- `accumulated` = call 0 的 prompt（1–5）+ call 0 的回复（50、51），即**第一轮模型实际看到的内容和它实际说的话**。
- `prompt` = call 1 的 prompt，是 Agent 在第二轮**重新渲染整段对话**后得到的。

所以 `extension_suffix` 真正要回答的是：**第二轮重新渲染的 prompt，开头是否原封不动地保留了第一轮的全部内容？** 若是，多出的 `[6, 7, 8]` 就是新一轮新增的部分（turn 标记、环境反馈、下一轮的 prefill）。

### 2.3 为什么会"不保留"：模板写回历史的过程

第二轮 prompt 中，第一轮的回复不是直接复制过去的，而是经过了"**去掉停止符 → 模板写回历史时加回 `<|im_end|>`**"的过程：

- **消息列表**中要去掉停止符：模板写回 assistant 历史的规则是 `<|im_start|>assistant\n + prefill + 内容 + <|im_end|>\n`，结尾的 `<|im_end|>` 由模板自己加，不去掉就会重复；
- **token 上下文**中保留停止符：Agent 直接复用 `completion.tokens`，停止符作为模型生成的 token，在训练中 mask = 1。

只要这"一去一加"不能互相抵消，或者模板删改了历史，开头就不再相同。下面在 token 层面逐个看四种情况。

**情况 A：以 `<|im_end|>` 结束 → 返回后缀 ✅**

```
call 0 回复: [50, 51]                         # 51 = <|im_end|>
去掉 51，模板再加回 <|im_end|>，正好抵消

accumulated = [1, 2, 3, 4, 5, 50, 51]
prompt      = [1, 2, 3, 4, 5, 50, 51, 6, 7, 8]
              └──────── 完全相同 ────────┘
返回 [6, 7, 8]
```

**情况 B：写满 `max_tokens` 被截断（没有停止符）→ 返回后缀 ✅**

```
call 0 回复: [50]                             # 没有停止符
没有可去掉的；模板写回时加上 <|im_end|>

accumulated = [1, 2, 3, 4, 5, 50]
prompt      = [1, 2, 3, 4, 5, 50, 51, 6, 7, 8]
              └───── 相同 ─────┘ └── 后缀 ──┘
返回 [51, 6, 7, 8]
```

51（`<|im_end|>`）落在后缀中。它是模板补上的、不是模型生成的，所以训练序列中 mask = 0。

**情况 C：以 `<|endoftext|>` 结束 → 返回 None ❌**

```
call 0 回复: [50, 52]                         # 52 = <|endoftext|>
去掉的是 52，模板加回的却是 51（<|im_end|>）

accumulated = [1, 2, 3, 4, 5, 50, 52]
prompt      = [1, 2, 3, 4, 5, 50, 51, 6, 7, 8]
                                  ↑ 52 ≠ 51
返回 None
```

`<|im_end|>` 表示"一条消息结束"，`<|endoftext|>` 表示"整篇文档结束"（来自预训练）。模板只认识前者，所以模型偶尔用后者结束时，扩展就会断开。

**情况 D：开启思考，且删除历史中的思考 → 返回 None ❌**

开启思考时，5 是 prefill `<think>\n`，模型回复中带有推理：

```
call 0 回复: [40, 41, 50, 51]                 # 40 = 推理文字，41 = </think>
模板写回历史时，把 prefill 和推理整段删除

accumulated = [1, 2, 3, 4, 5, 40, 41, 50, 51]
prompt      = [1, 2, 3, 4, 50, 51, 6, 7, 8]
                          ↑ 5 ≠ 50
返回 None
```

删除思考后，新 prompt 往往还比 `accumulated` **更短**，对应 docstring 中 `([1,2,3], [1,2])` 返回 `None` 的情况。

### 2.4 docstring 中的 5 个例子对应的实际场景

| 调用 | 返回 | 实际场景 |
|---|---|---|
| `([1,2,3], [1,2,3,9,9])` | `[9, 9]` | 情况 A、B：正常扩展 |
| `([1,2,3], [1,2,3])` | `[]` | 新增部分为空。实际中很少见（新一轮总有 turn 标记），但契约要求返回空列表而非 `None` |
| `([1,2,3], [1,9,3,4])` | `None` | 情况 C、D：中间某个 token 被替换 |
| `([1,2,3], [1,2])` | `None` | 情况 D：历史被删短，prompt 比已拼好的还短 |
| `([], [1,2])` | `[1, 2]` | 第一次调用，或扩展断开后重新开始：后缀就是完整的 prompt |

### 2.5 两个设计细节

**为什么返回后缀，而不是 True / False？** 调用方判断完马上就要追加这段后缀，直接返回后缀可以一步完成"判断"和"取出新增部分"。用 `None` 表示失败，正好能与"成功但后缀为空" `[]` 区分。所以调用方必须写 `if suffix is None`，而不是 `if not suffix`：后者会把 `[]` 误判为失败。

**为什么 `([], prompt)` 返回整个 prompt？** 这让 `stitch_trajectory` 不必为第一次调用写特殊逻辑：初始 `accumulated = []`，第一次调用的"后缀"自然就是完整 prompt；扩展断开后重新开始，同样从 `[]` 出发，套用同一套逻辑。

### 2.6 Agent 已经检查过一次，datagen 为什么还要再查

| | Agent 的 `_extend_context` | datagen 的 `extension_suffix` |
|---|---|---|
| 时机 | 回合进行中，每轮构造 prompt 时 | 回合结束后，拼接训练数据时 |
| 比较的层面 | **文本**：`full_text.startswith(旧文本 + 回复文本)` | **token**：`prompt[:len(acc)] == acc` |
| 结果的用途 | 决定复用旧 token 只编码尾部，还是整段重新编码 | 决定接着拼，还是另起一条序列 |

两者结论一致：Agent 判断文本一致时，会把旧 token 原样放在新 prompt 开头，token 前缀自然成立；判断不一致时会整段重新编码，旧回复的 token 被模板版本取代，token 前缀就对不上了。

但 **Agent 的判断结果没有记录在 Trajectory 中**：LLMCall 只存 prompt、completion 和 logprobs，没有"这一轮扩展是否成立"的标记。datagen 只能根据 token 数据重新推断，这就是 `extension_suffix` 存在的原因。

### 2.7 解答思路

1. 若 `accumulated` 比 `prompt` 长，直接返回 `None`；
2. 比较 `prompt` 的前 `len(accumulated)` 个 token 是否与 `accumulated` 完全相同；
3. 相同则返回 `prompt` 剩余部分（新列表，可能为空），否则返回 `None`。

### 2.8 小结

```
accumulated = 之前所有轮次中，模型实际看到的 + 实际生成的（包括它用的停止符）
prompt      = 下一轮重新渲染的 prompt（历史经过"去掉停止符 → 模板加回 <|im_end|>"，可能还删了思考）

开头一致   → 返回多出的部分（新一轮的 turn 标记、环境反馈、prefill，以及截断时补上的 <|im_end|>）
开头不一致 → 返回 None（停止符被替换、思考被删除）
```

一句话：**它检查的是模板重新渲染的历史，是否原封不动地保留了模型当时真正看到和说出的每一个 token。**只有保留了，才能把下一轮接在同一条训练序列上，否则训练时的上下文就不是模型当时真正看到的上下文。

---

## 3. 第 2 层：`stitch_trajectory(traj)`

这是 Part 1 的核心函数：把一个回合的多次 LLM 调用拼成一条（或几条）训练序列，同时标出哪些 token 由模型生成、对应的采样 logprob 是多少。

### 3.1 输入与输出

**输入**：`traj.calls = [LLMCall_0, LLMCall_1, ...]`，按调用顺序排列。每个 LLMCall 包含：

| 字段 | 含义 | 长度关系 |
|---|---|---|
| `prompt_tokens` | 这次调用模型实际收到的完整 prompt | — |
| `completion_tokens` | 模型这次生成的 token（含停止符，若有） | 记为 c |
| `completion_logprobs` | 每个生成 token 的采样 logprob（log π_old） | 也是 c，与 `completion_tokens` 一一对应 |

**输出**：`list[TokenSequence]`，可能为空。每条序列满足：

| 字段 | 长度 | 内容 |
|---|---|---|
| `tokens` | n | 拼接后的完整序列 |
| `mask` | n − 1 | 第 j 位为 1 当且仅当 `tokens[j+1]` 来自某次 completion |
| `logprobs` | n − 1 | mask = 1 处为采样 logprob，其余为 0.0 |
| `advantages` | n − 1 | **全部为 0.0**（由第 3 层再填） |

### 3.2 需要维护的状态

整个函数只需维护两样东西：

1. **结果列表** `results`：已经完成的 TokenSequence；
2. **当前正在拼的序列**，由三个**与 tokens 等长**的并行列表组成：

| 列表 | 每追加一个 token 时追加的值 |
|---|---|
| `cur_tokens` | 这个 token |
| `cur_is_action` | 1（模型生成的）或 0（模板 / 环境写的） |
| `cur_logprobs` | 采样 logprob（模型生成的）或 0.0 |

之所以让三个列表与 tokens **等长**，而不是直接维护长度 n − 1 的 mask，是为了避免 off-by-one：追加时只需关心"这个 token 是谁写的"，对齐留到收尾时一次处理（见 3.4）。

### 3.3 主循环

```
results = []
当前序列 = 空（三个空列表）

for call in traj.calls:
    ① suffix = extension_suffix(cur_tokens, call.prompt_tokens)

    ② if suffix is None:                     # 扩展断开
           收尾(当前序列)                      # 见 3.4：若有效则放入 results
           当前序列 = 空
           suffix = call.prompt_tokens        # 等价于 extension_suffix([], prompt)

    ③ 对 suffix 中的每个 token：
           追加 (token, 是否生成=0, logprob=0.0)

    ④ 对 completion 中的每个 (token, logprob)：
           追加 (token, 是否生成=1, logprob)

收尾(当前序列)                                # 循环结束后，最后一条也要收尾
return results
```

每一步的作用：

| 步骤 | 作用 | 对应的 token 来源 |
|---|---|---|
| ① | 判断这次调用的 prompt 是否以"当前序列"开头 | — |
| ② | 不是：先保存旧序列，再从这次调用的完整 prompt 重新开始 | — |
| ③ | 追加 prompt 中新增的部分 | 模板写的 turn 标记、环境观测、prefill；截断时补上的 `<\|im_end\|>`；断开后重新开始时的整个 prompt |
| ④ | 追加模型的回复 | 模型生成的 token，包括它输出的停止符 |

第一次调用时当前序列为空，`extension_suffix([], prompt)` 返回整个 prompt，因此第一次调用不需要任何特殊处理。

### 3.4 收尾：对齐、过滤、复制

"收尾"把当前序列转成 TokenSequence，要做三件事。

**① 对齐：去掉辅助列表的第一个元素**

`mask[j]` 描述的是 `tokens[j+1]`，所以把等长的 `cur_is_action`、`cur_logprobs` 各去掉第一个元素，就得到长度 n − 1、按 target 对齐的 `mask` 和 `logprobs`：

```
tokens        = [1,  2,  3,  4,  5,  50,   51,   6,  7,  8,  60,   61  ]
cur_is_action = [0,  0,  0,  0,  0,  1,    1,    0,  0,  0,  1,    1   ]   ← 与 tokens 等长
cur_logprobs  = [0,  0,  0,  0,  0, -0.4, -0.1,  0,  0,  0, -0.3, -0.05]

去掉第一个元素：
mask          = [    0,  0,  0,  0,  1,    1,    0,  0,  0,  1,    1   ]   ← 长度 11，第 j 位对应 tokens[j+1] ✓
logprobs      = [    0,  0,  0,  0, -0.4, -0.1,  0,  0,  0, -0.3, -0.05]
advantages    = [    0,  0,  0,  0,  0,    0,    0,  0,  0,  0,    0   ]   ← 全 0，长度 11
```

这与 docstring 中"长度为 L 的上下文后接长度为 c 的 completion，位置是 `L−1 : L−1+c`"完全等价：completion 0 前面有 L = 5 个 token，它的第一个 token 在等长列表中位于下标 5，去掉第一个元素后就到了下标 4 = L − 1。

被去掉的第一个元素描述的是 `tokens[0]`。它没有"前一个 token"，不可能作为预测目标，丢掉不会损失信息。

**② 过滤：两种序列不输出**

| 条件 | 为什么不输出 |
|---|---|
| token 少于 2 个 | 没有任何预测目标；`TokenSequence.__post_init__` 也会直接抛 `ValueError` |
| 对齐后的 `mask` 中没有任何 1 | 没有模型生成的 token，对训练没有贡献；docstring 明确要求不输出 |

注意：判断"有没有动作"要看**对齐后**的 mask，而不是等长的 `cur_is_action`。两者只在第一个 token 由模型生成时才会不同（例如某次调用的 prompt 为空），这在实际中几乎不会发生，但按对齐后的 mask 判断才严格符合契约。

**③ 复制：输出独立的列表**

放入 TokenSequence 的必须是新列表。如果当前序列一开始直接引用了 `call.prompt_tokens`，之后又原地追加，就会改坏 Trajectory 中保存的原始数据。

### 3.5 逐步演示 1：扩展一直成立

使用 [dataflow.md](dataflow.md) 的例子（Number Search，2 轮）：

```
call 0: prompt=[1,2,3,4,5],               completion=[50,51], logprobs=[-0.4,-0.1]
call 1: prompt=[1,2,3,4,5,50,51,6,7,8],   completion=[60,61], logprobs=[-0.3,-0.05]
```

| 步骤 | `extension_suffix` 的参数与返回 | 追加的内容 | 执行后的 `cur_tokens` |
|---|---|---|---|
| call 0 ① | `([], [1,2,3,4,5])` → `[1,2,3,4,5]` | — | `[]` |
| call 0 ③ | — | `1,2,3,4,5`（是否生成 = 0） | `[1,2,3,4,5]` |
| call 0 ④ | — | `50,51`（是否生成 = 1，logprob −0.4、−0.1） | `[1,2,3,4,5,50,51]` |
| call 1 ① | `([1,2,3,4,5,50,51], [1,2,3,4,5,50,51,6,7,8])` → `[6,7,8]` | — | 不变 |
| call 1 ③ | — | `6,7,8`（是否生成 = 0） | `[1,2,3,4,5,50,51,6,7,8]` |
| call 1 ④ | — | `60,61`（是否生成 = 1，logprob −0.3、−0.05） | `[1,2,3,4,5,50,51,6,7,8,60,61]` |
| 循环结束 | 收尾：12 个 token、4 个动作，有效 | — | 输出 1 条序列 |

结果：**1 条序列**，即 3.4 节中的 tokens / mask / logprobs。

### 3.6 逐步演示 2：扩展中途断开

模型第一轮以 `<|endoftext|>`（52）结束（见 2.3 节情况 C）：

```
call 0: prompt=[1,2,3,4,5],               completion=[50,52], logprobs=[-0.4,-0.1]
call 1: prompt=[1,2,3,4,5,50,51,6,7,8],   completion=[60,61], logprobs=[-0.3,-0.05]
                            ↑ 模板写回的是 51，不是 52
```

| 步骤 | `extension_suffix` 的参数与返回 | 动作 | 执行后的 `cur_tokens` |
|---|---|---|---|
| call 0 ①③④ | `([], [1,2,3,4,5])` → `[1,2,3,4,5]` | 追加 prompt，再追加 `50,52` | `[1,2,3,4,5,50,52]` |
| call 1 ① | `([1,2,3,4,5,50,52], [1,2,3,4,5,50,51,6,7,8])` → **`None`** | — | 不变 |
| call 1 ② | — | 收尾旧序列：7 个 token、2 个动作，有效 → **输出序列 1**；清空；suffix = 完整 prompt | `[]` |
| call 1 ③ | — | 追加 `1,2,3,4,5,50,51,6,7,8`（**全部**是否生成 = 0） | `[1,2,3,4,5,50,51,6,7,8]` |
| call 1 ④ | — | 追加 `60,61`（是否生成 = 1） | `[1,2,3,4,5,50,51,6,7,8,60,61]` |
| 循环结束 | 收尾：12 个 token、2 个动作，有效 → **输出序列 2** | — | — |

结果：**2 条序列**。

```
序列 1: tokens=[1,2,3,4,5,50,52]               mask=[0,0,0,0,1,1]
序列 2: tokens=[1,2,3,4,5,50,51,6,7,8,60,61]   mask=[0,0,0,0,0,0,0,0,0,1,1]
```

- 序列 1 训练模型第一轮**真正生成**的 `50, 52`；
- 序列 2 中的 `50, 51` 是模板写回的历史，属于 call 1 的 prompt，mask = 0，不会重复训练；
- 第一轮的回复没有丢失，也没有被训练两次。

docstring 的第二个例子是同样的情况，只是 token 更少：

```
call 0: prompt=[10, 11],     completion=[20, 21]
call 1: prompt=[10, 11, 30], completion=[40]

→ 序列 1: tokens=[10, 11, 20, 21]  mask=[0, 1, 1]
→ 序列 2: tokens=[10, 11, 30, 40]  mask=[0, 0, 1]
```

### 3.7 为什么要和"当前序列"比较，而不是和上一次调用的 prompt 比较

当前序列包含了**上一次调用的 completion**。如果错误地拿上一次调用的 prompt 来比较：

```
extension_suffix(call0.prompt, call1.prompt)
  = extension_suffix([1,2,3,4,5], [1,2,3,4,5,50,51,6,7,8])
  = [50, 51, 6, 7, 8]                       ← 后缀里混进了上一轮的回复
```

再把这个后缀追加到当前序列 `[1,2,3,4,5,50,51]` 后面，就会得到：

```
[1,2,3,4,5, 50,51, 50,51, 6,7,8, 60,61]
            ^^^^^  ^^^^^ 重复了
```

序列中出现了模型从未见过的内容，后面所有位置的上下文都是错的。

另外，这种写法也**检测不出**情况 C：`[1,2,3,4,5]` 仍是 call 1 prompt 的前缀，于是 52 被 51 替换的问题被漏掉了。只有拿包含 completion 的当前序列去比较，检查的才是"下一个 prompt = 之前全部内容 + 新尾部"。

### 3.8 特殊情况一览

| 情况 | 结果 | 说明 |
|---|---|---|
| `traj.calls` 为空 | `[]` | 循环不执行，收尾时当前序列为空，被过滤 |
| 只有一次调用 | 1 条 | prompt（mask 0）+ completion（mask 1） |
| 扩展一直成立 | 1 条 | 整个回合一条序列，最常见的情况 |
| 中途断开一次 | 2 条 | 见 3.6 |
| 每一轮都断开（开启思考且 `preserve_thinking_history: false`） | 每轮 1 条 | 每条序列 = 该轮完整 prompt + 该轮回复 |
| 某次 completion 被截断（无停止符） | 不断开 | 下一轮的后缀中会包含模板补上的 `<\|im_end\|>`，mask = 0（2.3 节情况 B） |
| 某次 completion 为空 | 不贡献动作 | 若整条序列因此没有动作，则被过滤 |
| 被放弃的回合（`stop_reason`） | 照常处理 | `run_episode` 已删除失败那一轮的调用，`calls` 中只有完整轮次 |
| 断开发生时当前序列为空 | 不会发生 | `extension_suffix([], …)` 永远不返回 `None` |

### 3.9 对照 docstring 的检查清单

- [ ] 按顺序处理每个 call，扩展成立时只追加新的 prompt 后缀，再追加 completion；
- [ ] 扩展断开时结束当前序列，从该 call 的**完整** prompt 开始新序列；
- [ ] `mask` 恰好在 completion token 对应的 target 位置为 1；
- [ ] `logprobs` 在这些位置为采样 logprob，其余为 0.0；
- [ ] `advantages` 全为 0.0；
- [ ] `mask` / `logprobs` / `advantages` 长度均为 `len(tokens) − 1`；
- [ ] 不输出少于 2 个 token 或没有动作的序列；
- [ ] 循环结束后输出最后一条序列；
- [ ] 不修改 `traj` 中的原始列表；
- [ ] 两个 docstring 例子的输出完全一致。

---

## 4. 第 3 层：`trajectory_to_training_sequences(traj, advantage)`

```
seqs = stitch_trajectory(traj)
对每条序列：mask = 1 的位置填 advantage，其余位置保持 0.0
```

一个回合中所有由模型生成的 token 共享同一个回合级的优势：REINFORCE / GRPO 只知道整个回合成功与否，无法区分是哪一步做对了。

**为什么不在 `stitch_trajectory` 中直接填？** 拼接只需要这一个回合本身的信息，而优势需要整组回合的奖励。拆成两个函数后，拼接逻辑可以独立测试，优势由上层 `groups_to_batch` 算好后再传入。

**advantage = 0.0 时也要输出序列**：`groups_to_batch` 的契约要求"每条剩余轨迹都贡献它的训练序列，即使优势为 0.0"。只有退化组（`compute_group_advantages` 返回 `None`）才整组跳过，这个判断在上层做，不在这里。

---

## 5. `compute_group_advantages(rewards, method)`

| method | 行为 |
|---|---|
| `"reinforce"` | 每条轨迹的优势就是它的原始奖励，无 baseline |
| `"grpo"` | 组大小 < 2 时抛 `ValueError`；奖励**完全相等**（无容差）时返回 `None`（退化组）；否则返回 `r_i − mean(r)` |
| 其它 | 抛 `ValueError` |

要点：

- 返回普通 Python `list[float]`，与 `rewards` 等长、同序；
- 遵循 Dr.GRPO，**不除以标准差**；
- 例：`[1, 0, 0, 1]` → 均值 0.5 → `[+0.5, −0.5, −0.5, +0.5]`；`[1, 1, 1, 1]` → `None`。

---

## 6. 常见坑

| 坑 | 后果 |
|---|---|
| 用 `if not suffix` 判断失败 | 把"后缀为空"误判为断开，多拆出序列 |
| mask 与 tokens 等长，或下标从 L 开始而不是 L−1 | logprob 错位，同步训练时 ρ 明显偏离 1 |
| 只拿上一次调用的 prompt 比较，而非当前累积的序列 | 漏掉 completion，前缀判断出错 |
| 断开时忘记先输出当前序列 | 之前几轮的训练数据丢失 |
| 循环结束后忘记输出最后一条序列 | 每个回合最后一段数据丢失（最常见的情况是整个回合都丢了） |
| 输出没有动作 token 的序列 | 违反契约；对训练无贡献，只浪费计算 |
| advantage 为 0 时直接返回空列表 | 违反 `groups_to_batch` 的契约 |
| 直接修改 `traj.calls` 中的列表（例如对 `prompt_tokens` 原地 `extend`） | 改坏原始数据，后续调用出错；初始化当前序列时要先复制 |
| GRPO 判断退化组时使用容差，或组大小 < 2 未报错 | 违反契约 |

---

## 7. 测试

```bash
uv run pytest tests/test_interfaces.py -k "advantages or datagen"
```

公开测试只检查形状。建议额外：

- 用 docstring 中的两个拼接例子和 `extension_suffix` 的 5 个例子手动验证；
- 用 `tests/test_envs.py` 中的 `ThinkStrippingPolicy` 构造一个扩展断开的回合来测试，隐藏测试一定会覆盖断开的情况；
- 自己构造 2.3 节的情况 B（截断）和情况 C（`<|endoftext|>`），检查后缀中补上的 `<|im_end|>` 的 mask 为 0，以及断开后产出两条序列。
