# 模板、历史、停止符与思维链

> 配合 [dataflow.md](dataflow.md) 和 [part0.md](part0.md) 阅读。本文从最基础的概念开始，用同一个猜数字的例子，逐步解释对话模板、生成提示、停止符、思维链、历史，最后说明这些概念为什么会导致"序列扩展"断开。模板规则以 [qwen3_5.jinja](../scottyrl/chat_templates/qwen3_5.jinja) 为准。

---

## 1. 起点：模型只会"续写一串 token"

大语言模型本质上只做一件事：**给定一串 token，预测下一个 token**。它没有"对话"、"角色"、"轮次"这些概念，看到的永远是一条长序列：

```
输入: [t0, t1, t2, ..., tn]  →  模型  →  预测 t(n+1)
```

生成一段回复，就是不断重复这一步：预测一个 token，接到末尾，再预测下一个……直到某个时刻停下。

那"聊天"从何而来？答案是：**把对话写成一段有固定格式的文本**，让模型续写。负责写这段文本的就是"模板"。

---

## 2. 模板：把消息列表变成一段文本

程序中的对话是结构化的消息列表：

```python
messages = [
  {"role": "system", "content": "You are Scotty ... 工具说明 ..."},
  {"role": "user",   "content": "Task: Guess the correct number between 1 and 6.\nI'm thinking of ..."},
]
```

**chat template（对话模板）**是一段规则，规定如何把这个列表写成文本。本作业模板的规则是：每条 system / user 消息写成

```
<|im_start|>角色\n内容<|im_end|>\n
```

渲染（`render`）结果如下：

```
<|im_start|>system
You are Scotty, an agent that solves tasks by calling tools. ...<|im_end|>
<|im_start|>user
Task: Guess the correct number between 1 and 6.
I'm thinking of a number between 1 and 6 (inclusive). You have 3 guesses.<|im_end|>
```

- `<|im_start|>`、`<|im_end|>` 是**特殊 token**，分别表示"一条消息开始"和"一条消息结束"，各自在词表中占一个 token。
- Qwen 在训练时见过海量这种格式的对话，所以它"认识"这个格式：`<|im_start|>user` 之后是用户的话，`<|im_start|>assistant` 之后该轮到自己说了。

**为什么必须用模型自己的模板？** 因为模型只在这种格式上训练过。换一种格式（例如 `User: ... Assistant: ...`），模型的表现会明显变差。

---

## 3. 生成提示：告诉模型"轮到你了"

上面的文本以 `<|im_end|>` 结束，模型并不知道接下来该谁说话。因此模板会在末尾再加一段 **generation prompt（生成提示）**：

```
<|im_start|>assistant
<think>

</think>

```

含义是："现在是 assistant 的回合，请从这里开始写。"后面的空 `<think>` 块见第 5 节。

完整的 prompt = 对话文本 + 生成提示。将其编码成 token 发给模型，模型就开始续写。

---

## 4. 停止符：模型怎么知道该停了

模型逐个 token 往后写：

```
<tool>  guess  (3)  </tool>  <|im_end|>
```

写到 `<|im_end|>` 时，采样器停止生成。模型在训练中学会了"说完一轮话就输出 `<|im_end|>`"，所以是它自己"喊停"的。

**停止符（stop sequence）**就是"一旦生成这个 token 就停止"的设置。本作业配置了两个（`ModelConfig.stop_sequences`）：

| 停止符 | 本来的含义 | 正常情况 |
|---|---|---|
| `<\|im_end\|>` | 一条消息结束 | 模型通常用它结束一轮回复 |
| `<\|endoftext\|>` | 整篇文档结束 | 偶尔会出现，也要能停下 |

因此这一轮模型返回的完整文本是 `<tool>guess(3)</tool><|im_end|>`。此外还有第三种情况：写满 `max_tokens` 被强行截断，这时结尾没有任何停止符。

---

## 5. 思维链：先想再答

**思维链（Chain of Thought，CoT）**是指让模型在回答前先写出推理过程。Qwen3.5 规定的格式为：

```
<think>
推理过程……
</think>

最终回答
```

`model.enable_thinking` 控制是否让模型思考，做法是改变生成提示的最后一部分，这部分称为 **prefill（预填充）**。

**关闭思考**（Number Search）：prefill 是一个**空的**思考块，相当于替模型写好了"我想完了"：

```
<|im_start|>assistant
<think>

</think>

            ← 模型从这里开始写，直接给出答案：<tool>guess(3)</tool><|im_end|>
```

**开启思考**（TextCraft）：prefill 只写了开头的 `<think>`，模型需要接着写推理：

```
<|im_start|>assistant
<think>
            ← 模型从这里开始写：
              区间是 1~6，取中间值 3。
              </think>

              <tool>guess(3)</tool><|im_end|>
```

**prefill 由模板写出，属于 prompt；prefill 之后的内容才是模型生成的。**因此开启思考时，模型的推理文字也属于它的"动作"，会参与训练。

---

## 6. 历史：多轮对话是如何继续的

环境回复 "3 is too low." 后，Agent 要发起第二轮：把刚才模型的回复和环境的回复加入消息列表，然后**把整段对话重新渲染一遍**：

```python
messages = [
  system, user,
  {"role": "assistant", "content": "<tool>guess(3)</tool>"},   # 模型上一轮的回复（去掉末尾停止符）
  {"role": "user",      "content": "3 is too low."},           # 环境的反馈
]
```

第 3 条之前的内容称为**历史**，即已经发生过的轮次。模板写回历史中 assistant 消息的规则是：

```
<|im_start|>assistant\n + prefill + 内容 + <|im_end|>\n
```

于是第二轮的 prompt 为：

```
<|im_start|>system ... <|im_end|>
<|im_start|>user Task: ... <|im_end|>                   ┐
<|im_start|>assistant                                    │ 第一轮的 prompt
<think>                                                  │
                                                         │
</think>                                                 │
                                                         ┘
<tool>guess(3)</tool><|im_end|>                          ← 第一轮模型生成的回复
                                                         ┐
<|im_start|>user                                         │ 新增的尾部
3 is too low.<|im_end|>                                  │
<|im_start|>assistant                                    │
<think>                                                  │
                                                         │
</think>                                                 │
                                                         ┘
```

可以看到：**第二轮的 prompt = 第一轮的 prompt + 第一轮的回复 + 新增的尾部**，这就是"扩展成立"。此时整个回合可以拼成一条训练序列，内容就等于最后一轮的 prompt 加上最后一轮的回复。

---

## 7. 历史中的思考是否保留

开启思考后，历史中的 assistant 消息带着很长的推理过程。TextCraft 最多 75 轮、每轮最多 512 个 token，全部保留会让上下文非常长。

`model.preserve_thinking_history` 控制模板如何写回历史中的思考。

**`true`（保留，本作业使用的设置）**：原样写回

```
<|im_start|>assistant
<think>
区间是 1~6，取中间值 3。
</think>

<tool>guess(3)</tool><|im_end|>
```

**`false`（删除，Qwen 官方模板的默认做法）**：只保留 `</think>` 之后的内容

```
<|im_start|>assistant
<tool>guess(3)</tool><|im_end|>
```

删除后上下文短得多，但历史与模型当初实际生成的内容**不再一致**。关闭思考时这个开关没有作用，因为历史中本来就没有需要删除的推理。

---

## 8. 回到问题：扩展为什么会断开

关键在于比较两段文字：

- **A**：第一轮时，模型**实际看到的** assistant 开头（prefill），加上它**实际生成的**回复；
- **B**：第二轮的 prompt 中，模板把这一轮**写回历史**后的样子。

A 和 B 一字不差，就能接着拼；有任何不同，就断开。下面逐个情况对比。

### 情况 1：正常结束（以 `<|im_end|>` 停止）✅

```
A（第一轮实际的）                    B（第二轮写回的）
<|im_start|>assistant               <|im_start|>assistant
<think>                             <think>

</think>                            </think>

<tool>guess(3)</tool><|im_end|>     <tool>guess(3)</tool><|im_end|>
```

两边完全相同，扩展成立，接着拼。

### 情况 2：被 `max_tokens` 截断（没有停止符）✅

```
A（第一轮实际的）                    B（第二轮写回的）
<|im_start|>assistant               <|im_start|>assistant
<think>                             <think>

</think>                            </think>

<tool>guess(3)</tool>               <tool>guess(3)</tool><|im_end|>
                                                         ↑ B 多出这一截
```

A 是 B 的开头，B 只是**在后面多了**一个 `<|im_end|>`。多出来的部分算作新尾部，扩展仍然成立。这个 `<|im_end|>` 不是模型生成的，所以 mask = 0，不参与训练。

### 情况 3：以 `<|endoftext|>` 停止 ❌

```
A（第一轮实际的）                    B（第二轮写回的）
<|im_start|>assistant               <|im_start|>assistant
<think>                             <think>

</think>                            </think>

<tool>guess(3)</tool><|endoftext|>  <tool>guess(3)</tool><|im_end|>
                    ↑ 从这里开始不同 ↑
```

模板写回历史时，结尾**固定写 `<|im_end|>`**，它不知道模型当初用的是哪个停止符。A 的最后一个 token 在 B 中被换掉了，A 不再是 B 的开头，扩展断开。

### 情况 4：开启思考，且删除历史中的思考（`preserve_thinking_history: false`）❌

```
A（第一轮实际的）                    B（第二轮写回的）
<|im_start|>assistant               <|im_start|>assistant
<think>                             <tool>guess(3)</tool><|im_end|>
区间是 1~6，取中间值 3。              ↑ 从这里开始不同
</think>

<tool>guess(3)</tool><|im_end|>
```

模板把 `<think>…</think>` 整段删掉了，B 在 `assistant` 之后直接是工具调用，而 A 在这里是 `<think>`，扩展断开。

### 汇总

| 情况 | A 与 B 的关系 | 结果 |
|---|---|---|
| 1. 正常结束 | 完全相同 | ✅ 接着拼 |
| 2. 被截断 | B 只在 A 后面多出一个停止符 | ✅ 接着拼，多出的部分 mask = 0 |
| 3. 以另一个停止符结束 | 最后一个 token 被替换 | ❌ 另起一条序列 |
| 4. 删除思考 | 中间的推理被删除 | ❌ 另起一条序列 |

一旦不一致，第二轮 prompt 就不再是"第一轮内容 + 新尾部"。如果硬拼成一条序列，序列中第二轮回复前面的内容就不是模型当时真正看到的上下文，训练就会出错。这时只能从第二轮的完整 prompt 另起一条序列。`datagen.extension_suffix` 就是在每一轮做这个判断（拼接的合法性条件与处理方式见 [dataflow.md 第 4、8 节](dataflow.md)）。

### 补充：文本相同，token 也可能不同

即使文本完全一致，**重新编码**得到的 token 也可能与模型生成的不同（示意）：

```
模型采样时逐个 token 生成:  "gu" + "ess"   → [1201, 3305]
把文本 "guess" 重新编码:    "guess"        → [9876]
```

模型生成 token 时不一定选择分词器的"标准切分"，而 BPE 编码会跨边界合并字符，所以"文本 → token"不是"token → 文本"的严格逆运算。

Agent **从设计上避免了**这一层问题：`_extend_context` 不重新编码模型生成的部分，而是直接复用 `completion.tokens`，只编码新增的尾部（见 [dataflow.md 2.2 节](dataflow.md)）。因此实践中导致断开的只有上表中的文本层原因。

### 补充：为什么 Agent 要重新渲染，而不是直接追加 token

因为**模板是格式的唯一权威**。每一轮需要的 turn 标记、换行、`<think>` prefill 如何书写，都由模板根据整段对话决定。如果 Agent 自己拼接格式，很容易与模型训练时见过的格式产生细微差别。

所以 Agent 的做法是"**以模板渲染的全文为准，再尽量复用已有的 token**"：

- 能复用（前缀一致）：只编码新尾部，扩展成立；
- 不能复用：整段重新编码，扩展断开。

---

## 9. 术语对照表

| 术语 | 一句话解释 | 本作业中的例子 |
|---|---|---|
| token | 模型处理的最小单位：一个词、半个词或一个符号 | `guess`、`(3)`、`<\|im_end\|>` |
| 特殊 token | 词表中表示格式标记的 token，不是普通文字 | `<\|im_start\|>`、`<\|im_end\|>`、`<\|endoftext\|>` |
| 消息列表 messages | 程序中结构化的对话 | `[{"role": "user", "content": ...}, ...]` |
| 对话模板 chat template | 把消息列表写成文本的规则 | [qwen3_5.jinja](../scottyrl/chat_templates/qwen3_5.jinja) |
| 渲染 render | 用模板把消息列表变成文本 | `policy.render(messages)` |
| 编码 encode | 把文本切分成 token | `policy.encode(text)` |
| 生成提示 generation prompt | prompt 末尾的"轮到 assistant 了" | `<\|im_start\|>assistant\n` + prefill |
| prefill | 替模型提前写好的开头，属于 prompt | `<think>\n`（开启思考）或空思考块（关闭思考） |
| 停止符 stop sequence | 生成到它就停止 | `<\|im_end\|>`、`<\|endoftext\|>` |
| 思维链 CoT | 回答前先写推理过程 | `<think>推理</think>` |
| `enable_thinking` | 是否让模型写推理 | Number Search 关，TextCraft 开 |
| 历史 history | 之前已经发生的轮次 | 第二轮 prompt 中第一轮的问答 |
| `preserve_thinking_history` | 写回历史时是否保留推理 | 本作业为 `true` |
| 扩展 extension | 下一轮 prompt = 之前全部内容 + 新尾部 | 成立时整个回合拼成一条序列 |
