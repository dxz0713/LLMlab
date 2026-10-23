# Lab 5 实验报告：LLM 注意力、RoPE 与 Transformer Block

## 1. 实验目的

本实验从零实现现代 LLM 内部的核心组件，从子词分词一直打通到可堆叠的 Transformer block。实验关注：

- 字节级 BPE 分词器的训练、编码与压缩效果度量；
- 因果掩码（causal mask）如何阻止自注意力"看见未来"；
- RoPE 旋转位置编码的构造及其相对位置性质；
- 多头因果注意力、GELU 前馈网络与 pre-norm Transformer block 的前向传播；
- 分词粒度对词元数量、压缩率以及后续语言模型建模的影响。

说明：本仓库按照教学实现采用**纯 NumPy**（`src/tokenizer.py`、`src/attention.py`、`src/transformer.py`），与 PDF 中以 PyTorch 张量表述的编程任务在结构上一一对应（Task 1–4 分别对应 `causal_mask`/带掩码注意力、`MultiHeadAttention`（含 `CausalSelfAttention`）、`FeedForward`、`TransformerBlock`）。

## 2. 数据集与环境

- 语料：`data/the-verdict.txt`（Stephen King 短篇小说片段，20,479 字符 / 20,479 字节，纯 ASCII）。
- 环境：Python 3.13 + NumPy（`uv` 管理依赖，见 `pyproject.toml`）。
- 测试：`uv run pytest tests/` 全部通过，**51 passed**（tokenizer 18、attention 20、transformer 13）。

## 3. 模型与实现

### 3.1 字节级 BPE 分词器（`src/tokenizer.py`）

词表从 256 个字节值出发，`train()` 每轮统计训练序列中相邻 id 对的出现频次，选出频次最高（并列时取字典序最小）的一对合并为新 token，重写训练序列，直至词表达到 `vocab_size`。特殊 token 不参训，其 id 预留在词表**最末尾**（`vocab_size - n_special` 起），因此 `len(vocab) + len(special_tokens) == vocab_size`——本次修复了合并 id 与特殊 token id 从 256 起重叠的冲突。`_merge` 从左到右做非重叠替换（`[a,a,a]` 配 `(a,a)` → `[new, a]`）。`_encode_ordinary` 按"最早学到的合并优先"重放合并规则；`encode` 先把特殊 token 按整体 id 切分路由，其余片段走 BPE；`decode` 按 id 拼回字节流并以 `errors="replace"` 解码。

`sliding_window` 以步长 `stride` 在 token 序列上滑 `max_length+1` 的窗口，输出 `(inputs, targets)` 两个 `(num_samples, max_length)` 的 int64 数组，`targets[i]` 恰为 `inputs[i]` 右移一位，即"预测下一个 token"的训练样本。本次修复：原实现误用 `yield` 变成了生成器，改为收集切片后一次性返回数组对。

### 3.2 因果注意力与掩码（`src/attention.py`）

`scaled_dot_product_attention` 计算 `softmax(q k^T / sqrt(d)) v`，掩码以加性方式进入分数矩阵：`causal_mask` 上三角为 `-inf`，softmax 将其映射为精确的 0。`MultiHeadAttention` 将 `d_out` 切分为 `num_heads` 个头，所有头以批量矩阵运算并行完成（无 Python 循环），最后拼回并经输出投影；`CausalSelfAttention` 在此基础上接入 RoPE。softmax 使用减最大值的数值稳定版本。

### 3.3 RoPE（`src/attention.py`）

`compute_rope_params` 对 `i = 0..head_dim/2-1` 生成角频率 `theta_base^(-2i/head_dim)`，角度表按 rotate-half 布局（前半与后半复制同一组角度）拼成 `(context_length, head_dim)`。`apply_rope` 将向量拆为前后两半，按
`[x1·cos − x2·sin, x2·cos + x1·sin]`
旋转，其中 cos/sin 取表的前 `head_dim//2` 列（本次修复：原实现用全宽表与半宽向量相乘导致广播失败）。旋转保持向量模长不变，且 `R_m(q)·R_n(k)` 只依赖相对偏移 `m − n`，由测试 `test_rope_dot_product_depends_only_on_relative_offset` 验证。

### 3.4 Transformer block（`src/transformer.py`）

pre-norm 结构：`x = x + att(norm1(x))`，`x = x + ff(norm2(x))`。`FeedForward` 为逐位置两层 MLP：`gelu(x @ W1 + b1) @ W2 + b2`，隐藏维度 `4 × emb_dim`；GELU 采用 GPT-2 使用的 tanh 近似。`LayerNorm` 用有偏方差，与 `torch.nn.LayerNorm` 一致。

## 4. 讨论问题解答

以下 Q1–Q3、Q4–Q6 的数值均由本仓库代码实测得到（脚本 `exp.py`，`uv run python exp.py` 可复现）。

### Q1：不使用因果掩码会发生什么？

位置 i 能 attend 到 i+1..T−1，即"偷看未来"。实测 6×8 的注意力权重中，未加掩码时第 0 行为
`[0.2424, 0.1540, 0.1769, 0.1367, 0.1209, 0.1692]`
——位置 0 把 76% 以上的注意力分配给了它之后的 token。生成式模型据此训练会直接"背答案"：预测下一 token 时输入里已含答案，损失被严重低估，推理时模型却看不到未来，训练与推理行为不一致。

### Q2：使用因果掩码会怎样？

加上掩码后，第 i 行只保留 0..i 的概率质量，其余为精确 0；每行仍和为 1，仍是合法分布。实测同一组输入：

```
[[1.     0.     0.     0.     0.     0.    ]
 [0.4009 0.5991 0.     0.     0.     0.    ]
 [0.3182 0.2826 0.3992 0.     0.     0.    ]
 ...
 [0.1706 0.1273 0.1625 0.1030 0.1900 0.2466]]
```

位置 0 只剩自己，故概率为 1（若一个 batch 中存在更短的样本需要 PAD，PAD 位被整行屏蔽后，剩余可见 token 的概率质量重新归一化——这正是短样本行"所有质量集中在合法位置"的原因）。

### Q3：加 PAD 会改变真实 token 的输出吗？

对于**尾随 PAD**：不会。位置 i 的输出只由 0..i 的 value 决定，尾随 PAD 在掩码之外，不参与任何真实位置的加权。实测 `MultiHeadAttention` 对同一句 `[hi]` 与 `[hi, PAD]`，真实位置的输出最大差异为 **0.0**。注意若 PAD 出现在序列**头部**（left padding）且只用因果掩码，则真实位置会 attend 到 PAD；实际工程还需配合 attention mask（本仓库按任务要求验证的是尾随 PAD 情形）。

### Q4：BPE 分词器的压缩率

在 `the-verdict.txt` 上训练（`train(text, vocab_size)`，`compression_ratio = 字节数/词元数`）：

| 分词器 | vocab_size | 学到的合并数 | 压缩率（bytes/token） |
|---|---:|---:|---:|
| Lab4 字符级（基线） | — | — | 1.0000 |
| 本 BPE | 270 | 14 | 1.2261 |
| 本 BPE | 5000 | 4744 | **10.7276** |

更大词表能合并出更长的常见片段（单词、词组），压缩率显著更高；270 版只有 14 次合并，大多只并到子词级，收益有限，但已优于逐字符切分。

### Q5：换训练语料会怎样？

用本项目的全部 Python 源码（`src/`、`tests/`、`visualize.py`，共 47,714 字符）以 `vocab_size=5000` 训练，在 `tests/test_tokenizer.py` 上测得：

| 分词器（vocab 5000） | 训练语料 | 对 test_tokenizer.py 的压缩率 |
|---|---|---:|
| 源码 BPE | 项目 Python 源码 | **10.0184**（4,889 字符 → 488 词元） |
| 小说 BPE | the-verdict.txt | 1.5848 |

符合预期：分词器对与训练分布同分布的文本压缩好得多（缩进、`def`、`self`、括号等模式成为高频合并），而对分布外文本（小说训练的 tokenizer 面对代码）几乎退化成字节切分。这说明**分词器必须匹配目标域**。

### Q6：三种分词器下 the-verdict.txt 的词元数

| 分词器 | 词元数 |
|---|---:|
| 字符级 | 20,479 |
| BPE vocab 270 | 16,702 |
| BPE vocab 5000 | **1,909** |

对 LLM 而言 5000 版更好：同样文本只需约 1/10 的词元，训练与推理的每 token 计算固定，词元少意味着同样上下文窗口能装下更多内容、更快收敛、更长的有效上下文；代价是嵌入/输出矩阵更大（5000+ 行 vs 270 行）、低频词更可能未被覆盖。"100%/99.9% 准确率"一类宣传对应的是编码后能否无损解码回原文——字节级 BPE 恒可无损还原（本实现 `decode(encode(s)) == s` 由测试覆盖），因此分词层面的"准确率"本身并不构成模型质量优势。

### Q7：困惑度分析

本仓库不含 Lab4 的 PyTorch LLM 工程与预训练 checkpoint（`checkpoints/wt103_en_char.pt`，训练于 WikiText-2 字符级数据），该实验无法在此实测，以下为方法说明与预期结论。做法：取 `the-verdict.txt` 前 25%，分别用字符级与 BPE 切分，对每个窗口内 token 计算交叉熵，困惑度 `= exp(mean CE)`，并与模型在其训练分布（wt103 验证集）上的困惑度对比。预期结果**更差**：一是 `the-verdict` 是与 WikiText-2 分布不同的小说文本；二是字符级模型根本不认识 BPE 词元（词表只有字符），按 BPE 输入只会得到接近随机的低概率，只有按字符输入才与训练分布一致——这正是"模型与其分词器绑定"的含义。

### Q8：BPE 的 bits-per-character

理论关系：`bits/char = 词元的 bits/token ÷ 每词元平均字符数`。BPE 每 token 困惑度虽高，但每个 token 覆盖约 10 个字符（10.7 bytes/token），摊薄后每字符比特数通常**低于**字符级 tokenizer 的对应值（Q7 的字符级每 token 即每字符、困惑度更高）——两者不能直接比"每 token"数值，必须换算到每字符才公平。同样因缺 checkpoint，本项未实测，结论为按公式可预测的方向性结果。

### Q9：训练动态对比

该任务需要 PyTorch 训练工程（GPT-2 small、WikiText-2、batch 8×context、Adam、warmup+退火），本仓库（纯 NumPy 前向实现，无反向传播）无法复现，预期趋势（见 `llms-from-scratch-cn` Ch04/Sec03/04）：`128` 上下文每次前向便宜、更新次数相对更多，初始损失下降快、更早进入平台；`1024` 上下文单步计算更贵、初期下降慢，但能看到更长程依赖，后期损失更低、验证集过拟合出现得更晚。短窗口曲线陡、长窗口曲线缓而低是典型形态。

## 5. 可视化输出

`uv run python visualize.py` 生成 `lab5.png`，含四个面板：词表从 256 增至 1000 时，编码 the-verdict.txt 所需词元数从 20,479 单调降至 6,878（BPE 压缩学习曲线）；同一 vocab=1000 分词器在 in-corpus 文本上的压缩率为 2.81 bytes/token，而对新造乱序文本仅 1.12——与 Q5 的域匹配结论一致；最后两个面板是因果注意力头的权重热图，均为严格下三角，直观验证了 Q2/Q3 的因果性。

![lab5 可视化](lab5.png)

## 6. 结论与复现

本实验完成了字节级 BPE 训练与编码、滑窗样本构造、因果多头注意力与 RoPE、GELU 前馈网络及 pre-norm Transformer block 的从零实现，测试全部通过（51 passed）。实测显示：合并次数（词表规模）与训练域匹配共同决定压缩率（1.0 → 1.23 → 10.7 bytes/token），因果掩码保证严格只看过去且尾随 PAD 不改变真实位置输出，RoPE 的注意力得分只依赖相对位置。Q7–Q9 依赖 Lab4 的 PyTorch LLM 工程与预训练权重，本仓库未包含，报告中给出了方法、公式与预期方向，未作数值实测。

复现步骤：

```bash
uv run pytest tests/          # 51 passed
uv run python exp.py          # Q1-Q6 实测数值
uv run python visualize.py    # 生成 lab5.png
```
