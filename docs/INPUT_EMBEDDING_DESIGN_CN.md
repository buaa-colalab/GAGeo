## 输入嵌入设计与融合方式说明

本文档从概念层面梳理整个 Cross-View Localization V3 模型中，**每一种输入是如何被编码为 embedding，并在主干网络中相互融合**的。重点不是实现细节，而是整体设计逻辑与信息流向。

我们将输入分为四类：
1. 图像输入：前视图 `front_view` 与卫星图 `satellite_view`
2. 几何提示：点、框、掩码（通过 SAM2 Prompt Encoder）
3. 可学习任务查询：用于 bbox/mask 与 heatmap 的 learnable queries
4. 额外监督相关输入：前视与卫星的语义掩码 `mono_mask`、`sat_mask`（仅用于对比学习）

---

### 1. 图像输入的嵌入与位置编码

**(1) 归一化与打包**

模型首先将前视图与卫星图按批次打包：
\[
I = \text{stack}([I_s, I_f]) \in \mathbb{R}^{B \times 2 \times 3 \times H \times W},
\]
然后使用 ImageNet 均值方差进行标准化：
\[
I_{\text{norm}} = \frac{I - \mu}{\sigma},
\]
其中 \(\mu,\sigma\) 为缓冲区中固定的 `image_mean` 与 `image_std`。

接着将两个视图展平到 batch 维度：
\[
I_{\text{flat}} \in \mathbb{R}^{(B \cdot 2) \times 3 \times H \times W}.
\]

**(2) DINOv2 Patch Embedding**

归一化后的图像输入到 DINOv2 编码器，得到 patch 级特征：
\[
X = \text{DINOv2}(I_{\text{flat}}) \in \mathbb{R}^{(B \cdot 2) \times N_p \times C},
\]
其中 \(N_p = 37 \times 37 = 1369\) 为 patch 数量。DINOv2 的 cls token 在此被丢弃，仅保留 patch tokens。

随后将 `X` reshape 回两视图形式：
\[
X \rightarrow \text{sate\_hidden} \in \mathbb{R}^{B \times N_p \times C},\quad
\text{front\_hidden} \in \mathbb{R}^{B \times N_p \times C}.
\]

**(3) Register Tokens 的拼接**

Pi3 解码器在每个视图前预置固定数量的“寄存器 token”（register tokens），用于聚合全局信息与为后续任务留出统一的“相机 token”位：
\[
\text{reg} \in \mathbb{R}^{1 \times 5 \times C},
\]
扩展到 batch 后分别拼接到两个视图的 patch 前端：
\[
\text{sate\_hidden} = [\text{reg},\, \text{sate\_patches}],\quad
\text{front\_hidden} = [\text{reg},\, \text{front\_patches}],
\]
因此每个视图的 token 数量变为 \(5 + N_p = 1374\)。此时，**图像输入的 embedding 由三部分组成**：
1. DINOv2 产生的 patch embedding；
2. 预置的 register token embedding；
3. 后续将会叠加的 RoPE 位置编码（见第 4 节）。

---

### 2. 几何提示（点、框、掩码）的嵌入方式

几何提示通过 `GeometryPromptEncoder` 进行编码，其内部遵循 SAM2 的设计，并通过随机傅里叶特征（`PositionEmbeddingRandom`）引入位置信息。

#### 2.1 点提示（points）的嵌入

点提示输入形式为 \((P, L)\)，其中
- \(P \in \mathbb{R}^{B \times N \times 2}\)：像素坐标，
- \(L \in \mathbb{R}^{B \times N}\)：标签（0=负点，1=正点，2/3=框角点）。

编码步骤概念上分为两层：

1. **几何位置编码（random Fourier PE）**
   - 首先将像素坐标归一化到 \([0,1]\)：
     \[
     \tilde{P}_{x} = \frac{P_x}{W},\quad \tilde{P}_{y} = \frac{P_y}{H},
     \]
   - 然后通过固定的高斯矩阵 \(G \in \mathbb{R}^{2 \times d}\) 映射并施加 \(\sin/\cos\) 变换：
     \[
     E_{\text{pos}}(P) = [\sin(2\pi \tilde{P}G),\, \cos(2\pi \tilde{P}G)],
     \]
   得到与内部维度相同的**几何位置 embedding**。

2. **类型 embedding 的加性注入**
   - 为“正点 / 负点 / 框左上角 / 框右下角 / 非点”分别维护若干可学习的向量 \(e_{\text{type}}\)；
   - 对于每个点，根据标签 \(L\) 选择对应的类型 embedding，并 **与位置 embedding 相加**：
     \[
     E_{\text{point}} = E_{\text{pos}} + e_{\text{type}}.
     \]

最终，所有点提示被堆叠为：
\[
\text{sparse\_points} \in \mathbb{R}^{B \times N' \times C_{\text{int}}},
\]
若仅有点而无框，会额外补一个“padding 点”，其坐标设为 \((0,0)\)，类型为“非点”。

#### 2.2 框提示（boxes）的嵌入

框提示输入为 \(\mathbb{R}^{B \times M \times 4}\)，格式为 \((x, y, w, h)\)。编码分为：

1. **角点几何编码**
   - 将中心+尺寸形式转换为左上、右下角坐标；
   - 同样以像素坐标形式输入 `PositionEmbeddingRandom`，得到两个角点的几何 embedding。

2. **角点类型 embedding**
   - 左上角与右下角分别加上专门的类型向量 \(e_{\text{top-left}}\)、\(e_{\text{bottom-right}}\)：
     \[
     E_{\text{corner, TL}} = E_{\text{pos, TL}} + e_{\text{top-left}},\quad
     E_{\text{corner, BR}} = E_{\text{pos, BR}} + e_{\text{bottom-right}}.
     \]

最终，每个框产生两个稀疏 token，将其沿查询维度展开后与点提示拼接，形成统一的：
\[
\text{sparse\_embeddings} \in \mathbb{R}^{B \times K \times C_{\text{int}}}.
\]

#### 2.3 掩码提示（masks）的嵌入

掩码提示输入为二值图：
\[
M \in \mathbb{R}^{B \times 1 \times H \times W}.
\]

编码流程：
1. 将掩码双线性插值到固定尺寸（约为 \(148 \times 148\)），匹配内部编码器预期输入；
2. 通过数层卷积 + LayerNorm + 激活函数序列，将掩码下采样到与 patch 网格同尺寸（约 \(37 \times 37\)）并提升通道维到内部维度 \(C_{\text{int}}\)：
   \[
   E_{\text{mask}} \in \mathbb{R}^{B \times C_{\text{int}} \times 37 \times 37}.
   \]
3. 若没有掩码提示，则使用一个可学习的 `no_mask_embed` 向量，在空间上广播为：
   \[
   E_{\text{mask, default}} \in \mathbb{R}^{B \times C_{\text{int}} \times 37 \times 37}.
   \]

#### 2.4 从 SAM 内部维度到主干维度的投影

Prompt Encoder 内部可以采用 SAM 原生维度（如 256），与主干所需的 1024/2048 维度不同。为此：
- 稀疏提示 token 通过一层线性层 + LayerNorm 投影到主干使用的维度：
  \[
  \text{sparse\_embeddings} \in \mathbb{R}^{B \times K \times C_{\text{int}}}
  \rightarrow \mathbb{R}^{B \times K \times C_{\text{dec}}};
  \]
- 掩码 embedding 通过 1×1 卷积 + LayerNorm 投影到相同维度：
  \[
  E_{\text{mask}} \in \mathbb{R}^{B \times C_{\text{int}} \times 37 \times 37}
  \rightarrow \mathbb{R}^{B \times C_{\text{dec}} \times 37 \times 37}.
  \]

这一步保证几何提示与 Pi3 解码器中的图像 patch embedding 处于统一特征空间，便于后续联合注意力。

---

### 3. 可学习任务查询的嵌入与注入位置

Pi3 Backbone V2 为前视图一侧引入固定数量的可学习查询：
\[
Q_{\text{learn}} \in \mathbb{R}^{1 \times N_{\text{learn}} \times C_{\text{dec}}},
\]
在 batch 维上复制并拼接到 front view token 序列尾部：
\[
\text{front\_hidden} = [\text{reg},\, \text{front\_patches},\, Q_{\text{learn}}].
\]

这些查询在后续被分成两类：
- 前 \(N_{\text{bbox}}\) 个作为 bbox/mask head 的查询；
- 其余 \(N_{\text{heat}}\) 个聚合后作为 heatmap head 的查询。

在整个解码过程中，**learnable queries 与其它 token 完全对称地参与局部和全局注意力**，但其初始表示是纯参数化的，不依赖任何显式输入；它们通过多层 cross-view self-attention 学到如何“提取”与任务相关的跨视角信息。

---

### 4. 提示 token 与图像 token 的融合方式

几何提示最终通过两条路径进入 Pi3 解码器：

1. **稀疏提示 token 作为额外 token 拼接到前视图尾部**
   - 经 `prompt_proj` 后的 `sparse_embeddings` 被直接拼接：
     \[
     \text{front\_hidden} = [\text{reg},\, \text{front\_patches},\, Q_{\text{learn}},\, T_{\text{prompt}}],
     \]
     其中 \(T_{\text{prompt}} \in \mathbb{R}^{B \times K \times C_{\text{dec}}}\)。
   - 这些 token 在局部与全局注意力中与图像 patch 与 learnable queries 共同作用，但通过注意力 mask 限制其与卫星视图之间的可见性，从而将提示信息主要“注入”到前视图侧与 learnable queries。

2. **掩码 embedding 作为加性项注入到全局注意力的 K 通道**
   - 对于全局注意力层，解码器构造：
     \[
     Q, V = [\text{sate},\, \text{front},\, Q_{\text{learn}},\, T_{\text{prompt}}],
     \]
     \[
     K = [\text{sate},\, \text{front} + E_{\text{mask}},\, Q_{\text{learn}},\, T_{\text{prompt}}],
     \]
     其中 \(E_{\text{mask}}\) 展平后与前视 patch tokens 逐元素相加。
   - **重要特性**：掩码并不直接修改前视 patch 本身在 Q/V 中的表示，而是仅作为 K 的加性偏置，影响“别人如何读取前视 patch”。这使得几何掩码更像是一种“注意力引导信号”，而非强行覆盖图像特征。

通过上述两条路径，点/框提示通过显式 token 与位置编码向 learnable queries 输送几何先验，掩码则通过加性偏置方式微调全局注意力的查询-键匹配模式。

---

### 5. RoPE 位置编码与 prompt 坐标的作用

Pi3 解码器使用二维 RoPE（Rotary Position Embedding）作为统一的位置编码方式。对所有参与注意力的 token，都需要提供一个二维“逻辑坐标”：

1. **卫星与前视 patch 的坐标**
   - 由 `PositionGetter` 在 patch 网格上生成整数网格；
   - 若存在 register tokens，则先整体平移一格，并为这几个特殊 token 指定 \((0,0)\)；
   - 得到：
     \[
     \text{sate\_pos} \in \mathbb{R}^{B \times (5+N_p) \times 2},\quad
     \text{front\_pos-base} \in \mathbb{R}^{B \times (5+N_p) \times 2}.
     \]

2. **learnable queries 的坐标**
   - 初始化时赋予 \((0,0)\) 这一“无几何偏好”的位置：
     \[
     \text{pos}_{Q_{\text{learn}}} = 0.
     \]

3. **prompt tokens 的坐标（由 `_build_prompt_coords` 提供）**
   - 对于点提示：直接使用像素坐标除以图像尺寸得到 \([0,1]\) 范围的归一化坐标；
   - 对于框提示：根据 \((x,y,w,h)\) 计算两角点坐标，并同样归一化；
   - 所有提示坐标沿 token 维拼接，形成：
     \[
     \text{prompt\_coords} \in \mathbb{R}^{B \times K \times 2}.
     \]
   - 在进入 RoPE 前，将 \([0,1]\) 坐标缩放到 patch 网格尺度（乘以 37），得到与 patch 一致的“类整数坐标”。

最终，前视侧的 RoPE 坐标为：
\[
\text{front\_pos} = [\text{reg+patch\_pos},\, \text{pos}_{Q_{\text{learn}}},\, \text{prompt\_pos}],
\]
卫星侧为 `sate_pos`。RoPE 在 attention 内部对 Q/K 进行旋转，从而注入相对几何信息。

---

### 6. 对比学习掩码 `mono_mask` / `sat_mask` 的使用

输入中的 `mono_mask` 与 `sat_mask` 并不进入主干作为显式 embedding，而是只在**对比学习头**中用于加权池化：
- 使用 `mono_mask`、`sat_mask` 对前视与卫星 patch 特征做掩码平均池化，得到全局向量；
- 通过 MLP 映射到对比空间，参与 MoCo 风格的 InfoNCE 损失。

因此，从“嵌入与加法/拼接”的角度看：
- 这两个掩码**不改变 backbone 内部 token 的 embedding**；
- 而是仅在 loss 分支中作为**池化权重**参与特征聚合。

---

### 7. 小结：各输入的 embedding “加在哪里”

综上，可以将每个输入的 embedding 融合位置概括为：

1. **前视图 / 卫星图像**
   - 先经 DINOv2 得到 patch embedding；
   - 拼接 register tokens；
   - 在解码器 self-attention / cross-attention 中乘上 RoPE 位置编码。

2. **点 / 框提示**
   - 通过 random Fourier PE + 类型 embedding 得到稀疏提示 token；
   - 经过维度投影后，作为 **额外 token 拼接到前视 token 序列尾部**；
   - 同时，其归一化坐标参与构建 RoPE 位置坐标，使提示 token 带有几何位置信息。

3. **掩码提示**
   - 通过卷积下采样与 1×1 投影得到与 patch 网格对齐的 dense embedding；
   - 在全局注意力层中，**仅对 K 通道的前视 patch tokens 做加法偏置**，引导其他 token 如何读取前视区域。

4. **可学习任务查询**
   - 作为参数向量初始化，在前视侧 **拼接到 token 序列尾部**；
   - 与图像 / 提示 token 共同参与所有注意力层，并被下游任务头读取。

5. **对比学习掩码**
   - 不直接进入 backbone，仅作为对比损失中的池化权重使用。

这种设计使得：  
**图像、几何提示与任务查询在同一 Transformer 空间内通过“拼接 + RoPE + 掩码注意力 + 加性偏置”统一交互，各输入的几何含义都通过位置编码或加法偏置精细地注入到注意力结构中。**

