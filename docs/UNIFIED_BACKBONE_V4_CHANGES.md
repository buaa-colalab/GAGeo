# Unified Backbone V4 修改说明

本文档记录 `feature/unified-backbone-v4` 分支中的两类核心变更：

1. `Pi3BackboneV2` 的 attention 机制调整（local + global）。
2. `DETRCriterionV2` 中 heatmap loss 从位置 MSE 改为 CornerNet 风格逐像素 focal loss。

---

## 1. Attention 机制改动

修改文件：

- `models/backbone/pi3_backbone_v2.py`

### 1.1 修改前（V3 风格）

#### Local Attention

- 采用“全 token 拼接后一次 self-attention”的形式。
- 布局：`[sate | front | learnable | prompt]`。
- 通过 `local_mask` 限制可见性（例如 sate/front 互不可见、prompt 互不可见等）。
- 本质上是 masked global attention，而不是 Pi3 原始的 frame-wise local self-attention。

#### Global Attention

- Q/V 布局为 `[sate | front | learnable | prompt]`。
- K 在此基础上可注入 dense mask（仅 front patch 区域）。
- `global_mask` 规则包括：
  - prompt 与 sate 互不可见；
  - prompt 与 prompt 互不可见（仅保留对角可见）；
  - 其余可见。

---

### 1.2 修改后（V4）

#### Local Attention（按 Pi3 frame attention 方式）

- 不再使用“全局拼接 + local mask”。
- 改为**每个 frame 独立 self-attention**：
  - 卫星分支：`[sate_tokens | learnable_queries]`
  - 前视分支：`[front_tokens | prompt_tokens]`
- 两个分支分别通过同一个 local block 前向（参数共享、调用两次），符合 Pi3 的 frame-wise local self-attention 思路。
- local stage 后再把 token 拆回：
  - `sate_hidden` 与 `learnable_hidden`
  - `front_hidden` 与 `prompt_hidden`

#### Global Attention（在原有基础上增加新 mask 规则）

- 全局阶段仍采用统一布局：`[sate | front | learnable | prompt]`。
- 保留原有规则（prompt<->sate 不可见、prompt 间互不可见）。
- **新增规则**：`learnable query` 与 `prompt token` 互不可见（双向屏蔽）。
- 这样可避免 learnable 分支直接读取 prompt token，减少任务分工耦合。

---

### 1.3 Token 流与监督接口差异

#### 修改前

- learnable token 直接挂在 `front_hidden` 尾部，intermediate/final 通过 front 切片提取 learnable 输出。

#### 修改后

- learnable token 作为独立流 `learnable_hidden` 在 local/global 交替中显式维护。
- intermediate 与 final learnable 输出直接来自 `learnable_hidden`，不再依赖 front 切片索引。
- 对下游 head 接口保持兼容（`learnable_out` 形状与语义不变）。

### 1.4 Embedding 设计（以及和原来的区别）

本节说明 backbone 中几类 embedding 的来源、注入位置以及 V3/V4 差异。

#### 视觉 token embedding

- `sate_hidden` / `front_hidden` 来自 DINOv2 patch token，经 register token 拼接后进入 Pi3 decoder。
- 该部分在 V3/V4 一致，基础表示均为 `[register | patches]`。

#### Learnable query embedding

- 参数形式：`learnable_queries ∈ R^{1×Q×C}`，训练中可学习，初始化为高斯分布（`std=0.02`）。
- **原来（V3）**：local 阶段拼接到 front token 尾部。
- **现在（V4）**：local 阶段拼接到 satellite token 尾部（按你的修正要求）。

#### Prompt embedding（稀疏提示）

- 来自 `sparse_embeddings`，若维度不匹配则经 `prompt_proj` 投影到 decoder 维度 `C`。
- prompt 的 RoPE 位置由 `_build_prompt_positions` 构造：
  - 若有 `prompt_coords`，映射到 patch 网格坐标；
  - 否则使用默认 `(0,0)`。
- **原来（V3）**：local 阶段拼接到 satellite 分支（历史实现）。
- **现在（V4）**：local 阶段拼接到 front token 尾部（按你的修正要求）。

#### Position embedding（RoPE）

- patch token 使用 `PositionGetter` 生成二维坐标；register/learnable 默认零坐标。
- Local 阶段位置布局：
  - satellite 分支：`[sate_pos | learnable_pos]`
  - front 分支：`[front_pos | prompt_pos]`
- Global 阶段位置布局：`[sate_pos | front_pos | learnable_pos | prompt_pos]`。

#### 与原来相比的核心变化

- token 归属对调：`learnable -> satellite(local)`，`prompt -> front(local)`。
- global 阶段新增 `learnable ↔ prompt` 双向不可见，进一步解耦功能 token。
- 输出接口保持不变：仍输出 `learnable_out` 给下游检测/heatmap 分支使用。

---

## 2. Heatmap Loss 改动

修改文件：

- `utils/losses_v2.py`

### 2.1 修改前

- `loss_heatmap = MSE(pred_position, target_position)`。
- 只在 2D 坐标上做回归，未显式约束热图分布形状。
- 与 heatmap head 的像素级输出耦合较弱。

### 2.2 修改后（CornerNet 变体）

- 基于 `heatmap_logits`（`[B, H, W]`）做逐像素监督：
  - `pred = sigmoid(logits)`；
  - 用目标位置生成高斯 target heatmap；
  - 使用 CornerNet 风格 focal 公式计算正负样本项。

核心形式（简化描述）：

- `pos_loss = log(pred) * (1 - pred)^alpha * pos_inds`
- `neg_loss = log(1 - pred) * pred^alpha * (1 - target)^beta * neg_inds`
- 归一化方式：`-(pos_loss + neg_loss) / num_pos`（无正样本时退化为 `-neg_loss`）

默认参数：

- `heatmap_sigma = 0.05`（归一化坐标系）
- `heatmap_focal_alpha = 2.0`
- `heatmap_focal_beta = 4.0`

### 2.3 行为差异总结

- 从“坐标点回归”变为“热图分布学习 + 隐式坐标定位”。
- 梯度覆盖整个空间网格，训练信号更密集。
- 对多峰/近邻不确定区域更鲁棒，不依赖 argmax 作为训练目标。

### 2.4 Heatmap loss 设计细节（实现级）

在 `DETRCriterionV2._compute_heatmap_loss` 中，heatmap loss 按以下步骤构建：

1. **输入与维度约束**
   - 使用 `outputs['heatmap_logits']` 作为监督输入，形状为 `[B, H, W]`。
   - 使用 `targets['camera_position']` 作为目标位置，形状 `[B, 2]`，坐标范围归一化到 `[0, 1]`。

2. **构造高斯目标热图**
   - 在归一化坐标平面生成网格 `grid_x, grid_y`。
   - 对每个样本以 `(x_t, y_t)` 为中心生成高斯：
     - `target = exp(-((x-x_t)^2 + (y-y_t)^2) / (2*sigma^2))`
   - 将 `target` 裁剪到 `[0, 1]`，用于定义正负样本与负样本权重。

3. **CornerNet 风格 focal 分解**
   - 概率：`pred = sigmoid(logits)`，并做 `clamp(1e-4, 1-1e-4)` 保证数值稳定。
   - 正样本：`target >= 0.999`，对应峰值中心区域。
   - 负样本：`target < 0.999`，并使用 `(1-target)^beta` 作为难度权重，离中心越远权重越大。
   - 正项：鼓励中心像素概率升高；负项：抑制背景与旁瓣响应。

4. **归一化策略**
   - 若存在正样本，按 `num_pos` 归一化，保证不同 batch/尺度下量级稳定。
   - 若不存在正样本（极端情况），退化为纯负样本项，避免 NaN。

5. **监控指标与训练目标解耦**
   - 训练目标：像素级 focal heatmap loss。
   - 监控指标：`pos_error` 继续保留，优先使用模型输出的 `position` 计算（若缺失则回退到 logits argmax 近似位置）。
   - 这样可以同时获得“分布监督”与“坐标可解释评估”。

### 2.5 Heatmap loss 数学公式定义

设第 `b` 个样本的目标位置为 `p_b=(x_b,y_b)`，预测 logits 为
`Z_b ∈ R^{H×W}`。令像素归一化坐标网格为：

\[
u_i=\frac{i}{W-1},\quad v_j=\frac{j}{H-1},
\]

其中 `i∈{0,\dots,W-1}`，`j∈{0,\dots,H-1}`。

1) 目标高斯热图：

\[
Y_b(i,j)=\exp\left(-\frac{(u_i-x_b)^2+(v_j-y_b)^2}{2\sigma^2}\right).
\]

2) 预测概率（数值稳定）：

\[
\hat{P}_b(i,j)=\mathrm{clip}\!\left(\mathrm{sigmoid}(Z_b(i,j)),\ \varepsilon,\ 1-\varepsilon\right),
\]

实现中 `\varepsilon=10^{-4}`。

3) 正负样本及负样本权重：

\[
\mathbf{1}^{+}_{b,i,j}=\mathbf{1}[Y_b(i,j)\ge 0.999],\quad
\mathbf{1}^{-}_{b,i,j}=\mathbf{1}[Y_b(i,j)<0.999],
\]

\[
w^{-}_{b,i,j}=(1-Y_b(i,j))^{\beta}.
\]

4) CornerNet 变体 focal loss：

\[
\mathcal{L}_{pos}=
\sum_{b,i,j}\log \hat{P}_b(i,j)\,(1-\hat{P}_b(i,j))^{\alpha}\,\mathbf{1}^{+}_{b,i,j},
\]

\[
\mathcal{L}_{neg}=
\sum_{b,i,j}\log(1-\hat{P}_b(i,j))\,\hat{P}_b(i,j)^{\alpha}\,w^{-}_{b,i,j}\,\mathbf{1}^{-}_{b,i,j}.
\]

设 `N_{pos}=\sum_{b,i,j}\mathbf{1}^{+}_{b,i,j}`，最终：

\[
\mathcal{L}_{heatmap}=
\begin{cases}
-\dfrac{\mathcal{L}_{pos}+\mathcal{L}_{neg}}{N_{pos}}, & N_{pos}>0,\\[6pt]
-\mathcal{L}_{neg}, & N_{pos}=0.
\end{cases}
\]

默认超参：`\alpha=2.0`，`\beta=4.0`，`\sigma=0.05`。

---

## 3. 兼容性与影响

- 对模型输出字段兼容：`heatmap`、`heatmap_logits`、`position` 均保留。
- 训练入口无需强制改参数（新 loss 参数均提供默认值）。
- deep supervision 下的中间层 heatmap loss 同步切换为 focal 版本。
- 监控指标 `pos_error` 继续保留（优先使用 `outputs['position']`）。

---

## 4. 变更清单

- `models/backbone/pi3_backbone_v2.py`
  - local attention 改为 frame-wise self-attention：
    - sate 分支拼接 learnable
    - front 分支拼接 prompt
  - global attention 增加 learnable<->prompt 双向不可见 mask
  - learnable token 改为独立 token 流维护并输出

- `utils/losses_v2.py`
  - heatmap loss 从 position MSE 改为 CornerNet 风格像素级 focal loss
  - 新增 heatmap focal 相关超参（含默认值）
  - 保留 `pos_error` 统计逻辑

