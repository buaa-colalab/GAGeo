# CHANGES_V3

分支：`feature/unified-backbone-v3`

本文档记录 V3 相对 V2 的核心改动与实现位置。

## 1) 可视化样本增加 mask 面板

- 文件：`utils/visualize.py`
- 改动：训练可视化从 3 列扩展为 4 列，新增 **Mask 对比面板**：
  - GT mask（绿色）与 Pred mask（红色）叠加显示
  - 显示 `IoU` 与 `Dice`
- 目的：更直观监控分割质量，便于早期发现 mask 训练异常。

## 2) mask token 不再直接加到 front 输入；仅在 global attention 的 K 中使用 front+mask

- 文件：`models/backbone/pi3_backbone_v2.py`
- 核心改动：
  - 移除将 dense mask embedding 直接加到 `front_hidden` 的路径
  - 构建 `front_with_mask`（front + dense_mask）只用于 global 阶段的 **Key**
  - global 阶段改为：
    - `Q` / `V`：来自原始 token 串（front + sate + prompt）
    - `K`：来自（front+mask）+ sate + prompt
- 目的：把 mask 信息约束到检索侧（K），避免直接污染 front 表征。

## 3) local 阶段改为更强的 masked-global 交互

- 文件：`models/backbone/pi3_backbone_v2.py`
- 核心改动：
  - local 阶段不再是简单局部自注意力，而是对全 token 做 masked-global attention
  - 通过 `_build_local_attn_mask(...)` 严格控制可见性：
    - learnable 可与 front / sate 双向交互
    - front 与 sate 保持互相屏蔽
    - prompt 不参与 local 阶段（整体屏蔽）
- 目的：提升 learnable token 的融合能力，同时保持跨视角屏蔽规则。

## 4) prompt 类型最多 1 个（point / bbox / mask 三选一）

- 文件：`utils/prompt_utils.py`
- 核心改动：
  - 随机 prompt 采样时强制仅选一种类型
  - `max_prompts` 限制为 1
- 目的：降低 prompt 组合复杂度，稳定训练分布，便于归因分析。

## 5) 在中间监督层（如 4、11）增加 heatmap 与 rotation 监督

- 文件：
  - `models/cross_view_localizer_v2.py`
  - `utils/losses_v2.py`
- 核心改动：
  - 在非最终监督层额外输出 `intermediate_heatmap_preds` 与 `intermediate_rotation_preds`
  - 在 loss 中新增对应中间层损失聚合：
    - `inter_*_loss_heatmap`
    - `inter_*_loss_rotation`
  - 最终 `total_loss` 纳入上述中间层 heatmap/rotation 项
- 目的：让中层特征更早学习空间定位与姿态信息，改善训练信号密度。

## 6) 代码结构性补充

- 文件：`models/backbone/pi3_backbone_v2.py`
- 新增：
  - 注意力模块支持解耦输入接口（`forward_qkv`），便于实现 `Q/K/V` 异源输入。
  - 中间输出新增 `front_patches`，便于上层中间监督复用 front 特征。

---

## 兼容性说明

- 上述改动保持了 V2 主体接口与训练脚本组织方式。
- 训练时建议优先做一次前向+损失 smoke test，确认：
  - 注意力 mask 维度
  - intermediate 监督键值完整性
  - 日志中新增损失项可正常记录
