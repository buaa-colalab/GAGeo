# Cross-View Localization with Unified Backbone and Multi-Task Learning

## Abstract

We present a unified architecture for cross-view drone localization that simultaneously predicts object bounding boxes, segmentation masks, camera position, and rotation from a monocular front-view image to a satellite map. Our approach leverages a unified Pi3 backbone with learnable queries, deep supervision, and a comprehensive multi-task loss function. The model supports flexible geometric prompts (points, bounding boxes, or masks) to specify the target object in the front view, enabling accurate localization in the satellite view.

---

## 1. Introduction

Cross-view localization is a fundamental task in drone navigation that requires establishing correspondences between a ground-level monocular view and an overhead satellite map. Unlike traditional approaches that use separate modules for different tasks, we propose a unified architecture that jointly learns object detection, segmentation, and camera pose estimation through a shared backbone with learnable queries.

---

## 2. Model Architecture

### 2.1 Overall Pipeline

Our model follows a three-stage pipeline: (1) **Feature Extraction** using a unified Pi3 backbone, (2) **Query-based Decoding** with learnable queries, and (3) **Multi-task Prediction** through specialized heads.

```
Input: Front-view image I_f, Satellite image I_s, Geometric prompts P
  ↓
[Pi3 Backbone] → F_f, F_s, Q_learnable
  ↓
[Task Heads] → BBox, Mask, Position, Rotation
```

### 2.2 Unified Backbone

The backbone is based on the Pi3 (Pose-conditioned Image-to-Image-to-Image) architecture, which consists of:

- **DINOv2 Encoder**: Extracts patch-level features from both views
  - Front view: \(F_f \in \mathbb{R}^{B \times N_p \times D}\) where \(N_p = 37 \times 37 = 1369\) patches
  - Satellite view: \(F_s \in \mathbb{R}^{B \times N_p \times D}\) with \(D = 2048\) for large decoder

- **Pi3 Decoder**: Processes cross-view features with learnable queries
  - Learnable queries: \(Q \in \mathbb{R}^{B \times N_q \times D}\) where \(N_q = N_{bbox} + N_{heatmap}\)
  - Custom attention masks control token interactions between views
  - RoPE (Rotary Position Embedding) for spatial awareness

The backbone outputs:
- Front view features: \(F_f \in \mathbb{R}^{B \times N_p \times D}\)
- Satellite view features: \(F_s \in \mathbb{R}^{B \times N_p \times D}\)
- Learnable query features: \(Q \in \mathbb{R}^{B \times N_q \times D}\)

### 2.3 Learnable Query Design

We split learnable queries into two groups:

- **BBox/Mask Queries** (\(Q_{bbox} \in \mathbb{R}^{B \times N_{bbox} \times D}\)): Used for object detection and segmentation
- **Heatmap Queries** (\(Q_{heat} \in \mathbb{R}^{B \times N_{heat} \times D}\)): Used for camera position prediction

The queries are initialized as learnable parameters and updated through the decoder layers via cross-attention with spatial features.

### 2.4 Task-Specific Heads

#### 2.4.1 Bounding Box Head

The bbox head follows a DETR-style design:

\[
\text{bbox\_head}(Q_{bbox}) = \{\text{pred\_boxes}, \text{bbox\_scores}, \text{class\_logits}\}
\]

where:
- \(\text{pred\_boxes} \in \mathbb{R}^{B \times N_{bbox} \times 4}\): Normalized bounding boxes \((c_x, c_y, w, h)\)
- \(\text{bbox\_scores} \in \mathbb{R}^{B \times N_{bbox}}\): Confidence scores
- \(\text{class\_logits} \in \mathbb{R}^{B \times N_{bbox} \times 1}\): Binary classification logits (object vs. no-object)

The head consists of:
- A 3-layer MLP for bbox regression: \(\text{MLP}: \mathbb{R}^D \rightarrow \mathbb{R}^4\)
- A linear layer for classification: \(\text{Linear}: \mathbb{R}^D \rightarrow \mathbb{R}^1\)

#### 2.4.2 Mask Head

The mask head adopts a SAM-style architecture:

\[
\text{mask\_head}(Q_{bbox}, F_s) = \{\text{mask\_logits}, \text{mask\_pred}, \text{iou\_pred}\}
\]

**Architecture**:
1. **Spatial Feature Upscaling**: \(F_s\) is upscaled from \(37 \times 37\) to \(148 \times 148\) via two transposed convolutions
2. **Hypernetwork**: Each query generates a dynamic convolution kernel
   \[
   K_q = \text{MLP}(Q_{bbox}[q]) \in \mathbb{R}^{D/8}
   \]
3. **Mask Generation**: Dot product between kernel and upscaled features
   \[
   M_q = K_q^T \cdot F_s^{upscaled} \in \mathbb{R}^{148 \times 148}
   \]
4. **Interpolation**: Upsampled to output size \(518 \times 518\)

The head outputs:
- \(\text{mask\_logits} \in \mathbb{R}^{B \times N_{bbox} \times 518 \times 518}\): Raw logits
- \(\text{mask\_pred} \in \mathbb{R}^{B \times N_{bbox} \times 518 \times 518}\): Sigmoid probabilities
- \(\text{iou\_pred} \in \mathbb{R}^{B \times N_{bbox}}\): Predicted IoU scores

#### 2.4.3 Heatmap Head

The heatmap head predicts camera position as a probability distribution:

\[
\text{heatmap\_head}(Q_{heat}, F_s) = \{\text{heatmap}, \text{position}\}
\]

**Process**:
1. **Query Combination**: Weighted combination of heatmap queries
   \[
   Q_{combined} = \sum_{i=1}^{N_{heat}} w_i \cdot Q_{heat}[i], \quad w = \text{softmax}(\text{Linear}(Q_{heat}))
   \]

2. **Dot Product**: Query embedding × spatial features
   \[
   H = Q_{combined}^T \cdot F_s \in \mathbb{R}^{37 \times 37}
   \]

3. **Upsampling & Softmax**: Convert to probability distribution
   \[
   H_{prob} = \text{softmax}(\text{upsample}(H)) \in \mathbb{R}^{518 \times 518}
   \]

4. **Position Extraction**: Soft-argmax to extract 2D coordinates
   \[
   p = \text{soft-argmax}(H_{prob}) \in \mathbb{R}^2
   \]

#### 2.4.4 Camera Rotation Head

The rotation head predicts camera orientation:

\[
\text{camera\_head}(F_f, F_s) = \{\text{rotation\_matrix}, \text{yaw}, \text{pitch}, \text{roll}\}
\]

**Architecture**:
1. **Transformer Decoder**: Processes patch features from both views with RoPE
2. **ResConv Blocks**: Extracts pose features
3. **Pose Prediction**: Outputs 4×4 SE(3) transformation matrix
4. **Euler Angles**: Extracts yaw, pitch, roll from rotation matrix

---

## 3. Loss Function Design

Our loss function combines multiple objectives with a Hungarian matching strategy for multi-query scenarios.

### 3.1 Hungarian Matching

For single-target detection with multiple queries, we use Hungarian matching to assign predictions to ground truth:

\[
\mathcal{C}_{ij} = \lambda_{bbox} \cdot \|\hat{b}_i - b_{gt}\|_1 + \lambda_{giou} \cdot (1 - \text{GIoU}(\hat{b}_i, b_{gt})) + \lambda_{class} \cdot (1 - \sigma(\hat{c}_i))
\]

where:
- \(\hat{b}_i\): Predicted bbox for query \(i\)
- \(b_{gt}\): Ground truth bbox
- \(\hat{c}_i\): Classification logit for query \(i\)
- \(\lambda_{bbox} = 5.0\), \(\lambda_{giou} = 2.0\), \(\lambda_{class} = 1.0\)

The matching selects the query with minimum cost for each ground truth.

### 3.2 Bounding Box Loss

For matched predictions, we compute:

\[
\mathcal{L}_{bbox} = \lambda_{bbox} \cdot \mathcal{L}_{L1} + \lambda_{giou} \cdot \mathcal{L}_{GIoU}
\]

where:
- **L1 Loss**: \(\mathcal{L}_{L1} = \frac{1}{B} \sum_{b=1}^{B} \|\hat{b}_b - b_{gt,b}\|_1\)
- **GIoU Loss**: \(\mathcal{L}_{GIoU} = \frac{1}{B} \sum_{b=1}^{B} (1 - \text{GIoU}(\hat{b}_b, b_{gt,b}))\)

### 3.3 Classification Loss

We use **Sigmoid Focal Loss** for binary classification (object vs. no-object):

\[
\mathcal{L}_{class} = \frac{1}{B \cdot N_q} \sum_{b=1}^{B} \sum_{q=1}^{N_q} \alpha_t \cdot \text{CE}_{b,q} \cdot (1 - p_t)^{\gamma}
\]

where:
- **Binary Cross-Entropy**: 
  \[
  \text{CE}_{b,q} = -[y_{b,q} \log(p_{b,q}) + (1-y_{b,q}) \log(1-p_{b,q})]
  \]
- **Prediction Probability**: \(p_{b,q} = \sigma(\text{class\_logit}_{b,q})\)
- **Correctness Probability**: 
  \[
  p_t = \begin{cases}
  p_{b,q} & \text{if } y_{b,q} = 1 \text{ (positive)} \\
  1 - p_{b,q} & \text{if } y_{b,q} = 0 \text{ (negative)}
  \end{cases}
  \]
- **Alpha Weighting**: 
  \[
  \alpha_t = \begin{cases}
  0.25 & \text{if } y_{b,q} = 1 \\
  0.75 & \text{if } y_{b,q} = 0
  \end{cases}
  \]
- **Focal Parameter**: \(\gamma = 2.0\)

**Key Design**:
- Matched queries: \(y = 1\) (positive samples)
- Unmatched queries: \(y = 0\) (negative samples)
- Focal loss focuses on hard examples (low \(p_t\)) and suppresses easy ones (high \(p_t\))

### 3.4 Mask Loss

For matched mask predictions, we compute:

\[
\mathcal{L}_{mask} = \lambda_{bce} \cdot \mathcal{L}_{BCE} + \lambda_{dice} \cdot \mathcal{L}_{Dice}
\]

where:
- **BCE Loss**: 
  \[
  \mathcal{L}_{BCE} = \frac{1}{B \cdot H \cdot W} \sum_{b=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} \text{BCE}(\hat{m}_{b,h,w}, m_{gt,b,h,w})
  \]

- **Dice Loss**: 
  \[
  \mathcal{L}_{Dice} = \frac{1}{B} \sum_{b=1}^{B} \left(1 - \frac{2 \cdot |\hat{m}_b \cap m_{gt,b}| + \epsilon}{|\hat{m}_b| + |m_{gt,b}| + \epsilon}\right)
  \]

where \(\hat{m}_b = \sigma(\text{mask\_logits}_b)\) and \(\epsilon = 1.0\) is a smoothing factor.

**Mask Selection**: Only the mask corresponding to the Hungarian-matched bbox query is used for loss computation.

### 3.5 Heatmap Loss

The heatmap loss is a simple MSE on the extracted position:

\[
\mathcal{L}_{heatmap} = \lambda_{heat} \cdot \|\hat{p} - p_{gt}\|_2^2
\]

where:
- \(\hat{p} \in \mathbb{R}^2\): Predicted position from soft-argmax
- \(p_{gt} \in \mathbb{R}^2\): Ground truth camera position (normalized to [0, 1])

### 3.6 Rotation Loss

The rotation loss uses geodesic distance on SO(3):

\[
\mathcal{L}_{rotation} = \lambda_{rot} \cdot d_{geodesic}(\hat{R}, R_{gt})
\]

For smooth training, we use:

\[
d_{geodesic}(\hat{R}, R_{gt}) = \|\log(\hat{R}^T R_{gt})\|_F
\]

where \(\log(\cdot)\) is the matrix logarithm mapping SO(3) to the Lie algebra \(\mathfrak{so}(3)\).

Alternatively, for numerical stability:

\[
d_{geodesic}(\hat{R}, R_{gt}) = \arccos\left(\frac{\text{tr}(\hat{R}^T R_{gt}) - 1}{2}\right)
\]

### 3.7 Contrastive Loss

We employ a MoCo-style contrastive loss for cross-view feature alignment:

\[
\mathcal{L}_{contrastive} = \lambda_{contrast} \cdot \mathcal{L}_{MoCo}
\]

**Process**:
1. **Feature Pooling**: Masked average pooling of patch features
   \[
   f_f = \frac{\sum_{i} m_f[i] \cdot F_f[i]}{\sum_{i} m_f[i]}, \quad f_s = \frac{\sum_{i} m_s[i] \cdot F_s[i]}{\sum_{i} m_s[i]}
   \]

2. **Projection**: Project to contrastive space
   \[
   z_f = \text{MLP}(f_f), \quad z_s = \text{MLP}(f_s)
   \]

3. **MoCo Loss**: InfoNCE with momentum encoder and queue
   \[
   \mathcal{L}_{MoCo} = -\log \frac{\exp(z_f \cdot z_s^+ / \tau)}{\exp(z_f \cdot z_s^+ / \tau) + \sum_{z^- \in Q} \exp(z_f \cdot z^- / \tau)}
   \]

where:
- \(z_s^+\): Positive sample (momentum encoder output)
- \(Q\): Negative queue (size 16384)
- \(\tau = 0.07\): Temperature parameter

### 3.8 Deep Supervision

To improve gradient flow and training stability, we apply deep supervision at intermediate decoder layers:

\[
\mathcal{L}_{deep} = \sum_{l \in \{4, 11, 17\}} w_l \cdot \mathcal{L}_{intermediate}^{(l)}
\]

where:
- \(w_4 = 0.1\), \(w_{11} = 0.3\), \(w_{17} = 0.6\): Layer-specific weights
- \(\mathcal{L}_{intermediate}^{(l)}\): Loss computed at layer \(l\) using intermediate predictions

**Intermediate Losses**:
- BBox loss: \(\mathcal{L}_{bbox}^{(l)}\)
- Mask loss: \(\mathcal{L}_{mask}^{(l)}\)
- Heatmap loss: \(\mathcal{L}_{heatmap}^{(l)}\) (only for layers 4, 11)
- Rotation loss: \(\mathcal{L}_{rotation}^{(l)}\) (only for layers 4, 11)

### 3.9 Total Loss

The final loss is a weighted combination of all components:

\[
\mathcal{L}_{total} = \mathcal{L}_{bbox} + \mathcal{L}_{giou} + \mathcal{L}_{class} + \mathcal{L}_{mask} + \mathcal{L}_{heatmap} + \mathcal{L}_{rotation} + \mathcal{L}_{contrastive} + \mathcal{L}_{deep}
\]

**Default Weights**:
- \(\lambda_{bbox} = 5.0\), \(\lambda_{giou} = 2.0\), \(\lambda_{class} = 2.0\)
- \(\lambda_{bce} = 2.0\), \(\lambda_{dice} = 5.0\)
- \(\lambda_{heat} = 0.1\), \(\lambda_{rot} = 0.1\), \(\lambda_{contrast} = 0.1\)

---

## 4. Training Strategy

### 4.1 Multi-Query Learning

With multiple queries (\(N_{bbox} > 1\)), the model learns to:
- **Specialize**: Different queries may focus on different aspects of the object
- **Robustness**: Multiple predictions provide redundancy
- **Selection**: At inference, we select the query with highest confidence score

### 4.2 Ablation Components

The model supports configurable ablation studies:
- **Deep Supervision**: Enable/disable intermediate layer supervision
- **Contrastive Loss**: Enable/disable MoCo-style cross-view alignment
- **Rotation/Position Supervision**: Enable/disable pose-related losses
- **Heatmap Loss**: Enable/disable heatmap-based position loss

### 4.3 Optimization

- **Optimizer**: AdamW with weight decay \(10^{-4}\)
- **Learning Rates**: 
  - Backbone: \(10^{-5}\) (low, pretrained)
  - New tokens: \(5 \times 10^{-4}\) (mid)
  - Task heads: \(10^{-4}\) (high)
- **Scheduler**: Cosine annealing with 5-epoch warmup
- **Mixed Precision**: BF16 for RTX 5090 GPUs

---

## 5. Key Design Choices

### 5.1 Unified Backbone

Unlike previous approaches that use separate encoders for different views, our unified backbone processes both views together, enabling better cross-view feature alignment.

### 5.2 Learnable Queries

Learnable queries provide a flexible mechanism for multi-task learning:
- Shared queries for bbox and mask (spatial consistency)
- Separate queries for heatmap (position-specific)
- Hungarian matching ensures proper assignment in multi-query scenarios

### 5.3 Deep Supervision

Deep supervision at multiple layers (early, mid, late) improves:
- Gradient flow through deep networks
- Feature learning at different abstraction levels
- Training stability and convergence speed

### 5.4 Focal Loss for Classification

Focal loss addresses class imbalance in single-target detection:
- Many negative queries (unmatched) vs. one positive query (matched)
- Automatically focuses on hard examples
- Prevents negative samples from dominating the loss

---

## 6. Conclusion

We present a unified architecture for cross-view localization that jointly learns object detection, segmentation, and camera pose estimation. The model leverages learnable queries, deep supervision, and a comprehensive multi-task loss function to achieve accurate localization from monocular front-view images to satellite maps. The flexible design supports various ablation studies and can be adapted to different scenarios.

---

## References

- DETR: End-to-End Object Detection with Transformers
- SAM: Segment Anything Model
- Pi3: Pose-conditioned Image-to-Image-to-Image
- MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
- Focal Loss for Dense Object Detection
