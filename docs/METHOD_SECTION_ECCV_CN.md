# 方法部分（中文翻译版）

## 3. 方法

我们提出了一种用于跨视角无人机定位的统一架构，该架构联合处理目标检测、分割和相机姿态估计。我们的方法利用具有可学习任务令牌的3D感知骨干网络，实现跨几何提示、前视图图像和卫星图的多模态推理。该架构通过统一的Transformer骨干网络处理所有模态，消除了对独立编码器的需求，并实现了跨视角对应关系的端到端学习。

### 3.1 整体架构

我们的模型以前视图图像$I_f$、卫星图像$I_s$和几何提示$P$作为输入，这些提示指定了前视图中的目标对象。提示可以是点、边界框或掩码，遵循SAM2提示编码范式。该架构遵循统一的基于令牌的处理流程，其中所有模态都被编码到共同的表示空间并联合处理。

流程首先使用DINOv2编码器对两张图像进行编码，该编码器从前视图和卫星视图分别提取补丁级特征$F_f \in \mathbb{R}^{B \times N_p \times D}$和$F_s \in \mathbb{R}^{B \times N_p \times D}$，其中$N_p = 37 \times 37 = 1369$个补丁，$D = 2048$用于大型解码器配置。几何提示通过预训练的SAM2提示编码器处理，该编码器产生编码点和框坐标的稀疏提示令牌$T_p \in \mathbb{R}^{B \times N_p \times D}$，以及用于掩码提示的密集嵌入$E_d \in \mathbb{R}^{B \times D \times H \times W}$。

关键创新在于这些令牌如何统一和处理。引入了可学习任务令牌$Q \in \mathbb{R}^{B \times N_q \times D}$，其中$N_q = N_{bbox} + N_{heatmap}$表示用于边界框检测和相机位置预测的查询。这些任务令牌与提示令牌和图像补丁令牌一起，通过具有交替局部和全局注意力机制的Pi3骨干网络进行处理。骨干网络通过允许来自不同模态的令牌通过精心设计的注意力掩码相互关注来实现跨视角推理，这些掩码尊重视图之间的几何关系。

可学习任务令牌从骨干网络中出现，具有丰富的跨模态表示，已经关注了提示编码的前视图和卫星空间特征。然后这些令牌由任务特定的头部解码以产生最终预测：用于目标定位的边界框和掩码，以及用于姿态估计的相机位置和旋转。统一处理确保所有预测都受益于相同的丰富跨视角特征表示，从而实现一致的多任务学习。

### 3.2 3D感知骨干网络

选择Pi3作为我们的骨干网络是出于其固有的3D几何感知能力，这对于跨视角定位任务至关重要。与独立处理视图的VGGT风格架构不同，Pi3被设计用于推理3D空间和相机几何，使其自然适合在地面级和俯视图之间建立对应关系。

Pi3骨干网络采用交替的局部和全局注意力层，其中局部注意力在空间邻域内操作以捕获细粒度细节，而全局注意力实现整个图像上的长程依赖。这种交替模式对于跨视角任务特别有效，因为它允许模型首先通过几何推理建立局部对应关系，然后全局传播这些信息以细化预测。

几何感知通过旋转位置嵌入（RoPE）嵌入，该嵌入将2D空间坐标直接编码到注意力机制中。这使得模型能够理解补丁之间的空间关系，这对于在不同视角下匹配对象至关重要。注意力掩码通过控制哪些令牌可以相互关注来进一步强制执行几何约束，确保跨视角推理尊重底层3D几何。

骨干网络同时处理两个视图在统一表示空间中的能力实现了姿态感知特征学习。当模型通过多个层处理令牌时，它学习在保持对它们之间相机姿态关系的感知的同时跨视图对齐特征。这种几何理解对于准确定位至关重要，因为它允许模型推理对象在地面级与俯视图中的不同 appearance。

### 3.3 任务特定组件

可学习任务令牌根据其预期用途分为两组：用于目标检测和分割的bbox/掩码查询$Q_{bbox} \in \mathbb{R}^{B \times N_{bbox} \times D}$，以及用于相机位置预测的热图查询$Q_{heat} \in \mathbb{R}^{B \times N_{heat} \times D}$。这种分离允许每组在共享来自骨干网络的相同丰富跨视角表示的同时进行专门化。

边界框头部遵循DETR风格设计，其中每个查询令牌预测一个归一化边界框$(c_x, c_y, w, h)$和置信度分数。头部由用于回归的3层MLP和用于二分类（对象vs.无对象）的线性层组成。使用相同查询进行bbox和掩码预测的设计选择确保了空间一致性，因为两个任务都需要在卫星视图中定位同一对象。

掩码头部采用SAM风格的超网络架构，其中每个查询令牌生成一个动态卷积核，该核应用于上采样的卫星空间特征。这种设计通过允许每个查询关注对象形状的不同方面来实现细粒度掩码预测。超网络生成维度为$D/8$的核，然后与上采样特征进行点积以产生掩码logits。这种方法比为每个查询使用独立解码器更参数高效，同时仍允许查询特定的掩码生成。

相机位置通过热图头部预测，该头部在卫星图像上生成概率分布。热图查询首先通过学习的加权组合进行组合，然后用于与卫星空间特征计算点积。生成的热图被上采样并通过softmax归一化以形成概率分布，从中使用soft-argmax提取相机位置。这种概率公式提供不确定性估计并实现端到端可微训练。

相机旋转由专用头部预测，该头部通过具有RoPE位置编码的Transformer解码器处理来自两个视图的补丁特征。头部输出4×4 SE(3)变换矩阵，从中提取欧拉角（偏航、俯仰、滚转）。使用补丁特征而不是单个相机令牌允许模型利用空间上下文进行更准确的姿态估计。

统一查询设计通过允许不同任务在保持任务特定专门化的同时共享表示来实现多任务学习。bbox和掩码查询共享相同的令牌，确保检测和分割预测在空间上一致。热图查询是独立的，允许它们专注于位置特定特征。这种设计平衡了共享表示学习与任务特定适应。

### 3.4 训练目标

我们的训练目标结合了多个损失和匈牙利匹配策略，以处理单目标检测中的多查询场景。匹配成本函数结合了边界框距离、GIoU重叠和分类置信度：

\begin{equation}
\mathcal{C}_{ij} = \lambda_{bbox} \|\hat{b}_i - b_{gt}\|_1 + \lambda_{giou} (1 - \text{GIoU}(\hat{b}_i, b_{gt})) + \lambda_{class} (1 - \sigma(\hat{c}_i))
\end{equation}

其中$\hat{b}_i$是查询$i$的预测边界框，$b_{gt}$是真实框，$\hat{c}_i$是分类logit，$\lambda_{bbox} = 5.0$、$\lambda_{giou} = 2.0$、$\lambda_{class} = 1.0$是权重因子。匈牙利算法为每个真实值选择成本最小的查询，确保只有一个查询被分配给单个目标，而其他查询学习预测"无对象"。

对于匹配的预测，边界框损失结合了L1距离和GIoU重叠：

\begin{equation}
\mathcal{L}_{bbox} = \lambda_{bbox} \frac{1}{B} \sum_{b=1}^{B} \|\hat{b}_b - b_{gt,b}\|_1 + \lambda_{giou} \frac{1}{B} \sum_{b=1}^{B} (1 - \text{GIoU}(\hat{b}_b, b_{gt,b}))
\end{equation}

分类损失使用sigmoid focal loss来解决单目标检测中的严重类别不平衡，其中许多查询是负的（未匹配），只有一个查询是正的（匹配）。Focal loss公式为：

\begin{equation}
\mathcal{L}_{class} = \frac{1}{B \cdot N_q} \sum_{b=1}^{B} \sum_{q=1}^{N_q} \alpha_t \cdot \text{CE}_{b,q} \cdot (1 - p_t)^{\gamma}
\end{equation}

其中$\text{CE}_{b,q} = -[y_{b,q} \log(p_{b,q}) + (1-y_{b,q}) \log(1-p_{b,q})]$是二分类交叉熵，$p_{b,q} = \sigma(\text{class\_logit}_{b,q})$是预测概率，$p_t$是正确预测的概率（对于正样本等于$p_{b,q}$，对于负样本等于$1-p_{b,q}$）。Focal参数$\gamma = 2.0$降低简单样本的权重，而$\alpha_t = 0.25$用于正样本，$0.75$用于负样本平衡每个类别的贡献。这种公式确保模型专注于困难样本，同时防止负样本主导损失。

掩码损失结合了匹配掩码预测的二进制交叉熵和Dice损失：

\begin{equation}
\mathcal{L}_{mask} = \lambda_{bce} \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} \text{BCE}(\hat{m}_{b,h,w}, m_{gt,b,h,w}) + \lambda_{dice} \frac{1}{B} \sum_{b=1}^{B} \left(1 - \frac{2|\hat{m}_b \cap m_{gt,b}| + \epsilon}{|\hat{m}_b| + |m_{gt,b}| + \epsilon}\right)
\end{equation}

其中$\hat{m}_b = \sigma(\text{mask\_logits}_b)$，$\epsilon = 1.0$是平滑因子。只有与匈牙利匹配的bbox查询对应的掩码用于损失计算，确保检测和分割之间的一致性。

相机位置通过提取位置的简单MSE损失进行监督：

\begin{equation}
\mathcal{L}_{heatmap} = \lambda_{heat} \|\hat{p} - p_{gt}\|_2^2
\end{equation}

其中$\hat{p} \in \mathbb{R}^2$是从soft-argmax提取的预测位置，$p_{gt}$是归一化到$[0,1]$的真实相机位置。

相机旋转使用SO(3)上的测地距离进行监督：

\begin{equation}
\mathcal{L}_{rotation} = \lambda_{rot} \|\log(\hat{R}^T R_{gt})\|_F
\end{equation}

其中$\log(\cdot)$将旋转矩阵映射到李代数$\mathfrak{so}(3)$。这种公式尊重旋转的流形结构，并在训练期间提供平滑梯度。

跨视图特征对齐通过MoCo风格的对比损失强制执行。掩码平均池化从前视图和卫星补丁特征中提取全局特征$f_f$和$f_s$，然后通过MLP投影到对比空间。具有动量编码器和负队列的InfoNCE损失鼓励对齐的表示：

\begin{equation}
\mathcal{L}_{contrastive} = \lambda_{contrast} \left(-\log \frac{\exp(z_f \cdot z_s^+ / \tau)}{\exp(z_f \cdot z_s^+ / \tau) + \sum_{z^- \in Q} \exp(z_f \cdot z^- / \tau)}\right)
\end{equation}

其中$z_s^+$是来自动量编码器的正样本，$Q$是大小为16384的负队列，$\tau = 0.07$是温度参数。

为了改善梯度流和训练稳定性，我们在中间解码器层（第4、11和17层）应用深度监督。使用附加到这些层的轻量级头部计算中间预测，并使用层特定权重$w_4 = 0.1$、$w_{11} = 0.3$、$w_{17} = 0.6$计算损失：

\begin{equation}
\mathcal{L}_{deep} = \sum_{l \in \{4, 11, 17\}} w_l \cdot \mathcal{L}_{intermediate}^{(l)}
\end{equation}

中间损失包括bbox、掩码、热图（对于第4和11层）和旋转（对于第4和11层）损失，使模型能够在不同抽象级别学习有意义的表示。

总训练损失是所有组件的加权组合：

\begin{equation}
\mathcal{L}_{total} = \mathcal{L}_{bbox} + \mathcal{L}_{giou} + \mathcal{L}_{class} + \mathcal{L}_{mask} + \mathcal{L}_{heatmap} + \mathcal{L}_{rotation} + \mathcal{L}_{contrastive} + \mathcal{L}_{deep}
\end{equation}

默认权重为$\lambda_{bbox} = 5.0$、$\lambda_{giou} = 2.0$、$\lambda_{class} = 2.0$、$\lambda_{bce} = 2.0$、$\lambda_{dice} = 5.0$、$\lambda_{heat} = 0.1$、$\lambda_{rot} = 0.1$和$\lambda_{contrast} = 0.1$。
