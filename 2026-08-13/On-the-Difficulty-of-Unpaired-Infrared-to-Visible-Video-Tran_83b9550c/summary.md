---
title: "On-the-Difficulty-of-Unpaired-Infrared-to-Visible-Video-Tran"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_On_the_Difficulty_of_Unpaired_Infrared-to-Visible_Video_Translation_Fine-Grained_Content-Rich_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:59:48"
field: "红外/可见光跨模态视频翻译"
keywords: ["红外到可见光视频翻译", "非配对图像翻译", "长尾分布优化", "内容感知优化", "时序一致性", "生成对抗网络"]
innovations: ["提出CPTrans框架解决长尾分布导致的梯度偏见问题", "设计内容感知优化模块通过PatchGAN梯度权重平衡富信息patch训练", "设计内容感知时间归一化模块构建与真实运动相关的光流实现像素级时序一致性"]
benchmarks: ["InfraredCity-Lite", "IRVI", "InfraredCity-Adverse"]
---

# 论文速读：On-the-Difficulty-of-Unpaired-Infrared-to-Visible-Video-Translation

## 一句话总结
本文提出 CPTrans 框架，通过分析非配对红外→可见光视频翻译中的长尾像素分布问题，设计内容感知优化（CO）与内容感知时间归一化（CTN）两个模块，平衡不同 patch 的梯度，从而显著提升内容丰富的 patch（如车辆、路牌）的细节生成质量。

## 研究问题与动机
1. **长尾分布导致优化偏见**：真实数据集（如 IRVI）中天空、道路等"贫信息 patch"占据大量像素，而车辆、路牌等"富信息 patch"属于少数类别，导致 GAN 训练中判别器的梯度被多数类别主导。
2. **现有方法忽视细粒度细节**：ROMA 等 SOTA 方法虽能保持结构一致性，但在复杂场景（多车辆、雨雪天气）下生成的视觉细节仍然混乱或语义错误。
3. **视频任务的时间一致性挑战**：红外视频中运动物体（如车辆）与静态背景（如天空）的运动特性不同，现有光流估计辅助方法在红外场景下精度不足。
4. **缺乏恶劣天气评测数据**：已有数据集多集中在夜间晴天场景，缺少雨雪等极端天气下的评测基准。

## 核心贡献（创新点）
1. **提出 CPTrans 框架**：通过梯度分析揭示长尾效应导致的优化偏见问题，并设计 CO 和 CTN 模块协同提升内容丰富 patch 的生成质量。
2. **内容感知优化模块（CO）**：利用 PatchGAN 梯度方向与全局梯度的余弦相似度定位富信息 patch，并通过自适应权重放大其梯度贡献，打破多数类别主导的优化偏见。
3. **内容感知时间归一化模块（CTN）**：基于权重图构建与真实运动相关的光流，使生成器对富信息 patch 的运动变化保持鲁棒，实现像素级时序一致性。
4. **扩展 InfraredCity-Adverse 数据集**：在原有 InfraredCity-Lite 基础上增加雨天和雪天场景，为红外相关研究提供更具挑战性的评测基准。

## 方法详解
**整体框架**：基于 Generator G 和 Discriminator D 的 GAN 架构，采用 PatchGAN 判别器（输出 w×h 个 patch 分类分数）， backbone 为 ResNet-9blocks。

**内容感知优化（CO）**：
- 计算每个 patch 梯度与全局平均梯度的余弦相似度：$\delta_i = \cos(\nabla_{\theta_D} \log p_i, \nabla_{\theta_D} \frac{1}{N}\sum_j \log p_j)$
- 选取偏差最大的 $\eta_{ratio}$（本文设为 40%）patch 作为富信息区域 $\mathcal{U}_r$
- 设计权重图：$w_i = \frac{\lambda_{inc}}{\exp(|\delta_i|)}$（富信息 patch），其余为 1.0
- 修改对抗损失：$\mathcal{L}_{co-adv}^{patch} = \mathbb{E}[\frac{1}{N}\sum w_i \log p_i] + \mathbb{E}[\frac{1}{N}\sum \tilde{w}_j \log(1-\tilde{p}_j)]$

**内容感知时间归一化（CTN）**：
- 利用 $\tilde{w}$ 构建内容感知光流：$F_{content} = Smooth(Normalize(\tilde{w}) \cdot \gamma_{stride} \cdot z)$，其中 $z$ 为标准高斯噪声
- 对 $\tilde{w}$ 归一化以减少贫信息 patch 的偏移影响
- 时序一致性损失：$\mathcal{L}_{ctn} = \mathbb{E}_x \|W(G(x), F_{content}) - G(W(x, F_{content}))\|_2$

**总损失函数**：
$$\min_{\theta_G} \max_{\theta_D} \mathcal{L} = \mathcal{L}_{co-adv}^{patch} + \lambda_1 \cdot \mathcal{L}_{cs} + \lambda_2 \cdot \mathcal{L}_{ctn}$$
其中 $\lambda_1 = 6.0$，$\lambda_2 = 10.0$，训练 100 epochs，学习率 $2\times10^{-6}$，batch size 为 1。

## 实验与结果
**数据集**：
- IRVI：约 20K 帧，白天交通/监控场景
- InfraredCity-Lite：约 40K 帧，夜间红外→白天可见光
- InfraredCity-Adverse：新增雨雪恶劣天气场景（含红外噪声如雨滴）

**评估指标**：FID、KID（degree=21）、YOLO score（车辆检测 AP）

**主要结果**（Table 1-2）：
- **InfraredCity-Lite**：CPTrans 在 Traffic City clear 场景取得 FID=0.3728（ROMA 为 0.4018），Monitoring all 场景 KID=0.4570（ROMA 为 0.7058）
- **IRVI**：Traffic FID=0.2936（ROMA 为 0.3467），Monitoring KID=2.3760（ROMA 为 3.3972）
- **InfraredCity-Adverse**：Rain 场景 FID=0.4760（ROMA 为 0.5577），Snow 场景 KID=2.6382（ROMA 为 4.9271），提升显著
- **YOLO score**：CPTrans 达到 58.1 AP，超越 ROMA 的 50.1

**训练效率**：CPTrans 训练速度约为 ROMA 的 5 倍，在各 epoch 节点均取得更优 KID 分数。

## 相关工作脉络
1. **CycleGAN [19]**：开创无监督图像翻译，引入 cycle consistency 保持内容，但未针对视频时序和红外场景优化。
2. **CUT [31] / F/LSeSim [51]**：单侧框架结合对比/自相似损失维持结构，但特征中仍混合风格信息，影响结构一致性。
3. **I2V-GAN [25]**：首个专用于红外→可见光视频翻译的方法，引入时序一致性损失，但未解决细粒度细节生成问题。
4. **ROMA [49]**：当前 SOTA，提出 cross-similarity 保持结构，但实验中发现在复杂场景下富信息 patch 细节混乱，本文正是针对此缺陷改进。
5. **Recycle-GAN [3] / Mocycle-GAN [7]**：基于光流辅助的时序一致性能，但辅助系统在红外场景下运动估计不准确。

## 局限性与未来方向
1. **CO 模块依赖梯度分析**：使用 GradCAM++ 思想定位富信息 patch，但梯度计算可能引入数值稳定性问题。
2. **CTN 的光流生成方式**：采用加权噪声构造伪光流，虽比随机 warping 更接近真实运动，但非精确光流估计。
3. **数据集扩展有限**：InfraredCity-Adverse 仅覆盖雨/雪，未涉及雾、沙尘等其他恶劣天气。
4. **未见与其他先进 Transformer 基视频翻译方法的对比**。

## 研究启发与可借鉴点
1. **梯度偏见分析思路**：通过 PatchGAN 梯度方向与全局梯度的余弦相似度识别优化偏移，这一分析范式可迁移至其他长尾分布的生成任务。
2. **权重图辅助光流构造**：将判别器权重图用于指导时序约束，使 motion prior 与内容分布对齐，这一设计避免了额外光流网络的训练成本。
3. **单侧梯度增强无需额外网络**：CO 模块仅修改损失函数的权重项，不增加模型复杂度，在提升性能的同时加速训练，适合资源受限场景。
4. **应用验证闭环**：从翻译质量（FID/KID）到下游任务（YOLO 检测、语义分割、视频融合）的多级验证，强化了方法实用性的论证力度。

## 关键术语表
**Unpaired Infrared-to-Visible Translation**：无需配对数据的红外图像到可见光图像的翻译任务，旨在将稳定的红外视频转换为具丰富视觉信息的可见光视频。

**Content-rich Patches**：包含丰富视觉细节的图像区域（如车辆、路牌），在数据集像素分布中属于少数类别，易被长尾效应忽视。

**Cross-similarity**：ROMA 提出的结构一致性损失，通过比较输入与输出在 ViT token 空间中的相似度矩阵，保持跨域结构不变性。

**GradCAM++**：基于分类分数梯度的可视化方法，本文用于分析判别器在不同 patch 上的优化关注点。

**PatchGAN Discriminator**：输出 w×h 个 patch 二分类分数的判别器，用于 GAN 的风格优化，是本文梯度分析的基础。

**Content-Aware Temporal Normalization (CTN)**：利用权重图构建与真实运动相关的光流，实现像素级时序一致性的模块。

## 可复现要素
- **数据集**：InfraredCity-Lite 和 IRVI 已公开；InfraredCity-Adverse 由作者扩展并提供（论文注明扩展数据集细节在 Supplementary Material）
- **代码**：作者声明已开源（"released codes"）
- **关键超参**：$\eta_{ratio} = 40\%$，$\lambda_1 = 6.0$，$\lambda_2 = 10.0$，训练 100 epochs，学习率 $2\times10^{-6}$，batch size = 1，Generator 使用 ResNet-9blocks
