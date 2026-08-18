---
title: "Open-Vocabulary-Panoptic-Segmentation-with-Text-to-Image-Dif"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Xu_Open-Vocabulary_Panoptic_Segmentation_With_Text-to-Image_Diffusion_Models_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:00:09"
field: "开放词汇视觉分割"
keywords: ["开放词汇分割", "全景分割", "扩散模型", "文本到图像生成", "CLIP", "隐式标题生成", "特征融合"]
innovations: ["首次将冻结文本到图像扩散模型的内部表示用于开放词汇全景分割", "提出隐式标题生成器桥接图像与扩散模型文本空间", "融合扩散模型与判别模型双路特征实现开放词汇推理"]
benchmarks: ["ADE20K", "COCO", "Pascal VOC", "Pascal Context"]
---

# 论文速读：Open-Vocabulary-Panoptic-Segmentation-with-Text-to-Image-Diffusion-Models

## 一句话总结
本文提出 ODISE，首次将大规模文本到图像扩散模型（Stable Diffusion）的冻结内部表示用于开放词汇全景分割任务，通过隐式标题生成器、mask 生成器和基于 CLIP 的开放词汇分类模块，实现了对任意词汇的实例+语义联合分割，在 ADE20K 上较 prior SOTA 提升 8.3 PQ。

## 研究问题与动机
- **现有方法缺乏统一的全景理解**：开放词汇识别大多仅做检测+实例分割或仅做语义分割，缺少能同时解析所有物体实例（things）和场景背景（stuff）的统一框架。
- **判别模型的空间表征不足**：CLIP 等文本-图像判别模型擅长开放词汇分类，但在空间关系理解上存在瓶颈，其内部表征对分割任务的适配性不如扩散模型。
- **扩散模型内部表征潜力未被探索**：文本到图像扩散模型在去噪过程中通过 cross-attention 将文本语义注入视觉特征，其内部表示具有高度的语义可分性和空间定位能力，但尚未被用于开放词汇分割。
- **缺乏开放词汇全景分割基准方法**：现有生成模型驱动分割工作（如 DDPM-Seg）仅面向小词汇表、闭集的语义分割，无法处理开放词汇全景分割。

## 核心贡献（创新点）
- **首次探索大规模文本到图像扩散模型用于开放词汇分割**：利用 Stable Diffusion 冻结 UNet 的内部特征作为分割表征，此前无工作如此系统性地将扩散生成模型的内部表示用于下游识别任务。
- **提出隐式标题生成器（Implicit Captioner）**：通过 CLIP 图像编码器 + MLP 直接生成隐式文本嵌入，避免显式 caption 依赖，解决无配对标注图像的特征提取问题。
- **统一扩散模型与判别模型的 open-vocabulary 推理**：推理时同时利用扩散模型 mask embedding 和 CLIP 图像编码器 masked pooling 特征，通过几何平均融合分类概率，显著提升开放词汇分类精度。
- **在多个基准上建立新的 SOTA**：在 ADE20K 上 PQ 达到 23.4（比 MaskCLIP 高 8.3），语义分割 mIoU 显著超越 OpenSeg、GroupViT 等方法。

## 方法详解
- **整体框架**：冻结的文本到图像扩散 UNet（Stable Diffusion）作为特征提取器，输入图像 x 与其隐式文本嵌入，提取多尺度特征金字塔；随后经过 Mask Generator（基于 Mask2Former）预测 N=100 个 class-agnostic 二值 mask 及其 mask embedding；最后通过 Mask Classification 模块结合 CLIP 文本嵌入完成开放词汇分类。
- **隐式标题生成器**：对无配对 caption 的图像，使用冻结的 CLIP 图像编码器 V(x) 提取图像特征，经一个可学习的 MLP 投影为隐式文本嵌入，替代显式 caption。该 MLP 在训练时仅更新此部分参数。
- **扩散特征提取**：给定图像-文本对 (x, s)，在扩散步 t=0 处添加高斯噪声 x_t，将 x_t 和文本嵌入 T(s) 输入冻结 UNet，提取多层特征（每隔三层 UNet block 取特征，构建 FPN 式特征金字塔）。
- **Mask 生成器**：采用 Mask2Former 架构，接收扩散特征金字塔，输出 N 个二值 mask 和对应的 mask embedding z_i；掩码损失为 pixel-wise binary cross-entropy。
- **Mask 分类（两种监督信号）**：
  - **类别标签监督（Label Supervision）**：将训练集类别名通过冻结文本编码器 T 编码，计算 mask embedding z_i 与各类别文本嵌入的余弦相似度，经 Softmax 后与 GT 类别 y_i 计算交叉熵损失 L_C。
  - **图像标题监督（Caption Supervision）**：从图像 caption 中提取名词作为 grounding 类别，计算所有 mask 与名词的相似度加权聚合为图像-标题对相似度 g(x,s)，再用 InfoNCE 形式的 grounding 损失 L_G 进行对比学习。
- **开放词汇推理融合**：除扩散模型的 mask embedding z_i 外，额外用 CLIP 图像编码器 V(x) 对每个 mask 做 masked pooling 得到 z'_i，分别计算两类特征对测试类别 C_test 的分类概率，最终通过几何平均融合：p_final ∝ p_diffusion^λ · p_discriminative^(1-λ)。

## 实验与结果
- **训练数据**：COCO 全景分割标注（仅 COCO 训练集）。
- **评测数据集**：ADE20K（开放词汇全景/实例/语义分割）、Pascal VOC/Pascal Context（语义分割）。
- **主要结果**：
  - **开放词汇全景分割（ADE20K）**：ODISE（caption 监督）PQ=23.4，mAP=13.9，mIoU=28.7；较 MaskCLIP（PQ=15.1）提升 **+8.3 PQ**、**+8.4 mAP**。
  - **开放词汇语义分割**：在 A-150 上 mIoU=29.9（+6.2 vs MaskCLIP）；在 PC-459 上 mIoU=57.3（label 监督）/55.3（caption 监督），显著超越 OpenSeg、GroupViT 等。
  - **COCO 泛化测试**：在 COCO 测试集上 PQ=55.4，mAP=46.0，mIoU=65.2。
- **消融结论**：
  - 扩散模型内部表征显著优于各类判别模型（DeiT-v3、Swin、ConvNeXt、CLIP 等）和自监督模型（MoCo-v3、DINO、MAE 等）。
  - 隐式标题生成器优于空文本输入和外部 caption 模型（BLIP）。
- **效率**：可训练参数仅 28.1M（占总参数 1.8%），推理速度 1.26 FPS（V100），显存占用 11.9GB。

## 相关工作脉络
- **MaskCLIP**：同期的开放词汇全景分割方法，仅依赖 CLIP 判别模型，本文证明扩散模型内部表征在空间分割上显著优于 CLIP 特征。
- **LSeg / OpenSeg / GroupViT**：基于 CLIP 或 ViT 的开放词汇语义分割方法，本文在此基础上引入扩散模型，统一实现全景分割（实例+语义）。
- **DDPM-Seg**：早期利用扩散模型内部表征做语义分割的工作，但面向小词汇表闭集场景，本文扩展到开放词汇全景分割。
- **传统全景分割（Mask2Former、Panoptic-DeepLab 等）**：依赖闭集标注训练，无法泛化到训练外类别；本文结合开放词汇分类能力实现任意类别分割。
- **CLIP 相关分割方法（DenseCLIP、RegionCLIP 等）**：仅利用 CLIP 的空间粗粒度特征，本文通过扩散模型更细粒度的内部表示和 mask 级特征融合提升分割精度。

## 局限性与未来方向
- **计算开销较大**：需加载 Stable Diffusion 和 CLIP 两个大模型，推理速度（1.26 FPS）仍有提升空间。
- **隐式标题生成器依赖 CLIP 编码器**：若 CLIP 对某些罕见类别的图像表征不佳，隐式标题质量可能受限。
- **N=100 的 mask 数量固定**：对复杂场景可能 mask 不足，对简单场景可能冗余，缺乏自适应机制。
- **消融中 mask 生成器固定为 Mask2Former**：未探索其他 backbone 架构的兼容性。
- **未系统在 open-world instance segmentation 和 open-vocabulary detection 上深入对比**（仅在 supplement 有补充）。

## 研究启发与可借鉴点
- **扩散模型内部表征的分割利用范式**：将冻结扩散 UNet 的多层特征直接用作分割 backbone 特征，为后续研究提供了"以生成模型表征做识别任务"的新思路。
- **隐式标题生成器的设计**：通过轻量 MLP 桥接图像编码器与扩散模型文本输入空间，巧妙解决无 caption 图像的推理问题，可迁移到其他需要文本条件输入的任务。
- **双路特征融合（扩散特征 + 判别特征）**：推理时同时利用生成模型和判别模型的表征，通过几何平均融合，显著提升开放词汇分类鲁棒性，可作为通用策略推广。
- **仅需 COCO 标注即可泛化至任意词汇**：验证了大规模预训练生成模型的零样本/开放泛化能力，为后续低资源场景下的开放词汇分割研究提供了可行路径。

## 关键术语表
- **Open-Vocabulary Panoptic Segmentation**：同时输出图像中所有实例（things）和场景区域（stuff）的分割掩码，且类别集合不限于训练集，可识别任意文本描述的类别。
- **Text-to-Image Diffusion Model**：通过逐步去噪从纯高斯噪声生成图像的生成模型，训练时使用大规模图文对（如 LAION），内部表示蕴含丰富语义信息。
- **Implicit Captioner**：通过冻结图像编码器+可学习 MLP，将图像直接映射为隐式文本嵌入，用于替代显式图像 caption 输入扩散模型。
- **Mask Generator**：基于 Transformer 的分割头（如 Mask2Former），接收多尺度特征金字塔，输出 class-agnostic 的二值 mask 及其 embedding。
- **Grounding Loss**：基于图像-文本配对的对比学习损失，通过 mask 预测与 caption 名词的相似度聚合，实现无需类别标签的开放词汇监督。
- **Panoptic Quality (PQ)**：全景分割的综合评价指标，结合分割质量（SQ）和识别质量（RQ），衡量实例/ Stuff 分割与分类的综合性能。
- **CLIP**：OpenAI 提出的图文对比学习模型，包含图像编码器和文本编码器，广泛用于开放词汇视觉任务的零样本分类和特征提取。

## 可复现要素
- **数据集**：COCO（训练）、ADE20K、Pascal VOC、Pascal Context（评测）；COCC 全景标注公开可用。
- **代码**：已开源，地址 https://github.com/NVlabs/ODISE。
- **权重**：Stable Diffusion（LAION 预训练）、CLIP 权重均使用官方公开版本；ODISE 模型权重见开源仓库。
- **关键超参**：训练 90k 迭代，图像尺寸 1024²，batch size=64，AdamW 优化器，lr=0.0001，weight decay=0.05，mask 数量 N=100，扩散步 t=0，每三层 UNet block 取特征构建特征金字塔。
