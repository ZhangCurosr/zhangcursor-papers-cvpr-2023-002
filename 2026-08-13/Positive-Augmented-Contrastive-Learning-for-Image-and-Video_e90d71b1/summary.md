---
title: "Positive-Augmented-Contrastive-Learning-for-Image-and-Video"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Sarto_Positive-Augmented_Contrastive_Learning_for_Image_and_Video_Captioning_Evaluation_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:02:06"
field: "多模态评估度量"
keywords: ["图像描述评估", "对比学习", "CLIP", "视频描述评估", "物体幻觉检测", "正样本增强", "PAC-S"]
innovations: ["提出正样本增强对比学习框架PAC-S，融合清洗数据与生成数据训练跨模态评估空间", "设计粗细粒度联合的视频描述评估策略", "在多种backbone上验证度量通用性，统一图像与视频评估"]
benchmarks: ["Flickr8k-Expert", "Flickr8k-CF", "Composite", "VATEX-EVAL", "Pascal-50S", "FOIL", "ActivityNet-FOIL"]
---

# 论文速读：Positive-Augmented-Contrastive-Learning-for-Image-and-Video

## 一句话总结
本文提出 PAC-S（Positive-Augmented Contrastive learning Score），一种基于正样本增强对比学习的图像/视频描述评估度量，通过在 CLIP 预训练基础上融合清洗数据与合成生成的图像-文本对，显著提升与人类判断的相关性，并在参考-free 和参考-based 设置下均优于现有指标。

## 研究问题与动机
1. **CLIP-Score 等现有对比度量的局限性**：大规模预训练模型基于网页 alt-tag 数据，文本风格质量低，且图像分布与 captioning 评测集不对齐。
2. **仅用清洗数据训练的对比度量效果差**：直接在小规模清洗数据集（如 COCO）上训练对比学习指标，效果甚至不如传统方法，原因是训练数据不足。
3. **缺乏统一的视频描述评估度量**：视频 caption 评估领域仅有 EMScore 一款可学习度量，覆盖面有限。
4. **需要检测物体幻觉的敏感度量**：自动评估需能区分正确描述与包含幻觉对象（hallucinated objects）的错误描述。

## 核心贡献（创新点）
1. **提出 PAC-S 正样本增强对比学习度量**：将真实清洗数据与由图像生成器/文本生成器合成的图像-文本对联合训练，本质区别在于用合成数据弥补清洗数据规模不足的问题，而非单纯扩大训练量。
2. **统一图像与视频描述评估框架**：在图像端引入参考-based 调和平均策略，在视频端设计粗粒度（全局表示匹配）和细粒度（词-帧匹配）两级评分，填补视频 caption 评估方法的空白。
3. **系统验证多种跨模态 backbone 的有效性**：在 CLIP ViT-B/16、ViT-L/14、OpenCLIP ViT-B/32、ViT-L/14 等多种骨干网络上验证 PAC-S 的通用性，证明方法不依赖特定架构。
4. **提供幻觉敏感度评测与分析**：在 FOIL 和 ActivityNet-FOIL 数据集上证明 PAC-S 对物体幻觉具有更强的识别能力，弥补了以往度量在此维度的缺失。

## 方法详解
- **基础架构**：基于 CLIP ViT-B/32 的双编码器（图像编码器 + 文本编码器），固定预训练权重，仅训练最终投影层 toward embedding space。
- **正样本生成**：
  - 以 COCO 数据集（>120k 图像，每图5条 human caption）为源。
  - 对每张真实图像 $v$，用 BLIP 生成合成 caption $t'$。
  - 对每条真实 caption $t$，用 Stable Diffusion（LAION-Aesthetics 微调版，512×512）生成合成图像 $v'$。
  - 构建四元组 $(v, t, v', t')$。
- **损失函数**：对称 InfoNCE 损失，包含三项加权组合：
  - $L_{\mathcal{V},\mathcal{T}}$：真实图像 vs 真实文本。
  - $\lambda_v L_{\mathcal{V}',\mathcal{T}}$：生成图像 vs 真实文本。
  - $\lambda_t L_{\mathcal{V},\mathcal{T}'}$：真实图像 vs 生成文本。
  - 总损失：$L = L_{\mathcal{V},\mathcal{T}} + \lambda_v L_{\mathcal{V}',\mathcal{T}} + \lambda_t L_{\mathcal{V},\mathcal{T}'}$，其中 $\lambda_v=0.05$，$\lambda_t=0.1$（网格搜索选定）。
- **图像评估分数**：
  - **参考-free**：$\text{Score}(t,v) = w \cdot \max(\cos(t,v), 0)$，$w=2^1$。
  - **参考-based (Ref-PAC-S)**：取参考-free 分数与候选文本和所有参考文本最大余弦相似度的调和平均。
- **视频评估分数**：
  - **粗粒度**：全局视频表示与全局文本表示的相似度。
  - **细粒度**：对每帧和每个词提取嵌入，计算 F1-score 结合 TF-IDF 加权。
  - 最终 score 为两者均值；参考版本类似扩展。

## 实验与结果
- **图像数据集**：Flickr8k-Expert（5,664 图像，1-4分专家评分）、Flickr8k-CF（48k 图像，二元 CrowdFlower 评分）、Composite（12k 判断，1-5分）。
- **视频数据集**：VATEX-EVAL（3,000 视频，54k 人类评分，1-5分）。
- **幻觉检测**：FOIL（8k 图像）、ActivityNet-FOIL（1,900 视频-段落对）。
- **配对排序**：Pascal-50S（4k 句子对）、Abstract-50S（400 对 clip-art 图像）。
- **主要结果**：
  - **Flickr8k-Expert**：PAC-S Kendall $\tau_b=53.9$（+2.8 vs CLIP-S 51.1）；RefPAC-S=55.5（+2.9）。
  - **Flickr8k-CF**：PAC-S $\tau_b=36.0$（+1.6）；RefPAC-S=37.6（+1.2）。
  - **Composite**：PAC-S $\tau_b=51.5$（+1.7）；RefPAC-S=53.0（+1.8）。
  - **VATEX-EVAL（无参考）**：PAC-S $\tau_b=25.1$（+1.9）、Spearman $\rho=32.6$（+2.3）；显著优于 EMScore。
  - **Pascal-50S**：PAC-S 均值 accuracy=82.4（+1.5 vs CLIP-S）。
  - **FOIL**：PAC-S accuracy=89.9（+2.7 vs CLIP-S 87.2）；RefPAC-S=93.7（+2.7）。
  - **ActivityNet-FOIL**：PAC-S=90.1（+0.6 vs EMScore 89.5）；RefPAC-S=93.5（+1.1）。
- **最强结果**：在所有图像/视频数据集上均排名第一，相比 CLIP-Score 平均提升约 2~3 个 Kendall/Spearman 点数，且参考-free 版本已超越传统参考-based 指标（如 CIDEr、SPICE）超10分。

## 相关工作脉络
1. **CLIP-Score [17]**：最直接的基线，用 CLIP 余弦相似度做参考-free 评估；本文在相同 backbone 上通过正样本增强进一步提升。
2. **TIGEr [21]**：基于 COCO 训练的跨模态匹配模型，仅利用真实图像-文本对；本文方法扩展至大规模跨域数据并加入生成增强。
3. **ViLBERTScore [26] / UMIC [25]**：基于 BERT/ViL-BERT 的文本或对比学习度量；本文引入生成式正样本作为额外监督信号，与它们正交。
4. **MID [24]**：用负高斯交叉互信息做评估；本文聚焦对比学习空间的 fine-tuning，方法路线不同。
5. **EMScore [49]**：唯一针对视频 caption 的对比度量；本文将其策略扩展到视频并验证正样本增强的有效性。
6. **ImaginE [69]**：利用扩散模型生成图像辅助评估，但仅用于纯文本任务；本文将其思想扩展至视觉-语言联合场景。

## 局限性与未来方向
1. **微调数据依赖 COCO**：在 COCO 子集上微调，可能限制在其他图像分布（如 Abstract-50S 的 clip-art）上的泛化，尽管实验显示仍有提升。
2. **生成模型质量依赖**：合成图像/文本的质量直接影响增强效果；若生成器能力不足，负样本可能引入噪声。
3. **未探索其他跨模态任务**：目前仅评估 captioning，对其他任务（如视觉问答、图文检索）的适用性待验证。
4. **温度参数 $\tau$ 和超参固定**：超参通过网格搜索在多个数据集上取平均，未深入分析其对不同数据分布的敏感性。

## 研究启发与可借鉴点
1. **正样本增强策略的可迁移性**：将真实数据与生成数据联合训练对比空间的想法，可推广至视觉问答、图文检索等跨模态任务的微调。
2. **粗细粒度联合评分设计**：视频端同时利用全局语义和局部词-帧对齐的思路，可借鉴到多尺度图像描述或视频时序建模任务中。
3. **参考-based 调和平均策略**：将参考-free 分数与参考匹配分数的调和平均，平衡了生成质量和参考相关性，该方法简洁且有效，可在其他度量中复用。
4. **幻觉敏感度作为评估维度**：将 FOIL/ActivityNet-FOIL 纳入评估体系，为后续工作提供了可复用的幻觉检测基准。

## 关键术语表
**PAC-S**：Positive-Augmented Contrastive learning Score，本文提出的图像/视频描述评估度量，通过正样本增强对比学习训练。
**InfoNCE Loss**：对称对比损失，最大化正对相似度、最小化负对相似度，是本文训练对比空间的核心损失函数。
**CLIP-Score**：基于 CLIP 模型的余弦相似度参考-free 图像描述评估指标，本文的主要基线。
**FOIL Dataset**：通过替换名词短语构造错误 caption 的数据集，用于评估度量对物体幻觉的敏感度。
**EMScore**：针对视频描述评估的对比度量，利用 CLIP 嵌入计算词-帧级别的细粒度和粗粒度相似度。
**Ref-PAC-S**：PAC-S 的参考-based 版本，通过调和平均融合参考-free 分数与参考文本匹配分数。

## 可复现要素
- **数据集**：COCO（训练用）、Flickr8k-Expert、Flickr8k-CF、Composite、VATEX-EVAL、Pascal-50S、Abstract-50S、FOIL、ActivityNet-FOIL——均为公开数据集。
- **代码/模型**：源码及训练模型已开源，地址：https://github.com/aimagelab/pacscore
- **关键超参**：$\lambda_v=0.05$，$\lambda_t=0.1$，learning rate=0.0001，batch size=256，AdamW 优化器，早停 patience=1,500 次迭代。
- **Backbone**：CLIP ViT-B/32（主实验），另验证 ViT-B/16、ViT-L/14、OpenCLIP ViT-B/32、ViT-L/14。
- **生成器**：Stable Diffusion（LAION-Aesthetics 微调版，512×512）用于图像生成；BLIP ViT-L/14（COCO 微调版）用于文本生成。
