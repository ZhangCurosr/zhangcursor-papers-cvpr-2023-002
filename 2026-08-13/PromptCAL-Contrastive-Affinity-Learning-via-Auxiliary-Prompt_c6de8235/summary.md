---
title: "PromptCAL-Contrastive-Affinity-Learning-via-Auxiliary-Prompt"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_PromptCAL_Contrastive_Affinity_Learning_via_Auxiliary_Prompts_for_Generalized_Novel_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:03:08"
---

# 论文速读：PromptCAL-Contrastive-Affinity-Learning-via-Auxiliary-Prompt

## 一句话总结
本文针对广义新类别发现（GNCD）任务，提出两阶段**PromptCAL**框架：通过判别性Prompt正则化（DPR）激活冻结ViT的语义判别灵活性，并结合基于共识KNN与图扩散的半监督亲和图生成（SemiAG）与对比亲和学习（CAL），在已知类标注有限的条件下有效抑制对比学习中的假负样本，在通用与细粒度基准上均取得SOTA。

## 研究问题与动机
- **核心问题**：GNCD要求仅利用已知类的部分标注数据，对同时包含已知类与新类别的未标注数据进行聚类发现；现有半监督方法受限于封闭世界假设，无法处理开放集样本。
- **现有方法不足**：
  1. **假负样本灾难**：主流GNCD方法（如GCD）依赖标准对比损失，将同语义或相近语义的不同未标注图像误判为负样本，导致聚类紧密度与纯度下降（类别冲突问题）。
  2. **骨干网络僵化**：为防过拟合已知类，现有工作多冻结预训练ViT主干，限制了模型对下游数据集语义分布的自适应能力。
  3. **朴素伪标签不可靠**：直接采用KNN或成对预测生成伪正样本的方法对语义偏移鲁棒性差，在开放集噪声下易引入错误监督信号。
  4. **Prompt训练缺乏语义引导**：直接对ViT末端注入视觉Prompt虽能提升灵活性，但未经语义正则化时易在小数据集上过拟合，难以直接服务于新类别发现。

## 核心贡献（创新点）
1. **两阶段协同学习框架**：提出Warm-up阶段与CAL阶段的交替强化机制，语义Prompt学习与对比亲和学习在迭代中互为支撑。*本质区别在于将单阶段对比自训练扩展为双阶段互相校准的框架，显著提升伪正样本质量。*
2. **判别性Prompt正则化（DPR）损失**：仅在ViT最后一层插入少量[P] token并施加弱监督对比损失，其余Prompt无监督学习。*区别于全量微调或纯冻结策略，以正则化形式注入语义判别信号，兼顾表达能力与防过拟合。*
3. **半监督亲和图生成（SemiAG）**：融合共识KNN统计、图扩散传播与已知类标签先验（SemiPriori）构建二值化亲和图。*区别于朴素KNN，通过多跳路径恢复被假负样本遮蔽的真实正样本，并显式利用已知类约束校准伪标签。*
4. **对比亲和学习（CAL）损失与在线子图机制**：基于EMA教师-学生记忆库动态采样子图，在SemiAG生成的伪正样本集合上计算对比损失。*区别于标准MoCo式固定负样本池，正样本集合随迭代在线更新，更贴合开放集动态分布。*
5. **全面SOTA与低资源鲁棒性**：在6个通用/细粒度基准上超越GCD/ORCA等现有方法，且在少类别、低标注比例设定下仍保持显著优势。*验证了方法在真实低资源开放世界场景中的实用价值。*

## 方法详解
- **基础架构**：采用DINO预训练的ViT-B/16作为骨干，末端添加视觉Prompt适配器。每个mini-batch生成两个增强视图，提取[class] token嵌入 $\mathbf{h}$ 经投影头得到特征 $\mathbf{z}$；末端[P] token输出平均后得 $\mathbf{h}_P$ 与 $\mathbf{z}_P$。
- **阶段一：Warm-up + DPR**：
  - 总损失 $L_1 = L_{semi}^{CLS}(\mathbf{z}) + \gamma L_{semi}^{P}(\mathbf{z}_P)$。
  - $L_{semi}^{CLS}$ 融合自监督对比项（同批次所有增强视图互为正）与监督对比项（仅标注样本按同类标签拉拢），温度参数分别为 $\tau, \tau_a$。
  - DPR损失将末端所有[P] tokenEmbedding平均后投影，施加与主分支相同形式的弱监督对比损失，权重 $\gamma$ 较小，起到正则化作用。
- **阶段二：SemiAG + CAL**：
  - **共识KNN图**：构建全局嵌入图 $G_\mathcal{H}$，边权 $g_{i,j}$ 定义为两节点互为K近邻的全局频次，再行归一化得 $\tilde{G}_c$。
  - **亲和传播**：执行 $\eta$ 步扩散 $\tilde{G}_d^{(t+1)} = \tilde{G}_c \tilde{G}_d^{(t)} \tilde{G}_c^T + I$，捕获高阶结构信息以召回潜在正样本。
  - **SemiPriori校准**：利用已知类标签强制约束，同类边置1、异类边剪枝，并按分位数 $q$ 稀疏化得到二值亲和图 $G_b$。
  - **在线子图采样**：为降低全图计算开销，维护FIFO记忆库 $\mathcal{M}$ 保存EMA教师的稳定嵌入；当前batch的teacher嵌入入队，子图 $G'_\mathcal{H}$ 由 $\mathcal{M} \cup \{h_T\}$ 构成，在其上快速执行SemiAG。
  - **CAL损失**：以student嵌入为查询，子图节点为键，按 $G_b'$ 的1/0边划分正/负集合计算对比损失 $L_{CAL}^{CLS}$；同时保留student-teacher一致性约束的 $L_{sup}^{CLS}$ 与 $L_{self}^{CLS}$。
  - 阶段二总损失：$L_2 = (1-\alpha)L_{sup}^{CLS} + \alpha(\beta L_{CAL}^{CLS} + (1-\beta)L_{self}^{CLS}) + \gamma L_2^P$。推理时用student的[CLS]嵌入做SemiKMeans聚类，匈牙利算法匹配真值。

## 实验与
