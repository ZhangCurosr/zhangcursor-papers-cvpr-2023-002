---
title: "Robust-Outlier-Rejection-for-3D-Registration-with-Variationa"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_Robust_Outlier_Rejection_for_3D_Registration_With_Variational_Bayes_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:04:16"
field: "3D点云配准与鲁棒特征学习"
keywords: ["3D点云配准", "异常值剔除", "变分贝叶斯", "非局部网络", "鲁棒估计"]
innovations: ["将变分贝叶斯推理引入非局部网络，通过先验/后验分布对齐学习判别性对应关系特征", "设计Wilson score投票内点搜索机制，理论上证明高异常值率下优于RANSAC"]
benchmarks: ["3DMatch", "3DLoMatch", "KITTI"]
---

# 论文速读：Robust-Outlier-Rejection-for-3D-Registration-with-Variational-Bayes

## 一句话总结
本文提出了一种基于**变分贝叶斯推理**的非局部网络（VBNonlocal），通过在查询/键/值分量中注入随机特征并对齐先验与标签相关的后验分布，学习更具判别性的对应关系特征；再结合Wilson score投票机制进行内点搜索，显著提升了高异常值率场景下的3D点云配准鲁棒性。

## 研究问题与动机
- **高异常值率下的特征模糊问题**：现有最优方法PointDSC依赖空间一致性非局部网络，但在低重叠、重复结构等挑战性场景中，几乎所有30%（FCGF）和17%（Predator）的inlier-outlier对仍拥有正的兼容性值，误导注意力权重，导致长程依赖建模失效。
- **缺乏不确定性建模**：对称或重复场景中inlier/outlier预测存在显著不确定性，确定性特征无法表征这种模糊性。
- **RANSAC效率瓶颈**：传统采样验证方法在高异常值率下时间开销巨大，且在小样本稀疏点云下易丢失所有inlier种子。
- **端到端方法的局限**：尽管端到端配准方法进步显著，但在(descriptor-based)大规模场景配准中，仍需依赖有效的异常值剔除模块来保证鲁棒性。

## 核心贡献（创新点）
- **提出变分非局部网络（VBNonlocal）**：通过将变分贝叶斯推理引入非局部注意力，用随机特征捕获长程依赖的模糊性，先验分布经训练后逼近判别性后验分布，使测试时采样的特征具有更高判别力——与PointDSC确定性非局部网络的本质区别在于显式建模了特征不确定性。
- **设计后验编码器与ELBO优化目标**：构建了概率图模型，推导变分下界（ELBO）作为训练目标，使标签相关的后验分布通过KL散度正则化约束先验——本质区别于仅添加分类损失的特征引导方式（实验证明后者效果有限甚至退化）。
- **提出Wilson score投票内点搜索机制**：将多轮非局部迭代特征作为多个"投票者"，通过粗粒度投票聚类+细粒度Wilson score融合，确定性搜索高质量假设内点集；理论上证明在高异常值率下比RANSAC更可能获得纯内点子集。
- **保守种子选择策略（CS）**：针对稀疏点云设置种子数量下限，避免激进筛选导致无inlier种子被选入，在低点数场景（250点）提升显著（+3.4%）。

## 方法详解
**变分非局部网络核心流程：**
- 给定假设对应关系集 $\mathcal{C}$，初始特征 $\tilde{\mathbf{F}}^0$ 通过线性投影获得。
- 每轮迭代 $l$ 使用GRU更新隐藏查询/键/值特征，汇总历史信息：
  $$\mathbf{h}_{q,i}^l = \text{GRU}_q(\mathbf{h}_{q,i}^{l-1}, [\mathbf{z}_{q,i}^{l-1}, \tilde{\mathbf{F}}_i^{l-1}])$$
- **前向编码器** $p_\theta(\cdot)$ 预测随机特征的高斯先验分布 $\mu, \sigma$，并采样 $\mathbf{z}^l$ 与隐藏特征拼接后经 $f_\theta^{q,k,v}$ 生成 $\tilde{\mathbf{Q}}^l, \tilde{\mathbf{K}}^l, \tilde{\mathbf{V}}^l$。
- **后验编码器** $q_\phi(\cdot)$ 为标签相关编码器，输入隐藏特征与扩展标签向量 $[b_i]_{\times k}$，输出条件高斯分布，用于采样后验随机特征。
- 经L轮迭代后，最终特征 $\tilde{\mathbf{F}}^L$ 输入标签预测网络得到内/外点概率。

**ELBO优化目标：**
$$\text{ELBO}(\theta,\phi) = \mathbb{E}_{q_\phi}[\ln y_\theta(\mathbf{b}|\tilde{\mathbf{F}}^L)] - \sum_{l=0}^{L-1}\mathbb{E}_{q_\phi}[\text{D}_{KL}(q_\phi(z^l|\mathbf{h}^l,\mathbf{b}) || p_\theta(z^l|\mathbf{h}^l))]$$
最大化ELBO等价于：提高标签预测准确率 + 最小化先验与后验分布的KL散度，迫使先验学会判别性特征。

**投票内点搜索：**
- 基于置信度与NMS选择种子集 $\mathcal{C}_{seed}$（种子比例 $v=0.1$，下限 $n=1000$）。
- 每轮迭代特征 $\tilde{\mathbf{F}}^l$ 作为独立投票者，通过兼容性矩阵 $\mathbf{S}_{i,j}^l = \text{clip}(1-(1-\text{cos}(\tilde{\mathbf{F}}_i^l,\tilde{\mathbf{F}}_j^l))/\sigma^2, 0, 1) \cdot \beta_{i,j}$ 选择 $\kappa-1$ 个最兼容对应关系形成假设内点集。
- Wilson score细融合：$W_{i,j}^n = \frac{1}{1+z^2/n}[\hat{p}_{i,j}^{(n)} + \frac{z^2}{2n} - z\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}]$，取各轮最大得分作为最终投票结果。
- 理论分析（Theorem 1）：当异常值服从Poisson分布时，若 $\alpha < \mathcal{U}$，则本文方法获得纯内点子集的概率不低于RANSAC；且内比 $p_{in}$ 越低，优势越明显。

**变换估计：** 对每个假设内点集使用Procrustes方法（加权最小二乘），选取使重叠对应数最大化的变换，最后用所有恢复内点做最小二乘精修。

## 实验与结果
**数据集与设置：**
- **3DMatch**（室内RGB-D）：46训练/8验证/8测试场景，5cm体素下采样，对比FCGF/FPFH描述子。
- **3DLoMatch**（低重叠，10%-30%）：对比FCGF/Predator描述子，不同点数设置（5000→250）。
- **KITTI**（室外LiDAR）：30cm下采样，序列0-5/6-7/8-10分训练/验证/测试。

**主要结果：**
| 数据集 | 描述子 | VBReg RR | PointDSC RR | 提升 |
|--------|--------|---------|-------------|------|
| 3DMatch | FCGF | **93.53%** | 92.42% | +1.11% |
| 3DMatch | FPFH | **82.75%** | 77.51% | +5.24% |
| 3DLoMatch (500点) | FCGF | **47.2%** | 37.7% | **+9.5%** |
| 3DLoMatch (250点) | Predator | **63.0%** | 60.5% | +2.5% |
| KITTI | FCGF | **98.02%** | 97.66% | +0.36% |
| KITTI | FPFH | **98.92%** | 98.20% | +0.72% |

- 3DLoMatch上在多数设置下均取得SOTA，挑战场景（250点FCGF）提升达9.5%。
- KITTI上FPFH设置各项指标均为最优；速度方面VBReg（0.22s）略慢于PointDSC（0.11s），但可接受。
- 消融实验证明：VBNonlocal vs SCNonlocal在3DMatch FPFH上提升2.77%，3DLoMatch低点数场景提升5-6%；投票机制与保守种子选择均有稳定增益。

## 相关工作脉络
- **PointDSC [2]**（核心基线）：空间一致性驱动非局部网络+神经谱匹配，本文替换其非局部模块为变分版本，并提出改进投票策略。
- **Predator [21] / FCGF [11]**：描述子方法，本文为descriptor-agnostic框架，兼容多种描述子。
- **RANSAC [16]**：传统随机采样验证，本文在理论上证明投票策略在高异常值率下优于RANSAC。
- **SC2 PCR [9]**：二阶空间兼容性方法，在3DMatch FPFH设置下略优于本文（83.92% vs 82.75%），但在挑战性3DLoMatch场景中被超越。
- **Deep Global Registration (DGR) [10]**：端到端配准方法，本文聚焦descriptor-based pipeline中的异常值剔除，可与DGR类方法互补。
- **RPM-Net [47] / RegTR [48]**：基于Sinkhorn/Transformer的端到端匹配方法，与本文非局部特征学习思路不同但可在高层Pipeline中结合。

## 局限性与未来方向
- **计算效率**：变分非局部网络需进行随机特征采样与ELBO优化，比PointDSC慢约2倍（0.22s vs 0.11s），在实时SLAM应用中可能受限。
- **仅针对配对配准**：目前方法局限于pairwise registration，未扩展到多帧/全局配准场景。
- **对描述子质量的依赖**：仍采用两阶段pipeline（先提取描述子建立假设对应关系），极端低重叠（<10%）下描述子本身质量可能不足。
- **随机特征维度敏感**：消融显示 $\tilde{d}=32$ 或 $64$ 时性能略有下降，最优 $\tilde{d}=128$ 增加参数量。
- **未来方向**：端到端学习描述子与异常值剔除的联合优化、引入Transformer架构处理超大尺度场景、扩展至多帧SLAM流水线。

## 研究启发与可借鉴点
- **变分贝叶斯+非局部注意力的结合范式**：将随机特征注入Q/K/V并通过先验/后验对齐学习判别性表征的思路，可迁移至其他3D视觉任务（如特征匹配、姿态估计、分割）。
- **标签相关后验编码器的设计**：通过扩展标签向量 $[b_i]_{\times k}$  conditioning后验分布，实现监督信号对特征不确定性的直接引导，该机制可复用至其他异常值过滤任务。
- **Wilson score投票融合策略**：将多轮迭代的中间结果作为独立"投票者"并基于统计置信度融合，替代RANSAC随机采样，为确定性鲁棒估计提供新思路。
- **保守种子选择策略**：在稀疏场景下通过设置下限保护小样本，这一思想可用于任何基于采样的鲁棒估计方法。
- **ELBO作为非局部网络训练目标**：将变分下界引入特征学习而非仅用于生成模型，展示了概率推断与判别学习的结合潜力。

## 关键术语表
- **Outlier Rejection（异常值剔除）**：从假设对应关系中识别并移除错误匹配（外点），保留正确匹配（内点）的过程。
- **Variational Bayes（变分贝叶斯）**：一种近似后验分布的概率推断方法，通过优化变分下界（ELBO）避免直接计算难解的后验。
- **Non-local Network（非局部网络）**：通过全局注意力机制聚合长程上下文信息来增强特征判别力的网络结构。
- **Prior/Posterior Feature Distribution（先验/后验特征分布）**：先验为测试时采样特征的条件分布，后验为训练时加入标签信息的条件分布；两者通过KL散度对齐。
- **ELBO（Evidence Lower Bound）**：对数似然的变分下界，作为可微优化目标替代难以计算的边缘似然。
- **Hypothetical Inlier（假设内点）**：基于种子点和兼容性关系推断出的候选内点子集。
- **Wilson Score（威尔逊分数）**：基于二项分布置信区间的统计量，用于评估投票结果的可靠性并融合多轮投票。
- **Geometric Compatibility（几何兼容性）**：基于对应关系间距离一致性的矩阵，衡量一对对应关系是否满足刚体变换约束。

## 可复现要素
- **数据集**：3DMatch（公开）、3DLoMatch（公开）、KITTI（公开）。
- **代码**：已开源，https://github.com/Jiang-HB/VBReg。
- **关键超参**：迭代次数 L=12；特征维度 d=128，随机特征维度 $\tilde{d}=128$，隐藏维度 d'=256；假设内点大小 κ=40；种子比例 v=0.1；种子下限 n=1000；学习率 1e-4；权重衰减 1e-6；训练50个epoch；Adam优化器。
