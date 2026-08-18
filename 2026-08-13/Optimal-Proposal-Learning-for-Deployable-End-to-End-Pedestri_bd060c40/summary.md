---
title: "Optimal-Proposal-Learning-for-Deployable-End-to-End-Pedestri"
source: https://openaccess.thecvf.com//content/CVPR2023/papers/Song_Optimal_Proposal_Learning_for_Deployable_End-to-End_Pedestrian_Detection_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:00:27"
field: "行人检测与密集场景理解"
keywords: ["end-to-end pedestrian detection", "NMS-free detection", "one-stage detector", "coarse-to-fine learning", "completed proposal network", "crowded scene detection"]
innovations: ["C2F渐进分类边界精化策略：通过递减One-to-Mi标签分配逐步消除one-to-one分类歧义", "CPN多流互补网络：融合分类与回归特征并引入MSFE局部增强与Neg(f_reg)反转，补偿小尺度/遮挡难样本", "轻量可部署的去NMS端到端行人检测框架：在FCOS基础上实现工业级性价比"]
benchmarks: ["CrowdHuman", "TJU-Ped-campus", "TJU-Ped-traffic", "Caltech"]
---

# 论文速读：Optimal-Proposal-Learning-for-Deployable-End-to-End-Pedestrian-Detection

## 一句话总结
本文针对拥挤场景下传统NMS后处理效果差的问题，在轻量级FCOS单阶段检测器上设计了OPL框架，通过C2F渐进分类边界精化策略解决one-to-one样本分配下的歧义性问题，并借助CPN为难样本提供额外响应补偿，实现了高性能且可部署的去NMS端到端行人检测。

## 研究问题与动机
- **NMS在密集人群场景下失效**：crowd scenes（如火车站、机场）中行人高度重叠，单一IoU阈值的NMS要么漏检高重叠正样本，要么引入过多误检。
- **已有去NMS方法部署成本过高**：PED[30]、DETR系方法虽实现end-to-end，但依赖Transformer/queries，训练慢、计算量大，无法在工业资源受限设备上部署。
- **简单one-to-one赋值无法根本消除歧义**：近邻候选框因共享相同人体像素，CNN提取的特征相似，分类分支难以形成紧凑的决策边界，导致仍有高置信度重复框产生。
- **难样本（小尺度/重度遮挡）召回困难**：one-to-one赋值下难样本的正样本更少，且分类分支倾向于关注显著人体部位，易忽略关键部位被遮挡的行人。

## 核心贡献（创新点）
1. **提出OPL框架**：首次将去NMS end-to-end检测建立在轻量CNN-one-stage检测器（FCOS）之上，区别于DETR类query-based方法的笨重设计，实现工业级可部署性。
2. **设计C2F渐进学习策略**：通过逐步收紧每个GT实例的正样本数量（One-to-Mi，Mi递减），让分类分支有机会渐进探索最优决策边界；与DeFCN[61]同时用one-to-many+one-to-one辅助损失的做法不同，C2F通过严格的序列精化引导模型自适应产生无歧义正样本。
3. **提出CPN模块**：融合分类特征（关注显著人体部位）与回归特征（关注完整人体边界）的双重信息流，并引入MSFE局部最大值增强和Neg(f_reg)反向增强，专门补偿难样本的响应；本质区别在于CPN不是额外训练一个辅助分支，而是与主分类分支联合输出最终得分。

## 方法详解
- **整体架构**：以ResNet-50+FPN为骨干，共享检测头包含三个分支：回归分支（保持FCOS原结构）、分类分支（C2F模块替换原最终conv层）、CPN。最终得分为 $S = S_{cls} \cdot S_{cpn}$（Hadamard积）。
- **C2F（Coarse-to-Fine）**：由 $n$ 个级联分类块组成，每块含两个conv层；第 $i$ 步采用 One-to-$M_i$ 标签分配策略（平均每GT实例分配 $M_i$ 个正样本），满足 $M_{i-1} > M_i$，loss为各步focal loss之和：
  $$L_{c2f} = \sum_{i=1}^{n} \frac{1}{N_{pos,i}} \sum_{x,y} L_{cls}(s_{x,y,i}, c_{x,y,i}^*)$$
  仅最后一步的输出参与推理。实验表明2步（$M=\{16,4\}$）效果最佳。
- **CPN（Completed Proposal Network）**：三流结构：
  - $F_1$（残差流）：直接concat $f_{cls}$ 与 $f_{reg}$，确保所有proposal参与优化，避免过拟合。
  - $F_2$（MSFE增强流）：对concat后特征做Multi-scale Feature Enhancement——跨相邻feature level双线性插值对齐分辨率后，经3D max pooling寻找邻近区域最大值，提升漏检难样本的响应。
  - $F_3$（难样本专向补偿流）：对 $f_{reg}$ 取负（Neg操作）使小目标响应更高，再与 $f_{cls}$ concat后经MSFE增强，专门补偿小尺度/遮挡行人。
  最终CPN得分：$S_{cpn} = \sigma(\text{Conv}(\text{ReLU}(\text{GN}(F_1+F_2+F_3))))$。
- **总损失**：$L = L_{reg}(\boldsymbol{B}) + L_{cls}(\boldsymbol{S}) + L_{c2f}$，其中 $L_{cls}$ 为focal loss（one-to-one标签），$L_{reg}$ 为IoU loss。

## 实验与结果
- **数据集**：CrowdHuman（~23人/图）、TJU-Ped-campus（55K图/33万实例）、TJU-Ped-traffic（20K图/4.4万实例）、Caltech（4万+测试图）。
- **评估指标**：mMR（log-average miss rate over FPPI∈[10⁻²,10⁰]，越低越好），另附AP/Recall及按可见度/尺寸划分的子集（R/RS/HO/R+HO/A）。
- **CrowdHuman**：OPL达 **44.9 mMR / 91.0 AP / 97.7 Recall**，较最强NMS-free基线PED[30]（45.6 mMR）**提升0.7 mMR**，且全面超越所有NMS-based方法；DeFCN[61]为48.9 mMR。
- **TJU-Ped-campus**：OPL在所有子集（R=31.5, RS=61.7, HO=72.4）均最优；TJU-Ped-traffic（R=23.4, RS=28.8, HO=62.7）同样最佳。
- **Caltech**：OPL在R=5.2、HO=30.1、R+HO=11.7，领先DeFCN（7.1/34.4/14.3）约2 mMR。
- **Ablation**：C2F单独+2.1 mMR，CPN单独+2.3 mMR，联合+4.4 mMR；CPN三流（F1/F2/F3）分别贡献0.2/1.0/1.1 mMR；ResNet-101仅提升0.2 mMR，说明小骨干已足够，适合部署。

## 相关工作脉络
- **OneNet[55]/DeFCN[61]**：同属去NMS one-stage思路，用one-to-one替代one-to-many；但DeFCN需额外auxiliary loss+3DMF模块，OPL直接用C2F+CPN更简洁且专门针对行人crowd场景优化。
- **PED[30]**：最早针对行人的query-based去NMS方法；OPL定位为工业可部署的轻量替代，摒弃Transformer带来的高计算开销。
- **DETR[8]/Deformable DETR[77]**：通用端到端检测器；OPL定位差异在于面向密集人群+资源受限部署，不依赖dense self-attention。
- **AdaptiveNMS[33]/Repulsion Loss[62]**：改进NMS策略或生成紧凑框的NMS-based方法；OPL从根本上废弃NMS，实现真正端到端。
- **稀疏query类方法（Sparse-RCNN[56]）**：用learnable proposals替代anchor；OPL选择全卷积one-stage路线，更适合工业部署。

## 局限性与未来方向
- **C2F仅验证了2步最优**，更深层数（如3步）反而略有下降，策略的泛化性在更复杂场景下未充分验证。
- **未报告推理速度/FLOPs/参数量等部署指标**，"deployable" claim缺乏量化支撑。
- **仅测试三个行人检测数据集**，未验证在其他密集人群/细粒度检测任务上的迁移能力。
- 论文自述未来工作：将OPL核心思想扩展到其他检测任务或实例级任务。

## 研究启发与可借鉴点
1. **C2F渐进约束思想可迁移**：任何需要one-to-one赋值的一阶段检测器（如FCOS类）均可借鉴渐进收紧正样本数量的策略，解决分类边界模糊问题。
2. **CPN的多流互补设计**：将分类特征（ discriminative）与回归特征（locational）分而治之后融合，并引入MSFE局部增强，对难样本召回问题有普适参考价值。
3. **Neg(f_reg)反转策略**：对回归特征取负来放大小目标响应，是一个巧妙且廉价的难样本增强技巧，可应用于其他尺度敏感任务。
4. **工业可部署性设计范式**：以轻量one-stage检测器为基础，通过模块化增量而非替换主干来实现新能力，是平衡性能与部署成本的有效思路。

## 关键术语表
**OPL（Optimal Proposal Learning）**：本文提出的端到端行人检测框架，在无NMS前提下通过C2F+CPN实现精确提案生成。
**C2F（Coarse-to-Fine Learning）**：渐进式分类学习策略，通过逐阶段减少每GT分配的正样本数（One-to-Mi递减），精化分类决策边界。
**CPN（Completed Proposal Network）**：完成提案网络，融合分类与回归特征，为小尺度/遮挡等难样本提供额外的响应补偿。
**One-to-One Label Assignment**：每个GT实例仅分配一个正样本的标签策略，是去NMS端到端检测的核心训练约束。
**mMR（log-average Miss Rate）**：行人检测标准指标，对FPPI范围内miss rate取对数平均，越低越好。
**MSFE（Multi-scale Feature Enhancement）**：CPN中的模块，通过跨尺度3D max pooling增强局部响应，提升难样本召回。
**FCOS**：Fully Convolutional One-stage Object Detection，本文OPL所依托的基础检测器。
**NMS-free / End-to-End Detection**：去除Non-Maximum Suppression后处理、模型直接输出最终检测结果的检测范式。

## 可复现要素
- **数据集**：CrowdHuman、TJU-Ped、Caltech，均为公开数据集。
- **代码/权重**：论文未明确声明开源（CVPR 2023，常见做法但需进一步确认）；"论文未提及"代码开源状态。
- **骨干网络**：默认ResNet-50（ImageNet预训练），另测试ResNet-101。
- **GPU配置**：CrowdHuman/Caltech用4×Tesla-V100（每卡2张图），TJU-Ped用8×Tesla-V100（每卡4张图）。
- **标签分配策略**：采用DeFCN[61]中的one-to-one/one-to-many方案（论文未提及具体超参细节）。
- **C2F步数**：最优为2步，$M=\{16, 4\}$。
