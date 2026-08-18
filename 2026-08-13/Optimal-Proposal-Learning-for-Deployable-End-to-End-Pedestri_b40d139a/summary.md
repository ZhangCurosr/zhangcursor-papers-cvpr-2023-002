---
title: "Optimal-Proposal-Learning-for-Deployable-End-to-End-Pedestri"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Optimal_Proposal_Learning_for_Deployable_End-to-End_Pedestrian_Detection_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:00:15"
field: "行人检测与密集场景感知"
keywords: ["end-to-end detection", "pedestrian detection", "NMS-free", "one-to-one assignment", "crowded scenes"]
innovations: ["C2F渐进式分类边界精炼策略解决one-to-one模糊性", "CPN多流特征补偿网络增强困难样本召回", "轻量CNN端到端检测框架替代重型query-based方案"]
benchmarks: ["CrowdHuman", "TJU-Ped-campus", "TJU-Ped-traffic", "Caltech"]
---

# 论文速读：Optimal-Proposal-Learning-for-Deployable-End-to-End-Pedestrian-Detection

## 一句话总结
本文提出了 Optimal Proposal Learning (OPL) 框架，通过在轻量级 CNN 检测器（FCOS）上消除 NMS 后处理，实现可部署的端到端行人检测。通过 C2F 渐进式分类边界精炼和 CPN 硬样本补偿，在 CrowdHuman/TJU-Ped/Caltech 三个数据集上均取得 SOTA 性能。

## 研究问题与动机
1. **NMS 在高密度场景失效**：拥挤场景（城市中心、火车站等）中，单一 IoU 阈值的 NMS 无法合理去除重复框，低阈值漏检、高阈值引入误检。
2. **现有端到端方法部署困难**：DETR 类 query-based 方法训练慢、计算成本高，不适合工业资源受限设备部署。
3. **One-to-one 分配存在模糊性**：CNN 对邻近候选框提取相似特征，分类分支难以找到紧致决策边界区分同一 GT 的多个候选。
4. **困难样本表征能力不足**：小尺度/重度遮挡行人难以获得高置信度，且 one-to-one 正样本更少，加剧学习难度。

## 核心贡献（创新点）
1. **提出 OPL 端到端检测框架**：在 FCOS 基础上构建无需 NMS 的轻量检测管线，区别于 DETR 类重型 query-based 方案。
2. **设计 C2F 渐进学习策略**：通过逐步减少每 GT 分配的正样本数（One-to-Mi 递减），引导分类分支自适应探索最优决策边界，解决正样本模糊问题。
3. **提出 CPN 硬样本补偿网络**：融合分类/回归双分支特征，通过 MSFE 局部最大值增强和 Negation 逆映射，为困难样本提供额外响应补偿，提升召回率。
4. **在三个标准数据集上验证有效性**：CrowdHuman mMR 达 44.9（较 PED 提升 0.7%），TJU-Ped/Caltech 全子集 SOTA。

## 方法详解
**整体架构**：基于 FCOS，Backbone（ResNet-50）+ FPN 提取多尺度特征，共享 Detection Head 包含三个分支：Regression、Classification、CPN。最终得分 $S = S_{cls} \cdot S_{cpn}$（Hadamard 积）。

**Loss 函数**：
$$L = L_{reg}(B) + L_{cls}(S) + L_{c2f}$$
其中 $L_{reg}$ 为 IoU loss，$L_{cls}$ 为 Focal loss（one-to-one 标签分配）。

**C2F（Coarse-to-Fine）**：
- 由 n 个堆叠的分类块组成，每块含 2 个 conv 层
- 第 i 步采用 One-to-$M_i$ 标签分配，满足 $M_{i-1} > M_i$
- 损失：$L_i = \frac{1}{N_{pos,i}} \sum_{x,y} L_{cls}(s_{x,y,i}, c^*_{x,y,i})$
- 总损失 $L_{c2f} = \sum_{i=1}^{n} L_i$
- **关键思想**：粗粒度→细粒度逐步收紧，让模型有探索决策边界的迭代过程

**CPN（Completed Proposal Network）**：
- 输入：分类特征 $f_{cls}$ + 回归特征 $f_{reg}$
- 三路流设计：
  - $F_1 = C(f_{cls}, f_{reg})$（残差连接，防止过拟合）
  - $F_2 = \text{MSFE}(\text{Conv}(C(f_{cls}, f_{reg})))$（局部最大值增强，回忆困难样本）
  - $F_3 = \text{MSFE}(f_1 \cdot f_2)$，其中 $f_2 = \sigma(\text{Conv}(C(\text{Neg}(f_{reg}), f_{cls})))$（Negation 逆转大样本偏好，额外补偿小/遮挡样本）
- 输出：$S_{cpn} = \sigma(\text{Conv}(\text{ReLU}(\text{GN}(F_1+F_2+F_3))))$
- **MSFE 模块**：跨层双线性插值对齐分辨率，3D max pooling 搜索邻域最大值更新响应

## 实验与结果
**数据集**：
- **CrowdHuman**：15K 训练/4.37K 验证，平均 23 人/图
- **TJU-Ped**：Campus（55K/329K 实例）+ Traffic（20K/43K 实例）
- **Caltech**：42.8K 训练/4K 测试

**评估指标**：mMR（log-average miss rate over FPPI ∈ [10⁻², 10⁰]），越低越好；另报告 AP/Recall

**主要结果**：
- **CrowdHuman**：OPL mMR=44.9，较 PED 提升 0.7% mMR / 1.5% AP / 3.7% Recall；较 DeFCN 提升 3.3% mMR
- **TJU-Ped-Campus**：R=31.5, RS=61.7, HO=72.4（全部最优）
- **TJU-Ped-Traffic**：R=23.4, RS=28.8, HO=62.7（全部最优）
- **Caltech**：R=5.2, HO=30.1, R+HO=11.7（全部最优）

**消融**：
- C2F 贡献 2.1% mMR，CPN 贡献 2.3% mMR，联合 4.4% mMR
- C2F-2step {16,4} 最优（44.9），证明"陡峭"转换更有效
- CPN 三流全开最佳，$F_3$ 贡献 1.1% mMR
- ResNet-101 仅提升 0.2% mMR，证明轻量 Backbone 已足够

## 相关工作脉络
1. **DETR 系列**（Carion et al., 2020; 77）：Query-based 端到端检测，计算复杂、收敛慢，不适合工业部署；本文聚焦轻量 CNN 路线。
2. **OneNet/DeFCN**（55, 61）：同样采用 one-to-one 分配，但 DeFCN 需辅助 loss + 3DMF；本文 C2F 以渐进方式自然引导决策边界，无需额外辅助任务。
3. **PED**（Lin et al., 2020）：首个面向行人检测的 DETR 变体，仍受 query-based 瓶颈限制；本文 OPL 在同等指标下显著更轻量。
4. **Adaptive NMS / Repulsion Loss**（33, 62）：改进 NMS 阈值或生成紧凑框，但未消除 NMS；本文彻底去除后处理实现端到端。
5. **Visible/Head Box**（27, 12, 13）：预测可见框或头盒辅助去重，属多监督信号策略；本文通过特征补偿直接学习可靠置信度。
6. **FCOS**（60）：Anchor-free 单阶段检测器基线；本文在其上扩展 C2F+CPN 实现无 NMS 检测。

## 局限性与未来方向
1. **C2F 步数与 M 序列需手动设计**：当前依赖实验调参（最优为 {16,4}），缺乏自适应确定机制。
2. **仅验证行人检测任务**：方法通用性未在其他目标（如车辆、小动物）或实例分割任务上验证。
3. **未对比更重量的 Transformer 基线**：如 DINO、Deformable DETR 等在密集人群场景的端到端性能。
4. **实际部署延迟未详细分析**：虽强调轻量，但缺少与工业部署框架（TensorRT、ONNX）的实测 FPS 对比。
5. **未来方向**：可迁移至其他检测/实例级任务；探索动态 M 调度策略。

## 研究启发与可借鉴点
1. **渐进式标签收紧策略**：C2F 的"从多正样本到少正样本"思路可迁移至其他 one-to-one 分配任务（如 YOLO-X、RTMDet 改进）。
2. **多分支特征补偿机制**：CPN 中 $F_3$ 的 Negation 逆映射设计（将大样本偏好转为小样本偏好）对困难样本挖掘有借鉴价值。
3. **MSFE 跨层局部最大值增强**：类似 "response compensation" 思想可在小目标检测、密集计数等任务中复用。
4. **Lightweight 端到端优先原则**：工业落地场景下，比 DETR 更重视推理效率和部署友好性，值得工程化团队参考。
5. **消融中"陡峭转换"发现**：C2F-2step {16,4} 优于 3step，提示激进的任务转换可能比平滑过渡更高效——可成为后续超参搜索的启发。

## 关键术语表
**One-to-one label assignment**：每个 GT 只分配一个正样本，替代传统 one-to-many，是实现端到端去 NMS 的核心前提。
**Coarse-to-Fine (C2F)**：渐进式学习策略，通过逐步减少正样本数（One-to-Mi 递减）引导模型精细分类边界。
**Completed Proposal Network (CPN)**：辅助分类分支的补偿网络，融合分类/回归特征并为困难样本提供额外响应。
**Multi-scale Feature Enhancement (MSFE)**：CPN 内部模块，跨层拼接后做 3D max pooling 增强局部响应。
**Negation function**：对回归特征取反，使小尺度/遮挡样本获得更高响应，平衡大样本偏好。
**mMR (log-average miss rate)**：行人检测主流指标，在 FPPI [0.01, 1] 区间取对数平均漏检率，越低越好。
**NMS-free**：无需非极大值抑制后处理的端到端检测范式。
**Anchor-free**：不依赖预定义锚框，直接在特征点预测目标的检测架构（如 FCOS）。

## 可复现要素
- **数据集**：CrowdHuman（公开）、TJU-Ped（公开）、Caltech（公开）
- **代码**：论文未提供开源链接（CVPR 2023 常见做法，需联系作者）
- **权重**：未提供预训练权重
- **关键超参**：Backbone ResNet-50（ImageNet 预训练）；CrowdHuman 4×V100，batch=8；TJU-Ped 8×V100，batch=32；C2F 步数 n=2，M={16,4}；Focal loss + IoU loss；未见明确学习率/优化器声明（论文未提及）
- **框架**：PyTorch（推断）
