---
title: "NoPe-NeRF-Optimising-Neural-Radiance-Field-with-No-Pose-Prio"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Bian_NoPe-NeRF_Optimising_Neural_Radiance_Field_With_No_Pose_Prior_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:59:41"
---

# 论文速读：NoPe-NeRF-Optimising-Neural-Radiance-Field-with-No-Pose-Prio

## 一句话总结
本文提出 NoPe-NeRF，通过在联合优化过程中显式校正单目深度图的尺度与平移畸变，将其转化为多视图一致的几何先验，并利用相邻帧点云 Chamfer 距离与表面光度一致性损失约束相对位姿，实现了在大尺度相机运动下相机位姿与 NeRF 的端到端联合优化。

## 研究问题与动机
- **大幅运动下的优化失效**：现有无预计算位姿的 NeRF 方法（BARF、NeRFmm、SC-NeRF）仅适用于前向平滑相机轨迹，面对手持视频等剧烈运动时极易陷入局部最优或发散。
- **独立位姿优化的结构缺陷**：前述方法对每帧位姿独立优化，忽略了相邻帧之间的相对位姿连续性，导致轨迹缺乏全局一致性。
- **形状-辐射度歧义叠加**：NeRF 本身存在 shape-radiance ambiguity，联合优化相机参数进一步放大歧义性，造成收敛缓慢且渲染几何不稳定。
- **单目深度的多视图不一致性**：直接引入单目深度图因尺度与平移畸变无法跨帧对齐，传统做法仅将其作为 depth-wise loss 附加，未从根本上解决深度估计与 NeRF 几何空间的标量对齐问题。

## 核心贡献（创新点）
- **单目深度的显式动态标定**：在训练过程中为每张单目深度图独立学习尺度 $\alpha$ 与偏移 $\beta$，通过 NeRF 渲染深度的多视图一致性强制将其校正为度量级深度；与以往静态 depth loss 的本质区别在于深度畸变参数随优化动态收敛，实现先验与场景几何的自适应对齐。
- **帧间点云 Chamfer 相对位姿约束**：将校正后的多视图一致深度反投影为点云，利用相邻帧点云间的 Chamfer Distance 损失显式注入相对位姿先验；与仅依赖单帧光度误差的独立优化相比，首次在无位姿 NeRF 中引入显式时序几何一致性。
- **表面光度一致性正则化**：将校正深度视为真实几何表面，通过相机投影将相邻帧点对应至图像平面计算光度误差；区别于纯 3D-3D 匹配，该损失融合 2D 外观信息，有效缓解点云匹配歧义并提升相对位姿估计鲁棒性。

## 方法详解
- **输入与优化变量**：给定 RGB 图像序列 $\mathcal{I}$、已知相机内参 $\mathbf{K}$、以及离线单目深度网络（DPT）生成的原始深度序列 $\mathcal{D}$。联合优化目标为 NeRF 参数 $\Theta$、相机位姿 $\Pi=\{\mathbf{T}_i\}$ 及深度畸变参数 $\Psi=\{(\alpha_i, \beta_i)\}$。
- **深度畸变校正**：对每帧单目深度施加线性变换 $D_i^* = \alpha_i D_i + \beta_i$，并通过深度渲染损失 $\mathcal{L}_{depth} = \sum_i \|D_i^* - \hat{D}_i\|$ 约束其与 NeRF 体渲染深度 $\hat{D}_i$ 的一致性，从而获得多视图一致的无畸变深度图。
- **点云相对位姿损失 ($\mathcal{L}_{pc}$)**：利用内参将 $D_i^*$ 反投影为点云 $P_i^*$，相邻帧间通过相对位姿 $\mathbf{T}_{ji} = \mathbf{T}_j \mathbf{T}_i^{-1}$ 对齐后计算双向 Chamfer Distance：
  $\mathcal{L}_{pc} = \sum_{(i,j)} \left( \sum_{p_i \in P_i^*} \min_{p_j} \|p_i - p_j\| + \sum_{p_j \in P_j^*} \min_{p_i} \|p_i - p_j\| \right)$。
- **表面光度一致性损失 ($\mathcal{L}_{rgb-s}$)**：将点云 $P_i^*$ 投影至 $I_i$ 与 $I_j$ 的图像平面，计算重投影像素间的光度差异：
  $\mathcal{L}_{rgb-s} = \sum_{(i,j)} \| I_i\langle \mathbf{K}_i P_i^* \rangle - I_j\langle \mathbf{K}_j \mathbf{T}_{ji} P_i^* \rangle \|$。
- **总损失与训练流程**：$\mathcal{L} = \mathcal{L}_{rgb} + \lambda_1 \mathcal{L}_{depth} + \lambda_2 \mathcal{L}_{pc} + \lambda_3 \mathcal{L}_{rgb-s}$（论文取 $\lambda_1=0.04, \lambda_2=1.0, \lambda_3=1.0$）。采用两阶段策略：初期联合优化所有损失直至帧间约束收敛，后期移除 $\mathcal{L}_{depth}$ 与帧间损失，仅保留 $\mathcal{L}_{rgb}$ 精细微调 NeRF，共 10,000 步。神经字段改用 Softplus 激活，射线均匀采样 128 点（范围 0.1~10），每步随机采样 1024 像素。

## 实验与结果
- **数据集与基线**：Tanks and Temples（8 个室内外场景）与 ScanNet（4 个室内场景）；对比基线包括 BARF、NeRFmm、SC-NeRF 及 COLMAP+NeRF。
- **新视角合成（NVS）**：ScanNet 平均 PSNR 31.86 / SSIM 0.83 / LPIPS 0.38，显著优于 SC-NeRF（29.83/0.81/0.41）、NeRFmm（29.97/0.80
