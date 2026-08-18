---
title: "PlenVDB-Memory-Efficient-VDB-Based-Radiance-Fields-for-Fast"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_PlenVDB_Memory_Efficient_VDB-Based_Radiance_Fields_for_Fast_Training_and_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:01:42"
field: "神经辐射场加速与渲染"
keywords: ["NeRF", "VDB", "神经辐射场", "稀疏体积", "实时渲染", "移动端渲染", "光线 march"]
innovations: ["首次将VDB稀疏体积数据结构用于NeRF训练与渲染，统一训练渲染数据结构", "双层光线 march 算法利用密度阈值过滤无效采样点，渲染提速约5倍", "VDB合并与float16压缩使模型体积较DVGO缩小约10倍"]
benchmarks: ["NeRF-Synthetic"]
---

# 论文速读：PlenVDB-Memory-Efficient-VDB-Based-Radiance-Fields-for-Fast

## 一句话总结
论文提出了PlenVDB，一种基于工业级VDB稀疏体积数据结构的新表征，直接优化VDB而非先训练NeRF再转换，实现了快速训练（~20分钟）和小存储；同时设计了双层光线 march 算法支持移动端实时渲染（iPhone 12 上 30+ FPS）。

## 研究问题与动机
- NeRF训练和渲染计算开销巨大，难以支撑实时应用和移动端部署。
- 利用密集3D网格的方法（如DVGO）可实现快速训练但存储开销大，不适用于移动设备。
- 利用3D稀疏性的方法（如Plenoctrees、EfficientNeRF）虽能实现实时渲染和小存储，但通常需要先训练Vanilla NeRF或密集网格再转换，训练时间更长；且Plenoctrees不支持三线性插值。
- 移动端实时高质量渲染仍是未解决的挑战。

## 核心贡献（创新点）
- **首次将VDB引入NeRF加速**：提出PlenVDB利用VDB的四层B+树结构，在训练中直接优化稀疏体积，统一了训练和渲染阶段的数据结构，避免了转换步骤。
- **双层光线 march 渲染算法**：第一次遍历只查询DensityVDB并通过alpha阈值过滤，仅对有效点查询ColorVDB，减少约90%的采样点计算量，渲染速度较DVGO提升约5倍。
- **VDB合并与压缩策略**：将DensityVDB和ColorVDB合并为单一VDB消除拓扑冗余，并采用float16压缩，相比DVGO模型体积缩小约10倍（如Chair场景从206MB降至23MB）。
- **移动端实时渲染支持**：通过SH投影后处理去除MLP，将模型导出为NanoVDB Float16格式，在Metal/OpenCL上实现实时渲染，iPhone 12上达到30+ FPS（1280×720）。

## 方法详解
- **训练策略**：采用coarse-to-fine两阶段训练（仿照DVGO），coarse阶段用较大包围盒，fine阶段用从训练集提取的更紧包围盒。初始化时所有VDB为dense，随着训练逐渐稀疏化。
- **前向传播**：给定采样坐标，通过三线性插值从ColorVDB和DensityVDB获取颜色特征$\mathbf{c}$和密度$\sigma$，代入体积渲染公式计算像素颜色并求损失。
- **反向传播**：梯度存入GradVDB，传入OptVDB（存储Adam优化器参数），再更新ColorVDB和DensityVDB。
- **双层光线 march**：
  - 第一次：从$t_{min}$沿射线以步长$\Delta t$采样，查询DensityVDB，低于阈值$\tau$或超出包围盒的点被过滤，记录有效点数$N_{valid}$和$t_{first}$。
  - 第二次：仅对有效点查询ColorVDB获取颜色特征，在CUDA中实现轻量MLP映射为RGB。
- **VDB合并**：统计所有active voxel数量$n_{Voxels}$，创建$(n_{Voxels}+1) \times (1+3n)$缓冲区，将DensityVDB的值转为索引，ColorVDB的值拷贝进去，合并后只需查询一次VDB。
- **压缩**：float32存为float16，读取时转回float32。

## 实验与结果
- **数据集**：NeRF-Synthetic（8个场景，每场景100张训练图、200张测试图，800×800分辨率）。
- **训练速度**：PlenVDB训练时间约10–15分钟，与DVGO（3–6分钟）和Plenoxels（10–18分钟）相当。
- **渲染速度**：相比DVGO平均提升约5倍（Chair 30 FPS vs DVGO 5 FPS）；相比Plenoxels/Plenoctrees较慢但画质更高。
- **模型大小**：相比DVGO缩小约10倍（如Ship场景：205MB → 29MB）；相比Plenoxels缩小约20–40倍。
- **重建质量**：PSNR与DVGO几乎持平（如Chair 34.07 vs 34.06），明显优于Plenoxels和Plenoctrees。
- **移动端**：iPhone 12上实现30+ FPS（1280×720），人像类场景可达50–60 FPS。

## 相关工作脉络
- **DVGO [30]**：直接在密集网格上优化NeRF，收敛快但存储大、渲染仍需MLP；PlenVDB替换密集网格为稀疏VDB，大幅压缩存储。
- **Plenoxels [5]**：用指针式密集索引阵列存储几何/颜色，渲染快但存储大（Ship达1331MB）；不支持三线性插值。
- **Plenoctrees [33]**：用球谐系数拟合稀疏八叉树，最快但仅支持最近邻插值、建模高频细节能力弱；PlenVDB支持三线性插值、画质更优。
- **Instant-NGP [18]**：多分辨率哈希表加速训练至分钟级，但内存开销大；PlenVDB存储更紧凑。
- **MobileNeRF [4]**：将NeRF表示为纹理多边形适配光栅化流水线，训练慢且无法处理透明物体；PlenVDB原生支持体积渲染和透明建模。
- **VDB/NanoVDB [19,20]**：工业级稀疏体积数据结构；本文首次将其应用于NeRF训练和渲染两阶段。

## 局限性与未来方向
- 仅适用于静态场景，动态场景建模留待未来探索。
- 三线性插值精度略低于MLP直接查询，合并时裁剪inactive voxel可能导致PSNR轻微下降。
- 移动端性能在渲染大面积区域（如face-forward或无界场景）时会下降。
- 论文未公开代码，复现需依赖自行实现。

## 研究启发与可借鉴点
- **数据结构迁移思路**：VDB/B+树等工业级稀疏体积结构可迁移至其他神经场表示（如SDF、隐式表面），兼顾访问速度与存储效率。
- **双层采样策略**：先通过稀疏性判断（密度阈值）粗筛再精细采样的思想，可推广至其他ray marching-based的神经渲染方法以节省计算。
- **训练-渲染统一数据结构**：避免"先训练dense再转换sparse"的pipeline，直接端到端优化目标数据结构，简化流程并减少信息损失。
- **移动端适配经验**：SH投影去除MLP + float16压缩 + NanoVDB导出，为神经渲染移动端部署提供了可复用的技术栈。
- **可结合本团队方向**：若团队关注低资源推理或移动端3D内容生成，PlenVDB的紧凑存储与实时渲染方案可直接参考。

## 关键术语表
- **VDB（Volume Data Base）**：四层B+树结构的稀疏体积数据结构，支持快速随机访问和缓存友好的顺序访问。
- **NeRF（Neural Radiance Fields）**：用MLP表示场景的辐射场，输入位置和方向输出密度和颜色。
- **DensityVDB / ColorVDB / GradVDB / OptVDB**：分别存储密度值、颜色特征、梯度和Adam优化器状态的VDB实例。
- **NanoVDB**：GPU友好的VDB格式，支持 portable header，便于在shader中实现快速索引和三线性插值。
- **Ray marching twice**：第一次光线 march 仅查询密度并过滤无效点，第二次仅对有效点查询颜色特征，大幅降低计算量。
- **Trilinear interpolation**：三线性插值，VDB中通过快速邻域查询实现，支持高质量的光滑渲染。
- **Coarse-to-fine training**：两阶段训练策略，先在较大包围盒内训练，再在精细包围盒内优化。
- **SH projection post-processing**：球谐函数投影后处理，将颜色特征转换为SH系数以消除渲染时MLP计算。

## 可复现要素
- **数据集**：NeRF-Synthetic，公开可获取。
- **代码**：论文未提及开源，项目页面为 plenvdb.github.io。
- **权重**：论文未提供开源权重。
- **关键超参**：color feature维度$n=4$（即3n=12），MLP两层各128 channels，batch size=8192 rays，coarse/fine各5k/20k iterations，学习率0.1（voxel grids）/1e-3（MLP），密度阈值coarse=$10^{-7}$、fine=$10^{-4}$，$\Delta t$为voxel size的一半，Adam($\beta_1=0.9, \beta_2=0.99, \epsilon=10^{-8}$)。
