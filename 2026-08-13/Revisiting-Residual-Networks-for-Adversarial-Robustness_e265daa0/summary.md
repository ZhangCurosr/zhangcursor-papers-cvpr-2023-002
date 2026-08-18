---
title: "Revisiting-Residual-Networks-for-Adversarial-Robustness"
source: https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Revisiting_Residual_Networks_for_Adversarial_Robustness_CVPR_2023_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 20:03:37"
---

# 论文速读：Revisiting-Residual-Networks-for-Adversarial-Robustness

## 一句话总结
本文系统性地剖析了残差网络架构组件与缩放策略对对抗鲁棒性的影响，提出新型鲁棒残差块（RobustResBlock）与深度-宽度复合缩放规则（RobustScaling），构建出 RobustResNet 系列；在同等 FLOPs 预算下以约 2 倍更少的参数超越主流 WRN 与 NAS 搜索型鲁棒架构，在无外部数据与使用 500K 外部数据下分别达到 61.1% 和 63.7% 的 AutoAttack 鲁棒准确率。

## 研究问题与动机
1. **核心问题**：现有对抗鲁棒性研究高度集中于改进对抗训练（AT）策略（损失设计、数据增强、权重集成等），严重忽视了网络架构（块拓扑、深度、宽度及其分配）对鲁棒性的独立贡献。
2. **现有方法不足**：主流鲁棒模型以 WRN 为基础，块结构局限于 basic block + post-activation；不同前作对平滑激活、SE 模块、网络缩放比例的实验结论相互矛盾，且缺乏跨容量/跨数据集的系统验证。
3. **知识缺口**：尚无经实证支持的深度-宽度联合缩放规则，导致同等计算预算下模型往往在“鲁棒性”与“参数效率”之间难以兼得。
4. **研究动机**：通过 >1200 个模型的系统性消融实验厘清架构设计原则，填补对抗鲁棒领域的架构设计工具空白，并提供一个更紧凑的新 baseline。

## 核心贡献（创新点）
1. **RobustResBlock 块级设计**：提出融合预激活、瓶颈拓扑、聚合/分层卷积与残差 SE 的新型块结构，本质区别在于打破 WRN 默认配置，将多类在标准 ERM 中有效的块级增强统一适配至对抗训练场景。
2. **残差 SE（Residual SE）改造**：发现标准 SE 模块会因过度抑制/放大通道而显著损害对抗鲁棒性；通过在 SE 外增加旁路残差连接，使其在 AT 下稳定增益。
3. **RobustScaling 复合缩放规则**：首次给出面向对抗训练的 depth-width 联合缩放准则，实证得出最优深度贡献率 $r_D \approx 0.7$，并给出阶段非均匀分配比例 $D_1:D_2:D_3=2:2:1$、$W_1:W_2:W_3=2:2.5:1$。
4. **RobustResNet 系列与 SotA 性能**：基于上述设计构建覆盖 5G–40G FLOPs 的模型谱系，参数量较同类 SotA 减少约 2×，同时在各数据集与攻击下实现更高鲁棒准确率。
5. **平滑激活函数的条件性再评估**：揭示平滑激活（如 SiLU）的优势高度依赖 AT 设置与 weight decay/BN 仿射参数的处理方式，修正了“平滑激活 universally 更好”的片面共识。

## 方法详解
- **块级设计（RobustResBlock）**：采用 pre-activation 拓扑（激活置于卷积前）；基础结构为 bottleneck，并分别嵌入 aggregated convolution（cardinality=4）与 hierarchical convolution（scales=8）以增强多尺度特征聚合；将标准 SE 替换为 residual SE（在通道重校准路径外保留恒等旁路，避免对抗扰动下信息瓶颈）；最终激活函数统一使用 ReLU 以保障训练稳定性。
- **网络级缩放（RobustScaling）**：在固定目标 FLOPs 下，将深度与宽度视为竞争资源，定义深度贡献率 $r_D = \sum D_i / (\sum D_i + \sum W_i)$。实证表明 $r_D$ 从 0.3 增至约 0.7 时鲁棒性单调上升，超过 0.7 后迅速下降，故最优比例约为深:宽 = 7:3；阶段分配遵循 $D_1:D_2:D_3 = 2:2:1$（前两层堆叠深度、末层收缩）与 $W_1:W_2:W_3 = 2:2.5:1$（第二层最宽、末层收窄），形成“深而窄”的紧凑布局。
- **训练与评估协议**：采用 Baseline AT (BAT) 与 Advanced AT (A
