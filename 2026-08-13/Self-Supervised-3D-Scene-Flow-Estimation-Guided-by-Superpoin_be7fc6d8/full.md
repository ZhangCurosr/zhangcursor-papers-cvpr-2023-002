# Self-Supervised 3D Scene Flow Estimation Guided by Superpoints

Yaqi Shen, Le Hui, Jin Xie<sup>∗</sup>, Jian Yang PCA Lab, Nanjing University of Science and Technology, Nanjing, China {syq, le.hui, csjxie, csjyang}@njust.edu.cn

## Abstract

3D scene flow estimation aims to estimate point-wise motions between two consecutive frames of point clouds. Superpoints, i.e., points with similar geometric features, are usually employed to capture similar motions of local regions in 3D scenes for scene flow estimation. However, in existing methods, superpoints are generated with the offline clustering methods, which cannot characterize local regions with similar motions for complex 3D scenes well, leading to inaccurate scene flow estimation. To this end, we propose an iterative end-to-end superpoint based scene flow estimation framework, where the superpoints can be dynamically updated to guide the point-level flow prediction. Specifically, our framework consists of a flow guided superpoint generation module and a superpoint guided flow refinement module. In our superpoint generation module, we utilize the bidirectional flow information at the previous iteration to obtain the matching points of points and superpoint centers for soft point-to-superpoint association construction, in which the superpoints are generated for pairwise point clouds. With the generated superpoints, we first reconstruct the flow for each point by adaptively aggregating the superpoint-level flow, and then encode the consistency between the reconstructed flow ofpairwise point clouds. Finally, we feed the consistency encoding along with the reconstructed flow into GRU to refine point-level flow. Extensive experiments on several different datasets show that our method can achieve promising performance. Code is available at https://github. com/supersyq/SPFlowNet.

## 1. Introduction

Scene flow estimation is one of the vital components of numerous applications such as 3D reconstruction [10], autonomous driving [37], and motion segmentation [2]. Estimating scene flow from stereo videos and RGB-D images has been studied for many years [17, 19]. Recently, with the rapid development of 3D sensors, estimating scene flow from two consecutive point clouds has receiving more and more attention. However, due to the irregularity and sparsity of point clouds, scene flow estimation from point clouds is still a challenging problem in real scenes.

![](images/671cd66c8255f523ab4eced1cc4b90be1b1fb49ed1085931617d36de825ae9c4.jpg)  
Figure 1. Comparison with other clustering-based methods. (a) Other clustering based methods utilize offline clustering algorithms to split the point clouds into some fixed superpoints for subsequent flow refinement, which is not learnable. (b) Our method embeds the differentiable clustering (superpoint generation) into our pipeline and generates dynamic superpoints at each iteration. We visualize part of the scene in FlyingThings3D [38] for better visualization. Different colors indicate different superpoints and red lines indicate the ground truth flow.

In recent years, many 3D scene flow estimation methods have been proposed [11, 34, 37, 56, 57, 59]. Most of these methods [34, 56] rely on dense ground truth scene flow as supervision for model training. However, collecting point-wise scene flow annotations is expensive and time-consuming. To avoid the expensive point-level annotations, some efforts have been dedicated to weaklysupervised and self-supervised scene flow estimation [9, 23, 46, 60]. For example, both Rigid3DSceneFlow [9] and LiDARSceneFlow [7] propose a weakly-supervised scene flow estimation framework, which only take the egomotion and background masks as inputs. Especially, they utilize the DBSCAN clustering algorithm [8] to segment the foreground points into local regions with flow rigidity constraints. In addition, RigidFlow [31] first utilizes the off-line oversegmentation method [32] to decompose the source point clouds into some fixed supervoxels, and then estimates the rigid transformations for supervoxels as pseudo scene flow labels for model training. In summary, these clustering based methods utilize offline clustering algorithms with hand-crafted features (i.e., coordinates and normals) to generate the superpoints and use the consistent flow constraints on these fixed superpoints for scene flow estimation. However, for some complex scenes, the offline clustering methods may cluster points with different flow patterns into the same superpoints. Figure 1(a) shows that [32] falsely clusters points with the entirely different flow into the same superpoint colored in purple (highlighted by the dotted circle). Thus, applying flow constraints to the incorrect and fixed superpoints for flow estimation will mislead the model to generate false flow results.

To address this issue, we propose an iterative end-toend superpoint guided scene flow estimation framework (dubbed as “SPFlowNet”), which consists of an online superpoint generation module and a flow refinement module. Our pipeline jointly optimizes the flow guided superpoint generation and superpoint guided flow refinement for more accurate flow prediction (Figure 1(b)). Specifically, we first utilize farthest point sampling (FPS) to obtain the initial superpoint centers, including the coordinate, flow, and feature information. Then, we use the superpoint-level and point-level flow information in the previous iteration to obtain the matching points of points and superpoint centers. With the pairs of points and superpoint centers, we can learn the soft point-to-superpoint association map. And we utilize the association map to adaptively aggregate the coordinates, features, and flow values of points for superpoint center updating. Next, based on the updated superpoint-wise flow values, we reconstruct the flow of each point via the generated association map. Furthermore, we encode the consistency between the reconstructed flow of pairwise point clouds. Finally, we feed the reconstructed flow along with the consistency encoding into a gated recurrent unit to refine the point-level flow. Extensive experiments on several benchmarks show that our approach achieves stateof-the-art performance.

Our main contributions are summarized as follows:

• We propose a novel end-to-end self-supervised scene flow estimation framework, which iteratively generates dynamic superpoints with similar flow patterns and refines the point-level flow with the superpoints.

• Different from other offline clustering based methods, we embed the online clustering into our model to dynamically segment point clouds with the guidance from pseudo flow labels generated at the last iteration.

• A superpoint guided flow refinement layer is introduced to refine the point-wise flow with superpointlevel flow information, where the superpoint-wise flow patterns are adaptively aggregated into the point-level with the learned association map.

• Our self-supervised scene flow estimation method outperforms state-of-the-art methods by a large margin.

## 2. Related Work

Supervised scene flow estimation on point clouds. The scene flow describes the 3D displacements of points between two temporal frames [52]. Estimating scene flow from stereo videos and RGB-D images has been investigated for many years [16, 19, 22, 50]. Recently, with the development of 3D sensor, directly estimating scene flow on point clouds has drawn the interest of many researchers. There are some supervised scene flow estimation methods [3, 25, 53, 55, 58, 59]. FlowNet3D [34] is the first end-toend scene flow estimation framework on point clouds with a flow embedding layer to capture the local correlation between source and target point clouds and a set upconv layer to propagate the flow embedding from the coarse scale to the finer scale for flow regression. Except for FlowNet3D, some other methods also involve multiscale analysis, such as [6, 12, 54–56]. Among them, Bi-PointFlowNet [6] propagates the features of two frames bidirectionally at different scales to obtain bidirectional correlations, which achieves promising performance. To explicitly encode the rigid motion, HCRF-Flow [29] uses [32] to segment the scenes into supervoxels and takes supervoxels as rigid objects for flow refinement with conditional random fields. Nevertheless, the above methods build local correlations within a limited search area, which fail to accurately estimate the large displacements. Therefore, FLOT [45] and SCTN [27] adopt optimal transport to build global correlation. In contrast, CamLiFlow [33] takes two consecutive synchronized camera and Lidar frames as inputs to estimate the optical flow and scene flow simultaneously and builds multiple bidirectional connections between its 2D and 3D branches to fuse the information of two modalities. Unlike other methods that focus on a pair of point clouds, SPCM-Net [14], MeteorNet [35], and [18] take a sequence of point clouds as input. Specifically, SPCM-Net computes spatiotemporal cost volumes between pairwise two frames and utilizes an order-invariant recurrent unit to aggregate the correlations across time. Although these supervised scene flow estimation methods achieve adorable performance, they need dense supervision for model training, while acquiring point-wise annotations is expensive.

Self-supervised scene flow estimation on point clouds. To address this drawback, there are some self-supervised and weakly-supervised methods [15,28,36,37,42,61]. The selfsupervised methods [41,44,51] utilize the cycle-consistency loss and nearest neighbor loss for model training. Besides, PointPWC-Net [60] combines the nearest neighbor loss with a flow smoothness loss and a Laplacian regularization loss as the self-supervised loss. [30] generates pseudo labels by optimal transport and refines the generated pseudo labels with the random walk. The generated pseudo labels are used for unsupervised model optimization. The followup RigidFlow [31] utilizes optimization-based point cloud oversegmentation method [32] to split point clouds into a set of supervoxels and then calculates the rigid transformation as pseudo flow labels. Rigid3DSceneFlow [9] and Li-DARSceneFlow [7] get rid of the requirement for expensive point-wise flow supervision with binary background masks as well as ego-motion and utilize the DBSCAN clustering algorithm [8] to segment the foreground points for flow rigidity constraints. LiDARSceneFlow expands [9] with a Gated Recurrent Unit (GRU) for flow refinement. The previous methods based on offline clustering mainly employ hand-crafted features (i.e., coordinates and normals) to offline cluster superpoints, which may cluster points with different motion patterns into the same clusters and further lead to worse results with rigidity constraints on the incorrect clusters. Our method attempts to dynamically cluster point clouds into superpoints and then refines the point-wise flow with superpoint-level flow information. In this way, our model can jointly optimize the superpoint generation and flow refinement for more accurate results. Additionally, other self-supervised methods [1, 11, 24] also achieve promising performance.

Point cloud oversegmentation. Point cloud oversegmentation semantically clusters points into superpoints. Recently, some optimization-based superpoint oversegmentation methods are proposed [13, 32]. Among them, [32] converts the point cloud oversegmentation into a subset selection problem and develops a heuristic algorithm to solve it. In contrast, SPNet [21] is the first end-to-end superpoint generation network. Due to low computational cost, superpoints are used for many down-stream tasks, such as point cloud segmentation [4,5,20,48]. In this paper, we introduce superpoints into scene flow estimation based on SPNet. Different from that SPNet focuses on generating superpoints in a single point cloud, our model utilizes the bidirectional flow information at the previous iteration to guide superpoint generation for pairwise point clouds.

## 3. Method

In this section, we illustrate our superpoint guided scene flow estimation (SPFlowNet) framework in detail. As shown in Figure 2, SPFlowNet consists of a flow guided superpoint generation module and a superpoint guided flow refinement module. It takes two consecutive point clouds $\mathbf { P } ~ = ~ \{ \mathbf { p } _ { i } ~ \in ~ \mathbb { R } ^ { 3 } ~ | ~ i ~ = ~ 1 , 2 , \ldots , n \}$ and $\mathbf { Q } \ = \ \{ \mathbf { q } _ { j } \in$ $\mathbb { R } ^ { 3 } \ \mid \ j \ = \ 1 , 2 , \ldots , m \}$ as inputs and outputs the flow $\mathbf { F } ^ { t } = \{ \mathbf { F } ^ { p , t } , \mathbf { F } ^ { q , t } \}$ at the t-th iteration for point clouds P and $\mathbf { Q } ,$ respectively. Note that the iteration subscript $t = 0$ means that our model is in the initialization stage.

## 3.1. Initialization

Initial flow. Firstly, we utilize the feature encoder used in FLOT [45] to extract the features for point clouds $\mathbf { P }$ and Q. The local features of P and Q can be denoted as $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ and $\textbf { Y } \in \ \mathbb { R } ^ { m \times d }$ , where d is the dimension of the feature. Then, we calculate the global correlation $\mathbf { W } \in \mathbb { R } ^ { n \times m }$ between the point clouds P and $\mathbf { Q } ,$ where W can be formulated as the dot product between their features. Next, we apply the Sinkhorn algorithm [47] to it for the final correlation map W. The initial flow $\bar { \mathbf { f } _ { i } ^ { p , 0 } } \in \mathbf { F } ^ { p , 0 }$ for each point $\mathbf { p } _ { i } \in \mathbf { P }$ can be defined as

$$
\mathbf { f } _ { i } ^ { p , 0 } = \frac { \sum _ { j = 1 } ^ { m } w _ { i , j } \mathbf { q } _ { j } } { \sum _ { j = 1 } ^ { m } w _ { i , j } } - \mathbf { p } _ { i }\tag{1}
$$

Similarly, we can obtain the initial flow $\mathbf { F } ^ { q , 0 }$ for point clouds Q by taking the same operations as P on Q.

Initial superpoint center. We obtain $L \ ( L \ \ll \ n$ and $L \ll m )$ initial superpoint centers $\mathbf { S P } ^ { 0 } = \{ \mathbf { S P } ^ { p , 0 } , \mathbf { S P } ^ { q , 0 } \}$ for point clouds P and Q by employing the FPS algorithm in the coordinate space. $\mathbf { S P } ^ { p , 0 }$ and $\bar { \mathbf { S } } \mathbf { P } ^ { \bar { q } , 0 }$ denote the initial superpoint centers for pairwise point clouds P and Q, respectively. Each superpoint center includes the coordinate, flow, and descriptor information, denoted by $\mathbf { S C } ^ { 0 } , \mathbf { S F } ^ { 0 }$ and $\mathbf { S D } ^ { 0 }$ , respectively.

## 3.2. Flow Guided Superpoint Generation

The scene flow estimation methods [9,31] usually exploit the offline clustering methods [8, 32] to decompose the point clouds into a collection of clusters and employ the flow rigidity constraints on the fixed clusters. However, the offline clustering methods usually generate false clusters, where the points with different flow patterns exist in the same cluster, as shown in Figure 1(a). Therefore, an online flow guided superpoint generation module is embedded in our framework, in which the point clouds are dynamically divided into superpoints. Due to the joint end-to-end optimization with the consequent flow refinement module, our model can relieve the above problem to some extent.

Point-to-Superpoint association calculation. Our method attempts to generate superpoints that satisfy the following requirements: (1) The points of the same superpoint are with similar flow patterns; (2) They are also close to the superpoint centers in the coordinate space; (3) Their features are semantically similar with each other. Thus, we follow SPNet [21] to build the soft association map between points and superpoint centers by adaptively learning the bilateral weights from both the coordinate and feature spaces. Different from SPNet, we introduce the previously iterated flow information at both point level and superpoint level to obtain the corresponding point/superpoint center via the bidirectional warping operation (source → target and target → source). Thus, we employ pairs of points and superpoint centers in the source and target point clouds to learn the similarity across the source and target while SPNet does not consider the pairs of corresponding points to learn the similarity. Note that following SPNet, we only calculate the association weights between each point and its K-nearest superpoint centers $( K \ll L )$ in the coordinate space, which is more efficient.

![](images/86656fe11396cb9c84c7871cfd9bed72504a8719df3d52e358357bc5ad971bc5.jpg)  
Figure 2. An overview of SPFlowNet. Given two consecutive point clouds P and Q, we first calculate the initial flow ${ \bf F } ^ { 0 } = \{ { \bf F } ^ { p , 0 } , { \bf F } ^ { q , 0 } \}$ and the initial superpoint centers ${ \bf S } { \bf P } ^ { 0 } = \{ { \bf S } { \bf P } ^ { p , 0 } , { \bf S } { \bf P } ^ { q , 0 } \}$ at the initialization stage $( t = 0 )$ . Then, our model iteratively performs the flow guided superpoint generation module and the superpoint guided flow refinement module for scene flow estimation. In the end, we can obtain final flow results after several iterations. Specifically, at the t-th iteration, the flow guided superpoint generation module clusters points into dynamic superpoints $\mathbf { S } \mathbf { P } ^ { t } = \{ \mathbf { S } \mathbf { P } ^ { p , t } , \mathbf { \hat { S } } \mathbf { P } ^ { q , t } \}$ with the pseudo superpoint-level and point-level flow labels generated at the previous iteration. With the generated superpoints, the superpoint guided flow refinement module feeds the superpoint-level flow and consistency encoding into GRU to obtain the updated point-level flow $\mathbf { F } ^ { p , t }$ and $\mathbf { F } ^ { q , t }$

We take the source point cloud P as an example to illustrate the point-to-superpoint association map calculation. Specifically, for the i-th point in source point cloud P, we use the Euclidean distance in the coordinate space to select the attended K superpoint centers N , where $\mathcal { N } _ { i } = \{ \mathbf { s c } _ { i , k } ^ { p , t - 1 } \in \mathbb { R } ^ { 3 } , \mathbf { s f } _ { i , k } ^ { p , t - 1 } \in \mathbb { R } ^ { 3 } , \mathbf { s d } _ { i , k } ^ { p , t - 1 } \in \mathbb { R } ^ { d } \} _ { k = 0 } ^ { K }$ includes the coordinate, flow, and feature information of the K-nearest superpoint centers. At the t-th iteration, the association $a _ { i , k } ^ { p , t } \in \bar { \mathbf { A } } ^ { p , t }$ between the i-th point and the k-th superpoint center in source point cloud P is defined as

$$
\begin{array} { r l } & { a _ { i , k } ^ { p , t } = \mathrm { M L P } \left( \mathbf { u } _ { i , k } \right) + \mathrm { M L P } \left( \mathbf { g } _ { i , k } \right) } \\ & { \mathbf { u } _ { i , k } = \left( \mathbf { x } _ { i } \right| \left| \hat { \mathbf { x } } _ { i } ^ { t - 1 } \right) - \left( \mathbf { s d } _ { i , k } ^ { p , t - 1 } \right| \left| \mathbf { \hat { s d } } _ { i , k } ^ { p , t - 1 } \right) } \\ & { \mathbf { g } _ { i , k } = \left( \mathbf { p } _ { i } \right| \left| \hat { \mathbf { p } } _ { i } ^ { t - 1 } \right) - \left( \mathbf { s c } _ { i , k } ^ { p , t - 1 } \right| \left| \mathbf { \hat { s c } } _ { i , k } ^ { p , t - 1 } \right) } \end{array}\tag{2}
$$

where || is the concatenation, $\mathbf { u } _ { i , k } \in \mathbb { R } ^ { 1 \times ( 2 * d ) }$ and $\mathbf { g } _ { i , k } \in$ $\mathbb { R } ^ { 1 \times ( 2 \ast \ddot { 3 } ) }$ represent the differences between the i-th point and the k-th superpoint center in feature and coordinate spaces, respectively. Besides, MLP(·) denotes a multi-layer perceptron followed by a sum-pooling operation, which is used to map the above difference information to association weights in both coordinate and feature spaces.

In Equation (2), we also utilize the feature and coordinate information of their corresponding points generated by the predicted point-level and superpoint-level flow in the previous iteration. The corresponding point $( \hat { \mathbf { p } } _ { i } ^ { t - 1 } , \hat { \mathbf { x } } _ { i } ^ { t - 1 } )$ and superpoint center $( \hat { \mathbf { s c } } _ { i , k } ^ { p , t - 1 } , \hat { \mathbf { s d } } _ { i , k } ^ { p , t - 1 } )$ for point $\mathbf { p } _ { i }$ and superpoint center $\mathbf { s c } _ { i , k } ^ { p , t - 1 }$ are defined as

$$
\begin{array} { r l } & { \hat { \mathbf { p } } _ { i } ^ { t - 1 } = \mathbf { p } _ { i } + \mathbf { f } _ { i } ^ { p , t - 1 } , \hat { \mathbf { s c } } _ { i , k } ^ { p , t - 1 } = \mathbf { s c } _ { i , k } ^ { p , t - 1 } + \mathbf { s f } _ { i , k } ^ { p , t - 1 } } \\ & { \hat { \mathbf { x } } _ { i } ^ { t - 1 } = \mathbf { Y } _ { \mathrm { N N } ( \hat { \mathbf { p } } _ { i } ^ { t - 1 } , \mathbf { Q } ) } , \hat { \mathbf { s d } } _ { i , k } ^ { p , t - 1 } = \mathbf { Y } _ { \mathrm { N N } ( \hat { \mathbf { s c } } _ { i , k } ^ { p , t - 1 } , \mathbf { Q } ) } } \end{array}\tag{3}
$$

where $\operatorname { N N } ( \cdot , \mathbf { Q } )$ is used to obtain the index of the nearest matching point in target point cloud $\mathbf { Q } .$

Next, we assign each point $\mathbf { p } _ { i } \in \mathbf { P }$ a probability vector over its K-nearest superpoint centers by

$$
a _ { i , k } ^ { p , t } = \mathrm { s o f t m a x } ( [ a _ { i , 1 } ^ { p , t } , . . . , a _ { i , K } ^ { p , t } ) ] ) _ { k }\tag{4}
$$

Similarly, we can obtain the association map $\mathbf { A } ^ { q , t }$ between the target point cloud $\mathbf { Q }$ and its superpoint centers.

Superpoint center updating. With the generated association map $\mathbf { A } ^ { t } .$ , we can assign each point to its $K \cdot$ -nearest superpoint centers with the learned weights. For each superpoint center, we adaptively aggregate the coordinate, flow, and feature information of the points belonging to it to update this superpoint center via the normalized association map. Specifically, given the local feature $\mathbf { X } ,$ flow $\mathbf { F } ^ { p , t - 1 } =$ $\{ \mathbf { f } _ { i } ^ { \bar { p , } t - 1 } | i \ = \ \bar { 1 , } . . . , n \}$ at the iteration $t \mathrm { ~ - ~ } 1$ and the association map $\mathbf { A } ^ { p , t }$ at the current t-th iteration of the source point cloud P, the updated l-th superpoint center in source point clouds can be formulated as

$$
\begin{array} { l } { { \displaystyle { \bf s c } _ { l } ^ { p , t } = \frac { 1 } { r } \sum _ { i = 1 } ^ { n } 1 \left\{ l \in \mathcal { N } _ { i } \right\} a _ { i , l } ^ { p , t } { \bf p } _ { i } } } \\ { { \displaystyle { \bf s f } _ { l } ^ { p , t } = \frac { 1 } { r } \sum _ { i = 1 } ^ { n } 1 \left\{ l \in \mathcal { N } _ { i } \right\} a _ { i , l } ^ { p , t } { \bf f } _ { i } ^ { p , t - 1 } } } \\ { { \displaystyle { \bf s d } _ { l } ^ { p , t } = \frac { 1 } { r } \sum _ { i = 1 } ^ { n } 1 \left\{ l \in \mathcal { N } _ { i } \right\} a _ { i , l } ^ { p , t } { \bf x } _ { i } } } \end{array}\tag{5}
$$

where $\mathbb { 1 } \left\{ l \in \mathcal { N } _ { i } \right\}$ is an indicator function that equals to one if the l-th superpoint center belongs to ${ \mathcal { N } } _ { i }$ , and zero otherwise. Besides, $\begin{array} { r c l } { r } & { = } & { \sum _ { i = 1 } ^ { n } \Im \left\{ l \in \mathcal { N } _ { i } \right\} a _ { i , l } ^ { p , t } } \end{array}$ is the normalization factor. Similarly, we update the superpoint centers in target point cloud Q. For brevity, we only visualize the pipeline of flow guided superpoint generation for source point cloud P in Figure 2.

## 3.3. Superpoint Guided Flow Refinement

Inspired by RAFT [49], many scene flow estimation methods [11, 26, 59] utilize a Gate Recurrent Unit (GRU) to iteratively update the predicted flow.

Gated recurrent unit. Given the hidden state $\mathbf { h } ^ { t - 1 }$ at the iteration $t - 1$ and the current iteration information $\mathbf { v } _ { } ^ { t }$ , the calculations of GRU can be written as

$$
\begin{array} { r l } & { \mathbf { z } ^ { t } = \sigma \left( \operatorname { S e t C o n v } _ { z } \left( \mathbf { h } ^ { t - 1 } | | \mathbf { v } ^ { t } \right) \right) } \\ & { \mathbf { r } ^ { t } = \sigma \left( \operatorname { S e t C o n v } _ { r } \left( \mathbf { h } ^ { t - 1 } | | \mathbf { v } ^ { t } \right) \right) } \\ & { \hat { \mathbf { h } } ^ { t } = \operatorname { t a n h } \left( \operatorname { S e t C o n v } _ { h } \left( \left( \mathbf { r } ^ { t } \odot \mathbf { h } ^ { t - 1 } \right) | | \mathbf { v } ^ { t } \right) \right) } \\ & { \mathbf { h } ^ { t } = \left( 1 - \mathbf { z } ^ { t } \right) \odot \mathbf { h } ^ { t - 1 } + \mathbf { z } ^ { t } \odot \hat { \mathbf { h } } ^ { t } } \end{array}\tag{6}
$$

where  is the Hadamard product, || is the concatenation, and $\sigma ( \cdot )$ is the sigmoid function. The SetConv layers are adopted from [26, 45].

The existing GRU-based methods usually concatenate the feature, flow, and flow embedding of each point as current iteration information $\mathbf { v } ^ { t } .$ , and regress the flow from the new hidden state $\mathbf { h } ^ { t }$ . Although these methods achieve promising results, most of them only involve point-level flow information. In contrast, [7] converts GRU output into rigid flow according to pre-clustered local regions, it is limited by pre-clustered regions. Our method adaptively learns the flow association at the superpoint level and does not rely on rigid object assumption. Specifically, we encode the superpoint-level flow information into the current iteration information $\mathbf { v } ^ { t }$ to guide the new hidden state $\mathbf { h } ^ { t }$ generation. Moreover, we utilize the consistency between the reconstructed flow values from the generated superpoints of pairwise point clouds to encode the confidence into the current iteration information $\mathbf { v } _ { } ^ { t }$ . Therefore, the current iteration information in our model simultaneously considers the superpoint-level flow information and its confidence.

Superpoint-level flow reconstruction. With the updated superpoint-level flow $\mathbf { S } \mathbf { F } ^ { p , t }$ and $\mathbf { S F } ^ { q , t }$ for superpoint centers in both point clouds $\mathbf { P }$ and $\mathbf { Q } ,$ , here we map the superpoint-level flow of K-nearest superpoint centers back onto each point in original point clouds via the learned association map $\mathbf { A } ^ { t }$ as follows

$$
\widetilde { \mathbf { F } } _ { i } ^ { p , t } = et { } { ' } { \sum } _ { k = 1 } ^ { K } a _ { i , k } ^ { p , t } \mathbf { s f } _ { k } ^ { p , t } , \widetilde { \mathbf { F } } _ { i } ^ { q , t } = et { } { ' } { \sum } _ { k = 1 } ^ { K } a _ { i , k } ^ { q , t } \mathbf { s f } _ { k } ^ { q , t }\tag{7}
$$

where $\widetilde { \mathbf { F } } _ { i } ^ { p , t }$ and $\widetilde { \mathbf { F } } _ { i } ^ { q , t }$ are the reconstructed superpointlevel flow for point clouds P and $\mathbf { Q } ,$ respectively. In this way, the reconstructed flow of each point in original point clouds adaptively aggregates the superpoint-level flow values of its $K \cdot$ -nearest superpoint centers. Since the superpoint-level flow values capture the flow patterns of the generated superpoints, we aim to utilize the superpointlevel flow pattern to guide the point-level flow refinement. Consistency encoding. We do a backward interpolation Ω used in [43] to propagate the reconstructed superpoint-level flow in the source point clouds to the target point clouds and vice versa. Next, we utilize the consistency between the interpolated flow and reconstructed flow to encode the confidence of the superpoint-level flow by

$$
{ \bf C } ^ { p , t } = \pi \left( \widetilde { \bf F } ^ { p , t } - \Omega ( \widetilde { \bf F } ^ { q , t } ) \right) , { \bf C } ^ { q , t } = \tau \left( \widetilde { \bf F } ^ { q , t } - \Omega ( \widetilde { \bf F } ^ { p , t } ) \right)\tag{8}
$$

where π and τ are the MLP layers with a sigmoid function. Besides, we send the reconstructed superpoint-level flow $\widetilde { \mathbf { F } } ^ { p , t }$ into a flow embedding layer used in [26] to obtain the correlation feature $\mathbf { F } _ { c } ^ { p , t }$ and a Linear layer to encode the flow feature $\mathbf { F } _ { e } ^ { p , t }$ . With the confidence $\mathbf { C } ^ { p , t }$ , the current iteration information $\mathbf { v } ^ { t }$ for source point cloud $\mathbf { P }$ can be defined as

$$
\mathbf { v } ^ { t } = \operatorname { S e t C o n v } _ { \mathrm { c } } \left( \mathbf { F } _ { c } ^ { p , t } \mathbf { C } ^ { p , t } \right) + \operatorname { S e t C o n v } _ { \mathrm { e } } \left( \mathbf { F } _ { e } ^ { p , t } \mathbf { C } ^ { p , t } \right)\tag{9}
$$

where the SetConv layers are adopted from [26, 45].

We send $\mathbf { v } ^ { t }$ into GRU to obtain the new hidden state h<sup>t</sup>. Finally, given the new hidden state $\mathbf { h } ^ { t } .$ , we use a flow regressor to obtain the residual flow $\triangle \mathbf { F } ^ { p , t }$ . Therefore, the updated flow for source point cloud P at the iteration t can be formulated as $\mathbf { F } ^ { p , t } = \widetilde { \mathbf { F } } ^ { p , t - 1 } + \Delta \mathbf { F } ^ { p , t }$ . Similarly, we can obtain the updated flow $\mathbf { F } ^ { q , t }$ for target point cloud Q.

## 3.4. Self-Supervised Loss Functions

At each iteration, we can obtain the estimated flow $\mathbf { F } _ { t } =$ $\{ \mathbf { F } ^ { p , t } , \mathbf { F } ^ { q , t } \}$ for pairwise point clouds P and $\mathbf { Q } .$ Without the ground truth scene flow, we utilize the following loss functions for model training. For simplicity, we omit the iteration subscript.

Chamfer loss. Following [26, 60], we warp the source P with the predicted flow F<sup>p</sup> and minimize the Chamfer Distance between the warped source $P ^ { \prime }$ and target Q by

$$
\begin{array} { r l } { { \displaystyle { \cal L } _ { c h } ( { \bf P } ^ { \prime } , { \bf Q } ) = \sum _ { { \bf q } _ { j } \in { \bf Q } } \operatorname* { m i n } _ { { \bf p } ^ { \prime } { \bf \Phi } _ { i } \in { \bf P } ^ { \prime } } \| { \bf q } _ { j } - { \bf p ^ { \prime } } { \bf \Phi } _ { i } \| _ { 2 } } + }  & { } \\ { { \displaystyle \sum _ { { \bf p ^ { \prime } } _ { i } \in { \bf P ^ { \prime } } } \operatorname* { m i n } _ { { \bf q } _ { j } \in { \bf Q } } \| { \bf p ^ { \prime } } _ { i } - { \bf q } _ { j } \| _ { 2 } } }  & { } \end{array}\tag{10}
$$

Smoothness loss. Following [26, 60], we also constrain the predicted scene flow values within a small local region to be similar. The smoothness loss is defined as

$$
L _ { s } = \sum _ { \mathbf { p } _ { i } \in P } \frac { 1 } { | \mathcal { N } _ { i } ^ { \prime } | } \sum _ { \mathbf { p } _ { j } \in \mathcal { N } _ { i } ^ { \prime } } \| \mathbf { f } _ { i } ^ { p } - \mathbf { f } _ { j } ^ { p } \| _ { 2 }\tag{11}
$$

where $\mathcal { N } _ { i } ^ { \prime }$ is the neighborhood around $\mathbf { p } _ { i } \in { \cal P } .$

Consistency loss. We enforce the backward-interpolated flow of the target point clouds to be consistent with the predicted flow of the source point clouds and vice versa.

$$
L _ { c } = \Vert \mathbf { F } ^ { p } - \boldsymbol { \Omega } \left( \mathbf { F } ^ { q } \right) \Vert _ { 2 } + \Vert \mathbf { F } ^ { q } - \boldsymbol { \Omega } \left( \mathbf { F } ^ { p } \right) \Vert _ { 2 }\tag{12}
$$

where Ω is backward interpolation.

The combined loss for self-supervised training can be written as

$$
L = L _ { c h } + \alpha L _ { s } + \beta L _ { c }\tag{13}
$$

where α and $\beta$ are the regularization parameters.

## 4. Experiment

## 4.1. Experimental Setups

Datasets. To validate the effectiveness of our proposed scene flow estimation framework, we conduct extensive experiments on two benchmarks, the FlyingThings3D [38] and the KITTI Scene Flow [39, 40]. There are two versions of datasets. The first version of the datasets is prepared by HPLFlowNet [12]. We denote these datasets without occluded points FT3D and $\mathrm { K I T T I } _ { \mathrm { s } } ,$ , respectively. $\mathrm { F T 3 D _ { s } }$ contains 19640 training examples and 3824 pairs in the test set. We only use one-quarter of the training data (4910 pairs). KITTI is a real-world scene flow dataset with 200 pairs for which 142 are used for testing without any fine-tuning. The second version of the datasets is prepared by FlowNet3D [34]. This version of datasets includes the occluded points, which are denoted $\mathrm { F T 3 D _ { o } }$ and $\mathrm { K I T T I _ { o } } .$ respectively. $\mathrm { F T 3 D _ { o } }$ contains 19999 training examples and 2003 pairs in the test set. $\mathrm { K I T T _ { o } }$ consists of 150 test examples. Besides, following Self-Point-Flow [30], we also split the $\mathrm { K I T T I _ { o } }$ dataset into KITTI<sub>f</sub> with 100 pairs and KITTI<sub>t</sub> with 50 pairs for evaluation. Moreover, [30] also extracts another self-supervised training dataset with 6026 pairs from the original KITTI dataset, denoted as KITTI<sub>r</sub>.

Implementation details. Our model is implemented with Pytorch and all experiments are executed on a NVIDIA TITAN RTX GPU. For the experiments on point clouds without occlusions, we train our model on synthetic $\mathrm { F T 3 D _ { s } }$ training data and evaluate it on both $\mathrm { F T 3 D _ { s } }$ test data and $\mathrm { K I T T I _ { s } }$ dataset. we feed randomly sampled 8192 points as inputs to our model, just like [31, 45] and other compared methods, and train it with a batch size of 2. Besides, we set the superpoint number and iteration number to 128 and 3, respectively. For occluded experiments, like [31], we also train our model on KITTI dataset and test it on $\mathrm { K I T T I _ { o } }$ and KITTI . The size of the input point clouds is set to 2048. Here, we set the batch size, iteration number, and superpoint number to 4, 3, and 30, respectively. The initial learning rate used in all experiments is 0.001 and our model is optimized with the ADAM optimizer. We multiply the learning rate by 0.7 at epochs 40, 55, and 70 and train our model for 100 epochs.

Evaluation metrics. We test our model with four evaluation metrics used in [12, 34], including End Point Error (EPE), Accuracy Strict (AS), Accuracy Relax (AR), and Outliers (Out). We denote the estimated scene flow and ground truth scene flow as F and $\mathbf { F } _ { g t } ,$ respectively. EPE(m): $| | \mathbf { F } - \mathbf { F } _ { g t } | | _ { 2 }$ averaged over all points. AS(%): the percentage of points whose EPE <0.05m or relative error <5%. AR(%): the percentage of points whose EPE <0.1m or relative error <10%. Out(%): the percentage of points whose EPE >0.3m or relative error >10%.

## 4.2. Results

Performance on point clouds without occlusions. We train our self-supervised model on FT3D<sub>s</sub> training data and evaluate it on both FT3D<sub>s</sub> test data and KITTI<sub>s</sub> dataset. And we compare our model with the recent state-of-the-art self-supervised scene flow estimation methods, including Ego-Motion [51], PointPWC-Net [60], SLIM [2], Self-Point-Flow [30], FlowStep3D [26], RCP [11], PDF-Flow [15], and RigidFlow [31]. The results are reported in Table 1. From the results, it can be found that our model can outperform all compared self-supervised methods in terms of the four metrics on the FT3D test data. Especially, our model brings 8.72% gains for metric AS. For the KITTI dataset, our model brings substantial improvements on all metrics. To be specific, our model outperforms the second best method RCP [11] by 8.68% and 6.58% on metrics AS and AR, respectively. Besides, it is worth noting that our model can even achieve an EPE metric of 3.62cm, which is much lower than the EPE (6.19cm) of recent RigidFlow.

We also compare our model with some supervised methods, such as FlowNet3D [34] and FLOT [45], etc. As shown in Table 1, our self-supervised model achieves comparable performance with supervised HPLFlowNet [12] on FT3D dataset. Without any fine-tuning on KITTI<sub>s</sub> dataset, our model can even outperform the supervised methods listed in Table 1, which proves that our model has better generalization ability. For real scenes, most local regions are with similar flow patterns. Thanks to dynamically clustering mechanism, our model clusters points with similar flow pattern into the same clusters and encodes the superpointlevel flow into the GRU for flow refinement, thereby leading to satisfactory performance on real scenes.

Performance on point clouds with occlusions. Following the experimental settings used in Self-Point-Flow [30] and RigidFlow [31], we train our model on KITTI dataset and evaluate our model on both $\mathrm { K I T T _ { o } }$ and $\mathrm { K I T T I _ { t } }$ datasets. The results on $\mathrm { K I T T _ { o } }$ and $\mathrm { K I T T I _ { t } }$ are shown in Tables 2 and 3, respectively. Although our model is not designed to deal with occluded cases, our model can also achieve the best performance on $\mathrm { K I T T _ { o } }$ dataset. This is due to that although there is no correspondence of the occluded points, our model employs the superpoint-level flow to guide the flow refinement rather than point-level flow information, which can alleviate the occluded problem to some extent. Due to the lack of point-level flow annotations for the real scenes, the supervised FLOT and FlowNet3D are trained on synthetic $\mathrm { F T 3 D _ { o } }$ dataset. The other two self-supervised methods [30, 31] and our model can be trained directly on unlabeled outdoor $\mathrm { K I T T I _ { r } }$ dataset. As shown in Table $^ { 2 , }$ our model can outperform all self-supervised methods including Self-Point-Flow and RigidFlow. To be specific, our model brings 12.7% gains on metric AS. Besides, it is worth noting that the Self-Point-Flow [30] needs additional normal and color information for pseudo label generation. Our model only needs the coordinate information of the consecutive frames of point clouds as inputs. For the KITTI<sub>t</sub> dataset, we compare our model with JGF [41] and WWL [44]. These two methods use the pre-trained model of FlowNet3D on $\mathrm { F T 3 D _ { o } }$ as the baseline and perform selfsupervised fine-tuning on $\mathrm { K I T T I _ { f } }$ , and then test their model on $\mathrm { K I T T I _ { t } }$ dataset. Our model and RigidFlow [31] get rid of the pre-trained model on synthetic $\mathrm { F T 3 D _ { o } }$ and only need to be trained on the unlabeled $\mathrm { K I T T I _ { r } }$ in a self-supervised manner. As shown in Table 3, our model can obtain 9.92% improvements on metric AR.

<table><tr><td>Methods</td><td>Sup. EPE↓</td><td></td><td>AS↑</td><td>AR↑</td><td>Out↓</td></tr><tr><td colspan="6"> $\mathrm { F T 3 D _ { s } }$ </td></tr><tr><td>FlowNet3D [34]</td><td>Full.</td><td>0.1136</td><td>41.25</td><td>77.06</td><td>60.16</td></tr><tr><td>HPLFlowNet [12]</td><td>Full.</td><td>0.0804</td><td>61.44</td><td>85.55</td><td>42.87</td></tr><tr><td>PointPWC-Net [60] Ego-Motion [51]</td><td>Full. Self.</td><td>0.0588 0.1696</td><td>73.79 25.32</td><td>92.76 55.01</td><td>34.24 80.46</td></tr><tr><td>PoinPWC-Net [60] Self-Point-Flow [30] FlowStep3D [26] PDF-Flow [15] RCP [11] RigidFlow [31]</td><td>Self. Self. Self. Self. Self. Self.</td><td>0.1246 0.1009 0.0852 0.075 0.0765 0.0692</td><td>30.68 42.31 53.63 58.9 58.58</td><td>65.52 77.47 82.62 86.2 86.02</td><td>70.32 60.58 41.98 47.0 41.42</td></tr><tr><td colspan="6">59.62 87.10 46.42 SPFlowNet (ours) Self. 0.0606 68.34 90.74 38.76</td></tr><tr><td>FlowNet3D [34] HPLFlowNet [12] PointPWC-Net [60] FLOT [45]</td><td>Full. Full. Full. Full.</td><td>KITTIS 0.1767 0.1169 0.0694 0.0560</td><td>37.38 47.83 72.81 75.50</td><td>66.77 77.76 88.84</td><td>52.71 41.03 26.48</td></tr><tr><td>Ego-Motion [51] PoinPWC-Net [60] SLIM [2] FlowStep3D [26]</td><td>Self. Self. Self. Self.</td><td>0.4154 0.2549 0.1207 0.1021</td><td>22.09 23.79 51.78 70.80</td><td>90.80 37.21 49.57 79.56 83.94</td><td>24.20 80.96 68.63 40.24</td></tr><tr><td>PDF-Flow [15]</td><td>Self.</td><td>0.092</td><td>74.7</td><td>87.0</td><td>24.56 28.3</td></tr><tr><td>Self-Point-Flow [30]</td><td>Self.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.0619</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.1120</td><td>52.76</td><td>79.36</td><td>40.86</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RigidFlow [31]</td><td>Self.</td><td></td><td>72.37</td><td>89.23</td><td>26.18</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RCP [11]</td><td>Self.</td><td>0.0763</td><td>78.56</td><td>89.21</td><td>18.49</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Self.</td><td></td><td></td><td></td><td></td></tr><tr><td>SPFlowNet (ours)</td><td></td><td>0.0362</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>87.24</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>95.79</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>17.71</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1. Comparison results on the FT3D<sub>s</sub> and KITTI<sub>s</sub> datasets. Our model is trained on FT3D<sub>s</sub> training part and evaluated on $\mathrm { F T 3 D _ { s } }$ test set and KITTI<sub>s</sub> dataset. Full. means the fullysupervised training manner. Self. represents the self-supervised training manner. Note that the best and the second-best results are emboldened and underlined, respectively.
<table><tr><td rowspan=1 colspan=2>Methods</td><td rowspan=1 colspan=1>|Sup.</td><td rowspan=1 colspan=1>|T. dat</td><td rowspan=1 colspan=1>a|EPE ↓ AS ↑ AR ↑ Out ↓</td></tr><tr><td rowspan=5 colspan=2>FlowNet3D [34]FLOT [45]Self-Point-Flow [30]]RigidFlow [31]SPFlowNet (ours)</td><td rowspan=1 colspan=1>|Full.</td><td rowspan=1 colspan=1> $\mathrm { F _ { o } }$ </td><td rowspan=4 colspan=1>0.17327.660.9 64.90.10745.174.046.30.10541.772.550.10.10248.475.644.2</td></tr><tr><td rowspan=1 colspan=1>Full.</td><td rowspan=1 colspan=1> $\mathrm { F _ { o } }$ </td></tr><tr><td rowspan=1 colspan=1>30]</td><td rowspan=1 colspan=1>Self.</td><td rowspan=1 colspan=1> $\mathrm { K } _ { \mathrm { r } }$ </td></tr><tr><td rowspan=2 colspan=1>Self.Self.</td><td rowspan=2 colspan=1> $\mathrm { K } _ { \mathrm { r } }$  $\mathrm { K } _ { \mathrm { r } }$ </td></tr><tr><td rowspan=1 colspan=1>0.08661.182.439.1</td></tr></table>

Table 2. Comparison results on KITTI<sub>o</sub> dataset. Our model is trained on KITTI and evaluated on KITTI dataset. T. data: training data. $\mathrm { F _ { o } \colon F T 3 D _ { o } } .$ $\mathrm { K } _ { \mathrm { r } }$ :KITTI .

## 4.3. Ablation Study

The effectiveness of key components. We conduct experiments to verify the effectiveness of key components in our model. Firstly, we remove the superpoint generation and superpoint guided flow refinement modules in our model. This variant takes a GRU without superpoint guidance for flow refinement (abbr. as “w/o superpoint”). Secondly, we adopt the SPNet [21] for superpoint generation without flow guidance (abbr. as “w/ SPNet”). The model “w/ FGSG (ours)” represents our model with flow guided superpoint generation module. The results of the above three models are listed in the top part of Table 4. From the results of the variant “w/o superpoint” and the other two models with superpoints, it can be found that introducing superpoints into scene flow estimation is effective. Besides, our proposed flow guided superpoint generation module can achieve better results than SPNet, which shows that flow guidance is crucial when there is no ground truth superpoint labels. Besides, we remove the consistency encoding from our model (abbr. as “w/o cons. encoding”). Table 4 shows that the performance drops a lot without the superpoint consistency encoding, which demonstrates that the consistency between the reconstructed superpointlevel flow of pairwise point clouds is important. Finally, we also remove the consistency loss and only utilize the Chamfer loss and smoothness loss for model training (abbr. as “w/o cons. loss”). The results of our model without consistency loss are worse than with it. According to the above comparisons, it can be observed that our model is less effective without any key components.

<table><tr><td>Methods</td><td>|Pre-T. </td><td>T. data</td><td>EPE↓ AS↑ AR↑</td></tr><tr><td>TGF [41]</td><td>√</td><td> $\mathrm { F _ { o } + K _ { f } }$ </td><td>0.218 10.17 34.38</td></tr><tr><td>WWL [44]</td><td>V</td><td> $\mathrm { F _ { o } + K _ { f } }$ </td><td>0.169 21.71 47.75</td></tr><tr><td>RigidFlow [31]</td><td></td><td> $\mathrm { K } _ { \mathrm { r } }$ </td><td>0.117 38.75 69.73</td></tr><tr><td>SPFlowNet (ours)</td><td></td><td> $\mathrm { K } _ { \mathrm { r } }$ </td><td>0.089 53.28 79.65</td></tr></table>

Table 3. Comparison results on KITTI<sub>t</sub> dataset. Our model is trained on $\mathrm { K I T T I } _ { \mathrm { { r } } }$ and evaluated on $\mathrm { K I T T I _ { t } }$ dataset.

<table><tr><td>Methods</td><td>EPE↓</td><td>AS↑</td><td>AR↑</td><td>Out↓</td></tr><tr><td>w/o superpoint</td><td>0.119 0.090</td><td>55.4</td><td>72.9</td><td>45.2</td></tr><tr><td>w/ SPNet w/ FGSG (ours)</td><td>0.086</td><td>60.0 61.1</td><td>80.7 82.4</td><td>40.2 39.1</td></tr><tr><td>w/o cons. encoding</td><td>0.103</td><td>57.7</td><td>76.1</td><td>44.3</td></tr><tr><td>w/o cons. loss</td><td>0.094</td><td>59.0</td><td>80.0</td><td>40.7</td></tr><tr><td>SPFlowNet (ours)</td><td>0.086</td><td>61.1</td><td>82.4</td><td>39.1</td></tr></table>

Table 4. Comparison results on the $\mathrm { K I T T _ { o } }$ dataset. All models are trained on KITTI and evaluated on $\mathrm { K I T T _ { o } }$ dataset.

Choices of the superpoint number L. In our superpoint generation layer, we generate L superpoints. We conduct the ablation study to choose a suitable superpoint number. We plot the results of the metrics AS and AR with different

![](images/b7bc1451191deb9e413b8700ce11ca34b069188d9a04e4ef80d648f606fa71a8.jpg)  
Figure 3. The ablation study results (AS and AR) of different hyper-parameters L, K, and T on the KITTI<sub>o</sub> dataset, where $L \in$ {10, 20, 30, 40, 50} and K, T ∈ {1, 2, 3, 4, 5}.

$L \in \{ 1 0 , 2 0 , 3 0 , 4 0 , 5 0 \}$ in Figure 3. It can be observed that choosing L = 30 achieves the best results.

Impact of the K-nearest superpoint centers. To prevent a point from being clustered to a distant superpoint, we only calculate the association map between each point and its K-nearest superpoint centers. Here we explore the impact on the performance of different K. We fix other superparameters and choice $K \in \{ 1 , 2 , 3 , 4 , 5 \}$ . The accuracy results are visualized in Figure 3. Figure 3 shows that our model achieves the best performance with $K = 2$

Number of iterations T. Our model iteratively generates superpoints and conducts the superpoint guided flow refinement. We plot the accuracy results of our model after each iteration. From Figure 3, it can be found that $T = 3$ can obtain state-of-the-art performance. Although $T \ = \ 4 , 5$ can achieve slightly high accuracy, it increases the inference time. Therefore, for a good trade-off between the accuracy and efficiency, we choose $T = 3 .$

## 5. Conclusion

We proposed a novel end-to-end superpoint guided scene flow estimation framework. Different from other offline clustering based scene flow estimation methods, our method can simultaneously optimize the flow guided superpoint generation and superpoint guided flow refinement. Thanks to the joint end-to-end optimization, our model can gradually generate more accurate flow results. Extensive experiments on the synthetic and real LiDAR scenes demonstrate that our self-supervised model can achieve outstanding performance in the scene flow estimation task.

## 6. Acknowledgements

This work was supported by the National Science Foundation of China (Grant Nos. U62276144, U1713208).

## References

[1] Ramy Battrawy, Rene Schuster, Mohammad-Ali Nikouei´ Mahani, and Didier Stricker. Rms-flownet: Efficient and robust multi-scale scene flow estimation for large-scale point clouds. In ICRA, 2022. 3

[2] Stefan Andreas Baur, David Josef Emmerichs, Frank Moosmann, Peter Pinggera, Bjorn Ommer, and Andreas¨ Geiger. Slim: Self-supervised lidar scene flow and motion segmentation. In ICCV, 2021. 1, 6, 7

[3] Aseem Behl, Despoina Paschalidou, Simon Donne, and´ Andreas Geiger. Pointflownet: Learning representations for rigid motion estimation from point clouds. In CVPR, 2019. 2

[4] Mingmei Cheng, Le Hui, Jin Xie, and Jian Yang. Sspcnet: Semi-supervised semantic 3d point cloud segmentation network. In AAAI, 2021. 3

[5] Mingmei Cheng, Le Hui, Jin Xie, Jian Yang, and Hui Kong. Cascaded non-local neural network for point cloud semantic segmentation. In IROS, 2020. 3

[6] Wencan Cheng and Jong Hwan Ko. Bi-pointflownet: Bidirectional learning for point cloud based scene flow estimation. In ECCV, 2022. 2

[7] Guanting Dong, Yueyi Zhang, Hanlin Li, Xiaoyan Sun, and Zhiwei Xiong. Exploiting rigidity constraints for lidar scene flow estimation. In CVPR, pages 12776–12785, 2022. 1, 3, 5

[8] Martin Ester, Hans-Peter Kriegel, Jorg Sander, Xiaowei Xu,¨ et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, 1996. 2, 3

[9] Zan Gojcic, Or Litany, Andreas Wieser, Leonidas J Guibas, and Tolga Birdal. Weakly supervised learning of rigid 3d scene flow. In CVPR, 2021. 1, 3

[10] Paulo FU Gotardo, Tomas Simon, Yaser Sheikh, and Iain Matthews. Photogeometric scene flow for high-detail dynamic 3d reconstruction. In ICCV, 2015. 1

[11] Xiaodong Gu, Chengzhou Tang, Weihao Yuan, Zuozhuo Dai, Siyu Zhu, and Ping Tan. Rcp: Recurrent closest point for point cloud. In CVPR, 2022. 1, 3, 5, 6, 7

[12] Xiuye Gu, Yijie Wang, Chongruo Wu, Yong Jae Lee, and Panqu Wang. Hplflownet: Hierarchical permutohedral lattice flownet for scene flow estimation on large-scale point clouds. In CVPR, 2019. 2, 6, 7

[13] Stephane Guinard and Loic Landrieu. Weakly supervised´ segmentation-aided classification of urban scenes from 3d lidar point clouds. In ISPRS Workshop, 2017. 3

[14] Pan He, Patrick Emami, Sanjay Ranka, and Anand Rangarajan. Learning scene dynamics from point cloud sequences. IJCV, 2022. 2

[15] Pan He, Patrick Emami, Sanjay Ranka, and Anand Rangarajan. Self-supervised robust scene flow estimation via the alignment of probability density functions. In AAAI, 2022. 3, 6, 7

[16] Evan Herbst, Xiaofeng Ren, and Dieter Fox. Rgb-d flow: Dense 3-d motion estimation using color and depth. In ICRA, 2013. 2

[17] Michael Hornacek, Andrew Fitzgibbon, and Carsten Rother. Sphereflow: 6 dof scene flow from rgb-d pairs. In CVPR, 2014. 1

[18] Shengyu Huang, Zan Gojcic, Jiahui Huang, and Konrad Schindler Andreas Wieser. Dynamic 3d scene analysis by point cloud accumulation. In ECCV, 2022. 2

[19] Fred´ eric Huguet and Fr´ ed´ eric Devernay. A variational´ method for scene flow estimation from stereo sequences. In ICCV, 2007. 1, 2

[20] Le Hui, Linghua Tang, Yaqi Shen, Jin Xie, and Jian Yang. Learning superpoint graph cut for 3d instance segmentation. In NeurIPS, 2022. 3

[21] Le Hui, Jia Yuan, Mingmei Cheng, Jin Xie, Xiaoya Zhang, and Jian Yang. Superpoint network for point cloud oversegmentation. In ICCV, 2021. 3, 4, 8

[22] Mariano Jaimez, Mohamed Souiai, Javier Gonzalez-Jimenez, and Daniel Cremers. A primal-dual framework for real-time dense rgb-d scene flow. In ICRA, 2015. 2

[23] Chaokang Jiang, Guangming Wang, Yanzi Miao, and Hesheng Wang. 3d scene flow estimation on pseudo-lidar: Bridging the gap on estimating point motion. TII, 2022. 1

[24] Zhao Jin, Yinjie Lei, Naveed Akhtar, Haifeng Li, and Munawar Hayat. Deformation and correspondence aware unsupervised synthetic-to-real scene flow estimation for point clouds. In CVPR, 2022. 3

[25] Philipp Jund, Chris Sweeney, Nichola Abdo, Zhifeng Chen, and Jonathon Shlens. Scalable scene flow from point clouds in the real world. RA-L, 2021. 2

[26] Yair Kittenplon, Yonina C Eldar, and Dan Raviv. Flowstep3d: Model unrolling for self-supervised scene flow estimation. In CVPR, 2021. 5, 6, 7

[27] Bing Li, Cheng Zheng, Silvio Giancola, and Bernard Ghanem. Sctn: Sparse convolution-transformer network for scene flow estimation. In AAAI, 2022. 2

[28] Bing Li, Cheng Zheng, Guohao Li, and Bernard Ghanem. Learning scene flow in 3d point clouds with noisy pseudo labels. arXiv preprint arXiv:2203.12655, 2022. 3

[29] Ruibo Li, Guosheng Lin, Tong He, Fayao Liu, and Chunhua Shen. Hcrf-flow: Scene flow from point clouds with continuous high-order crfs and position-aware flow embedding. In CVPR, 2021. 2

[30] Ruibo Li, Guosheng Lin, and Lihua Xie. Self-point-flow: Self-supervised scene flow estimation from point clouds with optimal transport and random walk. In CVPR, 2021. 3, 6, 7

[31] Ruibo Li, Chi Zhang, Guosheng Lin, Zhe Wang, and Chunhua Shen. Rigidflow: Self-supervised scene flow learning on point clouds by local rigidity prior. In CVPR, 2022. 2, 3, 6, 7, 8

[32] Yangbin Lin, Cheng Wang, Dawei Zhai, Wei Li, and Jonathan Li. Toward better boundary preserved supervoxel segmentation for 3d point clouds. ISPRS, 2018. 2, 3

[33] Haisong Liu, Tao Lu, Yihui Xu, Jia Liu, Wenjie Li, and Lijun Chen. Camliflow: Bidirectional camera-lidar fusion for joint optical flow and scene flow estimation. In CVPR, 2022. 2

[34] Xingyu Liu, Charles R Qi, and Leonidas J Guibas. Flownet3d: Learning scene flow in 3d point clouds. In CVPR, 2019. 1, 2, 6, 7

[35] Xingyu Liu, Mengyuan Yan, and Jeannette Bohg. Meteornet: Deep learning on dynamic 3d point cloud sequences. In CVPR, 2019. 2

[36] Yawen Lu, Yuhao Zhu, and Guoyu Lu. 3d sceneflownet: Self-supervised 3d scene flow estimation based on graph cnn. In ICIP, 2021. 3

[37] Chenxu Luo, Xiaodong Yang, and Alan Yuille. Selfsupervised pillar motion learning for autonomous driving. In CVPR, 2021. 1, 3

[38] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In CVPR, 2016. 1, 6

[39] Moritz Menze, Christian Heipke, and Andreas Geiger. Joint 3d estimation of vehicles and scene flow. ISPRS, 2015. 6

[40] Moritz Menze, Christian Heipke, and Andreas Geiger. Object scene flow. ISPRS, 2018. 6

[41] Himangi Mittal, Brian Okorn, and David Held. Just go with the flow: Self-supervised scene flow estimation. In CVPR, 2020. 3, 7, 8

[42] Bojun Ouyang and Dan Raviv. Occlusion guided scene flow estimation on 3d point clouds. In CVPRW, 2021. 3

[43] Bojun Ouyang and Dan Raviv. Occlusion guided selfsupervised scene flow estimation on 3d point clouds. In 3DV, 2021. 5

[44] Jhony Kaesemodel Pontes, James Hays, and Simon Lucey. Scene flow from point clouds with or without learning. In 3DV, 2020. 3, 7, 8

[45] Gilles Puy, Alexandre Boulch, and Renaud Marlet. Flot: Scene flow on point clouds guided by optimal transport. In ECCV, 2020. 2, 3, 5, 6, 7

[46] Yukang Shi and Kaisheng Ma. Safit: Segmentation-aware scene flow with improved transformer. In ICRA, 2022. 1

[47] Richard Sinkhorn. A relationship between arbitrary positive matrices and doubly stochastic matrices. JSTOR, 1964. 3

[48] Linghua Tang, Le Hui, and Jin Xie. Learning intersuperpoint affinity for weakly supervised 3d instance segmentation. In ACCV, 2022. 3

[49] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, 2020. 5

[50] Zachary Teed and Jia Deng. Raft-3d: Scene flow using rigidmotion embeddings. In CVPR, 2021. 2

[51] Ivan Tishchenko, Sandro Lombardi, Martin R Oswald, and Marc Pollefeys. Self-supervised learning of non-rigid residual flow and ego-motion. In 3DV, 2020. 3, 6, 7

[52] Sundar Vedula, Simon Baker, Peter Rander, Robert Collins, and Takeo Kanade. Three-dimensional scene flow. In ICCV, 1999. 2

[53] Guangming Wang, Yunzhe Hu, Zhe Liu, Yiyang Zhou, Masayoshi Tomizuka, Wei Zhan, and Hesheng Wang. What matters for 3d scene flow network. In ECCV, 2022. 2

[54] Guangming Wang, Yunzhe Hu, Xinrui Wu, and Hesheng Wang. Residual 3d scene flow learning with context-aware feature extraction. TIM, 2022. 2

[55] Guangming Wang, Xinrui Wu, Zhe Liu, and Hesheng Wang. Hierarchical attention learning of scene flow in 3d point clouds. TIP, 2021. 2

[56] Haiyan Wang, Jiahao Pang, Muhammad A Lodhi, Yingli Tian, and Dong Tian. Festa: Flow estimation via spatialtemporal attention for scene point clouds. In CVPR, 2021. 1, 2

[57] Ke Wang and Shaojie Shen. Estimation and propagation: Scene flow prediction on occluded point clouds. RA-L, 2022. 1

[58] Zirui Wang, Shuda Li, Henry Howard-Jenkins, Victor Prisacariu, and Min Chen. Flownet3d++: Geometric losses for deep scene flow estimation. In CVPR, 2020. 2

[59] Yi Wei, Ziyi Wang, Yongming Rao, Jiwen Lu, and Jie Zhou. Pv-raft: Point-voxel correlation fields for scene flow estimation of point clouds. In CVPR, 2021. 1, 2, 5

[60] Wenxuan Wu, Zhi Yuan Wang, Zhuwen Li, Wei Liu, and Li Fuxin. Pointpwc-net: Cost volume on point clouds for (self-) supervised scene flow estimation. In ECCV, 2020. 1, 3, 6, 7

[61] Victor Zuanazzi, Joris van Vugt, Olaf Booij, and Pascal Mettes. Adversarial self-supervised scene flow estimation. In 3DV, 2020. 3