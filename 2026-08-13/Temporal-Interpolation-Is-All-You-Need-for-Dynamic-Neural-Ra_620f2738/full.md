# Temporal Interpolation Is All You Need for Dynamic Neural Radiance Fields

Sungheon Park, Minjung Son, Seokhwan Jang, Young Chun Ahn, Ji-Yeon Kim, Nahyup Kang Samsung Advanced Institute of Technology (SAIT)

{sh2019.park, minjungs.son, swan.jang, ychun.ahn, jiyeon31.kim, nahyup.kang}@samsung.com

![](images/33aecb869e3959dc01017af6a242b9729b759383bb84318a1ce43800f1436fc2.jpg)  
Figure 1. We propose simple yet effective feature interpolation methods for training neural radiance fields of dynamic scenes based on tem poral interpolation. We provide two different feature vector representations, neural representation (top) and grid representation (bottom), both of which are the concatenation of static feature vectors and temporally-interpolated dynamic feature vectors. The neural representation exhibits high-quality rendering performance with small-sized models while the grid representation shows competitive rendering results with astonishingly fast training speed.

## Abstract

Temporal interpolation often plays a crucial role to learn meaningful representations in dynamic scenes. In this paper, we propose a novel method to train spatiotemporal neural radiance fields of dynamic scenes based on temporal interpolation offeature vectors. Two feature interpolation methods are suggested depending on underlying representations, neural networks or grids. In the neural representation, we extract features from space-time inputs via multiple neural network modules and interpolate them based on time frames. The proposed multi-level feature interpolation network effectively captures features of both short-term and long-term time ranges. In the grid representation, spacetime features are learned via four-dimensional hash grids, which remarkably reduces training time. The grid representation shows more than 100×faster training speed than the previous neural-net-based methods while maintaining

the rendering quality. Concatenating static and dynamic features and adding a simple smoothness term further improve the performance ofour proposed models. Despite the simplicity of the model architectures, our method achieved state-of-the-art performance both in rendering quality for the neural representation and in training speedfor the grid representation.

## 1. Introduction

3D reconstruction and photo-realistic rendering have been long-lasting problems in the fields of computer vision and graphics. Along with the advancements of deep learning, differentiable rendering [11, 14] or neural rendering, has emerged to bridge the gap between the two problems. Recently proposed Neural Radiance Field (NeRF) [18] has finally unleashed the era of neural rendering. Using NeRF, it is able to reconstruct accurate 3D structures from multiple 2D images and to synthesize photo-realistic images from unseen viewpoints. Tiny neural networks are sufficient to save and retrieve complex 3D scenes, which can be trained in a self-supervised manner given 2D images and camera parameters.

Meanwhile, as our world typically involves dynamic changes, it is crucial to reconstruct the captured scene through 4D spatiotemporal space. Since it is often not possible to capture scenes at different viewpoints simultaneously, reconstructing dynamic scenes from images is inherently an under-constrained problem. While NeRF was originally designed to deal with only static scenes, there have been a few approaches in the literature that extend NeRF to dynamic scenes [13,22,23,25], which are so called as dynamic NeRFs. Inspired by the non-rigid structurefrom-motion algorithms [1, 4] that reconstruct 3D structures of deformable objects, most previous works solved the under-constrained setting by estimating scene deformations [21,22,25] or 3D scene flows [5,9,12] for each frame.

However, since the parameters of deformation estimation modules are jointly optimized with NeRF network simultaneously, it is questionable that the modules can accurately estimate deformations or scene flows in accordance with its design principles. In many dynamic scenes, it is challenging to resolve the ambiguities whether a point was newly appeared, moved, or changed its color. It is expected that those ambiguities and deformation estimation can be separately solved within a single network, but in practice, it is hard to let the network implicitly learn the separation, especially for general dynamic scenes without any prior knowledge of deformation.

On the other hand, grid representations in NeRF training [7, 19, 24] have grabbed a lot of attentions mainly due to its fast training speed. Simple trilinear interpolation is enough to fill in the 3D space between grid points. While the representation can be directly adopted to dynamic NeRFs together with the warping estimation module [6], it still requires additional neural networks that affects training and rendering speed.

Motivated by the aforementioned analyses, we present a simple yet effective architecture for training dynamic NeRFs. The key idea in this paper is to apply feature interpolation to the temporal domain instead of using warping functions or 3D scene flows. While the feature interpolation in 2D or 3D space has been thoroughly studied, to the best of our knowledge, feature interpolation method in temporal domain for dynamic NeRF has not been proposed yet. We propose two multi-level feature interpolation methods depending on feature representation which is either neural nets or hash grids [19]. Overview of the two representations, namely the neural representation and the grid representation, are illustrated in Fig. 1. In addition, noting that 3D shapes deform smoothly over time in dynamic scenes, we additionally introduced a simple smoothness term that encourages feature similarity between the adjacent frames. We let the neural networks or the feature grids to learn meaningful representations implicitly without imposing any constraints or restrictions but the smoothness regularizer, which grants the flexibility to deal with various types of deformations. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of the proposed method. We summarized the main contributions of the proposed method as follows:

• We propose a simple yet effective feature extraction network that interpolates two feature vectors along the temporal axis. The proposed interpolation scheme outperforms existing methods without having a deformation or flow estimation module.

• We integrate temporal interpolation into hash-grid representation [19], which remarkably accelerates training speed more than 100× faster compared to the neural network models.

• We propose a smoothness regularizer which effectively improves the overall performance of dynamic NeRFs.

## 2. Related Work

Dynamic NeRF. There have been various attempts to extend NeRF to dynamic scenes. Existing methods can be categorized into three types: warping-based, flow-based, and direct inference from space-time inputs.

Warping-based methods learn how 3D structure of the scene is deformed. 3D radiance field for each frame is warped to single or multiple canonical frames using the estimated deformation. Deformation is parameterized as 3D translation [23], rotation and translation via angle-axis representation [21, 22], or weighted translation [25].

On the other hand, flow-based methods estimate the correspondence of 3d points in consecutive frames. Neural scene flow fields [13] estimates 3D scene flow between the radiance fields of two time stamps. Xian et al. [27] suggested irradiance fields which spans over 4D space-time fields with a few constraints derived from prior knowledge.

Although warping-based and flow-based approaches showed successful results, they have a few shortcomings. For instance, warping-based methods cannot deal with topological variations. Since those methods warp every input frame to a single canonical scene, it is hard to correctly represent newly appeared or disappeared 3D radiance fields in the canonical scene. HyperNeRF [22] learns hyperspace which represents multidimensional canonical shape bases. Neural scene flow fields [13] solve the problem by introducing occlusion masks which are used to estimate the regions where scene flows are not applicable. However, the masks and the flow fields should be learned simultaneously during the training, which makes the training procedure complicated and imposes dependence on extra information such as monocular depth or optical flow estimation.

Without estimating shape deformations or scene flows, DyNeRF [12] used a simple neural network to train NeRFs of dynamic scenes, although the experiments are only conducted on synchronized multi-view videos. It utilizes 3D position and time frame as inputs, and predicts color and density through a series of fully connected neural networks. Our neural representation applies temporal interpolation to intermediate features which enhances representation power for dynamic features while keeping the simple structure.

Category-specific NeRFs are one of the popular research directions for training dynamic objects such as humans [8, 26] or animals [28]. They are based on parametric models or use additional inputs such as pose information. Our method focuses on training NeRFs of general dynamic scenes without any prior knowledge about scenes or objects. Nevertheless, temporal interpolation proposed in this paper can be easily applied to the dynamic NeRFs with templates. Grid representations in NeRF. One of the major drawbacks of the original NeRF [18] is slow training speed. A few concurrent works are proposed to speed up NeRF training. Among them, grid-based representation such as Plenoxels [7], direct voxel grid optimization [24] show superior performance in terms of training time, where it takes only a few minutes to learn plausible radiance fields. Combining grid representation with a hash function further improves the efficiency and training speed of the feature grids [19]. Fang et al. [6] firstly applied a voxel grid representation to train dynamic NeRFs and achieved much faster training speed compared to the neural-network-based methods. Guo et al. [10] also proposed a deformable voxel grid method for fast training. Unlike aforementioned works, our grid representation does not estimate warping or deformation. Instead, it directly estimates color and density using the features obtained from 4D hash grids, which decreases the computational burden and enables faster training.

## 3. Temporal Interpolation for Dynamic NeRF

## 3.1. Preliminaries

Given 3D position $\textbf { x } \in \ \mathbb { R } ^ { 3 }$ and 2D viewing direction d $\in \mathbb { R } ^ { 2 }$ , NeRF [18] aims to estimate volume density $\sigma \in { \mathrm { ~ . ~ } }$ R and emitted RGB color $\mathbf { c } \in \mathbb { R } ^ { 3 }$ using neural networks. We formulate dynamic NeRF as a direct extension from 3D to 4D by adding time frame $t \in \mathbb { R }$ to inputs, i.e.

$$
( \mathbf { c } , \sigma ) = f ( \mathbf { v } ( \mathbf { x } , t ) , \mathbf { d } ) .\tag{1}
$$

$f$ is implemented as fully connected neural networks and a space-time feature representation v can be MLP-based neural nets or explicit grid values as explained in Sec. 3.2.

From the camera with origin o and ray direction d, the color of camera ray $\mathbf { r } ( u ) = \mathbf { o } + \mathbf { \eta }$ ud at time frame t is

$$
\mathbf { C } ( \mathbf { r } , t ) = \int _ { u _ { n } } ^ { u _ { f } } U ( u , t ) \sigma ( \mathbf { v } ) \mathbf { c } ( \mathbf { v } , \mathbf { d } ) d u ,\tag{2}
$$

where $u _ { n }$ and $u _ { f }$ denote the bounds of the scene volume, v is an abbreviation of $\mathbf { v } ( \mathbf { r } ( u ) , t )$ and $\begin{array} { r l } { \boldsymbol { U } ( \boldsymbol { u } , t ) } & { { } = } \end{array}$ $\begin{array} { r } { \exp ( - \int _ { u _ { n } } ^ { u } \sigma ( \mathbf { v } ( \mathbf { r } ( s ) , t ) ) d s ) } \end{array}$ is the accumulated transmittance from $u _ { n }$ to u. Then, the RGB color loss $L _ { c }$ is defined to minimize the $l _ { 2 } .$ -loss between the estimated colors $\hat { C } ( { \bf r } , t )$ and the ground truth colors $C ( \mathbf { r } , t )$ over all rays r in camera views R and time frames $t \in T$ as follows:

$$
L _ { c } = \sum _ { \mathbf { r } \in R , t \in T } | | \hat { C } ( \mathbf { r } , t ) - C ( \mathbf { r } , t ) | | _ { 2 } ^ { 2 } .\tag{3}
$$

## 3.2. Space-Time Feature Representation

The main contribution of our paper lies on the novel feature representation method for dynamic NeRFs. We define the feature vector v, which is fed into the neural network to determine a color and a volume density, as the concatenation of static and dynamic feature vectors, i.e.,

$$
\mathbf { v } ( \mathbf { x } , t ) = [ \mathbf { v } _ { s } ( \mathbf { x } ) , \mathbf { v } _ { d } ( \mathbf { x } , t ) ] ,\tag{4}
$$

where $[ \cdot , \cdot ]$ means concatenation operator of two vectors. The static feature vector ${ \bf v } _ { s }$ only depends on the 3D position x. As most dynamic scenes also contain static regions such as background, ${ \bf v } _ { s }$ is designed to learn the static features that are consistent across time. Meanwhile, $\mathbf { v } _ { d }$ learns dynamic features which may vary across time. We propose two novel feature representation methods in which temporal interpolation is applied. First, the neural representation, which is essentially a series of neural networks combined with linear interpolation, is suggested in Sec. 3.2.1. The model trained using the neural representation is able to render high-quality space-time view synthesis with small-sized neural networks (∼20MB). Second, the grid representation, which is an temporal extension of recently proposed voxel grid representations [19], is explained in Sec. 3.2.2. Dynamic NeRFs can be trained in less than 5 minutes with the proposed grid representation.

## 3.2.1 Neural Representation

In the neural representation, features that fed into the template NeRFs are determined by a series of neural nets. In other words, both static and dynamic feature extractor are formulated as multi-layer perceptrons (MLPs). First, the whole time frame is divided into equally spaced time slots. For each time slot, two MLPs are assigned. Then the whole feature vector is interpolated from the assigned MLPs. Overall feature extraction procedure is illustrated in Fig. 2.

![](images/791450d59be799e9361ddeaf3709e48158f3b1c873cb385e9167baec602b42ce.jpg)  
Figure 2. Illustration of the arrangements of MLPs (left) and the feature extraction (right) in the neural representation. $\phi _ { s }$ and $\phi _ { i j }$ are small MLPs that extract static and dynamic features respectively. The static features $\mathbf { v } _ { s }$ and the dynamic feature $\mathbf { v } _ { d }$ are concatenated and fed into the template NeRF. Detailed structures of the template NeRF can be found in the supplementary materials.

Concretely, let ${ \mathbf v } _ { d } ( { \mathbf x } , { \mathbf z } _ { t } )$ be the feature vector for a 3D point x at time $t ,$ which will be fed to the template NeRF. Here, we used $\mathbf { z } _ { t }$ as an embedding vector for an input time frame t [12, 16, 22]. For equally spaced n time slots, there are n+ 1 keyframes $\begin{array} { r } { t _ { i } = \frac { i } { n } ( i = 0 , 1 , 2 , . . . , n ) } \end{array}$ . An MLP ϕ<sub>i</sub> is assigned to each keyframe $t _ { i }$ which is responsible for two adjacent time slots $[ t _ { i - 1 } , t _ { i } ]$ and $[ t _ { i } , t _ { i + 1 } ]$ . For inputs with time $t \in [ t _ { i } , t _ { i + 1 } ]$ , x and $\mathbf { z } _ { t }$ are fed into $\phi _ { i }$ and $\phi _ { i + 1 }$ . Then, the outputs from two MLPs are interpolated as follows:

$$
\mathbf { v } _ { d } ( \mathbf { x } , \mathbf { z } _ { t } ) = \Delta t \cdot \phi _ { i } ( \mathbf { x } , \mathbf { z } _ { t } ) + ( 1 - \Delta t ) \cdot \phi _ { i + 1 } ( \mathbf { x } , \mathbf { z } _ { t } )\tag{5}
$$

where $\Delta t = ( t _ { i + 1 } - t ) / ( t _ { i + 1 } - t _ { i } )$

The purpose of this interpolation is to efficiently learn the features between keyframes in a scalable manner. Thanks to the temporal interpolation, we can learn the features of the continuous time range $[ t _ { i } , t _ { i + 1 } ]$ by enforcing the MLPs $\phi _ { i }$ and $\phi _ { i + 1 }$ responsible for that time range. While the static feature ${ \bf v } _ { s }$ represents the features across the whole timeline, $\mathbf { v } _ { d }$ represents the features that are focused more on dynamic regions. In addition, since each MLP for dynamic feature is responsible only for two adjacent time slots, it is able to make each $\phi _ { i }$ learn features that are specific to a certain period of time.

To exploit the features of both long-term and short-term, there can be multiple levels of dynamic features which have different number of keyframes. For multi-level dynamic feature extractor with level $l ,$ each level contains different number of keyframes $n _ { 1 } , n _ { 2 } , \cdots , n _ { l }$ . Let $\mathbf { v } _ { d i }$ denote the dynamic feature of i-th level, then the output dynamic feature is the concatenation of features from all levels, $i . e .$

$$
\mathbf { v } _ { d } ( \mathbf { x } , \mathbf { z } _ { t } ) = [ \mathbf { v } _ { d 1 } ( \mathbf { x } , \mathbf { z } _ { t } ) , \mathbf { v } _ { d 2 } ( \mathbf { x } , \mathbf { z } _ { t } ) , \cdot \cdot \cdot \mathbf { \nabla } , \mathbf { v } _ { d l } ( \mathbf { x } , \mathbf { z } _ { t } ) ] .\tag{6}
$$

In this paper, we used the settings $l = 2 , n _ { 1 } = 5 , n _ { 2 } =$ 20 otherwise stated. The dimensions of feature vectors are determined as 128 for $\mathbf { v } _ { s } ,$ and 64 for $\mathbf { v } _ { 0 }$ and $\mathbf { v } _ { 1 }$ . MLP with one hidden layer whose hidden size is the same as its output dimension is used for feature extraction of both ${ \bf v } _ { s }$ and $\mathbf { v } _ { d i }$

The concatenated feature vectors are fed to the template NeRF which outputs volume density $\sigma$ and emitted color $\mathbf { c } .$ We used the same structure as used in HyperNeRF [22] for the template NeRF of the neural representation.

## 3.2.2 Grid Representation

Recently, InstantNGP [19] suggested a novel multi-level grid representation with hash tables. We adopt the hash grid representation from [19] and extend it for fast dynamic NeRF training. Similar to the neural representation in Sec. 3.2.1, the feature vector from the proposed hash grid contains static and dynamic feature vectors. The static feature ${ \bf v } _ { s }$ is analogous to the one used in [19]. On the other hand, the dynamic feature $\mathbf { v } _ { d }$ comes from the 4D hash grids.

Concretely, to extract static and dynamic feature vectors whose dimensions are $m _ { s }$ and $m _ { d }$ respectively, a hash table of size H that contains $( m _ { s } + m _ { d } )$ dimension feature vectors is constructed. The hash function of d-dimensional vector, $h _ { d } ( \mathbf { x } )$ , is defined as

$$
h _ { d } ( { \bf x } ) = \bigoplus _ { i = 1 } ^ { d } ( x _ { i } P _ { i } ) \quad \mathrm { m o d } \ H ,\tag{7}
$$

where $P _ { i }$ is a large prime number and $\oplus$ is an XOR operator. Then, the feature vector v is retrieved by concatenating the outputs of 3D and 4D hash functions:

$$
\mathbf { v } ( \mathbf { x } , t ) = [ h _ { 3 } ( \mathbf { x } ) , h _ { 4 } ( \mathbf { x } , t ) ] .\tag{8}
$$

The 3D and 4D grids are constructed as the multi-level hash grids proposed in [19]. We applied different scaling factors for 3D space and time frame since the number of frames for training sequences are usually much smaller than the finest 3D grid resolutions.

## 3.3. Smoothness Regularization

As our dynamic world smoothly changes, it is reasonable to impose a smoothness term to adjacent time frames.

![](images/968821b8a3792c2644ddb37de57929113d526bc7a8263dc3a3704f84b1b4724d.jpg)  
Figure 3. Illustration of grid representation with hash tables at a certain level. a static feature vector of a 3D grid point is extracted via 3D hash function, while dynamic feature vector for 4D grid point is extracted via 4D hash. The static feature vector $\mathbf { v } _ { s }$ is determined as trilinear interpolation of 8 vectors of grid points, while the dynamic feature vector $\mathbf { v } _ { d }$ is calculated as quadrilinear interpolation of 16 vectors. In the figure, we only depicted feature retrieval process of two grid points for illustration purpose.

We note that the smoothness term is only applied to the input feature space, not to the estimated outputs such as RGB color or density. For the neural representation, we provided a simple smoothness term which is calculated as,

$$
L _ { s } ( \mathbf { x } , t ) = \| \mathbf { v } _ { d } ( \mathbf { x } , \mathbf { z } _ { t } ) - \mathbf { v } _ { d } ( \mathbf { x } , \mathbf { z } _ { t + 1 } ) \| _ { 2 } ^ { 2 } .\tag{9}
$$

This regularization term has two advantages. First, it reflects the intuition that the observed point x at time t will be stationary if there is no observation for x at time t + 1. Second, by propagating additional gradients to the feature networks or the hash grids, the smoothness term acts as a regularizer that stabilizes the training.

For the grid representation, we impose a smoothness term to the grid points that are temporally adjacent:

$$
L _ { s } ( \mathbf { x } , t ) = \frac { 1 } { n _ { f } ^ { 2 } } \| h _ { 4 } ( \mathbf { x } , t _ { a } ) - h _ { 4 } ( \mathbf { x } , t _ { b } ) \| _ { 2 } ^ { 2 } ,\tag{10}
$$

where $n _ { f }$ is the number of frames in the training sequence and $t _ { a } , t _ { b }$ are two adjacent grid points that satisfies $t _ { a } \leq t \leq$ $t _ { b } .$ In fact, Eq. (10) can be obtained from Eq. (9) in the grid representation, which indicates that imposing smoothness term to adjacent time frames is equivalent to add smoothness to two temporally adjacent grids with constant multiplier. Detailed derivation of this relationship is provided in the supplementary materials.

The smoothness term is added to the loss function with a weight λ, and the total loss is minimized together during the training. Accordingly, the total loss of our dynamic neural radiance fields is given by

$$
L = L _ { c } + \lambda L _ { s } .\tag{11}
$$

We observe that the smoothness term is especially powerful in the static background regions that appeared only in a small fraction of frames. In Fig. 4, we show examples of the rendered images which are trained with and without the smoothness term. It can be clearly seen that the boxed regions of the rendered image shows much plausible when the model is trained with the smoothness term.

![](images/4d0d4adb851eaaccde3a35a2f290e20d4a710b7612b77e68add1a5792ced14da.jpg)

![](images/49f8154298785354ba4496e20c2bd9413a97c107d5a2d0ccf2a65d63a1f5f827.jpg)  
(a) w/o smooth term

![](images/7460d4247a679b1a9986e6edb1a115ea68be208f2259b69730a9302b6886a13a.jpg)  
(b) w/ smooth term  
(c) GT  
Figure 4. Effectiveness of the smoothness term. It can be observed that the model trained with the smoothness term accurately renders the corner part which appears in only few frames within the whole sequence (blue box) and removes spurious artifacts (red box).

## 3.4. Implementation Details

In the neural representation, we adopted the template NeRF architecture of HyperNeRF [22] for density and color estimation. The network is an 8-layered MLP with hidden size of 256. The dimension of the embedding vector $\mathbf { z } _ { t }$ is set to 8. The smoothness weight λ is set to 0.01 or 0.001 depending on the characteristics of the datasets used in the experiments. We set λ to a large value in the sequences that the viewpoint change is significant in a short period of time although performance variations depending on the value of λ is not significant.

In the grid representation, λ is set to $1 e ^ { - 4 }$ . We applied the smoothness loss only for the finest two levels of the temporal dimension since applying it to every level slows training speed without much improvements in performance. After feature vectors are extracted from the hash grids, we fed them to a 3-layered MLP with hidden size of 128 to estimate volume density followed by one additional layer for RGB color estimation similar to [19]. We set $m _ { s } = 2 , m _ { d } = 6$ and $H = 2 ^ { 1 9 }$ for the grid representation experiments otherwise stated. The 3D and 4D hash grids are composed of 12 levels. For the spatial dimension, we set the base resolution to 8, and the scaling factor is set to 1.45. The base resolution of the temporal dimension is 2, which is multiplied by 1.4 in every other level. Detailed hyperparameter settings and network architectures can be found in the supplementary materials.

## 4. Experimental Results

## 4.1. Datasets

To validate the superiority of our method, we conducted extensive experiments on various datasets. We used three publicly available datasets that are used for dynamic NeRFs. For all experiments, we trained our models for each sequence individually, and then per-scene results are averaged and reported in this section.

D-NeRF Dataset [23]. The dataset consists of synthetically rendered images of moving and deforming 3D objects. For each frame, the image is rendered via a synthetic camera of random rotation. There are eight scenes in total, and each scene contains 50-200 training images with 20 test views.

HyperNeRF Dataset [22]. The dataset contains video sequences taken from mobile phones. There are four sequences in vrig-dataset which are taken using the camera rig with two phones vertically aligned. In addition, there are six sequences in interp-dataset which are taken from a single camera in order to estimate the image in the the middle of two consecutive training images.

DyNeRF Dataset [12]. The dataset consists of videos obtained from a capture system that consists of 21 GoPro Black Hero cameras. The cameras are located at a fixed position, and all video frames are synchronized to build the multi-view video dataset.

## 4.2. Neural Representation

In this section, we reported the performance of the proposed neural representation models. First, the experimental results on D-NeRF dataset are shown in Tab. 1. Peak signalto-noise ratio (PSNR), structural similarity (SSIM), and perceptual similarity (LPIPS) [29] are used as evaluation metrics following the previous works. We also reported the average metric (AVG) proposed in [3] which aggregates three metrics to a single value. Our method with the neural representation (Ours-NN) achieves state-of-the-art results on all evaluation metrics. It is worth noting that the smoothness term dramatically improves overall performance.

Next, we evaluated our method on HyperNeRF dataset [22] and compared with existing methods, which is shown in Tab. 2. Here, we reported PSNR and multiscale SSIM (MS-SSIM), and excluded LPIPS metric since its value cannot reliably reproduced [6]. Our method shows second-best results on both vrig and interp datasets. While flow-based method [13] suffers from interpolating motions between two consecutive frames, our method, which implicitly learns intermediate feature representation of inbetween frames, achieves competitive performance with warping-based methods [21, 22].

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PSNR↑</td><td rowspan=1 colspan=1>SSIM↑</td><td rowspan=1 colspan=1>LPIPS↓</td><td rowspan=1 colspan=1>AVG↓</td></tr><tr><td rowspan=4 colspan=1>NeRF [18]T-NeRF [23]D-NeRF [23]NDVG-full [10]TiNeuVox-B [6]</td><td rowspan=1 colspan=1>19.00</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.09</td></tr><tr><td rowspan=1 colspan=1>29.50</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=2 colspan=1>30.4327.8432.67</td><td rowspan=1 colspan=1>0.950.862</td><td rowspan=1 colspan=1>0.070.041</td><td rowspan=2 colspan=1>0.020.0290.016</td></tr><tr><td rowspan=1 colspan=1>0.971</td><td rowspan=1 colspan=1>0.041</td></tr><tr><td rowspan=2 colspan=1>Ours-NN (w/o smooth)Ours-NN (w/ smooth)</td><td rowspan=2 colspan=1>30.1832.73</td><td rowspan=1 colspan=1>0.963</td><td rowspan=1 colspan=1>0.038</td><td rowspan=2 colspan=1>0.0190.014</td></tr><tr><td rowspan=1 colspan=1>0.974</td><td rowspan=1 colspan=1>0.033</td></tr></table>

Table 1. Experimental results on D-NeRF [23] datasets.

Qualitative results for the neural representation are shown in Fig. 5. It is clearly observed that our method captures fine details compared to D-NeRF [23]. We highlighted the regions that show notable differences with colored boxes and placed zoomed-in images of the regions next to the rendered images. In addition, we also qualitatively compared our method with HyperNeRF [22] on their datasets. When the warping estimation module of [22] does not correctly estimate warping parameters, HyperNeRF produces implausible results. It can be observed that the head and the body of the chicken toy is not properly interlocked and the position of the hand peeling banana is incorrect. On the other hand, our method accurately recovers 3D geometry of those scenes. Thus, without using the physically mean ingful warping estimation module in neural networks, the proposed temporal interpolation and the smoothness regularizer provides simple yet effective way to learn complex deformations.

Lastly, performance evaluation on DyNeRF dataset [12] is presented in Tab. 3. We adopt PSNR, LPIPS, and FLIP [2] for evaluation metrics to compare with previous works. Our method achieves best PSNR while ranked second in LPIPS and FLIP. However, those two metrics are also better than DyNeRF<sup>†</sup> [12] which does not use importance sampling strategy in [12]. Since we do not use any sampling strategy during training, it can be concluded that our feature interpolation method is superior to the network architecture of [12]. Notably, our method outperforms DyNeRF with smaller network size (20MB) than the DyNeRF models (28MB).

## 4.3. Grid Representation

We used D-NeRF datasets to evaluate the performance of the proposed grid representation. Since this representation is mainly intended for fast training, we report the results in a short period time (∼8 minutes) and compare the results with the concurrent works [6, 10], both of which are based on voxel grid representation and showed the fastest training speed for dynamic NeRFs so far. We also examined the performance of the original implementation of InstantNGP [19] as a baseline with no temporal extension. All of the grid representation models in the experiments are trained on a single RTX 3090 GPU for fair comparison with [6, 10].

![](images/296809f5b69e0d2587046a2246ef79d29add794c6669d73c5f54f89c6d6ef4ba.jpg)  
D-NeRF [23]

![](images/d4632b8961c688735f7eeadf9f077f9231d4b0ff50c9d4622294cfbd3d2340bf.jpg)  
Ours-NN

![](images/40e84d1850207f4e4a859ecdd9222349dfdb61f15a164cd7e09ac063c2f28c4e.jpg)  
Ground truth

![](images/b2dfe87f107f6a5964ba8acdbc9533a420a25829fb6a2b8385154bfc287442b8.jpg)  
HyperNeRF [22]

Ours-NN  
![](images/29a12a9c7bea7c541b1f44dfd4ed9acc20e37902b44eb1f3751f917eb02db5da.jpg)  
Ground truth

Figure 5. Qualitative results of our method (Ours-NN) on of D-NeRF [23] and HyperNeRF [22] datasets. The regions that show significant difference with compared methods are scaled up (red and green boxes).
<table><tr><td rowspan="2"></td><td colspan="2">vrig</td><td colspan="2">interp</td></tr><tr><td>PSNR↑</td><td>MS-SSIM↑</td><td>PSNR↑</td><td>MS-SSIM↑</td></tr><tr><td>NeRF [18]</td><td>20.13</td><td>0.745</td><td>22.27</td><td>0.804</td></tr><tr><td>NV [15]</td><td>16.85</td><td>0.571</td><td>26.05</td><td>0.911</td></tr><tr><td>NSFF [13]</td><td>26.33</td><td>0.916</td><td>25.80</td><td>0.883</td></tr><tr><td>Nerfies [21]</td><td>22.23</td><td>0.803</td><td>28.47</td><td>0.939</td></tr><tr><td>HyperNeRF [22]</td><td>22.38</td><td>0.814</td><td>29.00</td><td>0.945</td></tr><tr><td>Ours-NN</td><td>24.35</td><td>0.867</td><td>28.67</td><td>0.940</td></tr></table>

Table 2. Experimental results on HyperNeRF [22] datasets.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Train time</td><td rowspan=1 colspan=1>PSNR</td><td rowspan=1 colspan=1>SSIM</td><td rowspan=1 colspan=1>LPIPS</td><td rowspan=1 colspan=1>AVG</td></tr><tr><td rowspan=1 colspan=1>InstantNGP [19]</td><td rowspan=1 colspan=1>5 min</td><td rowspan=1 colspan=1>20.28</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.146</td><td rowspan=1 colspan=1>0.077</td></tr><tr><td rowspan=1 colspan=1>D-NeRF [23]</td><td rowspan=1 colspan=1>20 hours</td><td rowspan=1 colspan=1>30.43</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>NDVG-full [10]Tinue Vox-B [6]</td><td rowspan=1 colspan=1>35 min30 min</td><td rowspan=1 colspan=1>27.8432.67</td><td rowspan=1 colspan=1>0.8620.971</td><td rowspan=2 colspan=1>0.0410.0410.048</td><td rowspan=1 colspan=1>0.0290.016</td></tr><tr><td rowspan=4 colspan=1>NDVG-half [10]TinueVox-S [6]</td><td rowspan=1 colspan=1>23 min</td><td rowspan=1 colspan=1>27.15</td><td rowspan=1 colspan=1>0.857</td><td></td></tr><tr><td></td><td></td><td></td><td rowspan=3 colspan=1>0.067</td><td></td></tr><tr><td></td><td></td><td></td><td rowspan=2 colspan=1>0.0330.023</td></tr><tr><td rowspan=1 colspan=1>8min</td><td rowspan=1 colspan=1>30.75</td><td rowspan=1 colspan=1>0.956</td></tr><tr><td rowspan=1 colspan=1>Ours-grid</td><td rowspan=1 colspan=1>1 min</td><td rowspan=1 colspan=1>26.77</td><td rowspan=1 colspan=1>0.933</td><td rowspan=3 colspan=1>0.1070.0630.062</td><td rowspan=3 colspan=1>0.0390.0240.023</td></tr><tr><td rowspan=1 colspan=1>Ours-grid</td><td rowspan=1 colspan=1>5 min</td><td rowspan=1 colspan=1>29.73</td><td rowspan=1 colspan=1>0.961</td></tr><tr><td rowspan=1 colspan=1>Ours-grid</td><td rowspan=1 colspan=1>8 min</td><td rowspan=1 colspan=1>29.84</td><td rowspan=1 colspan=1>0.962</td></tr></table>

<table><tr><td></td><td>PSNR↑</td><td>LPIPS↓</td><td>FLIP↓</td></tr><tr><td>MVS NeuralVolumes [15] LLFF [17] NeRF-T [12]</td><td>19.12 22.80 23.24 28.45</td><td>0.2599 0.2951 0.2346 0.100</td><td>0.2542 0.2049 0.1867 0.1415</td></tr><tr><td>DyNeRF† [12] DyNeRF [12]</td><td>28.50 29.58</td><td>0.0985 0.0832</td><td>0.1455 0.1347</td></tr><tr><td>Ours-NN</td><td>29.88</td><td>0.0960</td><td>0.1413</td></tr></table>

Table 3. Experimental results on DyNeRF [12] datasets.

The quantitative results are shown with elapsed training time in Tab. 4. The grid representation demonstrates much faster training speed even compared to the recent voxelgrid based methods [6, 10]. In comparison with neural network models [23], our method exceeds SSIM and LPIPS of [23] in 5 minutes which have arithmetically 240 times faster training speed. By taking the benefits of fully fused neural network [20] as well as the efficient hash grid representation, our method quickly learns meaningful features in dynamic scenes and generates acceptable novel view synthesis results with just one minute of training. Our method shows superior SSIM, LPIPS and slightly inferior PSNR to TinueVox-S [6], which indicates clearer and more detailed rendering results, when trained for the same period of time.

Table 4. Quantitative comparison of training time and performance on D-NeRF datasets using the grid representation.
<table><tr><td rowspan="2"></td><td colspan="3">D-NeRF dataset</td><td colspan="2">HyperNeRF vrig</td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>MS-SSIM</td></tr><tr><td>NN (dynamic only)</td><td>30.91</td><td>0.963</td><td>0.043</td><td>24.07</td><td>0.861</td></tr><tr><td>NN (dynamic+static)</td><td>32.73</td><td>0.974</td><td>0.033</td><td>24.35</td><td>0.866</td></tr><tr><td>Grid (dynamic only)</td><td>29.08</td><td>0.948</td><td>0.076</td><td>22.11</td><td>0.755</td></tr><tr><td>Grid (dynamic+static)</td><td>29.84</td><td>0.962</td><td>0.062</td><td>22.98</td><td>0.802</td></tr></table>

Table 5. Ablation study on the effectiveness of static features.

We depict training progress of the grid representation qualitatively in Fig. 6. We also compared the results of TinueVox-S [6] which are trained for 8 minutes. After one minute of training, the grid representation model produces blurry images but accurately render dynamic regions compared to InstantNGP [19]. After 5 minutes, the model successfully renders sharp images that are similar to the ground truth images. Notably, our model exhibits more sharp and clear results than TinueVox-S despite shorter training time, which results in better SSIM and LPIPS..

Finally, we compared the performance of the grid representation by varying the dimension of static and dynamic feature vectors, m<sub>s</sub> and $m _ { d } .$ . PSNR and LPIPS on test images are measured per minute and illustrated in Fig. 7. We trained the grid representation models in three different settings, $m _ { s } ~ = ~ 2 , m _ { d } ~ = ~ 6 , ~ m _ { s } ~ = ~ 2 , m _ { d } ~ = ~ 2 ,$ $m _ { s } = 4 , m _ { d } = 4$ . When smaller feature dimension is used, the training speed is faster, so PSNR increases fast in the early stage of training. However, the performance also saturates faster, and LPIPS is inferior to the other settings. The model with $m _ { s } = 2 , m _ { d } = 6$ significantly outperformed the others in terms of LPIPS.

![](images/957b7a8e60306bdc71e7c69479665cf99b3f0280a7bfc7cfbd5ba3b29d032a78.jpg)  
InstantNGP [19]

![](images/21e4b039d967ab0e114cfbbcf281762ff107fde9015c83828cea0de112cd0531.jpg)  
(8min)

![](images/351f024ea5984cc8b02dfc8ef2dfb4012c41472f93005a6f7dfa1c7f9f55e16f.jpg)  
Ours-grid (1min)

![](images/19519523835de24a3401cd6d78a88a4337ef79db88694a82fdc017d82b01080d.jpg)  
Ours-grid (3min)

![](images/4273cfb54feab1460574d7588517b1be04b0377efb7910607a650f5ca638a150.jpg)  
Ours-grid (5min)

![](images/29ef3515018d3dd4e7021492e749250749e90326bf37f3e0fd6a03147fd8a67d.jpg)  
Ground truth

Figure 6. Qualitative results of the grid representation. While InstantNGP [19] is not able to reconstruct dynamic regions, our method produces acceptable rendering results in one minute and performs superior to TiNueVox-S [6] with shorter training time.  
![](images/2532a1121a08e2516bbb36f944a73a7ea857988f015bc0224963ad5488822f16.jpg)

![](images/be64b06624ddbdc1a06f29c8779a8b6824c726cb8962e367b07b725cbaa02758.jpg)  
Ours-NN

![](images/dfba790565bd9027040f5f1b78a911c93c8f89f28f68a1f0385fba06b9cfb514.jpg)  
Ground truth

Figure 7. Performance comparison with different feature dimension settings. Using smaller feature dimensions results in faster training but quicker saturation of model performance.  
![](images/0e3e92a86ee6bf35e31b54b1edc5883d38b64a04beb07d58ee4fd979e67dca5f.jpg)  
Ours-grid

To validate the effectiveness of the static features, we conducted an ablation study on the static features and illustrated the results in Tab. 5. For both the neural representation (NN) and the grid representation (Grid), the models with static features performs superior to the ones using only dynamic features in all metrics. Additional ablation studies and qualitative results can be found in the supplementary materials.

## 4.5. Failure Cases

Although the proposed feature interpolation is able to learn meaningful spatiotemporal features in most cases,

## 4.4. Effectiveness of the Static Features

![](images/670189e342133830a91527c2132eceaa4b1258c5a531a583ef99df573487b9a5.jpg)  
Ground truth

![](images/adc0e2edb66f21b306d834115ed99cea3812abeeea3b5a22cb04bc16f59d9892.jpg)  
Figure 8. Examples of failure cases.

there are a few failure cases as presented in Fig. 8. For instance, our method has difficulty in recovering 3D structures when small objects in a sequence rapidly move (Fig. 8 left) or when there exist dynamic regions that are not observed in the training sequence (Fig. 8 right).

## 5. Conclusion

In this paper, we propose a simple yet effective feature interpolation method for training dynamic NeRFs. Both the neural representation and the grid representation showed impressive performance, and the smoothness term applied to the intermediate feature vectors further improves the performance. Since these methods are unrelated to the existing methods of modeling deformations or estimating scene flows, we believe that the proposed method suggests a new direction of training dynamic NeRFs.

While the neural representation model shows highquality rendering results owing to the representation power of neural networks, it requires hours of training and seconds of rendering which impose barriers to real-time applications. On the other hand, the grid representation is able to render dynamic scenes in less than a second after a few minutes of training, which makes it more practical for real-world applications. Both representations are mutually complementary, and investigating hybrid representations that take advantages of both representations would be an interesting research direction.

## References

[1] Ijaz Akhter, Yaser Sheikh, Sohaib Khan, and Takeo Kanade. Trajectory space: A dual representation for nonrigid structure from motion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(7):1442–1456, 2010. 2

[2] Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,¨ Magnus Oskarsson, Kalle Astr<sup>˚</sup> om, and Mark D. Fairchild.¨ FLIP: A Difference Evaluator for Alternating Images. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 3(2):15:1–15:23, 2020. 6

[3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5855–5864, 2021. 6

[4] Christoph Bregler, Aaron Hertzmann, and Henning Biermann. Recovering non-rigid 3d shape from image streams. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition. CVPR 2000 (Cat. No. PR00662), volume 2, pages 690–696. IEEE, 2000. 2

[5] Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenenbaum, and Jiajun Wu. Neural radiance flow for 4d view synthesis and video processing. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 14304–14314. IEEE Computer Society, 2021. 2

[6] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. arXiv preprint arXiv:2205.15285, 2022. 2, 3, 6, 7, 8

[7] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5501–5510, 2022. 2, 3

[8] Guy Gafni, Justus Thies, Michael Zollhofer, and Matthias Nießner. Dynamic neural radiance fields for monocular 4d facial avatar reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8649–8658, 2021. 3

[9] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5712–5721, 2021. 2

[10] Xiang Guo, Guanying Chen, Yuchao Dai, Xiaoqing Ye, Jiadai Sun, Xiao Tan, and Errui Ding. Neural deformable voxel grid for fast optimization of dynamic view synthesis. In Proceedings of the Asian Conference on Computer Vision (ACCV), 2022. 3, 6, 7

[11] Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada. Neural 3d mesh renderer. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3907– 3916, 2018. 1

[12] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5521–5531, 2022. 2, 3, 4, 6, 7

[13] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6498– 6508, 2021. 2, 6, 7

[14] Shichen Liu, Tianye Li, Weikai Chen, and Hao Li. Soft rasterizer: A differentiable renderer for image-based 3d reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7708–7717, 2019. 1

[15] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural volumes: Learning dynamic renderable volumes from images. ACM Trans. Graph., 38(4):65:1–65:14, July 2019. 7

[16] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7210–7219, 2021. 4

[17] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 38(4):1–14, 2019. 7

[18] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1):99–106, 2021. 1, 3, 6, 7

[19] Thomas Muller, Alex Evans, Christoph Schied, and Alexan-¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1– 102:15, July 2022. 2, 3, 4, 6, 7, 8

[20] Thomas Muller, Fabrice Rousselle, Jan Nov¨ ak, and Alexan-´ der Keller. Real-time neural radiance caching for path tracing. ACM Transactions on Graphics (TOG), 40(4):1–16, 2021. 7

[21] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5865–5874, 2021. 2, 6, 7

[22] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021. 2, 4, 5, 6, 7

[23] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318–10327, 2021. 2, 6, 7

[24] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields

reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5459– 5469, 2022. 2, 3

[25] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhofer, Christoph Lassner, and Christian Theobalt. Non-¨ rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12959–12970, 2021. 2

[26] Chung-Yi Weng, Brian Curless, Pratul P. Srinivasan, Jonathan T. Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16210–16220, June 2022. 3

[27] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time neural irradiance fields for free-viewpoint video. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9421–9431, 2021. 2

[28] Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, and Hanbyul Joo. Banmo: Building animatable 3d neural models from many casual videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2863–2873, 2022. 3

[29] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6