# SFD2: Semantic-guided Feature Detection and Description<sup>E</sup>

Fei Xue Ignas Budvytis Roberto Cipolla University of Cambridge {fx221, ib255, rc10001}@cam.ac.uk

## Abstract

Visual localization is a fundamental task for various applications including autonomous driving and robotics. Prior methods focus on extracting large amounts of often redundant locally reliable features, resulting in limited efficiency and accuracy, especially in large-scale environments under challenging conditions. Instead, we propose to extract globally reliablefeatures by implicitly embedding high-level semantics into both the detection and description processes. Specifically, our semantic-aware detector is able to detect keypoints from reliable regions (e.g. building, traffic lane) and suppress unreliable areas (e.g. sky, car) implicitly instead ofrelying on explicit semantic labels. This boosts the accuracy of keypoint matching by reducing the number offeatures sensitive to appearance changes and avoiding the need of additional segmentation networks at test time. Moreover, our descriptors are augmented with semantics and have stronger discriminative ability, providing more inliers at test time. Particularly, experiments on long-term large-scale visual localization Aachen Day-Night and RobotCar-Seasons datasets demonstrate that our model outperforms previous local features and gives competitive accuracy to advanced matchers but is about 2 and 3 times faster when using 2k and 4k keypoints, respectively. Code is available at https://github.com/feixue94/sfd2.

## 1. Introduction

Visual localization is key to various applications including autonomous driving and robotics. Structure-based algorithms [54, 57, 64, 69, 73, 79] involving mapping and localization processes still dominate in large-scale localization. Traditionally, handcrafted features (e.g. SIFT [3, 35], ORB [53]) are widely used. However, these features are mainly based on statistics of gradients of local patches and thus are prone to appearance changes such as illumination and season variations in the long-term visual localization task. With the success of CNNs, learning-based features [14, 16, 37, 45, 51, 76, 81] are introduced to replace handcrafted ones and have achieved excellent performance.

![](images/154665bd8c55b6bd9b663f8516b0af09441782f740e13c24582889ed2c4b0eeb.jpg)  
Figure 1. Overview of our framework. Our model implicitly incorporates semantics into the detection and description processes with guidance of an off-the-shelf segmentation network during the training process. Semantic- and feature-aware guidance are adopted to enhance its ability of embedding semantic information.

With massive data for training, these methods should be able to automatically extract keypoints from more reliable regions (e.g. building, traffic lane) by focusing on discriminative features [64]. Nevertheless, due to the lack of explicit semantic signals for training, their ability of selecting globally reliable keypoints is limited, as shown in Fig. 2 (detailed analysis is provided in Sec. B.1 in the supplementary material). Therefore, they prefer to extract locally reliable features from objects including those which are not useful for long-term localization (e.g. sky, tree, car), leading to limited accuracy, as demonstrated in Table 2.

Recently, advanced matchers based on sparse keypoints [8, 55, 65] or dense pixels [9, 18, 19, 33, 40, 47, 67, 74] are proposed to enhance keypoint/pixel-wise matching and have obtained remarkable accuracy. Yet, they have quadratic time complexity due to the attention and correlation volume computation. Moreover, advanced matchers rely on spatial connections of keypoints and perform imagewise matching as opposed to fast point-wise matching, so they take much longer time than nearest neighbor matching (NN) in both mapping and localization processes because of a large number of image pairs (much larger than the number of images) [8]. Alternatively, some works leverage semantics [64, 73, 79] to filter unstable features to eliminate wrong correspondences and report close even better accuracy than advanced matchers [79]. However, they require additional segmentation networks to provide semantic labels at test time and are fragile to segmentation errors.

Instead, we implicitly incorporate semantics into a local feature model, allowing it to extract robust features automatically from a single network in an end-to-end fashion. In the training process, as shown in Fig. 1, we provide explicit semantics as supervision to guide the detection and description behaviors. Specifically, in the detection process, unlike most previous methods [14, 16, 32, 37, 51] adopting exhaustive detection, we employ a semantic-aware detection loss to encourage our detector to favor features from reliable objects (e.g. building, traffic lane) and suppress those from unreliable objects (e.g. sky). In the description process, rather than utilizing triplet loss widely used for descriptor learning [16, 41], we employ a semantic-aware description loss consisting of two terms: inter- and intra-class losses. The inter-class loss embeds semantics into descriptors by enforcing features with the same label to be close and those with different labels to be far. The intra-class loss, which is a soft-ranking loss [23], operates on features in each class independently and differentiates these features from objects of the same label. Such use of soft-ranking loss avoids the conflict with inter-class loss and retains the diversity of features in each class (e.g. features from buildings usually have larger diversity than those from traffic lights). With semantic-aware descriptor loss, our model is capable of producing descriptors with stronger discriminative ability. Benefiting from implicit semantic embedding, our method avoids using additional segmentation networks at test time and is less fragile to segmentation errors.

As the local feature network is much simpler than typical segmentation networks e.g. UperNet [10], we also adopt an additional feature-consistency loss on the encoder to enhance its ability of learning semantic information. To avoid using costly to obtain ground-truth labels, we train our model with outputs of an off-the-shelf segmentation network [11, 34], which has achieved SOTA performance on the scene parsing task [83], but other semantic segmentation networks (e.g. [10]) can also be used.

An overview of our system is shown in Fig. 1. We embed semantics implicitly into the feature detection and description network via the feature-aware and semantic-aware guidance in the training process. At test time, our model produces semantic-aware features from a single network directly. We summarize contributions as follows:

• We propose a novel feature network which implicitly incorporates semantics into detection and description processes at training time, enabling the model to produce semantic-aware features end-to-end at test time.

• We adopt a combination of semantic-aware and feature-aware guidance strategy to make the model embed semantic information more effectively.

![](images/2cb7abdcbb30ed982a79532b8665281798106fa9bf92c923d1faf86131d033c0.jpg)  
SPP D2Net R2D2 ASLFeatFigure 2. Locally reliable features. We show top 1k keypoints (reliability high→low: 1-250 , 251-500 , 501-750 751-1000 of prior local features including SPP [14], pedestrian<sub>D2Net [16], R2D2 [51], and ASLFeat [37]. They indiscrimina-</sub> tively give high reliability to patches with rich textures even from objects e.g. sky, tree, pedestrian and car, which are reliable for car<sub>long-term localization (best view in color).</sub>

• Our method outperforms previous local features on the long-term localization task and gives competitive accuracy to advanced matchers but has higher efficiency.

Experiments show our method achieves a better tradeoff between accuracy and efficiency than advanced matchers [8, 55, 65] especially on devices with limited computing resources. We organize the rest of this paper as follows. In Sec. 2, we introduce related works. In Sec. 3, we describe our method in detail. We discuss experiments and limitations in Sec. 4 and Sec. 5 and conclude the paper in Sec. 6.

## 2. Related Work

In this section, we discuss related work on visual localization, feature extraction and matching, and knowledge distillation.

Visual localization. Visual localization methods can be roughly categorized as image-based and structure-based. Image-based systems recover camera poses by finding the most similar one in the database with global features, e.g. NetVLAD [2], CRN [27]. Due to the limited number of images in the database, they can only give approximate poses. To obtain more precise poses, structure-based methods build a sparse 3D map via SfM and estimate the pose via PnP from 2D-3D correspondences [12,54,57,58,69,79]. Some other works have tried to predict the camera pose directly from images, e.g. PoseNet [28] and its variations [80], or regress scene coordinates [5–7,26]. However, the former have been proved to perform similar to image retrieval [61] and latter are hard to scale to large-scale scenes [31].

Local features. Handcrafted features [3, 35, 53] have been investigated for decades and we refer readers a survey [38] for more details and focus on learned features. With the success of CNNs, learned features are proposed to replace handcrafted descriptors [15,17,36,41–43,49,71,72], detectors [13,68,70], or both [14,16,32,37,51,76,81]. Hard-Net [41] focuses on metric learning by maximizing the distance between the closest positive and negative examples. Instead of using pixel-wise correspondences for training, CAPS [77], PoSFeat [32] and PUMP [50] utilize camera pose and local consistency of matches for supervision. SuperPoint (SPP) [14] takes keypoint detection as a supervised task, training detector from synthetic geometric shapes. D2- Net [16] uses local maxima across the channels as score map. R2D2 [51] considers both the repeatability and reliability and adopts the average precision loss [23] for descriptor training. ASLFeat [37] employs deformable CNNs to learn shape-aware dense features. As they focus mainly on local reliability of features, regardless of their superior accuracy to handcrafted features, their performance is limited in the long-term large-scale localization task. To further improve the accuracy, some works [22,46,75] learn to filter unstable keypoints with extra matching score, repeatability or semantic labels. Essentially different with these methods, our model detects and extracts semantic-aware features automatically in an end-to-end fashion. As a result, our features are able to produce more accurate localization results.

Advanced matcher. As NN matching is unable to incorporate spatial connections of keypoints for matching, advanced matchers are proposed to enhance the accuracy by leveraging the spatial context of a set of keyppoints [8, 55, 65] or an image patch [9, 18, 33, 52, 67, 84]. SuperGlue (SPG) [55] utilizes graph neural networks with attention mechanism to propagate information among keypoints. It produces impressive accuracy, whereas its time complexity is quadratic to the number of keypoints. This problem is partially mitigated by using seeded matching [8] and cluster matching [65], but the time is still thousands of times slower than NN matching. Dense matchers [9, 33, 52, 67] compute pixel-wise correspondence from correlation volumes, so they undergo the high time and memory cost as sparse matchers [8, 55, 65]. Moreover, advanced matchers operates on image pairs as opposed to keypoints, so considering the number of image pairs, systems with advanced matchers could be much slower in real applications, as analyzed in [8]. In this paper, we embed high-level semantic information into local features implicitly to enhance both feature detection and description, enabling our model with simple NN matching to yield comparable results to advanced matchers. Our work provides a good trade-off of time and accuracy especially on devices with limited computing resources.

Visual semantic localization. Compared to local features, high-level semantics are more robust to appearance changes and have been widely used in visual localization [7, 25, 26, 30, 31, 44, 62–64, 66, 73, 78]. LLN [78] and SVL [63] use the discriminative landmarks for place recognition. ToDayGAN [1] transfers night images to day images with GAN [20]. MFC [30], SMC [73], SSM [64], and DASGIL [25] incorporate segmentation networks into a standard localization pipeline to reject semanticallyinconsistent matches. More recently, LBR [79] learns to recognize global instances for both coarse and fine localization. In fine localization, it filters unstable features and conducts instance-wise matching, achieving close accuracy to advanced matchers [55]. Unlike these methods, which require additional models to provide explicit semantic labels at test time, we embed the semantic information into the network and produce semantic-aware features directly from a single network.

Knowledge distillation. Knowledge distillation techniques have been widely used for tasks including model compression [54] and knowledge transfer [82]. Our usage of pseudo ground-truth local reliability and semantic labels predicted by off-the-shelf networks is more like a knowledge transfer task. In this paper, we focus mainly on how to effectively leverage the high-level semantics for low-level feature extraction.

## 3. Method

As shown in Fig. 1, our model consists of an encoder $\mathcal { F } _ { e n c }$ and two decoders $\mathcal { F } _ { d e t }$ and $\mathcal { F } _ { d e s c } . \ F _ { e n c }$ extracts highlevel features X from image $\mathbf { I } \in R ^ { 3 \times H \times W } . ~ \mathcal { F } _ { d e t }$ predicts the reliability map $\mathbf { S } \in R ^ { \tilde { H } \times W }$ and $\mathcal { F } _ { d e s c }$ produces descriptors $\mathbf { X } _ { d e s c } \doteq R ^ { \dot { 1 } 2 8 \times \frac { H } { 4 } \times \frac { W } { 4 } }$ . H and W are the height and width of the image. In this section, we give details about how to implicitly incorporate semantic information into our feature detection and description processes.

## 3.1. Semantic-guided Feature Detection

The detector predicts the reliability map as ${ \textbf { S } } =$ $\mathcal { F } _ { d e t } ( \mathbf { X } )$ . Previously, the reliability map S is defined by the richness of textures in patches (e.g. response to corners [14] or blobs [35]). Recently, learned local features [16, 32, 37, 51] define the reliability on the discriminative ability of descriptors. As shown in Fig. 2, these two definitions, however, only reveal the reliability of pixels at a local level but lack the stability at a global level. Instead, we redefine the reliability of features by taking both the local reliability ${ \bf { S } } _ { r e l }$ and global stability $\mathbf { S } _ { s t a }$ into consideration.

Local reliability. Local reliability shows the robustness of a keypoint to appearance changes and viewpoint variations. Previous learning-based features adopt two strategies for reliable feature learning: learning from groundtruth corners [14] and learning from the discriminative ability of descriptors [16, 32, 37, 51]. We find that corners [14] are more robust compared to purely learned detectors, as shown in [50,79], where SPP detector achieves better results when replacing other detectors. Therefore, following [54], we use the detection score ${ \bf { S } } _ { r e l }$ of SPP [14] as pseudo groundtruth, which is one of the best corner detectors. At the same time, local reliability is slightly adjusted by the discriminative ability of descriptors in the training process (see Sec. 3.2).

![](images/ce2e522596afe52eae46b9727c563cd689c7f7951bae3dbf199184888e763c73.jpg)  
Figure 3. Semantic-guided feature detection. From left to right: semantic segmentation mask predicted by [11, 34], stability map $\mathbf { S } _ { s t a }$ generated according to Table 1, local reliability map ${ \bf { S } } _ { r e l }$ produced by SuperPoint [14], and the final global reliability map S. Local reliability map gives very high score to clouds (red), trees (green), and pedestrians (pink) in addition to buildings, while the global reliability map removes unstable regions (sky, pedestrians), suppresses short-term objects (trees), and retains stable areas (buildings).

Global stability. The global stability of a pixel is assigned based on the semantic label which it belongs to. Specifically, we group all 120 semantic labels in ADE20k dataset [83], according to how they change over time, into four categories, denoted as Volatile, Dynamic, Short-term, and Long-term in Table 1. Volatile objects (e.g. sky, water) are constantly changing and are redundant for localization. Dynamic objects (e.g. car, pedestrian) are moving everyday and could cause localization error by introducing wrong matches. Short-term objects (e.g. tree) can be used for short-term localization tasks (e.g. VO/SLAM), yet they are sensitive to changes of illumination (low albedo) and season conditions. Long-term objects (e.g. building, traffic light) are resistant to aforementioned changes and are ideal for long-term localization.

Instead of directly filtering unstable features [64,73,79], we rerank features with stability values assigned empirically according to the extent of desired suppression. In detail, Long-term objects are robust for both short and longterm localization, so their stability value is set to 1.0. Shortterm objects are useful for short-term localization, so we set their stability to 0.5. The stability value of Volatile and Dynamic categories is set to 0.1 as they are not useful for both short/long-term localization. Note that we set it to 0.1 rather than 0. Our reranking strategy encourages the model to use stable features preferentially and uses keypoints from other objects as compensation when insufficient stable keypoints can be found, increasing the robustness of our model to various tasks (e.g. feature matching, short-term localization). Fig. 3 shows stability map $\mathbf { S } _ { s t a }$ transformed from Table 1. Our current global stability is assigned based on predefined semantic labels, but a learned one might provide better performance and deserves further exploration.

Semantic-guided detection. The global reliability map $\mathbf { S } _ { g t }$ is generated by multiplying the local reliability map ${ \bf { S } } _ { r e l }$ and global stability map, as ${ \bf S } _ { g t } = { \bf S } _ { r e l } \odot { \bf S } _ { s t a }$ ( is element-wise multiplication). Fig. 3 shows that local reliability map gives high score for all pixels with rich textures even those from the sky, pedestrians, and trees, which are useless for localization. However, the global reliability map considering both local reliability and global stability discards these sensitive features and suppresses short-term keypoints effectively. The detection loss is defined as:

<table><tr><td>Category</td><td>Volatile Dynamic Short-term Long-term Stability</td><td></td><td></td></tr><tr><td>sky, water</td><td>√</td><td rowspan="4"></td><td>0.1</td></tr><tr><td>vehicle, pedestrian</td><td>√</td><td>0.1</td></tr><tr><td>plant, grass</td><td></td><td>0.5</td></tr><tr><td>building, traffic light</td><td>√</td><td>√ 1.0</td></tr></table>

Table 1. Stability map. Semantic labels are categorized into four groups denoted as Volatile, Dynamic, Short-term, and Long-term. Four categories are empirically assigned with different stability values according to their robustness to appearance changes.

$$
{ \cal L } _ { d e t } = B C E ( { \bf S } , { \bf S } _ { g t } ) ,\tag{1}
$$

where BCE denotes the binary cross-entropy loss.

## 3.2. Semantic-guided Feature Description

We also enhance the discriminative ability of descriptors by embedding semantics into them directly. Unlike previous descriptors [14, 37, 41, 50, 51, 77], which only differentiate keypoints based on local patch information, our descriptors enforce similarities of features in the same class while retain dissimilarities for intra-class matching. However, the two forces conflict with each other during the training process, because class-level discriminative ability needs to squeeze the space of descriptors in the same class and intra-class discriminative ability has to increase the space. A simple solution could be to set a hard margin for all classes (Fig. 4 left), but it would lead to the loss of objects inner diversity (e.g., almost all traffic lights are similar but different buildings vary dramatically), which is indispensable for intra-class matching. To solve this problem, we design the inter-class and intra-class losses based on two different metrics (Fig. 4 right).

![](images/31a4b9c6b3a1ddb8ca0d16c88e8e6931db4bffd2471b4e580a93d66227138452.jpg)  
Figure 4. Semantic-guided feature description. Simply optimizing inter- and intra-class losses with hard margins may cause accuracy loss due to two conflicting forces (push forces of inter- and intra-class) (left). Instead, we optimize the intra-class force with a hard margin, but apply a soft ranking loss for inter-class force to avoid conflicts and retain the inner diversity of each object (right).

Inter-class loss. We first enforce the semantic consistency of features by maximizing the Euclidean distance between descriptors with different labels. This allows features to find correspondences from candidates with the same labels, reducing the search space and thus improving the matching accuracy. We define the inter-class loss based on triplet loss with a hard margin to separate all possible positive and negative keypoints with different labels in a batch:

$$
L _ { i n t e r } = \frac { 1 } { N } \sum ( | | \mathbf { x } _ { i } ^ { c _ { 1 } } - \mathbf { x } _ { j } ^ { c _ { 1 } } | | _ { 2 } - | | \mathbf { x } _ { i } ^ { c _ { 1 } } - \mathbf { x } _ { k } ^ { c _ { 2 } } | | _ { 2 } + m ) ,\tag{2}
$$

where $\mathbf { x } _ { i } ^ { c _ { 1 } } , \mathbf { x } _ { j } ^ { c _ { 1 } }$ , and $\mathbf { x } _ { k } ^ { c _ { 2 } }$ are vectorized descriptors with labels of $c _ { 1 } , c _ { 1 }$ , and $c _ { 2 } \left( c _ { 1 } \neq c _ { 2 } \right)$ . m is the margin and set to 1.0. This loss is conducted on features in the whole batch and N is the total number of features in a batch.

Intra-class loss. To make sure that the intra-class loss doesn’t conflict with the inter-class loss, we relax the limitation of distances between descriptors with the same label. Instead of using triplet loss with hard margins, we adopt a soft ranking loss [23] by optimizing the rank of positive and negative samples rather than their distances. We use the same strategy as [51] to generate positive and negative samples for each feature $\mathbf { x } _ { i } ^ { c }$ from self and the other images respectively, but enforce both positive and negative samples to share the same class label c as $\mathbf { x } _ { i } ^ { c } .$ . By optimizing ranks of all samples rather than forcing a hard boundary between positive and negative pairs as the triplet loss with a hard margin does, the soft ranking loss also retains the diversity of features on objects in the same class, as shown in Fig. 4 (right). The ranking loss is based on the averaging precision (AP) loss [23, 51]:

$$
L _ { i n t r a } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } ( 1 - A P ( \mathbf { x } _ { i } ^ { c } , \mathbf { S } ^ { \mathbf { x } _ { i } ^ { c } } ) ) ,\tag{3}
$$

where $\mathbf { x } _ { i } ^ { c }$ and $\mathbf { S } ^ { \mathbf { x } _ { i } ^ { c } }$ are the query descriptor with label c and corresponding predicted local reliability. C and $N _ { c }$ are the total number of classes and features in class c. Note that the AP loss for sample $\mathbf { x } _ { i } ^ { c }$ is weighted by its reliability $\mathbf { S } ^ { \mathbf { x } _ { i } ^ { c } }$

![](images/c21d6cdc42c2c6b956ab6b9eecce8a601dfd7c6cbf9118e0f1fcd24febe2222d.jpg)  
Figure 5. Architecture of our network. Features are 4× downsampling to save time and memory cost. Resblocks [24] are adopted to enhance the model’s capacity. We enforce the consistency between outputs of the middle two layers of our encoder and features of the segmentation network to enhance the ability of our model to embed semantic information during the training process.

Our final descriptor loss $L _ { d e s c }$ is the combination of $L _ { i n t e r }$ and $\boldsymbol { L } _ { i n t r a }$ , balanced by $w _ { i n t e r }$ and $w _ { i n t r a }$

$$
L _ { d e s c } = w _ { i n t e r } L _ { i n t e r } + w _ { i n t r a } L _ { i n t r a } .\tag{4}
$$

## 3.3. Implicit Feature Guidance

With semantic-aware detection and description losses, our model is able to learn semantic-aware features. However, compared with feature learning, semantic prediction is a more complicated task, requiring powerful encoders and aggregation layers [4, 34] for semantic-aware feature embedding. To further improve the ability of our model to embed semantic information, we take inspiration from current knowledge distillation tasks [21] and introduce a feature consistency loss on intermediate outputs of the first three layers of the encoder.

Fig. 5 shows the architecture of our network. We take intermediate outputs of the encoder of ConvNeXt [34] as supervision signal and enforce $l _ { 1 }$ loss on features of the middle 2 layers of our model:

$$
{ \cal L } _ { f e a t } = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { 2 } | { \bf X } _ { i } - { \bf X } _ { i } ^ { C o n v N e X t } | ,\tag{5}
$$

where $\mathbf { X } _ { i }$ and ${ \bf X } _ { i } ^ { C o n v N e X t }$ are features of the ith layer in our model and ConvNeXt [34], respectively. The total loss $L _ { t o t a l }$ is the combination of detection loss $L _ { d e t } .$ , description loss $L _ { d e s c } { \mathrm { . } }$ , and feature consistency loss $L _ { f e a t }$ with weights of $w _ { d e t } , w _ { d e s c } ,$ and $w _ { f e a t } .$

$$
L _ { t o t a l } = w _ { d e t } L _ { d e t } + w _ { d e s c } L _ { d e s c } + w _ { f e a t } L _ { f e a t } .\tag{6}
$$

## 4. Experiments

We first give implementation details. Then, we test our method on visual localization tasks in Sec. 4.1 and analyze the running time in Sec. 4.2. Finally, we perform an ablation study in Sec. 4.3. More implementation details, results and analysis are provided in the supplementary material.

<table><tr><td>Group</td><td>Method</td><td>Day</td><td>Night  $( 2 ^ { \circ } , 0 . 2 \bar { 5 } m ) / ( 5 ^ { \circ } , 0 . 5 m ) / ( \bar { 1 } 0 ^ { \circ } , 5 m )$ </td></tr><tr><td rowspan="4">C</td><td>AS_v1.1 [57]</td><td>85.3 / 92.2 / 97.9</td><td>39.8 / 49.0 / 64.3</td></tr><tr><td>CSL [69]</td><td>52.3 / 80.0 / 94.3</td><td>29.6 / 40.8 / 56.1</td></tr><tr><td>CPF [12]</td><td>76.7 / 88.6 / 95.8</td><td>33.7 / 48.0 / 62.2</td></tr><tr><td>Ours</td><td>88.2 / 96.0 / 98.7</td><td>87.8 / 94.9 / 100.0</td></tr><tr><td rowspan="5">S</td><td>SSM [64]</td><td>71.8 / 91.5 / 96.8</td><td>58.2 / 76.5 / 90.8</td></tr><tr><td>VLM [78]</td><td>62.4 / 71.8 / 79.9</td><td>35.7 / 44.9 / 54.1</td></tr><tr><td>SMC [73]</td><td>52.3 / 80.0 / 94.3</td><td>29.6 / 40.8 / 56.1</td></tr><tr><td>LBR [79]</td><td>88.3 / 95.6 / 98.8</td><td>84.7 / 93.9 / 100.0</td></tr><tr><td>Ours</td><td>88.2 / 96.0 / 98.7</td><td>87.8 / 94.9 / 100.0</td></tr><tr><td rowspan="17">L</td><td>SIFT [35] SPP [14]</td><td>82.8 / 88.1 / 93.1 80.5 / 87.4 / 94.2</td><td>30.6 / 43.9 / 58.2 42.9 / 62.2 / 76.5</td></tr><tr><td>D2Net [16] R2D2 [51]</td><td>84.8 / 92.6 / 97.5</td><td>84.7 / 90.8 / 96.9</td></tr><tr><td>SIFT+CAPS [35,77]</td><td></td><td>76.5 / 90.8 / 100.0 77.6 / 86.7 / 99.0</td></tr><tr><td>SPP+CAPS [14,77]</td><td></td><td>82.7 / 87.8 / 100.0</td></tr><tr><td>SPP+LISRD [14,49]</td><td></td><td>78.6 / 86.7 / 98.0</td></tr><tr><td>SPP+PUMP [14,50]</td><td></td><td>74.4 / 88.0 / 98.4</td></tr><tr><td></td><td>R2D2+PUMP [14,50] R2D2+LLF [14,68]</td><td></td><td>73.3 / 86.9 / 98.4</td></tr><tr><td></td><td>SOSNet+D2D [70,72]</td><td></td><td>72.4 / 90.8 / 99.0 73.5 / 83.7 / 96.9</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>PoSFeat [32]</td><td></td><td>81.6 / 90.8 / 100.0</td></tr><tr><td></td><td>ASLFeat [37]</td><td></td><td>81.6 / 87.8 / 100.0</td></tr><tr><td></td><td>Ours</td><td>88.2 / 96.0 / 98.7</td><td>87.8 / 94.9 / 100.0</td></tr><tr><td></td><td>ENCNet [52]</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>76.5 / 84.7 / 98.0</td></tr><tr><td></td><td>Dual-RCNet [33]</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>79.6 / 88.8 / 100.0</td></tr><tr><td>M</td><td>PDCNet [74]</td><td></td><td>80.6 / 87.8 / 100.0</td></tr><tr><td></td><td>DGCNet [40]</td><td>22.9 / 49.8 / 84.7</td><td>14.3 / 37.8 / 79.6</td></tr><tr><td></td><td>Pixloc [56]</td><td>84.7 / 94.2 / 98.8</td><td>81.6 / 93.9 / 100.0</td></tr><tr><td></td><td>AHM [18]</td><td>47.8 / 72.2 / 91.3</td><td>30.6 / 53.1 / 78.6</td></tr><tr><td></td><td>S2DNet [19]</td><td>84.5 / 90.3 / 95.3</td><td>74.5 / 82.7 / 94.9</td></tr><tr><td></td><td>Patch2Pix [84]</td><td>84.6 / 92.1 / 96.5</td><td>82.7 / 92.9 / 99.0</td></tr><tr><td></td><td></td><td>89.6 / 95.4 / 98.8</td><td></td></tr><tr><td></td><td>SPP+SPG [14,55]</td><td></td><td>86.7 / 93.9 / 100.0</td></tr><tr><td></td><td>SPP+SGMNet [8, 14]</td><td>86.8 / 94.2 / 97.7</td><td>83.7 / 91.8 / 99.0</td></tr><tr><td></td><td>SPP+ClusterGNN [14,65]</td><td>89.4 / 95.5 / 98.5</td><td>81.6 / 93.9 / 100.0</td></tr><tr><td>Ours</td><td></td><td>88.2 / 96.0 / 98.7</td><td>87.8 / 94.9 / 100.0</td></tr></table>

Table 2. Results on Aachen dataset [59,60]. The best and second best results are highlighted with bold and red fonts.

Implementation. We use the identical training dataset as R2D2 [51]. The training dataset consists of reference images in Aachen v1.0 dataset [60] and web images. As R2D2 [51] and LBR [79], training images are augmented with style transfer. To mitigate segmentation uncertainties caused by style transfer, semantic labels of stylized images are generated from their corresponding normal images. The network is implemented in PyTorch [48] and trained using Adam [29] optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9$ , batch size of 4, weight decay of 4 $\times 1 0 ^ { - 4 }$ on a single 3090Ti GPU for 40 epochs. The hyper-parameter $w _ { i n t r a }$ is set to 0.5, while $w _ { i n t e r } , w _ { d e t } , w _ { d e s c } ,$ and $w _ { f e a t }$ are set to 1.0.

## 4.1. Long-term Large-scale Localization

We test our method on Aachen (v1.0 and v1.1) [59, 60] and RobotCar-Seasons (RoCaS) [39, 60] datasets under various illumination, season, and weather conditions. Aachen v1.0 contains 4,328 reference and 922 (824 day, 98 night) query images captured around the Aachen city center. Aachen v1.1 expands v1.0 by adding 2,369 reference and 93 night query images. RoCaS has 26,121 reference and 11,934 query images. It is challenging because of various conditions of day query images (rain, snow, dusk, winter) and poor lighting of night query images in suburban areas. We adopt the success ratio at error thresholds of (2<sup>◦</sup>, 0.25m), (5<sup>◦</sup>, 0.5m), (10<sup>◦</sup>, 5m) as metric. We additionally provide results on Extended CMU-Seasons dataset in the supplementary material.

<table><tr><td>Group</td><td>Method</td><td>Day</td><td>Night  $( 2 ^ { \circ } , 0 . 2 \bar { 5 } m ) / ( 5 ^ { \circ } , 0 . 5 m ) / \breve { ( 1 0 ^ { \circ } , 5 m ) }$ </td></tr><tr><td rowspan="2">S</td><td>LBR [79]</td><td>89.1 / 96.1 / 99.3</td><td>77.0 / 90.1 / 99.5</td></tr><tr><td>Ours</td><td>88.2 / 96.0 / 98.7</td><td>78.0 / 92.1 / 99.5</td></tr><tr><td rowspan="8">H</td><td>SIFT [35]</td><td>72.2 / 78.4 / 81.7</td><td>19.4 / 23.0 / 27.2</td></tr><tr><td>SPP [14]</td><td>87.9 / 93.6 / 96.8</td><td>70.2 / 84.8 / 93.7</td></tr><tr><td>D2Net [16]</td><td>84.1 / 91.0 / 95.5</td><td>63.4 / 83.8 / 92.1</td></tr><tr><td>R2D2 [51]</td><td>88.8 / 95.3 / 97.8</td><td>72.3 / 88.5 / 94.2</td></tr><tr><td>ASLFeat [37]</td><td>88.0 / 95.4 / 98.2</td><td>70.7 / 84.3 / 94.2</td></tr><tr><td>CAPS+SIFT [35,77]</td><td>82.4 / 91.3 / 95.9</td><td>61.3 / 83.8 / 95.3</td></tr><tr><td>LISRD+SPP [14, 49]</td><td></td><td>73.3 / 86.9 / 97.9</td></tr><tr><td>LLF+R2D2 [14, 68]</td><td></td><td>71.2 / 81.2 / 94.2</td></tr><tr><td rowspan="5"></td><td>PoSFeat [32]</td><td></td><td>73.8 / 87.4 / 98.4</td></tr><tr><td>Ours</td><td>88.2 / 96.0 / 98.7</td><td>78.0 / 92.1 / 99.5</td></tr><tr><td>SPP+SGMNet [8,14]</td><td>88.7 / 96.2 / 98.9</td><td>75.9 / 89.0 / 99.0</td></tr><tr><td>SPP+SPG [14,55]</td><td>89.8 / 96.1 / 99.4</td><td>77.0 / 90.6 / 100.0</td></tr><tr><td>Patch2Pix [84]</td><td>86.4 / 93.0 / 97.5</td><td>72.3 / 88.5 / 97.9</td></tr><tr><td rowspan="4"></td><td>LoFTER [67]</td><td>88.7 / 95.6 / 99.0</td><td>78.5 / 90.6 / 99.0</td></tr><tr><td>ASpanFormer [9]</td><td>89.4 / 95.6 / 99.0</td><td>77.5 / 91.6 / 99.5</td></tr><tr><td>Ours</td><td>88.2 / 96.0 / 98.7</td><td>78.0 / 92.1 / 99.5</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 3. Results on Aachen v1.1 dataset [59, 60]. The best and second best results are highlighted with bold and red fonts.

Baselines. Baselines include classic systems (C) e.g. AS v1.1 [57], CSL [69], and CPF [12] and methods using semantics (S), e.g. LLN [78], SMC [73], SSM [64], DAS-GIL [25], ToDayGAN [1] and LBR [79]. We also compare our model with learned features [14, 16, 37, 50, 51, 77] (L). As prior methods [14, 16, 37, 50, 51, 77], we use HLoc [54] pipeline for reconstruction and mutual nearest matching (MNN). NetVLAD [2] is used to offer 50 and 20 candidates for Aachen and RoCaS datasets, respectively. We additionally compare our approach with advanced sparse/dense matchers (M) e.g., Superglue (SPG) [55], SGMNet [8], ClusterGNN [65] and ASpanFormer [9], LoFTER [67], Patch2Pix [84], Dual-RCNet [33]. Their results are obtained from the visual benchmark<sup>1</sup> or original papers.

Comparison with classic methods (C). As shown in Table 2 and 4, our model outperforms all classic methods. As most these methods use SIFT [35], they are more sensitive to weather and illumination changes than learned features.

Comparison with methods using explicit semantics (S). By leveraging semantic labels to filter potentially wrong matches, these models achieve better performance for day and night images in Table 2 and 4 but require segmentation results at test time. Our model outperforms all other approaches (except LBR [79]). LBR [79] reports excellent accuracy by selecting keypoints from buildings and performing instance-wise matching. Our method gives close results to LBR on day images but better performance on most night images, because our model does not require explicit semantic labels and is less fragile to segmentation errors especially for night images. LBR performs better than ours on night-train images in Table 4 because it is trained on augmented night rainy images, while our model is trained only on generated night images as R2D2.

<table><tr><td rowspan="2"></td><td rowspan="2">Group Method</td><td rowspan="2">day</td><td>night</td><td>night-rain</td></tr><tr><td> $( 2 ^ { \circ } , 0 . 2 5 m ) / ( 5 ^ { \circ } , \mathsf { \bar { 0 } } . 5 m ) / ( 1 0 ^ { \circ } , \mathsf { \bar { 5 } } m )$ </td><td></td></tr><tr><td rowspan="5">C</td><td>AS [57]</td><td>43.6 / 76.0 / 94.0</td><td>1.6 / 3.9 / 10.5</td><td>2.0 / 10.9 / 18.0</td></tr><tr><td>CSL [69]</td><td>45.3 / 73.5 / 90.1</td><td>0.2 / 0.9 / 5.3</td><td>0.9 / 4.3 / 9.1</td></tr><tr><td>CPF [12]</td><td>48.0 / 78.0 / 94.2</td><td>2.3 / 6.6 / 15.3</td><td>4.5 / 12.3 / 18.6</td></tr><tr><td>Ours</td><td>56.9 / 81.6 / 97.4</td><td>27.6 / 66.2 / 90.2</td><td>43.0 / 71.1 / 90.0</td></tr><tr><td>SSM [64]</td><td></td><td></td><td>14.5 / 33.2 / 47.5</td></tr><tr><td rowspan="6">S</td><td>VLM [78]</td><td>54.5 / 81.6 / 96.7 7.9 / 30.0 / 85.9</td><td>10.0 / 23.7 / 45.4</td><td></td></tr><tr><td>SMC [73]</td><td></td><td>11.9 / 26.0 / 55.0</td><td>15.7 / 34.5 / 60.5</td></tr><tr><td></td><td>50.3 / 79.3 / 95.2</td><td>6.2 / 18.5 / 44.3</td><td>8.0 / 26.4 / 46.4</td></tr><tr><td>DASGIL-FD [25]</td><td>8.7 / 30.7 / 81.3</td><td>1.6 / 4.8 / 19.9</td><td>1.8 / 4.3 / 21.6</td></tr><tr><td>ToDayGAN [1, 16]</td><td>52.2 / 80.1 / 95.9</td><td>16.4 / 43.2 / 73.3</td><td>24.1 / 50.5 / 74.1</td></tr><tr><td>LBR [79] Ours</td><td>56.7 / 81.7 / 98.2 56.9 / 81.6 / 97.4</td><td>24.9 / 62.3 / 86.1 27.6 / 66.2 / 90.2</td><td>47.5 / 73.4 / 90.0 43.0 / 71.1 / 90.0</td></tr><tr><td rowspan="6">L</td><td>SIFT [35]</td><td>53.5 / 77.6 / 92.6</td><td>7.8 / 13.9 / 22.1</td><td>9.5 / 14.5 / 17.0</td></tr><tr><td>SPP [14]</td><td>56.5 / 81.5 / 97.1</td><td>16.9 / 41.6 / 71.5</td><td>22.0 / 45.0 / 68.0</td></tr><tr><td>D2Net [16]</td><td>54.5 / 80.0 / 95.3</td><td>18.0 / 39.7 / 53.9</td><td>22.7 / 40.5 / 56.1</td></tr><tr><td>R2D2 [51]</td><td>57.4 / 81.9 / 97.9</td><td>18.3 / 43.4 / 67.8</td><td>29.1 / 50.2 / 68.2</td></tr><tr><td>CAPS [77]</td><td>56.0 / 81.5 / 96.5</td><td>21.9 / 54.3 / 86.8</td><td>27.0 / 58.9 / 85.9</td></tr><tr><td rowspan="4"></td><td>ASLFeat [37]</td><td>57.1 / 81.9 / 98.4</td><td>23.5 / 55.9 / 80.1</td><td>41.1 / 66.8 / 86.1</td></tr><tr><td>Ours</td><td>56.9 / 81.6 / 97.4</td><td>27.6 / 66.2 / 90.2</td><td>43.0 / 71.1 / 90.0</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>SPP+SPG [14,55]</td><td>56.9 / 81.7 / 98.1</td><td>24.2 / 62.6 / 87.4</td><td>42.3 / 69.3 / 90.2</td></tr><tr><td rowspan="4">M</td><td>Pixloc [56]</td><td>56.9 / 82.0 / 98.1</td><td>24.2 / 62.8 / 88.4</td><td>45.5 / 72.5 / 90.7</td></tr><tr><td>AHM [18]</td><td>45.7 / 78.0 / 95.1</td><td>16.2 / 55.3 / 93.6</td><td>28.4 / 68.4 / 95.5</td></tr><tr><td>Ours</td><td>56.9 / 81.6 / 97.4</td><td>27.6 / 66.2 / 90.2</td><td></td></tr><tr><td></td><td></td><td></td><td>43.0 / 71.1 / 90.0</td></tr></table>

Table 4. Results on RobotCar-Seasons dataset [39,59]. The best and second best results are highlighted with bold and red fonts.

Comparison with learned features (L). Benefiting from training with massive data, learned features such as R2D2 [51], ASLFeat [37] and PoSFeat [32], outperform SIFT [3, 35]. As they extract keypoints indiscriminately, they are still more sensitive to appearance changes especially on night images than semantic-aware methods [79], as shown in Table 2 and 4. Our model extracts semanticaware features directly, so it gives higher accuracy.

Comparison with advanced matchers (M). In Table 2, 3 and 4, we also show the results of previous advanced matchers. We find that our approach outperforms the recent efficient variations of SPG [55] (e.g. SGMNet [8], ClusterGNN [65]) and gives competitive results to SPG, which achieves the best accuracy and is also the slowest method. Note that our model only uses the simple MNN for matching. We provide a detailed analysis of time and memory usage in Sec. 4.2 and argue that our method provides a good trade off between running time and accuracy especially on devices with limited computing resources.

Robustness to the number of keypoints. We observe that most previous methods [16, 32, 37, 50, 51] extract keypoints with the number ranging from 10k (R2D2, ASLFeat)

![](images/16495ff9f193fa5ac054e1cef3fbd42e2c19a7ef03465ea18d6c2e10e8780930.jpg)

![](images/9e92f9441ed2ff6d82d0911bf1370027f0b811b8f3d33f4e06bcc669ef42a521.jpg)  
Figure 6. Influence of the number of keypoints. We report results of different number (from 4k to 1k) of keypoints on Aachen v1.1 [59, 60] at error threshold of (2<sup>◦</sup>, 0.25m).

to 40k (PosFeat) for evaluation. Although increasing the number improves the accuracy, it causes the time cost as well, which should be taken into consideration especially on devices with limited computing resources. In this experiment, we test the ability of previous and our methods of extracting fewer but more robust features by reducing the number of keypoints from 4k to 1k. Note that results of [14, 51, 55, 79] are from LBR [79] and results of ASLFeat [37] are obtained by running official source code.

Fig. 6 shows that as the number of keypoints decreases, all previous features [14, 37, 51] undergo dramatic accuracy loss especially for night images. SPP+SPG and LBR perform more robust because of the global context or semantics. With implicitly embedded semantics, our feature outperforms R2D2, ASLFeat, and SPP especially on night images and gives competitive results to SPP+SPG and LBR.

Qualitative comparison. Fig. 7 shows the detection and matching results of query images under conditions of large illumination and season changes. Compared with prior features [14, 37, 51], which prefer keypoints from areas with rich textures, our method favors keypoints from objects robust for long-term localization (e.g. buildings). When insufficient keypoints can be found from stale regions, our model also uses keypoints from Short-term objects e.g. trees from compensation but assigns them with lower reliability. Besides, our feature gives more inliers for query images with large occlusions of trees and severe illumination changes.

## 4.2. Running time analysis

Table 5 demonstrates the test time of previous features [14, 37, 51, 79], matchers [8, 55], and our method. Our method (33.2ms) is faster than R2D2 (72.4ms) [51] and slower than SPP (13.1ms) [14], but has higher accuracy. Besides, our method is faster than LBR (9.2+30.1ms) [79], which uses explicit instances to filter keypoints and advanced matchers including SPP+SPG (13.1+52.2/146.5ms) [14,55] and SPP+SGM (13.1+33.2/97.6ms) [8, 14]. As the matching method is extensively used for each image pair in mapping and localization processes, SPG/SGMNet are about 18.3/3.3 times slower than NN on Aachen dataset [59, 60] in the mapping process as discussed in [8]. Therefore, our approach could be a good trade off between accuracy and efficiency.

![](images/82d48058b9eef4716c1cddc943d40bb8f2b7152a970695ab30f4ed6000809870.jpg)

![](images/7ae70f9e96c78ca240c18882ef6efe5c521510df6e37b2d633122a6f15b4f3e6.jpg)

![](images/b81f8d59108086f9097e4df862d05919692e8000fecab8c8f0c65865ccc96cab.jpg)

![](images/c4626e0fef2f4291655db8f5608162e0cffdcc762dafd665a0fd012b2c3d42cc.jpg)  
Figure 7. Qualitative comparison of detection and matching. Left→right: SPP [14], R2D2 [51], ASLFeat [37] and our method. Our model favors keypoints on stable areas (reliability high→low: 1-250 , 251-500 , 501-750 , 751-1000 ) and gives more inliers.

<table><tr><td>Model</td><td>Input size</td><td>Running time (ms)</td></tr><tr><td>SPP [14]</td><td>1024×1024</td><td>13.1</td></tr><tr><td>R2D2 [51]</td><td>1024×1024</td><td>72.4</td></tr><tr><td>ASLFeat* [37]</td><td>1024×1024</td><td>112.3</td></tr><tr><td>LBR (feature) [79]</td><td>1024×1024</td><td>30.1</td></tr><tr><td>LBR (segmentation) [79]</td><td>256×256</td><td>9.2</td></tr><tr><td>SPG [55]</td><td>2k×2k,4k×4k</td><td>52.2, 146.5</td></tr><tr><td>SGMNet [8]</td><td>2k×2k,4k×4k</td><td>85.5,97.6</td></tr><tr><td>Ours</td><td>1024×1024</td><td>33.2</td></tr></table>

Table 5. Running time. We report test time of prior features [14, 37, 51], matchers [8, 55] and our method (\*in TensorFlow).

## 4.3. Ablation study

In Table 6, we verify the effectiveness of all components in our network by progressively adding the semantic detection (SD), description (SS), and feature consistency (SF) losses. We also compare results of SS loss with triplet and ranking as intra-class loss. Our baseline is trained with detection scores of SPP [14] as detector supervision and ap [23] loss for descriptor learning. After adding SD loss, the model performs better especially for night images. The accuracy is further improved by introducing SS loss because it augments the discriminative ability of descriptors with semantics. Compared to triplet loss with carefully tuned margin, ranking loss improves more accuracy as objects’ inner diversity can be better retained by optimizing ranks of samples. SF loss enhances the network’s ability of embedding semantic information, leading to further improvements.

## 5. Limitations

The first limitation is the hand-defined stability values. A learned stability map from training data could be more robust and further improve the localization accuracy. Besides, semantic labels used in the paper are from ADE20K [83] and the number of these labels is limited. Fine grained semantic labels [31] from automatic segmentation might be more reliable in real applications. Moreover, this work focuses mainly on outdoor localization and may not work very well in indoor scenarios due to the significant differences of object classes. Better performance for indoor scenes can be achieved by retraining the model with redefined stability map for indoor objects.

<table><tr><td>SD</td><td>SS</td><td>SF</td><td>Day</td><td>Night  $( 2 ^ { \circ } , 0 . 2 \bar { 5 } m ) / ( 5 ^ { \circ } , 0 . 5 m ) / ( \tilde { 1 0 ^ { \circ } } , 5 m )$ </td></tr><tr><td>x</td><td>x</td><td>x</td><td>85.4 / 93.6 / 97.9</td><td>71.2 / 84.3 / 98.4</td></tr><tr><td>√</td><td>x</td><td>x</td><td>87.3 / 94.3 / 97.8</td><td>72.8 / 88.5 / 99.5</td></tr><tr><td>x</td><td>√(triplet)</td><td>x</td><td>87.9 / 95.1 / 98.9</td><td>73.8 / 86.9 / 99.0</td></tr><tr><td>x</td><td>√(ranking)</td><td>x</td><td>87.9 / 95.3 / 98.7</td><td>74.9 / 89.5 / 99.0</td></tr><tr><td>√</td><td>√</td><td>x</td><td>88.2 / 95.5 / 98.8</td><td>75.9 / 89.0 / 99.5</td></tr><tr><td>√</td><td>√</td><td>√</td><td>88.2 / 96.0 / 98.7</td><td>78.0 / 92.1 / 99.5</td></tr></table>

Table 6. Ablation study. We test the efficacy of semantic detection (SD), semantic description (SS), and semantic feature consistency (SF) losses. The best results are highlighted.

## 6. Conclusions

In this paper, we implicitly incorporate semantic information into the feature detection and description processes, enabling the model to extract globally reliable features from a single network end-to-end. Specifically, we leverage outputs of an off-the-shelf semantic segmentation network as guidance and adopt a combination of semantic- and featureaware guidance strategies to enhance the ability of embedding semantic information at training time. Experiments on large-scale visual localization datasets demonstrate that our method outperforms prior local features and gives competitive performance to advanced matchers but has higher efficiency. We argue that our approach could be a good tradeoff between accuracy and efficiency.

## References

[1] Asha Anoosheh, Torsten Sattler, Radu Timofte, Marc Pollefeys, and Luc Van Gool. Night-to-day image translation for retrieval-based localization. In ICRA, 2019. 3, 6, 7

[2] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. NetVLAD: CNN architecture for weakly supervised place recognition. In CVPR, 2016. 2, 6

[3] Relja Arandjelovic and Andrew Zisserman. Three things ev-´ eryone should know to improve object retrieval. In CVPR, 2012. 1, 2, 7

[4] Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. TPAMI, 2017. 5

[5] Eric Brachmann, Alexander Krull, Sebastian Nowozin, Jamie Shotton, Frank Michel, Stefan Gumhold, and Carsten Rother. DSAC-differentiable RANSAC for camera localization. In CVPR, 2017. 2

[6] Eric Brachmann and Carsten Rother. Learning less is more-6d camera localization via 3d surface regression. In CVPR, 2018. 2

[7] Ignas Budvytis, Marvin Teichmann, Tomas Vojir, and Roberto Cipolla. Large scale joint semantic re-localisation and scene understanding via globally unique instance coordinate regression. In BMVC, 2019. 2, 3

[8] Hongkai Chen, Zixin Luo, Jiahui Zhang, Lei Zhou, Xuyang Bai, Zeyu Hu, Chiew-Lan Tai, and Long Quan. Learning to match features with seeded graph matching network. In ICCV, 2021. 1, 2, 3, 6, 7, 8

[9] Hongkai Chen, Zixin Luo, Lei Zhou, Yurun Tian, Mingmin Zhen, Tian Fang, David Mckinnon, Yanghai Tsin, and Long Quan. ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer. In ECCV, 2022. 1, 3, 6

[10] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, 2018. 2

[11] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, 2018. 2, 4

[12] Wentao Cheng, Weisi Lin, Kan Chen, and Xinfeng Zhang. Cascaded parallel filtering for memory-efficient image-based localization. In ICCV, 2019. 2, 6, 7

[13] Titus Cieslewski, Konstantinos G Derpanis, and Davide Scaramuzza. SIPs: Succinct interest points from unsupervised inlierness probability learning. In 3DV, 2019. 3

[14] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In CVPRW, 2018. 1, 2, 3, 4, 6, 7, 8

[15] Mihai Dusmanu, Ondrej Miksik, Johannes L Schonberger, and Marc Pollefeys. Cross-Descriptor Visual Localization and Mapping. In ICCV, 2021. 3

[16] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-Net: A trainable CNN for joint description and detection of local features. In CVPR, 2019. 1, 2, 3, 6, 7

[17] Patrick Ebel, Anastasiia Mishchuk, Kwang Moo Yi, Pascal Fua, and Eduard Trulls. Beyond cartesian representations for local descriptors. In ICCV, 2019. 3

[18] Hugo Germain, Guillaume Bourmaud, and Vincent Lepetit. Sparse-to-dense hypercolumn matching for long-term visual localization. In 3DV, 2019. 1, 3, 6, 7

[19] Hugo Germain, Guillaume Bourmaud, and Vincent Lepetit. S2DNet: Learning accurate correspondences for sparse-todense feature matching. In ECCV, 2022. 1, 6

[20] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. In NIPS, 2014. 3

[21] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. IJCV, 2021. 5

[22] Wilfried Hartmann, Michal Havlena, and Konrad Schindler. Predicting matchability. In CVPR, 2014. 3

[23] Kun He, Yan Lu, and Stan Sclaroff. Local descriptors optimized for average precision. In CVPR, 2018. 2, 3, 5, 8

[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 5

[25] Hanjiang Hu, Zhijian Qiao, Ming Cheng, Zhe Liu, and Hesheng Wang. DASGIL: Domain adaptation for semantic and geometric-aware image-based localization. TIP, 2020. 3, 6, 7

[26] Zhaoyang Huang, Han Zhou, Yijin Li, Bangbang Yang, Yan Xu, Xiaowei Zhou, Hujun Bao, Guofeng Zhang, and Hongsheng Li. VS-Net: Voting with segmentation for visual localization. In CVPR, 2021. 2, 3

[27] Hyo Jin Kim, Enrique Dunn, and Jan-Michael Frahm. Learned contextual feature reweighting for image geolocalization. In CVPR, 2017. 2

[28] Alex Kendall, Matthew Grimes, and Roberto Cipolla. Posenet: A convolutional network for real-time 6-dof camera relocalization. In ICCV, 2015. 2

[29] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 6

[30] Nikolay Kobyshev, Hayko Riemenschneider, and Luc Van Gool. Matching features correctly through semantic understanding. In 3DV, 2014. 3

[31] Mans Larsson, Erik Stenborg, Carl Toft, Lars Hammarstrand, Torsten Sattler, and Fredrik Kahl. Fine-grained segmentation networks: Self-supervised segmentation for improved long-term visual localization. In ICCV, 2019. 2, 3, 8

[32] Kunhong Li, Longguang Wang, Li Liu, Qing Ran, Kai Xu, and Yulan Guo. Decoupling Makes Weakly Supervised Local Feature Better. In CVPR, 2022. 2, 3, 6, 7

[33] Xinghui Li, Kai Han, Shuda Li, and Victor Prisacariu. Dualresolution correspondence networks. In NeurIPS, 2020. 1, 3, 6

[34] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, 2022. 2, 4, 5

[35] David G Lowe. Distinctive image features from scaleinvariant keypoints. IJCV, 2004. 1, 2, 3, 6, 7

[36] Zixin Luo, Tianwei Shen, Lei Zhou, Jiahui Zhang, Yao Yao, Shiwei Li, Tian Fang, and Long Quan. Contextdesc: Local descriptor augmentation with cross-modality context. In CVPR, 2019. 3

[37] Zixin Luo, Lei Zhou, Xuyang Bai, Hongkai Chen, Jiahui Zhang, Yao Yao, Shiwei Li, Tian Fang, and Long Quan. ASLFeat: Learning local features of accurate shape and localization. In CVPR, 2020. 1, 2, 3, 4, 6, 7, 8

[38] Jiayi Ma, Xingyu Jiang, Aoxiang Fan, Junjun Jiang, and Junchi Yan. Image matching from handcrafted to deep features: A survey. IJCV, 2021. 2

[39] Will Maddern, Geoffrey Pascoe, Chris Linegar, and Paul Newman. 1 year, 1000 km: The Oxford RobotCar dataset. IJRR, 2017. 6, 7

[40] Iaroslav Melekhov, Aleksei Tiulpin, Torsten Sattler, Marc Pollefeys, Esa Rahtu, and Juho Kannala. DGCNet: Dense geometric correspondence network. In WACV, 2019. 1, 6

[41] Anastasiia Mishchuk, Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Working hard to know your neighbor’s margins: Local descriptor learning loss. In NeurIPS, 2017. 2, 3, 4

[42] Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Repeatability is not enough: Learning affine regions via discriminability. In ECCV, 2018. 3

[43] Arun Mukundan, Giorgos Tolias, and Ondrej Chum. Explicit spatial encoding for deep local descriptors. In CVPR, 2019. 3

[44] Tayyab Naseer, Gabriel L Oliveira, Thomas Brox, and Wolfram Burgard. Semantics-aware visual localization under challenging perceptual conditions. In ICRA, 2017. 3

[45] Yuki Ono, Eduard Trulls, Pascal Fua, and Kwang Moo Yi. LF-Net: learning local features from images. In NeurIPS, 2018. 1

[46] Alexandra I Papadaki and Ronny Hansch. Match or no match: Keypoint filtering based on matching probability. In CVPRW, 2020. 3

[47] Adam Paszke, Abhishek Chaurasia, Sangpil Kim, and Eugenio Culurciello. Enet: A deep neural network architecture for real-time semantic segmentation. arXiv preprint arXiv:1606.02147, 2016. 1

[48] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019. 6

[49] Remi Pautrat, Viktor Larsson, Martin R Oswald, and Marc´ Pollefeys. Online invariance selection for local feature descriptors. In ECCV, 2020. 3, 6

[50] Jerome Revaud, Vincent Leroy, Philippe Weinzaepfel, and´ Boris Chidlovskii. PUMP: Pyramidal and Uniqueness Matching Priors for Unsupervised Learning of Local Descriptors. In CVPR, 2022. 3, 4, 6, 7

[51] Jerome Revaud, Philippe Weinzaepfel, Cesar Roberto de´ Souza, and Martin Humenberger. R2D2: Repeatable and reliable detector and descriptor. In NeurIPS, 2019. 1, 2, 3, 4, 5, 6, 7, 8

[52] Ignacio Rocco, Relja Arandjelovic, and Josef Sivic. Efficient´ neighbourhood consensus networks via submanifold sparse convolutions. In ECCV, 2020. 3, 6

[53] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. ORB: An efficient alternative to SIFT or SURF. In ICCV, 2011. 1, 2

[54] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From Coarse to Fine: Robust Hierarchical Localization at Large Scale. In CVPR, 2019. 1, 2, 3, 6

[55] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In CVPR, 2020. 1, 2, 3, 6, 7, 8

[56] Paul-Edouard Sarlin, Ajaykumar Unagar, Mans Larsson, Hugo Germain, Carl Toft, Viktor Larsson, Marc Pollefeys, Vincent Lepetit, Lars Hammarstrand, Fredrik Kahl, et al. Back to the feature: learning robust camera localization from pixels to pose. In CVPR, 2021. 6, 7

[57] Torsten Sattler, Bastian Leibe, and Leif Kobbelt. Efficient & effective prioritized for large-scale image-based localization. TPAMI, 2016. 1, 2, 6, 7

[58] Torsten Sattler, Will Maddern, Carl Toft, Akihiko Torii, Lars Hammarstrand, Erik Stenborg, Daniel Safari, Masatoshi Okutomi, Marc Pollefeys, Josef Sivic, et al. Benchmarking 6dof outdoor visual localization in changing conditions. In CVPR, 2018. 2

[59] Torsten Sattler, Will Maddern, Carl Toft, Akihiko Torii, Lars Hammarstrand, Erik Stenborg, Daniel Safari, Masatoshi Okutomi, Marc Pollefeys, Josef Sivic, et al. Benchmarking 6dof outdoor visual localization in changing conditions. In CVPR, 2018. 6, 7, 8

[60] Torsten Sattler, Tobias Weyand, Bastian Leibe, and Leif Kobbelt. Image retrieval for image-based localization revisited. In BMVC, 2012. 6, 7, 8

[61] Torsten Sattler, Qunjie Zhou, Marc Pollefeys, and Laura Leal-Taixe. Understanding the limitations of cnn-based absolute camera pose regression. In CVPR, 2019. 2

[62] Johannes L Schonberger, Marc Pollefeys, Andreas Geiger,¨ and Torsten Sattler. Semantic visual localization. In CVPR, 2018. 3

[63] Johannes L Schonberger, Marc Pollefeys, Andreas Geiger,¨ and Torsten Sattler. Semantic visual localization. In CVPR, 2018. 3

[64] Tianxin Shi, Shuhan Shen, Xiang Gao, and Lingjie Zhu. Visual localization using sparse semantic 3D map. In ICIP, 2019. 1, 3, 4, 6, 7

[65] Yan Shi, Jun-Xiong Cai, Yoli Shavit, Tai-Jiang Mu, Wensen Feng, and Kai Zhang. ClusterGNN: Cluster-based Coarseto-Fine Graph Neural Network for Efficient Feature Matching. In CVPR, 2022. 1, 2, 3, 6, 7

[66] Erik Stenborg, Carl Toft, and Lars Hammarstrand. Longterm visual localization using semantically segmented images. In ICRA, 2018. 3

[67] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. LoFTR: Detector-free local feature matching with transformers. In CVPR, 2021. 1, 3, 6

[68] Suwichaya Suwanwimolkul, Satoshi Komorita, and Kazuyuki Tasaka. Learning of low-level feature keypoints for accurate and robust detection. In WACV, 2021. 3, 6

[69] Linus Svarm, Olof Enqvist, Fredrik Kahl, and Magnus Os-¨ karsson. City-scale localization for cameras with known vertical direction. TPAMI, 2016. 1, 2, 6, 7

[70] Yurun Tian, Vassileios Balntas, Tony Ng, Axel Barroso-Laguna, Yiannis Demiris, and Krystian Mikolajczyk. D2D: Keypoint extraction with describe to detect approach. In ACCV, 2020. 3, 6

[71] Yurun Tian, Bin Fan, and Fuchao Wu. L2-net: Deep learning of discriminative patch descriptor in euclidean space. In CVPR, 2017. 3

[72] Yurun Tian, Xin Yu, Bin Fan, Fuchao Wu, Huub Heijnen, and Vassileios Balntas. SOSNet: Second order similarity regularization for local descriptor learning. In CVPR, 2019. 3, 6

[73] Carl Toft, Erik Stenborg, Lars Hammarstrand, Lucas Brynte, Marc Pollefeys, Torsten Sattler, and Fredrik Kahl. Semantic match consistency for long-term visual localization. In ECCV, 2018. 1, 3, 4, 6, 7

[74] Prune Truong, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning accurate dense correspondences and when to trust them. In CVPR, 2021. 1, 6

[75] Prune Truong, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning accurate dense correspondences and when to trust them. In CVPR, 2021. 3

[76] Michał J Tyszkiewicz, Pascal Fua, and Eduard Trulls. DISK: Learning local features with policy gradient. In NeurIPS, 2020. 1, 3

[77] Qianqian Wang, Xiaowei Zhou, Bharath Hariharan, and Noah Snavely. Learning feature descriptors using camera pose supervision. In ECCV, 2020. 3, 4, 6, 7

[78] Zhe Xin, Yinghao Cai, Tao Lu, Xiaoxia Xing, Shaojun Cai, Jixiang Zhang, Yiping Yang, and Yanqing Wang. Localizing discriminative visual landmarks for place recognition. In ICRA, 2019. 3, 6, 7

[79] Fei Xue, Ignas Budvytis, Daniel Olmeda Reino, and Roberto Cipolla. Efficient Large-scale Localization by Global Instance Recognition. In CVPR, 2022. 1, 2, 3, 4, 6, 7, 8

[80] Fei Xue, Xin Wu, Shaojun Cai, and Junqiu Wang. Learning multi-view camera relocalization with graph neural networks. In CVPR, 2020. 2

[81] Kwang Moo Yi, Eduard Trulls, Vincent Lepetit, and Pascal Fua. Lift: Learned invariant feature transform. In ECCV, 2016. 1, 3

[82] Long Zhao, Xi Peng, Yuxiao Chen, Mubbasir Kapadia, and Dimitris N Metaxas. Knowledge as priors: Crossmodal knowledge generalization for datasets without superior knowledge. In CVPR, 2020. 3

[83] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. IJCV, 2019. 2, 4, 8

[84] Qunjie Zhou, Torsten Sattler, and Laura Leal-Taixe. Patch2pix: Epipolar-guided pixel-level correspondences. In CVPR, 2021. 3, 6