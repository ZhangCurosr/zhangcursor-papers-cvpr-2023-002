: Spatial features aggregation : Temporal features aggregation : Point-wise feature : Cluster-wise feature : Cluster/Scene-wise feature

# Spatiotemporal Self-supervised Learning for Point Clouds in the Wild

Yanhao Wu <sup>1</sup> Tong Zhang<sup>2</sup> Wei Ke<sup>1</sup> Sabine Susstrunk¨ <sup>2</sup> Mathieu Salzmann<sup>2</sup> <sup>1</sup> School of Software Engineering, Xi’an Jiaotong University, China <sup>2</sup> School of Computer and Communication Sciences, EPFL Switzerland

## Abstract

Self-supervised learning (SSL) has the potential to benefit many applications, particularly those where manually annotating data is cumbersome. One such situation is the semantic segmentation ofpoint clouds. In this context, existing methods employ contrastive learning strategies and define positive pairs by performing various augmentation of point clusters in a single frame. As such, these methods do not exploit the temporal nature of LiDAR data. In this paper, we introduce an SSL strategy that leverages positive pairs in both the spatial and temporal domain. To this end, we design (i) a point-to-cluster learning strategy that aggregates spatial information to distinguish objects; and (ii) a cluster-to-cluster learning strategy based on unsupervised object tracking that exploits temporal correspondences. We demonstrate the benefits of our approach via extensive experiments performed by self-supervised training on two large-scale LiDAR datasets and transferring the resulting models to other point cloud segmentation benchmarks. Our results evidence that our method outperforms the state-of-the-art point cloud SSL methods. <sup>1</sup>

## 1. Introduction

Semantic segmentation from LiDAR point clouds can be highly beneficial in practical applications, e.g., for selfdriving vehicles to safely interact with their surroundings. Nowadays, state-of-the-art methods [13, 36, 46] achieve this with deep neural networks. While effective, the training of such semantic segmentation networks requires large amounts of annotated data, which is prohibitively costly to acquire, particularly for point-level LiDAR annotations [45]. By contrast, with the rapid proliferation of selfdriving vehicles, large amounts of unlabeled LiDAR data are generated. Here, we develop a method to exploit such unlabeled data in a self-supervised learning framework.

Self-supervised learning (SSL) aims to learn features without any human annotations [1, 2, 22, 26, 33, 35, 40, 45] but so that they can be effectively used for fine-tuning on a downstream task with a small number of labeled samples. This is achieved by defining a pre-task that does not require annotations. While many pre-tasks have been proposed [27], contrastive learning has nowadays become a highly popular choice [30,33,40,41,45]. In general, it aims to maximize the similarity of positive pairs while potentially minimizing that of negative ones. In this context, most of the point cloud SSL literature focuses on indoor scenes, for which relatively dense point clouds are available. Unfortunately, for outdoor scenes, such as the ones we consider here, the data is more complex and much sparser, and creating effective pairs remains a challenge.

![](images/25788283ed41c99f8b43fdf4f247ffc2b6fae6b7eb6ae3c04e1e043312782465.jpg)  
Figure 1. Our method vs existing ones. (Top) Previous methods create positive pairs for SSL by applying different augmentations, τ<sub>1</sub> and τ<sub>2</sub> (e.g., random flipping, clipping), to a single frame. (Bottom) By contrast, we leverage both spatial and temporal information via a point-to-cluster and an inter-frame SSL strategy. Points in the same color are from the same cluster in the latent space.

![](images/119fc8491a80824be3b53a809fe34a330b935fcc2b511c5a0c8bb0820450ab97.jpg)  
Figure 2. Cars in the same frame but under different illumination angles. Note that the main source of difference between the two instance point clouds arises from the different illumination angles.

Several approaches [33, 45] have nonetheless been proposed to perform SSL on outdoor LiDAR point cloud data. As illustrated in the top portion of Fig. 1, they construct positive pairs of point clusters or scenes by applying augmentations to a single frame. As such, they neglect the temporal information of the LiDAR data. By contrast, in this paper, we introduce an SSL approach to LiDAR point cloud segmentation based on extracting effective positive pairs in both the spatial and temporal domain.

To achieve this without requiring any pose sensor as in [24, 40], we introduce (i) a point-to-cluster (P2C) SSL strategy that maximizes the similarity between the features encoding a cluster and those of its individual points, thus encouraging the points belonging to the same object to be close in feature space; (ii) a cluster-level inter-frame selfsupervised learning strategy that tracks an object across consecutive frames in an unsupervised manner and encourages feature similarity between the different frames. These two strategies are depicted in the bottom portion of Fig. 1.

Note that the illumination angle of one object seen in two different frames typically differs. As shown in Fig. 2, this is also the main source of difference between two objects of the same class in the same frame. Therefore, our interframe SSL strategy lets us encode not only temporal information, but also the fact that points from different objects from the same class should be close to each other in feature space. As simulating different illumination angles via data augmentation is challenging, our approach yields positive pairs that better reflects the intra-class variations in LiDAR point clouds than existing single-frame methods [33, 45].

Our contribution can be summarized as follows:

• We introduce an SSL strategy for point cloud segmentation based only on positive pairs. It does not require any external information, such as pose, GPS, and IMU.

• We propose a novel Point-to-Cluster (P2C) training paradigm that combines the advantages of point-level and cluster-level representations to learn a structured point-level embedding space.

• We introduce the use of cluster-level inter-frame selfsupervised leaning on point clouds generated by a Li-DAR sensor, which introduces a new way to integrate temporal information into SSL.

Our experiments on several datasets, including KITTI [17], nuScene [5], SemanticKITTI [4] and SemanticPOSS [34], evidence that our method outperforms the state-of-the-art SSL techniques for point cloud data.

## 2. Related Work

Self-supervised learning for images. Self-supervised learning for images has developed at a fast pace in recent years [7–9,11,18,21,38]. Existing methods follow different paradigms, such as generation-based methods [32], clustering methods [6, 25, 42, 43] and contrastive learning methods [10, 12, 20]. Currently, BYOL [19], a self-supervised learning method that uses only positive pairs in its loss function, constitutes the state of the art. Intrigued by the success of such contrastive learning strategies, several works have studied the principles behind this approach, with a particular focus on the role of data augmentation [3, 23, 28, 37]. In [39], it was observed that data augmentation creates a certain degree of “chaos” between the intra-class samples that helps them to become more similar. Similarly, LoGo [44] also introduce local and global crops differently to handle the variance due to the augmentation. Our method is inspired by BYOL but targets 3D data. Because of the fundamentally different nature of 2D images and 3D point clouds, data augmentation designed for images does not directly apply to the 3D domain.

Self-supervised learning for 3D data. As in the image case, the number of self-supervised learning methods for 3D data has grown rapidly [1, 2, 22, 26, 33, 35, 40, 45], with examples such as DepthContrast [45], PointContrast [40], GCC-3D [30], ProposalContrast [41], STRL [24] and Seg-Contrast [33]. Nevertheless, these methods still suffer from severe limitations. In particular, many methods [24, 40] need the camera pose in each frame to find correspondences to use as positive pairs. While effective for indoor scenes, the points in outdoor scenes are much sparser, and even with the ground-truth poses, correspondences between points are hard to obtain. By contrast, SegContrast, ProposalContrast, and DepthContrast [33, 41, 45] specifically tackle the outdoor scenario, without requiring camera poses. However, they aggregate features in each region through either max or average pooling, and pull region-level features from different views together, therefore, it does not have constraint for each point. More importantly, they fail to find a way to associate the points in different time frames. By contrast, our method only utilizes point cloud data and does not rely on camera calibration to aggregate spatial and temporal features. Furthermore, we propose a point-to-cluster training paradigm that combines the advantages of point-level and cluster-level discrimination.

![](images/93ccb803e6dbd24969ef3802df66ea859615662ffbc1cb4f2f284602ecf70c1f.jpg)  
Figure 3. Overview of our STSSL. Given a sequence of LiDAR point clouds, we first perform clustering and unsupervised tracking to associate clusters in different frames. At each training iteration, we select two frames and apply augmentations to generate two views for each frame $( \mathrm { i . e . , } P _ { m } ^ { 1 } , P _ { n } ^ { 1 } , P _ { m } ^ { 2 } , P _ { n } ^ { 2 } ) .$ . A feature extractor (Backbone) is then used to obtain point-wise features in the four views, and we collect the features belonging to each cluster. In $P _ { m } ^ { 2 } , P _ { n } ^ { 2 }$ , we further apply a cluster-wise pooling layer to the features to generate cluster-wise features. Finally, we minimize the distance between the point features and the corresponding cluster features from $\bar { P } _ { m } ^ { 1 } , \ : P _ { n } ^ { 1 }$ and between the cluster features obtained from associated clusters in $P _ { m } ^ { 2 } , P _ { n } ^ { 2 } . ~ \tau _ { 1 }$ and $\tau _ { 2 }$ are data augmentations, such as random flipping and random clipping.

## 3. Method

The overall framework is depicted in Fig. 3 and contains three parts: clustering and unsupervised tracking, pointwise and cluster-wise feature extraction, and spatialtemporal feature aggregation. Below, we discuss these components in detail.

## 3.1. Clustering and Unsupervised Tracking

Let $P = \left\{ P ^ { 1 } , P ^ { 2 } , . . . , P ^ { T } \right\}$ denote a sequence of LiDAR point clouds with T frames, where $P ^ { k } = \left\{ p _ { 1 } ^ { k } , p _ { 2 } ^ { k } , . . . , p _ { N _ { k } } ^ { k } \right\}$ represents the k-th point cloud with $N _ { k } \ 3 \mathrm { D }$ points $p _ { i } ^ { k } \in \mathbb { R } ^ { 3 }$ The segmentation map of each $P ^ { k }$ is obtained by applying cluster to the non-ground points, where the ground points are eliminated by RANSAC [16]. Thanks to the over segmentation property of DBSCAN [15], each point has high possibility to represent the same semantic meaning with other points in the same cluster. This process yields a set of $M _ { k }$ clusters ${ \cal S } ^ { k } = \left\{ S _ { 1 } ^ { k } , S _ { 2 } ^ { k } , . . . , S _ { M _ { k } } ^ { k } \right\}$

We will leverage these clusters to define a point-tocluster loss for SSL, encoding a notion of spatial similarity. Furthermore, we will also exploit them to create temporal positive pairs for SSL via the unsupervised tracking strategy described below. Thus, the mechanism allows similar clusters to be merged into the same one in later stage to achieve final segmentation.

Specifically, unsupervised tracking is achieved by matching the clusters in two adjacent frames, $e . g .$ , frames k with $M _ { k }$ clusters and (k + 1) with $M _ { k + 1 }$ clusters. To this end, we define a matching degree matrix $D \in R ^ { M _ { k } \times M _ { k + 1 } }$ as

$$
D = D _ { l o c } + \alpha * D _ { f e a t } ,\tag{1}
$$

where $D _ { l o c }$ is the matrix of pairwise Euclidean distances between the cluster centers in the two frames, $D _ { f e a t }$ is the matrix of pairwise feature distance, and $\alpha ~ \in ~ ( 0 , 1 )$ is a weight balancing the two matrices. The center of cluster $j$ in frame k is taken as the average of all 3D points belong to this cluster. More details regarding the cluster features is provided in Section 3.2. We then use D to match the clusters in both frames using the Hungarian algorithm [29]. For the unmatched clusters, we will create trajectories for the one just appears in current frame, and abandon the trajectories of the clustering no longer exists. More details can be found in the supplementary

Thanks to the combination of 3D information and learned representations, this matching strategy allows us to robustly track a cluster across multiple frames. This lets us construct long-range positive pairs where a cluster is observed under different illumination angles, thus corresponding to a challenging positive sample for SSL. We will discuss how we exploit such pairs in Section 3.3.

## 3.2. Feature Extraction

As discussed above, we extract learned features from the input point clouds. Specifically, we extract two types of features: point-level ones and cluster-level ones. To this end, given an input point cloud $P ^ { k }$ , we first apply data augmentation to obtain two view $\tilde { P } ^ { k }$ and $\bar { P } ^ { k }$ . One view will be used to extract point-level features and the other for cluster-level features. This will let us create more challenging point-tocluster pairs for the SSL strategy discussed in Section 3.3.

Point-level Features. Following the BYOL [19] format, let $f$ denote the backbone encoder. In our case, $f$ is MinkUnet [14]. We forward pass $\tilde { P } ^ { k }$ through the backbone encoder to obtain a feature vector $y _ { q } ^ { k } \ = \ f ( \tilde { p } _ { q } ^ { k } )$ for every 3D point. We then group these representations according to the cluster to which each point belongs, giving us a set $F ^ { k } = \left\{ F _ { 1 } ^ { k } , F _ { 2 } ^ { k } , . . . , F _ { M _ { k } } ^ { k } \right\}$ , where $F _ { i } ^ { k } \in \mathbb { R } ^ { N _ { k , i } \times d }$ , with $N _ { k , : }$ the number of points in cluster i from point cloud k, and d the feature dimension of each $y _ { q } ^ { k }$

Cluster-level Features. To extract cluster-level features, we first process $\bar { P } ^ { k }$ as above to extract d-dimensional pointlevel features. However, instead of simply grouping these features according to the clusters, we max-pool them according to the clusters. This yields a set of cluster-level features $C ^ { k } = \left\{ c _ { 1 } ^ { k } , c _ { 2 } ^ { k } , . . . , c _ { M _ { k } } ^ { k } \right\}$ , where $c _ { i } ^ { k } \in \mathbb { R } ^ { 1 \times d }$

## 3.3. Spatialtemporal Feature Aggregation

Let us now describe our spatiotemporal SSL framework. It relies on two loss functions encoding two goals: i) points from one object should be close in feature space; ii) points from the same class should be closer to each other than other classes. We materialize these two objectives via the Pointto-Cluster and Cluster-level Inter-frame Self-supervised Leaning strategies discussed below.

Point-to-Cluster Learning Strategy. To encourage points from the same object to be close to each other, we minimize the distance between the point features of $\tilde { P } ^ { k }$ and the corresponding cluster features in view $\bar { P } ^ { k }$ . Given the features discussed in Section 3.2, this is achieved via the

loss function

$$
L _ { p 2 c } = \sum _ { i = 1 } ^ { M _ { k } } \sum _ { j = 1 } ^ { N _ { k , i } } \left\| \frac { f _ { i , j } ^ { k } } { \| f _ { i , j } ^ { k } \| _ { 2 } } - \frac { c _ { i } ^ { k } } { \| c _ { i } ^ { k } \| _ { 2 } } \right\| _ { 2 } ^ { 2 } ,\tag{2}
$$

where $f _ { i , j } ^ { k } \in \mathbb { R } ^ { 1 \times d }$ denotes the feature vector from $F _ { i } ^ { k }$ corresponding to point $j$ in cluster i. In essence, this encourages the network to learn similar features for all points in the same cluster while being robust to different views of the point cloud.

Cluster-level Inter-frame Self-supervised Leaning. To encourage points from the same class to be close to each other, we build on the observation that the main source of differences between two objects from the same class in point cloud data is the illumination angles under which they are observed. Thanks to the unsupervised tracking strategy, we can extract pairs of clusters in two distant frames, where the object is then seen under different illumination angles. Given two frames m and $n ,$ let $N ^ { m n }$ denote the number of matched clusters across the two frames. Then, we use the cluster-level features to write the loss

$$
L _ { i n t e r - f r a m e } = \sum _ { i = 0 } ^ { N ^ { m n } } \left\| \frac { c _ { i } ^ { m } } { \| c _ { i } ^ { m } \| _ { 2 } } - \frac { c _ { i } ^ { n } } { \| c _ { i } ^ { n } \| _ { 2 } } \right\| _ { 2 } ^ { 2 } ,\tag{3}
$$

where $c _ { i } ^ { m }$ and $c _ { i } ^ { n }$ are the cluster-level feature vectors of two matched clusters in frame m and $n .$ Hence, the total loss can be written as

$$
L _ { t o t a l } = L _ { p 2 c } + \lambda L _ { i n t e r - f r a m e } ,\tag{4}
$$

where $\lambda$ is a weight balancing the two loss terms. In practice, the inter-frame information can be better used with the feature of SSL on intra-frame. Thus, we choose a strategy of progressively increasing the λ.

## 4. Experiments

We first describe our experimental settings, including datasets, unsupervised tracking, and implementation details. Then, we demonstrate the benefits of our selfsupervised pre-trained model on downstream tasks, and finally analyze different aspects of our method.

## 4.1. Experimental Settings

Datasets. We use the KITTI [17] and nuScene [5] datasets for pre-training, and SemanticKITTI [4] and SemanticPOSS [34] for the down-stream tasks.

KITTI [17] has 21 sequences, and its sampling rate is 10hz. Following [33], we use only the point clouds captured by the Velodyne LiDAR sensor rather than all the information obtained from the position sensors. The sequences 0- 10 are used for pre-training, with the exception of sequence $^ { 8 , }$ which we use as validation data. nuScene [5] is much larger than KITTI. It comprises 1000 scenes and is divided into 10 sequences. The LiDAR data is acquired at 20hz. Because of limited computational resources, we only use the point clouds captured by the Velodyne LiDAR sensor in sequence 1 and 2 (scenes 0 - 149) for pre-training. SemanticKITTI [4] provides dense point-wise annotations for almost every point in KITTI [4]. A total of 23,201 scans are annotated on sequences 0-10 of KITTI for training and validation. SemanticPOSS [34] is also used for semantic segmentation, and contains 2988 diverse and complicated LiDAR scans with 14 classes. The scans are divided into 6 splits, with 500 scans per split. Splits 4 and 5 are used for testing and the other ones are used for training.

<table><tr><td>Method name</td><td>mIoU</td><td>car</td><td>road</td><td>sidewalk</td><td>building</td><td>fence</td><td>vegetation</td><td>terrain</td><td>parking</td><td>pole</td></tr><tr><td>From scratch</td><td>29.17</td><td>82.61</td><td>74.32</td><td>52.06</td><td>78.99</td><td>19.29</td><td>83.13</td><td>68.20</td><td>9.04</td><td>30.09</td></tr><tr><td>STRL [24]</td><td>16.64</td><td>47.66</td><td>56.17</td><td>23.17</td><td>58.63</td><td>13.68</td><td>69.96</td><td>41.91</td><td>0</td><td>3.12</td></tr><tr><td>DepthContrast [45]</td><td>30.91</td><td>88.80</td><td>69.51</td><td>49.87</td><td>82.67</td><td>22.70</td><td>83.36</td><td>67.38</td><td>9.32</td><td>48.69</td></tr><tr><td>SegContrast [33]</td><td>34.01</td><td>89.22</td><td>78.72</td><td>57.19</td><td>82.80</td><td>21.99</td><td>83.42</td><td>67.26</td><td>14.06</td><td>50.91</td></tr><tr><td>STSSL (ours)</td><td>37.71</td><td>91.11</td><td>85.34</td><td>66.09</td><td>85.43</td><td>25.63</td><td>84.79</td><td>72.57</td><td>22.61</td><td>48.67</td></tr></table>

Table 1. Per-class IoU when fine-tuning with 0.1 % labels.  
![](images/d54db347f97788a84d29a76993370fb4ae1194b2068ac72791ef85fbdbb419f6.jpg)  
Figure 4. Segmentation results on different frames (rows). The models are fine-tuned with 0.1% labels on KITTI. We compare SegContrast [33], STSSL (ours) and training from scratch (without pre-training). Our method better distinguishes the different structures shown in the highlighted area (red circle).

Unsupervised Tracking. We set the relative weight between spatial distance and feature distance in Eq. (1) to α = 0.5, and the threshold of RANSAC distance to 0.25 as in [33]. The threshold of DBSCAN distance is set to 0.25 in KITTI and 0.5 in nuScene. In each frame, we drop clusters with fewer than 200 points or more than 20000 points to filter out noise and retain up to 50 clusters.

Implementation Details. We compare our approach with DepthContrast [45], STRL [24], SegContrast [33], and training from scratch. We use MinkUnet [14] as backbone for all approaches and build our approach on the basis of BYOL [19]. We pre-train the backbone on KITTI and nuScene for 200 epochs using an SGD optimizer with a momentum of 0.9, and set the weight decay to 0.0004 following SegContrast [33]. The learning rate is initially set to 0.036 with a linear annealing scheme with a minimum learning rate equal to 0.009. In the early training stages, we set λ to 0, and to 4 in the later stages. When incorporating the inter-frame loss term, we re-initialize different MLPs after the backbone networks to avoid information leaks from spatial. The batch size is set to 8 for each GPU, and we use 8×GTX3090 GPUs to pre-train the models, which leads to the total batch size of 64.

For fine-tuning on the down-stream semantic segmentation task, we use an SGD optimizer with a cosine learning rate schedule. The fine-tuned models are evaluated on the validation sequences, i.e., sequence 8 for SemanticKITTI and sequences 4 and 5 for SemanticPOSS [33]. The batch size is set to 2 for each GPU, and 4×GTX2080Ti GPUs are used for the experiments.

Evaluation Metrics. We evaluate point cloud semantic segmentation using the mean intersection over union (mIoU) and the overall point classification accuracy (Acc).

## 4.2. Outdoor Scene Understanding

Label Efficiency. To assess the label efficiency of our STSSL approach, we fine-tune the model pre-trained on KITTI on SemanticKITTI. Following [33], SemanticKITTI is divided into different regimes corresponding to different percentages of labels. Specifically, we use 0.1%, 1%, 10%, and 100% of the training data to fine-tune the pre-trained model for semantic segmentation.

<table><tr><td>0.1%</td><td>1%</td><td>10%</td><td>100%</td></tr><tr><td>From Scratch</td><td>29.17 48.11</td><td>51.00</td><td>56.14</td></tr><tr><td>STRL [24]</td><td>16.64 31.88</td><td>30.88</td><td>55.71</td></tr><tr><td>DepthContrast [45]</td><td>30.91 42.41</td><td>42.38</td><td>45.48</td></tr><tr><td>SegContrast [33]</td><td>34.01 48.02</td><td>52.26</td><td>55.45</td></tr><tr><td>STSSL (ours)</td><td>37.71 52.60</td><td>54.51</td><td>57.33</td></tr></table>

Table 2. Pre-training on KITTI and evaluating the fine-tuned models in different label regimes on SemanticKITTI for semantic segmentation. We report the mIoU.
<table><tr><td>mIoU / Acc</td><td>seq1</td><td>seq 1-2</td></tr><tr><td>From Scratch</td><td>29.17 / 82.57</td><td>29.17 / 82.57</td></tr><tr><td>STRL [24]</td><td>19.11 / 74.56</td><td>18.74 / 70.85</td></tr><tr><td>SegContrast [33]</td><td>33.91 / 84.88</td><td>34.28 / 85.20</td></tr><tr><td>STSSL (ours)</td><td>34.43 / 85.34</td><td>35.08 / 85.75</td></tr></table>

Table 3. Pre-training on nuScene and evaluating the fine-tuned models in the 0.1% label regime on SemanticKITTI for semantic segmentation. We report mIoU/Acc.
<table><tr><td>pre-train dataset</td><td>KITTI</td><td>nuScene(seq 1)</td></tr><tr><td>From Scratch</td><td>39.64 / 88.66</td><td>39.64 / 88.66</td></tr><tr><td>STRL [24]</td><td>38.43 / 88.32</td><td>36.63 / 87.53</td></tr><tr><td>SegContrast [33]</td><td>43.88 / 89.64</td><td>42.86 / 89.28</td></tr><tr><td>STSSL (ours)</td><td>43.84 / 89.47</td><td>43.55 / 89.38</td></tr></table>

Table 4. Pre-training on KITTI and nuScene, and evaluating the fine-tuned models on SemanticPOSS for semantic segmentation. We report the mIoU/Acc.
<table><tr><td>mIoU / Acc</td><td>0.1%</td><td>1%</td></tr><tr><td>From Scratch</td><td>29.17/ 82.57</td><td>48.11 / 89.94</td></tr><tr><td>STRL [24]</td><td>16.64 / 69.36</td><td>31.88 / 85.35</td></tr><tr><td>DepthContrast [45]</td><td>30.91 / 82.82</td><td>42.41 / 88.95</td></tr><tr><td>SegContrast [33]</td><td>34.01 / 84.72</td><td>48.02 / 88.84</td></tr><tr><td>P2C</td><td>35.48 / 86.18</td><td>50.83 / 90.14</td></tr></table>

Table 5. Ablation study on the pre-training strategy with 0.1% and 1% labels. We report mIoU/Acc.

In Table 1, we compare the mIoU and per-class IoU of the proposed STSSL and of the state-of-the-art approaches when using 0.1% of the labels. Our method outperforms the baselines by an impressive margin, yielding an mIoU of 37.71%, which is 3.7% better than SegContrast [33] and 8.54% than training from scratch. Fig. 4 evidences that our method yields more complete and accurate segmentation masks than the other methods.

The per-class comparison shows that our approach greatly improves the network’s performance on most classes when there are few annotations. Our STSSL yields much better results than the baselines, especially for car, building, vegetation, terrain, and parking. We attribute this to the fact that these classes have clearly different appearances under different illumination angles as shown in Fig.2, which is exactly the problem that our inter-frame selfsupervised learning addresses.

The results obtained by fine-tuning with different percentages of training data are provided in Table 2. Our method consistently outperforms the state-of-the-art selfsupervised approach for all label regimes. Specifically, the mIoU of our approach outperforms the SegContrast and From Scratch ones by 2.25% and 3.51%, with 10% labels.

Feature Representation Transferability. To confirm the transferability of the features learned by our approach, we pre-train our models on nuScene and design two settings for pre-training: i) only using sequence 1 (seq1), and ii) using sequence 1 and 2 (seq 1-2) with uniform down-sampling to keep the number of frames consistent with seq1 and the frame rate consistent with KITTI.

As shown in Table 3, our method outperforms training from scratch and Segcontrast [33] when using only seq 1 from nuScene for pre-training and fine-tuning on SemanticKITTI [4]. Our approach improves the segmentation performance by 5.26% in mIoU and 2.77% in Acc. When seq 1 and 2 are used for pre-training, our approach improves the segmentation performance by 5.91% in mIoU and 3.18% in Acc. We also fine-tune the models pretrained on nuScene or KITTI to the semantic segmentation task using SemanticPOSS [34]. As SemanticPOSS is small, we fine-tune on the entire dataset. Table 4 shows that our method yields better mIoU results than training from scratch. When the network is pre-trained on KITTI, our approach improves the mIoU by 4.20% compared to the network without pre-training.

## 4.3. Analysis

Guaranteed over-segmentation assumptions. P2C SSL strategy relies on the over-segmentation assumptions and the hyper-parameter leading to the over-segmented clusters is easy to set. To show this, we performed the following experiment. We varied the DBSCAN distance threshold from 0.15 to 0.45 and measured the proportion of clusters having at least 90 % of their points from the same semantic class. The higher this proportion, the higher the chance that the clusters are over-segmented and not undersegmented. In the worst case, we found 73.45% of the clusters being over-segmented witch evidences the stability of the hyper-parameter of DBSCAN.

Performance of unsupervised tracking. We measured that 63.73% of the clusters are tracked for at least 3 frames, and 31.33% for at least 8 frames. Such tracking times are sufficient for us to create positive pairs of clusters observed under different illumination angles as shown in Fig. 7.

## 4.4. Ablation Study

Experiment in this section, if it is not specially stated, the model pre-training on KITTI, and fine-tuning and eval-

![](images/87764bb5e8e2c7d99be6e384e4f74b2b5aa5968df4195e167327dcc6f39b5026.jpg)  
(a)

![](images/9a4e0bed6efe7196f5b4c4328bdd737a6c6982c5f6316e2ab4627d976c2f67d9.jpg)  
(b)

![](images/7dd47ec9bed3ba0e9d0be5ddc1ace857c599989eee4269d5e28f2d06c3707565.jpg)  
(c)

![](images/b0a37e9df76c46329d22934b103eb8861570742b90454aae724581b302613c3b.jpg)  
(d)

Figure 5. Comparison of the features generated by different pre-trained models. (a) Results with P2C. (b) Zoomed in car from (a). (c) Zoomed in car from (d). (d) Results with SegContrast [33]. The points with the same color are in the same cluster(clustering in feature space). The colors of (a, b) and (c, d) are independent and have no relationship with each other. For better visualization, we only colorize the car in (b) and (c).
<table><tr><td>Method name</td><td>mIoU</td><td>car</td><td>road</td><td>sidewalk</td><td>building</td><td>fence</td><td>vegetation</td><td>terrain</td><td>parking</td><td>pole</td></tr><tr><td>SegContrast [33]</td><td>34.01</td><td>89.22</td><td>78.72</td><td>57.19</td><td>82.80</td><td>21.99</td><td>83.42</td><td>67.26</td><td>14.06</td><td>50.91</td></tr><tr><td>SegContrast-BYOL</td><td>33.94</td><td>89.34</td><td>78.56</td><td>57.29</td><td>82.86</td><td>21.19</td><td>83.26</td><td>67.30</td><td>14.09</td><td>50.54</td></tr><tr><td>SegContrast-Inter</td><td>36.03</td><td>90.52</td><td>81.45</td><td>62.90</td><td>83.54</td><td>21.46</td><td>84.37</td><td>72.52</td><td>17.05</td><td>51.78</td></tr><tr><td>P2C</td><td>35.48</td><td>90.18</td><td>80.90</td><td>61.91</td><td>83.92</td><td>25.45</td><td>84.30</td><td>71.35</td><td>18.73</td><td>46.95</td></tr><tr><td>STSSL (ours)</td><td>37.71</td><td>91.11</td><td>85.34</td><td>66.09</td><td>85.43</td><td>25.63</td><td>84.79</td><td>72.57</td><td>22.61</td><td>48.67</td></tr></table>

Table 6. Per-class IoU when fine-tuning with 0.1 % labels. SegContrast-BYOL: SegContrast built on the basis of BYOL. SegContrast-Inter: SegContrast with a additional inter-frame self-supervised learning stage after the original SegContrast. P2C: Our approach with only the point-to-cluster loss function.

uating on SemanticKITTI.

Scene- vs. Cluster-level Pre-training. To demonstrate the effectiveness of the Point-to-Cluster learning strategy proposed in Section 3.3, we conduct an experiment with only the point-to-cluster loss function of Eq. (2) for pretraining. We dub this setting P2C. As shown in Table 5 and Table 6, a cluster-level pre-trained SegContrast performs better than scene-level pre-trained DepthContrast and STRL. Our proposed P2C pre-training method achieves better results than SegContrast. To better understand the advantages of P2C, we have designed the following visualization. We select a point cloud frame, use the pre-trained model (i.e., SegContrast and P2C) to extract features for each point, and use K-Means [31] to cluster these features. Specifically, we use 20 clusters, which corresponds to the number of categories in the annotations. In Fig. 5, we visualize the points by coloring them according to the clusters. With features extracted by the SegContrast pre-trained model, the car zoomed in in Fig. 5 (c) is divided into several colors. This indicates that the points from the same category can be distant in the learned feature space. By contrast, encouraging the points in the same cluster to have similar features, P2C yields feature such that most of the car points are in the same cluster, as shown in Fig. 5 (b), thus indicating that the features are more representative of the object instances.

Effectiveness of Inter-frame Self-supervised Learning. To better demonstrate the role of inter-frame selfsupervised learning, we add a new stage after the original SegContrast, in which we activate our inter-frame loss function, as in our method. Note that we maintain the total number of training epochs to 200. We dub the resulting model SegContrast-inter, and show its per-class IoU in Table 6. For most classes, SegContrast-inter performs much better than SegContrast. However, SegContrast performs better in the fence class. Fence has almost the same appearance when they are viewed from different angles (the z axis remains unchanged). This experiment also supports our motivation that the illumination angle is an important factor, making the appearance of an object differ.

Interval between Two Frames. Since we choose positive pairs from different frames, the interval between the two selected frames is a hyper-parameter affecting the results. In our approach, we gradually increase the interval between two frames to avoid having to set such a hyperparameter. Here we evaluate the impact of the interval on a model using a fixed interval between two frames. The experiments are repeated for 3 times to reduce randomness, and we report the average. As shown in Fig. 8, with the increase of interval, the performance first increases and then decreases. We think it is because greater interval can reduce the impact of more illumination angles, but the number of clusters that can be tracked across frames decreases as the interval increases. The best-performing model corresponds to an interval of 5, reaching an mIoU of 36.05% when it reaches the balance of illumination angles change and number of tracking clusters. Note that the worst performance (interval of 9) is still better than that of SegContrast.

In Fig. 6, we visualize a frame with multiple cars under different LiDAR illumination angles. We also render the points in the same cluster with the same color. Note that SegContrast and STSSL with interval=0 cannot extract similar features for the cars under different illumination angles. By contrast, STSSL with interval=4 yields similar features for all cars, benefiting from inter-frame matching.

![](images/9d3b3a49ce40c61d50e6d6703b2b8160830e14125a56c6732146dfdf87c144b6.jpg)

Figure 6. Comparison of models pre-trained with different intervals between two frames. Left: SegContrast. Middle: (STSSL, interval=0). Right: (STSSL, interval=4). The pre-trained models are used to extract point features for visualization, and the points with the same color are adjacent in feature space. For better visualization, we only colorize the points belonging to the specified clusters.  
![](images/bd35901c1cd187d8c2b620af72a4e1c13f858f57cdf91875cb16d88c9c0c919b.jpg)

Figure 7. Associated clusters in multiple frames. Some clusters with clear semantic information are selected for display. The same color represents the same cluster. Red: car; Blue: motorcycle; Orange: person.  
![](images/867a7e240890764feacdd68693a780a8446a7188511db17e4e5b806ac4a85968.jpg)

Figure 8. Comparison mIoU on SemanticKITTI dataset, which is fine-tuned by the models pre-trained with different intervals between two frames.
<table><tr><td>mIoU / Acc</td><td>0.1 %</td><td>1 %</td></tr><tr><td>From Scratch</td><td>29.17 / 82.57</td><td>48.11 / 89.94</td></tr><tr><td>SegContrast [33]</td><td>34.01 / 84.72</td><td>48.02 / 88.84</td></tr><tr><td>STSSL-n</td><td>36.01 / 87.44</td><td>51.06 / 89.68</td></tr><tr><td>STSSL</td><td>37.71 / 87.67</td><td>52.60 / 90.24</td></tr></table>

Table 7. Ablation study on feature similarity for tracking. STSSL-n only considers location similarity in tracking stage.

Tracking with vs. without a Model. In Eq. (1), we use both location and feature similarities to compute the matching degree matrix for tracking. To illustrate the effectiveness of the feature similarity, we re-conduct the tracking with only location similarity. This is indicated by STSSL-n in Table 7. Note that the resulting method still outperforms

<table><tr><td>Method</td><td>mIoU (%)</td><td>Acc (%)</td></tr><tr><td>SegContrast [33]</td><td>34.01</td><td>84.72</td></tr><tr><td>SegContrast-BYOL</td><td>33.94</td><td>84.67</td></tr><tr><td>STSSL (Ours)</td><td>37.71</td><td>87.67</td></tr></table>

Table 8. Ablation study on the framework. SegContrast-BYOL: SegContrast built on the basis of BYOL.

SegContrast but not our complete STSSL.

Effect of BYOL. To evidence that the benefits of our approach over SegContrast are not only due to our use of BYOL instead of MoCo but truly to our training formalism, we replace the MoCo in SegContrast with BYOL. As shown in Table 8, SegContrast-BYOL performs on par with the original SegContrast (with MoCo) in the 0.1% label regime. It implies that the networks are not the key factor in point clouds SSL. Due to the over-segmentation, we used in our method, where each cluster could belong to the same class, building negative pairs for each point in our point-to-cluster strategy will harm the optimization. By contrast, it is more intuitive for us to only build positive pairs to avoid pushing the nearby clusters away, which offers a chance to merge the cluster with the help of temporal information.

## 5. Conclusion and Limitation

In this paper, we have introduced an SSL strategy for point cloud segmentation without external supervision. It relies on a novel Point-to-Cluster (P2C) training paradigm to exploit spatial information, and further introduces an inter-frame self-supervised learning strategy to capture temporal information. Altogether, our approach provides a practical tool for pre-training with point clouds in the wild. Experiments evidence that our approach outperforms the state-of-the-art SSL techniques for point cloud in the wild. However, such an improvement does not materialize for objects whose appearance is invariant to viewpoint changes, such as fences. In the future, we will therefore focus on how to improve the segmentation ability of such objects.

Acknowledgement. This work was supported in part by the National Natural Science Foundation of China under Grant No. 62006182 and the Swiss National Science Foundation via the Sinergia grant CRSII5-180359.

## References

[1] Idan Achituve, Haggai Maron, and Gal Chechik. Selfsupervised learning for domain adaptation on point clouds. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 123–133, 2021. 1, 2

[2] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3d point clouds. In International Conference on Machine Learning, pages 40–49. PMLR, 2018. 1, 2

[3] Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler. Scarf: Self-supervised contrastive learning using random feature corruption. arXiv preprint arXiv:2106.15147, 2021. 2

[4] Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen Gall. Semantickitti: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9297–9307, 2019. 2, 4, 5, 6

[5] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020. 2, 4

[6] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proceedings of the European Conference on Computer Vision (ECCV), pages 132–149, 2018. 2

[7] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in Neural Information Processing Systems, 33:9912– 9924, 2020. 2

[8] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning, pages 1597–1607. PMLR, 2020. 2

[9] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. Advances in Neural Information Processing Systems, 33:22243–22255, 2020.

[10] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020. 2

[11] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15750–15758, 2021. 2

[12] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9640–9649, 2021. 2

[13] Ran Cheng, Ryan Razani, Ehsan Taghavi, Enxu Li, and Bingbing Liu. 2-s3net: Attentive feature fusion with adaptive feature selection for sparse semantic segmentation network. In Proceedings ofthe IEEE/CVF Conference on Com-

puter Vision and Pattern Recognition, pages 12547–12556, 2021. 1

[14] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3075– 3084, 2019. 4, 5

[15] Martin Ester, Hans-Peter Kriegel, Jorg Sander, Xiaowei Xu,¨ et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, volume 96, pages 226–231, 1996. 3

[16] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications ofthe ACM, 24(6):381–395, 1981. 3

[17] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237, 2013. 2, 4

[18] Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In Proceedings of the ieee/cvf International Conference on Computer Vision, pages 6391–6400, 2019. 2

[19] Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin´ Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in Neural Information Processing Systems, 33:21271–21284, 2020. 2, 4, 5

[20] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9729–9738, 2020. 2

[21] Olivier Henaff. Data-efficient image recognition with contrastive predictive coding. In International Conference on Machine Learning, pages 4182–4192. PMLR, 2020. 2

[22] Ji Hou, Benjamin Graham, Matthias Nießner, and Saining Xie. Exploring data-efficient 3d scene understanding with contrastive scene contexts. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15587–15597, 2021. 1, 2

[23] Dapeng Hu, Shipeng Yan, Qizhengqiu Lu, HONG Lanqing, Hailin Hu, Yifan Zhang, Zhenguo Li, Xinchao Wang, and Jiashi Feng. How well does self-supervised pre-training perform with streaming data? In International Conference on Learning Representations, 2021. 2

[24] Siyuan Huang, Yichen Xie, Song-Chun Zhu, and Yixin Zhu. Spatio-temporal self-supervised representation learning for 3d point clouds. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6535–6545, 2021. 2, 5, 6

[25] Pan Ji, Tong Zhang, Hongdong Li, Mathieu Salzmann, and Ian Reid. Deep subspace clustering networks. Advances in Neural Information Processing Systems, 30, 2017. 2

[26] Longlong Jing, Yucheng Chen, Ling Zhang, Mingyi He, and Yingli Tian. Self-supervised modal and view invariant feature learning. arXiv preprint arXiv:2005.14169, 2020. 1, 2

[27] Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 43(11):4037–4058, 2020. 1

[28] Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive selfsupervised learning. arXiv preprint arXiv:2110.09348, 2021. 2

[29] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955. 4

[30] Hanxue Liang, Chenhan Jiang, Dapeng Feng, Xin Chen, Hang Xu, Xiaodan Liang, Wei Zhang, Zhenguo Li, and Luc Van Gool. Exploring geometry-aware contrast and clustering harmonization for self-supervised 3d object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3293–3302, 2021. 1, 2

[31] J MacQueen. Classification and analysis of multivariate observations. In 5th Berkeley Symp. Math. Statist. Probability, pages 281–297, 1967. 7

[32] Lars Mescheder, Sebastian Nowozin, and Andreas Geiger. Adversarial variational bayes: Unifying variational autoencoders and generative adversarial networks. In International Conference on Machine Learning, pages 2391–2400. PMLR, 2017. 2

[33] Lucas Nunes, Rodrigo Marcuzzi, Xieyuanli Chen, Jens Behley, and Cyrill Stachniss. Segcontrast: 3d point cloud feature representation learning through self-supervised segment discrimination. IEEE Robotics Autom. Lett., 7(2):2116–2123, 2022. 1, 2, 4, 5, 6, 7, 8

[34] Yancheng Pan, Biao Gao, Jilin Mei, Sibo Geng, Chengkun Li, and Huijing Zhao. Semanticposs: A point cloud dataset with large quantity of dynamic instances. In 2020 IEEE Intelligent Vehicles Symposium (IV), pages 687–693. IEEE, 2020. 2, 4, 5, 6

[35] Jonathan Sauder and Bjarne Sievers. Self-supervised deep learning on point clouds by reconstructing space. Advances in Neural Information Processing Systems, 32, 2019. 1, 2

[36] Haotian Tang, Zhijian Liu, Shengyu Zhao, Yujun Lin, Ji Lin, Hanrui Wang, and Song Han. Searching efficient 3d architectures with sparse point-voxel convolution. In European Conference on Computer Vision, pages 685–702. Springer, 2020. 1

[37] Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? Advances in Neural Information Processing Systems, 33:6827–6839, 2020. 2

[38] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th International Conference on Machine Learning, pages 1096–1103, 2008. 2

[39] Yifei Wang, Qi Zhang, Yisen Wang, Jiansheng Yang, and Zhouchen Lin. Chaos is a ladder: A new theoretical understanding of contrastive learning via augmentation overlap. arXiv preprint arXiv:2203.13457, 2022. 2

[40] Saining Xie, Jiatao Gu, Demi Guo, Charles R Qi, Leonidas Guibas, and Or Litany. Pointcontrast: Unsupervised pretraining for 3d point cloud understanding. In European Conference on Computer Vision, pages 574–591. Springer, 2020. 1, 2

[41] Junbo Yin, Dingfu Zhou, Liangjun Zhang, Jin Fang, Cheng-Zhong Xu, Jianbing Shen, and Wenguan Wang. Proposalcontrast: Unsupervised pre-training for lidar-based 3d object detection. arXiv preprint arXiv:2207.12654, 2022. 1, 2

[42] Tong Zhang, Pan Ji, Mehrtash Harandi, Richard Hartley, and Ian Reid. Scalable deep k-subspace clustering. In Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part V 14, pages 466–481. Springer, 2019. 2

[43] Tong Zhang, Pan Ji, Mehrtash Harandi, Wenbing Huang, and Hongdong Li. Neural collaborative subspace clustering. In International Conference on Machine Learning, pages 7384–7393. PMLR, 2019. 2

[44] Tong Zhang, Congpei Qiu, Wei Ke, Sabine Susstrunk, and¨ Mathieu Salzmann. Leverage your local and global representations: A new self-supervised learning strategy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16580–16589, 2022. 2

[45] Zaiwei Zhang, Rohit Girdhar, Armand Joulin, and Ishan Misra. Self-supervised pretraining of 3d features on any point-cloud. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10252–10263, 2021. 1, 2, 5, 6

[46] Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, and Dahua Lin. Cylindrical and asymmetrical 3d convolution networks for lidar segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9939– 9948, 2021. 1