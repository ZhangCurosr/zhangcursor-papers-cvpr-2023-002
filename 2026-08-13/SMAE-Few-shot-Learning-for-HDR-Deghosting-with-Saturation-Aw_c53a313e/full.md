# SMAE: Few-shot Learning for HDR Deghosting with Saturation-Aware Masked Autoencoders

Qingsen Yan<sup>†1</sup> Song Zhang<sup>†2</sup> Weiye Chen<sup>†2</sup> Hao Tang<sup>3</sup> Yu Zhu<sup>1</sup> Jinqiu Sun<sup>1</sup> Luc Van Gool<sup>3</sup> Yanning Zhang<sup>1\*</sup> <sup>1</sup>Northwestern Polytechnical University <sup>2</sup>Xidian University <sup>3</sup>CVL, ETH Zurich

## Abstract

Generating a high-quality High Dynamic Range (HDR) image from dynamic scenes has recently been extensively studied by exploiting Deep Neural Networks (DNNs). Most DNNs-based methods require a large amount of training data with ground truth, requiring tedious and timeconsuming work. Few-shot HDR imaging aims to generate satisfactory images with limited data. However, it is difficult for modern DNNs to avoid overfitting when trained on only a few images. In this work, we propose a novel semi-supervised approach to realize few-shot HDR imaging via two stages oftraining, called SSHDR. Unlikely previous methods, directly recovering content and removing ghosts simultaneously, which is hard to achieve optimum, we first generate content of saturated regions with a selfsupervised mechanism and then address ghosts via an iterative semi-supervised learning framework. Concretely, considering that saturated regions can be regarded as masking Low Dynamic Range (LDR) input regions, we design a Saturated Mask AutoEncoder (SMAE) to learn a robustfeature representation and reconstruct a non-saturated HDR image. We also propose an adaptive pseudo-label selection strategy to pick high-quality HDR pseudo-labels in the second stage to avoid the effect of mislabeled samples. Experiments demonstrate that SSHDR outperforms state-ofthe-art methods quantitatively and qualitatively within and across different datasets, achieving appealing HDR visualization with few labeled samples.

## 1. Introduction

Standard digital photography sensors are unable to capture the wide range of illumination present in natural scenes, resulting in Low Dynamic Range (LDR) images that often suffer from over or underexposed regions, which can damage the details of the scene. High Dynamic Range (HDR) imaging has been developed to address these limitations. This technique combines several LDR images with different exposures to generate an HDR image. While HDR imaging can effectively recover details in static scenes, it may produce ghosting artifacts when used with dynamic scenes or hand-held camera scenarios.

![](images/3cf04ad12c59dcbb9a8b07462629576749764203c16a14a78afd60a966ef91cb.jpg)  
Figure 1. The proposed method generates high-quality images with few labeled samples when compared with several methods.

Historically, various techniques have been suggested to address such issues, such as alignment-based methods [3,10,27,37], patch-based methods [8,15,24], and rejectionbased methods [5, 11, 19, 20, 35, 40]. Two categories of alignment-based approaches exist: rigid alignment (e.g., homographies) that fail to address foreground motions, and non-rigid alignment (e.g., optical flow) that is error-prone. Patch-based techniques merge similar regions using patchlevel alignment and produce superior results, but suffer from high complexity. Rejection-based methods aim to eliminate misaligned areas before fusing images, but may result in a loss of information in motion regions.

As Deep Neural Networks (DNNs) become increasingly prevalent, the DNN-based HDR deghosting methods [9, 33, 36] achieve better visual results compared to traditional methods. However, these alignment approaches are error-prone and inevitably cause ghosting artifacts (see Figure 1 Kalantari’s results). AHDR [31, 32] proposes spatial attention to suppress motion and saturation, which effectively alleviate misalignment problems. Based on AHDR, ADNET [14] proposes a dual branch architecture using spatial attention and PCD-alignment [29] to remove ghosting artifacts. All these above methods directly learn the complicated HDR mapping function with abundant HDR ground truth data. However, it’s challenging to collect a large amount of HDR-labeled data. The reasons can be attributed to that 1) generating a ghost-free HDR ground truth sample requires an absolute static background, and 2) it is time-consuming and requires considerable manpower to do manual post-examination. This generates a new setting that only uses a few labeled data for HDR imaging.

Recently, FSHDR [22] attempts to generate a ghost-free HDR image with only few labeled data. They use a preliminary model trained with a large amount of unlabeled dynamic samples, and a few dynamic and static labeled samples to generate HDR pseudo-labels and synthesize artificial dynamic LDR inputs to improve the model performance of dynamic scenes further. This approach expects the model to handle both the saturation and the ghosting problems simultaneously, but it is hard to achieve under the condition of few labeled data, especially misaligned regions caused by saturation and motion (see Figure 1 FSHDR). In addition, FSHDR uses optical flow to forcibly synthesize dynamic LDR inputs from poorly generated HDR pseudo-labels, the errors in optical flow further intensify the degraded quality of artificial dynamic LDR images, resulting in an apparent distribution shift between LDR training and testing data, which hampers the performance of the network.

The above analysis makes it very challenging to directly generate a high-quality and ghost-free HDR image with few labeled samples. A reasonable way is to address the saturation problems first and then cope with the ghosting problems with a few labeled samples. In this paper, we propose a semi-supervised approach for HDR deghosting, named SSHDR, which consists of two stages: selfsupervised learning network for content completion and sample-quality-based iterative semi-supervised learning for deghosting. In the first stage, we pretrain a Saturated Mask AutoEncoder (SMAE), which learns the representation of HDR features to generate content of saturated regions by self-supervised learning. Specifically, considering that the saturated regions can be regarded as masking the short LDR input patches, inspired by [6], we randomly mask a high proportion of the short LDR input and expect the model to reconstruct a no-saturated HDR image from the remaining LDR patches in the first stage. This self-supervised approach allows the model to recover the saturated regions with the capability to effectively learn a robust representation for the HDR domain and map an LDR image to an HDR image. In the second stage, to prevent the overfitting problem with a few labeled training samples and make full use of the unlabeled samples, we iteratively train the model with a few labeled samples and a large amount of HDR pseudo-labels from unlabeled data. Based on the pretrained SMAE, a sample-quality-based iterative semisupervised learning framework is proposed to address the ghosting artifacts. Considering the quality of pseudo-labels is uneven, we develop an adaptive pseudo-labels selection strategy to pick high-quality HDR pseudo-labels (i.e., wellexposed, ghost-free) to avoid awful pseudo-labels hampering the optimization process. This selection strategy is guided by a few labeled samples and enhances the diversity of training samples in each epoch. The experiments demonstrate that our proposed approach can generate highquality HDR images with few labeled samples and achieves state-of-the-art performance on individual and cross-public datasets. Our contributions can be summarized as follows:

• We propose a novel and generalized HDR self-supervised pretraining model, which uses mask strategy to reconstruct an HDR image and addresses saturated problems from one LDR image.

• We propose a sample-quality-based semi-supervised training approach to select well-exposed and ghost-free HDR pseudo-labels, which improves ghost removal.

• We perform both qualitative and quantitative experiments, which show that our method achieves state-of-theart results on individual and cross-public datasets.

## 2. Related Work

## 2.1. HDR Deghosting Methods

The existing HDR deghosting methods include four categories: alignment-based method, patch-based method, rejection-based method, and CNN-based method.

Alignment-based Method. Rigid or non-rigid registration is mainly used in alignment-based approaches. Bogoni [3] estimated flow vectors to align with the reference images. Kang et al. [10] utilized optical flow to align images in the luminance domain to remove ghosting artifacts. Tomaszewska et al. [27] used SIFT feature to perform global alignments. Since the dense correspondence computed by alignment methods are error-prone, they cannot handle large motion and occlusion.

Rejection-based Method. Rejection-based methods detect and eliminate motion from static regions. Then they merge static inputs to get HDR images. Grosch et al. [5] estimated a motion map and used it to generate ghost-free HDR. Zhang et al. [40] obtained a motion weighting map using quality measurement on image gradients. Lee et al. [11] and Oh et al. [19] detected motion regions using rank minimization. However, rejection-based methods remove the misalignment of regions. It will result in a lack of content in moving regions.

Patch-based Method. Patch-based methods use patchlevel alignment to merge similar contents. Sen et al. [24] proposed a patch-based energy minimization approach that optimizes alignment and reconstruction simultaneously. Hu et al. [8] utilized a patch-match mechanism to produce aligned images. Although these methods have good performance, they suffer from high computational costs.

CNN-based Method. Kalantari et al. [9] used a CNN network to fuse LDR images that are aligned with optical flow. Wu et al. [30] used homography to align the camera motion and reconstructed HDR images by CNN. Yan et al. [31] proposed an attention mechanism to suppress motion and saturation. Yan et al. [34] designed a nonlocal block to release the constraint of locality receptive field for global merging HDR. Niu et al. [18] proposed HDR-GAN to recover missing content using generative adversarial networks. Ye et al. [38] proposed multi-step feature fusion to generate ghost-free images. Liu et al. [14] utilized the PCD alignment subnetwork to remove ghosts. However, these methods require a large number of labeled samples, which is difficult to collect.

## 2.2. Few-shot Learning (FSL)

Humans can successfully learn new ideas with relatively little supervision. Inspired by such ability, FSL aims to learn robust representations with few labeled samples. There are three main categories for FSL methods: databased category [1, 23, 25], which augment the experience with prior knowledge; model-based category [2, 17, 26], which shrinks the size of the hypothesis space using prior knowledge; algorithm-based category [4, 12, 39], which modifies the search for the optimal hypothesis using prior knowledge. For HDR deghosting, Prabhakar et al. [22] proposed a data-based category deghosting method, which uses artificial dynamic sequences synthesis for motion transfer. Still, it is hard to handle both the saturation and the ghosting problems simultaneously with few labeled data.

## 3. The Proposed Method

## 3.1. Data Distribution

Following the setting of few-shot HDR imaging [22], we utilize 1) N dynamic unlabeled LDR samples $\bar { U } { = } \{ L _ { 1 } ^ { U } , \ldots , L _ { N } ^ { U } \}$ , where each $L ^ { U }$ consists of three LDRs $( X _ { 1 } ^ { U } , X _ { 2 } ^ { U } , X _ { 3 } ^ { U } )$ with different exposures. 2) M static labeled LDR samples $S = \{ L _ { 1 } ^ { S } , \ldots , L _ { M } ^ { S ^ { - } } \}$ , where each $L ^ { S }$ consists of three LDRs $( X _ { 1 } ^ { \bar { S } } , \bar { X } _ { 2 } ^ { S } , X _ { 3 } ^ { S } )$ with different exposures and ground truth $Y ^ { S }$ . 3) K dynamic labeled LDR samples $\mathsf { \bar { D } } = \{ L _ { 1 } ^ { D } , \dots , L _ { K } ^ { D } \}$ , where each $L ^ { D }$ consists of three LDRs $( X _ { 1 } ^ { D } , \bar { X _ { 2 } ^ { D } } , X _ { 3 } ^ { D } )$ and ground truth $Y ^ { D }$ . Since it is difficult to collect labeled samples, we set K to be less than or equal to 5, and M is fixed at 5. While it is easy to capture unlabeled samples, N can be arbitrary.

## 3.2. Model Overview

Generating a non-saturated and ghost-free HDR image with few labeled samples is challenging. It is a proper way to address saturated problems first and then handle ghosting problems. As shown in Figure 2, we propose a semisupervised approach for HDR deghosting. Our approach consists of two stages: a self-supervised learning network for content completion and a sample-quality-based iterative semi-supervised learning for deghosting. In the first stage, we propose a multi-scale Transformer model based on selfsupervised learning with a saturated-masked autoencoder to make it capable of recovering saturated regions. In a word, we randomly mask LDR patches and reconstruct nonsaturated HDR images from the remaining LDR patches.

In the second stage, we propose a sample-quality-based iterative semi-supervised learning approach that learns to address ghosting problems. We finetune the pretrained model based on the first stage with a few labeled samples. Then, we iteratively train the model with labeled samples and unlabeled samples with pseudo-labels. Considering that the HDR pseudo-labels inevitably contain saturated and ghosting regions, which deteriorate the model performance, we propose an adaptive pseudo-labels selection strategy to pick high-quality HDR pseudo-labels to avoid awful pseudo-labels hampering the optimization process.

## 3.3. Self-supervised Learning Stage

Input. Considering that there are more saturated regions in the medium $( X _ { 2 } ^ { U } )$ and long exposure frames $( X _ { 3 } ^ { U } )$ of unlabeled data $U ,$ we first transform the short exposure frame $( X _ { 1 } ^ { U } )$ ) into a new medium $( X _ { 2 ^ { \prime } } ^ { U } )$ and long exposure frames $( X _ { 3 ^ { ' } } ^ { U } )$ by exposure adjustment,

$$
X _ { i ^ { \prime } } ^ { U } = c l i p ( ( \frac { ( X _ { 1 } ^ { U } ) ^ { \gamma } \times t _ { i } } { t _ { 1 } } ) ^ { \frac { 1 } { \gamma } } ) , \quad i = 2 , 3 .\tag{1}
$$

Then following previous work [9, 30], we map the LDR input images $X _ { 1 } ^ { U } , X _ { 2 ^ { \prime } } ^ { U } , X _ { 3 ^ { \prime } } ^ { U }$ to HDR domain by gamma correction to get $H _ { i }$

$$
H _ { i ^ { \prime } } = ( X _ { i ^ { \prime } } ^ { U } ) ^ { \gamma } / t _ { i } .\tag{2}
$$

Note that $X _ { 1 ^ { \prime } } ^ { U } { = } X _ { 1 } ^ { U } , t _ { i }$ denotes the exposure time of LDR image $X _ { i } , \cdot$ γ represents the gamma correction parameter, we set $\gamma$ to 2.2. Then, we concatenate $X _ { i ^ { \prime } }$ and $H _ { i ^ { \prime } }$ along the channel dimension to get a 6-channel input $I _ { i } { = } [ X _ { i ^ { \prime } } , H _ { i ^ { \prime } } ] ,$ we subsequently mask the input $I _ { i }$ patches to get $I _ { i } ^ { ' }$ . Concretely, we divide the input into non-overlapping patches and randomly mask a subset of these patches with a high mask ratio (75%) (see Figure 3). Note that the patch size is $8 \times 8 .$ . Considering that the masking strategy is another way to destruct the saturated regions, we intend the model to learn a robust representation to recover these saturated regions. Finally, $I ^ { ' } { = } \{ I _ { 1 } ^ { ' } , I _ { 2 } ^ { ' } , I _ { 3 } ^ { ' } \}$ is the input of the model.

![](images/e6ccc7728d8fdd17e9439c667cfcd94cda6e5b3fe6d4957d101dcdbdb79372d5.jpg)  
Figure 2. The overview of our framework.

![](images/a5ac96687355737241d7f5ff8117242284bec98618ede3b73576110b5437f7a8.jpg)  
Figure 3. The detailed procedure of Stage 1. To recover the saturated regions, we utilize the short exposure frame as input and ground truth.

Model. Our SMAE self-supervised training-based multiscale Transformer consists of a feature extraction module, hallucination module, and Multi-Scale Residual Swin Transformer fusion Module (MSRSTM). The details in our model are included in the Appendix.

Hallucination Module. We first adopt three convolutional layers to extract shallow feature $F _ { i }$ . Then, we divide the shallow feature $F _ { i }$ into non-overlapping patches ${ \overline { { F _ { i } } } } ,$ and map each patch $\overline { { F _ { i } } }$ into query, keys and values. Subsequently, we calculate the similarity map between $\overline { { q } }$ and ${ \overline { { k } } } ,$ and perform the Softmax function to get the attention weight. Finally, we apply the attention weight to v to get $F _ { s } ^ { i }$

$$
\begin{array} { r l } & { \overline { { q } } = \overline { { F _ { 2 } } } W _ { q } , \quad \overline { { k _ { i } } } = \overline { { F _ { i } } } W _ { k } , \quad \overline { { v _ { i } } } = \overline { { F _ { i } } } W _ { v } , \quad i = 1 , 3 } \\ & { \overline { { F _ { s } ^ { i } } } = \mathrm { S o f t m a x } ( \overline { { q } } \overline { { k _ { i } } } ^ { T } / \sqrt { d } + b ) \overline { { v _ { i } } } , } \end{array}\tag{3}
$$

where b represents a learnable position encoding, d denotes the dimension of ${ \overline { { q } } } .$

MSRSTM. To merge more information from different exposure regions, inspired by [13], we propose a Multi-Scale Residual Swin Transformer Module (MSRSTM). First, $F _ { s } ^ { 1 } , F _ { s } ^ { 2 } , F _ { s } ^ { 3 }$ is concatenated along the channel dimension to get the input of MSRSTM. Note that $F _ { s } ^ { 2 }$ denotes $F _ { 2 }$ Then, MSRSTM merges a long range of information from different exposure regions. MSRSTM consists of multiple multi-scale Swin Transformer layers (STL), a few convolutional layers, and a residual connection. Given the input feature $F _ { o u t , i } ^ { N - 1 }$ of i-th MSRSTM, the output $F _ { o u t , i } ^ { N }$ ofMSRSTM can be formulated as follows :

$$
\begin{array} { r } { F _ { S T L , i } ^ { N } = C o n v ( ( C o n c a t ( S T L _ { i } ^ { N , l _ { 1 } } ( F _ { o u t , i } ^ { N - 1 } ) , } \\ { S T L _ { i } ^ { N , l _ { 2 } } ( F _ { o u t , i } ^ { N - 1 } ) , S T L _ { i } ^ { N , l _ { 3 } } ( F _ { o u t , i } ^ { N - 1 } ) ) , } \end{array}\tag{4}
$$

$$
F _ { o u t , i } ^ { N } = C o n v ( F _ { S T L , i } ^ { N } ) + F _ { o u t , i } ^ { N - 1 } ,\tag{5}
$$

where $S T L _ { i } ^ { N , l _ { j } } ( \cdot )$ represents the N-th Swin Transformer layer of the $l _ { j }$ scale in the i-th MSRSTM, $F _ { o u t , i } ^ { N - 1 }$ denotes the input feature of the N-th Swin Transformer layer in the i-th MSRSTM.

Loss Function. Since unlabeled samples do not have HDR ground truth labels, we calculate the self-supervised loss in the LDR domain. We first use function ω to transform the predicted HDR image $\hat { Y }$ to short, medium, and long exposure LDR images $\hat { Y _ { i } }$

$$
\hat { Y } _ { i } = \omega ( \hat { Y } ) = ( \hat { Y } \times t _ { i } ) ^ { \frac { 1 } { \gamma } } .\tag{6}
$$

To recover the saturated regions, we transform the short exposure frame (since the predicted HDR in this stage is aligned to the short exposure frame) to new short, medium, and long exposure frames by ground truth generation. Then, we regard the new exposure frames as the ground truth $X _ { i } ^ { G T }$ of the model,

$$
X _ { i } ^ { G T } = ( \frac { ( X _ { 1 } ^ { U } ) ^ { \gamma } \times t _ { i } } { t _ { 1 } } ) ^ { \frac { 1 } { \gamma } } , \quad i = 1 , 2 , 3 .\tag{7}
$$

Finally, we calculate $L _ { 1 }$ self-supervised loss between $\hat { Y _ { i } }$ and $X _ { i } ^ { G T }$

$$
L _ { S S L } = \sum _ { i = 1 } ^ { 3 } | | \hat { Y } _ { i } - X _ { i } ^ { G T } | | _ { 1 } .\tag{8}
$$

## 3.4. Semi-supervised Learning Stage

Finetune. At the beginning of this stage, to improve the saturated regions and further learn to handle ghosting regions, we first finetune the pretrained model with a few dynamic samples D and static labeled samples S. Here we apply $\mu \cdot$ -law to map the linear domain image to the tonemapped domain image,

$$
T ( x ) = \frac { l o g ( 1 + \mu x ) } { l o g ( 1 + \mu ) } ,\tag{9}
$$

where $T ( x )$ is the tonemap function, $\mu { = } 5 0 0 0$ . Then we calculate the reconstruction loss $L _ { r e c o n }$ and perceptual loss $L _ { p e r c e p }$ between the predicted HDR $\hat { Y } _ { 0 } ^ { D } , \hat { Y } _ { 0 } ^ { S }$ and ground truth HDR $Y _ { 0 } ^ { D } , Y _ { 0 } ^ { S }$

$$
L _ { r e c o n } = | | T ( \hat { Y } _ { 0 } ^ { D } ) - T ( Y _ { 0 } ^ { D } ) | | _ { 1 } + | | T ( \hat { Y } _ { 0 } ^ { S } ) - T ( Y _ { 0 } ^ { S } ) | | _ { 1 } ,\tag{10}
$$

$$
L _ { p e r c e p } = | | \phi _ { i , j } ( T ( \hat { Y } _ { 0 } ^ { D } ) ) - \phi _ { i , j } ( T ( Y _ { 0 } ^ { D } ) ) | | _ { 1 }
$$

$$
+ | | \phi _ { i , j } ( T ( \hat { Y } _ { 0 } ^ { S } ) ) - \phi _ { i , j } ( T ( Y _ { 0 } ^ { S } ) ) | | _ { 1 } ,\tag{11}
$$

$$
L _ { f i n e t u n e } = L _ { r e c o n } + \lambda L _ { p e r c e p } ,\tag{12}
$$

where $\phi _ { i , j }$ represents the j-th convolutional layer and the i-th max-pooling layer in VGG19, $\lambda { = } 1 e ^ { - 2 }$

Iteration. To prevent the overfitting problem with a few labeled training samples and exploit unlabeled samples, we further generate the pseudo-labels $\hat { Y } _ { t } ^ { U }$ of unlabeled data. Concretely, we iteratively and adaptively train the model with a few dynamic and static samples $D$ and S and a large number of unlabeled samples U. Specifically, at timestep $t ,$ we use model $N _ { t }$ to predict the pseudo-labels $\hat { Y } _ { t } ^ { U }$ of unlabeled data. Then, we train the model $N _ { t }$ with a few labeled and pseudo-labeled samples to get the model $N _ { t + 1 }$ at timestep $t + 1$ . Note that we use finetune model to generate unlabel HDR pseudo-labels $\hat { Y } _ { 0 } ^ { U }$ at timestep $t { = } 0$ . Finally, at each timestep in the refinement stage, we calculate the reconstruction loss and perceptual loss as follows,

$$
\begin{array} { r l r } & { } & { { \cal L } _ { I t e r a t i o n } = { \cal L } _ { r e c o n , t + 1 } ^ { D } + { \cal L } _ { r e c o n , t + 1 } ^ { S } + \displaystyle \sum _ { i = 1 } ^ { N } W _ { t + 1 } ^ { U _ { i } } { \cal L } _ { r e c o n , t + 1 } ^ { U _ { i } } } \\ & { } & \\ & { } & { \quad \quad + \lambda ( { \cal L } _ { p e r c e p , t + 1 } ^ { D } + { \cal L } _ { p e r c e p , t + 1 } ^ { S } + \displaystyle \sum _ { i = 1 } ^ { N } W _ { t + 1 } ^ { U _ { i } } { \cal L } _ { p e r c e p , t + 1 } ^ { U _ { i } } ) , } \end{array}\tag{13}
$$

where $\lambda { = } 1 e ^ { - 2 } . W _ { t { + } 1 } ^ { U _ { i } }$ is the weight factor of unlabeled data $U _ { i }$ . To get loss weight $W _ { t + 1 } ^ { U _ { i } }$ , please refer to the next section in detail.

APSS. Since the HDR pseudo-labels inevitably contain saturated and ghosted samples, we propose an Adaptive Pseudo-labels Selection Strategy (APSS) to pick wellexposed and ghost-free HDR pseudo-labels to avoid awful pseudo-labels hampering the optimization process. Specifically, at timestep t, we use model $N _ { t }$ to predict HDR images with dynamic and static labeled samples $\hat { Y } _ { t } ^ { D }$ and $\hat { Y } _ { t } ^ { S }$ Then we use function $\omega$ to map the predicted HDR image to medium exposure image $\tilde { Y } _ { t } ^ { D \cup S }$ and calculate the loss between $\tilde { Y } _ { t } ^ { D \cup \tilde { S } }$ and original medium exposure LDR image $X _ { 2 , t } ^ { D \cup S }$ in well exposure regions to get $L _ { s e l e c t , t } ^ { \mathrm { \bar { D } \cup S } }$

$$
\begin{array} { r } { L _ { s e l e c t , t } ^ { D \cup S } = | | m a s k ( \omega ( \hat { Y } _ { t } ^ { D } ) ) - m a s k ( \omega ( X _ { 2 , t } ^ { D } ) ) | | _ { 1 } } \\ { + | | m a s k ( \omega ( \hat { Y } _ { t } ^ { S } ) ) - m a s k ( \omega ( X _ { 2 , t } ^ { S } ) ) | | _ { 1 } , } \end{array}\tag{14}
$$

where mask(·) denotes masking the over and underexposure regions. Subsequently, we sort all patches’ losses, and adopt $\sigma ( \cdot , \cdot )$ function to get $\beta$ percentile (85th) loss as a selection threshold $\tau _ { t }$

$$
\tau _ { t } = \sigma ( L _ { s e l e c t , t } ^ { D \cup S } , \beta ) .\tag{15}
$$

Furthermore, we use model $N _ { t }$ to predict pseudo-labels $\hat { Y } _ { t } ^ { U }$ of unlabeled samples, similar to the operation of labeled data mentioned above. We then use ω function to map $\hat { Y } _ { t } ^ { U }$ to medium exposure to get $\tilde { Y } _ { t } ^ { U }$ and calculate the loss between $\tilde { Y } _ { t } ^ { U }$ and original medium exposure LDR image $X _ { 2 , t } ^ { U }$ to get $L _ { s e l e c t , t } ^ { U } = \{ L _ { s e l e c t , t } ^ { U _ { 1 } } , L _ { s e l e c t , t } ^ { U _ { 2 } } , \ldots , L _ { s e l e c t , t } ^ { U _ { N } } \}$ . If the current loss $L _ { s e l e c t , t } ^ { U _ { i } }$ is greater than $\tau _ { t }$ , we consider the pseudo-label to be of poor quality, which has more saturated and ghosted regions. Then we will give a lower weight which tends to decay linearly in the next training iteration.

$$
L _ { s e l e c t , t } ^ { U } = | | m a s k ( \omega ( \hat { Y } _ { t } ^ { U } ) ) - m a s k ( \omega ( X _ { 2 , t } ^ { U } ) ) | | _ { 1 } ,\tag{16}
$$

$$
m _ { t } ^ { U } = m a x ( L _ { s e l e c t , t } ^ { U } ) ,\tag{17}
$$

$$
\begin{array} { r } { W _ { t + 1 } ^ { U _ { i } } = \left\{ \begin{array} { l l } { 1 } & { L _ { s e l e c t , t } ^ { U _ { i } } \leq \tau _ { t } } \\ { \frac { m _ { t } ^ { U } - L _ { s e l e c t , t } ^ { U _ { i } } } { m _ { t } ^ { U } - \tau _ { t } } } & { L _ { s e l e c t , t } ^ { U _ { i } } > \tau _ { t } } \end{array} \right. } \end{array}\tag{18}
$$

where $X _ { 2 , t } ^ { U }$ is the unlabeled medium exposure image in timestep $t , m _ { t } ^ { U }$ is the largest selection loss of unlabeled samples in timestep t, $\overline { { W _ { t + 1 } ^ { U _ { i } } } }$ is the weight factor of $U _ { i }$ smaple in the $t + 1$ training iteration.

## 4. Experiments

Datasets. We train all the methods on two public datasets, Kalantari’s [9] and Hu’s dataset [7]. Kalantari’s dataset includes 74 training samples and 15 testing samples. Three different LDR images in a sample are captured with exposure biases of $\{ - 2 , 0 , + 2 \}$ or $\left\{ - 3 , 0 , + 3 \right\}$ . Hu’s dataset is captured at three exposure levels $( i . e . , \{ - 2 , 0 , + 2 \} )$ ). There are

![](images/1349521eb4bcced1df0d49fdecee9a462a9fe62714bd50b80200ac23b8ee6e3e.jpg)  
Figure 4. Examples of Kalantari’s [9] and Hu’s [7] datasets (top row) and Tursun’s [28] and Prabhakar’s [21] datasets (bottom row). Note that we directly evaluate the methods on Tursun’s and Prabhakar’s datasets with the checkpoint trained on Kalantari’s dataset.

85 training samples and 15 testing samples in Hu’s dataset. We train all comparison methods with the same set of images. Concretely, we randomly choose $K \in \{ 1 , 5 \}$ dynamic labeled samples and $Q { = } 5$ static labeled samples for training in all methods. Furthermore, for each $K ,$ , we evaluate all methods for 5 runs denoted as 5-way in Table 1. In addition, since FSHDR [22] and our method exploit unlabeled samples, we also use the rest of the dataset samples as unlabeled data U. Finally, to verify generalization performance, we evaluate all methods on Tursun’s dataset [28] that does not have ground truth and Prabhakar’s dataset [21].

Evaluation Metrics. We calculate five common metrics used for testing, i.e., PSNR-L, PSRN-µ, SSIM-L, SSIM-µ, and HDR-VDP-2 [16], where ‘-L’ denotes linear domain, $\cdot _ { - \mu } ,$ denotes tonemapping domain.

Implementation Details. The window size in MSRSTM is

2×2, 4 ×4 and $8 \times 8 .$ In the training stage, we crop the 128 $\times \ 1 2 8$ patches with stride 64 for the training dataset. We use the Adam optimizer, and set the batch size and learning rate as 4 and $0 . 0 0 0 5$ , receptively. And we set $\beta _ { 1 } { = } 0 . 9$ $\beta _ { 2 } { = } 0 . 9 9 9$ , and $\epsilon { = } 1 e ^ { - 8 }$ in the Adam optimizer. We implement our model using PyTorch with 2 NVIDIA GeForce 3090 GPUs and train for 200 epochs.

## 4.1. Comparison with State-of-the-art Methods

To evaluate our model, we carry out quantitative and qualitative experiments comparing with several state-ofthe-art methods, including patch-based classical methods: Sen [24], Hu [8], and deep learning-based methods: Kalantari [9], DeepHDR [30], AHDRNet [31], ADNet [14], FSHDR [22]. We use the codes provided by the authors. Evaluation on Kalantari’s and Hu’s Datasets. In Figure 4 (a) and (b), we compare our method with other stateof-the-art methods in the 5-shot scenario. Due to insufficient labeled samples, large motion, and saturation, most comparing methods suffer from color distortion and ghosting artifacts in these two datasets. Kalantari’s method and DeepHDR produce undesirable artifacts and color distortion (see Figure 4 (a)(b)). There are two reasons behind that: misalignment of optical flow and homographies and the lack of labeled data. Although AHDRNet and ADNET are proposed to suppress motion and saturation with attention mechanisms, they cannot reconstruct ghost-free HDR images with few labeled samples. They also produce severe ghosting artifacts (see the red block in Figure 4 (a)(b)). FSHDR exploits unlabeled data to alleviate ghosts under the constraint of a few labeled samples, but it is difficult to handle both ghosting and saturation problems simultaneously. We can see that FSHDR still suffers from ghosting artifacts which leaves an obvious hand artifact in the car (see the red block in Figure 4 (a)). Thanks to the proposed SMAE and sample-quality-based iterative learning strategy, which first address the saturation problems using SMAE and then adaptively sample well-exposed and ghostfree pseudo-labels to handle ghosting problems, we can reconstruct ghost-free HDR images with only a few labeled samples.

Table 1. The evaluation results on Kalantari’s [9] and Hu’s [7] datasets. The best and the second best results are highlighted in Bold and Underline, respectively.
<table><tr><td>Dataset</td><td>Metric</td><td>Setting</td><td>Kalantari</td><td>DeepHDR</td><td>AHDRNet</td><td>ADNet</td><td>FSHDR</td><td>Ours</td></tr><tr><td rowspan="3">Kalantari</td><td>PSNR-l PSNR-µ</td><td>5way-5shot</td><td> $3 9 . 3 7 { \pm } 0 . 1 2 $   $3 9 . 8 6 { \pm } 0 . 1 9$ </td><td> $3 8 . 2 5 { \pm } 0 . 2 9 $  38.62±0.27</td><td> $4 0 . 6 1 { \pm } 0 . 1 0$   $4 1 . 0 5 { \pm } 0 . 3 2 $ </td><td> $4 0 . 7 8 { \pm } 0 . 1 5 $  40.93±0.38</td><td> $\underline { { 4 1 . 3 9 } } \pm 0 . 1 2$   $\underline { { 4 1 . 4 0 { \pm } 0 . 1 3 } }$ </td><td> $\pm 1 . 5 4 \pm 0 . 1 0$   $\mathbf { 4 1 . 6 1 } { \pm } 0 . 0 8$ </td></tr><tr><td>PSNR-l</td><td>5way-1shot</td><td>36.94±0.44 37.33±1.21</td><td>36.67±0.67 37.01±1.68</td><td>38.83±0.39</td><td>38.96±0.35</td><td> $\overline { { 4 1 . 0 4 2 0 . 1 1 } }$ </td><td> $\overline { { 4 1 . 1 4 \pm 0 . 1 1 } }$ </td></tr><tr><td>PSNR-µ PSNR-l</td><td></td><td> $4 1 . 3 6 { \pm } 0 . 2 5 $ </td><td>40.73±0.66</td><td>39.15±1.04 46.37±0.76</td><td>39.08±1.06</td><td> $4 1 . 1 3 { \pm } 0 . 0 7 $ </td><td> $\pm 1 . 2 5 { \pm } 0 . 0 5$   $\pm 7 . 4 1 \pm 0 . 1 2$ </td></tr><tr><td rowspan="3">Hu</td><td>PSNR-µ</td><td>5way-5shot</td><td>38.95±0.14</td><td>39.92±0.22</td><td>43.42±0.44</td><td>46.88±0.81 43.79±0.48</td><td> $\underline { { 4 7 . 1 3 } } \pm 0 . 1 3 $   $4 3 . 9 8 { \pm } 0 . 2 7$ </td><td> $4 4 . 2 4 \pm 0 . 1 7$ </td></tr><tr><td>PSNR-l  $\mathrm { P S N R } { \cdot } { \mu }$ </td><td>5way-1shot</td><td> $3 8 . 6 7 { \scriptstyle \pm 0 . 4 3 }$  36.83±0.62</td><td>37.82±0.86 38.49±1.07</td><td>44.64±0.80 42.37±1.42</td><td>44.75±0.84</td><td> $\overline { { 4 4 . 9 4 2 0 . 2 3 } }$ </td><td> $\overline { { 4 5 . 0 4 \pm 0 . 1 6 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>42.41±1.20</td><td> $4 2 . 5 0 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $4 2 . 5 5 { \scriptstyle \pm 0 . 4 4 }$ </td></tr></table>

Table 2. Further evaluation results on Kalantari’s [9], Hu’s [7] and Prabhakar’s datasets [21]. The best and the second best results are highlighted in Bold and Underline in each setting, respectively.
<table><tr><td colspan="6">Kalantari</td><td>Hu</td><td colspan="4"></td><td></td></tr><tr><td colspan="2"></td><td>PSNR-l</td><td>PSNR-µ</td><td>SSIM-l</td><td> $\overline { { \mathbf { S S I M } - \mu } }$ </td><td>HV2</td><td>PSNR-l</td><td> $\textstyle { \overline { { \operatorname { P S N R } - \mu } } }$ </td><td>SSIM-l</td><td> $\overline { { \mathbf { S S I M } - \mu } }$ </td><td>HV2</td></tr><tr><td rowspan="4"> $S _ { 1 }$ </td><td>Sen</td><td>38.57</td><td>40.94</td><td>0.9711</td><td>0.9780</td><td>64.71</td><td>33.58</td><td>31.48</td><td>0.9634</td><td>0.9531</td><td>66.39</td></tr><tr><td>Hu</td><td>30.84</td><td>32.19</td><td>0.9408</td><td>0.9632</td><td>62.05</td><td>36.94</td><td>36.56</td><td>0.9877</td><td>0.9824</td><td>67.58</td></tr><tr><td>FSHDR</td><td>40.97</td><td>41.11</td><td>0.9864</td><td>0.9827</td><td>67.08</td><td>42.15</td><td>41.14</td><td>0.9904</td><td>0.9891</td><td>71.35</td></tr><tr><td>Ours (K=0)</td><td>41.12</td><td>41.20</td><td>0.9866</td><td>0.9868</td><td>67.16</td><td>42.99</td><td>41.30</td><td>0.9912</td><td>0.9903</td><td>72.18</td></tr><tr><td rowspan="2"> $S _ { 2 }$ </td><td>Ours (K=1)</td><td>41.14</td><td>41.25</td><td>0.9866</td><td>0.9869</td><td>67.20</td><td>45.04</td><td>42.55</td><td>0.9938</td><td>0.9928</td><td>73.23</td></tr><tr><td>Ours (K=5)</td><td>41.54</td><td>41.61</td><td>0.9879</td><td>0.9880</td><td>67.33</td><td>47.41</td><td>44.24</td><td>0.9974</td><td>0.9936</td><td>74.49</td></tr><tr><td rowspan="6"> $S _ { 3 }$ </td><td>Kalantari</td><td>41.22 40.91</td><td>41.85</td><td>0.9848</td><td>0.9872</td><td>66.23</td><td>43.76</td><td>41.60</td><td>0.9938</td><td>0.9914</td><td>72.94</td></tr><tr><td>DeepHDR</td><td>41.23</td><td>41.64</td><td>0.9863</td><td>0.9857</td><td>67.42</td><td>41.20</td><td>41.13</td><td>0.9941</td><td>0.9870</td><td>70.82</td></tr><tr><td>AHDRNet</td><td></td><td>41.87</td><td>0.9868</td><td>0.9889</td><td>67.50</td><td>49.22</td><td>45.76</td><td>0.9980</td><td>0.9956</td><td>75.04</td></tr><tr><td>ADNET</td><td>41.31</td><td>41.80</td><td>0.9871</td><td>0.9883</td><td>67.57</td><td>50.38</td><td>46.79</td><td>0.9987</td><td>0.9948</td><td>76.32</td></tr><tr><td>FSHDR</td><td>41.79</td><td>41.92</td><td>0.9876</td><td>0.9851</td><td>67.70</td><td>49.56</td><td>45.90</td><td>0.9984</td><td>0.9945</td><td>75.25</td></tr><tr><td>Ours</td><td>41.68</td><td>41.97</td><td>0.9889</td><td>0.9895</td><td>67.77</td><td>50.31</td><td>46.88</td><td>0.9988</td><td>0.9957</td><td>76.21</td></tr><tr><td rowspan="6"> $S _ { 4 }$ </td><td>Kalantari</td><td>25.87</td><td>21.44</td><td>0.8610</td><td>0.9176</td><td>60.00</td><td>10.23</td><td>16.95</td><td>0.6903</td><td>0.8346</td><td>49.10</td></tr><tr><td>DeepHDR</td><td>25.92</td><td>21.43</td><td>0.8597</td><td>0.9170</td><td>60.02</td><td>25.48</td><td>20.86</td><td>0.9215</td><td>0.8354</td><td>66.83</td></tr><tr><td>AHDRNet</td><td>26.62</td><td>22.08</td><td>0.8737</td><td>0.9238</td><td>58.89</td><td>11.44</td><td>17.84</td><td>0.6732</td><td>0.8389</td><td>52.79</td></tr><tr><td>ADNET</td><td>25.76</td><td>21.39</td><td>0.8686</td><td>0.8217</td><td>60.36</td><td>10.86</td><td>18.09</td><td>0.6915</td><td>0.8399</td><td>49.28</td></tr><tr><td>FSHDR</td><td>28.03</td><td>22.01</td><td>0.8751</td><td>0.9203</td><td>60.53</td><td>12.82</td><td>19.37</td><td>0.7442</td><td>0.8347</td><td>55.34</td></tr><tr><td>Ours</td><td>27.91</td><td>22.45</td><td>0.8764</td><td>0.9252</td><td>61.02</td><td>30.29</td><td>21.56</td><td>0.9440</td><td>0.8456</td><td>67.07</td></tr><tr><td rowspan="6"> $S _ { 5 }$ </td><td>Kalantari</td><td>31.24</td><td>33.10</td><td>0.9527</td><td>0.9593</td><td>63.99</td><td>19.82</td><td>18.63</td><td>0.7679</td><td>0.8742</td><td>59.50</td></tr><tr><td>DeepHDR</td><td>30.75</td><td>29.01</td><td>0.9244</td><td>0.9223</td><td>63.26</td><td>19.84</td><td>18.70</td><td>0.7698</td><td>0.8752</td><td>59.48</td></tr><tr><td>AHDRNet</td><td>31.84</td><td>33.49</td><td>0.9588</td><td>0.9606</td><td>64.40</td><td>20.80</td><td>20.51</td><td>0.8259</td><td>0.9136</td><td>59.79</td></tr><tr><td>ADNET</td><td>31.08</td><td>33.50</td><td>0.9536</td><td>0.9636</td><td>63.88</td><td>20.78</td><td>20.80</td><td>0.8268</td><td>0.9173</td><td>59.71</td></tr><tr><td>FSHDR</td><td>32.70</td><td>32.24</td><td>0.9553</td><td>0.9465</td><td>64.37</td><td>20.23</td><td>19.71</td><td>0.7929</td><td>0.9026</td><td>59.63</td></tr><tr><td>Ours</td><td>32.72</td><td>34.49</td><td>0.9586</td><td>0.9713</td><td>64.45</td><td>20.69</td><td>21.96</td><td>0.8257</td><td>0.9207</td><td>59.76</td></tr></table>

The quantitative results under the constraint of few shot scenarios on two dataset are shown in Table 1. We report means and 95% margin of variations for 5 and 1 shot cases across 5 runs. Our method achieves state-of-the-art performance on all metrics of two datasets, while most other methods perform poorly with only a few labeled samples. We show that our proposed method surpasses second-best method by 0.15db and 0.21db in terms of PSNR-l and $\mathrm { P S N R } { \cdot } { \mu }$ for 5way-5shot setting on Kalantari’s dataset, and it also improves by 0.28db and 0.26db for 5way-5shot setting on Hu’s dataset. For 5way-1shot setting, our method consistently outperforms other approaches on two datasets.

In addition, as shown in Table 2, we further compare our method with major HDR deghosting approaches in zeroshot setting $S _ { 1 }$ , few-shot setting $S _ { 2 } ,$ , and fully supervised setting $S _ { 3 }$ . Note that we use all the dynamic labeled samples without static and unlabeled samples for plain training in setting $S _ { 3 }$ . Our zero-shot approach outperforms other methods in zero-shot setting on two datasets. It also outperforms some 5-shot and fully supervised methods in most metrics. Finally, our few-shot and fully supervised approaches achieve state-of-the-art performance among two datasets.

Table 3. Ablation study of 5 shot scenario on Kalantari’s dataset.
<table><tr><td>#</td><td>Model</td><td>PSNR-l</td><td> $\mathbf { P S N R } { \cdot } { \boldsymbol { \mu } }$ </td><td>HDR-VDP-2</td></tr><tr><td>B1</td><td>SSHDR</td><td>41.54</td><td>41.61</td><td>67.33</td></tr><tr><td>B2</td><td>Stage2Net</td><td>41.31</td><td>41.43</td><td>67.21</td></tr><tr><td>B3</td><td>w/o APSS</td><td>41.49</td><td>41.45</td><td>67.29</td></tr><tr><td>B4</td><td>AHDR*</td><td>41.48</td><td>41.51</td><td>67.30</td></tr><tr><td>B5</td><td>FSHDR*</td><td>41.41</td><td>41.43</td><td>67.26</td></tr><tr><td>B6</td><td>Vanilla-AHDR</td><td>40.61</td><td>41.05</td><td>66.95</td></tr><tr><td>B7</td><td>Vanilla-FSHDR</td><td>41.39</td><td>41.40</td><td>67.25</td></tr></table>

Evaluation Generalization Across Different Datasets. We compare our method against other approaches on Kalantari’s, Hu’s, Tursun’s, and Prabhakar’s datasets to verify generalization performance. We directly evaluate the methods with the checkpoint trained on Kalantari’s dataset and show the qualitative results on Tursun’s and Prabhakar’s datasets in Figure 4 (c)(d). More results are included in the Appendix. In Figure 4 (c), since the lady’s motion is large, all the comparison methods cannot remove the ghosting artifacts. In Figure 4 (d), the comparison methods have obvious color distortion and ghosting artifacts on the floor and in the ceil. It shows that other methods have poor generalization performance across different datasets. All these methods address both the saturation and ghosting problems simultaneously. They cannot learn a robust representation to reconstruct a high-quality HDR image. Thanks to our SMAE and sample-quality-based iterative learning strategy, we can learn a robust representation to recover saturated regions and remove ghosting artifacts.

In Table 2, setting $S _ { 4 }$ denotes that we utilize the checkpoint trained on Kalantari’s or Hu’s dataset under 5 shot scenario to evaluate on Hu’s or Kalantari’s dataset reversely. Setting $S _ { 5 }$ represents that we train on Kalantari’s or Hu’s dataset under 5 shot scenario and evaluate on Prabhakar’s dataset. Our method achieves better numerical performance in terms of PSNR-l and $\mathrm { P S N R } { - } \mu$ . It demonstrates that our method generalizes well across different datasets.

## 4.2. Ablation Studies

We conduct ablation studies on Kalantari’s dataset under the condition of 5 shot scenario across 5 runs and analyze the importance of each component. We use the following variants of our whole SSHDR model: 1) SSHDR: The full model of SSHDR network trained with two entire stages. 2) Stage2Net: The model only trained in the second stage without SMAE pre-training. 3) w/o APSS: The model trained with two stages without using sample-quality-based pseudo-labels selection strategy. 4) AHDR<sup>∗</sup>: The AHDR model is trained with our proposed two stages strategy. 5) FSHDR<sup>∗</sup>: Our model is trained with the FSHDR strategy. 6) Vanilla-AHDR: The vanilla AHDR model trained in 5 shot scenario. 7) Vanilla-FSHDR: The vanilla FSHDR model trained with 5 labeled samples.

![](images/a59dc88f07871738ff7347fda980554bf43bb6acf2c393b9ac58c7357761961f.jpg)  
Figure 5. Visual results of poor pseudo-labels.

SMAE Pre-training. As shown in Table 3, the performance of Stage2Net is significantly decreased compared with SSHDR. Since the SMAE learns a robust representation to generate content of saturated regions, it helps to improve the saturated regions. In a word, it demonstrates that the SMAE pre-training stage is an effective mechanism.

Pseudo-labels Selection Strategy. Since the samplequality-based pseudo-labels selection strategy can exclude saturated and ghosted samples (see Figure 5), the model can be guided in a correct optimization direction which is effective for ghost removal. When we remove the pseudolabels selection strategy, the performance of the model without APSS is dropped.

Two Stages Strategy. In Table 3, we report the performance of AHDR<sup>∗</sup>. It achieves a significant increment compared with the vanilla AHDR model, which demonstrates the effectiveness of the overall two stages strategy.

Proposed Model Architecture. When we replace our two stages strategy with the FSHDR strategy, the numerical results increase compared with FSHDR. It shows that our proposed model architecture is also sound.

## 5. Conclusion

We propose a novel semi-supervised deghosting method for few-shot HDR problem via two stages of completing saturation and deghosting. In the first stage, a Saturated Mask AutoEncoder is proposed to learn a robust representation and reconstruct a non-saturated HDR image with a self-supervised mechanism. In the second stage, we propose an adaptive pseudo-label selection strategy to avoid the effects of mislabeled samples. Finally, our approach shows superiority over the existing state-of-the-art methods.

## References

[1] Sagie Benaim and Lior Wolf. One-shot unsupervised cross domain translation. Advances in neural information processing systems (NIPS), 31, 2018. 3

[2] Luca Bertinetto, Joao F Henriques, Jack Valmadre, Philip˜ Torr, and Andrea Vedaldi. Learning feed-forward one-shot learners. Advances in neural information processing systems (NIPS), 29, 2016. 3

[3] L. Bogoni. Extending dynamic range of mono-chrome and color images through fusion. In IEEE International Conference on Pattern Recognition (ICPR), pages 7–12, 2000. 1, 2

[4] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Modelagnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning (ICML), pages 1126–1135. PMLR, 2017. 3

[5] Thorsten Grosch. Fast and robust high dynamic range image generation with camera and object movement. In IEEE Conference ofVision , Modeling and Visualization (VMV), 2006. 1, 2

[6] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´ vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, 2022. 2

[7] Jinhan Hu, Gyeongmin Choe, Zeeshan Nadir, Osama Nabil, Seok-Jun Lee, Hamid Sheikh, Youngjun Yoo, and Michael Polley. Sensor-realistic synthetic data engine for multiframe high dynamic range photography. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 516–517, 2020. 5, 6, 7

[8] Jun Hu, O. Gallo, K. Pulli, and Xiaobai Sun. HDR deghosting: How to deal with saturation? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1163–1170, 2013. 1, 3, 6

[9] Nima Khademi Kalantari and Ravi Ramamoorthi. Deep high dynamic range imaging of dynamic scenes. ACM Transactions on Graphics, 36(4):1–12, 2017. 2, 3, 5, 6, 7

[10] S. B. Kang, M. Uyttendaele, S. Winder, and R. Szeliski. High dynamic range video. ACM Transactions on Graphics, 22(3):319–325, 2003. 1, 2

[11] Chul Lee, Yuelong Li, and Vishal Monga. Ghost-free high dynamic range imaging via rank minimization. IEEE signal processing letters, 21(9):1045–1049, 2014. 1, 2

[12] Yoonho Lee and Seungjin Choi. Gradient-based metalearning with learned layerwise metric and subspace. In International Conference on Machine Learning (ICML), pages 2927–2936. PMLR, 2018. 3

[13] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 1833–1844, 2021. 4

[14] Zhen Liu, Lin Wenjie, Li Xinpeng, Rao Qing, Jiang Ting, Han Mingyan, Fan Haoqiang, Sun Jian, and Liu Shuaicheng. Adnet: Attention-guided deformable convolutional network

for high dynamic range imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 463–470, 2021. 2, 3, 6

[15] Kede Ma, Li Hui, Yong Hongwei, Wang Zhou, Meng Deyu, and Zhang. Lei. Robust multi-exposure image fusion: A structural patch decomposition approach. IEEE Transactions on Image Processing, 26(5):2519–2532, 2017. 1

[16] Rafat Mantiuk, Kil Joong Kim, Allan G. Rempel, and Wolfgang Heidrich. HDR-VDP-2:a calibrated visual metric for visibility and quality predictions in all luminance conditions. In ACM Siggraph, pages 1–14, 2011. 6

[17] Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston. Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126, 2016. 3

[18] Yuzhen Niu, Wu Jianbin, Liu Wenxi, Guo Wenzhong, and WH Lau. Rynson. Hdr-gan: Hdr image reconstruction from multi-exposed ldr images with large motions. IEEE Transactions on Image Processing, 30:3885–3896, 2021. 3

[19] Tae-Hyun Oh, Joon-Young Lee, Yu-Wing Tai, and In So Kweon. Robust high dynamic range imaging by rank minimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(6):1219–1232, 2015. 1, 2

[20] Fabrizio Pece and Jan Kautz. Bitmap movement detection: HDR for dynamic scenes. In Visual Media Production (CVMP), pages 1–8, 2010. 1

[21] K Ram Prabhakar, Rajat Arora, Adhitya Swaminathan, Kunal Pratap Singh, and R Venkatesh Babu. A fast, scalable, and reliable deghosting method for extreme exposure fusion. In 2019 IEEE International Conference on Computational Photography (ICCP), pages 1–8. IEEE, 2019. 6, 7

[22] K Ram Prabhakar, Gowtham Senthil, Susmit Agrawal, R Venkatesh Babu, and Rama Krishna Sai S Gorthi. Labeled from unlabeled: Exploiting unlabeled data for few-shot deep hdr deghosting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4875–4885, 2021. 2, 3, 6

[23] Hang Qi, Matthew Brown, and David G Lowe. Lowshot learning with imprinted weights. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5822–5830, 2018. 3

[24] Pradeep Sen, Khademi Kalantari Nima, Yaesoubi Maziar, Darabi Soheil, Dan B Goldman, and Eli Shechtman. Robust patch-based HDR reconstruction of dynamic scenes. ACM Transactions on Graphics, 31(6):1–11, 2012. 1, 3, 6

[25] Pranav Shyam, Shubham Gupta, and Ambedkar Dukkipati. Attentive recurrent comparators. In International conference on machine learning (ICML), pages 3173–3181. PMLR, 2017. 3

[26] Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. Endto-end memory networks. Advances in neural information processing systems (NIPS), 28, 2015. 3

[27] Anna Tomaszewska and Radoslaw Mantiuk. Image registration for multi-exposure high dynamic range image acquisition. In International Conference in Central Europe on Computer Graphics and Visualization (WSCG), 2007. 1, 2

[28] Okan Tarhan Tursun, Ahmet Oguz Aky˘ uz, Aykut Erdem, and¨ Erkut Erdem. An objective deghosting quality metric for HDR images. Comput. Graph. Forum, 35(2):139–152, 2016. 6

[29] Xintao Wang, Kelvin CK Chan, Ke Yu, Chao Dong, and Chen Change Loy. Edvr: Video restoration with enhanced deformable convolutional networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 0–0, 2019. 2

[30] Shangzhe Wu, Xu Jiarui, Tai Yu-Wing, and Tang. Chi-Keung. Deep high dynamic range imaging with large foreground motions. In Proceedings ofthe European Conference on Computer Vision (ECCV), pages 117–132, 2018. 3, 6

[31] Qingsen Yan, Gong Dong, Shi Qinfeng, van den Hengel Anton, Shen Chunhua, Reid Ian, and Zhang Yanning. Attentionguided network for ghost-free high dynamic range imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1751–1760, 2019. 2, 3, 6

[32] Qingsen Yan, Dong Gong, Javen Qinfeng Shi, Anton van den Hengel, Chunhua Shen, Ian Reid, and Yanning Zhang. Dual-attention-guided network for ghost-free high dynamic range imaging. International Journal of Computer Vision, 130(1):76–94, 2022. 2

[33] Qingsen Yan, Dong Gong, Javen Qinfeng Shi, Anton van den Hengel, Jinqiu Sun, Yu Zhu, and Yanning Zhang. High dynamic range imaging via gradient-aware context aggregation network. Pattern Recognition, 122:108342, 2022. 2

[34] Qingsen Yan, Zhang Lei, Liu Yu, Zhu Yu, Sun Jinqiu, Shi Qinfeng, and Zhang Yanning. Deep hdr imaging via a nonlocal network. IEEE Transactions on Image Processing, 29:4308–4322, 2020. 3

[35] Qingsen Yan, Jinqiu Sun, Haisen Li, Yu Zhu, and Yanning Zhang. High dynamic range imaging by sparse representation. Neurocomputing, 269:160–169, 2017. 1

[36] Qingsen Yan, Bo Wang, Peipei Li, Xianjun Li, Ao Zhang, Qinfeng Shi, Zheng You, Yu Zhu, Jinqiu Sun, and Yanning Zhang. Ghost removal via channel attention in exposure fusion. Computer Vision and Image Understanding, 201:103079, 2020. 2

[37] Qingsen Yan, Yu Zhu, and Yanning Zhang. Robust artifactfree high dynamic range imaging of dynamic scenes. Multimedia Tools and Applications, 78:11487–11505, 2019. 1

[38] Qian Ye, Jun Xiao, Kin-man Lam, and Takayuki Okatani. Progressive and selective fusion network for high dynamic range imaging. In Proceedings of the 29th ACM International Conference on Multimedia (ACM MM), pages 5290– 5297, 2021. 3

[39] Jaesik Yoon, Taesup Kim, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin Ahn. Bayesian model-agnostic meta-learning. Advances in neural information processing systems (NIPS), 31, 2018. 3

[40] Wei Zhang and Wai-Kuen Cham. Gradient-directed multiexposure composition. IEEE Transactions on Image Processing, 21(4):2318–2323, 2011. 1, 2