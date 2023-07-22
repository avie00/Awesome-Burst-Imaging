# Burst-Imaging
A Comprehensive Collection of Papers and Repositories from CVPR, ICCV, and WACV (2016-2023)

Burst imaging has emerged as a powerful technique in computer vision and image processing, allowing us to capture and analyze temporal sequences of images to gain valuable insights. In recent years, top-tier conferences in computer vision, including CVPR (Conference on Computer Vision and Pattern Recognition), ICCV (International Conference on Computer Vision), and WACV (Winter Conference on Applications of Computer Vision) and journals have witnessed a surge in research focused on burst imaging.

Please note that this collection is an ongoing effort, and encourage the community to contribute by suggesting additional papers or repositories that may have been missed. Happy exploring!
![Burst Example](https://github.com/avie00/Burst-Imaging/blob/main/imgs/burst-example.png)

# Conferences
## CVPR 2023
### Computational Photography (Image Restoration and Enhancement)
#### Burstormer: Burst Image Restoration and Enhancement Transformer | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.pdf) | [project](https://github.com/akshaydudhane16/Burstormer)
- Proposes Burstormer, a transformer-based architecture for burst image restoration and enhancement, addressing the challenges of misalignment and degradation in burst frames.
Exploits multi-scale local and non-local features to achieve improved alignment and feature fusion, enabling inter-frame communication and burst-wide context modeling.
- Introduces an enhanced deformable alignment module that not only aligns burst features but also exchanges feature information and maintains focused communication with the reference frame through a reference-based feature enrichment mechanism.

#### Gated Multi-Resolution Transfer Network for Burst Restoration and Enhancement | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Mehta_Gated_Multi-Resolution_Transfer_Network_for_Burst_Restoration_and_Enhancement_CVPR_2023_paper.pdf) | [project](https://github.com/nanmehta/GMTNet) |
- Proposes a novel Gated Multi-Resolution Transfer Network (GMTNet) that addresses the challenges of burst image processing, including multiple degradations, misalignments, and limited utilization of mutual correlation and contextual information among burst frames by incorporating three optimized modules: Multi-scale Burst Feature Alignment (MBFA) for denoising and alignment, Transposed-Attention Feature Merging (TAFM) for multi-frame aggregation, and Resolution Transfer Feature Up-sampler (RTFU) for high-quality image reconstruction.
  
## WACV 2023
### Computational Photography (Reflection Removal, Super Resolution)
#### Burst Reflection Removal using Reflection Motion Aggregation Cues | [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Prasad_Burst_Reflection_Removal_Using_Reflection_Motion_Aggregation_Cues_WACV_2023_paper.pdf)
- Introduces a multi-stage deep learning approach for burst reflection removal, addressing the limitations of existing multi-image methods that require different view points and wide baselines.
- Leverages a burst of images captured in a short time duration, exploiting subtle handshakes to separate reflection and transmission layers, and presents a novel reflection motion aggregation (RMA) cue that emphasizes the transmission layer and aids in better layer separation.

#### Kernel-Aware Burst Blind Super-Resolution | [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Lian_Kernel-Aware_Burst_Blind_Super-Resolution_WACV_2023_paper.pdf) | [project](https://github.com/shermanlian/KBNet) |
- Introduces a kernel-guided strategy consisting of two steps: kernel estimation and HR image restoration to effectively handle complicated and unknown degradations in real-world low-resolution (LR) images, overcoming the limitations of existing non-blind designed networks.
- Proposes a pyramid kernel-aware deformable alignment module that aligns raw images considering blurry priors, enhancing the accuracy of the restoration process.

### Quanta Burst Imaging (Low-Light Enhancement)
#### Burst Vision Using Single-Photon Cameras | [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Ma_Burst_Vision_Using_Single-Photon_Cameras_WACV_2023_paper.pdf) | [project](https://wisionlab.com/project/burst-vision-single-photon/) |
- Proposes the development of quanta vision algorithms based on burst processing to extract scene information from single-photon avalanche diodes (SPAD) photon streams, enabling the use of high-resolution SPAD arrays as passive sensors for general computer vision tasks.
- Demonstrates the capabilities of SPAD sensors, combined with burst processing, in handling extremely challenging imaging conditions such as fast motion, low light, and high dynamic range, and showcases their effectiveness in various real-world computer vision tasks including object detection, pose estimation, SLAM, and text recognition.

## CVPR 2022

### Computational Photography (Image Restoration and Enhancement, Low Light Enhancement, Large Shift Alignment)
#### Burst Image Restoration and Enhancement | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dudhane_Burst_Image_Restoration_and_Enhancement_CVPR_2022_paper.pdf) | [project](https://github.com/akshaydudhane16/BIPNet) |
- Introduces the concept of pseudo-burst features, combining complimentary information from input burst frames to facilitate seamless information exchange and uses an edge-boosting burst alignment module to properly align individual burst frames, enabling successful creation of pseudo-burst features.
- Enriches the pseudo-burst features using multi-scale contextual information, and adaptively aggregates information to progressively increase resolution and merge the pseudo-burst features.

#### A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_A_Differentiable_Two-Stage_Alignment_Scheme_for_Burst_Image_Reconstruction_With_CVPR_2022_paper.pdf) |
- Introduces joint denoising and demosaicking (JDD) for burst images (JDD-B) as a crucial step in reconstructing high-quality full-color images from raw data captured by modern imaging devices.
- Identifies robust alignment of burst image frames as a key challenge in JDD-B due to large shifts caused by camera and object motion, and ddresses the alignment challenges by proposing a differentiable two-stage alignment scheme, utilizing patch-level and pixel-level alignment sequentially.

#### NAN: Noise-Aware NeRFs for Burst-Denoising | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Pearl_NAN_Noise-Aware_NeRFs_for_Burst-Denoising_CVPR_2022_paper.pdf) | [project](https://noise-aware-nerf.github.io/) |
- Identifies a major challenge in burst denoising related to coping with pixel misalignment, especially in the presence of large motion and high noise levels.
- Proposes a novel approach called NAN1, which utilizes Neural Radiance Fields (NeRFs) originally designed for physics-based novel-view rendering, as a powerful framework for burst denoising.
  
## CVPR 2021
### Computational Photography (Super-Resolution)
#### Deep Burst Super-Resolution | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf) | [project](https://github.com/goutamgmb/deep-burst-sr) |
- Addresses the limitations of single-image super-resolution (SISR) approaches, which focus on learning image priors for adding high-frequency details, by introducing the multi-frame super-resolution (MFSR) technique
- Presents a novel architecture for burst super-resolution, where multiple noisy RAW images are taken as input, and a denoised, super-resolved RGB image is generated as output. This incorporates explicit alignment of deep embeddings of input frames using pixel-wise optical flow and adaptive merging of information from all frames using an attention-based fusion module.
  
## CVPR 2020
### Computational Photography (Low-Light Enhancement)
Basis Prediction Networks for Effective Burst Denoising with Large Kernels | [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Basis_Prediction_Networks_for_Effective_Burst_Den) | [project](https://github.com/likesum/bpn)
- Explores the self-similarity present in bursts of images across time and space, leading to the representation of kernels as linear combinations of a small set of basis elements.
- Introduces a novel basis prediction network that predicts global basis kernels shared within the image and pixel-specific mixing coefficients for individual pixels in the input burst.

# Classical Papers
## CVPR 2019
### Computational Photography (Low-Light Enhancement)
#### Iterative Residual CNNs for Burst Photography Applications | [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kokkinos_Iterative_Residual_CNNs_for_Burst_Photography_Applications_CVPR_2019_paper.pdf) | [project](https://fkokkinos.github.io/deep_burst/)
- Utilizes a forward physics-based model to accurately describe each frame in the burst sequence, enabling the restoration of a single higher-quality image through an optimization problem.
- Proposes a convolutional iterative network with a transparent architecture, inspired by the proximal gradient descent method for handling non-smooth functions and modern deep learning techniques.

## CVPR 2018
### Computational Photography (Low-Light Enhancement)
#### Burst Denoising with Kernel Prediction Networks | [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mildenhall_Burst_Denoising_With_CVPR_2018_paper.pdf) | [project](https://github.com/google/burst-denoising)
- Introduces a novel technique for denoising bursts of images captured from a handheld camera using a convolutional neural network architecture.
- Predicts spatially varying kernels, enabling simultaneous alignment and denoising of frames, by utilizing a synthetic data generation approach based on a realistic noise formation model and an optimization to prevent undesirable local minima.

## CVPR 2015 
### Computational Photography (Deblurring)
#### Burst Deblurring: Removing Camera Shake Through Fourier Burst Accumulation | [paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Delbracio_Burst_Deblurring_Removing_2015_CVPR_paper.pdf) | [project](https://roboticimaging.org/Projects/BurstSfM/)
- Introduces a novel approach for removing image blur caused by camera shake, particularly when a burst of images is available as input.
- Performs a weighted average in the Fourier domain, leveraging the random nature of camera shake, where each image in the burst is typically blurred differently.

# Journals
# 2022
## RA-L
### Robotic Vision (Low-Light Enhancement)
#### Burst Imaging for Light-Constrained Structure-From-Motion | [paper](https://roboticimaging.org/Papers/ravendran2022burst.pdf/) | [project](https://roboticimaging.org/Projects/BurstSfM/)
- Establishes the viability of using burst imaging to improve robotic vision in low light, and provide a set of recommendations for adopting this approach in reconstruction tasks such as SfM.
- Enables the use of direct methods for image registration within bursts by exploiting the small camera motion between frames to yield a strong SNR advantage and applies feature-based methods to handle large camera motions between bursts for reconstruction.
  
## ACM ToG
### Computational Photography (Super Resolution)
#### High Dynamic Range and Super-Resolution from Raw Image Bursts | [paper](https://dl.acm.org/doi/pdf/10.1145/3528223.3530180/) |
- Reconstructs high-resolution, high-dynamic range color images from raw photographic bursts taken by handheld cameras with exposure bracketing.
- Leverages a physically-accurate model of image formation by combining an iterative optimization algorithm for solving the inverse problem, a learned image representation for alignment, and a learned natural image prior.

## 2020
## ACM ToG
### Quanta Burst Imaging (Low-Light Enhancement)
#### Quanta Burst Photography | [paper](https://dl.acm.org/doi/pdf/10.1145/3528223.3530180/](https://wisionlab.com/project/quanta-burst-photography/) |
- Introduces qa computational photography technique that utilizes single-photon cameras (SPCs) as passive imaging devices for challenging conditions like ultra low-light and fast motion.
- Aligns and merges binary sequences from SPCs, producing intensity images with minimal motion blur, artifacts, high signal-to-noise ratio (SNR), and a wide dynamic range demonstrate the generation of high-quality images using SPADs.
