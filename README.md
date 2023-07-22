# Burst-Imaging
A Comprehensive Collection of Papers and Repositories from CVPR, ICCV, and WACV (2016-2023) and Top-Tier Journals

Burst imaging has emerged as a powerful technique in computer vision and image processing, allowing us to capture and analyze temporal sequences of images to gain valuable insights. In recent years, top-tier conferences in computer vision, including CVPR (Conference on Computer Vision and Pattern Recognition), ICCV (International Conference on Computer Vision), and WACV (Winter Conference on Applications of Computer Vision) and journals have witnessed a surge in research focused on burst imaging.

Please note that this collection is an ongoing effort, and encourage the community to contribute by suggesting additional papers or repositories that may have been missed. Happy exploring!
![Burst Example](https://github.com/avie00/Burst-Imaging/blob/main/imgs/burst-example.png)

# Conferences
## CVPR 2023
### Computational Photography (Image Enhancement)
#### Burstormer: Burst Image Restoration and Enhancement Transformer | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.pdf) | [project](https://github.com/akshaydudhane16/Burstormer) |
- Proposes Burstormer, a transformer-based architecture for burst image restoration and enhancement, addressing the challenges of misalignment and degradation in burst frames.
Exploits multi-scale local and non-local features to achieve improved alignment and feature fusion, enabling inter-frame communication and burst-wide context modeling.
- Introduces an enhanced deformable alignment module that not only aligns burst features but also exchanges feature information and maintains focused communication with the reference frame through a reference-based feature enrichment mechanism.

#### Gated Multi-Resolution Transfer Network for Burst Restoration and Enhancement | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Mehta_Gated_Multi-Resolution_Transfer_Network_for_Burst_Restoration_and_Enhancement_CVPR_2023_paper.pdf) | [project](https://github.com/nanmehta/GMTNet) |
- Proposes a novel Gated Multi-Resolution Transfer Network (GMTNet) that addresses the challenges of burst image processing, including multiple degradations, misalignments, and limited utilization of mutual correlation and contextual information among burst frames by incorporating three optimized modules: Multi-scale Burst Feature Alignment (MBFA) for denoising and alignment, Transposed-Attention Feature Merging (TAFM) for multi-frame aggregation, and Resolution Transfer Feature Up-sampler (RTFU) for high-quality image reconstruction.
  
## WACV 2023
### Computational Photography (Reflection Removal, Super Resolution)
#### Burst Reflection Removal using Reflection Motion Aggregation Cues | [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Prasad_Burst_Reflection_Removal_Using_Reflection_Motion_Aggregation_Cues_WACV_2023_paper.pdf) |
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
### Computational Photography (Image Enhancement, Low Light Enhancement, Large Shift Alignment)
#### Burst Image Restoration and Enhancement | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dudhane_Burst_Image_Restoration_and_Enhancement_CVPR_2022_paper.pdf) | [project](https://github.com/akshaydudhane16/BIPNet) |
- Introduces the concept of pseudo-burst features, combining complimentary information from input burst frames to facilitate seamless information exchange and uses an edge-boosting burst alignment module to properly align individual burst frames, enabling successful creation of pseudo-burst features.
- Enriches the pseudo-burst features using multi-scale contextual information, and adaptively aggregates information to progressively increase resolution and merge the pseudo-burst features.

#### Dancing Under the Stars: Video Denoising in Starlight | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Monakhova_Dancing_Under_the_Stars_Video_Denoising_in_Starlight_CVPR_2022_paper.pdf) | [project](https://kristinamonakhova.com/starlight_denoising/) |
- Demonstrates photorealistic video denoising in extremely low light conditions, specifically under starlight (<0.001 lux), a challenging scenario with very low photon counts.
- Develops a GAN-tuned physics-based noise model to accurately represent camera noise at the lowest light levels, enabling more realistic noise modeling for the denoising process, and trains a video denoiser using a combination of simulated noisy video clips and real noisy still burst of images, effectively learning to remove noise and improve video quality in low light conditions.
  
#### A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_A_Differentiable_Two-Stage_Alignment_Scheme_for_Burst_Image_Reconstruction_With_CVPR_2022_paper.pdf) |
- Introduces joint denoising and demosaicking (JDD) for burst images (JDD-B) as a crucial step in reconstructing high-quality full-color images from raw data captured by modern imaging devices.
- Identifies robust alignment of burst image frames as a key challenge in JDD-B due to large shifts caused by camera and object motion, and ddresses the alignment challenges by proposing a differentiable two-stage alignment scheme, utilizing patch-level and pixel-level alignment sequentially.

### Neural Radiance Field (Low-Light Enhancement)
#### NAN: Noise-Aware NeRFs for Burst-Denoising | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Pearl_NAN_Noise-Aware_NeRFs_for_Burst-Denoising_CVPR_2022_paper.pdf) | [project](https://noise-aware-nerf.github.io/) |
- Identifies a major challenge in burst denoising related to coping with pixel misalignment, especially in the presence of large motion and high noise levels.
- Proposes a novel approach called NAN1, which utilizes Neural Radiance Fields (NeRFs) originally designed for physics-based novel-view rendering, as a powerful framework for burst denoising.

#### NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Mildenhall_NeRF_in_the_Dark_High_Dynamic_Range_View_Synthesis_From_CVPR_2022_paper.pdf) | [project](https://bmild.github.io/rawnerf/) |
- Presents RawNeRF, a modification of Neural Radiance Fields (NeRF) for high-quality novel view synthesis from linear raw images, preserving the scene's full dynamic range.
- Enables novel high dynamic range (HDR) view synthesis tasks, allowing manipulation of focus, exposure, and tonemapping.
  
## CVPR 2021
### Computational Photography (Super-Resolution)
#### Deep Burst Super-Resolution | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf) | [project](https://github.com/goutamgmb/deep-burst-sr) |
- Addresses the limitations of single-image super-resolution (SISR) approaches, which focus on learning image priors for adding high-frequency details, by introducing the multi-frame super-resolution (MFSR) technique
- Presents a novel architecture for burst super-resolution, where multiple noisy RAW images are taken as input, and a denoised, super-resolved RGB image is generated as output. This incorporates explicit alignment of deep embeddings of input frames using pixel-wise optical flow and adaptive merging of information from all frames using an attention-based fusion module.
  
## CVPR 2020
### Computational Photography (Low-Light Enhancement)
#### Basis Prediction Networks for Effective Burst Denoising with Large Kernels | [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Basis_Prediction_Networks_for_Effective_Burst_Den) | [project](https://github.com/likesum/bpn) |
- Explores the self-similarity present in bursts of images across time and space, leading to the representation of kernels as linear combinations of a small set of basis elements.
- Introduces a novel basis prediction network that predicts global basis kernels shared within the image and pixel-specific mixing coefficients for individual pixels in the input burst.

## ECCV 2020
### Computational Photography (Low-Light Enhancement)
#### Burst Denoising via Temporally Shifted Wavelet Transforms | [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580239.pdf) |
- Proposes an end-to-end trainable burst denoising pipeline to address the challenges of low light imaging in mobile photography, where long exposures can improve signal-to-noise ratio (SNR) but may introduce undesirable motion blur in dynamic scenes.
- Highlights the use of computational photography techniques, such as fusing multiple short exposures, to enhance SNR and generate visually pleasing results with deep network-based methods, nd introduces a novel model that jointly captures high-resolution and high-frequency deep features derived from wavelet transforms. The model preserves precious local details in high-frequency sub-band features to enhance final perceptual quality and uses low-frequency sub-band features for faithful reconstruction and final objective quality.

#### A Decoupled Learning Scheme for Real-World Burst Denoising from Raw Images | [paper](https://www4.comp.polyu.edu.hk/~cslzhang/paper/conf/ECCV20/BDNet.pdf) | [project](https://github.com/zhetongliang/BDNet) |
- Addresses limitations of existing learning-based burst denoising methods, which are often trained on video sequences with synthetic noise, leading to visual artifacts when applied to real-world raw image sequences with different noise statistics.
- Introduces a carefully designed multi-frame CNN model that decouples the learning of motion from the learning of noise statistics, enabling the model to adapt to real-world noisy datasets of static scenes and perform effective burst denoising on dynamic sequences.

# Classical Conference Papers
## CVPR 2019
### Computational Photography (Low-Light Enhancement)
#### Iterative Residual CNNs for Burst Photography Applications | [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kokkinos_Iterative_Residual_CNNs_for_Burst_Photography_Applications_CVPR_2019_paper.pdf) | [project](https://fkokkinos.github.io/deep_burst/) |
- Utilizes a forward physics-based model to accurately describe each frame in the burst sequence, enabling the restoration of a single higher-quality image through an optimization problem.
- Proposes a convolutional iterative network with a transparent architecture, inspired by the proximal gradient descent method for handling non-smooth functions and modern deep learning techniques.

## ICIP 2019
### Computational Photography (Low-Light Enhancement)
#### Multi-Kernel Prediction Networks for Denoising of Burst Images | [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kokkinos_Iterative_Residual_CNNs_for_Burst_Photography_Applications_CVPR_2019_paper.pdf) | [project](https://fkokkinos.github.io/deep_burst/) |
- Utilizes a forward physics-based model to accurately describe each frame in the burst sequence, enabling the restoration of a single higher-quality image through an optimization problem.
- Proposes a convolutional iterative network with a transparent architecture, inspired by the proximal gradient descent method for handling non-smooth functions and modern deep learning techniques.

## ECCV 2018
### Computational Photography (Low-Light Enhancement, Deblurring)
#### Deep Burst Denoising | [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Clement_Godard_Deep_Burst_Denoising_ECCV_2018_paper.pdf) |
- Proposes a strategy for mitigating noise in low-light situations by capturing multiple short frames in a burst and intelligently integrating the content, avoiding issues associated with long exposures.
- Implements integration using a recurrent fully convolutional deep neural net (CNN), creating a novel multiframe architecture that can be added to any single frame denoising model.
  
## ECCV 2018
#### Burst Image Deblurring Using Permutation Invariant Convolutional Neural Networks | [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Miika_Aittala_Burst_Image_Deblurring_ECCV_2018_paper.pdf) | [project](https://github.com/FrederikWarburg/Burst-Image-Deblurring) |
- Presents a neural approach for fusing a burst of photographs with severe camera shake and noise into a sharp and noise-free image.
- Introduces a novel convolutional architecture that simultaneously views all frames in the burst in an order-independent manner, effectively detecting and leveraging subtle cues scattered across different frames.
  
## CVPR 2018
### Computational Photography (Low-Light Enhancement)
#### Burst Denoising with Kernel Prediction Networks | [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mildenhall_Burst_Denoising_With_CVPR_2018_paper.pdf) | [project](https://github.com/google/burst-denoising) |
- Introduces a novel technique for denoising bursts of images captured from a handheld camera using a convolutional neural network architecture.
- Predicts spatially varying kernels, enabling simultaneous alignment and denoising of frames, by utilizing a synthetic data generation approach based on a realistic noise formation model and an optimization to prevent undesirable local minima.
  
#### Learning to See in the Dark | [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf) | [project](https://github.com/cchen156/Learning-to-See-in-the-Dark) |
- Introduces a challenging problem of imaging in low light conditions with issues like low photon count and low SNR, where short-exposure images suffer from noise, and long exposure can induce blur, and highlights the limitations of existing denoising, deblurring, and enhancement techniques in extreme conditions, such as video-rate imaging at night.
- Processes low-light images using end-to-end training of a fully-convolutional network that operates directly on raw sensor data, replacing much of the traditional image processing pipeline that performs poorly on such data.
  
## CVPR 2015 
### Computational Photography (Deblurring)
#### Burst Deblurring: Removing Camera Shake Through Fourier Burst Accumulation | [paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Delbracio_Burst_Deblurring_Removing_2015_CVPR_paper.pdf) | [project](https://roboticimaging.org/Projects/BurstSfM/) |
- Introduces a novel approach for removing image blur caused by camera shake, particularly when a burst of images is available as input.
- Performs a weighted average in the Fourier domain, leveraging the random nature of camera shake, where each image in the burst is typically blurred differently.

# Journals
## RA-L 2022
### Robotic Vision (Low-Light Enhancement)
#### Burst Imaging for Light-Constrained Structure-From-Motion | [paper](https://roboticimaging.org/Papers/ravendran2022burst.pdf/) | [project](https://roboticimaging.org/Projects/BurstSfM/) |
- Establishes the viability of using burst imaging to improve robotic vision in low light, and provide a set of recommendations for adopting this approach in reconstruction tasks such as SfM.
- Enables the use of direct methods for image registration within bursts by exploiting the small camera motion between frames to yield a strong SNR advantage and applies feature-based methods to handle large camera motions between bursts for reconstruction.
  
## ACM ToG 2022
### Computational Photography (Super Resolution)
#### High Dynamic Range and Super-Resolution from Raw Image Bursts | [paper](https://dl.acm.org/doi/pdf/10.1145/3528223.3530180/) |
- Reconstructs high-resolution, high-dynamic range color images from raw photographic bursts taken by handheld cameras with exposure bracketing.
- Leverages a physically-accurate model of image formation by combining an iterative optimization algorithm for solving the inverse problem, a learned image representation for alignment, and a learned natural image prior.

## IJCV 2022
### Computational Phtography (Low-Light Enhancement)
#### Efficient Burst Raw Denoising with Variance Stabilization and Multi-frequency Denoising Network | [paper](https://dl.acm.org/doi/abs/10.1007/s11263-022-01627-3) |
- Proposes a three-stage design for the burst denoising process including noise prior integration, multi-frame alignment and multi-frame denoising
- Demonstrates the efficiency and strong performance of the proposed three-stage design on burst denoising through experiments on synthetic and real raw datasets.

## ACM ToG 2020
### Quanta Burst Imaging (Low-Light Enhancement)
#### Quanta Burst Photography | [paper](https://dl.acm.org/doi/pdf/10.1145/3528223.3530180/) | [project](https://wisionlab.com/project/quanta-burst-photography/) |
- Introduces qa computational photography technique that utilizes single-photon cameras (SPCs) as passive imaging devices for challenging conditions like ultra low-light and fast motion.
- Aligns and merges binary sequences from SPCs, producing intensity images with minimal motion blur, artifacts, high signal-to-noise ratio (SNR), and a wide dynamic range demonstrate the generation of high-quality images using SPADs.

# Classical Journal Papers
## IEEE Transactions on Image Processing 2021
### Computational Phtography (Low-Light Enhancement)
#### Burst Photography for Learning to Enhance Extremely Dark Images | [paper](https://web.cs.hacettepe.edu.tr/~erkut/publications/dark-burst-photography-lowres.pdf) |
- Proposes a learning-based approach using burst photography to significantly improve the performance and obtain sharper and more accurate RGB images from extremely dark raw images.
- Introduces a novel coarse-to-fine network architecture as the backbone of the proposed framework. The coarse network predicts a low-resolution, denoised raw image, which is further refined by the fine network to recover fine-scale details and realistic textures, and extends the network to a permutation invariant structure, enabling it to take a burst of low-light images as input and merge information from multiple images at the feature-level, reducing noise and improving color accuracy.

## ACM ToG 2019
### Computational Photography (Low-Light Enhancement, Super-Resolution)
#### Handheld multi-frame super-resolution | [paper](https://dl.acm.org/doi/abs/10.1145/3306346.3323024) |
- Proposes a novel approach to replace traditional demosaicing in single-frame and burst photography pipelines with a multiframe super-resolution algorithm, directly creating a complete RGB image from a burst of CFA raw images.
- Utilizes natural hand tremor, typical in handheld photography, to capture a burst of raw frames with small offsets, which are then aligned and merged to form a single image with RGB values at every pixel site.
  
#### Handheld Mobile Photography in Very Low Light | [paper](https://dl.acm.org/doi/10.1145/3355089.3356508) |
- Describes a system for capturing clean, sharp, and colorful photographs in extremely low light conditions (as low as 0.3 lux) using mobile phones, where human vision becomes monochromatic and indistinct, and addresses the challenges of low-light photography with mobile phones, considering factors such as read noise, photon shot noise, small apertures, and handheld usage with moving subjects.
- Employs a multi-frame technique using motion metering to estimate motion magnitudes, enabling the capture of handheld photographs without flash illumination. The system optimizes the number of frames and per-frame exposure times to minimize both noise and motion blur in the captured burst.

## ACM ToG 2016
### Computational Photography (Low-Light Enhancement)
#### Burst photography for high dynamic range and low-light imaging on mobile cameras | [paper](https://dl.acm.org/doi/abs/10.1145/2980179.2980254) | [project](https://fkokkinos.github.io/deep_burst/) |
- Captures, aligns, and merges a burst of frames, avoiding bracketed exposures. This approach ensures more robust alignment, while setting the exposure low enough to prevent blown-out highlights, resulting in a merged image exhibiting clean shadows and high bit depth, enabling standard HDR tone mapping methods.
- Adopts a novel FFT-based alignment algorithm and a hybrid 2D/3D Wiener filter for denoising and merging frames within a burst.

## ACM ToG 2014
### Computational Photography (Low-Light Enhancement)
#### Fast Burst Images Denoising (Low-Light Enhancement) | [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/FastBurstDenoising_SIGGRAPHASIA14.pdf) |
- Introduces a fast denoising method capable of generating a clean image from a burst of noisy images, and accelerates image alignment using a lightweight camera motion representation known as homography flow, reducing computational overhead.
  Implements efficient temporal and spatial fusion to denoise the aligned images, ensuring rapid per-pixel operations, with a mechanism for selecting consistent pixels during temporal fusion to synthesize a clean, ghost-free image, reducing the need for extensive motion tracking between frames.
