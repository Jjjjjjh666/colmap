// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"

namespace colmap {

struct SiftExtractionOptions {
  // Number of threads for feature extraction.
  int num_threads = -1;

  // Whether to use the GPU for feature extraction.
  bool use_gpu = true;

  // Index of the GPU used for feature extraction. For multi-GPU extraction,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum image size, otherwise image will be down-scaled.
  int max_image_size = 3200;

  // Maximum number of features to detect, keeping larger-scale features.
  int max_num_features = 8192;

  // First octave in the pyramid, i.e. -1 upsamples the image by one level.
  int first_octave = -1;

  // Number of octaves.
  int num_octaves = 4;

  // Number of levels per octave.
  int octave_resolution = 3;

  // Peak threshold for detection.
  double peak_threshold = 0.02 / octave_resolution;

  // Edge threshold for detection.
  double edge_threshold = 10.0;

  // Estimate affine shape of SIFT features in the form of oriented ellipses as
  // opposed to original SIFT which estimates oriented disks.
  bool estimate_affine_shape = false;

  // Maximum number of orientations per keypoint if not estimate_affine_shape.
  int max_num_orientations = 2;

  // Fix the orientation to 0 for upright features.
  bool upright = false;

  // Whether to adapt the feature detection depending on the image darkness.
  // Note that this feature is only available in the OpenGL SiftGPU version.
  bool darkness_adaptivity = false;

  // Domain-size pooling parameters. Domain-size pooling computes an average
  // SIFT descriptor across multiple scales around the detected scale. This was
  // proposed in "Domain-Size Pooling in Local Descriptors and Network
  // Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to
  // outperform other SIFT variants and learned descriptors in "Comparative
  // Evaluation of Hand-Crafted and Learned Local Features", Schönberger,
  // Hardmeier, Sattler, Pollefeys, CVPR 2016.
  bool domain_size_pooling = false;
  double dsp_min_scale = 1.0 / 6.0;
  double dsp_max_scale = 3.0;
  int dsp_num_scales = 10;

  // Whether to force usage of the covariant VLFeat implementation.
  // Otherwise, the covariant implementation is only used when
  // estimate_affine_shape or domain_size_pooling are enabled, since the normal
  // Sift implementation is faster.
  bool force_covariant_extractor = false;

  enum class Normalization {
    // L1-normalizes each descriptor followed by element-wise square rooting.
    // This normalization is usually better than standard L2-normalization.
    // See "Three things everyone should know to improve object retrieval",
    // Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
    L1_ROOT,
    // Each vector is L2-normalized.
    L2,
  };
  Normalization normalization = Normalization::L1_ROOT;

  bool Check() const;
};

// Create a Sift feature extractor based on the provided options. The same
// feature extractor instance can be used to extract features for multiple
// images in the same thread. Note that, for GPU based extraction, a OpenGL
// context must be made current in the thread of the caller. If the gpu_index is
// not -1, the CUDA version of SiftGPU is used, which produces slightly
// different results than the OpenGL implementation.
std::unique_ptr<FeatureExtractor> CreateSiftFeatureExtractor(
    const SiftExtractionOptions& options);

struct SiftMatchingOptions {
  // Number of threads for feature matching and geometric verification.
  int num_threads = -1;

  // Whether to use the GPU for feature matching.
  bool use_gpu = true;

  // Index of the GPU used for feature matching. For multi-GPU matching,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum distance ratio between first and second best match.
  double max_ratio = 0.8;

  // Maximum distance to best match.
  double max_distance = 0.7;

  // Whether to enable cross checking in matching.
  bool cross_check = true;

  // Maximum number of matches.
  int max_num_matches = 32768;

  // Whether to perform guided matching, if geometric verification succeeds.
  bool guided_matching = false;

  // Whether to use brute-force instead of FLANN based CPU matching.
  bool cpu_brute_force_matcher = false;

  // Cache for reusing descriptor index for feature matching.
  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>*
      cpu_descriptor_index_cache = nullptr;

  bool Check() const;
};

std::unique_ptr<FeatureMatcher> CreateSiftFeatureMatcher(
    const SiftMatchingOptions& options);

// Load keypoints and descriptors from text file in the following format:
//
//    LINE_0:            NUM_FEATURES DIM
//    LINE_1:            X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//    LINE_I:            ...
//    LINE_NUM_FEATURES: X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//
// where the first line specifies the number of features and the descriptor
// dimensionality followed by one line per feature: X, Y, SCALE, ORIENTATION are
// of type float and D_J represent the descriptor in the range [0, 255].
//
// For example:
//
//    2 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//    0.32 0.12 1.23 1.0 1 2 3 4
//
void LoadSiftFeaturesFromTextFile(const std::string& path,
                                  FeatureKeypoints* keypoints,
                                  FeatureDescriptors* descriptors);

}  // namespace colmap

/*

  sift算法：

  1. 构建尺度空间：
    1. 逐步增加高斯函数的标准差进行高斯卷积(高斯模糊)，形成不同尺度的图像。
    2. 下采样(减小分辨率)，使得新的图像的宽高小于原先的图像，形成不同的组(octave)。
    3. 尺度空间的坐标：x, y, scale

  2. 高斯差分金字塔：
    1. 理想情况下，计算尺度空间的LOG，其中的极值点就是可疑的特征点。
    2. 但是，实际使用DOG来近似LOG，简便计算。

  3. 通过泰勒展开拟合·，将离散的坐标优化为为连续的坐标。

  4. 删除边缘的点：
    1. 原因：容易受到噪音干扰。
    2. 边缘点的两个主曲率的比值一般较大，当找出某点的主曲率的比值大于阈值的时候，就舍去这些点。
    3. 对于曲面上的一个点，该点的黑森矩阵的两个特征值的大小正比于该点主曲率的大小。
    4. 问题就被转变为了：求两个特征值的比值的大小。
    5.    设特征值为a, b. 设t = a/ b.
          t + 1 / t = ((a + b)^2 - 2ab) / ab,
          a + b = tr(H),
          ab = det(H).

  5. 计算特征点参考方向：
    1. 原因：使得特征点具有旋转不变性。
    2. 计算特征点周围点的梯度大小，找到梯度最大值作为主方向，梯度值大于最大值的百分之八十作为辅方向。
    3. 如果特征点过于靠近边缘没有足够的相邻点也会被舍弃。

  6. 描述子的计算
    1. 将特征点周围一圈划分为4*4个子区域，每个子区域有8个方向。
    2. 重新定义坐标：以特征点的参考方向作为主方向。
    3. 对于每一个像素点计算其梯度，并将梯度分配到8个方向上去。
    4. 将每个子区域中像素点的8个方向上的梯度，按照距离特征点的距离的高斯函数加权平均。
    5. 得到了一个128个数字组成的的描述子


*/
