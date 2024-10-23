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

#include "colmap/feature/utils.h"

#include "colmap/math/math.h"

namespace colmap {

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const FeatureKeypoints& keypoints) {    //将特征点转化为一组2维点的向量的表示
  std::vector<Eigen::Vector2d> points(keypoints.size());  
  for (size_t i = 0; i < keypoints.size(); ++i) {
    points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);  //遍历特征点集合，将坐标（x,y）转化为Eigen::Vector2d类型
  }
  return points;
}

void L2NormalizeFeatureDescriptors(FeatureDescriptorsFloat* descriptors) {
  descriptors->rowwise().normalize();
}  //将特征描述符进行L2归一化处理
//对特征描述符进行 L2 归一化操作就是将特征描述符的向量转换为在单位球面上的向量，
//使得不同的特征描述符更具有可比性。其具体步骤是将每个特征描述符向量除以其 L2 范数。
//L2 范数是向量中每个元素的平方和的平方根。

void L1RootNormalizeFeatureDescriptors(FeatureDescriptorsFloat* descriptors) {
  for (Eigen::MatrixXf::Index r = 0; r < descriptors->rows(); ++r) {
    descriptors->row(r) *= 1 / descriptors->row(r).lpNorm<1>();
    descriptors->row(r) = descriptors->row(r).array().sqrt();
     //首先计算当前行的 L1 范数（即所有元素绝对值之和），然后将当前行的每个元素乘以 1 除以该 L1 范数，实现 L1 归一化。
    //再对L1归一化的每个元素取平方根
  }
}  //L1归一化处理
//L1 根归一化操作是一种数据预处理技术，主要用于将数据特征转换到特定的尺度范围
//主要操作：计算L1范数，将向量中所有元素的绝对值进行求和，再将向量中的各个元素除以向量的L1范数再对结果取平方根

FeatureDescriptors FeatureDescriptorsToUnsignedByte(
    const Eigen::Ref<const FeatureDescriptorsFloat>& descriptors) {
    //将特征的浮点描述转化为无符号的字节类型
  FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
                                               descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
          TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

void ExtractTopScaleFeatures(FeatureKeypoints* keypoints,
                             FeatureDescriptors* descriptors,
                             const size_t num_features) {
    //从给定的特征点和特征描述符集合中提取具有最大尺度的前num_features个特征
  THROW_CHECK_EQ(keypoints->size(), descriptors->rows());
  THROW_CHECK_GT(num_features, 0); 
//使用断言检查输入的特征点数量和特征描述符行数是否相等，并且检查要提取的特征数量是否大于 0。
  if (static_cast<size_t>(descriptors->rows()) <= num_features) {
    return;
  }
//如果输入的特征描述符行数小于或等于要提取的特征数量，则直接返回，不进行任何操作。
  std::vector<std::pair<size_t, float>> scales;  //创建容器用于存储特征点的索引和尺度值的对
  scales.reserve(keypoints->size());
  for (size_t i = 0; i < keypoints->size(); ++i) {
    scales.emplace_back(i, (*keypoints)[i].ComputeScale());
  }
    //遍历特征点集合将每个特征点的索引和尺度值存储到向量中

  std::partial_sort(scales.begin(),
                    scales.begin() + num_features,
                    scales.end(),
                    [](const std::pair<size_t, float>& scale1,
                       const std::pair<size_t, float>& scale2) {
                      return scale1.second > scale2.second;
                    });

  FeatureKeypoints top_scale_keypoints(num_features);
  FeatureDescriptors top_scale_descriptors(num_features, descriptors->cols());
  for (size_t i = 0; i < num_features; ++i) {
    top_scale_keypoints[i] = (*keypoints)[scales[i].first];
    top_scale_descriptors.row(i) = descriptors->row(scales[i].first);
  }
    //创建两个新容器分别遍历存储前num—features个排序后的特征点和特征点描述符

  *keypoints = std::move(top_scale_keypoints);
  *descriptors = std::move(top_scale_descriptors);
    //将新创建的容器移动赋值给输入的指针所指向的容器，完成对输入容器的更新，使其只包含具有最大尺度的特征点和特征描述符。
}

}  // namespace colmap
//这段函数整体上对特征点数据进行了一定处理，起到了特征工程的作用
