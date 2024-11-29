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

#include "colmap/feature/matcher.h"

#include <flann/flann.hpp>

namespace colmap {
namespace {

class FlannFeatureDescriptorIndex : public FeatureDescriptorIndex {
//定义了一个名为FlannFeatureDescriptorIndex的类，它继承自FeatureDescriptorIndex类。
 public:
  void Build(const FeatureDescriptors& index_descriptors) override {
    THROW_CHECK_EQ(index_descriptors.cols(), 128); //检查传入的特征描述符列数是否为128
    num_index_descriptors_ = index_descriptors.rows();  //获取传入的特征描述符的数量
    if (num_index_descriptors_ == 0) {
      // Flann is not happy when the input has no descriptors.
      index_ = nullptr;
      return;
    } //若特征描述符的数量为0，则将index_置为nullptr，表示没有有效的索引并返回
    const flann::Matrix<uint8_t> descriptors_matrix(
        const_cast<uint8_t*>(index_descriptors.data()),
        num_index_descriptors_,
        index_descriptors.cols()); 
   //创建一个flann::Matrix<uint8_t>类型的对象descriptors_matrix，用于存储特征描述符数据。
   //这个矩阵的构造函数接受特征描述符数据的指针、行数和列数作为参数。
    index_ = std::make_unique<FlannIndexType>(
        descriptors_matrix, flann::KDTreeIndexParams(kNumTreesInForest));
   //创建一个指向FlannIndexType类型的智能指针index_，
   //并使用特征描述符矩阵和指定的参数（这里使用了flann::KDTreeIndexParams，参数为kNumTreesInForest，一个常量表示在构建 KD 树时的树的数量）来初始化这个索引。
    index_->buildIndex(); //构建特征描述符索引
  }
//Build函数整体是对传入的特征描述符数据进行验证和预处理，再使用flann库来构建出对应的索引结构
//KD树（多维二叉树数据结构）详解：https://blog.csdn.net/Galaxy_yr/article/details/89285069?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_utm_term-1&spm=1001.2101.3001.4242


  void Search(int num_neighbors,
              const FeatureDescriptors& query_descriptors,
              Eigen::RowMajorMatrixXi& indices,
              Eigen::RowMajorMatrixXi& l2_dists) const override { 
   //函数用于对特征描述符的索引进行搜索
   //接收要搜索的近邻数量、查询特征描述符、存储搜索结果的索引矩阵和 L2 距离矩阵作为参数。
    THROW_CHECK_NOTNULL(index_);
    THROW_CHECK_EQ(query_descriptors.cols(), 128);
   //检查索引是否为空同时检查索引列数是否为128

    const int num_query_descriptors = query_descriptors.rows();
    if (num_query_descriptors == 0) {
      return;
    } //若特征描述符数量为0，则直接返回

    const int num_eff_neighbors =
        std::min(num_neighbors, num_index_descriptors_);
   //计算有效的近邻数量，取传入的近邻数量和索引中的特征描述符数量中的较小值

    indices.resize(num_query_descriptors, num_eff_neighbors);
    l2_dists.resize(num_query_descriptors, num_eff_neighbors);
   //调整索引矩阵和l2距离矩阵的大小
    const flann::Matrix<uint8_t> query_matrix(
        const_cast<uint8_t*>(query_descriptors.data()),
        num_query_descriptors,
        query_descriptors.cols());
   //创建对象用于存储待查询的特征描述符

    flann::Matrix<int> indices_matrix(
        indices.data(), num_query_descriptors, num_eff_neighbors);
    std::vector<float> l2_dist_vector(num_query_descriptors *
                                      num_eff_neighbors);
    flann::Matrix<float> l2_dist_matrix(
        l2_dist_vector.data(), num_query_descriptors, num_eff_neighbors);
   //创建用于存储搜索结果索引的flann::Matrix<int>类型的对象indices_matrix，
   //以及用于存储 L2 距离的std::vector<float>和flann::Matrix<float>类型的对象l2_dist_matrix。
    index_->knnSearch(query_matrix,
                      indices_matrix,
                      l2_dist_matrix,  //用于存储对应近邻的L2距离矩阵
                      num_eff_neighbors,
                      flann::SearchParams(kNumLeavesToVisit));
   //knn算法：当预测一个新的值x的时候，根据它距离最近的K个点是什么类别来判断x属于哪个类别。

    for (int query_idx = 0; query_idx < num_query_descriptors; ++query_idx) {
      for (int k = 0; k < num_eff_neighbors; ++k) {
        l2_dists(query_idx, k) = static_cast<int>(
            std::round(l2_dist_vector[query_idx * num_eff_neighbors + k]));
      }
    }
   // 遍历查询索引和近邻索引，将 L2 距离向量中的值转换为整数，并存储在 L2 距离矩阵中。
  }
//这个Search函数围绕已有的特征描述符索引，对传入的查询特征描述符进行全面的近邻搜索操作
//最终将搜索得到的近邻索引和对应的L2距离结果以合适的格式整理并存储在相应的矩阵中

 private:
  // Tuned to produce similar results to brute-force matching. If speed is
  // important, the parameters can be reduced. The biggest speed improvement can
  // be gained by reducing the number of leaves.
  constexpr static int kNumTreesInForest = 4;
  constexpr static int kNumLeavesToVisit = 128;

  using FlannIndexType = flann::Index<flann::L2<uint8_t>>; //表示flann索引的类型:用于存储和管理数据点的索引结构，可以快速找到某个点的近邻。
  std::unique_ptr<FlannIndexType> index_;  //定义了一个指向FlannIndexType类型的智能指针index_，用于存储特征描述符索引
  int num_index_descriptors_ = 0;  //存储索引中的特征描述符数量
};

}  // namespace

std::unique_ptr<FeatureDescriptorIndex> FeatureDescriptorIndex::Create() {
  return std::make_unique<FlannFeatureDescriptorIndex>();
}

}  // namespace colmap
//这段代码实现了一个基于 FLANN 算法的特征描述符索引的创建和搜索功能。
//FLANN是一个对大数据集和高维特征进行最近邻搜索的算法的集合，包括随机k-d树算法，优先搜索k-means树算法和层次聚类数算法

 
