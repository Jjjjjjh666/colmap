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

#pragma once  // 防止头文件被多次包含

// 引入依赖头文件
#include "colmap/estimators/two_view_geometry.h"  // 用于计算两视图几何关系
#include "colmap/feature/matcher.h"              // SIFT特征匹配相关工具
#include "colmap/feature/sift.h"                 // SIFT特征提取
#include "colmap/scene/database.h"               // 数据库操作工具
#include "colmap/util/opengl_utils.h"            // OpenGL上下文管理工具
#include "colmap/util/threading.h"               // 多线程支持工具

#include <array>
#include <memory>
#include <string>
#include <vector>  // 标准库头文件，用于容器和智能指针

namespace colmap {  // 定义在 colmap 命名空间下

// 数据结构：`FeatureMatcherData`
// 用于存储两个图像的匹配信息，包括图像ID、特征匹配和两视图几何关系
struct FeatureMatcherData {
  image_t image_id1 = kInvalidImageId;   // 第一个图像的ID，默认为无效值
  image_t image_id2 = kInvalidImageId;   // 第二个图像的ID，默认为无效值
  FeatureMatches matches;                // 两图像之间的特征匹配
  TwoViewGeometry two_view_geometry;     // 两视图几何信息
};

// 类：`FeatureMatcherWorker`
// 特征匹配的工作线程，用于处理一批图像对的特征匹配任务
class FeatureMatcherWorker : public Thread {
 public:
  typedef FeatureMatcherData Input;   // 输入数据类型
  typedef FeatureMatcherData Output;  // 输出数据类型

  // 构造函数
  // 参数：
  // - `matching_options`: SIFT匹配的参数选项
  // - `geometry_options`: 两视图几何验证的参数选项
  // - `cache`: 特征缓存，用于提升性能
  // - `input_queue`: 输入任务队列
  // - `output_queue`: 输出结果队列
  FeatureMatcherWorker(const SiftMatchingOptions& matching_options,
                       const TwoViewGeometryOptions& geometry_options,
                       const std::shared_ptr<FeatureMatcherCache>& cache,
                       JobQueue<Input>* input_queue,
                       JobQueue<Output>* output_queue);

  // 设置最大匹配点数
  void SetMaxNumMatches(int max_num_matches);

 private:
  void Run() override;  // 重写线程的主执行逻辑

  SiftMatchingOptions matching_options_;         // SIFT匹配参数
  TwoViewGeometryOptions geometry_options_;      // 两视图几何参数
  std::shared_ptr<FeatureMatcherCache> cache_;   // 特征缓存
  JobQueue<Input>* input_queue_;                 // 输入任务队列
  JobQueue<Output>* output_queue_;               // 输出结果队列
  std::unique_ptr<OpenGLContextManager> opengl_context_;  // OpenGL上下文，用于GPU加速
};

// 类：`FeatureMatcherController`
// 多线程、多GPU的SIFT特征匹配控制器，负责管理匹配任务并将结果写入数据库。
// 通过缓存和数据库事务提高性能。
class FeatureMatcherController {
 public:
  // 构造函数
  // 参数：
  // - `matching_options`: SIFT匹配参数
  // - `two_view_geometry_options`: 两视图几何参数
  // - `cache`: 特征缓存，用于快速读取图像特征
  FeatureMatcherController(
      const SiftMatchingOptions& matching_options,
      const TwoViewGeometryOptions& two_view_geometry_options,
      std::shared_ptr<FeatureMatcherCache> cache);

  // 析构函数
  ~FeatureMatcherController();

  // 设置匹配器并返回是否成功
  bool Setup();

  // 对一批图像对进行匹配
  void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs);

 private:
  SiftMatchingOptions matching_options_;        // SIFT匹配参数
  TwoViewGeometryOptions geometry_options_;     // 两视图几何参数
  std::shared_ptr<FeatureMatcherCache> cache_;  // 特征缓存

  bool is_setup_;  // 标记是否成功设置匹配器

  std::vector<std::unique_ptr<FeatureMatcherWorker>> matchers_;        // 普通匹配线程
  std::vector<std::unique_ptr<FeatureMatcherWorker>> guided_matchers_; // 引导匹配线程
  std::vector<std::unique_ptr<Thread>> verifiers_;                     // 几何验证线程
  std::unique_ptr<ThreadPool> thread_pool_;                            // 线程池管理

  JobQueue<FeatureMatcherData> matcher_queue_;        // 匹配任务队列
  JobQueue<FeatureMatcherData> verifier_queue_;       // 验证任务队列
  JobQueue<FeatureMatcherData> guided_matcher_queue_; // 引导匹配任务队列
  JobQueue<FeatureMatcherData> output_queue_;         // 输出任务队列
};

}  // namespace colmap
