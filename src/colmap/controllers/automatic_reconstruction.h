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

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/util/threading.h"

#include <memory>
#include <string>

namespace colmap {

class AutomaticReconstructionController : public Thread {
 public:
// 定义枚举类型DataType，表示输入数据的类型
  enum class DataType { INDIVIDUAL, VIDEO, INTERNET };
// 定义枚举类型Quality，表示重建质量的级别
  enum class Quality { LOW, MEDIUM, HIGH, EXTREME };
// 定义枚举类型Mesher，表示用于生成网格的算法类型
  enum class Mesher { POISSON, DELAUNAY };

  struct Options {
    // The path to the workspace folder in which all results are stored.
    std::string workspace_path;

    // The path to the image folder which are used as input.
    std::string image_path;

    // The path to the mask folder which are used as input.
    std::string mask_path;

    // The path to the vocabulary tree for feature matching.
    std::string vocab_tree_path;

    // The type of input data used to choose optimal mapper settings.
    DataType data_type = DataType::INDIVIDUAL;

    // Whether to perform low- or high-quality reconstruction.
    Quality quality = Quality::HIGH;

    // Whether to use shared intrinsics or not.
    bool single_camera = false;

    // Whether to use shared intrinsics or not for all images in the same
    // sub-folder.
    bool single_camera_per_folder = false;

    // Which camera model to use for images.
    std::string camera_model = "SIMPLE_RADIAL";

    // Initial camera params for all images.
    std::string camera_params;

    // Whether to perform sparse mapping.
    bool sparse = true;

// Whether to perform dense mapping.
#if defined(COLMAP_CUDA_ENABLED)
    bool dense = true;
#else
    bool dense = false;
#endif

    // The meshing algorithm to be used.
    Mesher mesher = Mesher::POISSON;

    // 在所有阶段要使用的线程数量
    int num_threads = -1;

  // 是否在特征提取、特征匹配和束调整阶段使用GPU
    bool use_gpu = true;

    // 用于GPU阶段的GPU索引。对于多GPU计算（在特征提取/匹配中），
        // 可以用逗号分隔多个GPU索引，例如 "0,1,2,3"。对于单GPU阶段，仅使用第一个GPU。
        // 默认情况下，所有可用的GPU将在所有阶段中使用
    std::string gpu_index = "-1";
  };

// 构造函数
    // 参数：
    // - options：包含自动重建的各种配置选项的结构体
    // - reconstruction_manager：指向重建管理器的共享智能指针
  AutomaticReconstructionController(
      const Options& options,
      std::shared_ptr<ReconstructionManager> reconstruction_manager);

  void Stop() override;

 private:
  void Run() override;
// 执行特征提取的函数
  void RunFeatureExtraction();
 // 执行特征匹配的函数
  void RunFeatureMatching();
// 执行稀疏映射的函数
  void RunSparseMapper();
// 执行密集映射的函数
  void RunDenseMapper();

// 存储自动重建的配置选项，不可修改（通过const修饰）
  const Options options_;
  OptionManager option_manager_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
  Thread* active_thread_;
  std::unique_ptr<Thread> feature_extractor_;
  std::unique_ptr<Thread> exhaustive_matcher_;
  std::unique_ptr<Thread> sequential_matcher_;
  std::unique_ptr<Thread> vocab_tree_matcher_;
};

}  // namespace colmap
