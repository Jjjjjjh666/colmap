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

#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/incremental_mapper.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/image/undistortion.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

namespace colmap {

// AutomaticReconstructionController类的构造函数实现
// 用于初始化自动重建控制器对象的各种成员变量和配置选项
AutomaticReconstructionController::AutomaticReconstructionController(
    const Options& options,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(options),
      reconstruction_manager_(std::move(reconstruction_manager)),
      active_thread_(nullptr) {
// 检查工作区路径对应的目录是否存在，如果不存在则抛出异常
  THROW_CHECK_DIR_EXISTS(options_.workspace_path);
// 检查图像路径对应的目录是否存在，如果不存在则抛出异常
  THROW_CHECK_DIR_EXISTS(options_.image_path);
 // 确保传入的重建管理器指针不为空，否则抛出异常
  THROW_CHECK_NOTNULL(reconstruction_manager_);

  option_manager_.AddAllOptions();

  *option_manager_.image_path = options_.image_path;
  *option_manager_.database_path =
      JoinPaths(options_.workspace_path, "database.db");

// 根据重建质量级别对选项管理器进行相应的修改
  if (options_.data_type == DataType::VIDEO) {
    option_manager_.ModifyForVideoData();
  } else if (options_.data_type == DataType::INDIVIDUAL) {
    option_manager_.ModifyForIndividualData();
  } else if (options_.data_type == DataType::INTERNET) {
    option_manager_.ModifyForInternetData();
  } else {
    LOG(FATAL_THROW) << "Data type not supported";
  }

  THROW_CHECK(ExistsCameraModelWithName(options_.camera_model));

  if (options_.quality == Quality::LOW) {
    option_manager_.ModifyForLowQuality();
  } else if (options_.quality == Quality::MEDIUM) {
    option_manager_.ModifyForMediumQuality();
  } else if (options_.quality == Quality::HIGH) {
    option_manager_.ModifyForHighQuality();
  } else if (options_.quality == Quality::EXTREME) {
    option_manager_.ModifyForExtremeQuality();
  }

  option_manager_.sift_extraction->num_threads = options_.num_threads;
  option_manager_.sift_matching->num_threads = options_.num_threads;
  option_manager_.mapper->num_threads = options_.num_threads;
  option_manager_.poisson_meshing->num_threads = options_.num_threads;

  ImageReaderOptions& reader_options = *option_manager_.image_reader;
  reader_options.database_path = *option_manager_.database_path;
  reader_options.image_path = *option_manager_.image_path;
  if (!options_.mask_path.empty()) {
    reader_options.mask_path = options_.mask_path;
    option_manager_.image_reader->mask_path = options_.mask_path;
    option_manager_.stereo_fusion->mask_path = options_.mask_path;
  }
  reader_options.single_camera = options_.single_camera;
  reader_options.single_camera_per_folder = options_.single_camera_per_folder;
  reader_options.camera_model = options_.camera_model;
  reader_options.camera_params = options_.camera_params;
        
// 设置各个阶段（如SIFT提取、匹配、束调整等）是否使用GPU
  option_manager_.sift_extraction->use_gpu = options_.use_gpu;
  option_manager_.sift_matching->use_gpu = options_.use_gpu;
  option_manager_.mapper->ba_use_gpu = options_.use_gpu;
  option_manager_.bundle_adjustment->use_gpu = options_.use_gpu;

  option_manager_.sift_extraction->gpu_index = options_.gpu_index;
  option_manager_.sift_matching->gpu_index = options_.gpu_index;
  option_manager_.patch_match_stereo->gpu_index = options_.gpu_index;
  option_manager_.mapper->ba_gpu_index = options_.gpu_index;
  option_manager_.bundle_adjustment->gpu_index = options_.gpu_index;

// 创建特征提取控制器对象，并传入相关参数
  feature_extractor_ = CreateFeatureExtractorController(
      reader_options, *option_manager_.sift_extraction);
// 创建穷举特征匹配器对象，并传入相关参数
  exhaustive_matcher_ =
      CreateExhaustiveFeatureMatcher(*option_manager_.exhaustive_matching,
                                     *option_manager_.sift_matching,
                                     *option_manager_.two_view_geometry,
                                     *option_manager_.database_path);

  if (!options_.vocab_tree_path.empty()) {
    option_manager_.sequential_matching->loop_detection = true;
    option_manager_.sequential_matching->vocab_tree_path =
        options_.vocab_tree_path;
  }

  sequential_matcher_ =
      CreateSequentialFeatureMatcher(*option_manager_.sequential_matching,
                                     *option_manager_.sift_matching,
                                     *option_manager_.two_view_geometry,
                                     *option_manager_.database_path);

  if (!options_.vocab_tree_path.empty()) {
    option_manager_.vocab_tree_matching->vocab_tree_path =
        options_.vocab_tree_path;
    vocab_tree_matcher_ =
        CreateVocabTreeFeatureMatcher(*option_manager_.vocab_tree_matching,
                                      *option_manager_.sift_matching,
                                      *option_manager_.two_view_geometry,
                                      *option_manager_.database_path);
  }
}

void AutomaticReconstructionController::Stop() {
  if (active_thread_ != nullptr) {
    active_thread_->Stop();
  }
  Thread::Stop();
}

void AutomaticReconstructionController::Run() {
  if (IsStopped()) {
    return;
  }

  RunFeatureExtraction();

  if (IsStopped()) {
    return;
  }

  RunFeatureMatching();

  if (IsStopped()) {
    return;
  }

  if (options_.sparse) {
    RunSparseMapper();
  }

  if (IsStopped()) {
    return;
  }

  if (options_.dense) {
    RunDenseMapper();
  }
}

void AutomaticReconstructionController::RunFeatureExtraction() {
  THROW_CHECK_NOTNULL(feature_extractor_);
  active_thread_ = feature_extractor_.get();
  feature_extractor_->Start();
  feature_extractor_->Wait();
  feature_extractor_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunFeatureMatching() {
  Thread* matcher = nullptr;
  if (options_.data_type == DataType::VIDEO) {
    matcher = sequential_matcher_.get();
  } else if (options_.data_type == DataType::INDIVIDUAL ||
             options_.data_type == DataType::INTERNET) {
    Database database(*option_manager_.database_path);
    const size_t num_images = database.NumImages();
    if (options_.vocab_tree_path.empty() || num_images < 200) {
      matcher = exhaustive_matcher_.get();
    } else {
      matcher = vocab_tree_matcher_.get();
    }
  }

  THROW_CHECK_NOTNULL(matcher);
  active_thread_ = matcher;
  matcher->Start();
  matcher->Wait();
  exhaustive_matcher_.reset();
  sequential_matcher_.reset();
  vocab_tree_matcher_.reset();
  active_thread_ = nullptr;
}

// 执行稀疏映射的具体函数实现
void AutomaticReconstructionController::RunSparseMapper() {
  const auto sparse_path = JoinPaths(options_.workspace_path, "sparse");
  if (ExistsDir(sparse_path)) {
    auto dir_list = GetDirList(sparse_path);
    std::sort(dir_list.begin(), dir_list.end());
    if (dir_list.size() > 0) {
      LOG(WARNING)
          << "Skipping sparse reconstruction because it is already computed";
      for (const auto& dir : dir_list) {
        reconstruction_manager_->Read(dir);
      }
      return;
    }
  }
// 创建增量映射器对象，并传入相关参数
  IncrementalPipeline mapper(option_manager_.mapper,
                             *option_manager_.image_path,
                             *option_manager_.database_path,
                             reconstruction_manager_);
  mapper.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
  mapper.Run();

  CreateDirIfNotExists(sparse_path);
  reconstruction_manager_->Write(sparse_path);
  option_manager_.Write(JoinPaths(sparse_path, "project.ini"));
}

// 执行密集映射的具体函数实现
void AutomaticReconstructionController::RunDenseMapper() {
  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "dense"));

  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    if (IsStopped()) {
      return;
    }

    const std::string dense_path =
        JoinPaths(options_.workspace_path, "dense", std::to_string(i));
    const std::string fused_path = JoinPaths(dense_path, "fused.ply");

    std::string meshing_path;
    if (options_.mesher == Mesher::POISSON) {
      meshing_path = JoinPaths(dense_path, "meshed-poisson.ply");
    } else if (options_.mesher == Mesher::DELAUNAY) {
      meshing_path = JoinPaths(dense_path, "meshed-delaunay.ply");
    }

    if (ExistsFile(fused_path) && ExistsFile(meshing_path)) {
      continue;
    }

    // Image undistortion.
// 图像去畸变操作
    if (!ExistsDir(dense_path)) {
      CreateDirIfNotExists(dense_path);

      UndistortCameraOptions undistortion_options;
      undistortion_options.max_image_size =
          option_manager_.patch_match_stereo->max_image_size;
      COLMAPUndistorter undistorter(undistortion_options,
                                    *reconstruction_manager_->Get(i),
                                    *option_manager_.image_path,
                                    dense_path);
      undistorter.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
      undistorter.Run();
    }

    if (IsStopped()) {
      return;
    }

    // Patch match stereo.
// 块匹配立体操作
#if defined(COLMAP_CUDA_ENABLED)
    {
      mvs::PatchMatchController patch_match_controller(
          *option_manager_.patch_match_stereo, dense_path, "COLMAP", "");
      patch_match_controller.SetCheckIfStoppedFunc(
          [&]() { return IsStopped(); });
      patch_match_controller.Run();
    }
#else   // COLMAP_CUDA_ENABLED
    LOG(WARNING) << "Skipping patch match stereo because CUDA is not available";
    return;
#endif  // COLMAP_CUDA_ENABLED

    if (IsStopped()) {
      return;
    }

   // 立体融合操作
    if (!ExistsFile(fused_path)) {
      auto fusion_options = *option_manager_.stereo_fusion;
      const int num_reg_images =
          reconstruction_manager_->Get(i)->NumRegImages();
      fusion_options.min_num_pixels =
          std::min(num_reg_images + 1, fusion_options.min_num_pixels);
      mvs::StereoFusion fuser(
          fusion_options,
          dense_path,
          "COLMAP",
          "",
          options_.quality == Quality::HIGH ? "geometric" : "photometric");
      fuser.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
      fuser.Run();

      LOG(INFO) << "Writing output: " << fused_path;
      WriteBinaryPlyPoints(fused_path, fuser.GetFusedPoints());
      mvs::WritePointsVisibility(fused_path + ".vis",
                                 fuser.GetFusedPointsVisibility());
    }

    if (IsStopped()) {
      return;
    }

 // 表面网格生成操作
    if (!ExistsFile(meshing_path)) {
      if (options_.mesher == Mesher::POISSON) {
        mvs::PoissonMeshing(
            *option_manager_.poisson_meshing, fused_path, meshing_path);
      } else if (options_.mesher == Mesher::DELAUNAY) {
#if defined(COLMAP_CGAL_ENABLED)
        mvs::DenseDelaunayMeshing(
            *option_manager_.delaunay_meshing, dense_path, meshing_path);
#else  // COLMAP_CGAL_ENABLED
        LOG(WARNING)
            << "Skipping Delaunay meshing because CGAL is not available";
        return;

#endif  // COLMAP_CGAL_ENABLED
      }
    }
  }
}

}  // namespace colmap
