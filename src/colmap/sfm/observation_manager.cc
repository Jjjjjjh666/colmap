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

#include "colmap/sfm/observation_manager.h"

#include "colmap/estimators/alignment.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/projection.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

namespace colmap {

bool MergeAndFilterReconstructions(const double max_reproj_error,  //最大重投影误差，通常用于衡量点在重建过程中投影到图像上的误差大小，超过此阈值的点会被视为不准确。
                                   const Reconstruction& src_reconstruction,  //源重建（Reconstruction类型的对象），即需要被合并到目标重建中的数据。
                                   Reconstruction& tgt_reconstruction) {    //目标重建（也是Reconstruction类型的对象），即合并后的结果将存储到此对象中。
  if (!MergeReconstructions(
          max_reproj_error, src_reconstruction, tgt_reconstruction)) {
    return false;
  }  //调用MergeReconstructions重建函数，负责合并源重建和目标重建，使用最大重投影误差作为合并过程中的评判标准
  ObservationManager(tgt_reconstruction)
      .FilterAllPoints3D(max_reproj_error, /*min_tri_angle=*/0);  //对目标重建中所有的3D点进行过滤操作
  return true;                                                    //具体过滤：1.第一个参数是对所有3D点进行过滤，过滤掉重投影误差过大的点
}                                                                           //2.第二个参数值默认为0，过滤掉三角形角度过小的3D点
//函数首先尝试合并源重建和目标重建，若合并成功，则接下来对目标重建中的所有3D点进行过滤，过滤掉重投影误差过大或
//三角形角度过小的点，确保合并后的3D重建更加稳定精确


const int ObservationManager::kNumPoint3DVisibilityPyramidLevels = 6;  //这是一个类级常量，定义了3D点在图像中的可见性金字塔的层数，设为6。这通常用于优化点云可见性的管理，以适应不同的尺度和视角。

ObservationManager::ObservationManager(  //初始化
    Reconstruction& reconstruction,   //3D重建对象，包含了3D重建的所有图像和点
    std::shared_ptr<const CorrespondenceGraph> correspondence_graph)  //智能指针，存储了不同图像间的对应关系
    : reconstruction_(reconstruction),   
      correspondence_graph_(std::move(correspondence_graph)) {
  // Add image pairs.处理图像对的对应关系
  if (correspondence_graph_) {  如果correspondence_graph_存在
    image_pair_stats_.reserve(correspondence_graph_->NumImagePairs()); //预分配image_pair_stats_容器的大小，以容纳图像对的统计数据。
    for (const auto& image_pair :    //遍历correspondence_graph_->NumCorrespondencesBetweenImages()，获取所有图像对及其对应点的数量（通过image_pair.second）
         correspondence_graph_->NumCorrespondencesBetweenImages()) {
      //为每个图像对创建一个ImagePairStat对象，并将该图像对及其对应统计信息插入到image_pair_stats_中。
      ImagePairStat image_pair_stat;
      image_pair_stat.num_total_corrs = image_pair.second;
      image_pair_stats_.emplace(image_pair.first, image_pair_stat);
    }
  }

  // Add image stats.处理每个图像的统计信息
  image_stats_.reserve(reconstruction_.NumImages());  //预分配容器大小
  for (const auto& [image_id, image] : reconstruction_.Images()) {  //遍历reconstruction_中的所有图像
    const Camera& camera = *image.CameraPtr();  //获取该图像的Camera对象
    ImageStat image_stat;  //创建一个ImageStat对象来存储该图像的统计信息
    image_stat.point3D_visibility_pyramid = VisibilityPyramid(
        kNumPoint3DVisibilityPyramidLevels, camera.width, camera.height);  //初始化该图像的point3D_visibility_pyramid，用于存储图像中3D点的可见性金字塔（它有6个层级，基于图像的宽度和高度来创建）。
    image_stat.num_correspondences_have_point3D.resize(image.NumPoints2D(), 0);  //初始化num_correspondences_have_point3D数组，它的大小等于图像中2D点的数量，所有值初始化为0，表示每个2D点的3D对应点的数量
    image_stat.num_visible_points3D = 0;  //初始化num_visible_points3D为0，表示图像中当前可见的3D点的数量。
    if (correspondence_graph_ && correspondence_graph_->ExistsImage(image_id)) {  //如果correspondence_graph_存在并且当前图像ID在该图像图中有数据
      image_stat.num_observations =
          correspondence_graph_->NumObservationsForImage(image_id);  //记录图像的观测次数（即该图像在3D重建中被看到的次数）。
      image_stat.num_correspondences =
          correspondence_graph_->NumCorrespondencesForImage(image_id);  //记录图像的对应点数量（即该图像的2D点与其他图像的对应关系的数量）。
    }
    image_stats_.emplace(image_id, image_stat);  //将每个图像的统计数据插入到image_stats_中
  }

  // If an existing model was loaded from disk and there were already images
  // registered previously, we need to set observations as triangulated.
  //处理已经三角化的点
  for (const auto image_id : reconstruction_.RegImageIds()) {  //遍历已经注册的图像
    const Image& image = reconstruction_.Image(image_id);  //通过reconstruction_.RegImageIds()获取图像ID
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {    //对每个图像，遍历它的所有2D点（通过image.NumPoints2D()）
      if (image.Point2D(point2D_idx).HasPoint3D()) {  //如果该点已经有一个3D点，则调用SetObservationAsTriangulated方法，表示该2D点的观测已被三角化。
        const bool kIsContinuedPoint3D = false;  //kIsContinuedPoint3D为false，表示该点不是一个继续点（即一个已经存在的3D点）。这个标志通常用于区分新三角化的点与继续观测的点。
        SetObservationAsTriangulated(
            image_id, point2D_idx, kIsContinuedPoint3D);
      }
    }
  }
}
//ObservationManager构造函数的主要目的是初始化与3D重建和图像对应关系相关的所有数据，以便后续操作中能够有效地管理和优化这些数据。
//它通过填充图像统计、图像对统计和处理三角化点，为后续的重建优化和数据管理提供了基础。

void ObservationManager::IncrementCorrespondenceHasPoint3D(
    const image_t image_id, const point2D_t point2D_idx) {
  const Image& image = reconstruction_.Image(image_id);
  const struct Point2D& point2D = image.Point2D(point2D_idx);
  ImageStat& stats = image_stats_.at(image_id);

  stats.num_correspondences_have_point3D[point2D_idx] += 1;
  if (stats.num_correspondences_have_point3D[point2D_idx] == 1) {
    stats.num_visible_points3D += 1;
  }

  stats.point3D_visibility_pyramid.SetPoint(point2D.xy(0), point2D.xy(1));

  assert(stats.num_visible_points3D <= stats.num_observations);
}

void ObservationManager::DecrementCorrespondenceHasPoint3D(
    const image_t image_id, const point2D_t point2D_idx) {
  const Image& image = reconstruction_.Image(image_id);
  const struct Point2D& point2D = image.Point2D(point2D_idx);
  ImageStat& stats = image_stats_.at(image_id);

  stats.num_correspondences_have_point3D[point2D_idx] -= 1;
  if (stats.num_correspondences_have_point3D[point2D_idx] == 0) {
    stats.num_visible_points3D -= 1;
  }

  stats.point3D_visibility_pyramid.ResetPoint(point2D.xy(0), point2D.xy(1));

  assert(stats.num_visible_points3D <= stats.num_observations);
}

void ObservationManager::SetObservationAsTriangulated(
    const image_t image_id,
    const point2D_t point2D_idx,
    const bool is_continued_point3D) {
  if (correspondence_graph_ == nullptr) {
    return;
  }
  const Image& image = reconstruction_.Image(image_id);
  THROW_CHECK(image.HasPose());

  const Point2D& point2D = image.Point2D(point2D_idx);
  THROW_CHECK(point2D.HasPoint3D());

  const auto corr_range =
      correspondence_graph_->FindCorrespondences(image_id, point2D_idx);
  for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
    Image& corr_image = reconstruction_.Image(corr->image_id);
    const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
    IncrementCorrespondenceHasPoint3D(corr->image_id, corr->point2D_idx);
    // Update number of shared 3D points between image pairs and make sure to
    // only count the correspondences once (not twice forward and backward).
    if (point2D.point3D_id == corr_point2D.point3D_id &&
        (is_continued_point3D || image_id < corr->image_id)) {
      const image_pair_t pair_id =
          Database::ImagePairToPairId(image_id, corr->image_id);
      auto& stats = image_pair_stats_[pair_id];
      stats.num_tri_corrs += 1;
      THROW_CHECK_LE(stats.num_tri_corrs, stats.num_total_corrs)
          << "The correspondence graph must not contain duplicate matches: "
          << corr->image_id << " " << corr->point2D_idx;
    }
  }
}

void ObservationManager::ResetTriObservations(const image_t image_id,
                                              const point2D_t point2D_idx,
                                              const bool is_deleted_point3D) {
  if (correspondence_graph_ == nullptr) {
    return;
  }
  const Image& image = reconstruction_.Image(image_id);
  THROW_CHECK(image.HasPose());
  const Point2D& point2D = image.Point2D(point2D_idx);
  THROW_CHECK(point2D.HasPoint3D());

  const auto corr_range =
      correspondence_graph_->FindCorrespondences(image_id, point2D_idx);
  for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
    Image& corr_image = reconstruction_.Image(corr->image_id);
    const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
    DecrementCorrespondenceHasPoint3D(corr->image_id, corr->point2D_idx);
    // Update number of shared 3D points between image pairs and make sure to
    // only count the correspondences once (not twice forward and backward).
    if (point2D.point3D_id == corr_point2D.point3D_id &&
        (!is_deleted_point3D || image_id < corr->image_id)) {
      const image_pair_t pair_id =
          Database::ImagePairToPairId(image_id, corr->image_id);
      THROW_CHECK_GT(image_pair_stats_[pair_id].num_tri_corrs, 0)
          << "The scene graph graph must not contain duplicate matches";
      image_pair_stats_[pair_id].num_tri_corrs -= 1;
    }
  }
}

point3D_t ObservationManager::AddPoint3D(const Eigen::Vector3d& xyz,
                                         const Track& track,
                                         const Eigen::Vector3ub& color) {
  const point3D_t point3D_id = reconstruction_.AddPoint3D(xyz, track, color);

  const bool kIsContinuedPoint3D = false;
  for (const auto& track_el : track.Elements()) {
    SetObservationAsTriangulated(
        track_el.image_id, track_el.point2D_idx, kIsContinuedPoint3D);
  }

  return point3D_id;
}

void ObservationManager::AddObservation(const point3D_t point3D_id,
                                        const TrackElement& track_el) {
  reconstruction_.AddObservation(point3D_id, track_el);
  const bool kIsContinuedPoint3D = true;
  SetObservationAsTriangulated(
      track_el.image_id, track_el.point2D_idx, kIsContinuedPoint3D);
}

void ObservationManager::DeletePoint3D(const point3D_t point3D_id) {
  // Note: Do not change order of these instructions, especially with respect to
  // `ObservationManager::ResetTriObservations`
  const Track& track = reconstruction_.Point3D(point3D_id).track;
  const bool kIsDeletedPoint3D = true;
  for (const auto& track_el : track.Elements()) {
    ResetTriObservations(
        track_el.image_id, track_el.point2D_idx, kIsDeletedPoint3D);
  }

  reconstruction_.DeletePoint3D(point3D_id);
}

void ObservationManager::DeleteObservation(const image_t image_id,
                                           const point2D_t point2D_idx) {
  // Note: Do not change order of these instructions, especially with respect to
  // `ObservationManager::ResetTriObservations`
  Image& image = reconstruction_.Image(image_id);
  const point3D_t point3D_id = image.Point2D(point2D_idx).point3D_id;
  struct Point3D& point3D = reconstruction_.Point3D(point3D_id);

  if (point3D.track.Length() <= 2) {
    DeletePoint3D(point3D_id);
    return;
  }

  const bool kIsDeletedPoint3D = false;
  ResetTriObservations(image_id, point2D_idx, kIsDeletedPoint3D);
  reconstruction_.DeleteObservation(image_id, point2D_idx);
}

point3D_t ObservationManager::MergePoints3D(const point3D_t point3D_id1,
                                            const point3D_t point3D_id2) {
  const bool kIsDeletedPoint3D = true;
  const Track& track1 = reconstruction_.Point3D(point3D_id1).track;
  for (const auto& track_el : track1.Elements()) {
    ResetTriObservations(
        track_el.image_id, track_el.point2D_idx, kIsDeletedPoint3D);
  }
  const Track& track2 = reconstruction_.Point3D(point3D_id2).track;
  for (const auto& track_el : track2.Elements()) {
    ResetTriObservations(
        track_el.image_id, track_el.point2D_idx, kIsDeletedPoint3D);
  }

  point3D_t merged_point3D_id =
      reconstruction_.MergePoints3D(point3D_id1, point3D_id2);

  const Track track = reconstruction_.Point3D(merged_point3D_id).track;
  const bool kIsContinuedPoint3D = false;
  for (const auto& track_el : track.Elements()) {
    SetObservationAsTriangulated(
        track_el.image_id, track_el.point2D_idx, kIsContinuedPoint3D);
  }
  return merged_point3D_id;
}

size_t ObservationManager::FilterPoints3D(
    const double max_reproj_error,
    const double min_tri_angle,
    const std::unordered_set<point3D_t>& point3D_ids) {
  size_t num_filtered = 0;
  num_filtered +=
      FilterPoints3DWithLargeReprojectionError(max_reproj_error, point3D_ids);
  num_filtered +=
      FilterPoints3DWithSmallTriangulationAngle(min_tri_angle, point3D_ids);
  return num_filtered;
}

size_t ObservationManager::FilterPoints3DInImages(
    const double max_reproj_error,
    const double min_tri_angle,
    const std::unordered_set<image_t>& image_ids) {
  std::unordered_set<point3D_t> point3D_ids;
  for (const image_t image_id : image_ids) {
    const Image& image = reconstruction_.Image(image_id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids.insert(point2D.point3D_id);
      }
    }
  }
  return FilterPoints3D(max_reproj_error, min_tri_angle, point3D_ids);
}

size_t ObservationManager::FilterAllPoints3D(const double max_reproj_error,
                                             const double min_tri_angle) {
  // Important: First filter observations and points with large reprojection
  // error, so that observations with large reprojection error do not make
  // a point stable through a large triangulation angle.
  const std::unordered_set<point3D_t>& point3D_ids =
      reconstruction_.Point3DIds();
  size_t num_filtered = 0;
  num_filtered +=
      FilterPoints3DWithLargeReprojectionError(max_reproj_error, point3D_ids);
  num_filtered +=
      FilterPoints3DWithSmallTriangulationAngle(min_tri_angle, point3D_ids);
  return num_filtered;
}

size_t ObservationManager::FilterObservationsWithNegativeDepth() {
  size_t num_filtered = 0;
  for (const auto image_id : reconstruction_.RegImageIds()) {
    const Image& image = reconstruction_.Image(image_id);
    const Eigen::Matrix3x4d cam_from_world = image.CamFromWorld().ToMatrix();
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        const struct Point3D& point3D =
            reconstruction_.Point3D(point2D.point3D_id);
        if (!HasPointPositiveDepth(cam_from_world, point3D.xyz)) {
          DeleteObservation(image_id, point2D_idx);
          num_filtered += 1;
        }
      }
    }
  }
  return num_filtered;
}

size_t ObservationManager::FilterPoints3DWithSmallTriangulationAngle(
    const double min_tri_angle,
    const std::unordered_set<point3D_t>& point3D_ids) {
  // Number of filtered points.
  size_t num_filtered = 0;

  // Minimum triangulation angle in radians.
  const double min_tri_angle_rad = DegToRad(min_tri_angle);

  // Cache for image projection centers.
  std::unordered_map<image_t, Eigen::Vector3d> proj_centers;

  for (const auto point3D_id : point3D_ids) {
    if (!reconstruction_.ExistsPoint3D(point3D_id)) {
      continue;
    }

    const struct Point3D& point3D = reconstruction_.Point3D(point3D_id);

    // Calculate triangulation angle for all pairwise combinations of image
    // poses in the track. Only delete point if none of the combinations
    // has a sufficient triangulation angle.
    bool keep_point = false;
    for (size_t i1 = 0; i1 < point3D.track.Length(); ++i1) {
      const image_t image_id1 = point3D.track.Element(i1).image_id;

      Eigen::Vector3d proj_center1;
      if (proj_centers.count(image_id1) == 0) {
        const Image& image1 = reconstruction_.Image(image_id1);
        proj_center1 = image1.ProjectionCenter();
        proj_centers.emplace(image_id1, proj_center1);
      } else {
        proj_center1 = proj_centers.at(image_id1);
      }

      for (size_t i2 = 0; i2 < i1; ++i2) {
        const image_t image_id2 = point3D.track.Element(i2).image_id;
        const Eigen::Vector3d proj_center2 = proj_centers.at(image_id2);

        const double tri_angle = CalculateTriangulationAngle(
            proj_center1, proj_center2, point3D.xyz);

        if (tri_angle >= min_tri_angle_rad) {
          keep_point = true;
          break;
        }
      }

      if (keep_point) {
        break;
      }
    }

    if (!keep_point) {
      num_filtered += 1;
      DeletePoint3D(point3D_id);
    }
  }

  return num_filtered;
}

size_t ObservationManager::FilterPoints3DWithLargeReprojectionError(
    const double max_reproj_error,
    const std::unordered_set<point3D_t>& point3D_ids) {
  const double max_squared_reproj_error = max_reproj_error * max_reproj_error;

  // Number of filtered points.
  size_t num_filtered = 0;

  for (const auto point3D_id : point3D_ids) {
    if (!reconstruction_.ExistsPoint3D(point3D_id)) {
      continue;
    }

    struct Point3D& point3D = reconstruction_.Point3D(point3D_id);

    if (point3D.track.Length() < 2) {
      num_filtered += point3D.track.Length();
      DeletePoint3D(point3D_id);
      continue;
    }

    double reproj_error_sum = 0.0;

    std::vector<TrackElement> track_els_to_delete;

    for (const auto& track_el : point3D.track.Elements()) {
      const Image& image = reconstruction_.Image(track_el.image_id);
      const struct Camera& camera = *image.CameraPtr();
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      const double squared_reproj_error = CalculateSquaredReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      if (squared_reproj_error > max_squared_reproj_error) {
        track_els_to_delete.push_back(track_el);
      } else {
        reproj_error_sum += std::sqrt(squared_reproj_error);
      }
    }

    if (track_els_to_delete.size() >= point3D.track.Length() - 1) {
      num_filtered += point3D.track.Length();
      DeletePoint3D(point3D_id);
    } else {
      num_filtered += track_els_to_delete.size();
      for (const auto& track_el : track_els_to_delete) {
        DeleteObservation(track_el.image_id, track_el.point2D_idx);
      }
      point3D.error = reproj_error_sum / point3D.track.Length();
    }
  }

  return num_filtered;
}

void ObservationManager::DeRegisterImage(const image_t image_id) {
  Image& image = reconstruction_.Image(image_id);
  const auto num_points2D = image.NumPoints2D();
  for (point2D_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
    if (image.Point2D(point2D_idx).HasPoint3D()) {
      DeleteObservation(image_id, point2D_idx);
    }
  }
  reconstruction_.DeRegisterImage(image_id);
}

std::vector<image_t> ObservationManager::FilterImages(
    const double min_focal_length_ratio,
    const double max_focal_length_ratio,
    const double max_extra_param) {
  std::vector<image_t> filtered_image_ids;
  for (const image_t image_id : reconstruction_.RegImageIds()) {
    const Image& image = reconstruction_.Image(image_id);
    if (image.NumPoints3D() == 0 ||
        image.CameraPtr()->HasBogusParams(
            min_focal_length_ratio, max_focal_length_ratio, max_extra_param)) {
      filtered_image_ids.push_back(image_id);
    }
  }

  // Only de-register after iterating over reg_image_ids_ to avoid
  // simultaneous iteration and modification of the vector.
  for (const image_t image_id : filtered_image_ids) {
    DeRegisterImage(image_id);
  }

  return filtered_image_ids;
}

}  // namespace colmap
