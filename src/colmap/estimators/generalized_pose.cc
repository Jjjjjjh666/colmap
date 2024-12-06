// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// 版权所有，保留所有权利。
//
// 允许在源代码和二进制形式中重新分发和使用，是否修改均可，前提是满足以下条件：
//
//     * 源代码的重新分发必须保留上述版权声明、条件列表以及以下免责声明。
//     * 二进制形式的重新分发必须在分发的文档或其他材料中复制上述版权声明、条件列表和免责声明。
//     * 未经事先书面许可，不得使用ETH Zurich和UNC Chapel Hill的名称及其贡献者的名称来为由本软件派生的产品做宣传或推广。
//
// 本软件由版权持有者和贡献者按“原样”提供，不作任何明示或暗示的担保，包括但不限于对适销性和特定用途适用性的暗示担保。在任何情况下，版权持有者或贡献者对因使用本软件而产生的任何直接、间接、附带、特殊、示范性或后果性损害（包括但不限于采购替代商品或服务的费用、使用、数据或利润损失，或业务中断）不承担责任，即使事先已被告知可能发生此类损害。

#include "colmap/estimators/generalized_pose.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/generalized_absolute_pose.h"
#include "colmap/estimators/manifold.h"
#include "colmap/estimators/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/matrix.h"
#include "colmap/optim/ransac.h"
#include "colmap/optim/support_measurement.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {
namespace {

// 计算相机最大误差函数
double ComputeMaxErrorInCamera(const std::vector<size_t>& camera_idxs,
                               const std::vector<Camera>& cameras,
                               const double max_error_px) {
  THROW_CHECK_GT(max_error_px, 0.0);  // 确保最大误差大于0
  double max_error_cam = 0.;
  for (const auto& camera_idx : camera_idxs) {
    max_error_cam += cameras[camera_idx].CamFromImgThreshold(max_error_px);
  }
  return max_error_cam / camera_idxs.size();  // 返回相机的最大误差平均值
}

// 比较两个3D向量大小
bool LowerVector3d(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
  if (v1.x() < v2.x()) {
    return true;
  } else if (v1.x() == v2.x()) {
    if (v1.y() < v2.y()) {
      return true;
    } else if (v1.y() == v2.y()) {
      return v1.z() < v2.z();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

// 计算3D点的唯一ID，防止在多相机系统中重复计数相同的点
std::vector<size_t> ComputeUniquePointIds(
    const std::vector<Eigen::Vector3d>& points3D) {
  std::vector<size_t> point3D_ids(points3D.size());
  std::iota(point3D_ids.begin(), point3D_ids.end(), 0);  // 为每个点分配唯一ID
  std::sort(point3D_ids.begin(), point3D_ids.end(), [&](size_t i, size_t j) {
    return LowerVector3d(points3D[i], points3D[j]);  // 按3D点的坐标进行排序
  });

  std::vector<size_t>::iterator unique_it = point3D_ids.begin();
  std::vector<size_t>::iterator current_it = point3D_ids.begin();
  std::vector<size_t> unique_point3D_ids(points3D.size());
  while (current_it != point3D_ids.end()) {
    if (!points3D[*unique_it].isApprox(points3D[*current_it], 1e-5)) {
      unique_it = current_it;
    }
    unique_point3D_ids[*current_it] = unique_it - point3D_ids.begin();  // 设置每个点的唯一ID
    current_it++;
  }
  return unique_point3D_ids;
}

}  // namespace

// 估计广义绝对姿态
bool EstimateGeneralizedAbsolutePose(
    const RANSACOptions& options,
    const std::vector<Eigen::Vector2d>& points2D,
    const std::vector<Eigen::Vector3d>& points3D,
    const std::vector<size_t>& camera_idxs,
    const std::vector<Rigid3d>& cams_from_rig,
    const std::vector<Camera>& cameras,
    Rigid3d* rig_from_world,
    size_t* num_inliers,
    std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());  // 确保2D点和3D点数目相同
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());  // 确保2D点和相机索引数目相同
  THROW_CHECK_EQ(cams_from_rig.size(), cameras.size());  // 确保相机和相机姿态数目相同
  THROW_CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);  // 检查相机索引有效
  THROW_CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()),
                 cameras.size());  // 确保相机索引在范围内
  options.Check();  // 检查RANSAC选项是否有效
  if (points2D.size() == 0) {
    return false;  // 如果没有点对，则返回失败
  }

  // 为每个2D点计算相机的射线，并根据相机位置生成rig
  std::vector<GP3PEstimator::X_t> rig_points2D(points2D.size());
  for (size_t i = 0; i < points2D.size(); i++) {
    const size_t camera_idx = camera_idxs[i];
    rig_points2D[i].ray_in_cam =
        cameras[camera_idx].CamFromImg(points2D[i]).homogeneous().normalized();
    rig_points2D[i].cam_from_rig = cams_from_rig[camera_idx];
  }

  // 为每个3D点生成唯一ID
  const std::vector<size_t> unique_point3D_ids =
      ComputeUniquePointIds(points3D);

  // 计算相机的最大误差，并为RANSAC设置误差容忍度
  RANSACOptions options_copy(options);
  options_copy.max_error =
      ComputeMaxErrorInCamera(camera_idxs, cameras, options.max_error);

  RANSAC<GP3PEstimator, UniqueInlierSupportMeasurer> ransac(options_copy);
  ransac.support_measurer.SetUniqueSampleIds(unique_point3D_ids);  // 设置唯一点ID
  ransac.estimator.residual_type =
      GP3PEstimator::ResidualType::ReprojectionError;  // 使用重投影误差进行估计
  const auto report = ransac.Estimate(rig_points2D, points3D);  // 执行RANSAC估计
  if (!report.success) {
    return false;  // 如果RANSAC估计失败，则返回失败
  }
  *rig_from_world = report.model;  // 设置最终模型
  *num_inliers = report.support.num_unique_inliers;  // 设置内点数目
  *inlier_mask = report.inlier_mask;  // 设置内点掩码
  return true;
}
// 函数：RefineGeneralizedAbsolutePose
// 该函数用于通过Ceres优化来细化一个广义绝对位姿的估计，基于2D-3D点对、相机内参和外参的初步估计。
// 参数：
//   options：包含优化相关设置的选项。
//   inlier_mask：一个布尔型向量，指示哪些匹配点是内点。
//   points2D：2D点坐标的列表。
//   points3D：3D点坐标的列表。
//   camera_idxs：每个2D点对应的相机索引。
//   cams_from_rig：每个相机的相机到rig的变换矩阵。
//   rig_from_world：输出参数，表示rig到世界坐标系的位姿。
//   cameras：输入输出参数，表示所有相机的内参和外参。
//   rig_from_world_cov：输出参数，表示rig从世界坐标系转换的协方差矩阵。
// 返回值：
//   如果优化成功，返回true，否则返回false。
bool RefineGeneralizedAbsolutePose(const AbsolutePoseRefinementOptions& options,
                                   const std::vector<char>& inlier_mask,
                                   const std::vector<Eigen::Vector2d>& points2D,
                                   const std::vector<Eigen::Vector3d>& points3D,
                                   const std::vector<size_t>& camera_idxs,
                                   const std::vector<Rigid3d>& cams_from_rig,
                                   Rigid3d* rig_from_world,
                                   std::vector<Camera>* cameras,
                                   Eigen::Matrix6d* rig_from_world_cov) {

  // 检查输入数据的大小是否一致
  THROW_CHECK_EQ(points2D.size(), inlier_mask.size());
  THROW_CHECK_EQ(points2D.size(), points3D.size());
  THROW_CHECK_EQ(points2D.size(), camera_idxs.size());
  THROW_CHECK_EQ(cams_from_rig.size(), cameras->size());
  
  // 检查相机索引范围是否有效
  THROW_CHECK_GE(*std::min_element(camera_idxs.begin(), camera_idxs.end()), 0);
  THROW_CHECK_LT(*std::max_element(camera_idxs.begin(), camera_idxs.end()), cameras->size());

  // 检查优化选项是否正确
  options.Check();

  // 使用Cauchy损失函数来减少对离群点的敏感度
  const auto loss_function = std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);

  // 准备相机内参数据
  std::vector<double*> cameras_params_data;
  for (size_t i = 0; i < cameras->size(); i++) {
    cameras_params_data.push_back(cameras->at(i).params.data());
  }

  // 初始化相机计数和rig到世界的旋转和平移参数
  std::vector<size_t> camera_counts(cameras->size(), 0);
  double* rig_from_world_rotation = rig_from_world->rotation.coeffs().data();
  double* rig_from_world_translation = rig_from_world->translation.data();

  // 创建问题并初始化优化
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  // 迭代所有2D-3D点对
  for (size_t i = 0; i < points2D.size(); ++i) {
    // 跳过离群点
    if (!inlier_mask[i]) {
      continue;
    }

    // 获取当前点对应的相机索引
    const size_t camera_idx = camera_idxs[i];
    camera_counts[camera_idx] += 1;

    // 将残差块添加到优化问题中
    problem.AddResidualBlock(
        CameraCostFunction<RigReprojErrorCostFunctor>(
            cameras->at(camera_idx).model_id, points2D[i]),
        loss_function.get(),
        cams_from_rig[camera_idx].rotation.coeffs().data(),
        cams_from_rig[camera_idx].translation.data(),
        rig_from_world_rotation,
        rig_from_world_translation,
        points3D[i].data(),
        cameras_params_data[camera_idx]);

    // 固定3D点的位置
    problem.SetParameterBlockConstant(points3D[i].data());
  }

  // 如果问题有残差，则继续进行优化
  if (problem.NumResiduals() > 0) {
    // 设置四元数曼哈顿集
    SetQuaternionManifold(&problem, rig_from_world_rotation);

    // 设置相机内参的优化
    for (size_t i = 0; i < cameras->size(); i++) {
      if (camera_counts[i] == 0) continue;
      Camera& camera = cameras->at(i);

      // 固定rig的参数，因为它通常是约束不足的
      problem.SetParameterBlockConstant(
          cams_from_rig[i].rotation.coeffs().data());
      problem.SetParameterBlockConstant(
          cams_from_rig[i].translation.data());

      // 根据选项来决定是否优化焦距和其他附加参数
      if (!options.refine_focal_length && !options.refine_extra_params) {
        problem.SetParameterBlockConstant(camera.params.data());
      } else {
        // 总是将主点设置为固定参数
        std::vector<int> camera_params_const;
        const span<const size_t> principal_point_idxs =
            camera.PrincipalPointIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   principal_point_idxs.begin(),
                                   principal_point_idxs.end());

        // 根据选项决定是否优化焦距
        if (!options.refine_focal_length) {
          const span<const size_t> focal_length_idxs = camera.FocalLengthIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     focal_length_idxs.begin(),
                                     focal_length_idxs.end());
        }

        // 根据选项决定是否优化额外参数
        if (!options.refine_extra_params) {
          const span<const size_t> extra_params_idxs = camera.ExtraParamsIdxs();
          camera_params_const.insert(camera_params_const.end(),
                                     extra_params_idxs.begin(),
                                     extra_params_idxs.end());
        }

        // 如果所有参数都需要优化，设置为常量
        if (camera_params_const.size() == camera.params.size()) {
          problem.SetParameterBlockConstant(camera.params.data());
        } else {
          // 否则，优化指定的子集参数
          SetSubsetManifold(static_cast<int>(camera.params.size()),
                            camera_params_const,
                            &problem,
                            camera.params.data());
        }
      }
    }
  }

  // 设置求解器选项并开始求解
  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.logging_type = ceres::LoggingType::SILENT;

  // 降低线程创建的开销
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  // 执行求解
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // 如果选项要求，打印求解器的总结信息
  if (options.print_summary || VLOG_IS_ON(1)) {
    PrintSolverSummary(summary, "Pose refinement report");
  }

  // 计算协方差矩阵（如果需要）
  if (problem.NumResiduals() > 0 && rig_from_world_cov != nullptr) {
    ceres::Covariance::Options covariance_options;
    ceres::Covariance covariance(covariance_options);
    std::vector<const double*> parameter_blocks = {rig_from_world_rotation,
                                                   rig_from_world_translation};
    if (!covariance.Compute(parameter_blocks, &problem)) {
      return false;
    }
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 rig_from_world_cov->data());
  }

  // 返回优化是否成功
  return summary.IsSolutionUsable();
}
