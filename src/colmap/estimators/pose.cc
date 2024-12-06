// 版权所有 (c) 2023，苏黎世联邦理工大学和北卡罗来纳大学教堂山分校。
// 保留所有权利。
//
// 允许在源代码和二进制形式中进行再分发和使用，无论是否修改，
// 只要满足以下条件：
//
//     * 源代码的再分发必须保留上述版权声明、条件列表和以下免责声明。
//     * 二进制形式的再分发必须在随附的文档或其他材料中复制上述版权声明、条件列表和以下免责声明。
//     * 在没有特定的书面许可的情况下，不能使用苏黎世联邦理工大学和北卡罗来纳大学教堂山分校的名称，
//       也不能使用其贡献者的名称来为衍生产品背书或宣传。
//
// 本软件是由版权持有者和贡献者“按原样”提供的，
// 不附带任何明示或暗示的担保，包括但不限于适销性和特定用途的适用性。
// 在任何情况下，版权持有者或贡献者均不对因使用本软件而产生的任何直接、间接、偶然、特殊、示范性或
// 间接损害（包括但不限于采购替代商品或服务、使用丧失、数据丧失或利润损失、业务中断等）负责，
// 无论是在合同、严格责任还是侵权（包括过失或其他）理论下产生的，
// 即使已被告知可能发生此类损害。

#include "colmap/estimators/pose.h"

#include "colmap/estimators/absolute_pose.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/cost_functions.h"
#include "colmap/estimators/essential_matrix.h"
#include "colmap/estimators/manifold.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"

namespace colmap {

// 估计绝对姿态的函数
// 该函数通过RANSAC方法估计给定3D点和2D点的相机姿态
bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Rigid3d* cam_from_world,
                          Camera* camera,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());  // 确保2D点和3D点的数量一致
  options.Check();  // 检查配置是否有效

  *num_inliers = 0;  // 初始化内点数量
  inlier_mask->clear();  // 清空内点掩码

  std::vector<Eigen::Vector2d> points2D_normalized(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    points2D_normalized[i] = camera->CamFromImg(points2D[i]);  // 将2D图像坐标转换为归一化相机坐标
  }

  auto custom_ransac_options = options.ransac_options;
  custom_ransac_options.max_error =
      camera->CamFromImgThreshold(options.ransac_options.max_error);  // 设置最大误差阈值

  if (options.estimate_focal_length) {
    // 如果需要估计焦距，使用P4PF估计器进行RANSAC
    RANSAC<P4PFEstimator> ransac(custom_ransac_options);
    auto report = ransac.Estimate(points2D_normalized, points3D);  // 使用RANSAC进行估计
    if (report.success) {  // 如果估计成功
      *cam_from_world =
          Rigid3d(Eigen::Quaterniond(report.model.cam_from_world.leftCols<3>()),
                  report.model.cam_from_world.col(3));  // 设置相机姿态
      for (const size_t idx : camera->FocalLengthIdxs()) {
        camera->params[idx] *= report.model.focal_length;  // 更新焦距
      }
      *num_inliers = report.support.num_inliers;  // 更新内点数量
      *inlier_mask = std::move(report.inlier_mask);  // 更新内点掩码
      return true;  // 返回估计成功
    }
  } else {
    // 如果不需要估计焦距，使用P3P估计器进行LORANSAC估计
    LORANSAC<P3PEstimator, EPNPEstimator> ransac(custom_ransac_options);
    auto report = ransac.Estimate(points2D_normalized, points3D);  // 使用LORANSAC进行估计
    if (report.success) {  // 如果估计成功
      *cam_from_world = Rigid3d(Eigen::Quaterniond(report.model.leftCols<3>()),
                                report.model.col(3));  // 设置相机姿态
      *num_inliers = report.support.num_inliers;  // 更新内点数量
      *inlier_mask = std::move(report.inlier_mask);  // 更新内点掩码
      return true;  // 返回估计成功
    }
  }

  return false;  // 如果估计失败，返回false
}

// 估计两台相机之间的相对姿态（使用本质矩阵法估计）
size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Rigid3d* cam2_from_cam1) {
  // 使用RANSAC估计本质矩阵
  RANSAC<EssentialMatrixFivePointEstimator> ransac(ransac_options);
  const auto report = ransac.Estimate(points1, points2);

  if (!report.success) {  // 如果估计失败，返回0
    return 0;
  }

  std::vector<Eigen::Vector2d> inliers1(report.support.num_inliers);
  std::vector<Eigen::Vector2d> inliers2(report.support.num_inliers);

  size_t j = 0;
  for (size_t i = 0; i < points1.size(); ++i) {
    if (report.inlier_mask[i]) {
      inliers1[j] = points1[i];  // 仅保留内点
      inliers2[j] = points2[i];  // 仅保留内点
      j += 1;
    }
  }

  std::vector<Eigen::Vector3d> points3D;
  // 从本质矩阵恢复相对姿态
  PoseFromEssentialMatrix(
      report.model, inliers1, inliers2, cam2_from_cam1, &points3D);

  if (cam2_from_cam1->rotation.coeffs().array().isNaN().any() ||
      cam2_from_cam1->translation.array().isNaN().any()) {  // 检查旋转和平移是否有效
    return 0;
  }

  return points3D.size();  // 返回恢复的3D点数量
}

// 优化绝对姿态，通过Ceres求解最小化重投影误差
bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Rigid3d* cam_from_world,
                        Camera* camera,
                        Eigen::Matrix6d* cam_from_world_cov) {
  THROW_CHECK_EQ(inlier_mask.size(), points2D.size());  // 确保内点掩码和2D点数量一致
  THROW_CHECK_EQ(points2D.size(), points3D.size());  // 确保2D点和3D点数量一致
  options.Check();  // 检查选项配置是否有效

  const auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);  // 使用Cauchy损失函数进行稳健优化

  double* camera_params = camera->params.data();
  double* cam_from_world_rotation = cam_from_world->rotation.coeffs().data();
  double* cam_from_world_translation = cam_from_world->translation.data();

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (size_t i = 0; i < points2D.size(); ++i) {
    // 跳过非内点
    if (!inlier_mask[i]) {
      continue;
    }
    problem.AddResidualBlock(
        CameraCostFunction<ReprojErrorConstantPoint3DCostFunctor>(
            camera->model_id, points2D[i], points3D[i]),
        loss_function.get(),
        cam_from_world_rotation,
        cam_from_world_translation,
        camera_params);  // 添加残差块
  }

  if (problem.NumResiduals() > 0) {
    SetQuaternionManifold(&problem, cam_from_world_rotation);  // 设置四元数的流形约束

    // 如果不需要优化焦距或其他参数，则将相机参数设置为常数
    if (!options.refine_focal_length && !options.refine_extra_params) {
      problem.SetParameterBlockConstant(camera->params.data());
    } else {
      // 设置哪些相机参数不优化
      std::vector<int> camera_params_const;
      const span<const size_t> principal_point_idxs =
          camera->PrincipalPointIdxs();
      camera_params_const.insert(camera_params_const.end(),
                                 principal_point_idxs.begin(),
                                 principal_point_idxs.end());

      if (!options.refine_focal_length) {
        const span<const size_t> focal_length_idxs = camera->FocalLengthIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   focal_length_idxs.begin(),
                                   focal_length_idxs.end());
      }

      if (!options.refine_extra_params) {
        const span<const size_t> extra_params_idxs = camera->ExtraParamsIdxs();
        camera_params_const.insert(camera_params_const.end(),
                                   extra_params_idxs.begin(),
                                   extra_params_idxs.end());
      }

      if (camera_params_const.size() == camera->params.size()) {
        problem.SetParameterBlockConstant(camera->params.data());
      } else {
        SetSubsetManifold(static_cast<int>(camera->params.size()),
                          camera_params_const,
                          &problem,
                          camera->params.data());
      }
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.gradient_tolerance = options.gradient_tolerance;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.logging_type = ceres::LoggingType::SILENT;

  // 使用单线程进行求解，因为线程创建的开销太大
  solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);  // 求解优化问题

  if (options.print_summary || VLOG_IS_ON(1)) {
    PrintSolverSummary(summary, "Pose refinement report");  // 打印求解报告
  }

  if (!summary.IsSolutionUsable()) {  // 检查是否找到有效的解
    return false;
  }

  if (problem.NumResiduals() > 0 && cam_from_world_cov != nullptr) {
    ceres::Covariance::Options options;
    ceres::Covariance covariance(options);
    std::vector<const double*> parameter_blocks = {cam_from_world_rotation,
                                                   cam_from_world_translation};
    if (!covariance.Compute(parameter_blocks, &problem)) {  // 计算协方差
      return false;
    }
    // 将协方差矩阵转换为切空间形式
    covariance.GetCovarianceMatrixInTangentSpace(parameter_blocks,
                                                 cam_from_world_cov->data());
  }

  return true;  // 返回优化成功
}

// 优化相对姿态
bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Rigid3d* cam2_from_cam1) {
  THROW_CHECK_EQ(points1.size(), points2.size());  // 确保点的数量一致

  // 成本函数假定单位四元数
  cam2_from_cam1->rotation.normalize();  // 确保旋转矩阵是单位四元数

  double* cam2_from_cam1_rotation = cam2_from_cam1->rotation.coeffs().data();
  double* cam2_from_cam1_translation = cam2_from_cam1->translation.data();

  const double kMaxL2Error = 1.0;  // 设置最大L2误差
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(kMaxL2Error);  // 使用Cauchy损失函数

  ceres::Problem problem;

  for (size_t i = 0; i < points1.size(); ++i) {
    ceres::CostFunction* cost_function =
        SampsonErrorCostFunctor::Create(points1[i], points2[i]);  // 计算样本点的误差
    problem.AddResidualBlock(cost_function,
                             loss_function,
                             cam2_from_cam1_rotation,
                             cam2_from_cam1_translation);  // 添加残差块
  }

  SetQuaternionManifold(&problem, cam2_from_cam1_rotation);  // 设置四元数流形约束
  SetSphereManifold<3>(&problem, cam2_from_cam1_translation);  // 设置平移流形约束

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);  // 求解优化问题

  return summary.IsSolutionUsable();  // 检查是否得到有效解
}

// 优化本质矩阵
bool RefineEssentialMatrix(const ceres::Solver::Options& options,
                           const std::vector<Eigen::Vector2d>& points1,
                           const std::vector<Eigen::Vector2d>& points2,
                           const std::vector<char>& inlier_mask,
                           Eigen::Matrix3d* E) {
  THROW_CHECK_EQ(points1.size(), points2.size());  // 确保点的数量一致
  THROW_CHECK_EQ(points1.size(), inlier_mask.size());  // 确保内点掩码和点的数量一致

  // 提取内点
  size_t num_inliers = 0;
  for (const auto inlier : inlier_mask) {
    if (inlier) {
      num_inliers += 1;  // 统计内点的数量
    }
  }

  std::vector<Eigen::Vector2d> inlier_points1(num_inliers);
  std::vector<Eigen::Vector2d> inlier_points2(num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_points1[j] = points1[i];  // 仅保留内点
      inlier_points2[j] = points2[i];  // 仅保留内点
      j += 1;
    }
  }

  // 从本质矩阵恢复相对姿态
  Rigid3d cam2_from_cam1;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(
      *E, inlier_points1, inlier_points2, &cam2_from_cam1, &points3D);

  if (points3D.size() == 0) {  // 如果没有恢复3D点，返回false
    return false;
  }

  // 使用内点对本质矩阵进行优化
  const bool refinement_success = RefineRelativePose(
      options, inlier_points1, inlier_points2, &cam2_from_cam1);

  if (!refinement_success) {  // 如果优化失败，返回false
    return false;
  }

  *E = EssentialMatrixFromPose(cam2_from_cam1);  // 使用优化后的姿态恢复本质矩阵

  return true;  // 返回优化成功
}

}  // namespace colmap
