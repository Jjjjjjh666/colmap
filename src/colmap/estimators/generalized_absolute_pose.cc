// 版权所有 (c) 2023, ETH Zurich 和 UNC Chapel Hill。
// 保留所有权利。
// 
// 允许在以下条件下进行源代码和二进制形式的重新分发和使用，无论是否修改：
// 
//     * 重新分发的源代码必须保留上述版权声明、此条件列表以及以下免责声明。
// 
//     * 以二进制形式重新分发必须在分发的文档和/或其他材料中复制上述版权声明、此条件列表和免责声明。
// 
//     * 未经特定书面许可，ETH Zurich 和 UNC Chapel Hill 的名称及其贡献者的名称不得用于支持或推广基于此软件的产品。
// 
// 本软件按 "原样" 提供，版权持有者和贡献者不对任何明示或暗示的保证负责，包括但不限于对适销性和特定用途适用性的隐含保证。
// 在任何情况下，版权持有者或贡献者均不对任何因使用本软件而产生的直接、间接、附带、特殊、惩罚性或后果性损害负责，
// 包括但不限于采购替代商品或服务、使用损失、数据丢失、利润损失或业务中断，无论是基于合同、严格责任或侵权（包括过失或其他）
// 理论产生的，甚至在已被告知可能发生此类损害的情况下。

#include "colmap/estimators/generalized_absolute_pose.h"

#include "colmap/estimators/generalized_absolute_pose_coeffs.h"
#include "colmap/math/polynomial.h"
#include "colmap/util/logging.h"

#include <array>

namespace colmap {
namespace {

// 检查三条射线是否接近平行
bool CheckParallelRays(const Eigen::Vector3d& ray1,
                       const Eigen::Vector3d& ray2,
                       const Eigen::Vector3d& ray3) {
  const double kParallelThreshold = 1e-5;  // 定义射线平行的阈值
  // 通过叉乘判断射线是否平行
  return ray1.cross(ray2).isApproxToConstant(0, kParallelThreshold) &&
         ray1.cross(ray3).isApproxToConstant(0, kParallelThreshold);
}

// 检查三点是否接近共线
bool CheckCollinearPoints(const Eigen::Vector3d& X1,
                          const Eigen::Vector3d& X2,
                          const Eigen::Vector3d& X3) {
  const double kMinNonCollinearity = 1e-5;  // 定义非共线性的最小阈值
  const Eigen::Vector3d X12 = X2 - X1;
  const double non_collinearity_measure =
      X12.cross(X1 - X3).squaredNorm() / X12.squaredNorm();  // 计算非共线性度量
  return non_collinearity_measure < kMinNonCollinearity;
}

// 将刚体变换和射线转换为Plücker坐标系下的6维表示
Eigen::Vector6d ComposePlueckerLine(const Rigid3d& rig_from_cam,
                                    const Eigen::Vector3d& ray_in_cam) {
  const Eigen::Vector3d ray_in_rig =
      (rig_from_cam.rotation * ray_in_cam).normalized();  // 将射线从相机坐标系转换为刚体坐标系
  Eigen::Vector6d pluecker;
  pluecker << ray_in_rig, rig_from_cam.translation.cross(ray_in_rig);  // 计算Plücker坐标
  return pluecker;
}

// 根据Plücker坐标和深度计算三维点
Eigen::Vector3d PointFromPlueckerLineAndDepth(const Eigen::Vector6d& pluecker,
                                              const double depth) {
  return pluecker.head<3>().cross(pluecker.tail<3>()) +
         depth * pluecker.head<3>();  // 使用Plücker坐标计算三维点
}

// 通过求解三组非线性方程计算多项式系数。输入为三条Plücker线和对应的三维点的位置。
// 方程来源于三维点之间的距离约束。
Eigen::Matrix<double, 3, 6> ComputePolynomialCoefficients(
    const std::vector<Eigen::Vector6d>& plueckers,
    const std::vector<Eigen::Vector3d>& points3D) {
  THROW_CHECK_EQ(plueckers.size(), 3);  // 检查Plücker线数量
  THROW_CHECK_EQ(points3D.size(), 3);  // 检查三维点数量

  Eigen::Matrix<double, 3, 6> K;
  const std::array<int, 3> is = {{0, 0, 1}};  // 选择方程的索引
  const std::array<int, 3> js = {{1, 2, 2}};  // 选择方程的索引

  for (int k = 0; k < 3; ++k) {
    const int i = is[k];
    const int j = js[k];
    const Eigen::Vector3d moment_difference =
        plueckers[i].head<3>().cross(plueckers[i].tail<3>()) -
        plueckers[j].head<3>().cross(plueckers[j].tail<3>());
    K(k, 0) = 1;
    K(k, 1) = -2 * plueckers[i].head<3>().dot(plueckers[j].head<3>());
    K(k, 2) = 2 * moment_difference.dot(plueckers[i].head<3>());
    K(k, 3) = 1;
    K(k, 4) = -2 * moment_difference.dot(plueckers[j].head<3>());
    K(k, 5) = moment_difference.squaredNorm() -
              (points3D[i] - points3D[j]).squaredNorm();
  }

  return K;
}

// 求解二次方程 x^2 + bx + c = 0
int SolveQuadratic(const double b, const double c, double* roots) {
  const double delta = b * b - 4 * c;  // 计算判别式
  if (delta >= 0) {
    const double sqrt_delta = std::sqrt(delta);  // 如果有实根，计算平方根
    roots[0] = -0.5 * (b + sqrt_delta);
    roots[1] = -0.5 * (b - sqrt_delta);
    return 2;  // 返回根的数量
  } else {
    return 0;  // 无实根
  }
}

// 根据给定的lambda_j，计算lambda_i的值
void ComputeLambdaValues(const Eigen::Matrix<double, 3, 6>::ConstRowXpr& k,
                         const double lambda_j,
                         std::vector<double>* lambdas_i) {
  // 求解方程 x^2 + bx + c = 0
  double roots[2];
  const int num_solutions =
      SolveQuadratic(k(1) * lambda_j + k(2),
                     lambda_j * (k(3) * lambda_j + k(4)) + k(5),
                     roots);
  for (int i = 0; i < num_solutions; ++i) {
    if (roots[i] > 0) {
      lambdas_i->push_back(roots[i]);  // 仅保留正的解
    }
  }
}
// 给定多项式系统的系数，返回沿Pluecker直线的深度。
// 使用Sylvester结果法得到8次多项式的lambda_3，并在原始方程中进行回代。
std::vector<Eigen::Vector3d> ComputeDepthsSylvester(
    const Eigen::Matrix<double, 3, 6>& K) {
  
  // 计算深度的Sylvester系数
  const Eigen::Matrix<double, 9, 1> coeffs = ComputeDepthsSylvesterCoeffs(K);

  Eigen::VectorXd roots_real;  // 存储实数根
  Eigen::VectorXd roots_imag;  // 存储虚数根
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return std::vector<Eigen::Vector3d>();  // 找不到根时返回空结果
  }

  // 对每个lambda_3根进行回代
  std::vector<Eigen::Vector3d> depths;  // 存储深度结果
  depths.reserve(roots_real.size());  // 预分配空间
  for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
    const double kMaxRootImagRatio = 1e-3;
    
    // 如果虚部较大，则跳过该根
    if (std::abs(roots_imag(i)) > kMaxRootImagRatio * std::abs(roots_real(i))) {
      continue;
    }

    const double lambda_3 = roots_real(i);
    // 如果lambda_3小于等于0，跳过
    if (lambda_3 <= 0) {
      continue;
    }

    // 计算lambda_2的值
    std::vector<double> lambdas_2;
    ComputeLambdaValues(K.row(2), lambda_3, &lambdas_2);

    // 对于每个lambda_2，计算lambda_1值，并验证其一致性
    for (const double lambda_2 : lambdas_2) {
      std::vector<double> lambdas_1_1;
      ComputeLambdaValues(K.row(0), lambda_2, &lambdas_1_1);
      std::vector<double> lambdas_1_2;
      ComputeLambdaValues(K.row(1), lambda_3, &lambdas_1_2);

      // 遍历每个lambda_1_1和lambda_1_2，确保它们的值一致
      for (const double lambda_1_1 : lambdas_1_1) {
        for (const double lambda_1_2 : lambdas_1_2) {
          const double kMaxLambdaRatio = 1e-2;
          
          // 如果两个lambda_1值的差异较小，则视为一致
          if (std::abs(lambda_1_1 - lambda_1_2) <
              kMaxLambdaRatio * std::max(lambda_1_1, lambda_1_2)) {
            const double lambda_1 = (lambda_1_1 + lambda_1_2) / 2;
            depths.emplace_back(lambda_1, lambda_2, lambda_3);  // 保存深度结果
          }
        }
      }
    }
  }

  return depths;  // 返回所有有效的深度值
}

}  // namespace

void GP3PEstimator::Estimate(const std::vector<X_t>& points2D,
                             const std::vector<Y_t>& points3D,
                             std::vector<M_t>* models) {
  THROW_CHECK_EQ(points2D.size(), 3);  // 检查2D点数为3
  THROW_CHECK_EQ(points3D.size(), 3);  // 检查3D点数为3
  THROW_CHECK(models != nullptr);  // 检查模型指针非空

  models->clear();  // 清空模型

  // 如果3D点共线，返回
  if (CheckCollinearPoints(points3D[0], points3D[1], points3D[2])) {
    return;
  }

  // 将2D点转换为Pluecker线的紧凑表示
  std::vector<Eigen::Vector6d> plueckers(3);
  for (size_t i = 0; i < 3; ++i) {
    plueckers[i] = ComposePlueckerLine(Inverse(points2D[i].cam_from_rig),
                                       points2D[i].ray_in_cam);
  }

  // 如果光线平行，返回
  if (CheckParallelRays(plueckers[0].head<3>(),
                        plueckers[1].head<3>(),
                        plueckers[2].head<3>())) {
    return;
  }

  // 计算系数k1, k2, k3（根据公式4）
  const Eigen::Matrix<double, 3, 6> K =
      ComputePolynomialCoefficients(plueckers, points3D);

  // 计算沿Pluecker直线的深度
  const std::vector<Eigen::Vector3d> depths = ComputeDepthsSylvester(K);
  if (depths.empty()) {
    return;
  }

  // 对所有有效的深度值，计算从摄像机到世界坐标系的变换
  Eigen::Matrix3d points3D_in_world;
  for (size_t i = 0; i < 3; ++i) {
    points3D_in_world.col(i) = points3D[i];
  }

  models->resize(depths.size());
  for (size_t i = 0; i < depths.size(); ++i) {
    Eigen::Matrix3d points3D_in_rig;
    for (size_t j = 0; j < 3; ++j) {
      points3D_in_rig.col(j) =
          PointFromPlueckerLineAndDepth(plueckers[j], depths[i][j]);
    }

    // 使用Umeyama方法计算从世界坐标系到rig坐标系的变换矩阵
    const Eigen::Matrix4d rig_from_world =
        Eigen::umeyama(points3D_in_world, points3D_in_rig, false);
    (*models)[i] =
        Rigid3d(Eigen::Quaterniond(rig_from_world.topLeftCorner<3, 3>()),
                rig_from_world.topRightCorner<3, 1>());
  }
}

void GP3PEstimator::Residuals(const std::vector<X_t>& points2D,
                              const std::vector<Y_t>& points3D,
                              const M_t& rig_from_world,
                              std::vector<double>* residuals) {
  THROW_CHECK_EQ(points2D.size(), points3D.size());  // 确保2D和3D点数相等
  residuals->resize(points2D.size(), 0);  // 初始化残差

  for (size_t i = 0; i < points2D.size(); ++i) {
    // 将3D点转换到摄像机坐标系
    const Eigen::Vector3d point3D_in_cam =
        points2D[i].cam_from_rig * (rig_from_world * points3D[i]);
    
    // 检查3D点是否在摄像机前面
    if (point3D_in_cam.z() > std::numeric_limits<double>::epsilon()) {
      // 根据不同的残差类型计算残差
      if (residual_type == ResidualType::CosineDistance) {
        const double cosine_dist =
            1 - point3D_in_cam.normalized().dot(points2D[i].ray_in_cam);
        (*residuals)[i] = cosine_dist * cosine_dist;
      } else if (residual_type == ResidualType::ReprojectionError) {
        (*residuals)[i] = (point3D_in_cam.hnormalized() -
                           points2D[i].ray_in_cam.hnormalized())
                              .squaredNorm();
      } else {
        LOG(FATAL_THROW) << "无效的残差类型";  // 错误处理
      }
    } else {
      (*residuals)[i] = std::numeric_limits<double>::max();  // 点在摄像机后面，设为最大残差
    }
  }
}
