// 版权所有 (c) 2023，ETH Zurich 和 UNC Chapel Hill。
// 保留所有权利。
// 
// 以源代码或二进制形式重新分发和使用，或不使用修改，均可，前提是符合以下条件：
// 
//     * 源代码的重新分发必须保留上述版权声明、此条件列表和以下免责声明。
// 
//     * 二进制形式的重新分发必须在文档和/或随分发提供的其他材料中重复上述版权声明、此条件列表和以下免责声明。
// 
//     * 未经明确书面许可，不得使用ETH Zurich 和 UNC Chapel Hill的名称或其贡献者的名称来支持或推广基于本软件的产品。
// 
// 本软件由版权持有者和贡献者 "按原样" 提供，并且不附带任何明示或暗示的担保，包括但不限于适销性和特定用途适用性的暗示担保。
// 在任何情况下，版权持有者或贡献者都不对因使用本软件引起的任何直接、间接、附带、特殊、典型或后果性损害负责（包括但不限于采购替代商品或服务；使用、数据或利润的丧失；或业务中断），无论基于何种责任理论，无论是在合同、严格责任还是侵权（包括过失或其他）下，甚至在已被告知可能发生此类损害的情况下。
// 
// 版权持有者和贡献者不对本软件的使用承担任何责任。

#include "colmap/estimators/generalized_relative_pose.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/loransac.h"

#include <array>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// 测试GeneralizedRelativePose的Estimate函数
TEST(GeneralizedRelativePose, Estimate) {
  const size_t kNumPoints = 100;  // 使用100个3D点进行测试

  // 随机生成100个3D点
  std::vector<Eigen::Vector3d> points3D;
  for (size_t i = 0; i < kNumPoints; ++i) {
    points3D.emplace_back(Eigen::Vector3d::Random());
  }

  // 遍历qx和tx的不同值进行测试
  for (double qx = 0; qx < 0.4; qx += 0.1) {
    for (double tx = 0; tx < 0.5; tx += 0.1) {
      const int kRefCamIdx = 1;  // 参考摄像机索引
      const int kNumCams = 3;  // 摄像机数量

      // 定义三个摄像机从世界坐标系到摄像机坐标系的变换（旋转+平移）
      const std::array<Rigid3d, kNumCams> cams_from_world = {{
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.1, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx + 0.05, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.2, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx + 0.1, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.3, 0)),
      }};

      // 计算从rig坐标系到摄像机坐标系的变换
      std::array<Rigid3d, kNumCams> cams_from_rig;
      for (size_t i = 0; i < kNumCams; ++i) {
        cams_from_rig[i] =
            cams_from_world[i] * Inverse(cams_from_world[kRefCamIdx]);
      }

      // 将3D点投影到摄像机
      std::vector<GR6PEstimator::X_t> points1;
      std::vector<GR6PEstimator::Y_t> points2;
      for (size_t i = 0; i < points3D.size(); ++i) {
        const Eigen::Vector3d point3D_camera1 =
            cams_from_rig[i % kNumCams] * points3D[i];
        const Eigen::Vector3d point3D_camera2 =
            cams_from_world[(i + 1) % kNumCams] * points3D[i];
        
        // 如果投影的3D点在摄像机后面，则跳过
        if (point3D_camera1.z() < 0 || point3D_camera2.z() < 0) {
          continue;
        }

        // 将有效的投影点添加到点集
        points1.emplace_back();
        points1.back().cam_from_rig = cams_from_rig[i % kNumCams];
        points1.back().ray_in_cam = point3D_camera1.normalized();

        points2.emplace_back();
        points2.back().cam_from_rig = cams_from_rig[(i + 1) % kNumCams];
        points2.back().ray_in_cam = point3D_camera2.normalized();
      }

      // 配置RANSAC参数
      RANSACOptions options;
      options.max_error = 1e-3;  // 最大容许误差
      LORANSAC<GR6PEstimator, GR6PEstimator> ransac(options);
      
      // 使用RANSAC估计模型
      const auto report = ransac.Estimate(points1, points2);

      EXPECT_TRUE(report.success);  // 确保估计成功
      // 确保估计的模型与参考摄像机的变换足够接近
      EXPECT_LT(
          (cams_from_world[kRefCamIdx].ToMatrix() - report.model.ToMatrix())
              .norm(),
          1e-2);

      // 计算残差并验证其是否符合要求
      std::vector<double> residuals;
      GR6PEstimator::Residuals(points1, points2, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LE(residuals[i], options.max_error);  // 残差应该小于最大误差
      }
    }
  }
}

}  // namespace
}  // namespace colmap
