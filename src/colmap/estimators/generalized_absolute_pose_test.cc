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

#include "colmap/estimators/generalized_absolute_pose.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"

#include <array>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// 测试GeneralizedAbsolutePose（广义绝对姿态）估计功能
TEST(GeneralizedAbsolutePose, Estimate) {
  // 定义一个包含3D点的向量
  std::vector<Eigen::Vector3d> points3D;
  points3D.emplace_back(1, 1, 1); // 添加第一个点
  points3D.emplace_back(0, 1, 1); // 添加第二个点
  points3D.emplace_back(3, 1.0, 4); // 添加第三个点
  points3D.emplace_back(3, 1.1, 4); // 添加第四个点
  points3D.emplace_back(3, 1.2, 4); // 添加第五个点
  points3D.emplace_back(3, 1.3, 4); // 添加第六个点
  points3D.emplace_back(3, 1.4, 4); // 添加第七个点
  points3D.emplace_back(2, 1, 7); // 添加第八个点

  // 创建一个错误的3D点集合，用于测试
  auto points3D_faulty = points3D;
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D_faulty; // 让这些点的x坐标发生变化，模拟错误
  }

  // 遍历不同的相机旋转角度（qx）和平移量（tx），测试不同情况
  for (double qx = 0; qx < 1; qx += 0.2) { // 循环遍历不同的qx值
    for (double tx = 0; tx < 1; tx += 0.1) { // 循环遍历不同的tx值
      const int kRefCamIdx = 1; // 设置参考相机的索引
      const int kNumCams = 3; // 相机数量

      // 创建3个不同的相机位置和方向（rigid transformations）
      const std::array<Rigid3d, kNumCams> cams_from_world = {{
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, -0.1, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0, 0)),
          Rigid3d(Eigen::Quaterniond(1, qx, 0, 0).normalized(),
                  Eigen::Vector3d(tx, 0.1, 0)),
      }};

      // 选择参考相机（rig_from_world）
      const Rigid3d& rig_from_world = cams_from_world[kRefCamIdx];

      // 计算相对于参考相机的变换矩阵
      std::array<Rigid3d, kNumCams> cams_from_rig;
      for (size_t i = 0; i < kNumCams; ++i) {
        cams_from_rig[i] = cams_from_world[i] * Inverse(rig_from_world);
      }

      // 将3D点投影到相机坐标系中
      std::vector<GP3PEstimator::X_t> points2D;
      for (size_t i = 0; i < points3D.size(); ++i) {
        points2D.emplace_back();
        points2D.back().cam_from_rig = cams_from_rig[i % kNumCams]; // 每个点的相机坐标
        points2D.back().ray_in_cam =
            (cams_from_world[i % kNumCams] * points3D[i]).normalized(); // 通过相机坐标变换得到射线方向
      }

      // 设置RANSAC（随机样本一致性）算法的参数
      RANSACOptions options;
      options.max_error = 1e-5; // 设置最大容忍误差为1e-5
      RANSAC<GP3PEstimator> ransac(options);

      // 使用RANSAC估计广义绝对姿态
      const auto report = ransac.Estimate(points2D, points3D);

      // 验证RANSAC是否成功估计到正确的姿态
      EXPECT_TRUE(report.success);
      EXPECT_LT((rig_from_world.ToMatrix() - report.model.ToMatrix()).norm(),
                1e-2)
          << report.model.ToMatrix() << "\n\n"
          << rig_from_world.ToMatrix();

      // 测试精确点的残差
      std::vector<double> residuals;
      ransac.estimator.Residuals(points2D, points3D, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LT(residuals[i], 1e-10); // 残差应非常小
      }

      // 测试错误点的残差
      ransac.estimator.Residuals(
          points2D, points3D_faulty, report.model, &residuals);
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_GT(residuals[i], 1e-10); // 错误点的残差应较大
      }
    }
  }
}

}  // namespace
}  // namespace colmap
