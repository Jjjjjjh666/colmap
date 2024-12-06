// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// 许可和使用条款说明：
// 本软件遵循以下条件，允许在源代码或二进制形式下进行分发和使用，无论是否修改：
// - 源代码的分发必须保留上述版权声明、条件列表和免责声明。
// - 二进制形式的分发必须在文档和/或随附材料中复制上述版权声明、条件列表和免责声明。
// - 不得使用ETH Zurich和UNC Chapel Hill的名称或其贡献者的名称来支持或推广基于该软件的产品，除非事先获得书面许可。

#include "colmap/estimators/homography_matrix.h"  // 引入同伦矩阵估算器的头文件
#include "colmap/util/eigen_alignment.h"         // 引入Eigen对齐工具

#include <Eigen/Core>                            // 引入Eigen核心模块
#include <Eigen/Geometry>                        // 引入Eigen几何模块
#include <gtest/gtest.h>                         // 引入Google Test框架

namespace colmap {
namespace {

// 定义一个测试类，用于测试HomographyMatrixEstimator
class HomographyMatrixTests : public ::testing::TestWithParam<size_t> {};

// **Nominal测试**
// 验证HomographyMatrixEstimator在正常情况下的估计精度。
TEST_P(HomographyMatrixTests, Nominal) {
  const size_t kNumPoints = GetParam();  // 获取测试参数（点的数量）
  for (int x = 0; x < 10; ++x) {        // 进行10轮测试
    Eigen::Matrix3d expected_H;         // 定义期望的同伦矩阵
    expected_H << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;  // 源点集合
    std::vector<Eigen::Vector2d> dst;  // 目标点集合
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(Eigen::Vector2d::Random());  // 随机生成源点
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized());  // 根据同伦矩阵计算目标点
    }

    HomographyMatrixEstimator estimator;  // 创建同伦矩阵估算器
    std::vector<Eigen::Matrix3d> models;  // 存储估算的同伦矩阵
    estimator.Estimate(src, dst, &models);  // 估算同伦矩阵

    ASSERT_EQ(models.size(), 1);  // 确保估算结果仅有一个模型

    std::vector<double> residuals;  // 存储残差
    estimator.Residuals(src, dst, models[0], &residuals);  // 计算残差

    // 检查所有点的残差是否在允许范围内
    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

// **数值稳定性测试**
// 验证HomographyMatrixEstimator在坐标较大的情况下的数值稳定性。
TEST_P(HomographyMatrixTests, NumericalStability) {
  const size_t kNumPoints = GetParam();          // 获取测试参数
  constexpr double kCoordinateScale = 1e6;      // 坐标放大倍数
  for (int x = 1; x < 10; ++x) {                // 进行9轮测试
    Eigen::Matrix3d expected_H = Eigen::Matrix3d::Identity();  // 单位矩阵
    expected_H(0, 0) = x;                       // 修改矩阵元素

    std::vector<Eigen::Vector2d> src;  // 源点集合
    std::vector<Eigen::Vector2d> dst;  // 目标点集合
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(Eigen::Vector2d::Random() * kCoordinateScale);  // 生成大坐标的源点
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized());  // 计算目标点
    }

    HomographyMatrixEstimator estimator;  // 创建同伦矩阵估算器
    std::vector<Eigen::Matrix3d> models;  // 存储估算的同伦矩阵
    estimator.Estimate(src, dst, &models);
    ASSERT_EQ(models.size(), 1);  // 确保成功估算一个模型

    std::vector<double> residuals;  // 存储残差
    estimator.Residuals(src, dst, models[0], &residuals);  // 计算残差

    // 检查所有点的残差是否在允许范围内
    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

// **噪声稳定性测试**
// 验证HomographyMatrixEstimator在添加噪声的情况下的表现。
TEST_P(HomographyMatrixTests, NoiseStability) {
  const size_t kNumPoints = GetParam();    // 获取测试参数
  constexpr double kNoise = 1e-3;         // 噪声幅度
  for (int x = 1; x < 10; ++x) {          // 进行9轮测试
    Eigen::Matrix3d expected_H = Eigen::Matrix3d::Identity();  // 单位矩阵
    expected_H(0, 0) = x;

    std::vector<Eigen::Vector2d> src;  // 源点集合
    std::vector<Eigen::Vector2d> dst;  // 目标点集合
    for (size_t i = 0; i < kNumPoints; ++i) {
      src.push_back(Eigen::Vector2d::Random());  // 生成随机源点
      dst.push_back((expected_H * src[i].homogeneous()).hnormalized() +
                    Eigen::Vector2d::Random() * kNoise);  // 添加噪声后计算目标点
    }

    HomographyMatrixEstimator estimator;  // 创建同伦矩阵估算器
    std::vector<Eigen::Matrix3d> models;  // 存储估算的同伦矩阵
    estimator.Estimate(src, dst, &models);
    ASSERT_EQ(models.size(), 1);  // 确保成功估算一个模型

    std::vector<double> residuals;  // 存储残差
    estimator.Residuals(src, dst, models[0], &residuals);  // 计算残差

    // 检查所有点的残差是否在允许范围内
    for (size_t i = 0; i < kNumPoints; ++i) {
      EXPECT_LT(residuals[i], 1e-5);
    }
  }
}

// **奇异情况测试**
// 验证HomographyMatrixEstimator在冗余点或退化情况下的行为。
TEST_P(HomographyMatrixTests, Degenerate) {
  const size_t kNumPoints = GetParam();  // 获取测试参数
  constexpr double kNoise = 1e-3;       // 噪声幅度

  for (int x = 0; x < 10; ++x) {        // 进行10轮测试
    Eigen::Matrix3d expected_H;         // 期望的同伦矩阵
    expected_H << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;  // 源点集合
    src.emplace_back(2, 1);            // 手动添加点
    src.emplace_back(3, 1);
    src.emplace_back(10, 30);
    ASSERT_GE(kNumPoints, 4);          // 确保点数大于等于4
    const size_t num_redundant_points = kNumPoints - src.size();  // 计算冗余点数量
    for (size_t i = 0; i < num_redundant_points; ++i) {
      src.emplace_back(src.front());  // 添加冗余点
    }

    std::vector<Eigen::Vector2d> dst;  // 目标点集合
    for (size_t i = 0; i < src.size(); ++i) {
      const Eigen::Vector3d dsth = expected_H * src[i].homogeneous();
      dst.push_back(dsth.hnormalized() + Eigen::Vector2d::Random() * kNoise);  // 添加噪声
    }

    HomographyMatrixEstimator estimator;  // 创建同伦矩阵估算器
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(src, dst, &models
// 实例化测试用例，传入不同的点数进行测试
INSTANTIATE_TEST_SUITE_P(HomographyMatrix,
                         HomographyMatrixTests,
                         ::testing::Values(4, 8, 64, 1024));

}  // namespace
}  // namespace colmap
