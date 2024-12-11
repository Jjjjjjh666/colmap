//这个文件定义了两个类，用于估计本质矩阵（Essential Matrix），这是计算相机之间相对姿态的关键步骤
//本质矩阵用于描述两个摄像机之间的相对旋转和平移
//该文件提供了两种本质矩阵问题的解决方法：5点法和8点法
#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>
#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// 从对应的标准化点对中估计本质矩阵。
//
// 该算法解决了5点问题，基于以下论文：
//
//    D. Nister, An efficient solution to the five-point relative pose problem,
//    IEEE-T-PAMI, 26(6), 2004.
//    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.86.8769
class EssentialMatrixFivePointEstimator {
 public:
  typedef Eigen::Vector2d X_t;  // 点类型定义
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix3d M_t;  // 矩阵类型定义

  // 估计模型所需的最小样本数。
  static const int kMinNumSamples = 5;

  // 从一组对应点中估计最多10个可能的本质矩阵解。
  //
  // 对应点的数量必须至少为5。
  //
  // @param points1  第一组对应点。
  // @param points2  第二组对应点。
  //
  // @return         最多10个3x3本质矩阵的解。
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // 计算一组对应点和给定本质矩阵的残差。
  //
  // 残差定义为平方的Sampson误差。
  //
  // @param points1    第一组对应点。
  // @param points2    第二组对应点。
  // @param E          3x3本质矩阵。
  // @param residuals  输出的残差向量。
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& E,
                        std::vector<double>* residuals);
};

// 从对应的标准化点对中估计本质矩阵。
//
// 该算法解决了8点问题，基于以下论文：
//
//    Hartley and Zisserman, Multiple View Geometry, algorithm 11.1, page 282.
class EssentialMatrixEightPointEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix3d M_t;

  // 估计模型所需的最小样本数。
  static const int kMinNumSamples = 8;

  // 从一组对应点中估计本质矩阵解。
  //
  // 对应点的数量必须至少为8。
  //
  // @param points1  第一组对应点。
  // @param points2  第二组对应点。
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // 计算一组对应点和给定本质矩阵的残差。
  //
  // 残差定义为平方的Sampson误差。
  //
  // @param points1    第一组对应点。
  // @param points2    第二组对应点。
  // @param E          3x3本质矩阵。
  // @param residuals  输出的残差向量。
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& E,
                        std::vector<double>* residuals);
};

}  // namespace colmap
