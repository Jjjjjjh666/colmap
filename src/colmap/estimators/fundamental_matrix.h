//用于估计基础矩阵
//基础矩阵在计算机视觉中用于描述两个图像之间的几何关系(特别是在立体视觉和多视图几何中)


#pragma once

#include "colmap/estimators/homography_matrix.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// 从对应点对估计基础矩阵的估计器。
//
// 该算法解决了7点问题，基于以下论文：
//    Zhengyou Zhang 和 T. Kanade，确定极线几何及其不确定性：综述，国际计算机视觉杂志，1998。
//    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.4540

//  七点算法用于从七对图像点中估计基础矩阵
//  七对图像点指的是在两幅图像中对应的七组点,通常通过匹配算法(sift之类的)得到
class FundamentalMatrixSevenPointEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix3d M_t;

  // 估计模型所需的最小样本数。
  static const int kMinNumSamples = 7;

  // 从一组对应点估计1或3个可能的基础矩阵解。
  //
  // 对应点的数量必须正好为7。
  //
  // @param points1  第一组对应点。
  // @param points2  第二组对应点。
  //
  // @return         最多4个解，作为3x3基础矩阵的向量。
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // 计算一组对应点和给定基础矩阵的残差。
  //
  // 残差定义为平方的Sampson误差。
  //
  // @param points1    第一组对应点，作为Nx2矩阵。
  // @param points2    第二组对应点，作为Nx2矩阵。
  // @param F          3x3基础矩阵。
  // @param residuals  输出残差向量。
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& F,
                        std::vector<double>* residuals);
};



// 从对应点对估计基础矩阵的估计器。
//
// 该算法解决了8点问题，基于以下论文：
// Hartley 和 Zisserman，多视图几何，算法11.1，第282页。


//八点算法用于从至少八对图像点中估计基础矩阵
class FundamentalMatrixEightPointEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix3d M_t;

  // 估计模型所需的最小样本数。
  static const int kMinNumSamples = 8;

  // 从一组对应点估计基础矩阵解。
  //
  // 对应点的数量必须至少为8。
  //
  // @param points1  第一组对应点。
  // @param points2  第二组对应点。
  //
  // @return         单个解，作为3x3基础矩阵的向量。
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // 计算一组对应点和给定基础矩阵的残差。
  //
  // 残差定义为平方的Sampson误差。
  //
  // @param points1    第一组对应点，作为Nx2矩阵。
  // @param points2    第二组对应点，作为Nx2矩阵。
  // @param F          3x3基础矩阵。
  // @param residuals  输出残差向量。
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& F,
                        std::vector<double>* residuals);
};

}  // namespace colmap
