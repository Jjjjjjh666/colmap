// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// 许可和使用条款说明：
// 本软件遵循以下条件，允许在源代码或二进制形式下进行分发和使用，无论是否修改：
// - 源代码的分发必须保留上述版权声明、条件列表和免责声明。
// - 二进制形式的分发必须在文档和/或随附材料中复制上述版权声明、条件列表和免责声明。
// - 不得使用ETH Zurich和UNC Chapel Hill的名称或其贡献者的名称来支持或推广基于该软件的产品，除非事先获得书面许可。

#include "colmap/estimators/homography_matrix.h"  // 引入同伦矩阵估算器的头文件
#include "colmap/util/eigen_alignment.h"         // 引入Eigen对齐工具的头文件
#include "colmap/util/logging.h"                 // 引入日志记录工具的头文件

#include <Eigen/Geometry>  // 引入Eigen库的几何模块
#include <Eigen/LU>        // 引入Eigen库的LU分解模块
#include <Eigen/SVD>       // 引入Eigen库的SVD模块

namespace colmap {  // 开始colmap命名空间

// 同伦矩阵估算器的实现类
void HomographyMatrixEstimator::Estimate(const std::vector<X_t>& points1,
                                         const std::vector<Y_t>& points2,
                                         std::vector<M_t>* models) {
  // 检查输入的点集大小是否一致
  THROW_CHECK_EQ(points1.size(), points2.size());
  // 至少需要4个点来估算同伦矩阵
  THROW_CHECK_GE(points1.size(), 4);
  // 检查models指针是否有效
  THROW_CHECK(models != nullptr);

  models->clear();  // 清空输出模型的容器

  const size_t num_points = points1.size();  // 获取点对的数量

  // 构建约束矩阵A，大小为(2 * num_points, 9)
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(2 * num_points, 9);
  for (size_t i = 0; i < num_points; ++i) {
    // 填充矩阵A的每一行，表示每对点的约束关系
    A.block<1, 3>(2 * i, 0) = points1[i].transpose().homogeneous();  // 第i个点在图像1中的坐标
    A.block<1, 3>(2 * i, 3).setZero();  // 填充零块
    A.block<1, 3>(2 * i, 6) =
        -points2[i].x() * points1[i].transpose().homogeneous();  // 图像2中的x坐标对应的约束
    A.block<1, 3>(2 * i + 1, 0).setZero();  // 填充零块
    A.block<1, 3>(2 * i + 1, 3) = points1[i].transpose().homogeneous();  // 第i个点在图像1中的坐标
    A.block<1, 3>(2 * i + 1, 6) =
        -points2[i].y() * points1[i].transpose().homogeneous();  // 图像2中的y坐标对应的约束
  }

  Eigen::Matrix3d H;  // 用于存储估算的同伦矩阵H

  // 如果点对为4个，使用线性最小二乘法直接求解
  if (num_points == 4) {
    const Eigen::Matrix<double, 9, 1> h = A.block<8, 8>(0, 0)
                                              .partialPivLu()
                                              .solve(-A.block<8, 1>(0, 8))
                                              .homogeneous();  // 求解Ax = b，得到同伦矩阵H的元素
    if (h.hasNaN()) {  // 如果求解结果包含NaN，返回
      return;
    }
    H = Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose();  // 转置得到3x3的同伦矩阵
  } else {
    // 使用SVD方法求解约束矩阵的零空间
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);  // 对矩阵A进行SVD分解
    if (svd.rank() < 8) {  // 如果矩阵的秩小于8，无法求解
      return;
    }
    const Eigen::VectorXd nullspace = svd.matrixV().col(8);  // 获取零空间的最后一列
    H = Eigen::Map<const Eigen::Matrix3d>(nullspace.data()).transpose();  // 转置得到同伦矩阵
  }

  // 如果同伦矩阵的行列式接近零，说明矩阵可能无效，返回
  if (std::abs(H.determinant()) < 1e-8) {
    return;
  }

  models->resize(1);  // 分配空间存储一个模型
  (*models)[0] = H;  // 将计算得到的同伦矩阵H存入模型容器
}

// 计算给定同伦矩阵下的残差
void HomographyMatrixEstimator::Residuals(const std::vector<X_t>& points1,
                                          const std::vector<Y_t>& points2,
                                          const M_t& H,
                                          std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());  // 检查输入点集大小是否一致

  residuals->resize(points1.size());  // 初始化残差容器

  // 提取同伦矩阵H的元素
  const double H_00 = H(0, 0);
  const double H_01 = H(0, 1);
  const double H_02 = H(0, 2);
  const double H_10 = H(1, 0);
  const double H_11 = H(1, 1);
  const double H_12 = H(1, 2);
  const double H_20 = H(2, 0);
  const double H_21 = H(2, 1);
  const double H_22 = H(2, 2);

  // 计算每个点对的残差
  for (size_t i = 0; i < points1.size(); ++i) {
    const double s_0 = points1 图像1中的x坐标
    const double s_1 = points1 ;  /的y坐标
    const double d_0 = points2 ;  // 图像2    const double d_1 = points2 ;  // 图像2中的y坐标// 使用同伦矩阵变换图像1中的点，得到图像2中的点
    const double pd_0 = H_00 * s_0 + H_01 * s_1 + H_02;
    const double pd_1 = H_10 * s_0 + H_11 * s_1 + H_12;
    const double pd_2 = H_20 * s_0 + H_21 * s_1 + H_22;

    const double inv_pd_2 = 1.0 / pd_2;  // 对pd_2取倒数
    const double dd_0 = d_0 - pd_0 * inv_pd_2;  // 计算x方向的残差
    const double dd_1 = d_1 - pd_1 * inv_pd_2;  // 计算y方向的残差

    (*residuals)[i] = dd_0 * dd_0 + dd_1 * dd_1;  // 计算平方的残差
  }
}

}  // 结束colmap命名空间
