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

#pragma once

// 包含相关头文件，Eigen 是用于数学运算的库
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>
#include <Eigen/Core>

namespace colmap {

// 直接线性变换（DLT）算法用于计算单应性矩阵
// 给定点对，算法计算从至少4个对应点对中得到的最小二乘估计的单应性矩阵
class HomographyMatrixEstimator {
 public:
  typedef Eigen::Vector2d X_t;  // 定义二维点类型X_t，代表第一个点集中的点
  typedef Eigen::Vector2d Y_t;  // 定义二维点类型Y_t，代表第二个点集中的点
  typedef Eigen::Matrix3d M_t;  // 定义矩阵类型M_t，代表3x3的单应性矩阵

  // 估计模型所需的最小样本数
  static const int kMinNumSamples = 4;  // 至少需要4个点来估计单应性矩阵

  // 估计单应性矩阵（投影变换）
  //
  // 对应点的数量必须至少为4个。
  //
  // @param points1    第一组对应点。
  // @param points2    第二组对应点。
  //
  // @return         3x3的单应性变换矩阵。
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // 计算每对对应点的变换误差
  //
  // 残差定义为将源点变换到目标点的平方误差。
  //
  // @param points1    第一组对应点。
  // @param points2    第二组对应点。
  // @param H          3x3的投影变换矩阵。
  // @param residuals  输出的残差向量。
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& H,
                        std::vector<double>* residuals);
};

}  // namespace colmap
