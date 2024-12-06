// 版权所有 (c) 2023, 苏黎世联邦理工大学 (ETH Zurich) 和北卡罗来纳大学教堂山分校 (UNC Chapel Hill)
// 保留所有权利。
//
// 允许以源代码和二进制形式进行再分发和使用，无论是否经过修改，
// 但必须满足以下条件：
//
//     * 源代码的再分发必须保留上述版权声明、此条件列表以及以下免责声明。
//
//     * 二进制形式的再分发必须在分发的文档和/或其他材料中，复制上述版权声明、此条件列表及以下免责声明。
//
//     * 不得使用苏黎世联邦理工大学、北卡罗来纳大学教堂山分校的名称或其贡献者的名称来为本软件衍生的产品进行背书或宣传，
//       除非事先获得特定的书面许可。
//
// 本软件由版权持有者和贡献者提供，按“原样”提供，
// 不提供任何明示或暗示的保证，包括但不限于对适销性和特定用途适用性的暗示保证。
// 在任何情况下，版权持有者或贡献者都不对因使用本软件而产生的任何直接、间接、附带、特殊、典型或间接损害负责，
// 无论是合同责任、严格责任还是侵权（包括过失或其他原因），
// 即使在被告知可能发生此类损害的情况下。

#pragma once  // 防止头文件被多次包含

// 引入外部库和自定义模块
#include "colmap/geometry/rigid3.h"  // 引入刚性变换 (rigid transformation) 的定义
#include "colmap/util/eigen_alignment.h"  // 引入Eigen库的对齐支持
#include "colmap/util/types.h"  // 引入基本数据类型的定义

#include <vector>  // 引入C++标准库中的向量容器
#include <Eigen/Core>  // 引入Eigen库的核心功能

namespace colmap {

// GP3P问题（广义P3P问题）的求解器类（基于Lee等人的论文）
class GP3PEstimator {
 public:
  // 描述广义相机系统中相机相对位姿和射线的结构体
  struct X_t {
    Rigid3d cam_from_rig;  // 表示相机相对于相机组的刚性变换
    Eigen::Vector3d ray_in_cam;  // 表示在相机坐标系中的射线
  };

  // 3D特征点在世界坐标系中的表示
  typedef Eigen::Vector3d Y_t;
  // 表示估计出的广义相机在世界坐标系中的位姿
  typedef Rigid3d M_t;

  // 估计模型所需的最小样本数量
  static const int kMinNumSamples = 3;

  // 定义残差类型的枚举类
  enum class ResidualType {
    CosineDistance,  // 余弦距离
    ReprojectionError,  // 重投影误差
  };

  // 表示是否计算余弦相似度或重投影误差
  // [警告] 由于重投影误差是在归一化坐标中进行计算的，RANSAC的唯一误差阈值在不同相机中可能对应不同的像素值，
  // 如果相机具有不同的内参，则需要注意这一点。
  ResidualType residual_type = ResidualType::CosineDistance;

  // 从一组三维到二维的点对应关系中估计广义P3P问题的最可能解
  static void Estimate(const std::vector<X_t>& points2D,  // 2D点集
                       const std::vector<Y_t>& points3D,  // 3D点集
                       std::vector<M_t>* models);  // 存储估计出的模型

  // 计算给定一组三维到二维点的对应关系和广义相机位姿下，射线之间的平方余弦距离误差
  void Residuals(const std::vector<X_t>& points2D,  // 2D点集
                 const std::vector<Y_t>& points3D,  // 3D点集
                 const M_t& rig_from_world,  // 广义相机在世界坐标系中的位姿
                 std::vector<double>* residuals);  // 存储计算出的残差
};

}  // namespace colmap
