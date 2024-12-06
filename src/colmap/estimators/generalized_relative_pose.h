// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// 许可和使用条款说明：
// 本软件遵循以下条件，允许在源代码或二进制形式下进行分发和使用，无论是否修改：
// - 源代码的分发必须保留上述版权声明、条件列表和免责声明。
// - 二进制形式的分发必须在文档和/或随附材料中复制上述版权声明、条件列表和免责声明。
// - 不得使用ETH Zurich和UNC Chapel Hill的名称或其贡献者的名称来支持或推广基于该软件的产品，除非事先获得书面许可。

#pragma once

#include "colmap/geometry/rigid3.h"         // 引入刚体变换类（3D空间）
#include "colmap/util/eigen_alignment.h"    // 引入Eigen对齐工具
#include "colmap/util/types.h"              // 引入常用类型定义

#include <vector>
#include <Eigen/Core>                      // 引入Eigen核心模块

namespace colmap {

// 该类用于解决广义相对姿态（GR6P）问题，通过最小8个2D-2D对应点进行估计。
// 该实现基于以下文献：
//    "Efficient Computation of Relative Pose for Multi-Camera Systems",
//    Kneip 和 Li，CVPR 2014年。
// 需要注意的是，当所有对应点都来自同一相机或当相对运动是纯平移时，该问题的解会退化。
//
// 该实现是Kneip在OpenGV中的原始实现的修改和改进版本，OpenGV使用BSD许可。

class GR6PEstimator {
 public:
  // 左相机的广义图像观测数据，包括相机相对姿态和相机坐标系中的光线。
  struct X_t {
    Rigid3d cam_from_rig;        // 相机到rig坐标系的变换（刚体变换）
    Eigen::Vector3d ray_in_cam;  // 相机坐标系中的光线向量
  };

  // 右相机中归一化的图像特征点（与左相机对应的特征点）
  typedef X_t Y_t;  // 右相机的观测点类型和左相机相同
  // 估算出的相对姿态（rig2相机到rig1相机的变换）
  typedef Rigid3d M_t;

  // 估算模型所需的最小样本数。理论上，最少需要6个样本，但根据Laurent Kneip的论文，使用8个样本会更稳定。
  static const int kMinNumSamples = 8;

  // 通过一组2D-2D点对应关系估算GR6P问题的最可能解。
  // 输入：points1和points2是左相机和右相机的特征点集合
  // 输出：models是估算的相对姿态（rig2_from_rig1）的集合
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);

  // 计算对应点之间的Sampson误差的平方。
  // 输入：points1和points2是左相机和右相机的特征点集合，rig2_from_rig1是估算的相对姿态
  // 输出：residuals是每个点对应的Sampson误差的平方
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& rig2_from_rig1,
                        std::vector<double>* residuals);
};

}  // namespace colmap
