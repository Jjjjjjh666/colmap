//这个代码定义了一个名为 AffineTransformEstimator 的类，用于估计2D仿射变换

//仿射变换是一种线性变换，常用于计算机视觉和图像处理。它可以描述图像中点的以下操作
//缩放：改变图像的大小。
//旋转：围绕某个点旋转图像。
//平移：移动图像的位置。
//剪切：将图像的形状拉伸或压缩。
//仿射变换相关视频可以看：b站BV13T411d75F

//这个类的主要功能是从一组对应的2D点中估计出仿射变换矩阵。具体用途包括：

// 图像配准：将两幅图像对齐，使它们在相同的坐标系下进行比较或合并。
// 这在图像拼接、全景图生成中非常重要。

// 图像校正：修正图像中的几何失真。
// 例如，校正因相机角度或镜头畸变导致的图像变形。

// 特征匹配：在图像处理和计算机视觉中，
// 经常需要在不同视角下识别相同的物体。
// 仿射变换帮助在不同图像之间建立对应关系。


#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

class AffineTransformEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix<double, 2, 3> M_t;

  // 估计模型所需的最小样本数量。
  static const int kMinNumSamples = 3;

  // 从至少3个对应点估计仿射变换。
  //接受两个点集 points1 和 points2，每个点集代表一幅图像中的特征点。
  //估计出一个或多个仿射变换矩阵 models，这些矩阵描述了如何从 points1 变换到 points2。
  static void Estimate(const std::vector<X_t>& points1,
                       const std::vector<Y_t>& points2,
                       std::vector<M_t>* models);


  //计算变换后的点与目标点之间的误差（平方误差）。
  //评估变换的准确性，误差越小，变换越精确
  // 计算平方变换误差。(Residuals 函数用于评估 Estimate 得到的仿射变换矩阵的可靠性)
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& E,
                        std::vector<double>* residuals);
};

}  // namespace colmap
