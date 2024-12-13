//本文件中的内容与坐标系相关
#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>

namespace colmap {

// 曼哈顿世界坐标系估计的选项结构体
// 配置用于曼哈顿世界坐标系估计的参数。
// 曼哈顿世界坐标系中：假设场景中的结构主要由正交的平面组成（如城市中的建筑物）
struct ManhattanWorldFrameEstimationOptions {
  // 线段检测的最大图像尺寸。
  int max_image_size = 1024;
  // 线段的最小长度（像素）。
  double min_line_length = 3;
  // 将线分类为水平/垂直的容差。
  double line_orientation_tolerance = 0.2;
  // 线与消失点之间的最大距离（像素）。
  double max_line_vp_distance = 0.5;
  // 估计轴之间作为内点的最大余弦距离。
  double max_axis_distance = 0.05;
};

// 一个函数
// 通过图像的方向估计重力向量。
// 图像的方向信息，输出：估计的重力向量，用于调整模型的方向
Eigen::Vector3d EstimateGravityVectorFromImageOrientation(
    const Reconstruction& reconstruction, double max_axis_distance = 0.05);

// 假设为曼哈顿世界，通过在每个图像中找到主要消失点来估计重建的坐标系。
// 此函数假设大多数图像是直立拍摄的，即人们在图像中是直立的。
// 估计坐标系的正交轴将作为返回矩阵的列给出。
// 如果某个轴无法确定，相应的列将为零。
// 这些轴在世界坐标系中按右、下、前的顺序指定。
#ifdef COLMAP_LSD_ENABLED
Eigen::Matrix3d EstimateManhattanWorldFrame(
    const ManhattanWorldFrameEstimationOptions& options,
    const Reconstruction& reconstruction,
    const std::string& image_path);
#endif

// 使用主成分分析将模型对齐到主要平面。
//输入：模型的点云数据
//输出：对齐后的模型数据，使其主要方向与坐标轴一致
void AlignToPrincipalPlane(Reconstruction* recon, Sim3d* tform);


// 将模型对齐到东-北-上（ENU）平面
//输入：模型的点云数据和可选的unscaled参数
//输出：对齐后的模型数据，使其符合地理坐标系

//ENU是一个地理坐标系，定义了东（E）、北（N）、上（U）的局部方向。
//用于将3D模型与地理位置关联，特别是在导航和地理信息系统中

// 旋转重建，使得x-y平面与点云质心处的ENU切平面对齐，并将原点平移到质心。
// 如果unscaled == true，则模型的原始比例保持不变。
void AlignToENUPlane(Reconstruction* recon, Sim3d* tform, bool unscaled);

}  // namespace colmap
