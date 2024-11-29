// 本文件定义了一些用于对齐和合并三维重建的函数（稀疏重建阶段）
// 对齐：把两个不同的三维重建结果调整到一个共同的坐标系中，让它们看起来像是从同一个视角拍摄的
// 合并：将对齐后的稀疏重建结果合并，整合相机、图像和3D点数据，形成一个更完整的稀疏模型


#pragma once

#include "colmap/geometry/sim3.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {

// 将重建与给定的地理位置对齐。
bool AlignReconstructionToLocations(
    const Reconstruction& reconstruction,//包含相机、图像和3D点的重建对象。
    const std::vector<std::string>& image_names,//需要对齐的图像名称列表
    const std::vector<Eigen::Vector3d>& locations,//对应于图像的地理位置
    int min_common_images,//用于对齐的最少公共图像数量
    const RANSACOptions& ransac_options,//RANSAC（随机抽样一致性）是一种稳健估计方法，用于在数据中识别内点和异常值
    Sim3d* tform);

//用于对齐两个重建结果(应该是稀疏点云),通过最小化重投影误差来实现。
//重投影是指将三维空间中的点通过相机模型投影到二维图像平面上
bool AlignReconstructionsViaReprojections(
    const Reconstruction& src_reconstruction,//源重建对象，需要对齐的重建
    const Reconstruction& tgt_reconstruction,//目标重建对象，作为对齐基准的重建。
    double min_inlier_observations,//用于对齐的最小内点观测数。
    double max_reproj_error,//允许的最大重投影误差
    Sim3d* tgt_from_src);//输出的相似变换（Sim3d），用于将源重建对齐到目标重建

//用于对齐两个重建结果(应该是稀疏点云),通过对齐相机投影中心进行对齐。
//相机投影中心是相机在三维空间中的位置。
//比较两个重建中相机位置的差异，并通过计算一个变换0
//使得源重建的相机位置与目标重建的相机位置尽可能接近
bool AlignReconstructionsViaProjCenters(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    double max_proj_center_error,
    Sim3d* tgt_from_src);


//通过公共点的匹配，将源重建对齐到目标重建
//在两个重建中找到相同的三维点，通常通过匹配特征点的标识符实现
bool AlignReconstructionsViaPoints(const Reconstruction& src_reconstruction,
                                   const Reconstruction& tgt_reconstruction,
                                   size_t min_common_observations,
                                   double max_error,
                                   double min_inlier_ratio,
                                   Sim3d* tgt_from_src);


// 标识储存图像的名称以及它的旋转和投影中心误差
struct ImageAlignmentError {
  std::string image_name;
  double rotation_error_deg = -1;
  double proj_center_error = -1;
};
//用于计算对齐误差的函数
//返回一个 std::vector<ImageAlignmentError>，其中每个元素包含一个图像的对齐误差信息。
std::vector<ImageAlignmentError> ComputeImageAlignmentError(
    const Reconstruction& src_reconstruction,
    const Reconstruction& tgt_reconstruction,
    const Sim3d& tgt_from_src);

// 将源重建对齐到目标重建，并使用对齐合并相机、图像和 3D 点到目标中。
// 如果失败则返回 false。
bool MergeReconstructions(double max_reproj_error,
                          const Reconstruction& src_reconstruction,
                          Reconstruction& tgt_reconstruction);

}  // namespace colmap
