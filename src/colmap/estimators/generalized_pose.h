// 版权所有 (c) 2023, 瑞士苏黎世大学和美国北卡罗来纳大学教堂山分校。
// 保留所有权利。
//
// 允许以源代码和二进制形式进行再分发和使用，无论是否经过修改，
// 只要满足以下条件：
//
//     * 源代码的再分发必须保留上述版权声明、此条件列表以及以下免责声明。
//
//     * 二进制形式的再分发必须在提供的文档和/或其他材料中，
//       复制上述版权声明、此条件列表和以下免责声明。
//
//     * 未经明确书面许可，不能使用瑞士苏黎世大学和美国北卡罗来纳大学教堂山分校的名称，
//       或其贡献者的名称来支持或推广从本软件衍生的产品。
//
// 本软件是由版权持有者和贡献者“按原样”提供的，
// 不提供任何明示或暗示的保证，包括但不限于适销性和特定用途适用性的暗示保证。
// 在任何情况下，版权持有者或贡献者均不对因使用本软件而导致的任何直接、间接、附带、特殊、示范性或后果性损害负责，
// 无论是基于合同、严格责任还是侵权（包括过失或其他）理论，
// 即使事先已被告知可能会发生此类损害。

#pragma once

#include "colmap/estimators/pose.h"           // 导入位姿估计相关头文件
#include "colmap/geometry/rigid3.h"           // 导入刚体3D变换相关头文件
#include "colmap/optim/ransac.h"             // 导入RANSAC优化算法头文件
#include "colmap/scene/camera.h"             // 导入相机模型相关头文件
#include "colmap/util/eigen_alignment.h"     // 导入Eigen库对齐相关头文件

#include <vector>                           // 导入标准库中的vector容器
#include <Eigen/Core>                       // 导入Eigen库核心功能

namespace colmap {

// 估计来自2D-3D对应点的广义绝对位姿。
// 
// @param options              RANSAC算法的选项。
// @param points2D             对应的2D点。
// @param points3D             对应的3D点。
// @param camera_idxs          每个对应点的相机索引。
// @param cams_from_rig        从相机框架到rig的相对位姿。
// @param cameras              参与估计的相机。
// @param rig_from_world       从世界到rig的估计位姿。
// @param num_inliers          RANSAC中的内点数量。
// @param inlier_mask          2D-3D对应点的内点掩码。
// 
// @return                     位姿是否成功估计。
bool EstimateGeneralizedAbsolutePose(
    const RANSACOptions& options,               // RANSAC算法配置
    const std::vector<Eigen::Vector2d>& points2D, // 输入的2D点
    const std::vector<Eigen::Vector3d>& points3D, // 输入的3D点
    const std::vector<size_t>& camera_idxs,      // 对应点的相机索引
    const std::vector<Rigid3d>& cams_from_rig,   // 相机与rig之间的相对位姿
    const std::vector<Camera>& cameras,          // 相机列表
    Rigid3d* rig_from_world,                    // 输出：rig到世界的位姿
    size_t* num_inliers,                        // 输出：RANSAC算法中的内点数量
    std::vector<char>* inlier_mask              // 输出：内点掩码
);

// 通过2D-3D对应点优化（可选的焦距优化）来精细化广义绝对位姿。
// 
// @param options              精细化的配置选项。
// @param inlier_mask          2D-3D对应点的内点掩码。
// @param points2D             对应的2D点。
// @param points3D             对应的3D点。
// @param camera_idxs          每个对应点的相机索引。
// @param cams_from_rig        从rig到相机框架的相对位姿。
// @param rig_from_world       估计的rig到世界的位姿。
// @param cameras              参与精细化的相机。返回时会更新相机的焦距。
// @param rig_from_world_cov   可选的输出，表示估计的rig到世界位姿的协方差矩阵（6x6），
//                             包含旋转（轴角形式）和平移的协方差信息。
// 
// @return                     解是否可用。
bool RefineGeneralizedAbsolutePose(
    const AbsolutePoseRefinementOptions& options, // 精细化配置
    const std::vector<char>& inlier_mask,         // 内点掩码
    const std::vector<Eigen::Vector2d>& points2D, // 输入的2D点
    const std::vector<Eigen::Vector3d>& points3D, // 输入的3D点
    const std::vector<size_t>& camera_idxs,      // 对应点的相机索引
    const std::vector<Rigid3d>& cams_from_rig,   // 相机与rig的相对位姿
    Rigid3d* rig_from_world,                     // 输出：rig到世界的位姿
    std::vector<Camera>* cameras,                // 输出：精细化后的相机，包含焦距信息
    Eigen::Matrix6d* rig_from_world_cov = nullptr // 可选：输出位姿的协方差矩阵
);

}  // namespace colmap
