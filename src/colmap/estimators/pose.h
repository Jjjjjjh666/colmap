// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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

// 引入必要的头文件
#include "colmap/geometry/rigid3.h"
#include "colmap/optim/loransac.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// 绝对姿态估计选项结构体
struct AbsolutePoseEstimationOptions {
  bool estimate_focal_length = false; // 是否估计焦距
  RANSACOptions ransac_options;      // 用于P3P RANSAC的选项

  AbsolutePoseEstimationOptions() {
    ransac_options.max_error = 12.0;  // RANSAC允许的最大误差
    ransac_options.min_num_trials = 100; // 最小试验次数
    ransac_options.max_num_trials = 10000; // 最大试验次数
    ransac_options.confidence = 0.99999; // 高置信度，以避免过早中止
  }

  void Check() const { ransac_options.Check(); } // 检查配置是否有效
};

// 绝对姿态优化选项结构体
struct AbsolutePoseRefinementOptions {
  // 收敛准则
  double gradient_tolerance = 1.0;

  // 最大求解器迭代次数
  int max_num_iterations = 100;

  // 缩放因子，确定何时进行稳健化处理
  double loss_function_scale = 1.0;

  // 是否优化焦距参数
  bool refine_focal_length = false;

  // 是否优化额外的参数
  bool refine_extra_params = false;

  // 是否打印最终总结
  bool print_summary = false;

  void Check() const {
    THROW_CHECK_GE(gradient_tolerance, 0.0); // 确保梯度容忍度不小于0
    THROW_CHECK_GE(max_num_iterations, 0);   // 确保最大迭代次数不小于0
    THROW_CHECK_GE(loss_function_scale, 0.0); // 确保损失函数缩放因子不小于0
  }
};

// 根据2D-3D对应点估计绝对姿态（可选地估计焦距）
// 焦距估计通过对给定相机的焦距进行离散采样，选择使得内点数最多的焦距。
// 焦距估计过程采用RANSAC方法进行内点选择。
//
// @param options              绝对姿态估计选项。
// @param points2D             2D点对应的图像坐标。
// @param points3D             3D点对应的世界坐标。
// @param cam_from_world       估计得到的绝对相机姿态。
// @param camera               需要估计姿态的相机，估计结果会修改该相机对象。
// @param num_inliers          RANSAC算法中内点的数量。
// @param inlier_mask          2D-3D对应点的内点掩码。
//
// @return                     是否成功估计出姿态。
bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options,
                          const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D,
                          Rigid3d* cam_from_world,
                          Camera* camera,
                          size_t* num_inliers,
                          std::vector<char>* inlier_mask);

// 通过2D-2D对应点估计相对姿态。
// 假设第一台相机的姿态是位于原点且无旋转，第二台相机的姿态是世界到相机的变换，
// 即 `x2 = [R | t] * X2`。
//
// @param ransac_options       RANSAC选项。
// @param points1              第一组2D对应点。
// @param points2              第二组2D对应点。
// @param cam2_from_cam1       估计的两台相机之间的姿态。
//
// @return                     RANSAC内点的数量。
size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Rigid3d* cam2_from_cam1);

// 通过2D-3D对应点优化绝对姿态（可选地优化焦距）。
//
// @param options              优化选项。
// @param inlier_mask          2D-3D对应点的内点掩码。
// @param points2D             2D点对应的图像坐标。
// @param points3D             3D点对应的世界坐标。
// @param cam_from_world       优化后的绝对相机姿态。
// @param camera               需要优化姿态的相机，优化结果会修改该相机对象。
// @param cam_from_world_cov   估计得到的6x6协方差矩阵（可选）。
//
// @return                     是否优化成功。
bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options,
                        const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        Rigid3d* cam_from_world,
                        Camera* camera,
                        Eigen::Matrix6d* cam_from_world_cov = nullptr);

// 优化两台相机的相对姿态。
// 最小化相应的标准误差，使用稳健的代价函数进行优化。
// 对应的点不一定是内点，但有足够好的初始估计时可以进行优化。
//
// 假设第一台相机的投影矩阵为 `P = [I | 0]`，第二台相机的姿态为从世界到相机坐标系的变换。
//
// 假设给定的平移向量是标准化的，并且优化平移向量，直到恢复到单位向量。
// 
// @param options          优化选项。
// @param points1          第一组2D对应点。
// @param points2          第二组2D对应点。
// @param cam_from_world   优化后的两台相机的姿态。
//
// @return                 是否优化成功。
bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Rigid3d* cam_from_world);

// 优化本质矩阵。
// 将本质矩阵分解为旋转和平移分量，并使用 `RefineRelativePose` 函数优化相对姿态。
//
// @param E                3x3本质矩阵。
// @param points1          第一组对应点。
// @param points2          第二组对应点。
// @param inlier_mask      对应点的内点掩码。
// @param options          优化选项。
//
// @return                 是否优化成功。
bool RefineEssentialMatrix(const ceres::Solver::Options& options,
                           const std::vector<Eigen::Vector2d>& points1,
                           const std::vector<Eigen::Vector2d>& points2,
                           const std::vector<char>& inlier_mask,
                           Eigen::Matrix3d* E);

}  // namespace colmap
