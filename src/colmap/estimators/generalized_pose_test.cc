// 版权所有 (c) 2023, ETH Zurich 和 UNC Chapel Hill。
// 保留所有权利。
//
// 允许在满足以下条件的情况下对源代码和二进制形式进行复制和使用，无论是否经过修改：
//
//     * 源代码的再分发必须保留上述版权声明、此条件列表和以下免责声明。
//     * 二进制形式的再分发必须在随附的文档和/或其他材料中包含上述版权声明、此条件列表和以下免责声明。
//     * 未经特定的事先书面许可，不得使用 ETH Zurich 和 UNC Chapel Hill 的名称及其贡献者的名称来背书或推广基于本软件的产品。
//
// 本软件由版权所有者及其贡献者按“原样”提供，不提供任何明示或暗示的担保，包括但不限于适销性和特定用途适用性的默示担保。
// 在任何情况下，版权所有者或贡献者均不对任何直接、间接、附带、特殊、惩罚性或后果性损害（包括但不限于采购替代货物或服务的费用、使用损失、数据或利润损失、业务中断）负责，无论是由于合同、严格责任或侵权行为（包括疏忽或其他）引起的，即使已被告知此类损害的可能性。

#include "colmap/estimators/generalized_pose.h" // 包含广义位姿估计相关的头文件
#include "colmap/estimators/generalized_absolute_pose.h" // 包含绝对位姿估计的头文件
#include "colmap/geometry/rigid3.h" // 包含刚体变换相关定义
#include "colmap/math/random.h" // 包含随机数生成工具
#include "colmap/optim/ransac.h" // 包含 RANSAC 优化算法
#include "colmap/scene/camera.h" // 包含相机建模相关功能
#include "colmap/scene/reconstruction.h" // 包含重建场景相关功能
#include "colmap/scene/synthetic.h" // 包含合成数据集生成相关功能

#include <numeric> // 提供常用的数值计算功能
#include <gtest/gtest.h> // Google Test 框架，用于单元测试

namespace colmap {
namespace {

// 定义一个结构体用于存储广义相机问题相关的数据
struct GeneralizedCameraProblem {
  Rigid3d gt_rig_from_world; // 世界坐标到刚体坐标系的真实变换
  std::vector<Eigen::Vector2d> points2D; // 图像上的 2D 点集合
  std::vector<Eigen::Vector3d> points3D; // 空间中的 3D 点集合
  std::vector<size_t> point3D_ids; // 每个 3D 点的唯一 ID
  std::vector<size_t> camera_idxs; // 对应的相机索引
  std::vector<Rigid3d> cams_from_rig; // 各相机相对于刚体的变换
  std::vector<Camera> cameras; // 相机模型集合
};

// 构建一个用于广义相机问题的合成数据集
GeneralizedCameraProblem BuildGeneralizedCameraProblem() {
  Reconstruction reconstruction; // 定义一个场景重建对象
  SyntheticDatasetOptions synthetic_dataset_options; // 定义合成数据集选项
  synthetic_dataset_options.num_cameras = 3; // 指定相机数量
  synthetic_dataset_options.num_images = 3; // 指定图像数量
  synthetic_dataset_options.num_points3D = 50; // 指定 3D 点数量
  synthetic_dataset_options.point2D_stddev = 0; // 设置 2D 点的噪声为 0
  SynthesizeDataset(synthetic_dataset_options, &reconstruction); // 生成合成数据集

  GeneralizedCameraProblem problem;
  problem.gt_rig_from_world =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()); // 随机初始化真实的刚体变换
  for (const image_t image_id : reconstruction.RegImageIds()) { // 遍历所有注册图像
    const auto& image = reconstruction.Image(image_id); // 获取图像数据
    for (const auto& point2D : image.Points2D()) { // 遍历每个 2D 点
      if (point2D.HasPoint3D()) { // 如果 2D 点对应一个 3D 点
        problem.points2D.push_back(point2D.xy); // 添加 2D 坐标
        problem.points3D.push_back(
            reconstruction.Point3D(point2D.point3D_id).xyz); // 添加对应的 3D 坐标
        problem.point3D_ids.push_back(point2D.point3D_id); // 添加点 ID
        problem.camera_idxs.push_back(problem.cameras.size()); // 添加相机索引
      }
    }
    problem.cameras.push_back(*image.CameraPtr()); // 存储相机模型
    problem.cams_from_rig.push_back(image.CamFromWorld() *
                                    Inverse(problem.gt_rig_from_world)); // 计算相机到刚体的变换
  }
  return problem;
}

// 测试广义绝对位姿的估计
TEST(EstimateGeneralizedAbsolutePose, Nominal) {
  GeneralizedCameraProblem problem = BuildGeneralizedCameraProblem(); // 构建广义相机问题
  const size_t num_points = problem.points2D.size(); // 获取点的总数

  // 设置地面真实值的内点比例和噪声
  const double gt_inlier_ratio = 0.8;
  const double outlier_distance = 50;
  const size_t gt_num_inliers =
      std::max(static_cast<size_t>(gt_inlier_ratio * num_points),
               static_cast<size_t>(GP3PEstimator::kMinNumSamples)); // 确保内点数量至少为估计器所需的最小样本数
  std::vector<size_t> shuffled_idxs(num_points);
  std::iota(shuffled_idxs.begin(), shuffled_idxs.end(), 0); // 初始化索引
  std::shuffle(shuffled_idxs.begin(), shuffled_idxs.end(), *PRNG); // 随机打乱索引

  // 随机选择内点
  std::unordered_set<size_t> unique_inlier_ids;
  unique_inlier_ids.reserve(gt_num_inliers);
  for (size_t i = 0; i < gt_num_inliers; ++i) {
    unique_inlier_ids.insert(problem.point3D_ids[shuffled_idxs[i]]);
  }

  // 为外点添加噪声
  std::vector<char> gt_inlier_mask(num_points, true);
  for (size_t i = gt_num_inliers; i < num_points; ++i) {
    problem.points2D[shuffled_idxs[i]] +=
        Eigen::Vector2d::Random().normalized() * outlier_distance; // 添加随机噪声
    gt_inlier_mask[shuffled_idxs[i]] = false; // 标记为外点
  }

  // 配置 RANSAC 参数
  RANSACOptions ransac_options;
  ransac_options.max_error = 2; // 最大误差
  ransac_options.min_inlier_ratio = gt_inlier_ratio / 2; // 最小内点比例
  ransac_options.confidence = 0.99999; // 置信度

  // 运行估计器
  Rigid3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateGeneralizedAbsolutePose(ransac_options,
                                              problem.points2D,
                                              problem.points3D,
                                              problem.camera_idxs,
                                              problem.cams_from_rig,
                                              problem.cameras,
                                              &rig_from_world,
                                              &num_inliers,
                                              &inlier_mask)); // 调用位姿估计器
  EXPECT_EQ(num_inliers, unique_inlier_ids.size()); // 验证内点数量
  EXPECT_EQ(inlier_mask, gt_inlier_mask); // 验证内点掩码
  EXPECT_LT(problem.gt_rig_from_world.rotation.angularDistance(
                rig_from_world.rotation),
            1e-6); // 验证旋转误差
  EXPECT_LT((problem.gt_rig_from_world.translation - rig_from_world.translation)
                .norm(),
            1e-6); // 验证平移误差
}

// 测试广义绝对位姿的优化
TEST(RefineGeneralizedAbsolutePose, Nominal) {
  GeneralizedCameraProblem problem = BuildGeneralizedCameraProblem(); // 构建广义相机问题
  const std::vector<char> gt_inlier_mask(problem.points2D.size(), true); // 假设所有点都是内点

  // 添加初始位姿的噪声
  const double rotation_noise_degree = 1; // 旋转噪声（度）
  const double translation_noise = 0.1; // 平移噪声
  const Rigid3d rig_from_gt_rig(Eigen::Quaterniond(Eigen::AngleAxisd(
                                    DegToRad(rotation_noise_degree
