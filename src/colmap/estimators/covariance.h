#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/reconstruction.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/ceres.h>

namespace colmap {

// 捆绑调整（或扩展）问题的协方差估计。
// 该接口适用于所有基于捆绑调整扩展的 Ceres 问题。
// 显式计算舒尔补以消除所有 3D 点的 Hessian 块，
// 这是避免大规模重建中雅可比秩亏的关键。
class BundleAdjustmentCovarianceEstimatorBase {
 public:
  // 使用 COLMAP 重建构造
  BundleAdjustmentCovarianceEstimatorBase(ceres::Problem* problem,
                                          Reconstruction* reconstruction);
  // 通过指定姿态块和点块构造
  BundleAdjustmentCovarianceEstimatorBase(
      ceres::Problem* problem,
      const std::vector<const double*>& pose_blocks,
      const std::vector<const double*>& point_blocks);
  virtual ~BundleAdjustmentCovarianceEstimatorBase() = default;

  // 手动设置感兴趣的姿态块
  void SetPoseBlocks(const std::vector<const double*>& pose_blocks);

  // 计算所有参数的协方差（不包括 3D 点）。
  // 将完整矩阵存储在 cov_variables_ 中，并将子块副本存储在 cov_poses_ 中；
  virtual bool ComputeFull() = 0;

  // 计算姿态参数的协方差。
  // 存储在 cov_poses_ 中；
  virtual bool Compute() = 0;

  // 接口
  // 测试块是否对应于问题中的任何参数（不包括 3D 点）
  bool HasBlock(const double* params) const;
  // 测试块是否对应于姿态块中的任何参数
  bool HasPoseBlock(const double* params) const;
  // 测试估计器是否使用 COLMAP 重建构造
  bool HasReconstruction() const;
  // 测试姿态是否作为非常量变量在问题中
  bool HasPose(image_t image_id) const;

  // 姿态参数
  Eigen::MatrixXd GetPoseCovariance() const;
  Eigen::MatrixXd GetPoseCovariance(image_t image_id) const;
  Eigen::MatrixXd GetPoseCovariance(
      const std::vector<image_t>& image_ids) const;
  Eigen::MatrixXd GetPoseCovariance(image_t image_id1, image_t image_id2) const;
  Eigen::MatrixXd GetPoseCovariance(double* parameter_block) const;
  Eigen::MatrixXd GetPoseCovariance(
      const std::vector<double*>& parameter_blocks) const;
  Eigen::MatrixXd GetPoseCovariance(double* parameter_block1,
                                    double* parameter_block2) const;

  // 所有参数（不包括 3D 点）
  Eigen::MatrixXd GetCovariance(double* parameter_block) const;
  Eigen::MatrixXd GetCovariance(
      const std::vector<double*>& parameter_blocks) const;
  Eigen::MatrixXd GetCovariance(double* parameter_block1,
                                double* parameter_block2) const;

  // 测试是否调用过 ``ComputeFull()`` 或 ``Compute()``
  bool HasValidPoseCovariance() const;
  // 测试是否调用过 ``ComputeFull()``
  bool HasValidFullCovariance() const;

 protected:
  // 索引协方差矩阵
  virtual double GetCovarianceByIndex(int row, int col) const;
  virtual Eigen::MatrixXd GetCovarianceBlockOperation(int row_start,
                                                      int col_start,
                                                      int row_block_size,
                                                      int col_block_size) const;
  virtual double GetPoseCovarianceByIndex(int row, int col) const;
  virtual Eigen::MatrixXd GetPoseCovarianceBlockOperation(
      int row_start,
      int col_start,
      int row_block_size,
      int col_block_size) const;

  // 从重建中解析的块（在构造时初始化）
  std::vector<const double*> pose_blocks_;
  int num_params_poses_ = 0;
  std::vector<const double*> other_variables_blocks_;
  int num_params_other_variables_ = 0;
  std::vector<const double*> point_blocks_;
  int num_params_points_ = 0;

  // 获取矩阵中参数块的起始索引
  // 顺序：[pose_blocks, other_variables_blocks, point_blocks]
  std::map<const double*, int> map_block_to_index_;

  int GetBlockIndex(const double* params) const;
  int GetBlockTangentSize(const double* params) const;
  int GetPoseIndex(image_t image_id) const;
  int GetPoseTangentSize(image_t image_id) const;

  // 所有参数（不包括 3D 点）的协方差
  Eigen::MatrixXd cov_variables_;

  // 姿态参数的协方差
  Eigen::MatrixXd cov_poses_;

  // ceres 问题
  ceres::Problem* problem_;

  // 重建
  Reconstruction* reconstruction_ = nullptr;

 private:
  // 设置参数块
  void SetUpOtherVariablesBlocks();
};

class BundleAdjustmentCovarianceEstimatorCeresBackend
    : public BundleAdjustmentCovarianceEstimatorBase {
 public:
  BundleAdjustmentCovarianceEstimatorCeresBackend(
      ceres::Problem* problem, Reconstruction* reconstruction)
      : BundleAdjustmentCovarianceEstimatorBase(problem, reconstruction) {}

  BundleAdjustmentCovarianceEstimatorCeresBackend(
      ceres::Problem* problem,
      const std::vector<const double*>& pose_blocks,
      const std::vector<const double*>& point_blocks)
      : BundleAdjustmentCovarianceEstimatorBase(
            problem, pose_blocks, point_blocks) {}

  bool ComputeFull() override;
  bool Compute() override;
};

class BundleAdjustmentCovarianceEstimator
    : public BundleAdjustmentCovarianceEstimatorBase {
 public:
  BundleAdjustmentCovarianceEstimator(ceres::Problem* problem,
                                      Reconstruction* reconstruction,
                                      double lambda = 1e-8)
      : BundleAdjustmentCovarianceEstimatorBase(problem, reconstruction),
        lambda_(lambda) {}
  BundleAdjustmentCovarianceEstimator(
      ceres::Problem* problem,
      const std::vector<const double*>& pose_blocks,
      const std::vector<const double*>& point_blocks,
      double lambda = 1e-8)
      : BundleAdjustmentCovarianceEstimatorBase(
            problem, pose_blocks, point_blocks),
        lambda_(lambda) {}

  bool ComputeFull() override;
  bool Compute() override;

  // 分解
  bool FactorizeFull();
  bool Factorize();
  bool HasValidFullFactorization() const;
  bool HasValidPoseFactorization() const;

 private:
  // 索引协方差矩阵
  double GetCovarianceByIndex(int row, int col) const override;
  Eigen::MatrixXd GetCovarianceBlockOperation(
      int row_start,
      int col_start,
      int row_block_size,
      int col_block_size) const override;
  double GetPoseCovarianceByIndex(int row, int col) const override;
  Eigen::MatrixXd GetPoseCovarianceBlockOperation(
      int row_start,
      int col_start,
      int row_block_size,
      int col_block_size) const override;

  // 通过舒尔消去后所有参数（不包括 3D 点）的舒尔补
  Eigen::SparseMatrix<double> S_matrix_;

  // 为避免秩亏的阻尼因子
  const double lambda_ = 1e-8;

  // 通过消除 3D 点计算姿态和其他变量的舒尔补
  void ComputeSchurComplement();
  bool HasValidSchurComplement() const;

  // Cholesky 分解后的 L 矩阵的逆
  Eigen::MatrixXd L_matrix_variables_inv_;
  Eigen::MatrixXd L_matrix_poses_inv_;
};

// 每个图像的协方差顺序为 [R, t]，两者可能都在流形上
// （R 至少总是用 ceres::QuaternionManifold 在 Lie 代数上参数化）。
// 因此，协方差仅在每个变量的非常量部分上计算。
// 如果旋转和平移的完整部分都在问题中，协方差矩阵将为 6x6。
bool EstimatePoseCovarianceCeresBackend(
    ceres::Problem* problem,
    Reconstruction* reconstruction,
    std::map<image_t, Eigen::MatrixXd>& image_id_to_covar);

// 与 ``EstimatePoseCovarianceCeresBackend`` 的约定类似。
bool EstimatePoseCovariance(
    ceres::Problem* problem,
    Reconstruction* reconstruction,
    std::map<image_t, Eigen::MatrixXd>& image_id_to_covar,
    double lambda = 1e-8);

}  // namespace colmap
