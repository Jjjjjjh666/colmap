//本文件主要用于处理相机的绝对位姿估计问题
//在计算机视觉中，绝对位姿估计是通过已知的2D图像点和对应的3D世界点来确定相机的位姿
//这个文件定义了COLMAP项目中的一些绝对位姿估计器类，用于解决相机位姿估计问题。主要包括P3P、P4PF和EPNP估计器

//P3P问题是指通过三对已知的二维图像点和其对应的三维世界点来估计相机的姿态（旋转和平移）
//P4PF是P3P的扩展，除了估计相机的姿态外，还要估计相机的焦距。需要四对2D-3D点对应关系
//PNP问题是在已知相机内参的情况下，通过n对二维图像点及其对应的三维世界点来估计相机的姿态（旋转和平移）


#pragma once

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <array>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// P3PEstimator类用于解决P3P问题，通过三对2D-3D点来估计相机的旋转和平移。

// 输入：
// points2D: 归一化的2D图像点，类型为std::vector<Eigen::Vector2d>。
// points3D: 3D世界点，类型为std::vector<Eigen::Vector3d>。

// 输出：
// cams_from_world: 最可能的位姿，类型为std::vector<Eigen::Matrix3x4d>。

// 方法：
// Estimate: 计算相机的姿态。
// Residuals: 计算平方重投影误差。
class P3PEstimator {
 public:
  // 2D图像特征观测。
  // TODO(jsch): 可能更改为3D射线方向，并将残差表示为角度误差。需要一些评估。
  typedef Eigen::Vector2d X_t;
  // 世界坐标系中的3D特征。
  typedef Eigen::Vector3d Y_t;
  // 从世界坐标系到相机坐标系的变换。
  typedef Eigen::Matrix3x4d M_t;

  // 估计模型所需的最小样本数量。
  static const int kMinNumSamples = 3;

  // 从一组三对2D-3D点对应关系中估计P3P问题的最可能解。
  //
  // @param points2D   归一化的2D图像点，3x2矩阵。
  // @param points3D   3D世界点，3x3矩阵。
  //
  // @return           最可能的位姿，作为一个3x4矩阵的长度为1的向量返回。
  static void Estimate(const std::vector<X_t>& points2D,
                       const std::vector<Y_t>& points3D,
                       std::vector<M_t>* cams_from_world);

  // 给定一组2D-3D点对应关系和投影矩阵，计算平方重投影误差。
  //
  // @param points2D        归一化的2D图像点，Nx2矩阵。
  // @param points3D        3D世界点，Nx3矩阵。
  // @param cam_from_world  3x4投影矩阵。
  // @param residuals       输出残差向量。
  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& cam_from_world,
                        std::vector<double>* residuals);
};




// P4PFEstimator类用于解决P4PF问题，估计相机的旋转、平移和焦距。

// 输入：
// points2D: 归一化的2D图像点。
// points3D: 3D世界点。

// 输出：
// models: 包含相机姿态和焦距的模型，类型为std::vector<M_t>，其中M_t包括cam_from_world和focal_length。

// 方法：
// Estimate: 计算相机的姿态和焦距。
// Residuals: 计算平方重投影误差。

class P4PFEstimator {
 public:
  // 2D图像特征观测。
  // 期望被主点归一化。
  typedef Eigen::Vector2d X_t;
  // 世界坐标系中的3D特征。
  typedef Eigen::Vector3d Y_t;
  struct M_t {
    // 从世界坐标系到相机坐标系的变换。
    Eigen::Matrix3x4d cam_from_world;
    // 相机的焦距。
    double focal_length = 0.;
  };

  static const int kMinNumSamples = 4;

  static void Estimate(const std::vector<X_t>& points2D,
                       const std::vector<Y_t>& points3D,
                       std::vector<M_t>* models);

  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& model,
                        std::vector<double>* residuals);
};



// EPNPEstimator类用于解决PNP问题，适用于4个或更多2D-3D点。

// 输入：
// points2D: 归一化的2D图像点。
// points3D: 3D世界点。

// 输出：
// cams_from_world: 最可能的位姿，类型为std::vector<Eigen::Matrix3x4d>。

// 方法：
// Estimate: 计算相机的姿态。
// Residuals: 计算平方重投影误差。

class EPNPEstimator {
 public:
  // 2D图像特征观测。
  typedef Eigen::Vector2d X_t;
  // 世界坐标系中的3D特征。
  typedef Eigen::Vector3d Y_t;
  // 从世界坐标系到相机坐标系的变换。
  typedef Eigen::Matrix3x4d M_t;

  // 估计模型所需的最小样本数量。
  static const int kMinNumSamples = 4;

  // 从一组三对2D-3D点对应关系中估计P3P问题的最可能解。
  //
  // @param points2D   归一化的2D图像点，3x2矩阵。
  // @param points3D   3D世界点，3x3矩阵。
  //
  // @return           最可能的位姿，作为一个3x4矩阵的长度为1的向量返回。
  static void Estimate(const std::vector<X_t>& points2D,
                       const std::vector<Y_t>& points3D,
                       std::vector<M_t>* cams_from_world);

  // 给定一组2D-3D点对应关系和投影矩阵，计算平方重投影误差。
  //
  // @param points2D        归一化的2D图像点，Nx2矩阵。
  // @param points3D        3D世界点，Nx3矩阵。
  // @param cam_from_world  3x4投影矩阵。
  // @param residuals       输出残差向量。
  static void Residuals(const std::vector<X_t>& points2D,
                        const std::vector<Y_t>& points3D,
                        const M_t& cam_from_world,
                        std::vector<double>* residuals);

 private:
  bool ComputePose(const std::vector<Eigen::Vector2d>& points2D,
                   const std::vector<Eigen::Vector3d>& points3D,
                   Eigen::Matrix3x4d* cam_from_world);

  void ChooseControlPoints();
  bool ComputeBarycentricCoordinates();

  Eigen::Matrix<double, Eigen::Dynamic, 12> ComputeM();
  Eigen::Matrix<double, 6, 10> ComputeL6x10(
      const Eigen::Matrix<double, 12, 12>& Ut);
  Eigen::Matrix<double, 6, 1> ComputeRho();

  void FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);
  void FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L_6x10,
                        const Eigen::Matrix<double, 6, 1>& rho,
                        Eigen::Vector4d* betas);

  void RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L_6x10,
                      const Eigen::Matrix<double, 6, 1>& rho,
                      Eigen::Vector4d* betas);

  double ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut,
                   const Eigen::Vector4d& betas,
                   Eigen::Matrix3d* R,
                   Eigen::Vector3d* t);

  void ComputeCcs(const Eigen::Vector4d& betas,
                  const Eigen::Matrix<double, 12, 12>& Ut);
  void ComputePcs();

  void SolveForSign();

  void EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t);

  double ComputeTotalReprojectionError(const Eigen::Matrix3d& R,
                                       const Eigen::Vector3d& t);

  const std::vector<Eigen::Vector2d>* points2D_ = nullptr;
  const std::vector<Eigen::Vector3d>* points3D_ = nullptr;
  std::vector<Eigen::Vector3d> pcs_;
  std::vector<Eigen::Vector4d> alphas_;
  std::array<Eigen::Vector3d, 4> cws_;
  std::array<Eigen::Vector3d, 4> ccs_;
};

}  // namespace colmap
