#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/conditioned_cost_function.h>
#include <ceres/rotation.h>

namespace colmap {

template <typename T>
using EigenVector3Map = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
template <typename T>
using EigenQuaternionMap = Eigen::Map<const Eigen::Quaternion<T>>;
using EigenMatrix6d = Eigen::Matrix<double, 6, 6>;

inline Eigen::MatrixXd SqrtInformation(const Eigen::MatrixXd& covariance) {
  return covariance.inverse().llt().matrixL().transpose();
}

// Standard bundle adjustment cost function for variable
// camera pose, calibration, and point parameters.
template <typename CameraModel>
class ReprojErrorCostFunctor {
 public:
  explicit ReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<ReprojErrorCostFunctor<CameraModel>,
                                            2,
                                            4,
                                            3,
                                            3,
                                            CameraModel::num_params>(
        new ReprojErrorCostFunctor(point2D)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_world_rotation) *
            EigenVector3Map<T>(point3D) +
        EigenVector3Map<T>(cam_from_world_translation);
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class ReprojErrorConstantPoseCostFunctor
    : public ReprojErrorCostFunctor<CameraModel> {
  using Parent = ReprojErrorCostFunctor<CameraModel>;

 public:
  ReprojErrorConstantPoseCostFunctor(const Rigid3d& cam_from_world,
                                     const Eigen::Vector2d& point2D)
      : Parent(point2D), cam_from_world_(cam_from_world) {}

  static ceres::CostFunction* Create(const Rigid3d& cam_from_world,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorConstantPoseCostFunctor<CameraModel>,
            2,
            3,
            CameraModel::num_params>(
        new ReprojErrorConstantPoseCostFunctor(cam_from_world, point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Quaternion<T> cam_from_world_rotation =
        cam_from_world_.rotation.cast<T>();
    const Eigen::Matrix<T, 3, 1> cam_from_world_translation =
        cam_from_world_.translation.cast<T>();
    return Parent::operator()(cam_from_world_rotation.coeffs().data(),
                              cam_from_world_translation.data(),
                              point3D,
                              camera_params,
                              residuals);
  }

 private:
  const Rigid3d& cam_from_world_;
};

// Bundle adjustment cost function for variable
// camera pose and calibration parameters, and fixed point.
template <typename CameraModel>
class ReprojErrorConstantPoint3DCostFunctor
    : public ReprojErrorCostFunctor<CameraModel> {
  using Parent = ReprojErrorCostFunctor<CameraModel>;

 public:
  ReprojErrorConstantPoint3DCostFunctor(const Eigen::Vector2d& point2D,
                                        const Eigen::Vector3d& point3D)
      : Parent(point2D),
        point3D_x_(point3D(0)),
        point3D_y_(point3D(1)),
        point3D_z_(point3D(2)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
                                     const Eigen::Vector3d& point3D) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorConstantPoint3DCostFunctor<CameraModel>,
            2,
            4,
            3,
            CameraModel::num_params>(
        new ReprojErrorConstantPoint3DCostFunctor(point2D, point3D)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const camera_params,
                  T* residuals) const {
    const T point3D[3] = {T(point3D_x_), T(point3D_y_), T(point3D_z_)};
    return Parent::operator()(cam_from_world_rotation,
                              cam_from_world_translation,
                              point3D,
                              camera_params,
                              residuals);
  }

 private:
  const double point3D_x_;
  const double point3D_y_;
  const double point3D_z_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigReprojErrorCostFunctor {
 public:
  explicit RigReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (
        new ceres::AutoDiffCostFunction<RigReprojErrorCostFunctor<CameraModel>,
                                        2,
                                        4,
                                        3,
                                        4,
                                        3,
                                        3,
                                        CameraModel::num_params>(
            new RigReprojErrorCostFunctor(point2D)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_rig_rotation,
                  const T* const cam_from_rig_translation,
                  const T* const rig_from_world_rotation,
                  const T* const rig_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_rig_rotation) *
            (EigenQuaternionMap<T>(rig_from_world_rotation) *
                 EigenVector3Map<T>(point3D) +
             EigenVector3Map<T>(rig_from_world_translation)) +
        EigenVector3Map<T>(cam_from_rig_translation);
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Rig bundle adjustment cost function for variable camera pose and camera
// calibration and point parameters but fixed rig extrinsic poses.
template <typename CameraModel>
class RigReprojErrorConstantRigCostFunctor
    : public RigReprojErrorCostFunctor<CameraModel> {
  using Parent = RigReprojErrorCostFunctor<CameraModel>;

 public:
  explicit RigReprojErrorConstantRigCostFunctor(const Rigid3d& cam_from_rig,
                                                const Eigen::Vector2d& point2D)
      : Parent(point2D), cam_from_rig_(cam_from_rig) {}

  static ceres::CostFunction* Create(const Rigid3d& cam_from_rig,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            RigReprojErrorConstantRigCostFunctor<CameraModel>,
            2,
            4,
            3,
            3,
            CameraModel::num_params>(
        new RigReprojErrorConstantRigCostFunctor(cam_from_rig, point2D)));
  }

  template <typename T>
  bool operator()(const T* const rig_from_world_rotation,
                  const T* const rig_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Quaternion<T> cam_from_rig_rotation =
        cam_from_rig_.rotation.cast<T>();
    const Eigen::Matrix<T, 3, 1> cam_from_rig_translation =
        cam_from_rig_.translation.cast<T>();
    return Parent::operator()(cam_from_rig_rotation.coeffs().data(),
                              cam_from_rig_translation.data(),
                              rig_from_world_rotation,
                              rig_from_world_translation,
                              point3D,
                              camera_params,
                              residuals);
  }

 private:
  const Rigid3d& cam_from_rig_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `SphereManifold`.
class SampsonErrorCostFunctor {
 public:
  SampsonErrorCostFunctor(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<SampsonErrorCostFunctor, 1, 4, 3>(
        new SampsonErrorCostFunctor(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const cam2_from_cam1_rotation,
                  const T* const cam2_from_cam1_translation,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 3> R =
        EigenQuaternionMap<T>(cam2_from_cam1_rotation).toRotationMatrix();

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -cam2_from_cam1_translation[2], cam2_from_cam1_translation[1],
        cam2_from_cam1_translation[2], T(0), -cam2_from_cam1_translation[0],
        -cam2_from_cam1_translation[1], cam2_from_cam1_translation[0], T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Homogeneous image coordinates.
    const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
    const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
    const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
    const T x2tEx1 = x2_h.transpose() * Ex1;
    residuals[0] = x2tEx1 * x2tEx1 /
                   (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                    Etx2(1) * Etx2(1));

    return true;
  }

 private:
  const double x1_;
  const double y1_;
  const double x2_;
  const double y2_;
};

template <typename T>
inline void EigenQuaternionToAngleAxis(const T* eigen_quaternion,
                                       T* angle_axis) {
  const T quaternion[4] = {eigen_quaternion[3],
                           eigen_quaternion[0],
                           eigen_quaternion[1],
                           eigen_quaternion[2]};
  ceres::QuaternionToAngleAxis(quaternion, angle_axis);
}

// 6-DoF error on the absolute pose. The residual is the log of the error pose,
// splitting SE(3) into SO(3) x R^3. The 6x6 covariance matrix is defined in the
// reference frame of the camera. Its first and last three components correspond
// to the rotation and translation errors, respectively.
struct AbsolutePoseErrorCostFunctor {
 public:
  AbsolutePoseErrorCostFunctor(const Rigid3d& cam_from_world,
                               const EigenMatrix6d& covariance_cam)
      : world_from_cam_(Inverse(cam_from_world)),
        sqrt_information_cam_(SqrtInformation(covariance_cam)) {}

  static ceres::CostFunction* Create(const Rigid3d& cam_from_world,
                                     const EigenMatrix6d& covariance_cam) {
    return (
        new ceres::AutoDiffCostFunction<AbsolutePoseErrorCostFunctor, 6, 4, 3>(
            new AbsolutePoseErrorCostFunctor(cam_from_world, covariance_cam)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_q,
                  const T* const cam_from_world_t,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_measured_q =
        EigenQuaternionMap<T>(cam_from_world_q) *
        world_from_cam_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_t(residuals_ptr + 3);
    param_from_measured_t = EigenVector3Map<T>(cam_from_world_t) +
                            EigenQuaternionMap<T>(cam_from_world_q) *
                                world_from_cam_.translation.cast<T>();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(sqrt_information_cam_.template cast<T>());
    return true;
  }

 private:
  const Rigid3d world_from_cam_;
  const EigenMatrix6d sqrt_information_cam_;
};

// 6-DoF error between two absolute poses based on a measurement that is their
// relative pose, with identical scale for the translation. The covariance is
// defined in the reference frame of the camera j.
// Its first and last three components correspond to the rotation and
// translation errors, respectively.
//
// Derivation:
// j_T_w = ΔT_j·j_T_i·i_T_w
// where ΔT_j = exp(η_j) is the residual in SE(3) and η_j in tangent space.
// Thus η_j = log(j_T_w·i_T_w⁻¹·i_T_j)
// Rotation term: ΔR = log(j_R_w·i_R_w⁻¹·i_R_j)
// Translation term: Δt = j_t_w + j_R_w·i_R_w⁻¹·(i_t_j -i_t_w)
struct MetricRelativePoseErrorCostFunctor {
 public:
  MetricRelativePoseErrorCostFunctor(const Rigid3d& i_from_j,
                                     const EigenMatrix6d& covariance_j)
      : i_from_j_(i_from_j),
        sqrt_information_j_(SqrtInformation(covariance_j)) {}

  static ceres::CostFunction* Create(const Rigid3d& i_from_j,
                                     const EigenMatrix6d& covariance_j) {
    return (new ceres::AutoDiffCostFunction<MetricRelativePoseErrorCostFunctor,
                                            6,
                                            4,
                                            3,
                                            4,
                                            3>(
        new MetricRelativePoseErrorCostFunctor(i_from_j, covariance_j)));
  }

  template <typename T>
  bool operator()(const T* const i_from_world_q,
                  const T* const i_from_world_t,
                  const T* const j_from_world_q,
                  const T* const j_from_world_t,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> j_from_i_q =
        EigenQuaternionMap<T>(j_from_world_q) *
        EigenQuaternionMap<T>(i_from_world_q).inverse();
    const Eigen::Quaternion<T> param_from_measured_q =
        j_from_i_q * i_from_j_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals_ptr);

    Eigen::Matrix<T, 3, 1> i_from_jw_t =
        i_from_j_.translation.cast<T>() - EigenVector3Map<T>(i_from_world_t);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_t(residuals_ptr + 3);
    param_from_measured_t =
        EigenVector3Map<T>(j_from_world_t) + j_from_i_q * i_from_jw_t;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(sqrt_information_j_.template cast<T>());
    return true;
  }

 private:
  const Rigid3d& i_from_j_;
  const EigenMatrix6d sqrt_information_j_;
};

// Cost function for aligning one 3D point with a reference 3D point with
// covariance. Convention is similar to colmap::Sim3d
// residual = scale_b_from_a * R_b_from_a * point_in_a + t_b_from_a -
// ref_point_in_b
struct Point3dAlignmentCostFunctor {
 public:
  Point3dAlignmentCostFunctor(const Eigen::Vector3d& ref_point,
                              const Eigen::Matrix3d& covariance_point)
      : ref_point_(ref_point),
        sqrt_information_point_(SqrtInformation(covariance_point)) {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& ref_point,
                                     const Eigen::Matrix3d& covariance_point) {
    return (
        new ceres::
            AutoDiffCostFunction<Point3dAlignmentCostFunctor, 3, 3, 4, 3, 1>(
                new Point3dAlignmentCostFunctor(ref_point, covariance_point)));
  }

  template <typename T>
  bool operator()(const T* const point,
                  const T* const transform_q,
                  const T* const transform_t,
                  const T* const scale,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> T_q = EigenQuaternionMap<T>(transform_q);
    const Eigen::Matrix<T, 3, 1> transform_point =
        T_q * EigenVector3Map<T>(point) * scale[0] +
        EigenVector3Map<T>(transform_t);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = transform_point - ref_point_.cast<T>();
    residuals.applyOnTheLeft(sqrt_information_point_.template cast<T>());
    return true;
  }

 private:
  const Eigen::Vector3d ref_point_;
  const Eigen::Matrix3d sqrt_information_point_;
};

// 3-DoF error on the cam to world position.
struct PositionPriorErrorCostFunctor {
 public:
  PositionPriorErrorCostFunctor(
      const Eigen::Vector3d& world_from_cam_position_prior,
      const Eigen::Matrix3d& covariance)
      : world_from_cam_position_prior_(world_from_cam_position_prior),
        sqrt_information_prior_(SqrtInformation(covariance)) {}

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& world_from_cam_position_prior,
      const Eigen::Matrix3d& covariance) {
    return (
        new ceres::AutoDiffCostFunction<PositionPriorErrorCostFunctor, 3, 4, 3>(
            new PositionPriorErrorCostFunctor(world_from_cam_position_prior,
                                              covariance)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_q,
                  const T* const cam_from_world_t,
                  T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = world_from_cam_position_prior_.cast<T>() +
                EigenQuaternionMap<T>(cam_from_world_q).inverse() *
                    EigenVector3Map<T>(cam_from_world_t);
    residuals.applyOnTheLeft(sqrt_information_prior_.template cast<T>());
    return true;
  }

 private:
  const Eigen::Vector3d world_from_cam_position_prior_;
  const Eigen::Matrix3d sqrt_information_prior_;
};

// A cost function that wraps another one and whiten its residuals with an
// isotropic covariance, i.e. assuming that the variance is identical in and
// independent between each dimension of the residual.
template <class CostFunctor>
class IsotropicNoiseCostFunctorWrapper {
  class LinearCostFunction : public ceres::CostFunction {
   public:
    explicit LinearCostFunction(const double s) : s_(s) {
      set_num_residuals(1);
      mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const final {
      *residuals = **parameters * s_;
      if (jacobians && *jacobians) {
        **jacobians = s_;
      }
      return true;
    }

   private:
    const double s_;
  };

 public:
  template <typename... Args>
  static ceres::CostFunction* Create(const double stddev, Args&&... args) {
    THROW_CHECK_GT(stddev, 0.0);
    ceres::CostFunction* cost_function =
        CostFunctor::Create(std::forward<Args>(args)...);
    const double scale = 1.0 / stddev;
    std::vector<ceres::CostFunction*> conditioners(
#if CERES_VERSION_MAJOR < 2
        cost_function->num_residuals());
    // Ceres <2.0 does not allow reusing the same conditioner multiple times.
    for (size_t i = 0; i < conditioners.size(); ++i) {
      conditioners[i] = new LinearCostFunction(scale);
    }
#else
        cost_function->num_residuals(), new LinearCostFunction(scale));
#endif
    return new ceres::ConditionedCostFunction(
        cost_function, conditioners, ceres::TAKE_OWNERSHIP);
  }
};

template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction* CameraCostFunction(const CameraModelId camera_model_id,
                                        Args&&... args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::model_id:                                             \
    return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

}  // namespace colmap
