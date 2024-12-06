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

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// 包含头文件，使用 Ceres 库进行优化

#include <cmath>

#include "ceres/ceres.h"

// 设置四元数流形，确保优化时四元数的参数化是单位四元数
inline void SetQuaternionManifold(ceres::Problem* problem, double* quat_xyzw) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(quat_xyzw, new ceres::EigenQuaternionManifold);  // 使用Eigen四元数流形
#else
  problem->SetParameterization(quat_xyzw,
                               new ceres::EigenQuaternionParameterization);  // 使用旧版四元数参数化
#endif
}

// 设置子集流形，用于在优化中固定部分参数
inline void SetSubsetManifold(int size,
                              const std::vector<int>& constant_params,
                              ceres::Problem* problem,
                              double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params,
                       new ceres::SubsetManifold(size, constant_params));  // 使用子集流形
#else
  problem->SetParameterization(
      params, new ceres::SubsetParameterization(size, constant_params));  // 使用旧版子集参数化
#endif
}

// 设置球面流形，通常用于单位向量或者旋转矩阵
template <int size>
inline void SetSphereManifold(ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params, new ceres::SphereManifold<size>);  // 使用球面流形
#else
  problem->SetParameterization(
      params, new ceres::HomogeneousVectorParameterization(size));  // 使用旧版同质向量参数化
#endif
}

// 使用指数函数来确保变量严格为正，适用于尺度参数
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
template <int AmbientSpaceDimension>
class PositiveExponentialManifold : public ceres::Manifold {
 public:
  static_assert(ceres::DYNAMIC == Eigen::Dynamic,
                "ceres::DYNAMIC needs to be the same as Eigen::Dynamic.");

  PositiveExponentialManifold() : size_{AmbientSpaceDimension} {}
  explicit PositiveExponentialManifold(int size) : size_{size} {
    if (AmbientSpaceDimension != Eigen::Dynamic) {
      CHECK_EQ(AmbientSpaceDimension, size)
          << "指定的模板参数大小与提供的大小不同。";
    } else {
      CHECK_GT(size_, 0)
          << "流形的大小必须是正整数。";
    }
  }

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    for (int i = 0; i < size_; ++i) {
      x_plus_delta[i] = x[i] * std::exp(delta[i]);  // 计算加法操作
    }
    return true;
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    for (int i = 0; i < size_; ++i) {
      jacobian[size_ * i + i] = x[i];  // 计算加法的雅可比矩阵
    }
    return true;
  }

  virtual bool Minus(const double* y,
                     const double* x,
                     double* y_minus_x) const override {
    for (int i = 0; i < size_; ++i) {
      y_minus_x[i] = std::log(y[i] / x[i]);  // 计算减法操作
    }
    return true;
  }

  virtual bool MinusJacobian(const double* x, double* jacobian) const override {
    for (int i = 0; i < size_; ++i) {
      jacobian[size_ * i + i] = 1.0 / x[i];  // 计算减法的雅可比矩阵
    }
    return true;
  }

  int AmbientSize() const override {
    return AmbientSpaceDimension == ceres::DYNAMIC ? size_
                                                   : AmbientSpaceDimension;
  }
  int TangentSize() const override { return AmbientSize(); }

 private:
  const int size_{};
};
#else
// 旧版实现，用于保证变量正值
class PositiveExponentialParameterization
    : public ceres::LocalParameterization {
 public:
  explicit PositiveExponentialParameterization(int size) : size_{size} {
    CHECK_GT(size_, 0)
        << "流形的大小必须是正整数。";
  }
  ~PositiveExponentialParameterization() {}

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    for (int i = 0; i < size_; ++i) {
      x_plus_delta[i] = x[i] * std::exp(delta[i]);  // 计算加法操作
    }
    return true;
  }

  bool ComputeJacobian(const double* x, double* jacobian) const override {
    for (int i = 0; i < size_; ++i) {
      jacobian[size_ * i + i] = x[i];  // 计算加法的雅可比矩阵
    }
    return true;
  }

  int GlobalSize() const override { return size_; }
  int LocalSize() const override { return size_; }

 private:
  const int size_{};
};

#endif

// 设置正指数流形，确保优化变量严格为正
template <int size>
inline void SetPositiveExponentialManifold(ceres::Problem* problem,
                                           double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params, new PositiveExponentialManifold<size>);  // 设置正指数流形
#else
  problem->SetParameterization(params,
                               new PositiveExponentialParameterization(size));  // 使用旧版正指数参数化
#endif
}

// 获取参数块的切空间大小
inline int ParameterBlockTangentSize(ceres::Problem* problem,
                                     const double* param) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  return problem->ParameterBlockTangentSize(param);  // 获取切空间大小
#else
  return problem->ParameterBlockLocalSize(param);  // 获取局部大小
#endif
}

}  // namespace colmap
