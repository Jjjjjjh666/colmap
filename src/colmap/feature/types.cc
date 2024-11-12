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

#include "colmap/feature/types.h"

#include "colmap/util/logging.h"

namespace colmap {

FeatureKeypoint::FeatureKeypoint() : FeatureKeypoint(0, 0) {} //默认构造函数，将特征点初始坐标设置为（0,0）

FeatureKeypoint::FeatureKeypoint(const float x, const float y)
    : FeatureKeypoint(x, y, 1, 0, 0, 1) {}  //接收表示特征点的坐标（x，y），同时将尺度设置为1，方向设置为0

FeatureKeypoint::FeatureKeypoint(const float x_,
                                 const float y_,
                                 const float scale,
                                 const float orientation)
    : x(x_), y(y_) {
  THROW_CHECK_GE(scale, 0.0);  //检查尺度是否大于0
  const float scale_cos_orientation = scale * std::cos(orientation);  //根据尺度和方向计算矩阵中的元素
  const float scale_sin_orientation = scale * std::sin(orientation);
  a11 = scale_cos_orientation;
  a12 = -scale_sin_orientation;
  a21 = scale_sin_orientation;
  a22 = scale_cos_orientation;
}

/*
  a11, a12
  a21, a22
  构成一个表示旋转的矩阵
*/

FeatureKeypoint::FeatureKeypoint(const float x_,
                                 const float y_,
                                 const float a11_,
                                 const float a12_,
                                 const float a21_,
                                 const float a22_)
    : x(x_), y(y_), a11(a11_), a12(a12_), a21(a21_), a22(a22_) {}  //直接初始化成员变量

FeatureKeypoint FeatureKeypoint::FromShapeParameters(const float x,
                                                     const float y,
                                                     const float scale_x,
                                                     const float scale_y,
                                                     const float orientation,
                                                     const float shear) {
  THROW_CHECK_GE(scale_x, 0.0);
  THROW_CHECK_GE(scale_y, 0.0);
  return FeatureKeypoint(x,
                         y,
                         scale_x * std::cos(orientation),
                         -scale_y * std::sin(orientation + shear),
                         scale_x * std::sin(orientation),
                         scale_y * std::cos(orientation + shear));
} //根据给定的形状参数创建一个新的FeatureKeypoint对象

void FeatureKeypoint::Rescale(const float scale) { Rescale(scale, scale); } //使用相同的尺度参数进行缩放

void FeatureKeypoint::Rescale(const float scale_x, const float scale_y) {
  THROW_CHECK_GT(scale_x, 0);
  THROW_CHECK_GT(scale_y, 0);
  x *= scale_x;
  y *= scale_y;
  a11 *= scale_x;
  a12 *= scale_y;
  a21 *= scale_x;
  a22 *= scale_y;
}  //接收两个尺度参数对特征点的坐标和变换矩阵的元素进行缩放

float FeatureKeypoint::ComputeScale() const {
  return (ComputeScaleX() + ComputeScaleY()) / 2.0f;
}  //计算特征点的平均尺度

float FeatureKeypoint::ComputeScaleX() const {
  return std::sqrt(a11 * a11 + a21 * a21);
}  //计算特征点在x方向上的尺度，通过计算变换矩阵中对应元素平方和的平方根得到

float FeatureKeypoint::ComputeScaleY() const {
  return std::sqrt(a12 * a12 + a22 * a22);
}  //同上

float FeatureKeypoint::ComputeOrientation() const {
  return std::atan2(a21, a11);
}  //计算特征点的方向，通过反正切函数计算变换矩阵中两个元素的比值得到

float FeatureKeypoint::ComputeShear() const {
  return std::atan2(-a12, a22) - ComputeOrientation();
}  //计算特征点的剪切，通过计算两个反正切函数值的差得到。

}  // namespace colmap
//整体上定义了一个名为FeatureKeypoint的类，用于维护图像中的特征点，并对特征点进行构造，缩放和各种属性的计算
