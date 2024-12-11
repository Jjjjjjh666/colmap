//euclidean_transform是欧几里得变换的意思，欧几里得变换包括旋转和平移，可能还包括缩放
//本文件模板类型EuclideanTransformEstimator，用于从源坐标系和目标坐标系的对应点对中估计这种变换。

//需要输入——对应点对：
//一组在源坐标系中的点。
//一组在目标坐标系中的点。
//这些点需要一一对应。

//输出结果——变换参数：
//旋转矩阵。
//平移向量。
//（如果考虑缩放）缩放因子。


#pragma once

#include "colmap/geometry/sim3.h"

namespace colmap {

// 从源坐标系和目标坐标系中的对应点对估计N维欧几里得变换的估计器。
//
// 该算法基于以下论文：
//
//      S. Umeyama. 最小二乘估计两个点模式之间的变换参数。
//      IEEE模式分析与机器智能汇刊，卷13，第4期，第376-380页，1991。
//      http://www.stanford.edu/class/cs273/refs/umeyama.pdf
//
// 并使用了Eigen实现。
template <int kDim>
using EuclideanTransformEstimator = SimilarityTransformEstimator<kDim, false>;

}  // namespace colmap