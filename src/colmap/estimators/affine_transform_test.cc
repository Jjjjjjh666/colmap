
#include "colmap/estimators/affine_transform.h"

#include "colmap/util/eigen_alignment.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(AffineTransform, Nominal) {
  for (int x = 0; x < 10; ++x) {
    Eigen::Matrix<double, 2, 3> A;
    A << x / 10.0, 0.2, 0.3, 30, 0.2, 0.1;

    std::vector<Eigen::Vector2d> src;
    src.emplace_back(x, 0);
    src.emplace_back(1, 0);
    src.emplace_back(2, 1);

    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < 3; ++i) {
      dst.push_back(A * src[i].homogeneous());
    }

    AffineTransformEstimator estimator;
    std::vector<Eigen::Matrix<double, 2, 3>> models;
    estimator.Estimate(src, dst, &models);

    ASSERT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    EXPECT_EQ(residuals.size(), 3);

    for (size_t i = 0; i < 3; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

}  // namespace
}  // namespace colmap
