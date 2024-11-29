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

#include "colmap/feature/sift.h"

#include "colmap/feature/utils.h"
#include "colmap/math/math.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/opengl_utils.h"

#if defined(COLMAP_GPU_ENABLED)
#include "thirdparty/SiftGPU/SiftGPU.h"
#if !defined(COLMAP_GUI_ENABLED)
// GLEW symbols are already defined by Qt.
#include <GL/glew.h>
#endif // COLMAP_GUI_ENABLED
#endif // COLMAP_GPU_ENABLED
#include "colmap/util/eigen_alignment.h"

#include "thirdparty/VLFeat/covdet.h"
#include "thirdparty/VLFeat/sift.h"

#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>

#include <Eigen/Geometry>

namespace colmap
{

    // SIFT descriptors are normalized to length 512 (w/ quantization errors).
    constexpr int kSqSiftDescriptorNorm = 512 * 512;

    bool SiftExtractionOptions::Check() const
    {
        if (use_gpu)
        {
            CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
        }
        CHECK_OPTION_GT(max_image_size, 0);
        CHECK_OPTION_GT(max_num_features, 0);
        CHECK_OPTION_GT(octave_resolution, 0);
        CHECK_OPTION_GT(peak_threshold, 0.0);
        CHECK_OPTION_GT(edge_threshold, 0.0);
        CHECK_OPTION_GT(max_num_orientations, 0);
        if (domain_size_pooling)
        {
            CHECK_OPTION_GT(dsp_min_scale, 0);
            CHECK_OPTION_GE(dsp_max_scale, dsp_min_scale);
            CHECK_OPTION_GT(dsp_num_scales, 0);
        }
        return true;
    }

    bool SiftMatchingOptions::Check() const
    {
        if (use_gpu)
        {
            CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
        }
        CHECK_OPTION_GT(max_ratio, 0.0);
        CHECK_OPTION_GT(max_distance, 0.0);
        CHECK_OPTION_GT(max_num_matches, 0);
        return true;
    }
    // 以上两个函数主要是确保特征匹配和提取时，配置选项的合法性

    namespace
    {

        void WarnDarknessAdaptivityNotAvailable()
        {
            LOG(WARNING) << "Darkness adaptivity only available for GLSL SiftGPU.";
        }

        // VLFeat uses a different convention to store its descriptors. This transforms
        // the VLFeat format into the original SIFT format that is also used by SiftGPU.
        TransformVLFeatToUBCFeatureDescriptors(
            const FeatureDescriptors &vlfeat_descriptors)
        { // 将输入的基于VLFeat的特征描述符转化为UBC的特征描述符
            FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
                                               vlfeat_descriptors.cols());  //创建了一个与输入矩阵大小相同的空矩阵，用于存储转换后的结果
            const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};//用于交换位置的索引映射表，描述符中的8个子维度被以反转顺序重新排列
            for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n)  //遍历每个特征点的描述符
            {
                for (int i = 0; i < 4; ++i)
                {
                    for (int j = 0; j < 4; ++j)  //对描述符的每个4*4小块进行重新排列
                    {
                        for (int k = 0; k < 8; ++k)
                        {
                            ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
                                vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
                            // 将输入特征描述符 vlfeat_descriptors 中的特定位置的值，
                            //  按照 q 数组的顺序重新赋值给输出特征描述符 ubc_descriptors 的相应位置。
                        }
                    }
                }
            }
            return ubc_descriptors;
        }
     //代码的主要功能是将输入的基于 VLFeat 格式的特征描述符转换为 UBC (University of British Columbia) 格式的特征描述符。
     //通过q数组的反转，完成子维度的重新排列，实现格式转换

        FeatureDescriptors

            class SiftCPUFeatureExtractor : public FeatureExtractor
        {
            // SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）特征提取
        public:
            using VlSiftType = std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt *)>;
            // 定义了一个名为VlSiftType的类型别名，它代表一个std::unique_ptr智能指针，用于管理VlSiftFilt类型的对象。
            // 便于管理VlSiftType的生命周期

            explicit SiftCPUFeatureExtractor(const SiftExtractionOptions &options)
                : options_(options), sift_(nullptr, &vl_sift_delete)
            {
                THROW_CHECK(options_.Check());
                THROW_CHECK(!options_.estimate_affine_shape);
                THROW_CHECK(!options_.domain_size_pooling);
                if (options_.darkness_adaptivity)
                {
                    WarnDarknessAdaptivityNotAvailable();
                }
            }
            // 构造函数并确保某些特定的选项设置符合要求

            static std::unique_ptr<FeatureExtractor> Create(
                const SiftExtractionOptions &options)
            {
                return std::make_unique<SiftCPUFeatureExtractor>(options);
                // make_unique” 通常是 C++14 引入的一个函数模板，用于创建一个指向动态分配对象的 std::unique_ptr。
                // 它可以方便地创建一个独占所有权的智能指针，避免手动管理资源的复杂性，确保资源在适当的时候被正确释放
            }

            bool Extract(const Bitmap &bitmap, // bitmap表示输入图像
                         FeatureKeypoints *keypoints,
                         FeatureDescriptors *descriptors)
            {
                THROW_CHECK(bitmap.IsGrey());
                THROW_CHECK_NOTNULL(keypoints); // 检查输入图像是否灰质和特征描述符是否非空

                // 如果sift_智能指针为空，或者sift_所指向的对象的宽度和高度与输入图像的宽度和高度不匹配，那么创建一个新的VlSiftFilt对象
                if (sift_ == nullptr || sift_->width != bitmap.Width() ||
                    sift_->height != bitmap.Height())
                {
                    sift_ = VlSiftType(vl_sift_new(bitmap.Width(),
                                                   bitmap.Height(),
                                                   options_.num_octaves,
                                                   options_.octave_resolution,
                                                   options_.first_octave),
                                       &vl_sift_delete);
                    if (!sift_)
                    {
                        return false;
                    }
                }

                vl_sift_set_peak_thresh(sift_.get(), options_.peak_threshold);
                vl_sift_set_edge_thresh(sift_.get(), options_.edge_threshold); // 设置峰值阈值和边缘阈值

                // Iterate through octaves.
                std::vector<size_t> level_num_features;
                std::vector<FeatureKeypoints> level_keypoints;
                std::vector<FeatureDescriptors> level_descriptors; // 初始化用于存储不同尺度信息的容器，特征数量，特征关键点，特征描述符
                bool first_octave = true;
                while (true)
                {
                    if (first_octave)
                    { // 如果是第一个尺度
                        const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
                        // uint8_t” 是一种数据类型，通常在 C、C++ 等编程语言中使用。它代表无符号的 8 位整数类型，取值范围是 0 到 255。
                        std::vector<float> data_float(data_uint8.size());
                        for (size_t i = 0; i < data_uint8.size(); ++i)
                        {
                            data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
                        }
                        // 将输入图像转化为浮点型数据
                        if (vl_sift_process_first_octave(sift_.get(), data_float.data()))
                        {
                            break;
                        }
                        first_octave = false;
                    }
                    else
                    {
                        if (vl_sift_process_next_octave(sift_.get()))
                        {
                            break;
                        }
                    }

                    // Detect keypoints.
                    vl_sift_detect(sift_.get()); // 检测当前尺度的特征点

                    // Extract detected keypoints.
                    const VlSiftKeypoint *vl_keypoints = vl_sift_get_keypoints(sift_.get()); // 获取当前尺度检测到的特征点
                    const int num_keypoints = vl_sift_get_nkeypoints(sift_.get());           // 检查特征点数量是否为0，若为0，则处理下一个尺度
                    if (num_keypoints == 0)
                    {
                        continue;
                    }

                    // Extract features with different orientations per DOG level.
                    size_t level_idx = 0;                 // 记录当前层级索引
                    int prev_level = -1;                  // 记录上一个处理的层级索引
                    FeatureDescriptorsFloat desc(1, 128); // 创建特征描述符对象
                    for (int i = 0; i < num_keypoints; ++i)
                    { // 遍历所有特征点
                        if (vl_keypoints[i].is != prev_level)
                        { // 若当前特征点和上个处理的层级不同
                            if (i > 0)
                            { // 若不是第一个特征点
                                // Resize containers of previous DOG level.
                                level_keypoints.back().resize(level_idx);
                                if (descriptors != nullptr)
                                {
                                    level_descriptors.back().conservativeResize(level_idx, 128);
                                }
                            }

                            // Add containers for new DOG level.  //为新的层级添加容器
                            level_idx = 0;
                            level_num_features.push_back(0); // 重置参数
                            level_keypoints.emplace_back(options_.max_num_orientations *
                                                         num_keypoints);
                            if (descriptors != nullptr)
                            {
                                level_descriptors.emplace_back(
                                    options_.max_num_orientations * num_keypoints, 128);
                            }
                            // 添加描述符和特征点集合
                        }

                        level_num_features.back() += 1;  // 更新当前层级的特征点数量
                        prev_level = vl_keypoints[i].is; // 更新prev为当前层级标识

                        // Extract feature orientations.  //提取特征点方向
                        double angles[4];     // 存储方向角度
                        int num_orientations; // 存储方向数量
                        // 如果配置选项中的upright为真，则将方向数量设置为 1，角度设置为 0.0。
                        // 否则，调用vl_sift_calc_keypoint_orientations函数计算当前特征点的方向数量和角度
                        if (options_.upright)
                        {
                            num_orientations = 1;
                            angles[0] = 0.0;
                        }
                        else
                        {
                            num_orientations = vl_sift_calc_keypoint_orientations(
                                sift_.get(), angles, &vl_keypoints[i]);
                        }

                        // Note that this is different from SiftGPU, which selects the top
                        // global maxima as orientations while this selects the first two
                        // local maxima. It is not clear which procedure is better.
                        const int num_used_orientations =
                            std::min(num_orientations, options_.max_num_orientations);

                        for (int o = 0; o < num_used_orientations; ++o)
                        { // 遍历实际使用的方向数量
                            level_keypoints.back()[level_idx] =
                                FeatureKeypoint(vl_keypoints[i].x + 0.5f,
                                                vl_keypoints[i].y + 0.5f,
                                                vl_keypoints[i].sigma,
                                                angles[o]); // 创建一个新的FeatureKeypoint对象，并将其添加到当前层级的特征点集合中
                            if (descriptors != nullptr)
                            {
                                // 若特征描述符不为空 调用vl_sift_calc_keypoint_descriptor计算特征描述符，
                                // 然后根据配置选项进行归一化处理（L2 归一化或 L1_ROOT 归一化），最后将归一化后的描述符转换为无符号字节类型并添加到当前层级的特征描述符集合中。
                                vl_sift_calc_keypoint_descriptor(
                                    sift_.get(), desc.data(), &vl_keypoints[i], angles[o]);
                                if (options_.normalization ==
                                    SiftExtractionOptions::Normalization::L2)
                                {
                                    L2NormalizeFeatureDescriptors(&desc);
                                }
                                else if (options_.normalization ==
                                         SiftExtractionOptions::Normalization::L1_ROOT)
                                {
                                    L1RootNormalizeFeatureDescriptors(&desc);
                                }
                                else
                                {
                                    LOG(FATAL_THROW) << "Normalization type not supported";
                                }

                                level_descriptors.back().row(level_idx) =
                                    FeatureDescriptorsToUnsignedByte(desc);
                            }

                            level_idx += 1; // 更新层级索引
                        }
                    }

                    // Resize containers for last DOG level in octave.
                    // 最后处理最后一个尺度的容器大小
                    level_keypoints.back().resize(level_idx);
                    if (descriptors != nullptr)
                    {
                        level_descriptors.back().conservativeResize(level_idx, 128);
                    }
                }

                // Determine how many DOG levels to keep to satisfy max_num_features option.
                // 确定需要保留的尺度数量
                int first_level_to_keep = 0;
                int num_features = 0;
                int num_features_with_orientations = 0;
                for (int i = level_keypoints.size() - 1; i >= 0; --i)
                {
                    num_features += level_num_features[i];
                    num_features_with_orientations += level_keypoints[i].size();
                    if (num_features > options_.max_num_features)
                    {
                        first_level_to_keep = i; // 当总特征数量超过配置选项中的最大特征数量时，确定第一个需要保留的尺度索引。
                        break;
                    }
                }

                // Extract the features to be kept.  //提取需要保留的特征点
                {
                    size_t k = 0; // 记录特征点数量
                    keypoints->resize(num_features_with_orientations);
                    // 根据确定的第一个需要保留的尺度索引，遍历相应的尺度，将特征点存储在 keypoints 指针所指向的容器中
                    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i)
                    {
                        for (size_t j = 0; j < level_keypoints[i].size(); ++j)
                        {
                            (*keypoints)[k] = level_keypoints[i][j];
                            k += 1;
                        }
                    }
                }

                // Compute the descriptors for the detected keypoints.
                if (descriptors != nullptr)
                {
                    size_t k = 0;
                    descriptors->resize(num_features_with_orientations, 128);
                    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i)
                    {
                        for (size_t j = 0; j < level_keypoints[i].size(); ++j)
                        {
                            descriptors->row(k) = level_descriptors[i].row(j);
                            k += 1;
                        }
                    }
                    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors); // 将特征描述符转化为特定格式
                }

                return true;
            }

        private:
            const SiftExtractionOptions options_;
            VlSiftType sift_;
        };
        // 这段代码实现了在 CPU 上进行 SIFT 特征提取的功能。通过创建 VlSiftFilt 对象，
        // 处理不同尺度的图像，检测特征点，提取特征点的方向和描述符，并根据配置选项进行归一化和格式转换等操作，最终返回提取的特征点和描述符。

        class CovariantSiftCPUFeatureExtractor : public FeatureExtractor
        {
        public:
            explicit CovariantSiftCPUFeatureExtractor( // 构造函数
                const SiftExtractionOptions &options)
                : options_(options)
            {
                THROW_CHECK(options_.Check());
                if (options_.darkness_adaptivity)
                {
                    WarnDarknessAdaptivityNotAvailable();
                }
            }

            static std::unique_ptr<FeatureExtractor> Create(
                const SiftExtractionOptions &options)
            {
                return std::make_unique<CovariantSiftCPUFeatureExtractor>(options);
            }
            // 通过std::make_unique创建一个CovariantSiftCPUFeatureExtractor对象并返回，返回智能指针

            bool Extract(const Bitmap &bitmap,
                         FeatureKeypoints *keypoints,
                         FeatureDescriptors *descriptors)
            {
                THROW_CHECK(bitmap.IsGrey());   // 检查是否为灰度对象
                THROW_CHECK_NOTNULL(keypoints); // 检查指针是否为空

                // Setup covariant SIFT detector.
                // 创建检测器
                std::unique_ptr<VlCovDet, void (*)(VlCovDet *)> covdet(
                    vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
                if (!covdet)
                {
                    return false;
                }

                const int kMaxOctaveResolution = 1000;
                THROW_CHECK_LE(options_.octave_resolution, kMaxOctaveResolution);

                vl_covdet_set_first_octave(covdet.get(), options_.first_octave);           // 设置第一个八度参数
                vl_covdet_set_octave_resolution(covdet.get(), options_.octave_resolution); // 设置八度分辨率
                vl_covdet_set_peak_threshold(covdet.get(), options_.peak_threshold);       // 设置峰值阈值
                vl_covdet_set_edge_threshold(covdet.get(), options_.edge_threshold);       // 设置边缘阈值

                // 创建一个局部作用域
                {
                    const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray(); // 调用bitmap.ConvertToRowMajorArray()将bitmap对象转换为一个std::vector<uint8_t>类型的向量
                    std::vector<float> data_float(data_uint8.size());
                    // 通过循环将data_uint8中的每个元素转换为float类型，并除以255.0f进行归一化处理，存储到data_float中。
                    for (size_t i = 0; i < data_uint8.size(); ++i)
                    {
                        data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
                    }
                    // 调用vl_covdet_put_image将处理后的图像数据data_float以及图像的宽度（bitmap.Width()）和高度（bitmap.Height()）传递给协变 SIFT 检测器。
                    vl_covdet_put_image(
                        covdet.get(), data_float.data(), bitmap.Width(), bitmap.Height());
                }

                // 进行特征检测
                vl_covdet_detect(covdet.get(), options_.max_num_features);

                if (!options_.upright)
                {
                    if (options_.estimate_affine_shape)
                    {
                        vl_covdet_extract_affine_shape(covdet.get()); // 提取仿射形状
                    }
                    else
                    {
                        vl_covdet_extract_orientations(covdet.get()); // 提取方向
                    }
                }

                // 获取协变sift检测器检测到的特征数量和指向特征数组的指针
                const int num_features = vl_covdet_get_num_features(covdet.get());
                VlCovDetFeature *features = vl_covdet_get_features(covdet.get());

                // Sort features according to detected octave and scale.
                // 根据检测到的八度和尺度对特征进行排序
                std::sort(
                    features,
                    features + num_features, // 对整个特征数组进行排序
                    [](const VlCovDetFeature &feature1, const VlCovDetFeature &feature2)
                    {
                        // 若两个特征的八度相同，则按照尺度排序，若八度不同则直接按照八度排序
                        if (feature1.o == feature2.o)
                        {
                            return feature1.s > feature2.s;
                        }
                        else
                        {
                            return feature1.o > feature2.o;
                        }
                    });

                const size_t max_num_features =
                    static_cast<size_t>(options_.max_num_features); // 获取特征数量

                // Copy detected keypoints and clamp when maximum number of features
                // reached.
                // 复制检测到的关键点并在达到最大特征数量时进行截断
                int prev_octave_scale_idx = std::numeric_limits<int>::max(); // 用于跟踪上一个特征的八度和尺度的组合索引
                for (int i = 0; i < num_features; ++i)
                {
                    // 创建一个FeatureKeypoint类型的对象keypoint，并从当前特征的frame结构中复制坐标和仿射变换矩阵的元素。
                    FeatureKeypoint keypoint;
                    keypoint.x = features[i].frame.x + 0.5;
                    keypoint.y = features[i].frame.y + 0.5;
                    keypoint.a11 = features[i].frame.a11;
                    keypoint.a12 = features[i].frame.a12;
                    keypoint.a21 = features[i].frame.a21;
                    keypoint.a22 = features[i].frame.a22;
                    keypoints->push_back(keypoint);
                    // 计算当前特征的八度和尺度的组合索引octave_scale_idx
                    const int octave_scale_idx =
                        features[i].o * kMaxOctaveResolution + features[i].s;
                    THROW_CHECK_LE(octave_scale_idx, prev_octave_scale_idx); // 检查当前特征的组合索引是否小于等于上个特征的组合索引
                    // 如果当前特征的组合索引与上一个特征不同，并且已经达到了最大特征数量max_num_features，则跳出循环
                    if (octave_scale_idx != prev_octave_scale_idx &&
                        keypoints->size() >= max_num_features)
                    {
                        break;
                    }
                    // 更新prev_octave_scale_idx为当前特征的组合索引
                    prev_octave_scale_idx = octave_scale_idx;
                }

                // Compute the descriptors for the detected keypoints.
                // 这段代码的主要功能是为给定的关键点计算特征描述符
                if (descriptors != nullptr)
                {
                    descriptors->resize(keypoints->size(), 128); // 容器容量调整
                    // 定义补丁的各项数据
                    const size_t kPatchResolution = 15;                                // 分辨率
                    const size_t kPatchSide = 2 * kPatchResolution + 1;                // 边长
                    const double kPatchRelativeExtent = 7.5;                           // 相对范围
                    const double kPatchRelativeSmoothing = 1;                          // 平滑度
                    const double kPatchStep = kPatchRelativeExtent / kPatchResolution; // 步长
                    const double kSigma =
                        kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep; // 标准差

                    std::vector<float> patch(kPatchSide * kPatchSide);
                    std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

                    float dsp_min_scale = 1;
                    float dsp_scale_step = 0;
                    int dsp_num_scales = 1;
                    // 根据选项中的domain_size_pooling标志，设置不同的尺度参数
                    if (options_.domain_size_pooling)
                    {
                        dsp_min_scale = options_.dsp_min_scale;
                        dsp_scale_step = (options_.dsp_max_scale - options_.dsp_min_scale) /
                                         options_.dsp_num_scales;
                        dsp_num_scales = options_.dsp_num_scales;
                    }

                    FeatureDescriptorsFloat descriptor(1, 128);                      // 单尺度描述符
                    FeatureDescriptorsFloat scaled_descriptors(dsp_num_scales, 128); // 多尺度描述符

                    // 使用std::unique_ptr智能指针创建一个VlSiftFilt类型的对象sift
                    std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt *)> sift(
                        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
                    if (!sift)
                    {
                        return false;
                    }
                    // 设置sift对象的放大系数为3.0
                    vl_sift_set_magnif(sift.get(), 3.0);

                    // 外层循环遍历每个关键点，内层循环遍历不同的尺度。
                    for (size_t i = 0; i < keypoints->size(); ++i)
                    {
                        for (int s = 0; s < dsp_num_scales; ++s)
                        {
                            const double dsp_scale = dsp_min_scale + s * dsp_scale_step;
                            // 根据当前尺度调整关键点的框架
                            VlFrameOrientedEllipse scaled_frame = features[i].frame;
                            scaled_frame.a11 *= dsp_scale;
                            scaled_frame.a12 *= dsp_scale;
                            scaled_frame.a21 *= dsp_scale;
                            scaled_frame.a22 *= dsp_scale;
                            // 使用协变 SIFT 检测器提取当前尺度下的补丁数据，并存储在patch向量中
                            vl_covdet_extract_patch_for_frame(covdet.get(),
                                                              patch.data(),
                                                              kPatchResolution,
                                                              kPatchRelativeExtent,
                                                              kPatchRelativeSmoothing,
                                                              scaled_frame);
                            // 计算补丁数据的极坐标梯度，并存储在patchXY向量中
                            vl_imgradient_polar_f(patchXY.data(),
                                                  patchXY.data() + 1,
                                                  2,
                                                  2 * kPatchSide,
                                                  patch.data(),
                                                  kPatchSide,
                                                  kPatchSide,
                                                  kPatchSide);
                            // 使用sift对象计算当前尺度下的原始特征描述符，并存储在scaled_descriptors中。
                            vl_sift_calc_raw_descriptor(sift.get(),
                                                        patchXY.data(),
                                                        scaled_descriptors.row(s).data(),
                                                        kPatchSide,
                                                        kPatchSide,
                                                        kPatchResolution,
                                                        kPatchResolution,
                                                        kSigma,
                                                        0);
                        }
                        // 根据选项中的域大小池化标志，选择不同的方式来处理多尺度描述符。
                        //  如果启用了域大小池化，计算多尺度描述符的列均值作为最终描述符；否则，直接使用多尺度描述符
                        if (options_.domain_size_pooling)
                        {
                            descriptor = scaled_descriptors.colwise().mean();
                        }
                        else
                        {
                            descriptor = scaled_descriptors;
                        }

                        THROW_CHECK_EQ(descriptor.cols(), 128);
                        // 根据选项中的归一化类型，对描述符进行相应的归一化处理
                        if (options_.normalization ==
                            SiftExtractionOptions::Normalization::L2)
                        {
                            L2NormalizeFeatureDescriptors(&descriptor);
                        }
                        else if (options_.normalization ==
                                 SiftExtractionOptions::Normalization::L1_ROOT)
                        {
                            L1RootNormalizeFeatureDescriptors(&descriptor);
                        }
                        else
                        {
                            LOG(FATAL_THROW) << "Normalization type not supported";
                        }
                        // 将归一化后的描述符转换为无符号字节类型，并存储在descriptors所指向的容器中，对应每个关键点的位置
                        descriptors->row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
                    }

                    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
                }

                return true;
            }

        private:
            const SiftExtractionOptions options_;
        };

#if defined(COLMAP_GPU_ENABLED)
        // 如果宏COLMAP_GPU_ENABLED被定义，则声明一个静态的std::map，键为整数类型，值为指向std::mutex的智能指针。
        //  这个map的目的是确保在同一时间只有一个线程在同一台 GPU 上进行特征提取或匹配，因为SiftGPU内部使用了静态变量，可能会导致多线程冲突。
        //  Mutexes that ensure that only one thread extracts/matches on the same GPU
        //  at the same time, since SiftGPU internally uses static variables.
        static std::map<int, std::unique_ptr<std::mutex>> sift_gpu_mutexes_;

        class SiftGPUFeatureExtractor : public FeatureExtractor
        {
        public:
            // explicit关键字表明这个构造函数不能进行隐式类型转换
            explicit SiftGPUFeatureExtractor(const SiftExtractionOptions &options)
                : options_(options)
            {
                THROW_CHECK(options_.Check());
                THROW_CHECK(!options_.estimate_affine_shape);
                THROW_CHECK(!options_.domain_size_pooling);
            }

            static std::unique_ptr<FeatureExtractor> Create(
                const SiftExtractionOptions &options)
            {
                // SiftGPU uses many global static state variables and the initialization
                // must be thread-safe in order to work correctly. This is enforced here.
                // SiftGPU使用了许多全局静态状态变量，为了正确工作，初始化必须是线程安全的，这里通过一个静态的互斥锁来实现。
                static std::mutex mutex;
                std::lock_guard<std::mutex> lock(mutex);
                // 将选项中的 GPU 索引字符串转换为整数向量gpu_indices
                std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
                THROW_CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

                std::vector<std::string> sift_gpu_args;

                sift_gpu_args.push_back("./sift_gpu");
                // 将"./sift_gpu"字符串添加到sift_gpu_args中

#if defined(COLMAP_CUDA_ENABLED)
                // Use CUDA version by default if darkness adaptivity is disabled.
                // 如果黑暗适应功能未开启（!options.darkness_adaptivity）且 GPU 索引小于 0，则将 GPU 索引设为 0。
                if (!options.darkness_adaptivity && gpu_indices[0] < 0)
                {
                    gpu_indices[0] = 0;
                }

                // 如果 GPU 索引大于等于 0，则向sift_gpu_args中添加"-cuda"和 GPU 索引的字符串表示，以指定使用 CUDA 版本的SiftGPU。
                if (gpu_indices[0] >= 0)
                {
                    sift_gpu_args.push_back("-cuda");
                    sift_gpu_args.push_back(std::to_string(gpu_indices[0]));
                }
#endif // COLMAP_CUDA_ENABLED

                // Darkness adaptivity (hidden feature). Significantly improves
                // distribution of features. Only available in GLSL version.
                // 如果 GPU 索引大于等于 0，则调用WarnDarknessAdaptivityNotAvailable函数发出警告
                // 因为黑暗适应功能只在 GLSL 版本中可用
                if (options.darkness_adaptivity)
                {
                    if (gpu_indices[0] >= 0)
                    {
                        WarnDarknessAdaptivityNotAvailable();
                    }
                    // 将"-da"添加到sift_gpu_args中，以启用黑暗适应功能
                    sift_gpu_args.push_back("-da");
                }

                // No verbose logging.
                // 将"-v"和"0"添加到sift_gpu_args中，以禁用详细日志记录
                sift_gpu_args.push_back("-v");
                sift_gpu_args.push_back("0");

                // Set maximum image dimension.
                // Note the max dimension of SiftGPU is the maximum dimension of the
                // first octave in the pyramid (which is the 'first_octave').
                // 计算补偿因子compensation_factor，用于设置SiftGPU的最大图像维度。最大维度是金字塔中第一个八度的最大维度，由选项中的first_octave决定。
                // 将"-maxd"和最大图像维度的字符串表示添加到sift_gpu_args中
                const int compensation_factor = 1 << -std::min(0, options.first_octave);
                sift_gpu_args.push_back("-maxd");
                sift_gpu_args.push_back(
                    std::to_string(options.max_image_size * compensation_factor));

                // Keep the highest level features.
                // 将"-tc2"和最大特征数量的字符串表示添加到sift_gpu_args中，以保留最高级别的特征。
                sift_gpu_args.push_back("-tc2");
                sift_gpu_args.push_back(std::to_string(options.max_num_features));

                // First octave level.
                // 将"-fo"和第一个八度级别（options.first_octave）的字符串表示添加到sift_gpu_args中
                sift_gpu_args.push_back("-fo");
                sift_gpu_args.push_back(std::to_string(options.first_octave));

                // Number of octave levels.
                // 将"-d"和八度分辨率（options.octave_resolution）的字符串表示添加到sift_gpu_args中
                sift_gpu_args.push_back("-d");
                sift_gpu_args.push_back(std::to_string(options.octave_resolution));

                // Peak threshold.
                // 将"-t"和峰值阈值（options.peak_threshold）的字符串表示添加到sift_gpu_args中
                sift_gpu_args.push_back("-t");
                sift_gpu_args.push_back(std::to_string(options.peak_threshold));

                // Edge threshold.
                // 将"-e"和边缘阈值（options.edge_threshold）的字符串表示添加到sift_gpu_args中
                sift_gpu_args.push_back("-e");
                sift_gpu_args.push_back(std::to_string(options.edge_threshold));

                // 如果选项中开启了直立特征（options.upright）：
                // 将"-ofix"添加到sift_gpu_args中，以固定方向为 0。
                // 将"-mo"和"1"添加到sift_gpu_args中，以设置最大方向数为 1
                if (options.upright)
                {
                    // Fix the orientation to 0 for upright features.
                    sift_gpu_args.push_back("-ofix");
                    // Maximum number of orientations.
                    sift_gpu_args.push_back("-mo");
                    sift_gpu_args.push_back("1");
                }
                // 否则，将"-mo"和最大方向数（options.max_num_orientations）的字符串表示添加到sift_gpu_args中
                else
                {
                    // Maximum number of orientations.
                    sift_gpu_args.push_back("-mo");
                    sift_gpu_args.push_back(std::to_string(options.max_num_orientations));
                }

                // 创建一个std::vector<const char*>类型的向量sift_gpu_args_cstr，用于存储sift_gpu_args中字符串的常量字符指针。
                // 遍历sift_gpu_args，将每个字符串的常量字符指针添加到sift_gpu_args_cstr中
                std::vector<const char *> sift_gpu_args_cstr;
                sift_gpu_args_cstr.reserve(sift_gpu_args.size());
                for (const auto &arg : sift_gpu_args)
                {
                    sift_gpu_args_cstr.push_back(arg.c_str());
                }
                // 创建一个SiftGPUFeatureExtractor对象的智能指针extractor，使用传入的选项参数进行构造
                auto extractor = std::make_unique<SiftGPUFeatureExtractor>(options);

                // Note that the SiftGPU object is not movable (for whatever reason).
                // If we instead create the object here and move it to the constructor, the
                // program segfaults inside SiftGPU.
                // 注意到SiftGPU对象不可移动，所以不能在构造函数中创建并移动它。
                // 这里调用extractor所指向的对象的sift_gpu_成员的ParseParam函数，传入参数个数和参数指针数组，以解析参数
                extractor->sift_gpu_.ParseParam(sift_gpu_args_cstr.size(),
                                                sift_gpu_args_cstr.data());
                // 将extractor所指向的对象的sift_gpu_成员的gpu_index设置为gpu_indices[0]
                // 如果sift_gpu_mutexes_中没有对应gpu_indices[0]的互斥锁，则创建一个并添加到sift_gpu_mutexes_中
                extractor->sift_gpu_.gpu_index = gpu_indices[0];
                if (sift_gpu_mutexes_.count(gpu_indices[0]) == 0)
                {
                    sift_gpu_mutexes_.emplace(gpu_indices[0], std::make_unique<std::mutex>());
                }
                // 如果extractor所指向的对象的sift_gpu_成员的VerifyContextGL函数返回值不是SiftGPU::SIFTGPU_FULL_SUPPORTED
                // 表示 OpenGL 上下文不被完全支持，则返回nullptr。
                if (extractor->sift_gpu_.VerifyContextGL() !=
                    SiftGPU::SIFTGPU_FULL_SUPPORTED)
                {
                    return nullptr;
                }

                return extractor;
            }

            bool Extract(const Bitmap &bitmap,
                         FeatureKeypoints *keypoints,
                         FeatureDescriptors *descriptors) override
            {
                THROW_CHECK(bitmap.IsGrey());
                THROW_CHECK_NOTNULL(keypoints);
                THROW_CHECK_NOTNULL(descriptors);

                // Note the max dimension of SiftGPU is the maximum dimension of the
                // first octave in the pyramid (which is the 'first_octave').
                // 计算一个补偿因子，并检查经过补偿后的最大图像尺寸是否与SIFT GPU的最大维度相匹配。如果不匹配，则抛出异常
                const int compensation_factor = 1 << -std::min(0, options_.first_octave);
                THROW_CHECK_EQ(options_.max_image_size * compensation_factor,
                               sift_gpu_.GetMaxDimension());

                std::lock_guard<std::mutex> lock(*sift_gpu_mutexes_[sift_gpu_.gpu_index]);

                // Note, that this produces slightly different results than using SiftGPU
                // directly for RGB->GRAY conversion, since it uses different weights.
                // 将位图转换为SIFT GPU可以处理的格式
                const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits(); // 将bitmap转化为原始位
                // 调用SIFT GPU的RunSIFT方法来运行SIFT算法。传递了位图的宽度（Pitch）、高度、原始位数据以及两个OpenGL常量来表示图像格式
                const int code = sift_gpu_.RunSIFT(bitmap.Pitch(),
                                                   bitmap.Height(),
                                                   bitmap_raw_bits.data(),
                                                   GL_LUMINANCE,
                                                   GL_UNSIGNED_BYTE);
                // 检查sift算法是否成功运行
                const int kSuccessCode = 1;
                if (code != kSuccessCode)
                {
                    return false;
                }
                // 获取sift算法提取的特征点数量
                const size_t num_features = static_cast<size_t>(sift_gpu_.GetFeatureNum());

                keypoints_buffer_.resize(num_features);

                FeatureDescriptorsFloat descriptors_float(num_features, 128);

                // Download the extracted keypoints and descriptors.
                // 从SIFT GPU获取特征点和描述符的数据
                sift_gpu_.GetFeatureVector(keypoints_buffer_.data(),
                                           descriptors_float.data());
                // 调整容器大小
                keypoints->resize(num_features);
                // 遍历所有特征点，并使用keypoints_buffer_中的数据创建新的FeatureKeypoint对象
                for (size_t i = 0; i < num_features; ++i)
                {
                    (*keypoints)[i] = FeatureKeypoint(keypoints_buffer_[i].x,
                                                      keypoints_buffer_[i].y,
                                                      keypoints_buffer_[i].s,
                                                      keypoints_buffer_[i].o);
                }

                // Save and normalize the descriptors.处理描述符的归一化
                // 根据options_中的设置，使用L2归一化或L1根归一化来处理描述符。若设置了不支持的归一化处理，则记录错误并抛出异常
                if (options_.normalization == SiftExtractionOptions::Normalization::L2)
                {
                    L2NormalizeFeatureDescriptors(&descriptors_float);
                }
                else if (options_.normalization ==
                         SiftExtractionOptions::Normalization::L1_ROOT)
                {
                    L1RootNormalizeFeatureDescriptors(&descriptors_float);
                }
                else
                {
                    LOG(FATAL_THROW) << "Normalization type not supported";
                }
                // 将浮点型描述符转换为无符号字节格式，并将结果存储在descriptors指针指向的位置
                *descriptors = FeatureDescriptorsToUnsignedByte(descriptors_float);

                return true;
            }

        private:
            const SiftExtractionOptions options_;
            SiftGPU sift_gpu_;
            std::vector<SiftKeypoint> keypoints_buffer_;
        };
#endif // COLMAP_GPU_ENABLED

    } // namespace

    std::unique_ptr<FeatureExtractor> CreateSiftFeatureExtractor(
        const SiftExtractionOptions &options)
    {
        if (options.estimate_affine_shape || options.domain_size_pooling ||
            options.force_covariant_extractor)
        {
            // 检查options中的三个属性 若满足其中一个则穿件一个协变sift cpu特征提取器
            LOG(INFO) << "Creating Covariant SIFT CPU feature extractor";
            return CovariantSiftCPUFeatureExtractor::Create(options);
        }
        // 如果配置指定使用GPU进行特征提取，则尝试创建一个GPU版本的SIFT特征提取器
        else if (options.use_gpu)
        {
            // GPU支持已启用，于是记录一条日志信息并创建一个SIFT GPU特征提取器，否则返回nullptr
#if defined(COLMAP_GPU_ENABLED)
            LOG(INFO) << "Creating SIFT GPU feature extractor";
            return SiftGPUFeatureExtractor::Create(options);
#else
            return nullptr;
#endif // COLMAP_GPU_ENABLED
        }
        // 如果既不满足协变SIFT的条件，也不使用GPU，则创建一个标准的SIFT CPU特征提取器
        else
        {
            LOG(INFO) << "Creating SIFT CPU feature extractor";
            return SiftCPUFeatureExtractor::Create(options);
        }
    }
    // 这个函数根据传入的配置选项来动态决定创建哪种类型的SIFT特征提取器。
    // 它可能是协变的SIFT CPU特征提取器、SIFT GPU特征提取器或标准的SIFT CPU特征提取器。

    namespace
    {

        size_t FindBestMatchesOneWayBruteForce(
            const Eigen::RowMajorMatrixXi &dot_products, // 一个行主序的Eigen矩阵，存储了点积的结果。
            const float max_ratio,                       // 最佳和次佳匹配之间的最大允许比例
            const float max_distance,                    // 匹配的最大允许L2距离
            std::vector<int> *matches)
        { // 一个指向整数向量的指针，用于存储匹配结果。
            // 在给定的点积矩阵dot_products中，为每一行找到一个最佳的匹配列索引，并存储在matches向量中
            // 定义常量，表示描述符范数的倒数的浮点值
            constexpr float kInvSqDescriptorNorm =
                static_cast<float>(1. / kSqSiftDescriptorNorm);
            // 初始化匹配计数为0
            size_t num_matches = 0;
            // 调整matchs向量的大小以匹配dot_products的行数，并初始化为-1，表示未进行匹配
            matches->resize(dot_products.rows(), -1);
            // 外层循环，遍历点积矩阵的每一行
            for (Eigen::Index i1 = 0; i1 < dot_products.rows(); ++i1)
            {
                // 初始化最佳匹配和次最佳匹配的索引及点积值
                int best_d2_idx = -1;
                int best_dot_product = 0;
                int second_best_dot_product = 0;
                // 内层循环，遍历点积矩阵的每一列
                for (Eigen::Index i2 = 0; i2 < dot_products.cols(); ++i2)
                {
                    // 获取当前行列的点积值
                    const int dot_product = dot_products(i1, i2);
                    // 更新最佳和次最佳点积值及其对应的列索引
                    if (dot_product > best_dot_product)
                    {
                        best_d2_idx = i2;
                        second_best_dot_product = best_dot_product;
                        best_dot_product = dot_product;
                    }
                    else if (dot_product > second_best_dot_product)
                    {
                        second_best_dot_product = dot_product;
                    }
                }

                // Check if any match found.
                // 检查是否至少有一个匹配，否则跳过该行
                if (best_d2_idx == -1)
                {
                    continue;
                }

                // Convert to L2 distance in which the thresholds are defined.
                // 将最佳点积值转化为L2距离
                const float best_dist_normed =
                    std::acos(std::min(kInvSqDescriptorNorm * best_dot_product, 1.0f));

                // Check if match distance passes threshold.
                // 检查最佳匹配的距离是否超过最大允许的距离，若超过也跳过该行
                if (best_dist_normed > max_distance)
                {
                    continue;
                }
                // 将次佳点积值转化为L2距离
                const float second_best_dist_normed = std::acos(
                    std::min(kInvSqDescriptorNorm * second_best_dot_product, 1.0f));

                // Check if match passes ratio test. Keep this comparison >= in order to
                // ensure that the case of best == second_best is detected.
                // 检查最佳匹配是否满足比例测试（即最佳距离与次佳距离的比例不超过max_ratio）
                if (best_dist_normed >= max_ratio * second_best_dist_normed)
                {
                    continue;
                }
                // 检查最佳匹配是否满足比例测试（即最佳距离与次佳距离的比例不超过max_ratio）
                ++num_matches;
                (*matches)[i1] = best_d2_idx;
            }

            return num_matches;
        }
        // 以上函数实现了一个基于点积的暴力匹配算法，主要用于特征点匹配任务。
        // 它通过遍历输入的点积矩阵来查找每一行的最佳匹配，并应用了两个筛选条件：最大L2距离和最佳/次佳匹配的比例

        void FindBestMatchesBruteForce(const Eigen::RowMajorMatrixXi &dot_products,
                                       const float max_ratio,
                                       const float max_distance,
                                       const bool cross_check, // 是否进行交叉验证
                                       FeatureMatches *matches)
        {
            matches->clear(); // 清除可能已存在的任何旧匹配

            std::vector<int> matches_1to2; // 存储从第一个特征集合到第二个特征集合的匹配索引。
            // 调用FindBestMatchesOneWayBruteForce函数，找到1到2单向的最佳匹配
            const size_t num_matches_1to2 = FindBestMatchesOneWayBruteForce(
                dot_products, max_ratio, max_distance, &matches_1to2);

            if (cross_check)
            {                                  // 进行交叉检查
                std::vector<int> matches_2to1; // 找到2到1的最佳匹配
                const size_t num_matches_2to1 = FindBestMatchesOneWayBruteForce(
                    dot_products.transpose(), max_ratio, max_distance, &matches_2to1);
                // 为matches预留空间，大小是num_matches_1to2和num_matches_2to1中的较小值，以提高内存使用效率。
                matches->reserve(std::min(num_matches_1to2, num_matches_2to1));
                // 遍历matches_1to2，对于每个索引i1，检查其是否在matches_2to1中有对应的匹配，并且这个匹配在matches_2to1中对应的索引是否指回到i1（即交叉检查）
                for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1)
                {
                    if (matches_1to2[i1] != -1 && matches_2to1[matches_1to2[i1]] != -1 &&
                        matches_2to1[matches_1to2[i1]] == static_cast<int>(i1))
                    {
                        // 如果满足交叉匹配条件，则创建一个FeatureMatch对象，设置其索引，并将其添加到matches中。
                        FeatureMatch match;
                        match.point2D_idx1 = i1;
                        match.point2D_idx2 = matches_1to2[i1];
                        matches->push_back(match);
                    }
                }
            }
            else
            {                                       // 若不进行交叉检查，直接添加匹配
                matches->reserve(num_matches_1to2); // 空间预留
                // 遍历matches_1to2，对于每个有效的匹配索引（不等于-1），创建一个FeatureMatch对象，设置其索引，并将其添加到matches中
                for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1)
                {
                    if (matches_1to2[i1] != -1)
                    {
                        FeatureMatch match;
                        match.point2D_idx1 = i1;
                        match.point2D_idx2 = matches_1to2[i1];
                        matches->push_back(match);
                    }
                }
            }
        }
        // 函数实现了一个基于点积矩阵的双向暴力特征匹配算法。
        // 它首先调用FindBestMatchesOneWayBruteForce函数找到从第一个特征集合到第二个特征集合的最佳匹配（matches_1to2）。
        // 然后，根据cross_check参数的值决定是否进行交叉检查
        // 如果cross_check为true，则函数还会找到从第二个特征集合到第一个特征集合的最佳匹配（matches_2to1），并只保留那些在两个方向上都是最佳匹配的匹配对
        // 如果cross_check为false，则函数将直接添加所有matches_1to2中的有效匹配到最终的匹配结果中，不进行交叉检查

        // 找到两组特征点间的最佳单向匹配
        size_t FindBestMatchesOneWayIndex(const Eigen::RowMajorMatrixXi &indices,  // 特征点索引矩阵
                                          const Eigen::RowMajorMatrixXi &l2_dists, // L2距离矩阵
                                          const float max_ratio,                   // 最大比例阈值
                                          const float max_distance,                // 最大距离阈值
                                          std::vector<int> *matches)               // 存储匹配结果的向量矩阵
        {
            // max_l2_dist是根据max_distance计算出的最大L2距离的平方
            const int max_l2_dist = kSqSiftDescriptorNorm * max_distance * max_distance;

            size_t num_matches = 0;
            // matches向量被重新调整大小，以容纳与indices行数相同数量的元素，并初始化为-1，表示初始时没有匹配。
            matches->resize(indices.rows(), -1);

            for (int d1_idx = 0; d1_idx < indices.rows(); ++d1_idx)
            {
                int best_d2_idx = -1;
                // 初始化最佳L2距离和次最佳L2距离为整型最大值
                int best_l2_dist = std::numeric_limits<int>::max();
                int second_best_l2_dist = std::numeric_limits<int>::max();
                for (int n_idx = 0; n_idx < indices.cols(); ++n_idx)
                {
                    // 从indices矩阵中获取当前可能的匹配点的索引。
                    const int d2_idx = indices(d1_idx, n_idx);
                    // 从l2_dists矩阵中获取当前两个特征点之间的L2距离。
                    const int l2_dist = l2_dists(d1_idx, n_idx);
                    // 如果当前L2距离小于最佳L2距离，则更新最佳和次佳匹配信息。
                    if (l2_dist < best_l2_dist)
                    {
                        best_d2_idx = d2_idx;
                        second_best_l2_dist = best_l2_dist;
                        best_l2_dist = l2_dist;
                    }
                    // 如果当前L2距离介于最佳和次佳之间，则只更新次佳L2距离。
                    else if (l2_dist < second_best_l2_dist)
                    {
                        second_best_l2_dist = l2_dist;
                    }
                }

                // Check if any match found.
                if (best_d2_idx == -1)
                {
                    continue;
                }

                // Check if match distance passes threshold.
                if (best_l2_dist > max_l2_dist)
                {
                    continue;
                }

                // Check if match passes ratio test. Keep this comparison >= in order to
                // ensure that the case of best == second_best is detected.
                if (std::sqrt(static_cast<float>(best_l2_dist)) >=
                    max_ratio * std::sqrt(static_cast<float>(second_best_l2_dist)))
                {
                    continue;
                }

                ++num_matches;
                (*matches)[d1_idx] = best_d2_idx;
            }

            return num_matches;
        }

        void FindBestMatchesIndex(const Eigen::RowMajorMatrixXi &indices_1to2,  // 第一个集合到第二个集合的索引映射
                                  const Eigen::RowMajorMatrixXi &l2_dists_1to2, // 第一个集合到第二个集合的L2距离矩阵
                                  const Eigen::RowMajorMatrixXi &indices_2to1,  // 2到1的索引映射
                                  const Eigen::RowMajorMatrixXi &l2_dists_2to1, // 2到1的距离矩阵
                                  const float max_ratio,
                                  const float max_distance,
                                  const bool cross_check,
                                  FeatureMatches *matches)
        {
            matches->clear();
            // 找到1到2特征点集合的最佳匹配
            std::vector<int> matches_1to2;
            const size_t num_matches_1to2 = FindBestMatchesOneWayIndex(
                indices_1to2, l2_dists_1to2, max_ratio, max_distance, &matches_1to2);

            if (cross_check && indices_2to1.rows()) // 进行交叉验证且2到1的索引不为空
            {
                // 找到2到1特征点集合的最佳匹配
                std::vector<int> matches_2to1;
                const size_t num_matches_2to1 = FindBestMatchesOneWayIndex(
                    indices_2to1, l2_dists_2to1, max_ratio, max_distance, &matches_2to1);
                matches->reserve(std::min(num_matches_1to2, num_matches_2to1));
                for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1)
                {
                    // 如果当前匹配有效，并且在反向匹配中也找到了对应的点，并且该点恰好是当前点，则认为是双向匹配成功。
                    if (matches_1to2[i1] != -1 && matches_2to1[matches_1to2[i1]] != -1 &&
                        matches_2to1[matches_1to2[i1]] == static_cast<int>(i1))
                    {
                        // 创建一个新对象并设置其索引，并将其添加到matches中
                        FeatureMatch match;
                        match.point2D_idx1 = i1;
                        match.point2D_idx2 = matches_1to2[i1];
                        matches->push_back(match);
                    }
                }
            }
            else // 否则直接将其添加到匹配中
            {
                matches->reserve(num_matches_1to2);
                for (size_t i1 = 0; i1 < matches_1to2.size(); ++i1)
                {
                    if (matches_1to2[i1] != -1)
                    {
                        FeatureMatch match;
                        match.point2D_idx1 = i1;
                        match.point2D_idx2 = matches_1to2[i1];
                        matches->push_back(match);
                    }
                }
            }
        }
        // 函数的主要目的是在两个特征点集合之间找到最佳匹配点，并可以根据需要进行双向交叉验证。
        // 通过调用FindBestMatchesOneWayIndex函数来分别找到两个方向的匹配结果，然后根据是否需要交叉验证来决定最终的匹配结果。
        // 如果需要交叉验证，则会检查两个方向的匹配结果是否一致，只有一致的匹配才会被添加到最终的匹配结果中
        // 共用体类 距离类型
        enum class DistanceType
        {
            L2,
            DOT_PRODUCT,
        };

        Eigen::RowMajorMatrixXi ComputeSiftDistanceMatrix(
            const DistanceType distance_type, // 距离类型
            const FeatureKeypoints *keypoints1,
            const FeatureKeypoints *keypoints2,
            const FeatureDescriptors &descriptors1,
            const FeatureDescriptors &descriptors2,
            const std::function<bool(float, float, float, float)> &guided_filter) // 引导过滤函数
        {
            // 参数有效性检查
            // 如果提供了guided_filter函数，则检查keypoints1和keypoints2是否非空，并确保关键点数量与对应的描述子行数一致。
            if (guided_filter != nullptr)
            {
                THROW_CHECK_NOTNULL(keypoints1);
                THROW_CHECK_NOTNULL(keypoints2);
                THROW_CHECK_EQ(keypoints1->size(), descriptors1.rows());
                THROW_CHECK_EQ(keypoints2->size(), descriptors2.rows());
            }

            // 将输入的浮点型描述子descriptors1和descriptors2转换为整数型
            const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
                descriptors1.cast<int>();
            const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
                descriptors2.cast<int>();

            // 创建一个行主序的矩阵用于存储计算出的描述子之间的距离值
            Eigen::RowMajorMatrixXi distances(descriptors1.rows(), descriptors2.rows());
            for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1)
            {
                for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2)
                {
                    /// 如果guided_filter返回true，则根据distance_type设置特定的距离值
                    if (guided_filter != nullptr && guided_filter((*keypoints1)[i1].x,
                                                                  (*keypoints1)[i1].y,
                                                                  (*keypoints2)[i2].x,
                                                                  (*keypoints2)[i2].y))
                    {
                        // 对于L2距离，设置一个预定义的常量值
                        if (distance_type == DistanceType::L2)
                        {
                            distances(i1, i2) = kSqSiftDescriptorNorm;
                        }
                        // 对于点积，设置距离为0
                        else if (distance_type == DistanceType::DOT_PRODUCT)
                        {
                            distances(i1, i2) = 0;
                        }
                        // 否则抛出异常
                        else
                        {
                            LOG(FATAL_THROW) << "Distance type not supported";
                        }
                    }
                    else
                    {
                        // 如果guided_filter未返回true或未提供，则计算实际的L2距离或点积
                        if (distance_type == DistanceType::L2)
                        {
                            // 计算两个整数型描述子行之间的差的平方范数（L2距离）
                            distances(i1, i2) =
                                (descriptors1_int.row(i1) - descriptors2_int.row(i2))
                                    .squaredNorm();
                        }
                        else if (distance_type == DistanceType::DOT_PRODUCT)
                        {
                            // 计算两个整数型描述子行之间的点积
                            distances(i1, i2) =
                                descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
                        }
                        // 否则抛出异常
                        else
                        {
                            LOG(FATAL_THROW) << "Distance type not supported";
                        }
                    }
                }
            }

            return distances;
        }
        // ComputeSiftDistanceMatrix函数是一个用于计算两组SIFT特征描述子之间距离矩阵的函数。
        // 它支持L2距离和点积两种计算方式，并可选地使用一个guided_filter函数来根据关键点位置信息过滤某些距离计算。
        // 函数通过双层循环遍历所有描述子对，并根据所选的距离类型和过滤条件计算相应的距离值，最后返回一个包含所有计算结果的矩阵。

        class SiftCPUFeatureMatcher : public FeatureMatcher
        {
        public:
            explicit SiftCPUFeatureMatcher(const SiftMatchingOptions &options)
                : options_(options) // 初始化成员变量
            {
                // 参数有效性检查
                THROW_CHECK(options_.Check());
            }

            static std::unique_ptr<FeatureMatcher> Create( // 创建featurematcher对象，并返回一个智能指针
                const SiftMatchingOptions &options)
            {
                return std::make_unique<SiftCPUFeatureMatcher>(options);
            }

            void Match(const Image &image1,
                       const Image &image2,
                       FeatureMatches *matches) override
            {
                THROW_CHECK_NOTNULL(matches);
                THROW_CHECK_NE(image1.image_id, kInvalidImageId);
                THROW_CHECK_NE(image2.image_id, kInvalidImageId);
                THROW_CHECK_NOTNULL(image1.descriptors);
                THROW_CHECK_NOTNULL(image2.descriptors);
                THROW_CHECK_EQ(image1.descriptors->cols(), 128);
                THROW_CHECK_EQ(image2.descriptors->cols(), 128);

                matches->clear();

                if (!options_.cpu_brute_force_matcher &&
                    (prev_image_id1_ == kInvalidImageId ||
                     prev_image_id1_ != image1.image_id))
                {
                    index1_ = options_.cpu_descriptor_index_cache->Get(image1.image_id);
                    prev_image_id1_ = image1.image_id;
                }

                if (!options_.cpu_brute_force_matcher &&
                    (prev_image_id2_ == kInvalidImageId ||
                     prev_image_id2_ != image2.image_id))
                {
                    index2_ = options_.cpu_descriptor_index_cache->Get(image2.image_id);
                    prev_image_id2_ = image2.image_id;
                }

                if (image1.descriptors->rows() == 0 || image2.descriptors->rows() == 0)
                {
                    return;
                }

                if (options_.cpu_brute_force_matcher)
                {
                    const Eigen::RowMajorMatrixXi dot_products =
                        ComputeSiftDistanceMatrix(DistanceType::DOT_PRODUCT,
                                                  nullptr,
                                                  nullptr,
                                                  *image1.descriptors,
                                                  *image2.descriptors,
                                                  nullptr);
                    FindBestMatchesBruteForce(dot_products,
                                              options_.max_ratio,
                                              options_.max_distance,
                                              options_.cross_check,
                                              matches);
                    return;
                }

                Eigen::RowMajorMatrixXi indices_1to2;
                Eigen::RowMajorMatrixXi l2_dists_1to2;
                Eigen::RowMajorMatrixXi indices_2to1;
                Eigen::RowMajorMatrixXi l2_dists_2to1;
                index2_->Search(
                    /*num_neighbors=*/2, *image1.descriptors, indices_1to2, l2_dists_1to2);
                if (options_.cross_check)
                {
                    index1_->Search(/*num_neighbors=*/2,
                                    *image2.descriptors,
                                    indices_2to1,
                                    l2_dists_2to1);
                }

                FindBestMatchesIndex(indices_1to2,
                                     l2_dists_1to2,
                                     indices_2to1,
                                     l2_dists_2to1,
                                     options_.max_ratio,
                                     options_.max_distance,
                                     options_.cross_check,
                                     matches);
            }

            void MatchGuided(const double max_error,
                             const Image &image1,
                             const Image &image2,
                             TwoViewGeometry *two_view_geometry) override
            {
                // 参数有效性检查
                THROW_CHECK_NOTNULL(two_view_geometry);
                THROW_CHECK_NE(image1.image_id, kInvalidImageId);
                THROW_CHECK_NE(image2.image_id, kInvalidImageId);
                THROW_CHECK_NOTNULL(image1.descriptors);
                THROW_CHECK_NOTNULL(image1.keypoints);
                THROW_CHECK_NOTNULL(image2.descriptors);
                THROW_CHECK_NOTNULL(image2.keypoints);
                THROW_CHECK_EQ(image1.descriptors->rows(), image1.keypoints->size());
                THROW_CHECK_EQ(image2.descriptors->rows(), image2.keypoints->size());
                THROW_CHECK_EQ(image1.descriptors->cols(), 128);
                THROW_CHECK_EQ(image2.descriptors->cols(), 128);

                // 清除对象中旧的匹配结果
                two_view_geometry->inlier_matches.clear();
                // 检查是否应该使用暴力匹配器（cpu_brute_force_matcher）。
                // 如果不是，并且当前图像与上一次处理的图像不同，则从缓存中获取该图像的描述符索引。
                if (!options_.cpu_brute_force_matcher &&
                    (prev_image_id1_ == kInvalidImageId ||
                     prev_image_id1_ != image1.image_id))
                {
                    index1_ = options_.cpu_descriptor_index_cache->Get(image1.image_id);
                    prev_image_id1_ = image1.image_id;
                }

                if (!options_.cpu_brute_force_matcher &&
                    (prev_image_id2_ == kInvalidImageId ||
                     prev_image_id2_ != image2.image_id))
                {
                    index2_ = options_.cpu_descriptor_index_cache->Get(image2.image_id);
                    prev_image_id2_ = image2.image_id;
                }

                // 计算最大残差值 max_error的平方
                const float max_residual = max_error * max_error;
                // 将two_view_geometry中的F和H矩阵转换为float类型，并存储在局部变量F和H中。
                const Eigen::Matrix3f F = two_view_geometry->F.cast<float>();
                const Eigen::Matrix3f H = two_view_geometry->H.cast<float>();

                // 定义一个std::function类型的guided_filter，它接受四个float参数并返回一个bool值。
                // 这个过滤器将根据两幅图像之间的几何关系来决定哪些关键点对是有效的
                std::function<bool(float, float, float, float)> guided_filter;
                // 判断视图几何配置的两种可能情况 校准或未校准
                if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
                    two_view_geometry->config == TwoViewGeometry::UNCALIBRATED)
                {
                    // 如果上述条件满足，我们为guided_filter赋值一个lambda表达式。
                    // 这个lambda表达式定义了如何基于校准的或未校准的视图几何来过滤关键点对。
                    guided_filter =
                        [&](const float x1, const float y1, const float x2, const float y2)
                    {
                        const Eigen::Vector3f p1(x1, y1, 1.0f);
                        const Eigen::Vector3f p2(x2, y2, 1.0f);
                        // 使用基础矩阵F对p1进行变换，得到一个新的3D向量Fx1。
                        const Eigen::Vector3f Fx1 = F * p1;
                        // 使用F的转置矩阵对p2进行变换，得到一个新的3D向量Ftx2。
                        const Eigen::Vector3f Ftx2 = F.transpose() * p2;
                        const float x2tFx1 = p2.transpose() * Fx1; //// 计算p2的转置与Fx1的点积，得到一个浮点数x2tFx1。
                                                                   // 返回一个bool值比较了x2tFx1的平方与Fx1和Ftx2的某些元素的平方和之间的比率是否大于max_residual。
                        return x2tFx1 * x2tFx1 /
                                   return x2tFx1 * x2tFx1 /
                                   (Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) + Ftx2(0) * Ftx2(0) +
                                    Ftx2(1) * Ftx2(1)) >
                               max_residual;
                    };
                }
                // 否则，如果视图几何配置是平面或全景或平面或全景，则将guided_filter设置为lambda表达式。
                else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
                         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
                         two_view_geometry->config ==
                             TwoViewGeometry::PLANAR_OR_PANORAMIC)
                {
                    guided_filter =
                        [&](const float x1, const float y1, const float x2, const float y2)
                    {
                        const Eigen::Vector3f p1(x1, y1, 1.0f);
                        const Eigen::Vector2f p2(x2, y2);
                        // 使用单应性矩阵H对p1进行变换，并对结果进行齐次归一化，得到一个2D向量。
                        // 然后计算这个归一化后的向量与p2之间的平方欧氏距离。
                        return ((H * p1).hnormalized() - p2).squaredNorm() > max_residual;
                    };
                }
                else
                {
                    return;
                }

                THROW_CHECK(guided_filter);
                // 调用ComputeSiftDistanceMatrix函数计算从image1到image2的SIFT距离矩阵（l2_dists_1to2），
                // 以及从image2到image1的距离矩阵（通过转置得到l2_dists_2to1）。
                const Eigen::RowMajorMatrixXi l2_dists_1to2 =
                    ComputeSiftDistanceMatrix(DistanceType::L2,
                                              image1.keypoints.get(),
                                              image2.keypoints.get(),
                                              *image1.descriptors,
                                              *image2.descriptors,
                                              guided_filter);
                const Eigen::RowMajorMatrixXi l2_dists_2to1 = l2_dists_1to2.transpose();

                // 创建并初始化两个索引矩阵indices_1to2和indices_2to1
                Eigen::RowMajorMatrixXi indices_1to2(l2_dists_1to2.rows(),
                                                     l2_dists_1to2.cols());
                for (int i = 0; i < indices_1to2.rows(); ++i)
                {
                    indices_1to2.row(i) = Eigen::VectorXi::LinSpaced(
                        indices_1to2.cols(), 0, indices_1to2.cols() - 1);
                }
                Eigen::RowMajorMatrixXi indices_2to1(l2_dists_1to2.cols(),
                                                     l2_dists_1to2.rows());
                for (int i = 0; i < indices_2to1.rows(); ++i)
                {
                    indices_2to1.row(i) = Eigen::VectorXi::LinSpaced(
                        indices_2to1.cols(), 0, indices_2to1.cols() - 1);
                }
                // 调用FindBestMatchesIndex函数，使用之前计算的距离矩阵和索引矩阵来查找两幅图像之间的最佳匹配。
                // 这些匹配结果将存储在two_view_geometry->inlier_matches中。
                FindBestMatchesIndex(indices_1to2,
                                     l2_dists_1to2,
                                     indices_2to1,
                                     l2_dists_2to1,
                                     options_.max_ratio,
                                     options_.max_distance,
                                     options_.cross_check,
                                     &two_view_geometry->inlier_matches);
            }
            // MatchGuided函数的整体功能是实现两幅图像之间的特征点匹配。
            // 它首先检查输入参数的有效性，然后清除旧的匹配结果。接下来，根据配置和需要，它可能从缓存中获取描述符索引以优化性能
            // 。然后，它定义了一个过滤器来根据几何关系过滤关键点对，并计算SIFT距离矩阵和索引矩阵。最后，它调用一个函数来查找并存储两幅图像之间的最佳匹配结果
        private:
            const SiftMatchingOptions options_;
            image_t prev_image_id1_ = kInvalidImageId;
            image_t prev_image_id2_ = kInvalidImageId;
            std::shared_ptr<FeatureDescriptorIndex> index1_;
            std::shared_ptr<FeatureDescriptorIndex> index2_;
        };

#if defined(COLMAP_GPU_ENABLED)
        // Mutexes that ensure that only one thread extracts/matches on the same GPU
        // at the same time, since SiftGPU internally uses static variables.
        // 这几行注释和代码定义了一个静态的std::map，它映射整数（可能是GPU的索引）到std::mutex的智能指针。
        static std::map<int, std::unique_ptr<std::mutex>> sift_match_gpu_mutexes_;

        class SiftGPUFeatureMatcher : public FeatureMatcher
        {
        public:
            explicit SiftGPUFeatureMatcher(const SiftMatchingOptions &options)
                : options_(options)
            {
                THROW_CHECK(options_.Check());
            }

            static std::unique_ptr<FeatureMatcher> Create(
                const SiftMatchingOptions &options)
            {
                // SiftGPU uses many global static state variables and the initialization
                // must be thread-safe in order to work correctly. This is enforced here.
                // 由于SiftGPU使用了许多全局静态状态变量，其初始化必须是线程安全的。
                // 这里，通过使用一个静态的std::mutex和std::lock_guard来确保在Create函数执行期间，其他线程无法进入此代码段，从而保证了线程安全。
                static std::mutex mutex;
                std::lock_guard<std::mutex> lock(mutex);

                // 将传入的index选项转换为整数向量并检查该向量的大小是否为1，否则抛出错误
                const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
                THROW_CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";
                // 代码初始化了SiftGPU对象，并设置了其详细级别为0（即不输出详细信息）。
                SiftGPU sift_gpu;
                sift_gpu.SetVerbose(0);
                // 创建了一个SiftGPUFeatureMatcher的实例，并初始化了其内部的sift_match_gpu_成员，该成员是SiftMatchGPU类型的对象
                auto matcher = std::make_unique<SiftGPUFeatureMatcher>(options);

                // Note that the SiftMatchGPU object is not movable (for whatever reason).
                // If we instead create the object here and move it to the constructor, the
                // program segfaults inside SiftMatchGPU.

                matcher->sift_match_gpu_ = SiftMatchGPU(options.max_num_matches);

// 根据是否定义了COLMAP_CUDA_ENABLED，代码设置sift_match_gpu_使用的语言。
// 如果定义了，并且GPU索引非负，则使用CUDA；否则，使用OpenGL的GLSL。这是为了支持不同的GPU加速库。（？？）
#if defined(COLMAP_CUDA_ENABLED)
                if (gpu_indices[0] >= 0)
                {
                    matcher->sift_match_gpu_.SetLanguage(
                        SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 + gpu_indices[0]);
                }
                else
                {
                    matcher->sift_match_gpu_.SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
                }
#else  // COLMAP_CUDA_ENABLED
                matcher->sift_match_gpu_.SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
#endif // COLMAP_CUDA_ENABLED
       // 验证GPU上下文是否有效。如果无效（返回0），则函数返回nullptr，表示无法创建匹配器。
                if (matcher->sift_match_gpu_.VerifyContextGL() == 0)
                {
                    return nullptr;
                }
                // 尝试为sift_match_gpu_分配所需的内存。如果分配失败（可能是因为GPU内存不足），则记录一个错误并返回nullptr
                if (!matcher->sift_match_gpu_.Allocate(options.max_num_matches,
                                                       options.cross_check))
                {
                    LOG(ERROR) << StringPrintf(
                        "Not enough GPU memory to match %d features. "
                        "Reduce the maximum number of matches.",
                        options.max_num_matches);
                    return nullptr;
                }

// 如果没有定义COLMAP_CUDA_ENABLED，代码还会检查OpenGL版本的SiftGPU是否支持请求的最大匹配数。如果不支持，它会记录一个警告。
#if !defined(COLMAP_CUDA_ENABLED)
                if (matcher->sift_match_gpu_.GetMaxSift() < options.max_num_matches)
                {
                    LOG(WARNING) << StringPrintf(
                        "OpenGL version of SiftGPU only supports a "
                        "maximum of %d matches - consider changing to CUDA-based "
                        "feature matching to avoid this limitation.",
                        matcher->sift_match_gpu_.GetMaxSift());
                }
#endif // COLMAP_CUDA_ENABLED
       // 最后，代码设置sift_match_gpu_的gpu_index成员，并检查sift_match_gpu_mutexes_映射中是否已经有一个与该GPU索引对应的互斥锁。如果没有，它会添加一个。
                matcher->sift_match_gpu_.gpu_index = gpu_indices[0];
                if (sift_match_gpu_mutexes_.count(gpu_indices[0]) == 0)
                {
                    sift_match_gpu_mutexes_.emplace(gpu_indices[0],
                                                    std::make_unique<std::mutex>());
                }

                return matcher;
            }
            // 这个Create函数的目的是以线程安全的方式创建一个SiftGPUFeatureMatcher实例，并确保它正确地初始化了其内部的SiftGPU和SiftMatchGPU对象，以便进行后续的特征匹配操作。

            void Match(const Image &image1,
                       const Image &image2,
                       FeatureMatches *matches) override
            {
                // 参数有效性检查
                THROW_CHECK_NOTNULL(matches);
                THROW_CHECK_NE(image1.image_id, kInvalidImageId);
                THROW_CHECK_NE(image2.image_id, kInvalidImageId);
                THROW_CHECK_NOTNULL(image1.descriptors);
                THROW_CHECK_NOTNULL(image2.descriptors);
                THROW_CHECK_EQ(image1.descriptors->cols(), 128);
                THROW_CHECK_EQ(image2.descriptors->cols(), 128);
                // 清空matchs中的旧数据
                matches->clear();

                std::lock_guard<std::mutex> lock(
                    *sift_match_gpu_mutexes_[sift_match_gpu_.gpu_index]); // 使用互斥锁确保在同一GPU上不会同时进行多个匹配操作
                // 如果之前的图像ID无效、之前是引导匹配或之前的图像ID与当前图像ID不匹配，则会设置新的描述符。
                if (prev_image_id1_ == kInvalidImageId || prev_is_guided_ ||
                    prev_image_id1_ != image1.image_id)
                {
                    WarnIfMaxNumMatchesReachedGPU(*image1.descriptors);
                    sift_match_gpu_.SetDescriptors(
                        0, image1.descriptors->rows(), image1.descriptors->data());
                    prev_image_id1_ = image1.image_id;
                }

                if (prev_image_id2_ == kInvalidImageId || prev_is_guided_ ||
                    prev_image_id2_ != image2.image_id)
                {
                    WarnIfMaxNumMatchesReachedGPU(*image2.descriptors);
                    sift_match_gpu_.SetDescriptors(
                        1, image2.descriptors->rows(), image2.descriptors->data());
                    prev_image_id2_ = image2.image_id;
                }

                prev_is_guided_ = false; // 表示当前不是引导匹配

                matches->resize(static_cast<size_t>(options_.max_num_matches));
                // 调用SiftGPU的GetSiftMatch函数来执行实际的特征匹配
                const int num_matches = sift_match_gpu_.GetSiftMatch(
                    options_.max_num_matches,
                    reinterpret_cast<uint32_t(*)[2]>(matches->data()),
                    static_cast<float>(options_.max_distance),
                    static_cast<float>(options_.max_ratio),
                    options_.cross_check);

                if (num_matches < 0) // 若匹配数小于0 表示匹配失败，则清空匹配结果
                {
                    LOG(ERROR) << "Feature matching failed. This is probably caused by "
                                  "insufficient GPU memory. Consider reducing the maximum "
                                  "number of features and/or matches.";
                    matches->clear();
                }
                else // 否则检查返回的匹配数是否小于或等于预分配的内存大小，并调整matches的大小以匹配实际的匹配数。
                {
                    THROW_CHECK_LE(num_matches, matches->size());
                    matches->resize(num_matches);
                }
            }
            // 这个Match函数实现了在GPU上使用SiftGPU库匹配两个图像的特征点的功能。它首先检查传入的参数是否有效，然后确保在同一GPU上不会同时进行多个匹配操作。
            // 接着，它为两个图像设置描述符（如果需要的话），并执行实际的特征匹配。最后，它处理匹配结果，确保matches中只包含有效的匹配项。

            void MatchGuided(const double max_error, // 最大允许误差
                             const Image &image1,
                             const Image &image2,
                             TwoViewGeometry *two_view_geometry) override
            {
                static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float),
                              "Invalid keypoint format");
                static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float),
                              "Invalid keypoint format");
                static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float),
                              "Invalid keypoint format");

                THROW_CHECK_NOTNULL(two_view_geometry);
                THROW_CHECK_NE(image1.image_id, kInvalidImageId);
                THROW_CHECK_NE(image2.image_id, kInvalidImageId);
                THROW_CHECK_NOTNULL(image1.descriptors);
                THROW_CHECK_NOTNULL(image1.keypoints);
                THROW_CHECK_NOTNULL(image2.descriptors);
                THROW_CHECK_NOTNULL(image2.keypoints);
                THROW_CHECK_EQ(image1.descriptors->rows(), image1.keypoints->size());
                THROW_CHECK_EQ(image2.descriptors->rows(), image2.keypoints->size());
                THROW_CHECK_EQ(image1.descriptors->cols(), 128);
                THROW_CHECK_EQ(image2.descriptors->cols(), 128);

                two_view_geometry->inlier_matches.clear();

                std::lock_guard<std::mutex> lock(
                    *sift_match_gpu_mutexes_[sift_match_gpu_.gpu_index]);

                constexpr size_t kFeatureShapeNumElems = 4; // 特征形状的元素个数
                // 检查prev_image_id1_是否是无效ID、是否不是引导匹配，或者是否与当前image1的ID不同。
                if (prev_image_id1_ == kInvalidImageId || !prev_is_guided_ ||
                    prev_image_id1_ != image1.image_id)
                {
                    //// 如果image1的描述符数量超过了GPU上SIFT匹配器的最大匹配数量，则发出警告。
                    WarnIfMaxNumMatchesReachedGPU(*image1.descriptors);
                    const size_t kIndex = 0; // 表示这是第一个图像
                                             // 为GPU上的SIFT匹配器设置image1的描述符数据。
                    sift_match_gpu_.SetDescriptors(
                        kIndex, image1.descriptors->rows(), image1.descriptors->data());
                    // 为GPU上的SIFT匹配器设置image1的描述符数据。
                    sift_match_gpu_.SetFeautreLocation(
                        kIndex,
                        reinterpret_cast<const float *>(image1.keypoints->data()),
                        kFeatureShapeNumElems);
                    // 更新pre_image_id1
                    prev_image_id1_ = image1.image_id;
                }
                // 对image2的处理类似于image1
                if (prev_image_id2_ == kInvalidImageId || !prev_is_guided_ ||
                    prev_image_id2_ != image2.image_id)
                {
                    WarnIfMaxNumMatchesReachedGPU(*image2.descriptors);
                    const size_t kIndex = 1;
                    sift_match_gpu_.SetDescriptors(
                        kIndex, image2.descriptors->rows(), image2.descriptors->data());
                    sift_match_gpu_.SetFeautreLocation(
                        kIndex,
                        reinterpret_cast<const float *>(image2.keypoints->data()),
                        kFeatureShapeNumElems);
                    prev_image_id2_ = image2.image_id;
                }

                prev_is_guided_ = true;

                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F; // 定义两个3*3行主序浮点矩阵
                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
                float *F_ptr = nullptr;
                float *H_ptr = nullptr;
                if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
                    two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) // 判断配置类型
                {
                    // 如果是CALIBRATED或UNCALIBRATED类型，将two_view_geometry中的F矩阵转换为浮点类型并赋值给F
                    F = two_view_geometry->F.cast<float>();
                    F_ptr = F.data();
                }
                // 如果是PLANAR、PANORAMIC或PLANAR_OR_PANORAMIC类型，将two_view_geometry中的H矩阵转换为浮点类型并赋值给H
                H = two_view_geometry->H.cast<float>();
                else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
                         two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
                         two_view_geometry->config ==
                             TwoViewGeometry::PLANAR_OR_PANORAMIC)
                {
                    H = two_view_geometry->H.cast<float>();
                    H_ptr = H.data();
                }
                // 若配置不符合以上任何一种 直接返回
                else
                {
                    return;
                }
                // 检查F_ptr和H_ptr至少有一个不为nullptr，否则抛出异常
                THROW_CHECK(F_ptr != nullptr || H_ptr != nullptr);

                two_view_geometry->inlier_matches.resize(
                    static_cast<size_t>(options_.max_num_matches));
                // 计算最大残差并转化为浮点数
                const float max_residual = static_cast<float>(max_error * max_error);
                // 调用GPU上的SIFT匹配器进行引导匹配，获取匹配结果数量
                const int num_matches = sift_match_gpu_.GetGuidedSiftMatch(
                    options_.max_num_matches,
                    reinterpret_cast<uint32_t(*)[2]>(
                        two_view_geometry->inlier_matches.data()),
                    H_ptr,
                    F_ptr,
                    static_cast<float>(options_.max_distance),
                    static_cast<float>(options_.max_ratio),
                    max_residual,
                    max_residual,
                    options_.cross_check);

                if (num_matches < 0) // 检查匹配结果数量，如果小于0，说明匹配失败，记录错误信息并清空匹配结果
                {
                    LOG(ERROR) << "Feature matching failed. This is probably caused by "
                                  "insufficient GPU memory. Consider reducing the maximum "
                                  "number of features.";
                    two_view_geometry->inlier_matches.clear();
                }
                else
                {
                    // 如果匹配成功，检查匹配结果数量是否超过inlier_matches的大小，抛出异常
                    THROW_CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
                    // 调整inlier_matches的大小为实际的匹配结果数量
                    two_view_geometry->inlier_matches.resize(num_matches);
                }
            }
            // MatchGuided函数是一个复杂而高效的函数，它充分利用了GPU的并行处理能力来加速SIFT特征匹配过程。
            // 通过详细的参数检查、线程安全控制、GPU数据设置、匹配配置以及结果处理流程，该函数能够准确地找到两个图像之间的匹配特征点，并将这些信息存储在双视图几何对象中供后续使用。

        private:
            void WarnIfMaxNumMatchesReachedGPU(const FeatureDescriptors &descriptors)
            {
                if (sift_match_gpu_.GetMaxSift() < descriptors.rows())
                {
                    LOG(WARNING) << StringPrintf(
                        "Clamping features from %d to %d - consider "
                        "increasing the maximum number of matches.",
                        descriptors.rows(),
                        sift_match_gpu_.GetMaxSift());
                }
            }
            // 检查传入的特征点数量（descriptors.rows()）是否超过了GPU上SIFT匹配的最大数量（sift_match_gpu_.GetMaxSift()）。
            // 如果超过了，就记录一条警告日志，提示用户特征点数量被限制了，并建议增加最大匹配数量。

            const SiftMatchingOptions options_;
            SiftMatchGPU sift_match_gpu_;
            bool prev_is_guided_ = false;
            image_t prev_image_id1_ = kInvalidImageId;
            image_t prev_image_id2_ = kInvalidImageId;
        };
#endif // COLMAP_GPU_ENABLED

    } // namespace

    std::unique_ptr<FeatureMatcher> CreateSiftFeatureMatcher(
        const SiftMatchingOptions &options)
    {
        if (options.use_gpu)
        {
#if defined(COLMAP_GPU_ENABLED)
            // 如果定义了COLMAP_GPU_ENABLED，则创建SIFT GPU特征匹配器
            LOG(INFO) << "Creating SIFT GPU feature matcher";
            return SiftGPUFeatureMatcher::Create(options);
#else  // 如果没有定义COLMAP_GPU_ENABLED，则返回nullptr
            return nullptr;
#endif // COLMAP_GPU_ENABLED
        }
        else
        {
            // 如果没有定义COLMAP_GPU_ENABLED，则创建SIFT CPU特征匹配器
            LOG(INFO) << "Creating SIFT CPU feature matcher";
            return SiftCPUFeatureMatcher::Create(options);
        }
    }
 /*
     从文本文件中加载SIFT特征。
     文件具备以下特征：
     1. 第一行包含两个整数：特征点的数量和描述符的维度（维度必须为128）
     2. 后面的每一行包含四个整数：x坐标, y坐标, 尺度 scale, 方向 orientation

     参数：
     path : 文件路径
     keypoints : 存储加载好的特征点
     descriptors ：存储加载好的描述符
 */

    void LoadSiftFeaturesFromTextFile(const std::string &path,
                                      FeatureKeypoints *keypoints,
                                      FeatureDescriptors *descriptors)
    {
        THROW_CHECK_NOTNULL(keypoints);
        THROW_CHECK_NOTNULL(descriptors);

        std::ifstream file(path.c_str());
        THROW_CHECK_FILE_OPEN(file, path);

        std::string line;
        std::string item;

        std::getline(file, line);
        std::stringstream header_line_stream(line);
        //解析特征点数量
        std::getline(header_line_stream >> std::ws, item, ' ');
        const point2D_t num_features = std::stoul(item);
        //解析描述符维度
        std::getline(header_line_stream >> std::ws, item, ' ');
        const size_t dim = std::stoul(item);
       //检查描述符维度是否为128
        THROW_CHECK_EQ(dim, 128) << "SIFT features must have 128 dimensions";
       //调整关键点和描述符的容量以容纳所有特征
        keypoints->resize(num_features);
        descriptors->resize(num_features, dim);

        for (size_t i = 0; i < num_features; ++i)
        {
            std::getline(file, line);
            std::stringstream feature_line_stream(line);
            //解析x坐标
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float x = std::stold(item);
            //解析y坐标
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float y = std::stold(item);
            //解析尺度scale
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float scale = std::stold(item);
            //解析方向
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float orientation = std::stold(item);
            //储存特征点信息
            (*keypoints)[i] = FeatureKeypoint(x, y, scale, orientation);

            // Descriptor
            for (size_t j = 0; j < dim; ++j)
            {
                std::getline(feature_line_stream >> std::ws, item, ' ');
                const float value = std::stod(item);
                //检查描述符值是否在合法范围内（0到255）
                THROW_CHECK_GE(value, 0);
                THROW_CHECK_LE(value, 255);
                (*descriptors)(i, j) = TruncateCast<float, uint8_t>(value);
            }
        }
        // 这个函数LoadSiftFeaturesFromTextFile的主要功能是从一个文本文件中加载SIFT（尺度不变特征变换）特征的关键点和描述符。
    }

} //  na
