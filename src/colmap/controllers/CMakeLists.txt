# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# 设置一个变量FOLDER_NAME，其值为"controllers"，用于后续可能的路径或名称相关操作
set(FOLDER_NAME "controllers")
# 使用COLMAP_ADD_LIBRARY宏来创建一个名为colmap_controllers的库
# SRCS（源文件）部分列出了构成该库的所有源文件（.h头文件和对应的.cc实现文件）
# 这些源文件涉及到自动重建、束调整、分层映射、特征提取、特征匹配等多个功能模块相关的代码

COLMAP_ADD_LIBRARY(
    NAME colmap_controllers
    SRCS
        automatic_reconstruction.h automatic_reconstruction.cc
        bundle_adjustment.h bundle_adjustment.cc
        hierarchical_mapper.h hierarchical_mapper.cc
        feature_extraction.h feature_extraction.cc
        feature_matching.h feature_matching.cc
        feature_matching_utils.h feature_matching_utils.cc
        image_reader.h image_reader.cc
        incremental_mapper.h incremental_mapper.cc
        option_manager.h option_manager.cc
    # PUBLIC_LINK_LIBS部分列出了该库在链接时需要公开链接的其他库
    # 这些库包括colmap_estimators、colmap_feature等多个colmap相关的库以及Eigen3::Eigen、Boost::program_options等外部库
    PUBLIC_LINK_LIBS
        colmap_estimators
        colmap_feature
        colmap_geometry
        colmap_scene
        colmap_util
        Eigen3::Eigen
        Boost::program_options
    # PRIVATE_LINK_LIBS部分列出了该库在链接时需要私有链接的其他库
    # 同样包含多个colmap相关库以及Ceres::ceres、Boost::boost等外部库
    PRIVATE_LINK_LIBS
        colmap_image
        colmap_math
        colmap_mvs
        colmap_sfm
        Ceres::ceres
        Boost::boost
)
# 使用COLMAP_ADD_TEST宏来创建一个名为hierarchical_mapper_test的测试可执行文件
# SRCS指定了该测试的源文件为hierarchical_mapper_test.cc
# LINK_LIBS指定了该测试可执行文件在链接时需要链接的库为colmap_controllers库
COLMAP_ADD_TEST(
    NAME hierarchical_mapper_test
    SRCS hierarchical_mapper_test.cc
    LINK_LIBS colmap_controllers
)
# 使用COLMAP_ADD_TEST宏来创建一个名为incremental_mapper_test的测试可执行文件
# 类似地，SRCS指定源文件为incremental_mapper_test.cc，LINK_LIBS指定链接colmap_controllers库
COLMAP_ADD_TEST(
    NAME incremental_mapper_test
    SRCS incremental_mapper_test.cc
    LINK_LIBS colmap_controllers
)
