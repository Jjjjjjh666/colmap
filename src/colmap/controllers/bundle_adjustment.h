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

#pragma once

// 包含用于管理选项的头文件
#include "colmap/controllers/option_manager.h"
// 包含重建场景相关的数据结构和功能
#include "colmap/scene/reconstruction.h"
// 包含控制器的基础类
#include "colmap/util/base_controller.h"

namespace colmap {

// 全局捆绑调整控制器类，用于协调和执行捆绑调整（Bundle Adjustment）过程。
// 捆绑调整是计算机视觉中常用的优化方法，旨在通过调整相机参数和三维点的位置
// 来最小化重投影误差，从而提升三维重建的质量。
class BundleAdjustmentController : public BaseController {
 public:
  // 构造函数，初始化捆绑调整控制器。
  // 参数：
  // - options: 全局选项管理器，用于存储和管理捆绑调整相关的参数。
  // - reconstruction: 指向重建对象的共享指针，表示待优化的三维重建数据。
  BundleAdjustmentController(const OptionManager& options,
                             std::shared_ptr<Reconstruction> reconstruction);

  // 执行捆绑调整过程的主要函数。
  void Run();

 private:
  // 用于存储捆绑调整相关的参数选项。
  const OptionManager options_;
  // 用于存储三维重建数据的共享指针。
  std::shared_ptr<Reconstruction> reconstruction_;
};

}  // namespace colmap

