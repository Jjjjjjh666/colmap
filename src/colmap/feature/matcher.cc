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

#include "colmap/feature/matcher.h"

namespace colmap {

FeatureMatcherCache::FeatureMatcherCache(
    const size_t cache_size, const std::shared_ptr<Database>& database)
    : cache_size_(cache_size),  //初始化为传入的缓存大小
      database_(THROW_CHECK_NOTNULL(database)),  //初始化为传入的数据库指针并进行非空检查
      descriptor_index_cache_(cache_size_, [this](const image_t image_id) {  //接收缓存大小和lambda表达式作为参数
          //lambda表达式：当缓存缺失时，调用该函数以生成所需要的数据
        auto descriptors = GetDescriptors(image_id);  //获取描述符
        auto index = FeatureDescriptorIndex::Create();  //获取描述符索引对象
        index->Build(*descriptors);  //创建一个FeatureDescriptorIndex对象并调用Build方法构建索引。
        return index;
      }) {
  keypoints_cache_ =   //存储图像关键点信息
      std::make_unique<ThreadSafeLRUCache<image_t, FeatureKeypoints>>(   
          cache_size_, [this](const image_t image_id) {   //lambda表达式，缓存未命中时执行操作
            std::lock_guard<std::mutex> lock(database_mutex_);  //加锁，保护database的操作线程安全
            return std::make_shared<FeatureKeypoints>(    //智能指针，返回包含关键点的对象
                database_->ReadKeypoints(image_id));  //从数据库中读取相应图像ID的关键点
          });

  descriptors_cache_ =
      std::make_unique<ThreadSafeLRUCache<image_t, FeatureDescriptors>>(
          cache_size_, [this](const image_t image_id) {
            std::lock_guard<std::mutex> lock(database_mutex_);  // 在已加锁的情况下进行数据库操作
                                                                // 这里的代码会在互斥锁的保护下执行，保证线程安全
            return std::make_shared<FeatureDescriptors>(
                database_->ReadDescriptors(image_id));   //调用database_->ReadKeypoints从数据库中读取对应图像 ID 的关键点，并返回一个指向包含这些关键点的FeatureKeypoints对象的智能指针。
          });

  keypoints_exists_cache_ = std::make_unique<ThreadSafeLRUCache<image_t, bool>>(  //指向一个用于存储图像关键点是否存在的对象
      cache_size_, [this](const image_t image_id) {
        std::lock_guard<std::mutex> lock(database_mutex_);
        return std::make_shared<bool>(database_->ExistsKeypoints(image_id));
      });   //类似于上

  descriptors_exists_cache_ =
      std::make_unique<ThreadSafeLRUCache<image_t, bool>>(  //指向一个用于存储图像描述符是否存在的对象
          cache_size_, [this](const image_t image_id) {
            std::lock_guard<std::mutex> lock(database_mutex_);
            return std::make_shared<bool>(
                database_->ExistsDescriptors(image_id));
          });  //同上
}  
//该构造函数实现了对关键点的缓存，对描述符的缓存和对存在性信息的缓存，借助lambda表达式和互斥锁保证线程安全

void FeatureMatcherCache::AccessDatabase(
    const std::function<void(const Database& database)>& func) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  func(*database_);
} //在锁定数据库的情况下执行传入的操作，确保对数据库的操作是线程安全的

const Camera& FeatureMatcherCache::GetCamera(const camera_t camera_id) {
  MaybeLoadCameras();
  return cameras_cache_->at(camera_id);  //通过智能指针获取对应相机id的camera对象并返回其常量引用
}  //获取指定相机id的Camera对象

const Image& FeatureMatcherCache::GetImage(const image_t image_id) {
  MaybeLoadImages();
  return images_cache_->at(image_id);
} //同上，获取指定imageID的image对象

const PosePrior* FeatureMatcherCache::GetPosePriorOrNull(
    const image_t image_id) {  //获取图像id的姿态先验信息
    //图像 ID 的姿态先验信息指的是与特定图像 ID 相关联的、用于描述该图像姿态的先验知识。
    //例如，在某些场景中，可能根据图像的特征和一些先验模型，为每个图像 ID 存储一些关于其可能的拍摄角度、方向等方面的信息，这些信息就是图像 ID 的姿态先验信息。
  MaybeLoadPosePriors();  //加载姿态先验信息到缓存中
  const auto it = pose_priors_cache_->find(image_id);  
  if (it == pose_priors_cache_->end()) {
    return nullptr;
  } 
  return &it->second;
//在pose_priors_cache_中查找对应图像 ID 的姿态先验信息。
//如果找到，则返回指向该姿态先验信息的指针；如果未找到，则返回nullptr。
}

std::shared_ptr<FeatureKeypoints> FeatureMatcherCache::GetKeypoints(
    const image_t image_id) {
  return keypoints_cache_->Get(image_id);
}  //获取指定图像ID的特征关键点

std::shared_ptr<FeatureDescriptors> FeatureMatcherCache::GetDescriptors(
    const image_t image_id) {
  return descriptors_cache_->Get(image_id);
}  //获取指定图像ID的特征描述符

FeatureMatches FeatureMatcherCache::GetMatches(const image_t image_id1,
                                               const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ReadMatches(image_id1, image_id2);
}  //获取指定两个图像ID之间的特征匹配

std::vector<image_t> FeatureMatcherCache::GetImageIds() {
  MaybeLoadImages();  //加载图像信息到缓存中

  std::vector<image_t> image_ids; 
  image_ids.reserve(images_cache_->size());
  for (const auto& image : *images_cache_) {  //通过遍历images_cache_中的图像，将图像ID添加到image_ids向量中
    image_ids.push_back(image.first);
  }
  // Sort the images for deterministic behavior. Note that the images_cache_ is
  // an unordered_map, which does not guarantee a deterministic order across
  // different standard library implementations.
  std::sort(image_ids.begin(), image_ids.end());
  //保证可行性对image_ids进行排序
  return image_ids;
}

ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>&
FeatureMatcherCache::GetFeatureDescriptorIndexCache() {
  return descriptor_index_cache_;
}  //获取特征描述符索引缓存对象的引用

bool FeatureMatcherCache::ExistsKeypoints(const image_t image_id) {
  return *keypoints_exists_cache_->Get(image_id);
}  //检查指定图像 ID 的关键点是否存在

bool FeatureMatcherCache::ExistsDescriptors(const image_t image_id) {
  return *descriptors_exists_cache_->Get(image_id);
}  //检查指定图像 ID 的描述符是否存在

bool FeatureMatcherCache::ExistsMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ExistsMatches(image_id1, image_id2);
}  //检查指定两个图像 ID 之间的匹配是否存在

bool FeatureMatcherCache::ExistsInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->ExistsInlierMatches(image_id1, image_id2);
}  //检查指定两个图像 ID 之间的内点匹配是否存在

void FeatureMatcherCache::WriteMatches(const image_t image_id1,
                                       const image_t image_id2,
                                       const FeatureMatches& matches) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->WriteMatches(image_id1, image_id2, matches);
}  //将指定两个图像ID间的匹配信息写进数据库

void FeatureMatcherCache::WriteTwoViewGeometry(
    const image_t image_id1,
    const image_t image_id2,
    const TwoViewGeometry& two_view_geometry) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
}  //将指定两个图像 ID 之间的两视图几何信息写入数据库

void FeatureMatcherCache::DeleteMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->DeleteMatches(image_id1, image_id2);
}  //删除两个图像ID间的匹配信息

void FeatureMatcherCache::DeleteInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::lock_guard<std::mutex> lock(database_mutex_);
  database_->DeleteInlierMatches(image_id1, image_id2);
}  //删除两个图像ID之间的内点匹配信息

size_t FeatureMatcherCache::MaxNumKeypoints() {
  std::lock_guard<std::mutex> lock(database_mutex_);
  return database_->MaxNumKeypoints();
}  //获取数据库的最大关键点数量

void FeatureMatcherCache::MaybeLoadCameras() {  //加载相机信息到缓存中
  if (cameras_cache_) {
    return;
  }

  std::lock_guard<std::mutex> lock(database_mutex_);  //互斥锁锁定
  std::vector<Camera> cameras = database_->ReadAllCameras();  //获取所有的相机信息并将其存储在cameras中
  cameras_cache_ = std::make_unique<std::unordered_map<camera_t, Camera>>();
//创建一个指向std::unordered_map<camera_t, Camera>类型的智能指针cameras_cache_，用于存储相机信息的缓存。
  cameras_cache_->reserve(cameras.size()); //为unordered—map预留足够的空间
  for (Camera& camera : cameras) {
    cameras_cache_->emplace(camera.camera_id, std::move(camera));
  }
//遍历获取的相机信息向量cameras。对于每个相机，使用emplace方法将相机的 ID 和相机对象本身插入到cameras_cache_所指向的unordered_map中。    
//std::move 是 C++ 标准库中的一个函数模板，它的作用是将一个对象的状态从一个地方转移到另一个地方，通常用于高效地转移资源所有权而不进行拷贝操作。
}

void FeatureMatcherCache::MaybeLoadImages() {
  if (images_cache_) {
    return;
  }

  std::lock_guard<std::mutex> lock(database_mutex_);
  std::vector<Image> images = database_->ReadAllImages();
  images_cache_ = std::make_unique<std::unordered_map<image_t, Image>>();
  images_cache_->reserve(images.size());
  for (Image& image : images) {
    images_cache_->emplace(image.ImageId(), std::move(image));
  }
}
//同上

void FeatureMatcherCache::MaybeLoadPosePriors() {
  if (pose_priors_cache_) {
    return;
  }

  MaybeLoadImages();

  std::lock_guard<std::mutex> lock(database_mutex_);
  pose_priors_cache_ =
      std::make_unique<std::unordered_map<image_t, PosePrior>>();
  pose_priors_cache_->reserve(database_->NumPosePriors());
  for (const auto& image : *images_cache_) {
    if (database_->ExistsPosePrior(image.first)) {
        //检查数据库中是否存在该图像的姿态先验信息，若存在，插入到pose_priors_cache_所指向的unordered_map中。
      pose_priors_cache_->emplace(image.first,
                                  database_->ReadPosePrior(image.first));
    }
  }
}
//

}
// namespace colmap
//FeatureMatcherCache 类实现了对图像特征匹配相关数据的高效缓存管理以及安全的数据库操作，
//方便在多线程环境下对诸如相机、图像、特征匹配、姿态先验等各类数据进行快速获取、检查以及修改等操作
