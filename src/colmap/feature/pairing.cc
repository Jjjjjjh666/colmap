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

#include "colmap/feature/pairing.h"

#include "colmap/feature/utils.h"
#include "colmap/geometry/gps.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <fstream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {
namespace {

std::vector<std::pair<image_t, image_t>> ReadImagePairsText(
    const std::string& path,  //读取文本的路径
    const std::unordered_map<std::string, image_t>& image_name_to_image_id) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);  //尝试打开文件并检查是否打开成功

  std::string line;  //存储从文件中读取的每一行内容
  std::vector<std::pair<image_t, image_t>> image_pairs;  //存储有效的图像对
  std::unordered_set<image_pair_t> image_pairs_set;  //跟踪已处理过的图像对，避免重复添加相同的图像对，储存唯一的图像对id
  while (std::getline(file, line)) {   //逐行读取文件到line中
    StringTrim(&line);  //除去字符串中多余的空白字符，确保没有无效空格的干扰

    if (line.empty() || line[0] == '#') {  //跳过空行和注释行
      continue;
    }

    std::stringstream line_stream(line);   //创建一个std::stringstream对象line_stream，将当前行line的内容放入其中

    std::string image_name1;
    std::string image_name2;

    std::getline(line_stream, image_name1, ' ');  //声明两个string变量并通过getline以空格为分隔符从linestream中提取两个图像名称
    StringTrim(&image_name1);                     //分别代表两个图像的名称，并对他们进行stringtrim处理，去除图像名称中的多余空格符
    std::getline(line_stream, image_name2, ' ');
    StringTrim(&image_name2);

    if (image_name_to_image_id.count(image_name1) == 0) {    //检查图像名称是否在映射表中，返回0表示图像不存在
      LOG(ERROR) << "Image " << image_name1 << " does not exist.";  //输出错误日志
      continue;  //跳过该行前往下一行
    }
    if (image_name_to_image_id.count(image_name2) == 0) {
      LOG(ERROR) << "Image " << image_name2 << " does not exist.";
      continue;
    }
    //检查从当前行解析出的两个图像名称image_name1和image_name2是否在image_name_to_image_id映射中存在
    //若不存在，则输出错误信息，跳出当前循环，不处理该图像
    //若两个映射都存在，则从映射中获取各自对应的image_id
    const image_t image_id1 = image_name_to_image_id.at(image_name1);
    const image_t image_id2 = image_name_to_image_id.at(image_name2);
    const image_pair_t image_pair =
        Database::ImagePairToPairId(image_id1, image_id2);  //将两组图像id生成一个唯一的图像对
    const bool image_pair_exists = image_pairs_set.insert(image_pair).second;  //将生成的图像对尝试插入到image_pairs_set中，若插入成功
    //若插入成功，即先前没有处理过，则将其添加到image_pairs向量中                                                                          
    if (image_pair_exists) {
      image_pairs.emplace_back(image_id1, image_id2);  //储存有效的图像对 加入image_pairs中
    }
  }
  return image_pairs;
}
//该函数实现从用户文件中读取图像信息并将其进行筛选处理，转化为相应的图像ID生成图像对进行存储

}  // namespace

bool ExhaustiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 1);
  return true;
}

bool VocabTreeMatchingOptions::Check() const {
  CHECK_OPTION_GT(num_images, 0);
  CHECK_OPTION_GT(num_nearest_neighbors, 0);
  CHECK_OPTION_GT(num_checks, 0);
  return true;
}

bool SequentialMatchingOptions::Check() const {
  CHECK_OPTION_GT(overlap, 0);
  CHECK_OPTION_GT(loop_detection_period, 0);
  CHECK_OPTION_GT(loop_detection_num_images, 0);
  CHECK_OPTION_GT(loop_detection_num_nearest_neighbors, 0);
  CHECK_OPTION_GT(loop_detection_num_checks, 0);
  return true;
}
//以上三个函数均是对各自对象的参数所进行的有效性检查

VocabTreeMatchingOptions SequentialMatchingOptions::VocabTreeOptions() const {
  VocabTreeMatchingOptions options;
  options.num_images = loop_detection_num_images;
  options.num_nearest_neighbors = loop_detection_num_nearest_neighbors;
  options.num_checks = loop_detection_num_checks;
  options.num_images_after_verification =
      loop_detection_num_images_after_verification;
  options.max_num_features = loop_detection_max_num_features;
  options.vocab_tree_path = vocab_tree_path;
  return options;
}
//从SequentialMatchingOptions对象中提取一部分特定的参数值，创建一个VocabTreeMatchingOptions对象
//并将提取的参数值赋给新创建对象的相应成员变量，最后返回这个新的VocabTreeMatchingOptions对象（类似于深拷贝）

bool SpatialMatchingOptions::Check() const {
  CHECK_OPTION_GT(max_num_neighbors, 0);
  CHECK_OPTION_GT(max_distance, 0.0);
  return true;
}

bool TransitiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(batch_size, 0);
  CHECK_OPTION_GT(num_iterations, 0);
  return true;
}

bool ImagePairsMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 0);
  return true;
}

bool FeaturePairsMatchingOptions::Check() const { return true; }
//参数检查

std::vector<std::pair<image_t, image_t>> PairGenerator::AllPairs() {
  std::vector<std::pair<image_t, image_t>> image_pairs;
  while (!this->HasFinished()) {   //检查生成过程是否完成
    std::vector<std::pair<image_t, image_t>> image_pairs_block = this->Next();  //调用this->NEXT生成一组图像对块
    image_pairs.insert(image_pairs.end(),
                       std::make_move_iterator(image_pairs_block.begin()),
                       std::make_move_iterator(image_pairs_block.end()));
      //使用insert函数将image_pairs_block中的图像对移动到image_pairs向量中。
      //优化手段，move_iterator：将普通迭代器转化为一个移动迭代器，将元素从一个容器转移到另一个容器中
      //起到移动元素的作用，而非简单的拷贝
  }
  return image_pairs;
}
//该函数整体上实现了对容器的初始化，通过next函数生成一批图像对并存储在image_pairs_block中，借助移动语句
//将图像对高效地插入到image_pairs中，图像对生成结束后，循环结束并返回全部图像对

ExhaustivePairGenerator::ExhaustivePairGenerator(
    const ExhaustiveMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)  //shared_ptr是c++中的一种智能指针类型，用于管理动态分配对象的生命周期
    : options_(options),
      image_ids_(THROW_CHECK_NOTNULL(cache)->GetImageIds()),  //检查指针非空，获取图像标识
      block_size_(static_cast<size_t>(options_.block_size)),  //指定块大小
      num_blocks_(static_cast<size_t>(
          std::ceil(static_cast<double>(image_ids_.size()) / block_size_))) {  //通过计算标识符数量除以块大小并向上取整获得块的数量
  THROW_CHECK(options.Check());  //检查匹配选项是否有效
  LOG(INFO) << "Generating exhaustive image pairs...";
  const size_t num_pairs_per_block = block_size_ * (block_size_ - 1) / 2;  //计算每个块中的图像对数量
  image_pairs_.reserve(num_pairs_per_block);   //使用reserve函数为image_pairs_向量预留足够的空间
}

ExhaustivePairGenerator::ExhaustivePairGenerator(
    const ExhaustiveMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : ExhaustivePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}
//调用第一个构造函数来初始化对象，在调用第一个构造函数时，创建了一个FeatureMatcherCache对象
//使用options.CacheSize()设置缓存大小，并传入database指针（同样经过非空检查）。

void ExhaustivePairGenerator::Reset() {
  start_idx1_ = 0;
  start_idx2_ = 0;
}
//重置函数

bool ExhaustivePairGenerator::HasFinished() const {
  return start_idx1_ >= image_ids_.size();
}
//判断图像对的生成过程是否完成

std::vector<std::pair<image_t, image_t>> ExhaustivePairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }
//先清空image_pairs_向量，然后判断生成过程是否结束，若结束，则返回空的image_pairs_向量
  const size_t end_idx1 =
      std::min(image_ids_.size(), start_idx1_ + block_size_) - 1;
  const size_t end_idx2 =
      std::min(image_ids_.size(), start_idx2_ + block_size_) - 1;
//计算当前块的结束索引end_idx1和end_idx2。
//这里通过取image_ids_.size()和当前索引加上block_size_的较小值再减1来确定，确保不会超出图像标识符向量的范围

  LOG(INFO) << StringPrintf("Matching block [%d/%d, %d/%d]",
                            start_idx1_ / block_size_ + 1,
                            num_blocks_,
                            start_idx2_ / block_size_ + 1,
                            num_blocks_);
//输出显示当前匹配的块在整个块序列中的位置
//其中start_idx1_ / block_size_ + 1和start_idx2_ / block_size_ + 1分别表示当前块在start_idx1_和start_idx2_方向上的索引（从1开始计数）

  for (size_t idx1 = start_idx1_; idx1 <= end_idx1; ++idx1) {
    for (size_t idx2 = start_idx2_; idx2 <= end_idx2; ++idx2) {   //idx1和idx2遍历当前块所有的图像对组合
      const size_t block_id1 = idx1 % block_size_;
      const size_t block_id2 = idx2 % block_size_;    //计算块内位置，确保块内索引相对位置判断
      if ((idx1 > idx2 && block_id1 <= block_id2) ||   //避免同一对反向重复
          (idx1 < idx2 && block_id1 < block_id2)) {  // Avoid duplicate pairs  //确保按照升序生成  
        image_pairs_.emplace_back(image_ids_[idx1], image_ids_[idx2]);
      }
    }
  }
//通过嵌套循环用于生成当前块的图像对，同时通过条件判断避免生成重复的图像对
  start_idx2_ += block_size_;
  if (start_idx2_ >= image_ids_.size()) {
    start_idx2_ = 0;
    start_idx1_ += block_size_;
  }
//更新index2的值，若index2已经超过了idsize，说明当前index1下的所有情况已经得到处理，index2置零，index1+block_size切换到下一行处理
  return image_pairs_;
}
//生成图像对，通过分块方式处理大规模图像集合，同时避免重复组合
//整体实现逻辑：初始化并清空容器，计算当前接结束索引，嵌套循环生成图像对并避免重复，更新块索引，最后返回当前块的图像对



VocabTreePairGenerator::VocabTreePairGenerator(   //使用字典树生成图像对（Trie）
    const VocabTreeMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const std::vector<image_t>& query_image_ids)
    : options_(options),  
      cache_(THROW_CHECK_NOTNULL(cache)),
      thread_pool(options_.num_threads),  //初始化线程池
      queue(options_.num_threads) {
  THROW_CHECK(options.Check());  //检查匹配是否有效
  LOG(INFO) << "Generating image pairs with vocabulary tree...";

  // Read the pre-trained vocabulary tree from disk.
  visual_index_.Read(options_.vocab_tree_path);  //读取字典树

  const std::vector<image_t> all_image_ids = cache_->GetImageIds();  //获取所有图像标识符
  if (query_image_ids.size() > 0) {
    query_image_ids_ = query_image_ids;  //若查询图像标识符的向量大小大于0，则将query_image_ids_设置为传入的query_image_ids，表示使用传入的查询图像标识符。
  } else if (options_.match_list_path == "") {
    query_image_ids_ = cache_->GetImageIds();
    //如果options_.match_list_path为空字符串，则将query_image_ids_设置为all_image_ids，即使用所有图像作为查询图像
  } else {
    // Map image names to image identifiers.
    std::unordered_map<std::string, image_t> image_name_to_image_id; //将图像名称映射到图像标识符
    image_name_to_image_id.reserve(all_image_ids.size());  //空间预留
    for (const auto image_id : all_image_ids) {
      const auto& image = cache_->GetImage(image_id);
      image_name_to_image_id.emplace(image.Name(), image_id);
    }  //将每个图像名称和标识符添加到映射中

    // Read the match list path.
    std::ifstream file(options_.match_list_path);
    THROW_CHECK_FILE_OPEN(file, options_.match_list_path);  //打开文件并检查是否打开成功
    std::string line;
    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty() || line[0] == '#') {
        continue;
      }
//逐行读取内容，去掉空行和注释行
      if (image_name_to_image_id.count(line) == 0) {
        LOG(ERROR) << "Image " << line << " does not exist.";
      } else {
        query_image_ids_.push_back(image_name_to_image_id.at(line));
      }
    }
    //对于每行内容，如果在image_name_to_image_id映射中存在对应的图像名称
    //则将其对应的标识符添加到query_image_ids_向量中，不存在则输出错误信息
  }

  IndexImages(all_image_ids);

  query_options_.max_num_images = options_.num_images;
  query_options_.num_neighbors = options_.num_nearest_neighbors;
  query_options_.num_checks = options_.num_checks;
  query_options_.num_images_after_verification =
      options_.num_images_after_verification;
//将query_options_的各个成员变量设置为options_中相应的参数值
}
//初始化一个基于字典树的图像对生成器

VocabTreePairGenerator::VocabTreePairGenerator(
    const VocabTreeMatchingOptions& options,
    const std::shared_ptr<Database>& database,
    const std::vector<image_t>& query_image_ids)
    : VocabTreePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(options.CacheSize(),
                                                THROW_CHECK_NOTNULL(database)),
          query_image_ids) {}

void VocabTreePairGenerator::Reset() {
  query_idx_ = 0;
  result_idx_ = 0;
}
//重置内部索引

bool VocabTreePairGenerator::HasFinished() const {
  return result_idx_ >= query_image_ids_.size();
}
//检查处理过程是否完成

std::vector<std::pair<image_t, image_t>> VocabTreePairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }
  if (query_idx_ == 0) {  //处理刚开始
    // Initially, make all retrieval threads busy and continue with the
    // matching.
    const size_t init_num_tasks =
        std::min(query_image_ids_.size(), 2 * thread_pool.NumThreads());
      //计算初始化任务数量：查询图像数量和两倍线程池数量中的较小者
    for (; query_idx_ < init_num_tasks; ++query_idx_) {
      thread_pool.AddTask(
          &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_]);
    }  //通过循环将任务添加到线程池中
  }
    //启动多个任务，利用多线程处理

  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", result_idx_ + 1, query_image_ids_.size());

  // Push the next image to the retrieval queue.
  if (query_idx_ < query_image_ids_.size()) {
    thread_pool.AddTask(
        &VocabTreePairGenerator::Query, this, query_image_ids_[query_idx_++]);
  }
    //若还有未查询的图像，则添加新的查询任务

  // Pop the next results from the retrieval queue.
  auto retrieval = queue.Pop();
  THROW_CHECK(retrieval.IsValid());
//获取检索结果并检查其是否有效

  const auto& image_id = retrieval.Data().image_id;
  const auto& image_scores = retrieval.Data().image_scores;
//获取图像标识符和图像匹配分数信息

  // Compose the image pairs from the scores.
  image_pairs_.reserve(image_scores.size());
  for (const auto image_score : image_scores) {
    image_pairs_.emplace_back(image_id, image_score.image_id);  //为每个分数对应的图像创建一个图像对并将其添加到image_pairs_中
  }
  ++result_idx_;  //更新结果索引
  return image_pairs_;
}
//该函数通过并行查询和检索图像匹配结果，逐步生成图像对并返回，主要功能是利用多线程和队列实现高效的图像检索
//并将匹配到的图像对返回

void VocabTreePairGenerator::IndexImages(
    const std::vector<image_t>& image_ids) {  //对给定的图像标识符向量中的图像进行索引
  retrieval::VisualIndex<>::IndexOptions index_options;  //存储索引的相关选项
  index_options.num_threads = options_.num_threads;  
  index_options.num_checks = options_.num_checks;
//获取线程数和检查数并赋值给相应成员

  for (size_t i = 0; i < image_ids.size(); ++i) {
    Timer timer;
    timer.Start();
    LOG(INFO) << StringPrintf(
        "Indexing image [%d/%d]", i + 1, image_ids.size());
    //对于每个图像，创建一个Timer对象并启动它，用于记录索引该图像所花费的时间。
    //同时输出一条日志信息，显示正在索引的图像的进度（当前索引的图像序号和总图像数）。
    auto keypoints = *cache_->GetKeypoints(image_ids[i]);
    auto descriptors = *cache_->GetDescriptors(image_ids[i]);
    //根据图像标识通过cache_获取当前图像的关键点和描述符
      
    if (options_.max_num_features > 0 &&
        descriptors.rows() > options_.max_num_features) {
      ExtractTopScaleFeatures(
          &keypoints, &descriptors, options_.max_num_features);
    }
    //检查options_中设置的最大特征数量（max_num_features）是否大于 0，并且当前图像的描述符行数是否超过这个最大数量。
    //如果是，则调用ExtractTopScaleFeatures函数来提取前max个最重要的特征
    visual_index_.Add(index_options, image_ids[i], keypoints, descriptors);
    //将图像添加到视觉索引中
    LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  }

  // Compute the TF-IDF weights, etc.
  visual_index_.Prepare();
    //完成索引的准备工作，用于后续的图像检索
}
//函数整体功能实现：
//首先设置索引操作的相关选项，包括线程数和检查数
//再来遍历索引的图像集合，从缓存中获取图像的关键点和描述符并提取重要特征，再将图像添加到视觉索引中
//最后完成索引信息的准备工作

void VocabTreePairGenerator::Query(const image_t image_id) {
  auto keypoints = *cache_->GetKeypoints(image_id);
  auto descriptors = *cache_->GetDescriptors(image_id);  //获取图像关键点和描述符
  if (options_.max_num_features > 0 &&
      descriptors.rows() > options_.max_num_features) {
    ExtractTopScaleFeatures(
        &keypoints, &descriptors, options_.max_num_features);
  }  //提取重要特征

  Retrieval retrieval;  //储存查询结果相关信息
  retrieval.image_id = image_id;
  visual_index_.Query(
      query_options_, keypoints, descriptors, &retrieval.image_scores);  //查询

  THROW_CHECK(queue.Push(std::move(retrieval)));  //入队
}
//函数负责对给定图像 image_id 进行特征提取和检索，通过字典树查询算法生成图像匹配结果并将其存入检索队列 queue。
//该函数利用图像的关键点和描述符，通过字典树查询生成相似图像的匹配分数。

SequentialPairGenerator::SequentialPairGenerator(
    const SequentialMatchingOptions& options, //传入匹配选项
    const std::shared_ptr<FeatureMatcherCache>& cache)  //存储特征匹配的缓存信息
    : options_(options), cache_(THROW_CHECK_NOTNULL(cache)) {
  THROW_CHECK(options.Check());   //检查传入的匹配选项是否有效
  LOG(INFO) << "Generating sequential image pairs...";
  image_ids_ = GetOrderedImageIds();
  image_pairs_.reserve(options_.overlap);   //获取图像ID并预留空间

  if (options_.loop_detection) {
    std::vector<image_t> query_image_ids;
    for (size_t i = 0; i < image_ids_.size();
         i += options_.loop_detection_period) {
      query_image_ids.push_back(image_ids_[i]);
    }//按照loop_detection_period指定的步长从image_ids_中选取图像 ID，并将其添加到query_image_ids向量中
    vocab_tree_pair_generator_ = std::make_unique<VocabTreePairGenerator>(
        options_.VocabTreeOptions(), cache_, query_image_ids);
  }
}

SequentialPairGenerator::SequentialPairGenerator(
    const SequentialMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : SequentialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void SequentialPairGenerator::Reset() {
  image_idx_ = 0;
  if (vocab_tree_pair_generator_) {
    vocab_tree_pair_generator_->Reset();
  }
}
//重置函数

bool SequentialPairGenerator::HasFinished() const {
  return image_idx_ >= image_ids_.size() &&
         (vocab_tree_pair_generator_ ? vocab_tree_pair_generator_->HasFinished()
                                     : true);
}
//检查函数操作是否已全部完成

std::vector<std::pair<image_t, image_t>> SequentialPairGenerator::Next() {
  image_pairs_.clear();   //清空之前存储的图像信息，为生成新图像做准备
  if (image_idx_ >= image_ids_.size()) {   //检查是否完成所有的基本图像生成
    if (vocab_tree_pair_generator_) {   //若存在词汇树图像对生成器，则调用其next函数并返回结果
      return vocab_tree_pair_generator_->Next();
    }
    return image_pairs_;
  }
  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", image_idx_ + 1, image_ids_.size());

  const auto image_id1 = image_ids_.at(image_idx_);  //获取当前图像ID
  //用于生成当前图像和后续图像的图像对
  for (int i = 0; i < options_.overlap; ++i) {
    if (options_.quadratic_overlap) {  //若是指数形式重叠
        //通过image_idx_ + (1ull << i)计算下一个图像索引（1ull << i实现了以 2 为底数的指数增长）
      const size_t image_idx_2_quadratic = image_idx_ + (1ull << i);
      //如果该索引在image_ids_范围内，则将由image_id1和新获取的图像 ID 组成的图像对添加到image_pairs_容器中
      if (image_idx_2_quadratic < image_ids_.size()) {
        image_pairs_.emplace_back(image_id1,
                                  image_ids_.at(image_idx_2_quadratic));
      } else {
        break;  //否则若索引越界，则跳出循环
      }
    }
    //如果options_.quadratic_overlap为false，则通过image_idx_ + i + 1计算下一个图像索引，
    //并在索引合法的情况下将相应的图像对添加到image_pairs_容器中。
    else {
      const size_t image_idx_2 = image_idx_ + i + 1;
      if (image_idx_2 < image_ids_.size()) {
        image_pairs_.emplace_back(image_id1, image_ids_.at(image_idx_2));
      } else {
        break;
      }
    }
  }
  //更新图像对索引并返回图像对
  ++image_idx_;
  return image_pairs_;
}
//函数用于根据顺序遍历策略生成一批图像对。它按照固定或指数增长的步长，从一个图像 ID 开始，与后续的若干图像生成图像对，用于图像匹配任务。
//支持线性和指数重叠策略，当遍历完所有图像时，可以切换到基于词汇树的图像对生成器继续生成图像对。

std::vector<image_t> SequentialPairGenerator::GetOrderedImageIds() const {  //获取有序的图像ID
  const std::vector<image_t> image_ids = cache_->GetImageIds();  //获取所有图像ID

  std::vector<Image> ordered_images;
  ordered_images.reserve(image_ids.size());  //创建对象并预留空间
  for (const auto image_id : image_ids) {
    ordered_images.push_back(cache_->GetImage(image_id));
  }  //通过循环遍历每个图像ID并将其对应的图像添加到容器中

  std::sort(ordered_images.begin(),
            ordered_images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });  //按照图像名称的字典序对图像对象进行排序

  std::vector<image_t> ordered_image_ids;
  ordered_image_ids.reserve(image_ids.size());  //创建储存有序图像ID的容器并预留空间
  for (const auto& image : ordered_images) {
    ordered_image_ids.push_back(image.ImageId());
  }   //提取排序后图像的ID并将其添加到容器中

  return ordered_image_ids;
}
//该函数实现了对图像id的获取存储和根据字典序对其进行排序再添加到容器中返回

SpatialPairGenerator::SpatialPairGenerator(
    const SpatialMatchingOptions& options,  //匹配选项对象，包含空间图像配对的参数等
    const std::shared_ptr<FeatureMatcherCache>& cache)  //智能指针，用于缓存图像特征匹配的数据
    : options_(options), image_ids_(THROW_CHECK_NOTNULL(cache)->GetImageIds()) {
  LOG(INFO) << "Generating spatial image pairs...";
  THROW_CHECK(options.Check());  //检查空间匹配选项是否有效

  Timer timer;
  timer.Start();  //创建一个Timer对象timer并启动它，用于计时索引图像过程
  LOG(INFO) << "Indexing images...";

  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> position_matrix =
      ReadPositionPriorData(*cache);  //从缓存中读取图像位置的先验数据，存入矩阵中，每一行表示一个图像的位置
  const size_t num_positions = position_idxs_.size();   //获取位置索引的大小

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
  if (num_positions == 0) {  //若没有图像具有位置数据
    LOG(INFO) << "=> No images with location data.";
    return;
  }

  timer.Restart();
  LOG(INFO) << "Building search index...";
  //使用position_matrix的数据创建一个flann::Matrix<float>类型的positions对象
  //将位置矩阵包装成flann兼容矩阵
  flann::Matrix<float> positions(
      position_matrix.data(), num_positions, position_matrix.cols());

  flann::LinearIndexParams index_params;  //线性索引参数
  flann::LinearIndex<flann::L2<float>> search_index(index_params);   //创建搜索参数对象
  search_index.buildIndex(positions);    //根据位置创建线性搜索索引

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());

  timer.Restart();
  LOG(INFO) << "Searching for nearest neighbors...";
  //搜索最近邻部分
  knn_ = std::min<int>(options_.max_num_neighbors + 1, num_positions);  //根据options_.max_num_neighbors和num_positions确定要搜索的最近邻数量knn_
  image_pairs_.reserve(knn_);

  index_matrix_.resize(num_positions, knn_);//调整index_matrix_的大小为num_positions行和knn_列
  flann::Matrix<size_t> indices(index_matrix_.data(), num_positions, knn_); //使用其数据创建flann::Matrix<size_t>类型的indices对象，用于存储最近邻的索引

  distance_matrix_.resize(num_positions, knn_);  //储存最近邻的索引
  flann::Matrix<float> distances(distance_matrix_.data(), num_positions, knn_);  //存储最近邻的距离

  flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
  //？？？根据options_.num_threads和线程池最大线程数的关系设置search_params的cores属性
  // （如果options_.num_threads是最大线程数，则使用硬件并发线程数；否则使用options_.num_threads，如果计算出的线程数小于等于 0，则设置为 1）。
  if (options_.num_threads == ThreadPool::kMaxNumThreads) {
    search_params.cores = std::thread::hardware_concurrency();
  } else {
    search_params.cores = options_.num_threads;
  }
  if (search_params.cores <= 0) {
    search_params.cores = 1;
  }
  //调用knn算法进行最近邻搜索
  search_index.knnSearch(positions, indices, distances, knn_, search_params);

  LOG(INFO) << StringPrintf(" in %.3fs", timer.ElapsedSeconds());
}
//该函数是一个构造函数，用于生成基于图像位置数据的空间图像对
//通过读取图像位置的先验数据，使用flann库进行近邻搜索，构建空间上可能相关的图像对集合

SpatialPairGenerator::SpatialPairGenerator(
    const SpatialMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : SpatialPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void SpatialPairGenerator::Reset() { current_idx_ = 0; }

bool SpatialPairGenerator::HasFinished() const {
  return current_idx_ >= position_idxs_.size();
}

std::vector<std::pair<image_t, image_t>> SpatialPairGenerator::Next() {
  image_pairs_.clear();
  if (HasFinished()) {
    return image_pairs_;
  }

  LOG(INFO) << StringPrintf(
      "Matching image [%d/%d]", current_idx_ + 1, position_idxs_.size());
  const float max_distance =
      static_cast<float>(options_.max_distance * options_.max_distance);//计算最大阈值距离的平方
  for (int j = 0; j < knn_; ++j) {  //遍历当前图像索引的knn个最近邻
    // Check if query equals result.
    if (index_matrix_(current_idx_, j) == current_idx_) {
      continue;    //若当前最近邻索引与当前图像索引相同，则跳过本次循环
    }

    // Since the nearest neighbors are sorted by distance, we can break.
    if (distance_matrix_(current_idx_, j) > max_distance) {
      break;   //若当前索引下的图像距离大于最大距离，便可以直接break，因为最近邻是按照距离排序的
    }

    const image_t image_id = image_ids_.at(position_idxs_[current_idx_]);
    const size_t nn_idx = position_idxs_.at(index_matrix_(current_idx_, j));
    const image_t nn_image_id = image_ids_.at(nn_idx);
    image_pairs_.emplace_back(image_id, nn_image_id);
  }
  ++current_idx_;
  return image_pairs_;
}
//该函数主要用于生成一组空间图像对，将符合条件的最近邻添加到image_pairs_当中

Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>
SpatialPairGenerator::ReadPositionPriorData(FeatureMatcherCache& cache) {   //从cache中读取图像位置的先验数据
  GPSTransform gps_transform;
  std::vector<Eigen::Vector3d> ells(1);

  size_t num_positions = 0;
  position_idxs_.clear();
  position_idxs_.reserve(image_ids_.size());
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> position_matrix(
      image_ids_.size(), 3);  //用于存储图像位置信息的矩阵，每行存储一个图像的三维位置

  for (size_t i = 0; i < image_ids_.size(); ++i) {
    const PosePrior* pose_prior = cache.GetPosePriorOrNull(image_ids_[i]); //获取该图像ID下的位置先验指针
    if (pose_prior == nullptr) {
      continue;
    }
    const Eigen::Vector3d& position_prior = pose_prior->position;
    if ((position_prior(0) == 0 && position_prior(1) == 0 &&
         options_.ignore_z) ||
        (position_prior(0) == 0 && position_prior(1) == 0 &&
         position_prior(2) == 0 && !options_.ignore_z)) {  //若在水平面上没有位置信息或是完全没有位置信息，则跳过本次循环
      continue;
    }

    position_idxs_.push_back(i); //将当前图像的索引i添加到position_idxs_容器中，表示该索引对应的图像具有有效的位置信息。

    switch (pose_prior->coordinate_system) {
        //将position_prior向量中的值赋给ells向量的第一个元素（如果options_.ignore_z为true，则将高度设为 0）
      case PosePrior::CoordinateSystem::WGS84: {
        ells[0](0) = position_prior(0);
        ells[0](1) = position_prior(1);
        ells[0](2) = options_.ignore_z ? 0 : position_prior(2);
        //调用gps_transform.EllToXYZ函数将ells向量中的 GPS 坐标转换为笛卡尔坐标xyzs
        const auto xyzs = gps_transform.EllToXYZ(ells);
        position_matrix(num_positions, 0) = static_cast<float>(xyzs[0](0));
        position_matrix(num_positions, 1) = static_cast<float>(xyzs[0](1));
        position_matrix(num_positions, 2) = static_cast<float>(xyzs[0](2));
      } break;
    //如果坐标系统是UNDEFINED或其他未明确处理的情况，输出一条警告日志，表示未知的坐标系统，并按照笛卡尔坐标的方式处理：
    //将position_prior向量中的x、y和（根据options_.ignore_z决定的z）坐标值转换为float类型，并存储到position_matrix矩阵的当前行中。
      case PosePrior::CoordinateSystem::UNDEFINED:
      default:
        LOG(WARNING) << "Unknown coordinate system for image " << image_ids_[i]
                     << ", assuming cartesian.";
      case PosePrior::CoordinateSystem::CARTESIAN:
        position_matrix(num_positions, 0) =
            static_cast<float>(position_prior(0));
        position_matrix(num_positions, 1) =
            static_cast<float>(position_prior(1));
        position_matrix(num_positions, 2) =
            static_cast<float>(options_.ignore_z ? 0 : position_prior(2));
    }

    num_positions += 1;
  }
  return position_matrix;
}
//问
//这个函数的主要功能是从FeatureMatcherCache中读取图像的位置先验数据，
// 并将其转换为统一的笛卡尔坐标形式存储在position_matrix中。

TransitivePairGenerator::TransitivePairGenerator(
    const TransitiveMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options), cache_(cache) {
  THROW_CHECK(options.Check());
}
//接收匹配参数和特征匹配缓存，使用其数据初始化变量options和cache，检查options的参数合法性

TransitivePairGenerator::TransitivePairGenerator(
    const TransitiveMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : TransitivePairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}
//接收 TransitiveMatchingOptions 和 Database 对象。
//使用 std::make_shared<FeatureMatcherCache> 创建缓存，将缓存的大小设置为 options.CacheSize()，并确保 database 非空。
//调用第一个构造函数进行初始化。

void TransitivePairGenerator::Reset() {
  current_iteration_ = 0;
  current_batch_idx_ = 0;
  image_pairs_.clear();
  image_pair_ids_.clear();
}

bool TransitivePairGenerator::HasFinished() const {
  return current_iteration_ >= options_.num_iterations && image_pairs_.empty();
}

std::vector<std::pair<image_t, image_t>> TransitivePairGenerator::Next() {
  if (!image_pairs_.empty()) {  //若image_pairs_容器不为空，表示还有剩余图像对需要处理
    current_batch_idx_++;  //增加当前批次索引
    std::vector<std::pair<image_t, image_t>> batch;//创建一个空的std::vector<std::pair<image_t, image_t>>类型的batch容器，用于存储当前批次的图像对。
    while (!image_pairs_.empty() &&
           static_cast<int>(batch.size()) < options_.batch_size) {
      batch.push_back(image_pairs_.back());  //将图像对按批次大小填充到batch中，直到批次满或图像对耗尽
      image_pairs_.pop_back();
    }
    LOG(INFO) << StringPrintf(
        "Matching batch [%d/%d]", current_batch_idx_, current_num_batches_);
    return batch;
  }
  //若已完成所有迭代，则返回一个空的图像对
  if (current_iteration_ >= options_.num_iterations) {
    return {};
  }

  current_batch_idx_ = 0;
  current_num_batches_ = 0;
  current_iteration_++;   //初始化新一轮的迭代参数

  LOG(INFO) << StringPrintf(
      "Iteration [%d/%d]", current_iteration_, options_.num_iterations);

  std::vector<std::pair<image_t, image_t>> existing_image_pairs;  //存储已存在的图像对
  std::vector<int> existing_num_inliers;   //存储图像对对应的内点数
  //通过cache_->AccessDatabase函数访问数据库，并传入一个 lambda 函数。
  // 在 lambda 函数内部，调用database对象的ReadTwoViewGeometryNumInliers函数，将读取到的图像对和内点数分别存储到existing_image_pairs和existing_num_inliers容器中
  cache_->AccessDatabase(
      [&existing_image_pairs, &existing_num_inliers](const Database& database) {
        database.ReadTwoViewGeometryNumInliers(&existing_image_pairs,
                                               &existing_num_inliers);
      });
  //建立图像间的邻接关系
  std::unordered_map<image_t, std::vector<image_t>> adjacency;  //邻接表，记录每个图像与其直接相邻图像的关系
  for (const auto& image_pair : existing_image_pairs) {
   //将图像对中的第一个图像 ID 作为键，将第二个图像 ID 添加到对应的值向量中，表示这两个图像是相邻的。
   // 同样，将第二个图像 ID 作为键，将第一个图像 ID 添加到对应的值向量中
    adjacency[image_pair.first].push_back(image_pair.second);
    adjacency[image_pair.second].push_back(image_pair.first);
    image_pair_ids_.insert(
        Database::ImagePairToPairId(image_pair.first, image_pair.second));
  }
//基于现有的邻接关系生成新的图像对
  for (const auto& image : adjacency) {  //第一层，遍历每个图像image_id1
    const auto image_id1 = image.first;
    for (const auto& image_id2 : image.second) {   //第二层，遍历与image_id1相邻的image_id2
      const auto it = adjacency.find(image_id2);  //从image_id2的邻接表中查找第三个图像image_id3，构建（id1，id3）作为新的图像对
      if (it == adjacency.end()) {
        continue;
      }
      for (const auto& image_id3 : it->second) {
        if (image_id1 == image_id3) {  //排除image_id1==image_id3的情况
          continue;
        }
        const auto image_pair_id =
            Database::ImagePairToPairId(image_id1, image_id3);
        if (image_pair_ids_.count(image_pair_id) != 0) {  //避免生成已存在的图像对，利用 image_pair_ids_ 进行判重。
          continue;  
        }
        image_pairs_.emplace_back(image_id1, image_id3);
        image_pair_ids_.insert(image_pair_id);
      }
    }
  }
  //计算批次数量并递归调用
  current_num_batches_ =
      std::ceil(static_cast<double>(image_pairs_.size()) / options_.batch_size);

  return Next();
}
//该函数旨在逐步生成传递性图像对的下一批次
//它首先检查是否有剩余未处理的图像对，如果有，则按批次返回；如果没有剩余图像对，且当前迭代未完成，则基于已有图像对的邻接关系生成新的传递性图像对。


ImportedPairGenerator::ImportedPairGenerator(  //该类负责导入外部图像对
    const ImagePairsMatchingOptions& options,
    const std::shared_ptr<FeatureMatcherCache>& cache)
    : options_(options) {
  THROW_CHECK(options.Check());

  LOG(INFO) << "Importing image pairs...";
  const std::vector<image_t> image_ids = cache->GetImageIds();  //从缓存中获取图像id列表
  std::unordered_map<std::string, image_t> image_name_to_image_id;//将图像的名称映射到对应的图像ID
  image_name_to_image_id.reserve(image_ids.size());
  for (const auto image_id : image_ids) {  //循环遍历每个图像ID
    const auto& image = cache->GetImage(image_id);   //从缓存中获取对应的图像对象
    image_name_to_image_id.emplace(image.Name(), image_id);  //将图像名称和ID插入unordered_map，避免重复查找和储存
  }
  image_pairs_ =
      ReadImagePairsText(options_.match_list_path, image_name_to_image_id);  //读取指定路径中的图像对文件存储到导入的图像对
  block_image_pairs_.reserve(options_.block_size);
}

ImportedPairGenerator::ImportedPairGenerator(
    const ImagePairsMatchingOptions& options,
    const std::shared_ptr<Database>& database)
    : ImportedPairGenerator(
          options,
          std::make_shared<FeatureMatcherCache>(
              options.CacheSize(), THROW_CHECK_NOTNULL(database))) {}

void ImportedPairGenerator::Reset() { pair_idx_ = 0; }

bool ImportedPairGenerator::HasFinished() const {
  return pair_idx_ >= image_pairs_.size();
}

std::vector<std::pair<image_t, image_t>> ImportedPairGenerator::Next() {
  block_image_pairs_.clear();
  if (HasFinished()) {
    return block_image_pairs_;
  }

  LOG(INFO) << StringPrintf("Matching block [%d/%d]",
                            pair_idx_ / options_.block_size + 1,
                            image_pairs_.size() / options_.block_size + 1);
  //计算当前图像对块的结束索引block_end，取pair_idx_ + options_.block_size和image_pairs_.size()中的较小值。这确保了不会超出图像对的总数。
  //通过一个循环，从pair_idx_开始到block_end，将image_pairs_中的图像对依次添加到block_image_pairs_容器中，从而填充当前的图像对块。

  const size_t block_end =
      std::min(pair_idx_ + options_.block_size, image_pairs_.size());
  for (size_t j = pair_idx_; j < block_end; ++j) {
    block_image_pairs_.push_back(image_pairs_[j]);
  }
  pair_idx_ += options_.block_size;
  return block_image_pairs_;
}
//函数用于 按块返回导入的图像对。
//它从预先导入的图像对列表中，逐批次（块）提取一组图像对进行处理，直到所有图像对都被处理完。
//每次调用该函数都会返回下一批图像对，适用于分块执行特征匹配等操作。



}  // namespace colmap





  


     

  
  
   
  
  
  L
