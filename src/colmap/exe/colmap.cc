#include "colmap/exe/database.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/gui.h"
#include "colmap/exe/image.h"
#include "colmap/exe/model.h"
#include "colmap/exe/mvs.h"
#include "colmap/exe/sfm.h"
#include "colmap/exe/vocab_tree.h"
#include "colmap/util/version.h"

namespace {

// 定义一个类型，用于命令函数的指针
typedef std::function<int(int, char**)> command_func_t;

// 显示帮助信息的函数
int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  std::cout << colmap::StringPrintf(
                   "%s -- Structure-from-Motion and Multi-View Stereo\n(%s)",
                   colmap::GetVersionInfo().c_str(),
                   colmap::GetBuildInfo().c_str())
            << std::endl
            << std::endl;

  // 输出使用说明
  std::cout << "Usage:" << std::endl;
  std::cout << "  colmap [command] [options]" << std::endl << std::endl;

  // 输出文档链接
  std::cout << "Documentation:" << std::endl;
  std::cout << "  https://colmap.github.io/" << std::endl << std::endl;

  // 输出示例用法
  std::cout << "Example usage:" << std::endl;
  std::cout << "  colmap help [ -h, --help ]" << std::endl;
  std::cout << "  colmap gui" << std::endl;
  std::cout << "  colmap gui -h [ --help ]" << std::endl;
  std::cout << "  colmap automatic_reconstructor -h [ --help ]" << std::endl;
  std::cout << "  colmap automatic_reconstructor --image_path IMAGES "
               "--workspace_path WORKSPACE"
            << std::endl;
  std::cout << "  colmap feature_extractor --image_path IMAGES --database_path "
               "DATABASE"
            << std::endl;
  std::cout << "  colmap exhaustive_matcher --database_path DATABASE"
            << std::endl;
  std::cout << "  colmap mapper --image_path IMAGES --database_path DATABASE "
               "--output_path MODEL"
            << std::endl;
  std::cout << "  ..." << std::endl << std::endl;

  // 输出可用命令
  std::cout << "Available commands:" << std::endl;
  std::cout << "  help" << std::endl;
  for (const auto& command : commands) {
    std::cout << "  " << command.first << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

}  // namespace


int main(int argc, char** argv) {
  // 初始化日志系统
  colmap::InitializeGlog(argv);
#if defined(COLMAP_GUI_ENABLED)
  Q_INIT_RESOURCE(resources);
#endif
  //commands中存储了所有可以执行的功能
  // 定义命令列表，每个命令关联一个函数
  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("gui", &colmap::RunGraphicalUserInterface);
  commands.emplace_back("automatic_reconstructor",
                        &colmap::RunAutomaticReconstructor);
  commands.emplace_back("bundle_adjuster", &colmap::RunBundleAdjuster);
  commands.emplace_back("color_extractor", &colmap::RunColorExtractor);
  commands.emplace_back("database_cleaner", &colmap::RunDatabaseCleaner);
  commands.emplace_back("database_creator", &colmap::RunDatabaseCreator);
  commands.emplace_back("database_merger", &colmap::RunDatabaseMerger);
  commands.emplace_back("delaunay_mesher", &colmap::RunDelaunayMesher);
  commands.emplace_back("exhaustive_matcher", &colmap::RunExhaustiveMatcher);
  commands.emplace_back("feature_extractor", &colmap::RunFeatureExtractor);
  commands.emplace_back("feature_importer", &colmap::RunFeatureImporter);
  commands.emplace_back("hierarchical_mapper", &colmap::RunHierarchicalMapper);
  commands.emplace_back("image_deleter", &colmap::RunImageDeleter);
  commands.emplace_back("image_filterer", &colmap::RunImageFilterer);
  commands.emplace_back("image_rectifier", &colmap::RunImageRectifier);
  commands.emplace_back("image_registrator", &colmap::RunImageRegistrator);
  commands.emplace_back("image_undistorter", &colmap::RunImageUndistorter);
  commands.emplace_back("image_undistorter_standalone",
                        &colmap::RunImageUndistorterStandalone);
  commands.emplace_back("mapper", &colmap::RunMapper);
  commands.emplace_back("matches_importer", &colmap::RunMatchesImporter);
  commands.emplace_back("model_aligner", &colmap::RunModelAligner);
  commands.emplace_back("model_analyzer", &colmap::RunModelAnalyzer);
  commands.emplace_back("model_comparer", &colmap::RunModelComparer);
  commands.emplace_back("model_converter", &colmap::RunModelConverter);
  commands.emplace_back("model_cropper", &colmap::RunModelCropper);
  commands.emplace_back("model_merger", &colmap::RunModelMerger);
  commands.emplace_back("model_orientation_aligner",
                        &colmap::RunModelOrientationAligner);
  commands.emplace_back("model_splitter", &colmap::RunModelSplitter);
  commands.emplace_back("model_transformer", &colmap::RunModelTransformer);
  commands.emplace_back("patch_match_stereo", &colmap::RunPatchMatchStereo);
  commands.emplace_back("point_filtering", &colmap::RunPointFiltering);
  commands.emplace_back("point_triangulator", &colmap::RunPointTriangulator);
  commands.emplace_back("pose_prior_mapper", &colmap::RunPosePriorMapper);
  commands.emplace_back("poisson_mesher", &colmap::RunPoissonMesher);
  commands.emplace_back("project_generator", &colmap::RunProjectGenerator);
  commands.emplace_back("rig_bundle_adjuster", &colmap::RunRigBundleAdjuster);
  commands.emplace_back("sequential_matcher", &colmap::RunSequentialMatcher);
  commands.emplace_back("spatial_matcher", &colmap::RunSpatialMatcher);
  commands.emplace_back("stereo_fusion", &colmap::RunStereoFuser);
  commands.emplace_back("transitive_matcher", &colmap::RunTransitiveMatcher);
  commands.emplace_back("vocab_tree_builder", &colmap::RunVocabTreeBuilder);
  commands.emplace_back("vocab_tree_matcher", &colmap::RunVocabTreeMatcher);
  commands.emplace_back("vocab_tree_retriever", &colmap::RunVocabTreeRetriever);

  // 如果没有输入参数或请求帮助，显示帮助信息
  if (argc == 1) {
    return ShowHelp(commands);
  }

  // 解析用户输入的命令
  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    return ShowHelp(commands);
  } else {
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        matched_command_func = command_func.second;
        break;
      }
    }
    // 如果命令匹配，执行对应函数；否则，显示错误信息
    if (matched_command_func == nullptr) {
      LOG(ERROR) << colmap::StringPrintf(
          "Command `%s` not recognized. To list the "
          "available commands, run `colmap help`.",
          command.c_str());
      return EXIT_FAILURE;
    } else {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      return matched_command_func(command_argc, command_argv);
    }
  }

  return ShowHelp(commands);
}