/*
 * Copyright 2016 Jiang Chen <criver@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <memory>
#include <sys/stat.h>
#include <unordered_set>

#include "external/cppformat/format.h"

#include "src/base/base.h"
#include "src/data_store/data_store.h"
#include "src/data_store/flatfiles_data_store.h"
#include "src/data_store/tsv_data_store.h"
#include "src/gbdt_algo/evaluation.h"
#include "src/gbdt_algo/gbdt_algo.h"
#include "src/gbdt_algo/subsampling.h"
#include "src/gbdt_algo/utils.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/json_utils.h"
#include "src/utils/stopwatch.h"
#include "src/utils/utils.h"

DECLARE_string(config_file);
DECLARE_string(mode);
DECLARE_string(training_flatfiles_dirs);
DECLARE_string(testing_flatfiles_dirs);
DECLARE_string(training_tsvs);
DECLARE_string(testing_tsvs);
DECLARE_string(testing_model_file);
DECLARE_string(output_dir);
DECLARE_string(output_model_name);
DECLARE_int32(seed);

using gbdt::Config;
using gbdt::DataConfig;
using gbdt::DataStore;
using gbdt::FlatfilesDataStore;
using gbdt::TSVDataStore;
using gbdt::Forest;
using gbdt::Subsampling;

void Train();
void Test();

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_mode == "train") {
    Train();
  } else if (FLAGS_mode == "test") {
    Test();
  } else {
    LOG(FATAL) << "Wrong mode " << FLAGS_mode;
  }

  return 0;
}

string FeatureImportanceFormatted(const vector<pair<string, double>>& feature_importance) {
  vector<string> feature_importance_strs;
  for (const auto& p : feature_importance) {
    feature_importance_strs.push_back(fmt::format("{0}\t{1}", p.first, p.second));
  }
  return strings::JoinStrings(feature_importance_strs, "\n");
}

void GetHeaderAndTsvs(const string& tsv_flag, string* header_file, vector<string>* tsvs) {
  *tsvs = strings::split(tsv_flag, ",");
  CHECK_GE(tsvs->size(), 2) << "There should at least be two tsv files: one header and one tsv.";
  *header_file = (*tsvs)[0];
  tsvs->erase(tsvs->begin());
}

unique_ptr<DataStore> LoadTrainingDataStore(const DataConfig& config) {
  if (!FLAGS_training_flatfiles_dirs.empty()) {
    return unique_ptr<DataStore>(
    new FlatfilesDataStore(strings::split(FLAGS_training_flatfiles_dirs, ",")));
  } else if (!FLAGS_training_tsvs.empty()) {
    string header_file;
    vector<string> tsvs;
    GetHeaderAndTsvs(FLAGS_training_tsvs, &header_file, &tsvs);
    return unique_ptr<DataStore>(new TSVDataStore(header_file, tsvs, config.tsv_data_config()));
  } else {
    LOG(FATAL) << "Please specify --training_flatfiles_dirs or --training_tsvs.";
  }
  return nullptr;
}

unique_ptr<DataStore> LoadTestingDataStore(const DataConfig& config) {
  if (!FLAGS_testing_flatfiles_dirs.empty()) {
    return unique_ptr<DataStore>(
    new FlatfilesDataStore(strings::split(FLAGS_testing_flatfiles_dirs, ",")));
  } else if (!FLAGS_testing_tsvs.empty()) {
    string header_file;
    vector<string> tsvs;
    GetHeaderAndTsvs(FLAGS_testing_tsvs, &header_file, &tsvs);
    return unique_ptr<DataStore>(new TSVDataStore(header_file, tsvs, config.tsv_data_config()));
  } else {
    LOG(FATAL) << "Please specify --testing_flatfiles_dirs or --testing_tsvs.";
  }
  return nullptr;
}

void Train() {
  CHECK(!FLAGS_config_file.empty()) << "Please specify --config_file.";
  CHECK(!FLAGS_output_dir.empty()) << "Please specify --output_dir.";

  StopWatch stopwatch;
  stopwatch.Start();
  LOG(INFO) << "Start training.";

  // Reseed.
  Subsampling::Reseed(FLAGS_seed);

  // Load config.
  Config config;
  string config_text = ReadFileToStringOrDie(FLAGS_config_file);
  CHECK(JsonUtils::FromJson(config_text, &config))
      << "Failed to parse json to proto: " << config_text;

  // Load DataStore.
  auto data_store = LoadTrainingDataStore(config.data_config());
  CHECK(data_store) << "Failed to load DataStore.";

  // Start learning.
  unique_ptr<Forest> forest = TrainGBDT(config, data_store.get());
  CHECK(forest) << "Failed to tain a forest";

  // Write the model into a file.
  mkdir(FLAGS_output_dir.c_str(), 0744);
  string output_model_file = FLAGS_output_dir + "/" + FLAGS_output_model_name + ".json";
  WriteStringToFile(JsonUtils::ToJsonOrDie(*forest), output_model_file);
  LOG(INFO) << "Wrote the model to " << output_model_file;

  // Write the feature importance into a file.
  WriteStringToFile(FeatureImportanceFormatted(ComputeFeatureImportance(*forest)),
                    FLAGS_output_dir + "/" + FLAGS_output_model_name + ".fimps");
  stopwatch.End();
  LOG(INFO) << "Finished training in "
            << StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs()) << ".";
}

void Test() {
  CHECK(!FLAGS_config_file.empty()) << "Please specify --config_file.";
  CHECK(!FLAGS_testing_model_file.empty()) << "Please specify --testing_model_file";
  CHECK(!FLAGS_output_dir.empty()) << "Please specify --output_dir.";

  LOG(INFO) << "Start testing.";

  // Load config.
  string config_text = ReadFileToStringOrDie(FLAGS_config_file);
  Config config;
  CHECK(JsonUtils::FromJson(config_text, &config))
      << "Failed to parse json to proto " << config_text;

  // Load testing_model_file.
  string forest_text = ReadFileToStringOrDie(FLAGS_testing_model_file);
  Forest forest;
  CHECK(JsonUtils::FromJson(forest_text, &forest))
      << "Failed to parse json " << forest_text;
  LOG(INFO) << "Loaded a forest with " << forest.tree_size() << " trees.";

  // Load DataStore.
  auto data_store = LoadTestingDataStore(config.data_config());
  CHECK(data_store) << "Failed to load DataStore.";

  // Evaluate forest and write score out.
  mkdir(FLAGS_output_dir.c_str(), 0744);
  CHECK(EvaluateForest(data_store.get(), config, forest, FLAGS_output_dir))
      << "Failed to evaluate the forest.";

  LOG(INFO) << "Wrote testing outputs to " << FLAGS_output_dir;
  LOG(INFO) << "Finished testing.";
}
