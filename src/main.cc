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
#include <memory>
#include <sys/stat.h>

#include "external/cppformat/format.h"

#include "src/base/base.h"
#include "src/data_store/data_store.h"
#include "src/data_store/flatfiles_data_store.h"
#include "src/data_store/tsv_data_store.h"
#include "src/gbdt_algo/evaluation.h"
#include "src/gbdt_algo/gbdt_algo.h"
#include "src/gbdt_algo/utils.h"
#include "src/loss_func/loss_func.h"
#include "src/loss_func/loss_func_factory.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/json_utils.h"
#include "src/utils/stopwatch.h"
#include "src/utils/subsampling.h"
#include "src/utils/utils.h"

DECLARE_string(config_file);
DECLARE_string(mode);
DECLARE_string(flatfiles_dirs);
DECLARE_string(tsvs);
DECLARE_string(testing_model_file);
DECLARE_string(base_model_file);
DECLARE_string(output_dir);
DECLARE_string(output_model_name);
DECLARE_int32(seed);
DECLARE_int32(logbuflevel);

using gbdt::Config;
using gbdt::DataStore;
using gbdt::FlatfilesDataStore;
using gbdt::LoadForestOrDie;
using gbdt::LossFunc;
using gbdt::LossFuncFactory;
using gbdt::TSVDataStore;
using gbdt::Forest;
using gbdt::Subsampling;

void Train();
void Test();

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Set logbuflevel to -1 so that glog won't buffer LOG(INFO).
  FLAGS_logbuflevel = -1;
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

unique_ptr<DataStore> LoadDataStoreOrDie(const Config& config) {
  unique_ptr<DataStore> data_store;
  if (!FLAGS_flatfiles_dirs.empty()) {
    data_store.reset(
        new FlatfilesDataStore(strings::split(FLAGS_flatfiles_dirs, ",")));
  } else if (!FLAGS_tsvs.empty()) {
    data_store.reset(new TSVDataStore(strings::split(FLAGS_tsvs, ","), config));
  }
  if (!data_store) {
    LOG(FATAL) << "Failed to load data_store. Please either specify --flatfiles_dirs or --tsvs.";
  }
  if (!data_store->status().ok()) {
    LOG(FATAL) << "Failed to load data_store. Error message: " + data_store->status().ToString();
  }

  return data_store;
}

void Train() {
  CHECK(!FLAGS_config_file.empty()) << "Please specify --config_file.";
  CHECK(!FLAGS_output_dir.empty()) << "Please specify --output_dir.";

  LOG(INFO) << "Start training.";

  // Reseed.
  Subsampling::Reseed(FLAGS_seed);

  // Load config.
  Config config;
  string config_text = ReadFileToStringOrDie(FLAGS_config_file);
  auto status = JsonUtils::FromJson(config_text, &config);
  CHECK(status.ok()) << "Failed to parse json to proto: " << config_text;

  // Load DataStore.
  auto data_store = LoadDataStoreOrDie(config);

  // Load Base Forest if provided.
  unique_ptr<Forest> base_forest;
  if (!FLAGS_base_model_file.empty()) {
    base_forest.reset(new Forest);
    *base_forest = LoadForestOrDie(FLAGS_base_model_file);
  }

  // Initialize Loss Function.
  unique_ptr<LossFunc> loss_func = LossFuncFactory::CreateLossFunc(config);
  CHECK(loss_func) << "Failed to initialize loss func " << config.loss_func();

  // Start learning.
  Forest forest;
  status = TrainGBDT(data_store.get(),
                     GetFeaturesSetFromConfig(config),
                     GetSampleWeightsOrDie(config, data_store.get()),
                     GetTargetsOrDie(config, data_store.get()),
                     loss_func.get(),
                     config,
                     base_forest.get(),
                     &forest);
  CHECK(status.ok()) << "Failed to train a forest. Error message: " << status.ToString();

  // Write the model into a file.
  mkdir(FLAGS_output_dir.c_str(), 0744);
  string output_model_file = FLAGS_output_dir + "/" + FLAGS_output_model_name + ".json";
  string forest_text;
  status = JsonUtils::ToJson(forest, &forest_text);
  CHECK(status.ok()) << "Failed to output model to json.";
  WriteStringToFile(forest_text, output_model_file);
  LOG(INFO) << "Wrote the model to " << output_model_file;

  // Write the feature importance into a file.
  WriteStringToFile(FeatureImportanceFormatted(ComputeFeatureImportance(forest)),
                    FLAGS_output_dir + "/" + FLAGS_output_model_name + ".fimps");
}

void Test() {
  CHECK(!FLAGS_config_file.empty()) << "Please specify --config_file.";
  CHECK(!FLAGS_testing_model_file.empty()) << "Please specify --testing_model_file";
  CHECK(!FLAGS_output_dir.empty()) << "Please specify --output_dir.";

  StopWatch stopwatch;
  stopwatch.Start();
  LOG(INFO) << "Start testing.";

  // Load config.
  string config_text = ReadFileToStringOrDie(FLAGS_config_file);
  Config config;
  auto status = JsonUtils::FromJson(config_text, &config);
  CHECK(status.ok()) << "Failed to parse json to proto " << config_text;

  // Load testing_model_file.
  Forest forest = LoadForestOrDie(FLAGS_testing_model_file);

  // Load DataStore.
  auto data_store = LoadDataStoreOrDie(config);

  // Evaluate forest and write score out.
  mkdir(FLAGS_output_dir.c_str(), 0744);
  status = EvaluateForest(data_store.get(),
                          forest,
                          GetTestPoints(config, forest.tree_size()),
                          FLAGS_output_dir);
  CHECK(status.ok()) << "Failed to evaluate the forest: " << status.ToString();

  LOG(INFO) << "Wrote testing outputs to " << FLAGS_output_dir;
  stopwatch.End();
  LOG(INFO) << "Finished testing in "
            << StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs()) << ".";
}
