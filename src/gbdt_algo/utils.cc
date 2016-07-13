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

#include "utils.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "external/cppformat/format.h"

#include "src/base/base.h"
#include "src/data_store/column.h"
#include "src/data_store/data_store.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/json_utils.h"
#include "src/utils/stopwatch.h"
#include "src/utils/threadpool.h"
#include "src/utils/utils.h"

DECLARE_int32(num_threads);

namespace gbdt {

Status LoadFeatures(const unordered_set<string>& feature_names,
                    DataStore* data_store,
                    vector<const Column*>* columns) {
  vector<const Column*> features;
  LOG(INFO) << "Loading features...";
  StopWatch stopwatch;
  stopwatch.Start();

  {
    ThreadPool pool(FLAGS_num_threads);
    for (const auto& feature_name : feature_names) {
      pool.Enqueue([&]() { data_store->GetColumn(feature_name); });
    }
  }

  features.reserve(feature_names.size());
  for (const auto& feature_name : feature_names) {
    const auto* feature = data_store->GetColumn(feature_name);
    if (feature == nullptr) {
      return Status(error::NOT_FOUND, "Failed to load feature " + feature_name);
    }
    if (!feature->status().ok()) {
      return feature->status();
    }
    features.emplace_back(feature);
  }

  stopwatch.End();
  LOG(INFO) << fmt::format(
      "Loaded {0} features, each with {1} rows, in {2}.",
      features.size(),
      data_store->num_rows(),
      StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs()));
  if (columns) {
    (*columns) = std::move(features);
  }
  return Status::OK;
}

void ComputeFeatureImportance(const TreeNode& node,
                              unordered_map<string, double>* feature_importance_map) {
  if (node.has_left_child()) {
    (*feature_importance_map)[node.split().feature()] += node.split().gain();
    ComputeFeatureImportance(node.left_child(), feature_importance_map);
    ComputeFeatureImportance(node.right_child(), feature_importance_map);
  }
}

vector<pair<string, double>> ComputeFeatureImportance(const Forest& forest) {
  unordered_map<string, double> feature_importance_map;
  for (const auto& tree : forest.tree()) {
    ComputeFeatureImportance(tree, &feature_importance_map);
  }
  vector<pair<string, double>> feature_importance(feature_importance_map.begin(),
                                                  feature_importance_map.end());;

  sort(feature_importance.begin(), feature_importance.end(),
       [](const pair<string, double> x, const pair<string, double> y) {
         return x.second > y.second;
       });
  // Normalize feature importance map.
  if (feature_importance.size() > 0) {
    double max_importance = feature_importance[0].second;
    for (auto& p : feature_importance) {
      p.second /= max_importance;
    }
  }
  return feature_importance;
}

void CollectAllFeatures(const TreeNode& tree, unordered_set<string>* feature_names) {
  if (tree.has_split()) {
    feature_names->insert(tree.split().feature());
  }
  if (tree.has_left_child()) {
    CollectAllFeatures(tree.left_child(), feature_names);
  }
  if (tree.has_right_child()) {
    CollectAllFeatures(tree.right_child(), feature_names);
  }
}

unordered_set<string> CollectAllFeatures(const Forest& forest) {
  unordered_set<string> feature_names;
  for (const auto& tree: forest.tree()) {
    CollectAllFeatures(tree, &feature_names);
  }
  return feature_names;
}

bool IsSingleNodeTree(const TreeNode& tree) {
  return !tree.has_left_child();
}

list<int> GetTestPoints(const EvalConfig& config, int forest_size) {
  // Load test points
  // By default, we output the test scores of the final forest,
  // but if eval_interval is specified, we will test the forest at the intervals.
  list<int> test_points;
  if (config.eval_interval() > 0) {
    for (int test_point = forest_size; test_point > 0;
         test_point -= config.eval_interval()) {
      test_points.push_back(test_point);
    }
    std::reverse(test_points.begin(), test_points.end());
  } else {
    test_points.push_back(forest_size);
  }
  return test_points;
}

unordered_set<string> GetFeaturesSetFromConfig(const DataConfig& config) {
  unordered_set<string> feature_names(config.float_feature().begin(), config.float_feature().end());
  feature_names.insert(config.categorical_feature().begin(), config.categorical_feature().end());
  return feature_names;
}

FloatVector GetSampleWeightsOrDie(const DataConfig& config, DataStore* data_store) {
  const string& weight_column_name = config.weight_column();
  if (!weight_column_name.empty()) {
    const auto* sample_weights = data_store->GetRawFloatColumn(weight_column_name);
    CHECK(sample_weights) << "Failed to load sample weights";
    return [&](int i) {
      return (*sample_weights)[i];
    };
  }

  return [](int) { return 1.0; };
}

FloatVector GetTargetsOrDie(const DataConfig& config, DataStore* data_store) {
  const string& target_column_name = config.target_column();
  CHECK(!target_column_name.empty()) << "Please specify target_column.";

  auto targets = data_store->GetRawFloatColumn(target_column_name);
  CHECK(targets) << "Failed to get target column " << target_column_name;
  const auto& raw_floats = targets->raw_floats();

  if (config.binarize_target()) {
    return [&raw_floats=raw_floats](int i) { return raw_floats[i] > 0 ? 1 : -1; };
  } else {
    return [&raw_floats=raw_floats](int i) { return raw_floats[i]; };
  }
}

Forest LoadForestOrDie(const string& forest_file) {
  Forest forest;
  string forest_text = ReadFileToStringOrDie(forest_file);
  auto status = JsonUtils::FromJson(forest_text, &forest);
  CHECK(status.ok()) << "Failed to parse json " << forest_text;
  LOG(INFO) << "Loaded a forest with " << forest.tree_size() << " trees.";
  return forest;
}

}  // namespace gbdt
