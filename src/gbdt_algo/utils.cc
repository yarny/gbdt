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
#include "src/proto/tree.pb.h"
#include "src/utils/stopwatch.h"
#include "src/utils/threadpool.h"

DECLARE_int32(num_threads);

namespace gbdt {

vector<const Column*> LoadFeaturesOrDie(unordered_set<string>& feature_names,
                                        DataStore* data_store) {
  vector<const Column*> features;
  LOG(INFO) << "Loading features...";
  StopWatch stopwatch;
  stopwatch.Start();

  {
    auto task = [](DataStore* data_store, const string& feature_name) {
      data_store->GetColumn(feature_name);
    };
    // Load features in multi-threaded way.
    ThreadPool pool(FLAGS_num_threads);
    for (const auto& feature_name : feature_names) {
      pool.Enqueue(std::bind(task, data_store, feature_name));
    }
  }

  features.reserve(feature_names.size());
  for (const auto& feature_name : feature_names) {
    const auto* feature = data_store->GetColumn(feature_name);
    CHECK(feature != nullptr) << "Failed to load feature " << feature_name;
    features.emplace_back(feature);
  }

  stopwatch.End();
  LOG(INFO) << fmt::format(
      "Loaded {0} features, each with {1} rows, in {2}.",
      features.size(),
      data_store->num_rows(),
      StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs()));
  return features;
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

}  // namespace gbdt
