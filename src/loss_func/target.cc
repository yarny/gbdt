/* Copyright 2016 Jiang Chen <criver@gmail.com>
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

#include "target.h"

#include <unordered_set>
#include <vector>

#include "src/data_store/data_store.h"
#include "src/proto/config.pb.h"

namespace gbdt {

bool ComputeFloatBinaryTargets(DataStore* data_store, uint num_rows,
                               const BinaryTargetConfig& config, vector<float>* targets) {
  const string& target_column_name = config.target_column();
  if (target_column_name.empty()) {
    LOG(ERROR) << "Please specify binary target in your config.";
    return false;
  }
  auto target_column = data_store->GetRawFloatColumn(target_column_name);
  if (!target_column) {
    LOG(ERROR) << "Failed to get target column " << target_column_name;
    return false;
  }

  float threshold = config.threshold();
  targets->resize(target_column->size());
  for (uint i = 0; i < targets->size(); ++i) {
    (*targets)[i] = (*target_column)[i] > threshold ? 1 : -1;
  }
  return true;
}

bool ComputeStringBinaryTargets(DataStore* data_store, uint num_rows,
                                const BinaryTargetConfig& config, vector<float>* targets) {
  const string& target_column_name = config.target_column();
  if (target_column_name.empty()) {
    LOG(ERROR) << "Please specify target in your logloss config.";
    return false;
  }
  auto target_column = data_store->GetStringColumn(target_column_name);
  if (!target_column) {
    LOG(ERROR) << "Failed to get target column " << target_column_name;
    return false;
  }
  if (target_column->size() != num_rows) {
    LOG(ERROR) << "Column size consistency check failed for target " << target_column_name;
    return false;
  }

  unordered_set<uint> categories;
  for (const auto& cat: config.category().category()) {
    uint cat_index;
    if (target_column->get_cat_index(cat, &cat_index)) {
      categories.insert(cat_index);
    }
  }

  targets->resize(target_column->size());
  const auto& target_col = target_column->col();
  for (uint i = 0; i < targets->size(); ++i) {
    (*targets)[i] = categories.find(target_col[i]) != categories.end() ? 1 : -1;
  }
  return true;
}

bool ComputeBinaryTargets(DataStore* data_store, uint num_rows,
                          const BinaryTargetConfig& config, vector<float>* targets) {
  switch (config.positivity_test_case()) {
    case BinaryTargetConfig::kThreshold:
      return ComputeFloatBinaryTargets(data_store, num_rows, config, targets);
    case BinaryTargetConfig::kCategory:
      return ComputeStringBinaryTargets(data_store, num_rows, config, targets);
    default:
      LOG(ERROR) << "Failed to find setting for positivity test " << config.DebugString();
      return false;
  }
}

}  // namespace gbdt
