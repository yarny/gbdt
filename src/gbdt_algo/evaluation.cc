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

#include "evaluation.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <list>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

#include "external/cppformat/format.h"

#include "compute_tree_scores.h"
#include "split_algo.h"
#include "src/base/base.h"
#include "src/data_store/data_store.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/utils.h"
#include "utils.h"

namespace gbdt {

bool WriteScoreFile(const string& filename,
                    const vector<double>& scores) {
  ofstream out_file;
  out_file.open(filename.c_str());
  if (!out_file.is_open())
    return false;

  for (uint i = 0; i < scores.size(); ++i) {
    if (i != 0)
      out_file << "\n";
    out_file << scores[i];
  }

  out_file.close();
  return true;
}

Status EvaluateForest(DataStore* data_store,
                      const Forest& forest,
                      const list<int>& test_points_arg,
                      const string& output_dir) {
  auto test_points = test_points_arg;
  auto feature_names = CollectAllFeatures(forest);
  auto status = LoadFeatures(feature_names, data_store, nullptr);
  if (!status.ok()) return status;
  mkdir(output_dir.c_str(), 0744);

  ComputeTreeScores compute_tree_scores(data_store);

  vector<double> scores(data_store->num_rows(), 0.0);
  for (int i = 0; i < forest.tree_size() && !test_points.empty(); ++i) {
    compute_tree_scores.AddTreeScores(forest.tree(i), &scores);

    if (i+1 == test_points.front()) {
      string score_file =fmt::format("{0}/forest.{1}.score", output_dir, test_points.front());
      if (!WriteScoreFile(score_file, scores)) {
        return Status(error::ABORTED, "Failed to write into the score files.");
      }
      LOG(INFO) << fmt::format("Wrote {0}.", score_file);
    }

    while (!test_points.empty() && i+1 >= test_points.front()) {
      test_points.pop_front();
    }
  }

  return Status::OK;
}

Status EvaluateForest(DataStore* data_store,
                      const Forest& forest,
                      vector<double>* scores) {
  auto feature_names = CollectAllFeatures(forest);
  auto status = LoadFeatures(feature_names, data_store, nullptr);
  if (!status.ok()) return status;

  ComputeTreeScores compute_tree_scores(data_store);
  scores->clear();
  scores->resize(data_store->num_rows(), 0.0);
  for (const auto& tree : forest.tree()) {
    compute_tree_scores.AddTreeScores(tree, scores);
  }
  return Status::OK;
}

}  // namespace gbdt
