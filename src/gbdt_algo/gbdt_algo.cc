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

#include "gbdt_algo.h"

#include <algorithm>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "external/cppformat/format.h"

#include "compute_tree_scores.h"
#include "split_algo.h"
#include "src/base/base.h"
#include "src/data_store/data_store.h"
#include "src/loss_func/loss_func.h"
#include "src/loss_func/loss_func_factory.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/stopwatch.h"
#include "src/utils/subsampling.h"
#include "src/utils/threadpool.h"
#include "src/utils/utils.h"
#include "src/utils/vector_slice.h"
#include "tree_algo.h"
#include "utils.h"

DECLARE_int32(num_threads);

namespace gbdt {

void ApplyShrinkage(TreeNode* tree, float shrinkage) {
  tree->set_score(tree->score() * shrinkage);
  if (tree->has_left_child()) {
    ApplyShrinkage(tree->mutable_left_child(), shrinkage);
  }
  if (tree->has_right_child()) {
    ApplyShrinkage(tree->mutable_right_child(), shrinkage);
  }
}

void ClearInternalFields(TreeNode* tree) {
  if (!tree->has_left_child()) {
    tree->clear_split();
  }

  if (tree->has_split()) {
    if (tree->split().has_cat_split()) {
      tree->mutable_split()->mutable_cat_split()->clear_internal_categorical_index();
    }
  }
  if (tree->has_left_child()) {
    ClearInternalFields(tree->mutable_left_child());
  }
  if (tree->has_right_child()) {
    ClearInternalFields(tree->mutable_right_child());
  }
}

void ClearInternalFields(Forest* forest) {
  for (auto& tree : *forest->mutable_tree()) {
    ClearInternalFields(&tree);
  }
}

void InitializeWithBaseForest(const Forest* base_forest,
                              const ComputeTreeScores& compute_tree_scores,
                              Forest* forest,
                              vector<double>* f) {
  // Initialize forest with base forest.
  auto* constant_tree = forest->mutable_tree(0);
  for (const auto& tree : base_forest->tree()) {
    // Update function scores.
    compute_tree_scores.AddTreeScores(tree, f);
    if (IsSingleNodeTree(tree)) {
      // Single node tree contains constant only.
      constant_tree->set_score(constant_tree->score() + tree.score());
    } else{
      *forest->add_tree() = tree;
    }
  }
  LOG(INFO) << "Finished initializing forest with " << base_forest->tree_size() << " trees.";
}

Status TrainGBDT(DataStore* data_store,
                 const unordered_set<string>& feature_names,
                 LossFunc* loss_func,
                 const vector<float>& w,
                 const Config& config,
                 const Forest* base_forest,
                 unique_ptr<Forest>* output_forest) {
  const auto& tree_config = config.tree_config();
  const auto& sampling_config = config.sampling_config();
  LOG(INFO) << "TreeConfig:\n" << tree_config.DebugString();
  LOG(INFO) << "SamplingConfig:\n" << sampling_config.DebugString();

  // Find features from data_store.
  vector<const Column*> features;
  auto status = LoadFeatures(feature_names, data_store, &features);
  if (!status.ok()) return status;

  if (!loss_func->Init(data_store, w)) {
    return Status(error::INTERNAL, "Failed to initialize loss function with the data_store.");
  }

  uint num_rows = data_store->num_rows();
  vector<double> f(num_rows, 0);  // current function values
  vector<GradientData> gradient_data(num_rows);
  ComputeTreeScores compute_tree_scores(data_store);

  unique_ptr<Forest> forest(new Forest);
  // The first tree is constant tree. Throughout the learning process, we will keep
  // updating the constant. The main reason for doing that is to exclude constant
  // from being scaled down by shrinkage.
  auto* constant_tree = forest->add_tree();
  if (base_forest) {
    InitializeWithBaseForest(base_forest, compute_tree_scores, forest.get(), &f);
  }

  StopWatch stopwatch;
  for (int i = 0; i < tree_config.num_iterations(); ++i) {
    string time_progress;
    if (i > 0) {
      stopwatch.End();
      time_progress = fmt::format(
          "iter={0},etf={1},",
          StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs()),
          StopWatch::MSecsToFormattedString(
              stopwatch.ElapsedTimeInMSecs() * (tree_config.num_iterations() - i)));
    }
    stopwatch.Start();
    // Compute gradients and constant
    double constant = 0;
    string loss_func_progress;
    loss_func->ComputeFunctionalGradientsAndHessians(f, &constant, &gradient_data, &loss_func_progress);

    // When constant is NaN of Inf, the learning diverges and should be stopped.
    if (std::isnan(constant) || std::isinf(constant)) {
      return Status(error::INTERNAL,
                    "Stopped learning early because it diverges. "
                    "Please try adding regularization to the config.");
    }

    // Log progress
    LOG(INFO) << fmt::format("{0}: {1}{2}", i, time_progress, loss_func_progress);

    // Add a tree to forest
    auto* tree = forest->add_tree();
    // Fit a tree to gradients and apply the shrinkage
    *tree = FitTreeToGradients(w, gradient_data, features, tree_config, sampling_config);

    // Apply Shrinkage to the tree
    ApplyShrinkage(tree, tree_config.shrinkage());
    // Update the constant
    constant_tree->set_score(constant_tree->score() + constant);
    // Update function score.
    compute_tree_scores.AddTreeScores(*tree, constant, &f);
  }

  ClearInternalFields(forest.get());
  *output_forest = std::move(forest);

  return Status::OK;
}

}  // namespace gbdt
