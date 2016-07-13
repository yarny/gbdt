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

#include "tree_algo.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <queue>
#include <tuple>

#include "external/cppformat/format.h"

#include "split_algo.h"
#include "src/base/base.h"
#include "src/data_store/column.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/stopwatch.h"
#include "src/utils/subsampling.h"
#include "src/utils/threadpool.h"
#include "src/utils/utils.h"
#include "src/utils/vector_slice.h"

DECLARE_int32(num_threads);

using namespace std::placeholders;

namespace gbdt {

GradientData ComputeWeightedSum(FloatVector w,
                                const vector<GradientData>& gradient_data_vec,
                                const VectorSlice<uint>& samples) {
  // Divide samples into slices to parallelize the computation.
  auto slices = Subsampling::DivideSamples(samples, FLAGS_num_threads * 5);
  vector<GradientData> totals(slices.size());

  {
    ThreadPool pool(FLAGS_num_threads);
    for (int i = 0; i < slices.size(); ++i) {
      pool.Enqueue([&, &slice=slices[i], &total=totals[i]]() {
          for (auto index : slice) {
            total += w(index) * gradient_data_vec[index];
          }
        });
    }
  }

  return std::accumulate(totals.begin(), totals.end(), GradientData());
}

struct NodeData {
  NodeData(TreeNode* node_in, const Column* feature_in,
           VectorSlice<uint> subsamples_in)
      : node(node_in), feature(feature_in), subsamples(subsamples_in) {}
  TreeNode* node;
  const Column* feature;
  // Slices of samples that are routed to the node.
  VectorSlice<uint> subsamples;
};

// Finds the best split for the samples.
pair<Split, const Column*> FindBestFeatureAndSplit(const vector<const Column*>& features,
                                                   FloatVector w,
                                                   const vector<GradientData>& gradient_data_vec,
                                                   const VectorSlice<uint>& samples,
                                                   const GradientData& total,
                                                   const Config& config) {
  vector<uint> sample_features = Subsampling::UniformSubsample(
      features.size(), config.feature_sampling_rate());
  vector<Split> splits(sample_features.size());
  {

    ThreadPool pool(FLAGS_num_threads);

    for (uint i = 0; i < sample_features.size(); ++i) {
      pool.Enqueue([&,i] () {
          FindBestSplit(features[sample_features[i]], w, &gradient_data_vec, samples,
                        config, total, &splits[i]);
        });
    }
  }

  uint best_index = 0;
  for (uint i = 1; i < sample_features.size(); ++i) {
    if (splits[i].gain() > splits[best_index].gain()) {
      best_index = i;
    }
  }
  if (splits.size() > 0 && splits[best_index].gain() > 0.0) {
    const auto* feature = features[sample_features[best_index]];
    auto* split = &splits[best_index];
    split->set_feature(feature->name());
    if (split->has_cat_split()) {
      const auto* string_feature = static_cast<const StringColumn*>(feature);
      for (auto cat_index : split->cat_split().internal_categorical_index()) {
        split->mutable_cat_split()->add_category(string_feature->get_cat_string(cat_index));
      }
    }
    return make_pair(*split, feature);
  }
  return make_pair(Split(), nullptr);
}

TreeNode FitTreeToGradients(FloatVector w,
                            const vector<GradientData>& gradient_data_vec,
                            const vector<const Column*>& features,
                            const Config& config) {
  double lambda = config.l2_lambda();
  auto cmp = [] (const NodeData& x, const NodeData& y) {
      return x.node->split().gain() < y.node->split().gain();
  };
  priority_queue<NodeData, vector<NodeData>, decltype(cmp)> node_queue(cmp);
  TreeNode tree;

  // Subsampling.
  auto subsamples = Subsampling::UniformSubsample(
      gradient_data_vec.size(), config.example_sampling_rate());
  GradientData total = ComputeWeightedSum(w, gradient_data_vec, subsamples);

  tree.set_score(total.Score(lambda));
  auto root_split = FindBestFeatureAndSplit(
      features, w, gradient_data_vec, subsamples, total, config);
  if (root_split.first.gain() > 0) {
    *(tree.mutable_split()) = std::move(root_split.first);
  }
  node_queue.push(NodeData({&tree, root_split.second, VectorSlice<uint>(subsamples)}));

  // The size of queue is equal to the number of leaves
  while (!node_queue.empty() && node_queue.size() < config.num_leaves() &&
         node_queue.top().feature) {
    auto node_data = node_queue.top();
    auto* node = node_data.node;
    const auto* feature = node_data.feature;
    auto subsamples_slice = node_data.subsamples;

    // Partition.
    auto sub_slices = Partition(feature, node->split(), subsamples_slice);

    GradientData left_total = ComputeWeightedSum(w, gradient_data_vec, sub_slices.first);
    GradientData right_total = ComputeWeightedSum(w, gradient_data_vec, sub_slices.second);

    // Left.
    auto* left_child = node->mutable_left_child();
    left_child->set_score(left_total.Score(lambda));
    auto left_split = FindBestFeatureAndSplit(
        features, w, gradient_data_vec, sub_slices.first, left_total, config);
    if (left_split.first.gain() > 0) {
      *left_child->mutable_split() = std::move(left_split.first);
    }

    // Right.
    auto* right_child = node->mutable_right_child();
    right_child->set_score(right_total.Score(lambda));
    auto right_split = FindBestFeatureAndSplit(
        features, w, gradient_data_vec, sub_slices.second, right_total, config);
    if (right_split.first.gain() > 0) {
      *right_child->mutable_split() = std::move(right_split.first);
    }

    node_queue.pop();
    node_queue.push(NodeData(left_child, left_split.second, sub_slices.first));
    node_queue.push(NodeData(right_child, right_split.second, sub_slices.second));
  }

  return tree;
}

}  // namespace gbdt
