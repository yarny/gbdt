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

#include "compute_tree_scores.h"

#include "split_algo.h"
#include "src/base/base.h"
#include "src/data_store/data_store.h"
#include "src/proto/tree.pb.h"
#include "src/utils/subsampling.h"
#include "src/utils/threadpool.h"
#include "src/utils/vector_slice.h"

DECLARE_int32(num_threads);

namespace gbdt {

void AddSampleTreeScores(DataStore* data_store,
                         const TreeNode* tree,
                         double constant,
                         VectorSlice<uint> samples,
                         vector<double>* scores) {
  if (tree->has_left_child()) {
    auto* column = data_store->GetColumn(tree->split().feature());

    CHECK(column) << "Failed to load feature " << tree->split().feature();
    auto slices = Partition(column, tree->split(), samples);
    AddSampleTreeScores(data_store, &tree->left_child(), constant, slices.first, scores);
    AddSampleTreeScores(data_store, &tree->right_child(), constant, slices.second, scores);
  } else {
    for (auto index : samples) {
      (*scores)[index] += tree->score() + constant;
    }
  }
}

ComputeTreeScores::ComputeTreeScores(DataStore* data_store)
    : data_store_(data_store) {
  allsamples_ = Subsampling::CreateAllSamples(data_store->num_rows());
  slices_ = Subsampling::DivideSamples(allsamples_, FLAGS_num_threads * 5);
}

void ComputeTreeScores::AddTreeScores(const TreeNode& tree, double constant,
                                      vector<double>* scores) {
  // Compute tree scores
  ThreadPool pool(FLAGS_num_threads);
  for (auto slice : slices_) {
    pool.Enqueue(std::bind(AddSampleTreeScores, data_store_, &tree, constant, slice, scores));
  }
}

void ComputeTreeScores::AddTreeScores(const TreeNode& tree, vector<double>* scores) {
  AddTreeScores(tree, 0, scores);
}

}  // namespace gbdt
