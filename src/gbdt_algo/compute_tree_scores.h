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

#ifndef COMPUTE_TREE_SCORES_H_
#define COMPUTE_TREE_SCORES_H_

#include <vector>

#include "src/base/base.h"
#include "src/utils/vector_slice.h"

namespace gbdt {

class Column;
class DataStore;
class TreeNode;

class ComputeTreeScores {
 public:
  ComputeTreeScores(DataStore* data_store);

  void AddTreeScores(const TreeNode& tree, vector<double>* scores);
  void AddTreeScores(const TreeNode& tree, double constant, vector<double>* scores);

 private:
  DataStore* data_store_ = nullptr;
  vector<uint> allsamples_;
  vector<VectorSlice<uint>> slices_;
};

}  // namespace gbdt

#endif  // COMPUTE_TREE_SCORES_H_
