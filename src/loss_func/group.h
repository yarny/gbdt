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

#include <map>
#include <random>
#include <utility>
#include <vector>

#include "src/base/base.h"

namespace gbdt {

// Represents a group of instances in the training data set. In search ranking setting, a group are
// documents retrieved by the same query.
// Samples of pairs of instances with difference in targets are drawn from groups. In case when the
// group size is large, sampling is a feasible solution to avoid quadratic blow up. For some
// you can find linear algorithm to compute pairwise losses. But in this package, we adopt a general
// sampling strategy to support various types of pairwise losses.
class Group {
 public:
  Group(vector<uint>&& group, FloatVector y);

  // NOTE: Positive and negative are relative in this setting. Each pair has a positive and
  // a negative but each item can be positive in one pair but negative in another.

  // Randomly sample a pair from the group.
  pair<uint, uint> SamplePair(std::mt19937* generator) const;
  inline uint size() const {
    return group_.size();
  }
  inline uint operator [] (int i) const {
    return group_[i];
  }
  inline uint64 num_pairs() const {
    return num_pairs_;
  }
  inline uint rank(int i) const {
    return ranks_[i];
  }
  inline float y(int i) const {
    return y_(group_[i]);
  }

  void Rerank(const vector<double>& f);

private:
  vector<uint> group_;
  vector<uint> ranks_;
  uint64 num_pairs_ = 0;
  FloatVector y_;

  // The following data structure is used to map pair index to the actual
  // pair. Each entry represent a target block (instances with the
  // same target value), the key is the total accumulated up to this
  // target block. The value is (num_instances_in_block, start_of_negative) pair.
  map<uint64, pair<uint, uint64>> pair_map_;
};

}  // namespace gbdt
