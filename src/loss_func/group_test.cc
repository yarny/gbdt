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

#include "group.h"

#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "src/data_store/column.h"

namespace gbdt {

class GroupTest : public ::testing::Test {
 protected:
  void SetUp() {
  }
  vector<float> targets_ = {0, 1, 2, 0, 3, 0, 1, 1, 0, 2, 3, 1, 1, 0, 3};
  std::mt19937 generator_;
  const int kSampleCount_ = 10000;
};

TEST_F(GroupTest, TestGroup) {
  Group group({0, 2, 3, 4, 5, 7, 9, 10}, [this](int i) { return targets_[i]; });
  vector<float> targets(group.size());
  for (int i = 0; i < targets.size(); ++i) {
    targets[i] = targets_[group[i]];
  }
  EXPECT_EQ(8, group.size());
  // The group is sorted by targets.
  EXPECT_EQ(vector<float>({3, 3, 2, 2, 1, 0, 0, 0}), targets);
  // num_pairs = 2 * 6 + 2 * 4 + 3 = 23.
  EXPECT_EQ(23, group.num_pairs());
  map<pair<uint, uint>, double> counts;
  for (int i = 0; i < kSampleCount_; ++i) {
    auto p = group.SamplePair(&generator_);
    ASSERT_GT(group.size(), p.first);
    ASSERT_GT(group.size(), p.second);
    ASSERT_NE(p.first, p.second);
    ASSERT_GT(targets_[group[p.first]], targets_[group[p.second]]);
    ++counts[p];
  }

  // The probability of sampling each pair is 1/23.
  for (int i = 0; i < group.size(); ++i) {
    for (int j = i + 1; j < group.size(); ++j) {
      if (targets_[group[i]] != targets_[group[j]]) {
        EXPECT_LT(fabs(1.0 / 23 - counts[make_pair(i, j)] / (double) kSampleCount_), 1e-2);
      }
    }
  }
}

TEST_F(GroupTest, TestRerank) {
  Group group({0, 2, 3, 4, 5, 7, 9, 10}, [this](int i) { return i; });

  vector<double> f(targets_.size());
  // f is the reverse rank of targets.
  for (int i = 0; i < f.size(); ++i) {
    f[i] = -i;
  }

  group.Rerank(f);
  vector<uint> expected_ranks = {7, 6, 5, 4, 3, 2, 1, 0};
  vector<uint> ranks(group.size());
  for (int i = 0; i < group.size(); ++i) {
    ranks[i] = group.rank(i);
  }
  EXPECT_EQ(expected_ranks, ranks);
}

}  // namespace gbdt
