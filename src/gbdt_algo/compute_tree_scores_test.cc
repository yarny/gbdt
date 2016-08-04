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

#include <memory>
#include <google/protobuf/text_format.h>

#include "gtest/gtest.h"
#include "src/base/base.h"
#include "src/data_store/column.h"
#include "src/data_store/data_store.h"
#include "src/proto/tree.pb.h"

namespace gbdt {

class GetTreeScoreTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto color = Column::CreateStringColumn(
        "color",
        {"red", "blue", "green", "blue", "red", "red", "blue", "green", "red", "blue"});
    auto length = Column::CreateBucketizedFloatColumn(
        "length", vector<float>({2, 1, 1, 3, 2, 4, 10, 2, 7, 5}));
    auto width = Column::CreateBucketizedFloatColumn(
        "width", vector<float>({2, 3, 7, 3, 8, 4, 6, 2, 3, 5}));
    auto height = Column::CreateBucketizedFloatColumn(
        "height", vector<float>({12, 20, 6, 3, 11, 4, 10, 2, 20, 3}));

    data_store_.Add(std::move(color));
    data_store_.Add(std::move(length));
    data_store_.Add(std::move(width));
    data_store_.Add(std::move(height));

    string text = "score: 0.0 "
                  "split { feature: 'color' cat_split { category: ['red', 'green'] } }"
                  "left_child {"
                  "  score: 1 "
                  "  split { feature: 'length' float_split { threshold: 3.0 } }"
                  "  left_child { "
                  "    split { feature: 'width' float_split { threshold: 5.0 } } "
                  "    left_child { score: 0.0 }"
                  "    right_child { score: 1.0 }"
                  "  }"
                  "  right_child {"
                  "    split { feature: 'width' float_split {threshold: 4.0 } }"
                  "    left_child { score: 2.0 }"
                  "    right_child { score: 3.0 }"
                  "  }"
                  "}"
                  "right_child {"
                  "  split { feature: 'height' float_split { threshold: 10.0 } }"
                  "  left_child { score: 4.0 }"
                  "  right_child { score: 5.0 }"
                  "}";
    CHECK(google::protobuf::TextFormat::ParseFromString(text, &tree_));
    compute_tree_scores_.reset(new ComputeTreeScores(&data_store_));
    num_rows_ = data_store_.num_rows();
  }

  DataStore data_store_;
  uint num_rows_;
  TreeNode tree_;
  unique_ptr<ComputeTreeScores> compute_tree_scores_;
};

TEST_F(GetTreeScoreTest, GetTreeScore) {
  vector<double> scores(num_rows_, 0.0);
  vector<double> expected_scores = {0, 5, 1, 4, 1, 3, 5, 0, 2, 4};

  scores = vector<double>(num_rows_, 0);
  compute_tree_scores_->AddTreeScores(tree_, &scores);
  EXPECT_EQ(expected_scores, scores);

  // Set initial score to 1.
  scores = vector<double>(num_rows_, 1.0);
  for (auto& score : expected_scores) {
    score += 1.0;
  }
  compute_tree_scores_->AddTreeScores(tree_, &scores);
  EXPECT_EQ(expected_scores, scores);
}

}  // namespace gbdt
