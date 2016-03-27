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

#include "utils.h"

#include <google/protobuf/text_format.h>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "src/proto/tree.pb.h"
#include "src/utils/utils.h"

namespace gbdt {

class UtilsTest : public ::testing::Test {
 protected:
  void SetUp() {
      CHECK(google::protobuf::TextFormat::ParseFromString(
      "tree {"
      "  score: 3.4 "
      "  split {"
      "    feature: 'A' "
      "    gain: 7.0 "
      "  }"
      "  left_child { "
      "    score: 1.5 "
      "    split {"
      "      feature: 'B' "
      "      gain: 3.0 "
      "    }"
      "    left_child {"
      "      score: 1.0 "
      "    }"
      "    right_child {"
      "      score: -1.0 "
      "    }"
      "  }"
      "  right_child { "
      "    score: 1.2 "
      "    split {"
      "      feature: 'C' "
      "      gain: 1.0 "
      "    }"
      "    left_child { "
      "      score: 2.0 "
      "    }"
      "    right_child { "
      "      score: 1.0 "
      "    }"
      "  }"
      "}"
      "tree {"
      "  score: 3.4 "
      "  split {"
      "    feature: 'B' "
      "    gain: 2.0 "
      "  }"
      "  left_child { "
      "    score: 1.5 "
      "    split {"
      "      feature: 'A' "
      "      gain: 3.0 "
      "    }"
      "    left_child {"
      "      score: 1.0 "
      "    }"
      "    right_child {"
      "      score: -1.0 "
      "    }"
      "  }"
      "  right_child { "
      "    score: 1.2 "
      "    split {"
      "      feature: 'D' "
      "      gain: 4.0 "
      "    }"
      "    left_child { "
      "      score: 2.0 "
      "    }"
      "    right_child { "
      "      score: 1.0 "
      "    }"
      "  }"
      "}",
      &forest_));
  }
  Forest forest_;
};

TEST_F(UtilsTest, ComputeFeatureImportance) {
  auto feature_importance = ComputeFeatureImportance(forest_);
  ASSERT_EQ(4, feature_importance.size());
  EXPECT_EQ("A", feature_importance[0].first);
  EXPECT_FLOAT_EQ(1.0, feature_importance[0].second);
  EXPECT_EQ("B", feature_importance[1].first);
  EXPECT_FLOAT_EQ(0.5, feature_importance[1].second);
  EXPECT_EQ("D", feature_importance[2].first);
  EXPECT_FLOAT_EQ(0.4, feature_importance[2].second);
  EXPECT_EQ("C", feature_importance[3].first);
  EXPECT_FLOAT_EQ(0.1, feature_importance[3].second);
}

TEST_F(UtilsTest, CollectAllFeatures) {
  EXPECT_EQ(unordered_set<string>({"A", "B", "C", "D"}), CollectAllFeatures(forest_));
}

}  // namespace gbdt
