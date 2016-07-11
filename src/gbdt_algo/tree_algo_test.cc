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

// TODO(criver): figure out how to use ProtoEq and rewrite the following tests.

#include "tree_algo.h"

#include <vector>

#include "gtest/gtest.h"
#include "src/data_store/column.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/loss_func/gradient_data.h"
#include "src/utils/subsampling.h"

namespace gbdt {

class TreeBuildingTest : public testing::Test {
 public:
  void SetUp() {
    allsamples_ = Subsampling::CreateAllSamples(gradient_data_vec_.size());

    tree_config_.set_num_leaves(10);
    sampling_config_.set_example_sampling_rate(1.0);
    sampling_config_.set_feature_sampling_rate(1.0);
    const_float_feature_ = Column::CreateBinnedFloatColumn("feature0", vector<float>(16, 1.3));
    const_string_feature_ = Column::CreateStringColumn("feature1", vector<string>(16, "1.3"));
    vector<float> irrelevant_feature = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    irrelevant_feature_ = Column::CreateBinnedFloatColumn("feature2", irrelevant_feature);

    vector<float> parity_feature = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
    parity_feature_ = Column::CreateBinnedFloatColumn("feature3", parity_feature);
    vector<string> zero_feature = { "zero", "zero", "zero", "zero",
                                    "nonzero", "nonzero", "nonzero", "nonzero",
                                    "nonzero", "nonzero", "nonzero", "nonzero",
                                    "nonzero", "nonzero", "nonzero", "nonzero" };
    zero_feature_ = Column::CreateStringColumn("feature4", zero_feature);
    vector<float> three_feature0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0};
    vector<float> three_feature1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, NAN};
    three_feature0_ = Column::CreateBinnedFloatColumn("feature5", three_feature0);
    three_feature1_ = Column::CreateBinnedFloatColumn("feature6", three_feature1);
  }

  void RemoveGains(TreeNode* t) {
    if (t->has_split()) {
      t->mutable_split()->clear_gain();
    }
    if (t->has_left_child()) {
      RemoveGains(t->mutable_left_child());
    }
    if (t->has_right_child()) {
      RemoveGains(t->mutable_right_child());
    }
  }

  unique_ptr<Column> const_float_feature_;
  unique_ptr<Column> const_string_feature_;
  unique_ptr<Column> irrelevant_feature_;
  // The feature is 0 when gradient is even and 1 if gradient is odd.
  unique_ptr<Column> parity_feature_;
  // Feature that indicates zeros.
  unique_ptr<Column> zero_feature_;
  // Feature that indicates threes.
  unique_ptr<Column> three_feature0_;
  unique_ptr<Column> three_feature1_;
  TreeConfig tree_config_;
  SamplingConfig sampling_config_;
  vector<GradientData> gradient_data_vec_ =
  {
    {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0},
    {2.0, 1.0}, {2.0, 1.0}, {2.0, 1.0}, {2.0, 1.0},
    {3.0, 1.0}, {3.0, 1.0}, {3.0, 1.0}, {3.0, 1.0},
    {4.0, 1.0}, {4.0, 1.0}, {4.0, 1.0}, {4.0, 1.0}};
  FloatVector w_ = [] (int) { return 1.0; };

  vector<uint> allsamples_;
};

TEST_F(TreeBuildingTest, BuildTree) {
  // In this example, parity_feature_, zero_feature_ and three_feature_ perfectly
  // classify the target g_.
  vector<const Column*> features = { const_float_feature_.get(),
                                     const_string_feature_.get(),
                                     irrelevant_feature_.get(),
                                     parity_feature_.get(),
                                     zero_feature_.get(),
                                     three_feature0_.get(),
                                     three_feature1_.get() };
  TreeNode t = FitTreeToGradients(w_, gradient_data_vec_, features, tree_config_, sampling_config_);
  RemoveGains(&t);
  string expected_tree =
      "score: 2.5\n"
      "split {\n"
      "  feature: \"feature4\"\n"
      "  cat_split {\n"
      "    category: \"zero\"\n"
      "    internal_categorical_index: 1\n"
      "  }\n"
      "}\n"
      "left_child {\n"
      "  score: 1\n"
      "}\n"
      "right_child {\n"
      "  score: 3\n"
      "  split {\n"
      "    feature: \"feature5\"\n"
      "    float_split {\n"
      "      threshold: 1.5\n"
      "    }\n"
      "  }\n"
      "  left_child {\n"
      "    score: 2.8\n"
      "    split {\n"
      "      feature: \"feature6\"\n"
      "      float_split {\n"
      "        threshold: 1.5\n"
      "        missing_to_right_child: true\n"
      "      }\n"
      "    }\n"
      "    left_child {\n"
      "      score: 2.5\n"
      "      split {\n"
      "        feature: \"feature3\"\n"
      "        float_split {\n"
      "          threshold: 0.5\n"
      "        }\n"
      "      }\n"
      "      left_child {\n"
      "        score: 3\n"
      "      }\n"
      "      right_child {\n"
      "        score: 2\n"
      "      }\n"
      "    }\n"
      "    right_child {\n"
      "      score: 4\n"
      "    }\n"
      "  }\n"
      "  right_child {\n"
      "    score: 4\n"
      "  }\n"
      "}\n";
  EXPECT_EQ(expected_tree, t.DebugString());
}

TEST_F(TreeBuildingTest, BuildTreeWithIrrlevantFeatures) {
  vector<const Column*> features = { const_float_feature_.get(),
                                     const_string_feature_.get(),
                                     irrelevant_feature_.get() };

  TreeNode t = FitTreeToGradients(w_, gradient_data_vec_, features, tree_config_, sampling_config_);
  EXPECT_FALSE(t.has_left_child());
  EXPECT_FALSE(t.has_right_child());
  EXPECT_FALSE(t.has_split());
  EXPECT_EQ(2.5, t.score());
}

}  // namespace gbdt
