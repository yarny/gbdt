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

#include "json_utils.h"

#include "src/proto/tree.pb.h"
#include "gtest/gtest.h"

namespace gbdt {

class JsonUtilsTest : public ::testing::Test {
 protected:
  void SetUp() {
    tree_.set_score(0.2);

    auto* float_split = tree_.mutable_split()->mutable_float_split();
    float_split->set_threshold(0);
    tree_.mutable_split()->set_feature("A");
    auto* left_child = tree_.mutable_left_child();
    left_child->set_score(0);
    auto* right_child = tree_.mutable_right_child();
    right_child->mutable_split()->set_feature("B");
    right_child->set_score(0.3);
    auto* cat_split = right_child->mutable_split()->mutable_cat_split();
    cat_split->add_category("apple");
    cat_split->add_category("banana");
    right_child->mutable_left_child()->set_score(0.4);
    right_child->mutable_right_child()->set_score(0);

    tree_json_ = "{\"score\":0.2,\"split\":{\"feature\":\"A\",\"floatSplit\":"
                 "{\"threshold\":0,\"missingToRightChild\":false},\"gain\":0},\"leftChild\":"
                 "{\"score\":0},\"rightChild\":{\"score\":0.3,\"split\":"
                 "{\"feature\":\"B\",\"catSplit\":{\"category\":"
                 "[\"apple\",\"banana\"],\"internalCategoricalIndex\":[]},\"gain\":0},"
                 "\"leftChild\":{\"score\":0.4},\"rightChild\":{\"score\":0}}}";
  }

  TreeNode tree_;
  string tree_json_;
};

TEST_F(JsonUtilsTest, TestToJson) {
  EXPECT_EQ(tree_json_, JsonUtils::ToJsonOrDie(tree_));
}

TEST_F(JsonUtilsTest, FromJson) {
  TreeNode tree;
  EXPECT_TRUE(JsonUtils::FromJson(tree_json_, &tree));
  EXPECT_EQ(tree_.DebugString(), tree.DebugString());
}

}  // namespace gbdt
