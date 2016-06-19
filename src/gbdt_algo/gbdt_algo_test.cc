// Copyright 2010 Jiang Chen. All Rights Reserved.
// Author: Jiang Chen <criver@gmail.com>
//
// Unittest for gbdt_algo
// TODO(criver): figure out how to use ProtoEq and rewrite the following tests.

#include "gbdt_algo.h"

#include <google/protobuf/text_format.h>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "src/data_store/flatfiles_data_store.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/utils.h"

namespace gbdt {

namespace {
const float kFloatTolerance = 1e-6;
}  // namespace

class GBDTAlgoTest : public ::testing::Test {
 protected:
  void SetUp() {
    // The ground truth function is color='red' and width >=10 and height >= 5.
    data_store_.reset(new FlatfilesDataStore("src/gbdt_algo/testdata/gbdt_algo_test/flatfiles"));
    string config_text =
        " tree_config { "
        "  num_iterations: 2 "
        "  num_leaves: 3 "
        "  shrinkage: 0.1"
        "  split_config { "
        "  }"
        "}"
        "sampling_config { "
        "  example_sampling_rate: 1.0"
        "  feature_sampling_rate: 1.0"
        "}"
        "data_config { "
        "  float_feature: ["
        "   'width', "
        "   'height' "
        "  ]"
        " categorical_feature: ["
        "   'color' "
        " ]"
        "}"
        "loss_func_config {"
        "  loss_func: 'mse' "
        "  target_column: 'label'"
        "}";
    CHECK(google::protobuf::TextFormat::ParseFromString(config_text, &config_))
        << "Failed to parse proto " << config_text;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        ReadFileToStringOrDie("src/gbdt_algo/testdata/gbdt_algo_test/forest.model.txt"),
        &expected_forest_));
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
    if (fabs(t->score()) < kFloatTolerance) {
      t->set_score(0);
    }
  }

  Config config_;
  Forest expected_forest_;
  unique_ptr<DataStore> data_store_;
};

TEST_F(GBDTAlgoTest, TestBuildForest) {
  unique_ptr<Forest> forest = TrainGBDT(config_, data_store_.get());
  CHECK(forest) << "Failed to train forest.";
  for (auto& tree : *forest->mutable_tree()) {
    RemoveGains(&tree);
  }

  EXPECT_EQ(expected_forest_.tree_size(), forest->tree_size());
  for (int i = 0; i < forest->tree_size(); ++i) {
    EXPECT_EQ(expected_forest_.tree(i).DebugString(), forest->tree(i).DebugString());
  }
}

}  // namespace gbdt
