// Copyright 2010 Jiang Chen. All Rights Reserved.
// Author: Jiang Chen <criver@gmail.com>
//
// Unittest for gbdt_algo
// TODO(criver): figure out how to use ProtoEq and rewrite the following tests.

#include "gbdt_algo.h"

#include <google/protobuf/text_format.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "src/data_store/flatfiles_data_store.h"
#include "src/gbdt_algo/utils.h"
#include "src/loss_func/loss_func.h"
#include "src/loss_func/loss_func_factory.h"
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
        "  num_iterations: 4 "
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

    loss_func_ = LossFuncFactory::CreateLossFunc(config_.loss_func_config());
    CHECK(loss_func_);
    feature_names_ = GetFeaturesSetFromConfig(config_.data_config());
    w_ = GetSampleWeightsOrDie(config_.data_config(), data_store_.get());
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
  unique_ptr<LossFunc> loss_func_;
  unordered_set<string> feature_names_;
  FloatVector w_;
};

TEST_F(GBDTAlgoTest, TestBuildForest) {
  unique_ptr<Forest> forest;
  auto status = TrainGBDT(data_store_.get(),
                          feature_names_,
                          loss_func_.get(),
                          w_,
                          config_,
                          nullptr,
                          &forest);
  CHECK(status.ok()) << status.ToString();
  CHECK(forest) << "Failed to train forest.";
  for (auto& tree : *forest->mutable_tree()) {
    RemoveGains(&tree);
  }

  EXPECT_EQ(expected_forest_.tree_size(), forest->tree_size());
  for (int i = 0; i < forest->tree_size(); ++i) {
    EXPECT_EQ(expected_forest_.tree(i).DebugString(), forest->tree(i).DebugString());
  }
}

TEST_F(GBDTAlgoTest, TestBuildForestWithBaseForest) {
  // Setup a base forest from the first 3 trees of the expected_forest_.
  Forest base_forest;
  for (int i = 0; i < 3; ++i) {
    *base_forest.add_tree() = expected_forest_.tree(i);
  }

  // Train with base forest and train for additional 2 iterations.
  config_.mutable_tree_config()->set_num_iterations(2);

  unique_ptr<Forest> forest;
  auto status = TrainGBDT(data_store_.get(),
                          feature_names_,
                          loss_func_.get(),
                          w_,
                          config_,
                          &base_forest,
                          &forest);
  CHECK(status.ok()) << status.ToString();
  CHECK(forest) << "Failed to train forest.";

  for (auto& tree : *forest->mutable_tree()) {
    RemoveGains(&tree);
  }

  EXPECT_EQ(expected_forest_.tree_size(), forest->tree_size());
  // The first trees are from expected_forest_. They match exactly with those of
  // expected_forest_.
  for (int i = 0; i < expected_forest_.tree_size(); ++i) {
    EXPECT_EQ(expected_forest_.tree(i).DebugString(), forest->tree(i).DebugString());
  }
}

}  // namespace gbdt
