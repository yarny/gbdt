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

#include "loss_func_pairwise.h"

#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "loss_func_math.h"
#include "loss_func_pairwise_logloss.h"
#include "src/data_store/mem_data_store.h"

namespace gbdt {

class PairwiseTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto target = Column::CreateRawFloatColumn("target", vector<float>({0, 1, 2, 3}));
    auto group0 = Column::CreateStringColumn("group0", {"1", "1", "1", "1"});
    auto group1 = Column::CreateStringColumn("group1", {"0", "0", "1", "1"});
    data_store_.AddColumn(target->name(), std::move(target));
    data_store_.AddColumn(group0->name(), std::move(group0));
    data_store_.AddColumn(group1->name(), std::move(group1));

    config_.set_target_column("target");
    auto* pairwise_target_config = config_.mutable_pairwise_target();
    pairwise_target_config->set_group_column("group0");
    pairwise_target_config->set_pair_sampling_rate(kSamplingRate_);
  }

  void ExpectGradientEqual(const vector<GradientData>& expected,
                           const vector<GradientData>& gradient_data_vec) {
    for (int i = 0; i < expected.size(); ++i) {
      double avg_g = gradient_data_vec[i].g / kSamplingRate_;
      double avg_h = gradient_data_vec[i].h / kSamplingRate_;
      EXPECT_LT(fabs(expected[i].g - avg_g), 5e-2)
          << " at " << i << " actual " << avg_g << " vs " << expected[i].g;
      EXPECT_LT(fabs(expected[i].h - avg_h), 5e-2)
          << " at " << i << " actual " << avg_h << " vs " << expected[i].h;
    }
  }

  unique_ptr<Pairwise> CreateAndInitPairwiseLoss() {
    unique_ptr<Pairwise> pairwise(new PairwiseLogLoss(config_));
    pairwise->Init(&data_store_, sample_weights_);
    return std::move(pairwise);
  }

  MemDataStore data_store_;
  vector<float> sample_weights_ = {1.0, 1.0, 1.0, 1.0};
  vector<double> f_ = { 0, 0, 0, 0};
  // Set sampleing_rate to 10000 so that g and h are more stable.
  const int kSamplingRate_ = 10000;
  LossFuncConfig config_;
};

// All instances are in one group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansOneGroup) {
  vector<GradientData> gradient_data_vec;
  double c;
  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss();
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // The gradients reflect the relative order of the original targets.
  vector<GradientData> expected = { {-1.5, 1.5}, {-0.5, 1.5}, {0.5, 1.5}, {1.5, 1.5} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

// When on group_columnis specified, every instance is put in one group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansNoGroup) {
  // When no group specified, every instance is put in one group.
  vector<GradientData> gradient_data_vec;
  config_.mutable_pairwise_target()->clear_group_column();
  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss();
  double c;
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // The gradients reflect the relative order of the original targets.
  vector<GradientData> expected = { {-1.5, 1.5}, {-0.5, 1.5}, {0.5, 1.5}, {1.5, 1.5} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

// All instances are in two group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansTwoGroups) {
  vector<GradientData> gradient_data_vec;

  config_.mutable_pairwise_target()->set_group_column("group1");
  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss();
  double c;
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // Because of the grouping, the gradients of the two groups {0, 1} and {2, 3}
  // are similar in the magnitude since the targets are not compared
  // across groups.
  vector<GradientData> expected = { {-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansWeightByDeltaTarget) {
  vector<GradientData> gradient_data_vec;

  config_.mutable_pairwise_target()->set_weight_by_delta_target(true);
  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss();
  double c;
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // More weights are put on higher target separation.
  vector<GradientData> expected = { {-3, 3}, {-1, 2}, {1, 2}, {3, 3} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

}  // namespace gbdt
