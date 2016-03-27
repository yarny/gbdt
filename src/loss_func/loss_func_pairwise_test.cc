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
  }

  MemDataStore data_store_;
  vector<float> sample_weights_ = {1.0, 1.0, 1.0, 1.0};
  const int kSamplingRate_ = 1000;
};

// All instances are in one group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansOneGroup) {
  vector<double> f = { 0, 0, 0, 0};
  vector<double> g, h;
  double c;
  LossFuncConfig config;
  // Set sampleing_rate to 10000 so that g and h are more stable.
  auto* pairwise_target_config = config.mutable_pairwise_target();
  pairwise_target_config->set_target_column("target");
  pairwise_target_config->set_group_column("group0");
  pairwise_target_config->set_pair_sampling_rate(kSamplingRate_);
  unique_ptr<Pairwise> pairwise(new PairwiseLogLoss(config));
  pairwise->Init(&data_store_, sample_weights_);
  pairwise->ComputeFunctionalGradientsAndHessians(f, &c, &g, &h, nullptr);
  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);

  // The gradients reflect the relative order of the original targets.
  vector<double> expected_g = { -1.5, -0.5, 0.5, 1.5 };
  vector<double> expected_h = { 1.5, 1.5, 1.5, 1.5};
  for (int i = 0; i < g.size(); ++i) {
    EXPECT_LT(fabs(expected_g[i] - g[i] / kSamplingRate_), 5e-2);
    EXPECT_LT(fabs(expected_h[i] - h[i] / kSamplingRate_), 5e-2);
  }
}

// All instances are in two group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansTwoGroups) {
  vector<double> f = { 0, 0, 0, 0};
  vector<double> g, h;
  double c;
  LossFuncConfig config;
  auto* pairwise_target_config = config.mutable_pairwise_target();
  pairwise_target_config->set_target_column("target");
  pairwise_target_config->set_group_column("group1");
  pairwise_target_config->set_pair_sampling_rate(kSamplingRate_);
  unique_ptr<Pairwise> pairwise(new PairwiseLogLoss(config));
  pairwise->Init(&data_store_, sample_weights_);
  pairwise->ComputeFunctionalGradientsAndHessians(f, &c, &g, &h, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);

  // Because of the grouping, the gradients of the two groups {0, 1} and {2, 3}
  // are similar in the magnitude since the targets are not compared
  // across groups.
  vector<double> expected_g = { -0.5, 0.5, -0.5, 0.5 };
  vector<double> expected_h = { 0.5, 0.5, 0.5, 0.5};
  for (int i = 0; i < g.size(); ++i) {
    EXPECT_LT(fabs(expected_g[i] - g[i] / kSamplingRate_), 5e-2);
    EXPECT_LT(fabs(expected_h[i] - h[i] / kSamplingRate_), 5e-2);
  }
}

}  // namespace gbdt
