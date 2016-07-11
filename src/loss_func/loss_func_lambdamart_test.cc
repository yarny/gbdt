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

#include "loss_func_lambdamart.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "src/data_store/mem_data_store.h"

namespace gbdt {

class PairwiseTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto target = Column::CreateRawFloatColumn("target", vector<float>({0, 1, 2, 3}));
    auto group0 = Column::CreateStringColumn("group0", {"1", "1", "1", "1"});
    data_store_.AddColumn(target->name(), std::move(target));
    data_store_.AddColumn(group0->name(), std::move(group0));

    config_.set_target_column("target");
    auto* pairwise_config = config_.mutable_pairwise_config();
    pairwise_config->set_group_column("group0");
    pairwise_config->set_pair_sampling_rate(kSamplingRate_);
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

  MemDataStore data_store_;
  const int kSamplingRate_ = 1000;
  LossFuncConfig config_;
  FloatVector w_ = [](int) { return 1.0; };
};

TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessians) {
  vector<double> f = { 1, 0, 3, 2};
  vector<GradientData> gradient_data_vec;
  double c;

  unique_ptr<Pairwise> lambdamart(new LambdaMART(config_));
  lambdamart->Init(&data_store_, w_);
  lambdamart->ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);
  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);

  // The gradients reflect the relative order of the original targets.
  vector<GradientData> expected = { {-0.28, 0.4}, {-0.022, 0.16}, {-0.113, 0.4}, {0.416, 0.38} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

}  // namespace gbdt
