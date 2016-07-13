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

#include "loss_func_logloss.h"

#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class LossFuncLogLossTest : public ::testing::Test {
 protected:
  void SetUp() {
    LossFuncConfig config;
    config.set_loss_func("logloss");
    logloss_.reset(new LogLoss(config));
    CHECK(logloss_->Init(num_rows_, w_, y_, nullptr).ok());
  }

  unique_ptr<LogLoss> logloss_;
  FloatVector w_ = [](int i) { return i < 4 ? 1.0 : 2.0; };
  FloatVector y_ = [](int i) { return i < 4 ? -1.0 : 1.0; };
  int num_rows_ = 8;
};

TEST_F(LossFuncLogLossTest, TestLogLoss) {
  vector<double> f = { 0, 0, 0, 0, 0, 0, 0, 0 };
  vector<GradientData> gradient_data_vec;
  double c;
  logloss_->ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);
  EXPECT_FLOAT_EQ(log(2.0), c);

  vector<double> expected_g =
      { -2.0/3.0, -2.0/3.0, -2.0/3.0, -2.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0};
  vector<double> expected_h = vector<double>(num_rows_, 2/9.0);
  for (int i = 0; i < num_rows_; ++i) {
    EXPECT_FLOAT_EQ(expected_g[i], gradient_data_vec[i].g);
    EXPECT_FLOAT_EQ(expected_h[i], gradient_data_vec[i].h);
  }
}

}  // namespace gbdt
