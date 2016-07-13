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

#include "loss_func_mse.h"

#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class LossFuncMSETest : public ::testing::Test {
 protected:
  void SetUp() {
    Config config;
    config.set_loss_func("mse");
    mse_.reset(new MSE(config));
    CHECK(mse_->Init(num_rows_, w_, y_, nullptr).ok());
  }
  
  unique_ptr<MSE> mse_;
  FloatVector w_ = [](int) { return 1.0; };
  FloatVector y_ = [](int i) { return i < 4 ? 0 : 1.0; };
  int num_rows_ = 8;
};

TEST_F(LossFuncMSETest, TestMSE1) {
  vector<double> f = { 2, 2, 2, 2, 2, 2, 2, 2 };
  vector<GradientData> gradient_data_vec;
  double c;
  mse_->ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);
  EXPECT_FLOAT_EQ(-1.5, c);
  vector<double> expected_g = { -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5 };
  vector<double> expected_h = { 1, 1, 1, 1, 1, 1, 1, 1 };
  for (int i = 0; i < gradient_data_vec.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_g[i], gradient_data_vec[i].g);
    EXPECT_FLOAT_EQ(expected_h[i], gradient_data_vec[i].h);
  }

  auto total = accumulate(gradient_data_vec.begin(), gradient_data_vec.end(), GradientData());
  EXPECT_FLOAT_EQ(0, total.g);
}

TEST_F(LossFuncMSETest, TestMSE2) {
  vector<double> f = { 2, 2, 2, 2, 0, 0, 0, 0 };
  vector<GradientData> gradient_data_vec;
  double c;
  mse_->ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);
  EXPECT_FLOAT_EQ(-0.5, c);
  vector<double> expected_g = { -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5 };
  vector<double> expected_h = { 1, 1, 1, 1, 1, 1, 1, 1 };
  for (int i = 0; i < gradient_data_vec.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_g[i], gradient_data_vec[i].g);
    EXPECT_FLOAT_EQ(expected_h[i], gradient_data_vec[i].h);
  }
  auto total = accumulate(gradient_data_vec.begin(), gradient_data_vec.end(), GradientData());
  EXPECT_FLOAT_EQ(0, total.g);
}

}  // namespace gbdt
