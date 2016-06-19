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

#include "loss_func_pointwise.h"

#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "loss_func_math.h"

namespace gbdt {

class MockPointwise : public Pointwise {
 public:
  MockPointwise(std::function<LossFuncData(double, double)> loss_func,
                const vector<float>& y)
      : Pointwise(loss_func), mock_y_(y) {}
 protected:
  bool ProvideY(DataStore* data_store, vector<float>* y) {
    *y = mock_y_;
    return true;
  }

  vector<float> mock_y_;
};

TEST(PointwiseTest, TestMSE) {
  vector<double> f = { 2, 2, 2, 2, 2, 2, 2, 2 };
  vector<float> w =  {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vector<float> y = {0, 0, 0, 0, 1, 1, 1, 1};
  vector<GradientData> gradient_data_vec;
  double c;

  MockPointwise mse(ComputeMSE, y);
  mse.Init(nullptr, w);
  mse.ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);
  EXPECT_FLOAT_EQ(-1.5, c);

  vector<GradientData> expected =
      {{-0.5, 1}, {-0.5, 1}, {-0.5, 1}, {-0.5, 1}, {0.5, 1}, {0.5, 1}, {0.5, 1}, {0.5, 1}};

  for (int i = 0; i < gradient_data_vec.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i].g, gradient_data_vec[i].g);
    EXPECT_FLOAT_EQ(expected[i].h, gradient_data_vec[i].h);
  }
  auto total = accumulate(gradient_data_vec.begin(), gradient_data_vec.end(), GradientData());
  EXPECT_FLOAT_EQ(0, total.g);
}

TEST(PointwiseTest, TestLogLoss) {
  vector<float> w = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};
  vector<double> f = { 0, 0, 0, 0, 0, 0, 0, 0 };
  vector<float> y = {-1, -1, -1, -1, 1, 1, 1, 1};
  vector<GradientData> gradient_data_vec;
  double c;
  MockPointwise logloss(ComputeLogLoss, y);
  logloss.Init(nullptr, w);
  logloss.ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);

  EXPECT_FLOAT_EQ(log(2.0), c);

  vector<GradientData> expected =
      { {-2.0/3.0, 2/9.0}, {-2.0/3.0, 2/9.0}, {-2.0/3.0, 2/9.0}, {-2.0/3.0, 2/9.0},
        {1.0/3.0, 2/9.0}, {1.0/3.0, 2/9.0}, {1.0/3.0, 2/9.0}, {1.0/3.0, 2/9.0}};

  for (int i = 0; i < gradient_data_vec.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i].g, gradient_data_vec[i].g);
    EXPECT_FLOAT_EQ(expected[i].h, gradient_data_vec[i].h);
  }
}

}  // namespace gbdt
