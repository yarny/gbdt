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

namespace gbdt {

class MockPointwise : public Pointwise {
 public:
  MockPointwise(std::function<Pointwise::Data(double, double)> loss_func,
                const vector<float>& y)
      : Pointwise(loss_func), mock_y_(y) {}
 protected:
  bool ProvideY(DataStore* data_store, vector<float>* y) {
    *y = mock_y_;
    return true;
  }

  vector<float> mock_y_;
};

TEST(PointwiseTest, TestRMSE) {
  auto loss_func = [](double y, double f) {
    return Pointwise::Data((y - f) * (y -f),
                           y - f,
                           1.0);
  };
  vector<double> f = { 2, 2, 2, 2, 2, 2, 2, 2 };
  vector<float> w =  {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vector<float> y = {0, 0, 0, 0, 1, 1, 1, 1};
  vector<double> g, h;
  double c;

  MockPointwise mse(loss_func, y);
  mse.Init(nullptr, w);
  mse.ComputeFunctionalGradientsAndHessians(f, &c, &g, &h, nullptr);
  EXPECT_FLOAT_EQ(-1.5, c);

  EXPECT_EQ(vector<double>({ -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5 }),  g);
  EXPECT_EQ(vector<double>({ 1, 1, 1, 1, 1, 1, 1, 1 }),  h);
  EXPECT_FLOAT_EQ(0, accumulate(g.begin(), g.end(), 0.0));
}

TEST(PointwiseTest, TestLogLoss) {
  auto loss_func = [](double y, double f) {
    double e = exp(-y * f);
    return Pointwise::Data(log(1 + e),
                           y * e / (1 + e),
                           e / ((1 + e) * (1 + e)));
  };

  vector<float> w = {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};
  vector<double> f = { 0, 0, 0, 0, 0, 0, 0, 0 };
  vector<float> y = {-1, -1, -1, -1, 1, 1, 1, 1};
  vector<double> g, h;
  double c;
  MockPointwise logloss(loss_func, y);
  logloss.Init(nullptr, w);
  logloss.ComputeFunctionalGradientsAndHessians(f, &c, &g, &h, nullptr);

  EXPECT_FLOAT_EQ(log(2.0), c);

  vector<double> expected_g =
      { -2.0/3.0, -2.0/3.0, -2.0/3.0, -2.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0};
  vector<double> expected_h = vector<double>(g.size(), 2/9.0);
  for (int i = 0; i < g.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_g[i], g[i]);
    EXPECT_FLOAT_EQ(expected_h[i], h[i]);
  }
}

}  // namespace gbdt
