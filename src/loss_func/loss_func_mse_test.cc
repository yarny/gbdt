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
#include "src/data_store/column.h"
#include "src/data_store/mem_data_store.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class LossFuncMSETest : public ::testing::Test {
 protected:
  void SetUp() {
    auto target = Column::CreateRawFloatColumn("target", vector<float>({0, 0, 0, 0, 1, 1, 1, 1}));
    data_store_.AddColumn(target->name(), std::move(target));
    LossFuncConfig config;
    config.set_loss_func("mse");
    config.mutable_regression_target()->set_target_column("target");
    mse_.reset(new MSE(config));
    num_rows_ = data_store_.num_rows();
    w_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  }

  MemDataStore data_store_;
  unique_ptr<MSE> mse_;
  vector<float> w_;
  uint num_rows_;
};

TEST_F(LossFuncMSETest, TestMSE) {
  mse_->Init(&data_store_, w_);
  vector<double> f = { 2, 2, 2, 2, 2, 2, 2, 2 };
  vector<double> g;
  vector<double> h;
  double c;
  mse_->ComputeFunctionalGradientsAndHessians(f, &c, &g, &h, nullptr);
  EXPECT_FLOAT_EQ(-1.5, c);
  EXPECT_EQ(vector<double>({ -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5 }),  g);
  EXPECT_EQ(vector<double>({ 1, 1, 1, 1, 1, 1, 1, 1 }),  h);
  EXPECT_FLOAT_EQ(0, accumulate(g.begin(), g.end(), 0.0));

  f = { 2, 2, 2, 2, 0, 0, 0, 0 };
  mse_->ComputeFunctionalGradientsAndHessians(f, &c, &g, &h, nullptr);
  EXPECT_FLOAT_EQ(-0.5, c);
  EXPECT_EQ(vector<double>({ -1.5, -1.5, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5 }), g);
  EXPECT_EQ(vector<double>({ 1, 1, 1, 1, 1, 1, 1, 1 }),  h);
  EXPECT_FLOAT_EQ(0, accumulate(g.begin(), g.end(), 0.0));
}

}  // namespace gbdt
