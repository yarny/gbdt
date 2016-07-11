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
#include "src/data_store/column.h"
#include "src/data_store/mem_data_store.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class LossFuncLogLossTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto target = Column::CreateRawFloatColumn("target", vector<float>({0, 0, 0, 0, 1, 1, 1, 1}));
    data_store_.AddColumn(target->name(), std::move(target));
    LossFuncConfig config;
    config.set_loss_func("logloss");
    config.set_target_column("target");

    logloss_.reset(new LogLoss(config));
    num_rows_ = data_store_.num_rows();
  }

  MemDataStore data_store_;
  unique_ptr<LogLoss> logloss_;
  FloatVector w_ = [](int i) { return i < 4 ? 1.0 : 2.0; };
  uint num_rows_;
};

TEST_F(LossFuncLogLossTest, TestLogLoss) {
  CHECK(logloss_->Init(&data_store_, w_)) << "Failed to init loss func.";
  vector<double> f = { 0, 0, 0, 0, 0, 0, 0, 0 };
  vector<double> g, h;
  vector<GradientData> gradient_data_vec;
  double c;
  logloss_->ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);
  EXPECT_FLOAT_EQ(log(2.0), c);

  vector<double> expected_g =
      { -2.0/3.0, -2.0/3.0, -2.0/3.0, -2.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0};
  vector<double> expected_h = vector<double>(g.size(), 2/9.0);
  for (int i = 0; i < g.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_g[i], gradient_data_vec[i].g);
    EXPECT_FLOAT_EQ(expected_h[i], gradient_data_vec[i].h);
  }
}

}  // namespace gbdt
