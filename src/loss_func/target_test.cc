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

#include "target.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/data_store/column.h"
#include "src/data_store/mem_data_store.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class TargetTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto target_column = Column::CreateRawFloatColumn(
        "target", vector<float>({0, 1, 0, -0.5, -1, 0.8, 0, 1.3}));
    data_store_.AddColumn(target_column->name(), std::move(target_column));
  }

  MemDataStore data_store_;
};

TEST_F(TargetTest, ComputeFloatBinaryTargets) {
  LossFuncConfig config;
  config.set_target_column("target");
  vector<float> targets;
  EXPECT_TRUE(ComputeBinaryTargets(&data_store_, config, &targets));
  EXPECT_EQ(vector<float>({-1, 1, -1, -1, -1, 1, -1, 1}), targets);
}

}  // namespace gbdt
