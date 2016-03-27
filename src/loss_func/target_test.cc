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
    auto height_column = Column::CreateRawFloatColumn(
        "height", vector<float>({1, 6, 3, 2, 5, 7, 0, 20}));
    auto color_column = Column::CreateStringColumn(
        "color",
        vector<string>({"red", "green", "yellow", "green", "red", "blue", "yellow", "red"}));
    data_store_.AddColumn(height_column->name(), std::move(height_column));
    data_store_.AddColumn(color_column->name(), std::move(color_column));
    num_rows_ = data_store_.GetColumn("height")->size();
  }

  MemDataStore data_store_;
  uint num_rows_ = 0;
};

TEST_F(TargetTest, ComputeFloatBinaryTargets) {
  BinaryTargetConfig config;
  config.set_target_column("height");
  config.set_threshold(5);
  vector<float> targets;
  EXPECT_TRUE(ComputeBinaryTargets(&data_store_, num_rows_, config, &targets));
  EXPECT_EQ(vector<float>({-1, 1, -1, -1, -1, 1, -1, 1}), targets);
}

TEST_F(TargetTest, ComputeStringBinaryTargets) {
  BinaryTargetConfig config;
  config.set_target_column("color");
  config.mutable_category()->add_category("red");
  config.mutable_category()->add_category("blue");
  vector<float> targets;

  EXPECT_TRUE(ComputeBinaryTargets(&data_store_, num_rows_, config, &targets));
  EXPECT_EQ(vector<float>({1, -1, -1, -1, 1, 1, -1, 1}), targets);
}

}  // namespace gbdt
