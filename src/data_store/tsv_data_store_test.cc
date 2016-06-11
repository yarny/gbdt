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

#include "tsv_data_store.h"

#include <memory>

#include "gtest/gtest.h"

#include "column.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class TSVDataStoreTest : public ::testing::Test {
 protected:
  const string kTestFileDir = "src/data_store/testdata/tsv_data_store_test";
  const vector<string> blocks = { "block-0-with-header.tsv", "block-1.tsv", "block-2.tsv" };
  void SetUp() {

    vector<string> block_paths;
    for (const auto& block : blocks) {
      block_paths.push_back(kTestFileDir + "/" + block);
    }
    TSVDataConfig config;
    config.add_binned_float_column("foo");
    config.add_binned_float_column("bar");
    config.add_string_column("weather");
    config.add_raw_float_column("target");
    data_store_.reset(new TSVDataStore(block_paths, config));
  }

  static vector<string> GetColStrings(const StringColumn& column){
    vector<string> col_strings(column.size());

    for (uint i = 0; i < column.size(); ++i) {
      col_strings[i] = column.get_row_string(i);
    }
    return col_strings;
  }

  static vector<float> GetBinMax(const BinnedFloatColumn& column) {
    vector<float> array(column.size());
    for (int i = 0; i < column.size(); ++i) {
      array[i] = column.get_row_max(i);
    }
    return array;
  }

  static vector<float> GetRawFloat(const RawFloatColumn& column) {
    vector<float> array(column.size());
    for (int i = 0; i < column.size(); ++i) {
      array[i] = column[i];
    }
    return array;
  }

  unique_ptr<TSVDataStore> data_store_;
};

TEST_F(TSVDataStoreTest, Test) {
  ASSERT_NE(data_store_->GetRawFloatColumn("target"), nullptr);
  ASSERT_NE(data_store_->GetStringColumn("weather"), nullptr);
  ASSERT_NE(data_store_->GetBinnedFloatColumn("foo"), nullptr);
  ASSERT_NE(data_store_->GetBinnedFloatColumn("bar"), nullptr);
  // "color" is in the tsv but not loaded by our config.
  EXPECT_EQ(data_store_->GetStringColumn("color"), nullptr);

  EXPECT_EQ(9, data_store_->num_rows());
  EXPECT_EQ(vector<string>({ "rainy", "clear", "cloudy", "clear", "snowy", "rainy", "cloudy", "shower", "rainy" }),
            GetColStrings(*data_store_->GetStringColumn("weather")));
  EXPECT_EQ(vector<float>({ 213, 312, 395, 45, 672, 123, 56, 79, 321 }),
            GetBinMax(*data_store_->GetBinnedFloatColumn("foo")));
  EXPECT_EQ(vector<float>({ 5.4, 4.3, 3.2, 2.3, 4.5, 6.5, 6.5, 7.8, 9.9 }),
            GetBinMax(*data_store_->GetBinnedFloatColumn("bar")));
  EXPECT_EQ(vector<float>({ 0, 1, 2, 2, 3, 1, 1, 3, 2 }),
            GetRawFloat(*data_store_->GetRawFloatColumn("target")));
}

}  // namespace gbdt
