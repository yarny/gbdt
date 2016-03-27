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

#include "flatfiles_data_store.h"

#include <memory>

#include "gtest/gtest.h"

#include "column.h"

namespace gbdt {

class FlatfilesDataStoreTest : public ::testing::Test {
 protected:
  void SetUp() {
    data_store_.reset(new FlatfilesDataStore("src/data_store/testdata/flatfiles_data_store_test"));
  }
  unique_ptr<FlatfilesDataStore> data_store_;
};

TEST_F(FlatfilesDataStoreTest, ReadBinnedFloats) {
  auto column = data_store_->GetBinnedFloatColumn("foo");
  CHECK(column != nullptr);
  EXPECT_EQ(9, column->size());
  set<uint> missing_positions = {2, 6, 7};
  for (uint i = 0; i < column->size(); ++i) {
    if (missing_positions.find(i) == missing_positions.end()) {
      EXPECT_FALSE(column->col().missing(i)) << " at " << i;
    } else {
      EXPECT_TRUE(column->col().missing(i)) << " at " << i;
    }
  }
  EXPECT_FLOAT_EQ(0.1, column->get_row_max(0));
  EXPECT_FLOAT_EQ(1.2, column->get_row_max(1));
  EXPECT_FLOAT_EQ(2.3, column->get_row_max(3));
  EXPECT_FLOAT_EQ(3.4, column->get_row_max(4));
  EXPECT_FLOAT_EQ(4.5, column->get_row_max(5));
  EXPECT_FLOAT_EQ(5.6, column->get_row_max(8));

  // num_rows will be 9 after the first feature is loaded.
  EXPECT_EQ(9, data_store_->num_rows());
}

TEST_F(FlatfilesDataStoreTest, ReadRawFloats) {
  const auto* column = data_store_->GetRawFloatColumn("foo2");
  CHECK(column != nullptr);
  EXPECT_FLOAT_EQ(0.1, (*column)[0]);
  EXPECT_FLOAT_EQ(1.2, (*column)[1]);
  EXPECT_TRUE(isnan((*column)[2]));
  EXPECT_FLOAT_EQ(2.3, (*column)[3]);
  EXPECT_FLOAT_EQ(3.4, (*column)[4]);
  EXPECT_FLOAT_EQ(4.5, (*column)[5]);
  EXPECT_TRUE(isnan((*column)[6]));
  EXPECT_TRUE(isnan((*column)[7]));
  EXPECT_FLOAT_EQ(5.6, (*column)[8]);
}

TEST_F(FlatfilesDataStoreTest, ReadStrings) {
  auto column = data_store_->GetStringColumn("bar");
  CHECK(column != nullptr);
  EXPECT_EQ(9, column->size());
  set<uint> missing_positions = {2, 6, 7};
  for (uint i = 0; i < column->size(); ++i) {
    if (missing_positions.find(i) == missing_positions.end()) {
      EXPECT_FALSE(column->col().missing(i)) << " at " << i;
    } else {
      EXPECT_TRUE(column->col().missing(i)) << " at " << i;
    }
  }

  vector<string> raw_strings(column->size());
  for (uint i = 0; i < column->size(); ++i) {
    raw_strings[i] = column->get_row_string(i);
  }
  EXPECT_EQ(
    vector<string>({
        "foo", "bar", "__missing__", "foo", "hello", "world",
        "__missing__", "__missing__", "world"}),
    raw_strings);
}

TEST_F(FlatfilesDataStoreTest, NotExistentFlatfile) {
  EXPECT_NE(nullptr, data_store_->GetColumn("foo"));
  EXPECT_NE(nullptr, data_store_->GetColumn("bar"));
  EXPECT_NE(nullptr, data_store_->GetColumn("foo2"));
  EXPECT_EQ(nullptr, data_store_->GetColumn("foo3"));
  EXPECT_EQ(nullptr, data_store_->GetColumn("bar2"));
}

}  // namespace gbdt
