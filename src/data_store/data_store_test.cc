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

#include "data_store.h"

#include <memory>

#include "gtest/gtest.h"
#include "mem_data_store.h"
#include "flatfiles_data_store.h"

namespace gbdt {

class DataStoreTest : public ::testing::Test {
 protected:
  void SetUp() {
    data_store_.reset(new FlatfilesDataStore("src/data_store/testdata/flatfiles_data_store_test"));
  }
  unique_ptr<DataStore> data_store_;
};

TEST_F(DataStoreTest, TestGetColumn) {
  EXPECT_NE(nullptr, data_store_->GetBucketizedFloatColumn("foo"));
  EXPECT_NE(nullptr, data_store_->GetRawFloatColumn("foo2"));
  EXPECT_NE(nullptr, data_store_->GetStringColumn("bar"));

  EXPECT_EQ(nullptr, data_store_->GetRawFloatColumn("foo"));
  EXPECT_EQ(nullptr, data_store_->GetStringColumn("foo"));
  EXPECT_EQ(nullptr, data_store_->GetBucketizedFloatColumn("foo2"));
  EXPECT_EQ(nullptr, data_store_->GetStringColumn("foo2"));
  EXPECT_EQ(nullptr, data_store_->GetBucketizedFloatColumn("bar"));
  EXPECT_EQ(nullptr, data_store_->GetRawFloatColumn("bar"));

  EXPECT_EQ(nullptr, data_store_->GetRawFloatColumn("bar2"));
  EXPECT_EQ(nullptr, data_store_->GetBucketizedFloatColumn("bar2"));
  EXPECT_EQ(nullptr, data_store_->GetStringColumn("bar2"));
}

}  // namespace gbdt
