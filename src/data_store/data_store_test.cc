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

#include "gtest/gtest.h"
#include "column.h"

namespace gbdt {

class DataStoreTest : public ::testing::Test {
};

TEST_F(DataStoreTest, TestAddAndGet) {
  DataStore data_store;
  auto status = data_store.Add(Column::CreateStringColumn("foo", vector<string>({"a", "b", "c", "d"})));

  EXPECT_TRUE(status.ok());
  status = data_store.Add(Column::CreateStringColumn("foo", vector<string>({"a", "b", "c", "d"})));
  EXPECT_FALSE(status.ok());  // Column exists.
  status = data_store.Add(Column::CreateBucketizedFloatColumn("bar", vector<float>({0.3, 0.2, 0.45, 0.1, 0.3})));
  EXPECT_FALSE(status.ok());  // Lengthes mismatch.
  status = data_store.Add(Column::CreateBucketizedFloatColumn("bar", vector<float>({0.3, 0.2, 0.45, 0.1})));
  EXPECT_TRUE(status.ok());
  status = data_store.Add(Column::CreateRawFloatColumn("bar2", vector<float>({0.3, 0.2, 0.45, 0.1})));
  EXPECT_TRUE(status.ok());

  EXPECT_NE(nullptr, data_store.GetStringColumn("foo"));
  EXPECT_EQ(nullptr, data_store.GetRawFloatColumn("foo"));
  EXPECT_EQ(nullptr, data_store.GetBucketizedFloatColumn("foo"));

  EXPECT_NE(nullptr, data_store.GetBucketizedFloatColumn("bar"));
  EXPECT_EQ(nullptr, data_store.GetRawFloatColumn("bar"));
  EXPECT_EQ(nullptr, data_store.GetStringColumn("bar"));

  EXPECT_NE(nullptr, data_store.GetRawFloatColumn("bar2"));
  EXPECT_EQ(nullptr, data_store.GetBucketizedFloatColumn("bar2"));
  EXPECT_EQ(nullptr, data_store.GetStringColumn("bar2"));

  EXPECT_EQ(nullptr, data_store.GetRawFloatColumn("foo2"));
  EXPECT_EQ(nullptr, data_store.GetBucketizedFloatColumn("foo2"));
  EXPECT_EQ(nullptr, data_store.GetStringColumn("foo2"));

  data_store.RemoveColumnIfExists("foo");
  EXPECT_EQ(nullptr, data_store.GetStringColumn("foo"));
}

}  // namespace gbdt
