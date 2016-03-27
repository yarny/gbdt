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

#include "vector_slice.h"

#include <vector>

#include "gtest/gtest.h"

namespace gbdt {

class VectorSliceTest : public ::testing::Test {
 protected:
  void SetUp() {
  }
  vector<int> vec_ = {0, 1, 2, 3, 4, 5, 6, 7};
};

TEST_F(VectorSliceTest, TestSliceConstructor0) {
  VectorSlice<int> slice(vec_, 2, 4);
  EXPECT_EQ(4, slice.size());
  EXPECT_EQ(2, slice[0]);
  EXPECT_EQ(3, slice[1]);
  EXPECT_EQ(4, slice[2]);
  EXPECT_EQ(5, slice[3]);
}

TEST_F(VectorSliceTest, TestSlicesConstructor1) {
  VectorSlice<int> slice(vec_.begin() + 2, vec_.begin() + 6);
  EXPECT_EQ(4, slice.size());
  EXPECT_EQ(2, slice[0]);
  EXPECT_EQ(3, slice[1]);
  EXPECT_EQ(4, slice[2]);
  EXPECT_EQ(5, slice[3]);
}

TEST_F(VectorSliceTest, TestSliceConstructor2) {
  VectorSlice<int> all_slice(vec_);
  VectorSlice<int> slice(all_slice, 2, 4);
  EXPECT_EQ(4, slice.size());
  EXPECT_EQ(2, slice[0]);
  EXPECT_EQ(3, slice[1]);
  EXPECT_EQ(4, slice[2]);
  EXPECT_EQ(5, slice[3]);
}

TEST_F(VectorSliceTest, TestSliceSwap) {
  VectorSlice<int> slice(vec_.begin() + 2, vec_.begin() + 6);
  std::swap(slice[0], slice[3]);
  EXPECT_EQ(vector<int>({0, 1, 5, 3, 4, 2, 6, 7}), vec_);
}

}  // namespace gbdt
