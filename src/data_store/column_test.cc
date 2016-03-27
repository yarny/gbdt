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

#include "column.h"

#include "external/cppformat/format.h"
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace gbdt {

class StringColumnTest : public ::testing::Test {
 protected:
  vector<string> GetColStrings(const StringColumn& column){
    vector<string> col_strings(column.size());

    for (uint i = 0; i < column.size(); ++i) {
      col_strings[i] = column.get_row_string(i);
    }
    return col_strings;
  }

  void TestStringColumn(uint k) {
    int n = 10000;
    vector<string> raw_strings(n);
    for (int i = 0; i < n; ++i) {
      raw_strings[i] = fmt::format("{0}", i % k);
    }

    auto column = Column::CreateStringColumn("foo", raw_strings);
    const auto* string_column = static_cast<const StringColumn*>(column.get());
    ASSERT_NE(nullptr, column);
    EXPECT_EQ("foo", column->name());
    EXPECT_EQ(Column::kStringColumn, column->type());
    EXPECT_EQ(k + 1, string_column->max_int());
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(i % k + 1, string_column->col()[i]) << " at " << i;
    }
    for (int i = 0; i < string_column->size(); ++i) {
      EXPECT_FALSE(string_column->col().missing(i));
      EXPECT_EQ(raw_strings[i], string_column->get_row_string(i));

      EXPECT_EQ(raw_strings[i], string_column->get_cat_string(string_column->col()[i]));
      uint cat_index;
      EXPECT_TRUE(string_column->get_cat_index(raw_strings[i], &cat_index));
      EXPECT_EQ(string_column->col()[i], cat_index);
    }
  }
};

TEST_F(StringColumnTest, TestCreateStringColumn) {
  TestStringColumn(12);
  TestStringColumn(255);
  TestStringColumn(256);
  TestStringColumn(511);
}

TEST_F(StringColumnTest, TestMissingStrings) {
  vector<string> raw_strings = {"hello", "world", "__missing__", "foo", "__missing__", "bar"};
  auto column = Column::CreateStringColumn("foo", raw_strings);
  const auto* string_column = static_cast<const StringColumn*>(column.get());
  ASSERT_TRUE(string_column != nullptr);
  EXPECT_FALSE(string_column->col().missing(0));
  EXPECT_FALSE(string_column->col().missing(1));
  EXPECT_TRUE(string_column->col().missing(2));
  EXPECT_FALSE(string_column->col().missing(3));
  EXPECT_TRUE(string_column->col().missing(4));
  EXPECT_FALSE(string_column->col().missing(5));
  EXPECT_EQ(raw_strings, GetColStrings(*string_column));
}

class FloatColumnTest : public ::testing::Test {
 protected:
  void SetUp() {
    generator_.reset(new std::default_random_engine(1234));
  }

  void TestRandomVecWithEnoughBins(uint k, uint n, uint seed) {
    generator_.reset(new std::default_random_engine(seed));
    auto raw_floats = GenerateRandomFloatVector(k, n);
    // We use 30% more bins than the number of unique values to make sure
    // each unique value is binned to one unique bins.
    auto column = Column::CreateBinnedFloatColumn("foo", raw_floats, uint(1.3 * k));
    ASSERT_NE(nullptr, column);
    EXPECT_EQ(Column::kBinnedFloatColumn, column->type());

    vector<float> expected_max(column->size());
    for (int i = 0; i < column->size(); ++i) {
      expected_max[i] = raw_floats[i];
    }

    auto* float_column = static_cast<const BinnedFloatColumn*>(column.get());
    EXPECT_EQ(expected_max, GetMax(*float_column));
  }

  // Generates a random float vector of length n with k unique values.
  vector<float> GenerateRandomFloatVector(uint k, uint n) {
    std::uniform_int_distribution<int> uniform_int(0, k-1);
    std::uniform_real_distribution<float> uniform_real(0, 1);
    vector<float> unique_values(k);
    for (uint i = 0; i < k; ++i) {
      unique_values[i] = uniform_real(*generator_);
    }

    vector<float> random_floats(n);
    for (uint i = 0; i < n; ++i) {
      random_floats[i] = unique_values[uniform_int(*generator_)];
    }
    return random_floats;
  }

  static vector<uint> GetCol(const BinnedFloatColumn& column) {
    vector<uint> col(column.size());
    for (int i = 0; i < column.size(); ++i) {
      col[i] = column.col()[i];
    }
    return col;
  }

  static vector<float> GetMax(const BinnedFloatColumn& column) {
    vector<float> array(column.size());
    for (int i = 0; i < column.size(); ++i) {
      array[i] = column.get_row_max(i);
    }
    return array;
  }

  unique_ptr<std::default_random_engine> generator_;
};

TEST_F(FloatColumnTest, TestSimpleCreatBinnedFloatColumn) {
  vector<float> raw_floats = {0.2, 0.1, -0.34, 0.2, 0.1, 0.7, 0.8, 23.4, 23.4, 0.8};
  auto column = Column::CreateBinnedFloatColumn("foo", raw_floats, 10);
  ASSERT_NE(nullptr, column);
  EXPECT_EQ(Column::kBinnedFloatColumn, column->type());

  auto* float_column = static_cast<const BinnedFloatColumn*>(column.get());
  EXPECT_EQ(vector<uint>({ 3, 2, 1, 3, 2, 4, 5, 6, 6, 5 }), GetCol(*float_column));

  EXPECT_EQ(vector<float>({0.2, 0.1, -0.34, 0.2, 0.1, 0.7, 0.8, 23.4, 23.4, 0.8}),
            GetMax(*float_column));
}

TEST_F(FloatColumnTest, TestAddAfterBinBuilding) {
  unique_ptr<BinnedFloatColumn> column(new BinnedFloatColumn("foo", 10));
  auto raw_floats0 = vector<float>({1, 2, 3});
  auto raw_floats1 = vector<float>({0.5, 2.5, 3.5});
  column->Add(&raw_floats0);
  column->BuildBins();
  column->Add(&raw_floats1);
  column->Finalize();

  EXPECT_EQ(vector<uint>({1, 2, 3, 1, 3, 4}), GetCol(*column));
  EXPECT_EQ(vector<float>({1, 2, 3, 1, 3, numeric_limits<float>::max()}), GetMax(*column));
}

TEST_F(FloatColumnTest, TestImbalancedDistributionWithEnoughBins) {
  // This case contains 10000 0.0 and 1 of 1, 2, 3, 4, 5 ,6, 7, 8, 9.
  // With enough bins, all unique values are in its own bins.
  vector<float> raw_floats = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  for (uint i = 0; i < 10000; ++i) {
    raw_floats.push_back(0.0);
  }
  auto column = Column::CreateBinnedFloatColumn("foo", raw_floats, 10);
  ASSERT_NE(nullptr, column);
  EXPECT_EQ(Column::kBinnedFloatColumn, column->type());
  auto* float_column = static_cast<const BinnedFloatColumn*>(column.get());
  vector<uint> col(column->size());
  vector<uint> expected_col = {2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (uint i = 0; i < 10000; ++i) {
    expected_col.push_back(1);
  }

  EXPECT_EQ(expected_col, GetCol(*float_column));
  EXPECT_EQ(raw_floats, GetMax(*float_column));
}

TEST_F(FloatColumnTest, TestImbalancedDistributionWithoutEnoughBins) {
  // This case contains 10000 0.0 and 1 of 1, 2, 3, 4, 5 ,6, 7, 8, 9.
  // Without enough bins, 0.0 is in its own bin and the other values are
  // distributed into 5 uniform bins.
  const int kNumZeros = 10000;
  vector<float> raw_floats = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  for (uint i = 0; i < kNumZeros; ++i) {
    raw_floats.push_back(0);
  }

  auto column = Column::CreateBinnedFloatColumn("foo", raw_floats, 6);
  ASSERT_NE(nullptr, column);
  EXPECT_EQ(Column::kBinnedFloatColumn, column->type());
  auto* float_column = static_cast<const BinnedFloatColumn*>(column.get());
  vector<uint> col(column->size());
  vector<uint> expected_col = {2, 2, 3, 3, 4, 4, 5, 5, 6};
  for (uint i = 0; i < kNumZeros; ++i) {
    expected_col.push_back(1);
  }

  EXPECT_EQ(expected_col, GetCol(*float_column));
  vector<float> expected_max = {2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 9.0};
  for (int i = 0; i < kNumZeros; ++i) {
    expected_max.push_back(0);
  }
  EXPECT_EQ(expected_max, GetMax(*float_column));
}

TEST_F(FloatColumnTest, TestRandomVecWithEnoughBins) {
  TestRandomVecWithEnoughBins(100, 10000, 1);
  TestRandomVecWithEnoughBins(100, 10000, 12);
  TestRandomVecWithEnoughBins(100, 10000, 123);
  TestRandomVecWithEnoughBins(100, 10000, 1234);
  TestRandomVecWithEnoughBins(1000, 10000, 1234);
}

TEST_F(FloatColumnTest, TestMissingFloats) {
  vector<float> raw_floats = {0, 1, NAN, 2, 3, NAN, 4, 5, NAN, 6};
  auto column = Column::CreateBinnedFloatColumn("foo", raw_floats, 10);
  ASSERT_NE(nullptr, column);
  EXPECT_EQ(Column::kBinnedFloatColumn, column->type());
  const auto* float_column = static_cast<const BinnedFloatColumn*>(column.get());
  EXPECT_EQ(9, float_column->max_int());
  EXPECT_FALSE(float_column->col().missing(0));
  EXPECT_FALSE(float_column->col().missing(1));
  EXPECT_TRUE(float_column->col().missing(2));
  EXPECT_FALSE(float_column->col().missing(3));
  EXPECT_FALSE(float_column->col().missing(4));
  EXPECT_TRUE(float_column->col().missing(5));
  EXPECT_FALSE(float_column->col().missing(6));
  EXPECT_FALSE(float_column->col().missing(7));
  EXPECT_TRUE(float_column->col().missing(8));
  EXPECT_FALSE(float_column->col().missing(9));

  vector<pair<float, float> > expected_min_max(column->size());
  for (int i = 0; i < float_column->size(); ++i) {
    if (isnan(raw_floats[i])) {
      EXPECT_TRUE(isnan(float_column->get_row_max(i)));
    } else {
      EXPECT_EQ(raw_floats[i], float_column->get_row_max(i));
    }
  }
  EXPECT_FALSE(NAN < 0.2);
  EXPECT_FALSE(NAN >= 0.2);
}

TEST_F(FloatColumnTest, TestRawFloats) {
  vector<float> raw_floats = GenerateRandomFloatVector(10000, 10000);
  auto raw_floats_copy = raw_floats;

  auto column = Column::CreateRawFloatColumn("foo", std::move(raw_floats));
  ASSERT_NE(nullptr, column);
  EXPECT_EQ(Column::kRawFloatColumn, column->type());
  const auto* float_column = static_cast<const RawFloatColumn*>(column.get());
  vector<float> column_floats(float_column->size());
  for (int i = 0; i < float_column->size(); ++i) {
    column_floats[i] = (*float_column)[i];
  }
  EXPECT_EQ(raw_floats_copy, column_floats);
}

}  // namespace gbdt
