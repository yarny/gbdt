/*
 * Copyright 2016 Jiang Chen <criver@gmail.com>
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

#include "subsampling.h"

#include <vector>

#include "gtest/gtest.h"
#include "src/base/base.h"
#include "src/utils/utils.h"

namespace gbdt {

TEST(SubsamplingTest, CreateAllSamples) {
  EXPECT_EQ(Subsampling::CreateAllSamples(7),
            vector<uint>({0, 1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(Subsampling::CreateAllSamples(10),
            vector<uint>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(SubsamplingTest, UniformSubsampleSameSeed) {
  Subsampling::Reseed(1234);
  vector<uint> samples1 = Subsampling::UniformSubsample(100, 0.5);
  vector<uint> samples2 = Subsampling::UniformSubsample(100, 0.5);
  Subsampling::Reseed(1234);
  vector<uint> samples3 = Subsampling::UniformSubsample(100, 0.5);
  vector<uint> samples4 = Subsampling::UniformSubsample(100, 0.5);
  EXPECT_EQ(samples1, samples3);
  EXPECT_EQ(samples2, samples4);
}

TEST(SubsamplingTest, DivideSamples) {
  vector<uint> samples = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto slices = Subsampling::DivideSamples(samples, 3);
  EXPECT_EQ(3, slices.size());
  EXPECT_EQ(vector<uint>({0, 1, 2}), VectorSliceToVector(slices[0]));
  EXPECT_EQ(vector<uint>({3, 4, 5}), VectorSliceToVector(slices[1]));
  EXPECT_EQ(vector<uint>({6, 7, 8}), VectorSliceToVector(slices[2]));

  slices = Subsampling::DivideSamples(samples, 4);
  EXPECT_EQ(4, slices.size());
  EXPECT_EQ(vector<uint>({0, 1, 2}), VectorSliceToVector(slices[0]));
  EXPECT_EQ(vector<uint>({3, 4}), VectorSliceToVector(slices[1]));
  EXPECT_EQ(vector<uint>({5, 6}), VectorSliceToVector(slices[2]));
  EXPECT_EQ(vector<uint>({7, 8}), VectorSliceToVector(slices[3]));
}

}  // namespace gbdt
