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

#include <random>

#include "src/proto/config.pb.h"
#include "src/utils/utils.h"
#include "src/utils/vector_slice.h"

namespace gbdt {

std::mt19937 Subsampling::generator_;

std::uniform_real_distribution<double> Subsampling::uniform_01_(0.0, 1.0);

void Subsampling::Reseed(int seed) {
  generator_.seed(seed);
}

vector<uint> Subsampling::UniformSubsample(uint n, double rate) {
  vector<uint> samples;
  samples.reserve(max(1, int(rate * n)));
  for (uint i = 0; i < n; ++i) {
    if (uniform_01_(generator_) < rate) {
      samples.emplace_back(i);
    }
  }
  return samples;
}

vector<uint> Subsampling::CreateAllSamples(uint n) {
  vector<uint> samples(n);
  for (uint i = 0; i < n; ++i) {
    samples[i] = i;
  }
  return samples;
}

static vector<uint> DivideSamplesHelper(int num_samples, uint num_groups) {
  vector<uint> group_sizes(num_groups);
  for (int i = 0; i < group_sizes.size(); ++i) {
    group_sizes[i] = num_samples / num_groups;
  }

  int left = num_samples % num_groups;
  for (int i = 0; i < left; ++i) {
    group_sizes[i] += 1;
  }
  return group_sizes;
}

vector<pair<uint, uint>> Subsampling::DivideSamples(int num_samples, int num_groups) {
  vector<uint> group_sizes = DivideSamplesHelper(num_samples, num_groups);
  vector<pair<uint, uint>> slices;
  int start = 0;
  for (auto size : group_sizes) {
    slices.emplace_back(start, start + size);
    start += size;
  }
  return slices;
}

vector<VectorSlice<uint>> Subsampling::DivideSamples(
    VectorSlice<uint> samples, int num_groups) {
  vector<uint> group_sizes = DivideSamplesHelper(samples.size(), num_groups);
  vector<VectorSlice<uint>> slices;
  slices.reserve(num_groups);
  auto start = samples.begin();

  for (auto size : group_sizes) {
    slices.emplace_back(start, start + size);
    start += size;
  }
  return slices;
}

}  // namespace gbdt
