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

#ifndef SUBSAMPLING_H_
#define SUBSAMPLING_H_

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "src/base/base.h"
#include "src/utils/vector_slice.h"

namespace gbdt {

class SamplingConfig;
class UniformSamplingConfig;

class Subsampling {
public:
  static void Reseed(int seed);

  // Construct the sample set [0,n-1]
  static vector<uint> CreateAllSamples(uint n);
  static vector<uint> UniformSubsample(uint n, double rate);

  // Divide samples uniformly into gropus.
  static vector<VectorSlice<uint>> DivideSamples(VectorSlice<uint> samples, int num_groups);
  static vector<pair<uint, uint>> DivideSamples(int num_samples, int num_groups);
  static std::mt19937* get_generator() {
    return &generator_;
  }

private:
  static std::mt19937 generator_;
  static std::uniform_real_distribution<double> uniform_01_;
};

}  // namespace gbdt

#endif  // SUBSAMPLING_H_
