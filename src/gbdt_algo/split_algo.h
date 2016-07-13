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

#ifndef SPLIT_ALGO_H_
#define SPLIT_ALGO_H_

#include <unordered_set>
#include <utility>
#include <vector>

#include "src/base/base.h"
#include "src/loss_func/gradient_data.h"
#include "src/utils/utils.h"
#include "src/utils/vector_slice.h"

namespace gbdt {

class Column;
class IntegerizedColumn;
class Split;
class Config;

// Histogram contains weighted sums of gradients and hessians
// for each bucktized feature values.
class Histogram {
 public:
  Histogram(const IntegerizedColumn& feature,
            FloatVector w,
            const vector<GradientData>& gradient_data_vec,
            const VectorSlice<uint>& samples);
  inline int size() const {
    return non_zero_values_.size();
  }
  inline const GradientData& data(int i) const {
    return histograms_[value(i)];
  }
  inline const uint value(int i) const {
    return non_zero_values_[i];
  }
  bool HasMissingValue() const;
  inline const GradientData& DataOnMissing() const;

  void SortOnNodeScore(double lambda);

 private:
  void ComputeHistograms(const IntegerizedColumn& feature,
                         FloatVector w,
                         const vector<GradientData>& gradient_data,
                         const VectorSlice<uint>& samples);
  vector<GradientData> histograms_;
  vector<uint> non_zero_values_;
};

// Partitions samples into left and right according to the split.
pair<VectorSlice<uint>, VectorSlice<uint>>
Partition(const Column* feature, const Split& split, VectorSlice<uint> samples);

// Finds the best split for a feature. Returns false on failure.
//
// 1. For both float and strings features, the algorithm has complexity of O(n + k), where
//    k is the number of unique values of the data.
//
// 2. total is inputs to the algorithm to save one pass over the data.
bool FindBestSplit(const Column* feature,
                   FloatVector w,
                   const vector<GradientData>* gradient_data_vec,
                   const VectorSlice<uint>& samples,
                   const Config& config,
                   const GradientData& total,
                   Split* split);

}  // namespace gbdt

#endif  // SPLIT_ALGO_H_
