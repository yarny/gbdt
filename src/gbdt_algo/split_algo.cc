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

#include "split_algo.h"

#include <algorithm>
#include <glog/logging.h>
#include <numeric>
#include <unordered_set>

#include "src/base/base.h"
#include "src/data_store/column.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"
#include "src/utils/vector_slice.h"

using namespace std::placeholders;

namespace gbdt {

const double kFloatTolerance = 1e-6;

pair<VectorSlice<uint>, VectorSlice<uint>>
Partition(const BinnedFloatColumn* feature, const Split& split, VectorSlice<uint> samples) {
  CHECK(split.has_float_split()) << "Split and feature type mismatch for " << feature->name();
  uint left_size = 0;
  bool missing_to_right = split.float_split().missing_to_right_child();
  float threshold = split.float_split().threshold();
  const auto& col = feature->col();
  for (uint i = 0; i < samples.size(); ++i) {
    if ((col.missing(samples[i]) && !missing_to_right) ||
        (!col.missing(samples[i]) && feature->get_row_max(samples[i]) < threshold)) {
      swap(samples[i], samples[left_size++]);
    }
  }
  return make_pair(VectorSlice<uint>(samples, 0, left_size),
                   VectorSlice<uint>(samples, left_size, samples.size() - left_size));
}

pair<VectorSlice<uint>, VectorSlice<uint>>
Partition(const StringColumn* feature, const Split& split, VectorSlice<uint> samples) {
  CHECK(split.has_cat_split()) << "Split and feature type mismatch for " << feature->name();
  // TODO(criver): converting repeated fields everytime seems wasteful. Rewrite this if protobuf
  // has better support for set. However, this is arguably not so bad, since this is done at
  // per-node level not the per-instance level.
  unordered_set<uint> categories;
  if (split.cat_split().internal_categorical_index_size() > 0) {
    categories.insert(split.cat_split().internal_categorical_index().begin(),
                      split.cat_split().internal_categorical_index().end());
  } else {
    for (const auto& cat: split.cat_split().category()) {
      uint cat_index;
      if (feature->get_cat_index(cat, &cat_index)) {
        categories.insert(cat_index);
      }
    }
  }

  const auto& col = feature->col();
  uint left_size = 0;
  for (uint i = 0; i < samples.size(); ++i) {
    if (categories.find(col[samples[i]]) != categories.end())
      swap(samples[i], samples[left_size++]);
  }
  return make_pair(VectorSlice<uint>(samples, 0, left_size),
                   VectorSlice<uint>(samples, left_size, samples.size() - left_size));
}

pair<VectorSlice<uint>, VectorSlice<uint>>
Partition(const Column* feature, const Split& split, VectorSlice<uint> samples) {
  if (feature->type() == Column::kStringColumn) {
    return Partition(static_cast<const StringColumn*>(feature), split, samples);
  } else if (feature->type() == Column::kBinnedFloatColumn) {
    return Partition(static_cast<const BinnedFloatColumn*>(feature), split, samples);
  } else {
    return make_pair(VectorSlice<uint>(samples, 0, 0), VectorSlice<uint>(samples));
  }
}

Histogram::Histogram(const IntegerizedColumn& feature,
                     const vector<float>& w,
                     const vector<GradientData>& gradient_data_vec,
                     const VectorSlice<uint>& samples) {
  ComputeHistograms(feature, w, gradient_data_vec, samples);
}

// This is the main work horse of the whole algorithm. Please make sure
// it is written in an efficient way.
void Histogram::ComputeHistograms(const IntegerizedColumn& feature,
                                  const vector<float>& w,
                                  const vector<GradientData>& gradient_data_vec,
                                  const VectorSlice<uint>& samples) {
  uint max_int = feature.max_int();
  histograms_.resize(max_int);
  non_zero_values_.reserve(max_int);
  const auto& col = feature.col();
  // Compute histograms.
  for(auto index : samples) {
    auto& histogram = histograms_[col[index]];
    const auto& weight = w[index];
    const auto& gradient_data = gradient_data_vec[index];
    histogram.g += weight * gradient_data.g;
    histogram.h += weight * gradient_data.h;
  }

  for (uint i = 0; i < max_int; ++i) {
    if (histograms_[i].g != 0 && histograms_[i].h != 0) {
      non_zero_values_.push_back(i);
    }
  }
}

bool Histogram::HasMissingValue() const {
  // IntegerizedColumn put 0 as the missing value.
  return histograms_[0].g != 0 || histograms_[0].h != 0;
}

const GradientData& Histogram::DataOnMissing() const {
  // IntegerizedColumn put 0 as the missing value.
  return histograms_[0];
}

void Histogram::SortOnNodeScore(double lambda) {
  // Compare the Node Score.
  auto compare_node_score = [](uint x, uint y,
                               const vector<GradientData>* histograms, double lambda) {
    return ((*histograms)[x].Score(lambda) < (*histograms)[y].Score(lambda));
  };

  sort(non_zero_values_.begin(), non_zero_values_.end(), std::bind(compare_node_score, _1, _2, &histograms_, lambda));
}

// Data structure for holding the split.
struct SplitPoint {
  int left_point = -1;
  bool missing_on_right = false;
  double gain = 0;
};

// Finds the left and right points for the split.
// place_missing tells the algorithm to try placing missing on the left or right.
bool FindBestSplitPoint(const IntegerizedColumn& feature,
                        const SplitConfig& config,
                        const Histogram histogram,
                        const GradientData& total,
                        bool place_missing,
                        SplitPoint* split_point) {
  double lambda = config.l2_lambda();
  double total_energy = total.Energy(lambda);

  GradientData left;
  GradientData right;
  // Scanning to find the best splitting point.
  for (int i = 0; i < histogram.size(); ++i) {
    right = total - left;
    if (i > 0) {
      double gain = left.Energy(lambda) + right.Energy(lambda) - total_energy;
      bool missing_on_right = false;
      if (place_missing && histogram.HasMissingValue()) {
        // By default, for float missing value is placed on the left since they take index 0.
        // We will try to put them on the right to see if it improves the gain. If so, we will put
        // it on the right.
        const auto& data_on_missing = histogram.DataOnMissing();        
        double gain_on_right = (left - data_on_missing).Energy(lambda) +
                               (right + data_on_missing).Energy(lambda) -
                               total_energy;
        if (gain_on_right > gain) {
          gain = gain_on_right;
          missing_on_right = true;
        }
      }
      
      if (gain > max(split_point->gain, max(config.min_gain(), kFloatTolerance))) {
        split_point->gain = gain;
        split_point->left_point = i - 1;
        split_point->missing_on_right = missing_on_right;
      }
    }

    left += histogram.data(i);
  }

  return split_point->left_point >= 0;
}

// To best split point, we first aggregate the score and
// weights into each unique value represented by an integer,
// then we iterate over [0, num_unique_values) to find the best split
// point. The time complexity is O(n+num_unique_values).
bool FindBestFloatSplit(const BinnedFloatColumn& feature,
                        const vector<float>* w,
                        const vector<GradientData>* gradient_data_vec,
                        const VectorSlice<uint>& samples,
                        const SplitConfig& config,
                        const GradientData& total,
                        Split* split) {
  Histogram histogram(feature, *w, *gradient_data_vec, samples);

  SplitPoint split_point;
  if (!FindBestSplitPoint(feature, config, histogram, total, true, &split_point)) {
    return false;
  }

  split->set_gain(split_point.gain);
  split->mutable_float_split()->set_missing_to_right_child(split_point.missing_on_right);
  
  split->mutable_float_split()->set_threshold(
      (feature.get_bin_max(histogram.value(split_point.left_point)) +
       feature.get_bin_min(histogram.value(split_point.left_point + 1))) / 2.0);
  return true;
}

bool FindBestStringSplit(const StringColumn& feature,
                         const vector<float>* w,
                         const vector<GradientData>* gradient_data_vec,
                         const VectorSlice<uint>& samples,
                         const SplitConfig& config,
                         const GradientData& total,
                         Split* split) {
  Histogram histogram(feature, *w, *gradient_data_vec, samples);
  // For categorical features, since there is no preset order, we can
  // sort them based on their node scores and find the optimal subset.
  histogram.SortOnNodeScore(config.l2_lambda());

  SplitPoint split_point;
  if (!FindBestSplitPoint(feature, config, histogram, total, false, &split_point)) {
    return false;
  }

  split->set_gain(split_point.gain);
  // Construct the set of values. Always put the smaller set on the left.
  auto* cat_split = split->mutable_cat_split();
  if (split_point.left_point + 1 <= histogram.size() - split_point.left_point - 1) {
    for (uint i = 0; i <= split_point.left_point; ++i) {
      cat_split->add_internal_categorical_index(histogram.value(i));
    }
  } else {
    for (uint i = split_point.left_point + 1; i < histogram.size(); ++i) {
      cat_split->add_internal_categorical_index(histogram.value(i));
    }
  }

  return true;
}

bool FindBestSplit(const Column* feature,
                   const vector<float>* w,
                   const vector<GradientData>* gradient_data_vec,
                   const VectorSlice<uint>& samples,
                   const SplitConfig& config,
                   const GradientData& total,
                   Split* split) {
  switch (feature->type()) {
    case Column::kStringColumn:
      return FindBestStringSplit(static_cast<const StringColumn&>(*feature),
                                 w, gradient_data_vec, samples, config, total, split);
    case Column::kBinnedFloatColumn:
      return FindBestFloatSplit(static_cast<const BinnedFloatColumn&>(*feature),
                                w, gradient_data_vec, samples, config, total, split);
    default:
      return false;
  }
}

}  // namespace gbdt
