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

#include "loss_func_pairwise.h"

#include <algorithm>
#include <gflags/gflags.h>
#include <string>
#include <vector>

#include "external/cppformat/format.h"

#include "src/data_store/column.h"
#include "src/utils/subsampling.h"
#include "src/utils/threadpool.h"

using namespace std::placeholders;

DECLARE_int32(num_threads);

namespace gbdt {

namespace {

double ComputePairSamplingProbability(int num_rows, double pair_sampling_rate,
                                      const vector<Group>& groups) {
  uint64 num_total_pairs = 0;
  for (const auto& group : groups) {
    num_total_pairs += group.num_pairs();
  }
  return pair_sampling_rate * num_rows / num_total_pairs / 2.0;
}

}  // namespace

Pairwise::Pairwise(const Config& config, bool rerank, PointwiseLossFunc loss_func) :
    pair_sampling_rate_(config.pair_sampling_rate()),
    pair_weight_by_delta_target_(config.pair_weight_by_delta_target()),
    equal_group_weight_(config.equal_group_weight()),
    rerank_(rerank),
    loss_func_(loss_func) {
}

Status Pairwise::Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* group_column) {
  w_ = w;
  y_ = y;

  if (pair_sampling_rate_ <= 0) {
    return Status(error::INVALID_ARGUMENT,
                  fmt::format("pair_sampling_rate need to by positive (actual {0})", pair_sampling_rate_));
  }

  // Construct groups.
  vector<vector<uint>> groups;
  if (!group_column) {
    // When group is not specified, every thing is one group.
    groups.resize(1);
    groups[0] = Subsampling::CreateAllSamples(num_rows);
  } else {
    groups.resize(group_column->max_int() - 1);
    const auto& group_col = group_column->col();
    for (int i = 0; i < group_column->size(); ++i) {
      uint group_id = group_col[i];
      groups[group_id - 1].emplace_back(i);
    }
  }

  groups_.reserve(groups.size());
  for (auto& group : groups) {
    groups_.emplace_back(std::move(group), y);
  }

  min_num_pairs_ = 1;
  for (const auto& group : groups_) {
    min_num_pairs_ = max(min_num_pairs_, group.num_pairs());
  }

  pair_sampling_probability_ = ComputePairSamplingProbability(num_rows, pair_sampling_rate_,
                                                              groups_);

  slices_ = Subsampling::DivideSamples(groups_.size(), FLAGS_num_threads * 5);

  return Status::OK;
}

void Pairwise::ComputeFunctionalGradientsAndHessians(const vector<double>& f,
                                                     double* c,
                                                     vector<GradientData>* gradient_data_vec,
                                                     string* progress) {
  // Resize g and h if they haven't be resized yet.
  if (gradient_data_vec->size() != f.size()) {
    gradient_data_vec->resize(f.size());
  }
  *c = 0;
  auto set_zero = [](GradientData& x) { x = GradientData(); };
  std::for_each(gradient_data_vec->begin(), gradient_data_vec->end(), set_zero);

  // Sample pairs and compute pairwise loss.
  vector<double> losses(slices_.size(), 0.0);
  vector<double> weight_sums(slices_.size(), 0.0);
  {
    ThreadPool pool(FLAGS_num_threads);
    for (int j = 0; j < slices_.size(); ++j) {
      pool.Enqueue([&, this, &slice=slices_[j], &loss=losses[j], &weight_sum=weight_sums[j]]() {
          std::mt19937* generator = Subsampling::get_generator();
          for (int group_index = slice.first; group_index < slice.second; ++group_index) {
            auto& group = groups_[group_index];
            if (rerank_) group.Rerank(f);

            uint64 num_sample_pairs = group.num_pairs() * pair_sampling_probability_;

            // To make each group's weight constant, we rescale each group's weight by
            // 1.0 / group.num_pairs().
            double weight_rescaling_factor = equal_group_weight_ ?
                                             double(min_num_pairs_) / group.num_pairs() : 1.0;
            auto pair_weighting_func = PairWeightingFunc(group);
            for (int i = 0; i < num_sample_pairs; ++i) {
              auto p = group.SamplePair(generator);
              auto pos_sample = group[p.first];
              auto neg_sample = group[p.second];
              double weight = w_(pos_sample) * w_(neg_sample) * pair_weighting_func(p) *
                              weight_rescaling_factor;
              double delta_target = y_(pos_sample) - y_(neg_sample);
              double delta_func = f[pos_sample] - f[neg_sample];

              auto data = loss_func_(delta_target, delta_func);
              auto& pos_gradient_data = (*gradient_data_vec)[pos_sample];
              auto& neg_gradient_data = (*gradient_data_vec)[neg_sample];
              pos_gradient_data.g += weight * std::get<1>(data);
              neg_gradient_data.g -= weight * std::get<1>(data);
              pos_gradient_data.h += 2.0 * weight * std::get<2>(data);
              neg_gradient_data.h += 2.0 * weight * std::get<2>(data);
              loss += weight * std::get<0>(data);
              weight_sum += weight;
            }
          }
        });
    }
  }

  double loss = std::accumulate(losses.begin(), losses.end(), 0.0);
  double weight_sum = std::accumulate(weight_sums.begin(), weight_sums.end(), 0.0);

  loss /= weight_sum;
  if (progress) {
    *progress = PrepareProgressMessage(loss);
  }
}

// Basic pairwise loss uses uniform weighting.
function<double(const pair<uint, uint>&)> Pairwise::PairWeightingFunc(
    const Group& group) const {
  if (pair_weight_by_delta_target_) {
    return [&group] (const pair<uint, uint>& p) {
      return group.y(p.first) - group.y(p.second);
    };
  } else {
    return [](const pair<uint, uint>& p) { return 1;};
  }
}

string Pairwise::PrepareProgressMessage(double loss) {
  // Prepare for the progress message.
  if (initial_loss_ < 0) {
    initial_loss_ = loss;
  }

  double relative_reduction = initial_loss_ == 0.0 ? 0.0 : (initial_loss_ - loss) / initial_loss_;
  return fmt::format("loss={0},reduced={1:.2f}%", loss, relative_reduction * 100.0);
}

}  // namespace gbdt
