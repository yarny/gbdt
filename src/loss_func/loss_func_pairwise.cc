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
#include <string>
#include <vector>

#include "external/cppformat/format.h"

#include "src/data_store/column.h"
#include "src/utils/subsampling.h"
#include "src/utils/threadpool.h"

using namespace std::placeholders;

DECLARE_int32(num_threads);

namespace gbdt {

Pairwise::Pairwise(const LossFuncConfig& config, Pairwise::PairwiseLossFunc loss_func)
    : config_(config), loss_func_(loss_func) {
  CHECK(config_.pairwise_config().pair_sampling_rate() > 0)
      << "Please specify a non-zero pair sampling rate.";
}

Status Pairwise::Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* group_column) {
  w_ = w;
  y_ = y;

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
      groups[group_id - 1].push_back(i);
    }
  }

  groups_.reserve(groups.size());
  for (auto& group : groups) {
    groups_.emplace_back(std::move(group), y);
  }
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

  double sampling_rate = config_.pairwise_config().pair_sampling_rate();

  // Sample pairs and compute pairwise loss.
  vector<double> losses(slices_.size(), 0.0);
  vector<double> weight_sums(slices_.size(), 0.0);
  {
    ThreadPool pool(FLAGS_num_threads);
    for (int j = 0; j < slices_.size(); ++j) {
      pool.Enqueue([&, this, &slice=slices_[j], &loss=losses[j], &weight_sum=weight_sums[j]]() {
          std::mt19937* generator = Subsampling::get_generator();
          for (int group_index = slice.first; group_index < slice.second; ++group_index) {
            const auto& group = groups_[group_index];
            uint64 num_sample_pairs = group.num_pairs() * sampling_rate;
            auto pair_weighting_func = GeneratePairWeightingFunc(group.group(), f);
            for (int i = 0; i < num_sample_pairs; ++i) {
              auto p = group.SamplePair(generator);
              auto pos_sample = group[p.first];
              auto neg_sample = group[p.second];
              double weight = w_(pos_sample) * w_(neg_sample) * pair_weighting_func(p);
              double delta_target = y_(pos_sample) - y_(neg_sample);
              double delta_func = f[pos_sample] - f[neg_sample];

              auto data = loss_func_(delta_target, delta_func);
              auto& pos_gradient_data = (*gradient_data_vec)[pos_sample];
              auto& neg_gradient_data = (*gradient_data_vec)[neg_sample];
              pos_gradient_data.g += weight * data.gradient_data.g;
              neg_gradient_data.g -= weight * data.gradient_data.g;
              pos_gradient_data.h += 2.0 * weight * data.gradient_data.h;
              neg_gradient_data.h += 2.0 * weight * data.gradient_data.h;
              loss += weight * data.loss;
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
function<double(const pair<uint, uint>&)> Pairwise::GeneratePairWeightingFunc(
    const vector<uint>& group, const vector<double>& f) {
  if (config_.pairwise_config().weight_by_delta_target()) {
    return [&, this] (const pair<uint, uint>& p) {
      return y_(group[p.first]) - y_(group[p.second]);
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
