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

#include "src/data_store/data_store.h"
#include "src/utils/subsampling.h"
#include "src/utils/threadpool.h"

using namespace std::placeholders;

DECLARE_int32(num_threads);

namespace gbdt {

Pairwise::Group::Group(vector<uint>&& group, const RawFloatColumn* target_column,
                       std::mt19937* generator)
    : group_(group), generator_(generator) {
  // Sort groups on the descending order of targets.
  auto sort_by_target_desc = [](uint i, uint j, const RawFloatColumn* target) {
    return (*target)[i] > (*target)[j];
  };
  sort(group_.begin(), group_.end(), std::bind(sort_by_target_desc, _1, _2, target_column));

  // Find change boundaries of targets.
  int last_boundary = 0;
  for (int i = 1; i < group_.size(); ++i) {
    if ((*target_column)[group_[i - 1]] != (*target_column)[group_[i]]) {
      // The pairs consists of this target group and all items following it.
      int block_size = (i - last_boundary);
      num_pairs_ += block_size * (group_.size() - i);
      pair_map_.insert(make_pair(num_pairs_, make_pair(block_size, i)));
      last_boundary = i;
    }
  }
}

pair<uint, uint> Pairwise::Group::SamplePair() const {
  std::uniform_int_distribution<uint> sampler(0, num_pairs_ - 1);
  uint pair_index = sampler(*generator_);
  // Given a pair_index, try to find the actual pair.
  // TODO(criver): describe the algorithm.
  auto it = pair_map_.lower_bound(pair_index + 1);
  int block_size = it->second.first;
  int start_of_neg = it->second.second;
  int local_pair_index = it->first - pair_index - 1;
  uint pos_index = start_of_neg - 1 - local_pair_index % block_size;
  uint neg_index = start_of_neg + local_pair_index / block_size;
  return make_pair(pos_index, neg_index);
}

// TOOD(criver): figure out the how to deal with the genrator.
// One strategy is to use binary-wide generator, but the problem is if we have multi-thread,
// we might need per-thread seed.
unique_ptr<std::mt19937> Pairwise::generator_(new std::mt19937);

Pairwise::Pairwise(const LossFuncConfig& config, Pairwise::PairwiseLossFunc loss_func)
    : config_(config), loss_func_(loss_func) {
  CHECK(config_.pairwise_target().pair_sampling_rate() > 0)
      << "Please specify a non-zero pair sampling rate.";
}

bool Pairwise::Init(DataStore* data_store, const vector<float>& w) {
  const string& target_column_name = config_.target_column();
  const string& group_column_name = config_.pairwise_target().group_column();

  w_ = &w;

  if (target_column_name.empty()) {
    LOG(ERROR) << "Please specify target_column for Pairwise loss.";
    return false;
  }
  auto target_column = data_store->GetRawFloatColumn(target_column_name);
  if (!target_column) {
    LOG(ERROR) << "Failed to get target column " << target_column_name;
    return false;
  }
  target_column_ = target_column;

  // Construct groups.
  vector<vector<uint>> groups;
  if (group_column_name.empty()) {
    // When group is not specified, every thing is one group.
    groups.resize(1);
    groups[0] = Subsampling::CreateAllSamples(w_->size());
  } else {
    auto group_column = data_store->GetStringColumn(group_column_name);
    if (!group_column) {
      LOG(ERROR) << "Failed to get group column " << group_column_name;
      return false;
    }

    groups.resize(group_column->max_int() - 1);
    const auto& group_col = group_column->col();
    for (int i = 0; i < group_column->size(); ++i) {
      uint group_id = group_col[i];
      groups[group_id - 1].push_back(i);
    }
  }

  groups_.reserve(groups.size());
  for (auto& group : groups) {
    groups_.emplace_back(std::move(group), target_column, generator_.get());
  }
  slices_ = Subsampling::DivideSamples(groups_.size(), FLAGS_num_threads * 5);

  return true;
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

  double sampling_rate = config_.pairwise_target().pair_sampling_rate();
  if (sampling_rate <= 0) return;

  // Sample pairs and compute pairwise loss.
  vector<double> losses(slices_.size(), 0.0);
  vector<double> weight_sums(slices_.size(), 0.0);
  {
    ThreadPool pool(FLAGS_num_threads);
    for (int j = 0; j < slices_.size(); ++j) {
      pool.Enqueue([&, this, &slice=slices_[j], &loss=losses[j], &weight_sum=weight_sums[j]]() {
          for (int group_index = slice.first; group_index < slice.second; ++group_index) {
            const auto& group = groups_[group_index];
            int sample_pairs = group.num_pairs() * sampling_rate;
            auto pair_weighting_func = GeneratePairWeightingFunc(group.group(), f);
            for (int i = 0; i < sample_pairs; ++i) {
              auto p = group.SamplePair();
              auto pos_sample = group.group()[p.first];
              auto neg_sample = group.group()[p.second];
              double weight = (*w_)[pos_sample] * (*w_)[neg_sample] * pair_weighting_func(p);
              double delta_target = (*target_column_)[pos_sample] - (*target_column_)[neg_sample];
              double delta_func = f[pos_sample] - f[neg_sample];

              auto data = loss_func_(delta_target, delta_func);
              auto& pos_gradient_data = (*gradient_data_vec)[pos_sample];
              auto& neg_gradient_data = (*gradient_data_vec)[neg_sample];
              pos_gradient_data.g += data.gradient_data.g * weight;
              neg_gradient_data.g -= data.gradient_data.g * weight;
              pos_gradient_data.h += 2.0 * weight * data.gradient_data.h;
              neg_gradient_data.h += 2.0 * weight * data.gradient_data.h;
              loss += data.loss * weight;
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
  return [](const pair<uint, uint>& p) { return 1.0;};
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
