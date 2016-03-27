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

#include "loss_func_lambdamart.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "loss_func_math.h"
#include "src/data_store/column.h"
#include "src/proto/config.pb.h"

using namespace std::placeholders;

namespace gbdt {

const int kNumPrecomputedDiscounts = 100;

inline double discount(uint rank, float base) {
  return log(base) / log(base + rank);
}

LambdaMART::LambdaMART(const LossFuncConfig& config)
    : Pairwise(config, [](uint i, uint j, const vector<double>* f) {
        return ComputeLogLoss(1, (*f)[i] - (*f)[j]); }) {
  if (config.lambdamart_params().dcg_base() > 0) {
    dcg_base_ = config.lambdamart_params().dcg_base();
  }
  precomputed_discounts_.reserve(kNumPrecomputedDiscounts);
  for (int i = 0; i < kNumPrecomputedDiscounts; ++i) {
    precomputed_discounts_[i] = discount(i, dcg_base_);
  }

  auto precomputed_discount = [](uint rank, double base,
                                 const vector<double>* precomputed_discounts) {
    return (rank < precomputed_discounts->size() ? (*precomputed_discounts)[rank] :
            discount(rank, base));
  };

  discount_ = std::bind(precomputed_discount, _1, dcg_base_, &precomputed_discounts_);
}

// TODO(criver): Solve the following problem:
// At the beginning of the training, all scores are zero. We would sort them anyway and
// use the ranks to weight different pairs differently. Instead, we should know that
// and put uniform weights on aell pairs at the beginning.
//
// Proposal:
// Sort scores in descending order.
// Compute the rank delta between adjacent store by p = 1 / (1 + exp(f_i - f_j)).
// rank_j =  rank_i + 2 * (p - 0.5).
// At the beginning of the training, there are less separation of scores. All ranks will be 0.
// With more score separation, the ranks will be more separated out.
vector<uint> ComputeRanks(const vector<uint>& group, const vector<double>& f) {
  // Computes the ranking based on the current function.
  vector<uint> ranking(group.size());
  for (int i = 0; i < ranking.size(); ++i) {
    ranking[i] = i;
  }
  auto sort_by_f = [](uint i, uint j, const vector<uint>* group, const vector<double>* f) {
    return (*f)[(*group)[i]] > (*f)[(*group)[j]];
  };

  // Sort by f.
  sort(ranking.begin(), ranking.end(), std::bind(sort_by_f, _1, _2, &group, &f));

  // For each index, store its ranks.
  vector<uint> ranks(group.size());

  for (int i = 0; i < ranking.size(); ++i) {
    ranks[ranking[i]] = i;
  }

  return ranks;
}

function<double(const pair<uint, uint>&)> LambdaMART::GeneratePairWeightingFunc(
    const vector<uint>& group, const vector<double>& f) {
  ranks_ = ComputeRanks(group, f);

  // In LambdaMART the weight is the difference of DCG if the ranking of the pair was inverted.
  auto dcg_diff = [](const pair<uint, uint>& p,
                     const vector<uint>* ranks,
                     const vector<uint>* group,
                     const RawFloatColumn* target_column,
                     function<double(uint)> discount) {
    return (((*target_column)[(*group)[p.first]] - (*target_column)[(*group)[p.second]]) *
            fabs(discount((*ranks)[p.first]) - discount((*ranks)[p.second])));
  };

  return std::bind(dcg_diff, _1, &ranks_, &group, target_column_, discount_);
}

}  // namespace gbdt
