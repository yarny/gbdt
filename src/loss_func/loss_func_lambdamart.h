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

#ifndef LOSS_FUNC_LAMBDAMART_H_
#define LOSS_FUNC_LAMBDAMART_H_

#include "loss_func_pairwise.h"

namespace gbdt {

// LambdaMART (http://research.microsoft.com/pubs/132652/MSR-TR-2010-82.pdf)
// is implemented as pairwise logloss with pair weighted by dcg diff.
class LambdaMART : public Pairwise {
 public:
  LambdaMART(const Config& config);

 private:
  function<double(const pair<uint, uint>&)> PairWeightingFunc(const Group& group) const override;
  float dcg_base_ = 2.0;

  // ranks_ for each group. Used to generate pair weight function for LambdaMart.
  vector<uint> ranks_;
  // Precomputed discounts for fast computation.
  vector<double> precomputed_discounts_;
  function<double(uint)> discount_;
};

}  // namespace gbdt

#endif  // LOSS_FUNC_LAMBDAMART_H_
