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

#ifndef LOSS_FUNC_PAIRWISE_LOGLOSS_H_
#define LOSS_FUNC_PAIRWISE_LOGLOSS_H_

#include <functional>

#include "loss_func_math.h"
#include "loss_func_pairwise.h"

namespace gbdt {

class LossFuncConfig;

// PairwiseLogloss: \sum_(\forall pairs) log(1+exp(fn - fp)).
class PairwiseLogLoss : public Pairwise {
 public:
  PairwiseLogLoss(const Config& config)
      : Pairwise(config, false,
                 [](double delta_target,
                    double delta_func) {
                   return ComputeLogLoss(1.0, delta_func); }) {}
};

}  // namespace

#endif  // LOSS_FUNC_PAIRWISE_LOGLOSS_H_
