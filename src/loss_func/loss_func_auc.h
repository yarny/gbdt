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

#ifndef LOSS_FUNC_AUC_H_
#define LOSS_FUNC_AUC_H_

#include <functional>

#include "loss_func_math.h"
#include "loss_func_pairwise.h"

namespace gbdt {

class Config;

// AUC: \sum_(\forall pairs) max(0, f_n + 1.0 - f_p). We huberize the hinge loss so we
// can get hessian out of it.
class AUC : public Pairwise {
 public:
  AUC(const Config& config)
      : Pairwise(config,
                 [] (double delta_target, double delta_func) {
                   return ComputeHuberizedHinge(1, delta_func); }) {}
};

}  // namespace

#endif  // LOSS_FUNC_AUC_H_
