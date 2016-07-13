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

#ifndef LOSS_FUNC_GBRANK_H_
#define LOSS_FUNC_GBRANK_H_

#include <functional>
#include <map>
#include <vector>

#include "loss_func_pairwise.h"
#include "loss_func_math.h"
#include "src/base/base.h"
#include "src/proto/config.pb.h"

namespace gbdt {

// GBRank: Pairwise squared hinge loss: max(f(x) - f(y) - \tau, 0)^2
// http://www.cc.gatech.edu/~zha/papers/fp086-zheng.pdf
class GBRank : public Pairwise {
 public:
  GBRank(const Config& config)
      : Pairwise(config,
                 [](double delta_target, double delta_func) {
                   return ComputeSquaredHinge(delta_target, delta_func);
                 }) {}
};

}  // namespace gbdt

#endif  // LOSS_FUNC_GBRANK_H_
