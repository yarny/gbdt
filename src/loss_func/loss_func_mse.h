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

#ifndef LOSS_FUNC_MSE_H_
#define LOSS_FUNC_MSE_H_

#include "loss_func_math.h"
#include "loss_func_pointwise.h"
#include "src/proto/config.pb.h"

namespace gbdt {

// RMSE loss = sqrt(sum_i(w[i] * (x[i] - f[i])^2) / sum_i(w[i]))
// https://en.wikipedia.org/wiki/Root-mean-square_deviation
class MSE : public Pointwise {
 public:
  MSE(const Config& unused_config) : Pointwise(ComputeMSE) {}
};

}  // namespace gbdt

#endif  // LOSS_FUNC_MSE_H_
