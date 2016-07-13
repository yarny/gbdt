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

#ifndef LOSS_FUNC_LOGLOSS_H_
#define LOSS_FUNC_LOGLOSS_H_

#include "loss_func_pointwise.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class DataStore;

// LogLoss = sum_i (w[i] * log(1+ exp(-y[i] * f[i]))) / sum_i(w[i])
// y is {-1, 1} valued.
class LogLoss : public Pointwise {
 public:
  LogLoss(const LossFuncConfig& config);

  Status Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* unused_group_column) override;
 private:


  LossFuncConfig config_;
};

}  // namespace gbdt

#endif  // LOSS_FUNC_LOGLOSS_H_
