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

#include "loss_func_logloss.h"

#include <cmath>
#include <glog/logging.h>
#include <numeric>
#include <string>
#include <vector>

#include "loss_func_math.h"
#include "src/base/base.h"
#include "src/proto/config.pb.h"
#include "src/utils/utils.h"

namespace gbdt {

LogLoss::LogLoss(const LossFuncConfig& config)
    : Pointwise(ComputeLogLoss), config_(config) {
}

Status LogLoss::Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* group_column) {
  for (int i = 0; i < num_rows; ++i) {
    if (y(i) != 1.0 and y(i) != -1.0) {
      return Status(error::INVALID_ARGUMENT, "Binary targets should only take values +1 and -1.");
    }
  }
  Pointwise::Init(num_rows, w, y, group_column);
  return Status::OK;
}

}  // namespace gbdt
