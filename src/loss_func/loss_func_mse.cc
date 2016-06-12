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

#include "loss_func_mse.h"

#include <cmath>
#include <glog/logging.h>
#include <numeric>
#include <string>
#include <vector>

#include "loss_func_math.h"
#include "src/base/base.h"
#include "src/data_store/column.h"
#include "src/data_store/data_store.h"
#include "src/utils/utils.h"

namespace gbdt {

MSE::MSE(const LossFuncConfig& config) : Pointwise(ComputeMSE), config_(config) {
}

bool MSE::ProvideY(DataStore* data_store, vector<float>* y) {
  const string& target_column_name = config_.target_column();
  if (target_column_name.empty()) {
    LOG(ERROR) << "Please specify target_column for MSE loss.";
    return false;
  }

  auto targets = data_store->GetRawFloatColumn(target_column_name);
  if (!targets) {
    LOG(ERROR) << "Failed to get target column " << target_column_name;
    return false;
  }

  // Copy targets to y.
  y->resize(targets->size());
  for (int i = 0; i < targets->size(); ++i) {
    (*y)[i] = (*targets)[i];
  }

  return true;
}

}  // namespace gbdt
