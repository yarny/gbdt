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

#include "loss_func_huberized_hinge.h"

#include <cmath>
#include <glog/logging.h>
#include <numeric>
#include <string>
#include <vector>

#include "loss_func_math.h"
#include "src/base/base.h"
#include "src/data_store/column.h"
#include "src/data_store/data_store.h"
#include "src/proto/config.pb.h"
#include "src/utils/utils.h"
#include "target.h"

namespace gbdt {

HuberizedHinge::HuberizedHinge(const LossFuncConfig& config)
    : Pointwise(ComputeHuberizedHinge), config_(config) {
}

bool HuberizedHinge::ProvideY(DataStore* data_store, vector<float>* y) {
  return ComputeBinaryTargets(data_store, config_, y);
}

}  // namespace gbdt
