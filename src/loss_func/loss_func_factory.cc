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

#include "loss_func_factory.h"

#include <unordered_map>
#include <utility>

#include "loss_func_auc.h"
#include "loss_func_gbrank.h"
#include "loss_func_huberized_hinge.h"
#include "loss_func_lambdamart.h"
#include "loss_func_logloss.h"
#include "loss_func_mse.h"
#include "loss_func_pairwise_logloss.h"
#include "src/proto/config.pb.h"

namespace gbdt {

unordered_map<string, LossFuncFactory::Creator> LossFuncFactory::loss_func_creator_map_ = {
  {"mse", [](const LossFuncConfig& config) { return new MSE(config);}},
  {"logloss", [](const LossFuncConfig& config) { return new LogLoss(config);}},
  {"huberized_hinge", [](const LossFuncConfig& config) { return new HuberizedHinge(config);}},
  {"auc", [](const LossFuncConfig& config) { return new AUC(config);}},
  {"pairwise_logloss", [](const LossFuncConfig& config) { return new PairwiseLogLoss(config);}},
  {"gbrank", [](const LossFuncConfig& config) { return new GBRank(config);}},
  {"lambdamart", [](const LossFuncConfig& config) { return new LambdaMART(config);}}
};

// TODO(criver): find better registration mechanism implementation.
unique_ptr<LossFunc> LossFuncFactory::CreateLossFunc(const LossFuncConfig& config) {
  auto it = loss_func_creator_map_.find(config.loss_func());
  if (it != loss_func_creator_map_.end()) {
    return unique_ptr<LossFunc>(it->second(config));
  }

  return nullptr;
}

}  // namespace gbdt
