/*
 * Copyright 2016 Jiang Chen <criver@gmail.com>
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

#include "train_gbdt_py.h"

#include <vector>
#include <map>
#include <memory>

#include "external/cppformat/format.h"
#include "datastore_py.h"
#include "forest_py.h"
#include "src/gbdt_algo/gbdt_algo.h"
#include "src/gbdt_algo/utils.h"
#include "src/loss_func/loss_func.h"
#include "src/loss_func/loss_func_factory.h"
#include "src/proto/config.pb.h"
#include "src/utils/json_utils.h"
#include "src/utils/utils.h"

using gbdt::ForestPy;

DECLARE_int32(seed);
DECLARE_int32(num_threads);

namespace gbdt {

ForestPy TrainPy(DataStorePy* data_store,
                 const vector<float>& y,
                 const vector<float>& w,
                 const string& json_config,
                 ForestPy* base_forest_py,
                 int random_seed,
                 int num_threads) {
  FLAGS_seed = random_seed;
  FLAGS_num_threads = num_threads;

  Config config;
  auto status = JsonUtils::FromJson(json_config, &config);
  if (!status.ok()) ThrowException(status);
  status = CheckConfig(config);
  if (!status.ok()) ThrowException(status);

  FloatVector w_hat = [](int){ return 1.0f; };
  if (!w.empty()) {
    w_hat = ([&w=w](int i) { return w[i]; });
  }
  auto y_hat = [&y=y](int i) { return y[i]; };

  // Initialize Loss Function.
  unique_ptr<LossFunc> loss_func = LossFuncFactory::CreateLossFunc(config);
  if (!loss_func) ThrowException(Status(
          error::NOT_FOUND,
          fmt::format("Unknown loss function {0}. Supported loss functions: {1}",
                      config.loss_func(),
                      strings::JoinStrings(LossFuncFactory::LossFuncs(), ","))));

  const Forest* base_forest = base_forest_py ? &base_forest_py->forest() : nullptr;

  unordered_set<string> features(config.categorical_feature().begin(),
                                 config.categorical_feature().end());
  features.insert(config.float_feature().begin(),
                  config.float_feature().end());

  Forest forest;
  status = TrainGBDT(data_store->data_store(),
                     features,
                     w_hat,
                     y_hat,
                     loss_func.get(),
                     config,
                     base_forest,
                     &forest);
  if (!status.ok()) ThrowException(status);

  return ForestPy(std::move(forest));
}

}  // namespace gbdt

void InitTrainGBDTPy(py::module &m) {
  m.def("train",
        &gbdt::TrainPy,
        py::arg("data_store"),
        py::arg("y"),
        py::arg("w")=vector<float>(),
        py::arg("config"),
        py::arg("base_forest")=(ForestPy*) nullptr,
        py::arg("random_seed")=1232212,
        py::arg("num_threads")=16);
}
