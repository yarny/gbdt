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

#ifndef FOREST_PY_H_
#define FOREST_PY_H_

#include <vector>

#include "gbdt_py_base.h"
#include "src/proto/tree.pb.h"

namespace gbdt {

class DataStorePy;

class ForestPy {
 public:
  ForestPy(const string& str);
  ForestPy(Forest&& forest) : forest_(forest) {}

  string ToJson() const;
  vector<double> Predict(DataStorePy* data_store_py) const;
  void PredictAndOutput(DataStorePy* data_store_py,
                        const list<int>& test_points,
                        const string& output_dir) const;
  vector<pair<string, double>> FeatureImportance() const;
  const Forest& forest() const { return forest_; }

 private:
  Forest forest_;
};

}  // namespace gbdt

void InitForestPy(py::module &m);

#endif // FOREST_PY_H_
