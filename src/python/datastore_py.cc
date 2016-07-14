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

#include "datastore_py.h"

#include <memory>
#include <string>
#include <vector>

#include "gbdt_py_base.h"
#include "src/data_store/data_store.h"
#include "src/data_store/tsv_data_store.h"
#include "src/proto/config.pb.h"

using gbdt::DataStorePy;

namespace gbdt {

void DataStorePy::LoadTSV(const vector<string>& tsvs,
                          const vector<string>& binned_float_cols,
                          const vector<string>& raw_float_cols,
                          const vector<string>& string_cols) {
  Config config;
  for (const auto& col : binned_float_cols) {
    config.add_float_feature(col);
  }
  for (const auto& col : raw_float_cols) {
    config.add_additional_float_column(col);
  }
  for (const auto& col : string_cols) {
    config.add_additional_string_column(col);
  }
  unique_ptr<DataStore> data_store(new gbdt::TSVDataStore(tsvs, config));
  if (!data_store->status().ok()) {
    ThrowException(data_store->status());
    return;
  }
  data_store_ = std::move(data_store);
}

void DataStorePy::Clear() {
  data_store_.reset(nullptr);
}

}  // namespace gbdt

void InitDataStorePy(py::module &m) {
  py::class_<DataStorePy>(m, "DataStore")
      .def(py::init<>())
      .def("load_tsv", &DataStorePy::LoadTSV,
           py::arg("tsvs"),
           py::arg("binned_float_cols") = vector<string>(),
           py::arg("raw_float_cols") = vector<string>(),
           py::arg("string_cols") = vector<string>())
      .def("num_cols", &DataStorePy::num_cols)
      .def("num_rows", &DataStorePy::num_rows)
      .def("num_binned_float_cols", &DataStorePy::num_binned_float_cols)
      .def("num_raw_float_cols", &DataStorePy::num_raw_float_cols)
      .def("num_string_cols", &DataStorePy::num_string_cols)
      .def("clear", &DataStorePy::Clear)
      .def("__str__", &DataStorePy::Description);
}
