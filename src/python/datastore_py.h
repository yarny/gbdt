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

#ifndef DATASTORE_PY_H_
#define DATASTORE_PY_H_

#include "gbdt_py_base.h"
#include "src/data_store/data_store.h"

using gbdt::DataStore;

class DataStorePy {
 public:
  DataStorePy() {
  }

  void LoadTSV(const vector<string>& tsvs,
               const vector<string>& binned_float_cols,
               const vector<string>& raw_float_cols,
               const vector<string>& string_cols);
  string Description() const {
    return (data_store_) ? data_store_->Description() : "Empty data store.";
  }
  void Clear();
  int num_rows() const {
    return (data_store_) ? data_store_->num_rows() : 0;
  }
  int num_cols() {
    return (data_store_) ? data_store_->num_cols() : 0;
  }
  int num_binned_float_cols() const {
    return (data_store_) ? data_store_->num_binned_float_cols() : 0;
  }
  int num_raw_float_cols() const {
    return (data_store_) ? data_store_->num_raw_float_cols() : 0;
  }
  int num_string_cols() const {
    return (data_store_) ? data_store_->num_string_cols() : 0;
  }

  DataStore* data_store() {
    return data_store_.get();
  }

 private:
  unique_ptr<DataStore> data_store_;
};

void InitDataStorePy(py::module &m);

#endif // DATASTORE_PY_H_
