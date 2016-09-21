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

#include <string>
#include <vector>

#include "column_py.h"
#include "gbdt_py_base.h"
#include "src/data_store/data_store.h"

namespace gbdt {

class DataStorePy {
 public:
  DataStorePy();

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

  BucketizedFloatColumnPy GetBucketizedFloatColumn(const string& col) const;
  RawFloatColumnPy GetRawFloatColumn(const string& col) const;
  StringColumnPy GetStringColumn(const string& col) const;
  vector<BucketizedFloatColumnPy> GetBucketizedFloatColumns() const;
  vector<RawFloatColumnPy> GetRawFloatColumns() const;
  vector<StringColumnPy> GetStringColumns() const;
  void RemoveColumnIfExists(const string& col);
  bool ExistsBucketizedFloatColumn(const string& col) const;
  bool ExistsRawFloatColumn(const string& col) const;
  bool ExistsStringColumn(const string& col) const;

  DataStore* data_store() {
    return data_store_.get();
  }

  void LoadTSV(const vector<string>& tsvs,
               const vector<string>& bucketized_float_cols,
               const vector<string>& raw_float_cols,
               const vector<string>& string_cols);

  void AddStringColumn(const string& column_name,
                       const vector<string>& raw_strings);
  void AddBucketizedFloatColumn(const string& column_name,
                                const vector<float>& raw_floats);
  void AddRawFloatColumn(const string& column_name,
                         const vector<float>& raw_floats);

 private:
  unique_ptr<DataStore> data_store_;
};

}  // namespace gbdt

void InitDataStorePy(py::module &m);

#endif // DATASTORE_PY_H_
