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

#include "external/cppformat/format.h"
#include "gbdt_py_base.h"
#include "src/data_store/column.h"
#include "src/data_store/data_store.h"
#include "src/data_store/tsv_data_store.h"
#include "src/proto/config.pb.h"

using gbdt::DataStorePy;

namespace gbdt {

DataStorePy::DataStorePy() {}


void DataStorePy::LoadTSV(const vector<string>& tsvs,
                          const vector<string>& bucketized_float_cols,
                          const vector<string>& raw_float_cols,
                          const vector<string>& string_cols) {
  Config config;
  for (const auto& col : bucketized_float_cols) {
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

StringColumnPy DataStorePy::GetStringColumn(const string& col) const {
  const auto* column = data_store_ ? data_store_->GetStringColumn(col) : nullptr;
  if (!column) ThrowException(Status(error::NOT_FOUND,
                                      fmt::format("Failed to find {0} from data store", col)));
  return StringColumnPy(column);
}

RawFloatColumnPy DataStorePy::GetRawFloatColumn(const string& col) const {
  const auto* column = data_store_ ? data_store_->GetRawFloatColumn(col) : nullptr;
  if (!column) ThrowException(Status(error::NOT_FOUND,
                                      fmt::format("Failed to find {0} from data store", col)));
  return RawFloatColumnPy(column);
}

BucketizedFloatColumnPy DataStorePy::GetBucketizedFloatColumn(const string& col) const {
  const auto* column = data_store_ ? data_store_->GetBucketizedFloatColumn(col) : nullptr;
  if (!column) ThrowException(Status(error::NOT_FOUND,
                                      fmt::format("Failed to find {0} from data store", col)));
  return BucketizedFloatColumnPy(column);
}

bool DataStorePy::Exists(const string& col) const {
  return data_store_ && data_store_->GetColumn(col);
}

void DataStorePy::AddStringColumn(const string& column_name,
                                  const vector<string>& raw_strings) {
  if (!data_store_) data_store_.reset(new DataStore);
  auto status = data_store_->Add(Column::CreateStringColumn(column_name, raw_strings));
  if (!status.ok()) {
    ThrowException(status);
  }
}

void DataStorePy::AddBucketizedFloatColumn(const string& column_name,
                                           const vector<float>& raw_floats) {
  if (!data_store_) data_store_.reset(new DataStore);
  auto status = data_store_->Add(Column::CreateBucketizedFloatColumn(column_name, raw_floats));
  if (!status.ok()) {
    ThrowException(status);
  }
}

void DataStorePy::AddRawFloatColumn(const string& column_name,
                                    const vector<float>& raw_floats) {
  if (!data_store_) data_store_.reset(new DataStore);
  auto status = data_store_->Add(Column::CreateRawFloatColumn(column_name, vector<float>(raw_floats)));
  if (!status.ok()) {
    ThrowException(status);
  }
}

void DataStorePy::RemoveColumnIfExists(const string& column_name) {
  if (data_store_) {
    data_store_->RemoveColumnIfExists(column_name);
  }
}

vector<string> DataStorePy::BucketizedFloatColumnNames() const {
  if (!data_store_) return vector<string>();
  vector<string> column_names;
  for (const auto* col: data_store_->GetBucketizedFloatColumns()) {
    column_names.push_back(col->name());
  }
  return column_names;
}

vector<string> DataStorePy::RawFloatColumnNames() const {
  if (!data_store_) return vector<string>();
  vector<string> column_names;
  for (const auto* col: data_store_->GetRawFloatColumns()) {
    column_names.push_back(col->name());
  }
  return column_names;
}

vector<string> DataStorePy::StringColumnNames() const {
  if (!data_store_) return vector<string>();
  vector<string> column_names;
  for (const auto* col: data_store_->GetStringColumns()) {
    column_names.push_back(col->name());
  }
  return column_names;
}

vector<string> DataStorePy::AllColumnNames() const {
  vector<string> column_names;
  for (auto col : BucketizedFloatColumnNames()) {
    column_names.emplace_back(col);
  }
  for (auto col : StringColumnNames()) {
    column_names.emplace_back(col);
  }
  for (auto col : RawFloatColumnNames()) {
    column_names.emplace_back(col);
  }
  return column_names;
}

void DataStorePy::Clear() {
  data_store_.reset(nullptr);
}

string DataStorePy::GetColumnType(const string& col) const {
  if (!data_store_) {
    ThrowException(Status(error::NOT_FOUND, "Empty data store"));
  }

  const auto* column = data_store_->GetColumn(col);
  if (!column) {
    ThrowException(Status(error::NOT_FOUND,
                          fmt::format("Failed to find column '{0}' data store", col)));
  }

  switch(column->type()) {
    case Column::kStringColumn:
      return "string";
    case Column::kBucketizedFloatColumn:
      return "bucketized_float";
    case Column::kRawFloatColumn:
      return "raw_float";
    default:
      return "unknown";
  }
}

}  // namespace gbdt

void InitDataStorePy(py::module &m) {
  py::class_<DataStorePy>(m, "DataStore")
      .def(py::init<>())
      .def("load_tsv",
           &DataStorePy::LoadTSV,
           "Loads tsv into data_store. \nThe data store accepts both float and string columns."
           "The gbdt package bucketizes float features to reduce time complexity and memory footprint."
           "Please load all float features as bucketized_float_cols and other float cols like target "
           "or weights as raw_float_cols.",
           py::arg("tsvs"),
           py::arg("bucketized_float_cols")=vector<string>(),
           py::arg("raw_float_cols")=vector<string>(),
           py::arg("string_cols")=vector<string>())
      .def("get_bucketized_float_col", &DataStorePy::GetBucketizedFloatColumn)
      .def("get_raw_float_col", &DataStorePy::GetRawFloatColumn)
      .def("get_string_col", &DataStorePy::GetStringColumn)
      .def("add_bucketized_float_col", &DataStorePy::AddBucketizedFloatColumn)
      .def("add_raw_float_col", &DataStorePy::AddRawFloatColumn)
      .def("add_string_col", &DataStorePy::AddStringColumn)
      .def("erase", &DataStorePy::RemoveColumnIfExists)
      .def("clear", &DataStorePy::Clear)
      .def("cols", &DataStorePy::AllColumnNames)
      .def("bucketized_float_cols", &DataStorePy::BucketizedFloatColumnNames)
      .def("string_cols", &DataStorePy::StringColumnNames)
      .def("raw_float_cols", &DataStorePy::RawFloatColumnNames)
      .def("get_column_type", &DataStorePy::GetColumnType)
      .def("__len__", &DataStorePy::num_rows)
      .def("__contains__", &DataStorePy::Exists)
      .def("__repr__", &DataStorePy::Description)
      .def("__str__", &DataStorePy::Description);
}
