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

#include "column_py.h"

#include <string>

#include "external/cppformat/format.h"

using gbdt::BinnedFloatColumnPy;
using gbdt::RawFloatColumnPy;
using gbdt::StringColumnPy;

namespace gbdt {

int StringColumnPy::size() const {
  return column_ ? column_->size() : 0;
}

const string StringColumnPy::name() const {
  return column_ ? column_->name() : "empty column.";
}

const string& StringColumnPy::get(int i) const {
  if (!column_) ThrowException(Status(error::NOT_FOUND, "The column is null."));
  if (i >= column_->size()) ThrowException(Status(error::OUT_OF_RANGE, "Index out of range."));

  return column_->get_row_string(i);
}

int RawFloatColumnPy::size() const {
  return column_ ? column_->size() : 0;
}

const string RawFloatColumnPy::name() const {
  return column_ ? column_->name() : "empty column.";
}

float RawFloatColumnPy::get(int i) const {
  if (!column_) ThrowException(Status(error::NOT_FOUND, "The column is null."));
  if (i >= column_->size()) ThrowException(Status(error::OUT_OF_RANGE, "Index out of range."));

  return (*column_)[i];
}

int BinnedFloatColumnPy::size() const {
  return column_ ? column_->size() : 0;
}

const string BinnedFloatColumnPy::name() const {
  return column_ ? column_->name() : "empty column.";
}

pair<float, float> BinnedFloatColumnPy::get(int i) const {
  if (!column_) ThrowException(Status(error::NOT_FOUND, "The column is null."));
  if (i >= column_->size()) ThrowException(Status(error::OUT_OF_RANGE, "Index out of range."));

  return make_pair(column_->get_row_min(i), column_->get_row_max(i));
}

vector<pair<float, float>> BinnedFloatColumnPy::GetBins() const {
  if (!column_) ThrowException(Status(error::NOT_FOUND, "The column is null."));
  vector<pair<float, float>> bins;
  for (int i = 0; i < column_->max_int(); ++i) {
    bins.emplace_back(column_->get_bin_min(i), column_->get_bin_max(i));
  }
  return bins;
}

}  // namespace gbdt

void InitStringColumnPy(py::module &m) {
  py::class_<StringColumnPy>(m, "StringColumn")
      .def("__len__", &StringColumnPy::size)
      .def("__getitem__", &StringColumnPy::get)
      .def("__str__", &StringColumnPy::name);
}

void InitRawFloatColumnPy(py::module &m) {
  py::class_<RawFloatColumnPy>(m, "RawFloatColumn")
      .def("__len__", &RawFloatColumnPy::size)
      .def("__getitem__", &RawFloatColumnPy::get)
      .def("__str__", &RawFloatColumnPy::name);
}

void InitBinnedFloatColumnPy(py::module &m) {
  py::class_<BinnedFloatColumnPy>(m, "BinnedFloatColumn")
      .def("__len__", &BinnedFloatColumnPy::size)
      .def("__getitem__", &BinnedFloatColumnPy::get)
      .def("__str__", &BinnedFloatColumnPy::name)
      .def("bins", &BinnedFloatColumnPy::GetBins);
}
