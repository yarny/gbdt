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
#include "src/utils/utils.h"

using gbdt::BucketizedFloatColumnPy;
using gbdt::RawFloatColumnPy;
using gbdt::StringColumnPy;

namespace gbdt {

namespace {
const int NUM_ELEMENTS_TO_DISPLAY = 10;
}  // namespace

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

const string StringColumnPy::Description() const {
  int n = min(size(), NUM_ELEMENTS_TO_DISPLAY);
  vector<string> parts(n);
  for (int i = 0; i < n; ++i) {
    parts[i] = fmt::format("\"{0}\"", get(i));
  }
  return fmt::format("StringColumn([{0}{1}])", strings::JoinStrings(parts, ","),
                     size() > n ? " ..." : "");
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

const string RawFloatColumnPy::Description() const {
  int n = min(size(), NUM_ELEMENTS_TO_DISPLAY);
  vector<string> parts(n);
  for (int i = 0; i < n; ++i) {
    parts[i] = fmt::format("{0}", get(i));
  }
  return fmt::format("FloatColumn([{0}{1}])",
                     strings::JoinStrings(parts, ","),
                     size() > n ? " ...": "");
}

int BucketizedFloatColumnPy::size() const {
  return column_ ? column_->size() : 0;
}

const string BucketizedFloatColumnPy::name() const {
  return column_ ? column_->name() : "empty column.";
}

float BucketizedFloatColumnPy::get(int i) const {
  if (!column_) ThrowException(Status(error::NOT_FOUND, "The column is null."));
  if (i >= column_->size()) ThrowException(Status(error::OUT_OF_RANGE, "Index out of range."));

  return column_->get_row_min(i);
}

vector<pair<float, float>> BucketizedFloatColumnPy::GetBuckets() const {
  if (!column_) ThrowException(Status(error::NOT_FOUND, "The column is null."));
  vector<pair<float, float>> buckets;
  for (int i = 0; i < column_->max_int(); ++i) {
    buckets.emplace_back(column_->get_bucket_min(i), column_->get_bucket_max(i));
  }
  return buckets;
}

const string BucketizedFloatColumnPy::Description() const {
  int n = min(size(), NUM_ELEMENTS_TO_DISPLAY);
  vector<string> parts(n);
  for (int i = 0; i < n; ++i) {
    parts[i] = fmt::format("{0}", get(i));
  }
  return fmt::format("BucketizedFloatColumn([{0}{1}])",
                     strings::JoinStrings(parts, ","),
                     size() > n ? " ..." : "");
}

}  // namespace gbdt

void InitStringColumnPy(py::module &m) {
  py::class_<StringColumnPy>(m, "StringColumn")
      .def("__len__", &StringColumnPy::size)
      .def("__getitem__", &StringColumnPy::get)
      .def("__str__", &StringColumnPy::name)
      .def("__repr__", &StringColumnPy::Description);
}

void InitRawFloatColumnPy(py::module &m) {
  py::class_<RawFloatColumnPy>(m, "RawFloatColumn")
      .def("__len__", &RawFloatColumnPy::size)
      .def("__getitem__", &RawFloatColumnPy::get)
      .def("__str__", &RawFloatColumnPy::name)
      .def("__repr__", &RawFloatColumnPy::Description);
}

void InitBucketizedFloatColumnPy(py::module &m) {
  py::class_<BucketizedFloatColumnPy>(m, "BucketizedFloatColumn")
      .def("__len__", &BucketizedFloatColumnPy::size)
      .def("__getitem__", &BucketizedFloatColumnPy::get)
      .def("__str__", &BucketizedFloatColumnPy::name)
      .def("buckets", &BucketizedFloatColumnPy::GetBuckets)
      .def("__repr__", &BucketizedFloatColumnPy::Description);;
}
