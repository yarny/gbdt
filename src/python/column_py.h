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

#ifndef COLUMN_PY_H_
#define COLUMN_PY_H_

#include "gbdt_py_base.h"
#include "src/data_store/column.h"

#include <utility>

namespace gbdt {

class StringColumnPy {
 public:
  StringColumnPy(const StringColumn* column) : column_(column) {}
  int size() const;
  const string name() const;
  const string& get(int i) const;
 private:
  const StringColumn* column_ = nullptr;
};

class RawFloatColumnPy {
 public:
  RawFloatColumnPy(const RawFloatColumn* column) : column_(column) {}
  int size() const;
  const string name() const;
  float get(int i) const;
 private:
  const RawFloatColumn* column_ = nullptr;
};

class BucketizedFloatColumnPy {
 public:
  BucketizedFloatColumnPy(const BucketizedFloatColumn* column) : column_(column) {}
  int size() const;
  const string name() const;
  pair<float, float> get(int i) const;

  vector<pair<float, float> > GetBuckets() const;
 private:
  const BucketizedFloatColumn* column_ = nullptr;
};

}  // namespace gbdt

void InitStringColumnPy(py::module &m);
void InitRawFloatColumnPy(py::module &m);
void InitBucketizedFloatColumnPy(py::module &m);

#endif // COLUMN_PY_H_
