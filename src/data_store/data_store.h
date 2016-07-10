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

#ifndef DATA_STORE_H_
#define DATA_STORE_H_

#include <memory>
#include <unordered_map>

#include "column.h"
#include "src/base/base.h"

namespace gbdt {

// A data store is simply a collection of columns.
class DataStore {
public:
  virtual ~DataStore();

  const RawFloatColumn* GetRawFloatColumn(const string& column_name);
  const BinnedFloatColumn* GetBinnedFloatColumn(const string& column_name);
  const StringColumn* GetStringColumn(const string& column_name);
  virtual const Column* GetColumn(const string& column_name);
  uint num_rows() const;
  uint num_cols() const;
  uint num_binned_float_cols() const;
  uint num_raw_float_cols() const;
  uint num_string_cols() const;
  string Description() const;
  const Status& status() const {
    return status_;
  }

protected:
  unordered_map<string, unique_ptr<Column>> column_map_;
  Status status_;
};

}  // namespace gbdt

#endif  // DATA_STORE_H_
