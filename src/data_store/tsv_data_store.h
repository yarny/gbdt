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

#ifndef TSV_DATA_STORE_H_
#define TSV_DATA_STORE_H_

#include <string>
#include <utility>
#include <vector>

#include "column.h"
#include "data_store.h"
#include "src/base/base.h"
#include "src/utils/status.h"

namespace gbdt {

class TSVBlock;
class TSVDataConfig;
class DataConfig;

class TSVDataStore : public DataStore {
 public:
  // TSVs can be divided into blocks. The first tsv contains the header file.
  // data_config contains information on how to load the columns. The column
  // can be loaded as binned_floats, raw_float, or strings.
  TSVDataStore(const vector<string>& tsvs, const DataConfig& data_config);
  virtual ~TSVDataStore() {}

 protected:
  Status ProcessBlock(const TSVBlock* block);
  Status Finalize();
  Status SetupColumns(const string& first_tsv, const DataConfig& data_config);
  Status LoadTSVs(const vector<string>& tsvs, const DataConfig& data_config);

  vector<pair<BinnedFloatColumn*, int>> binned_float_columns_;
  vector<pair<RawFloatColumn*, int>> raw_float_columns_;
  vector<pair<StringColumn*, int>> string_columns_;
  vector<int> float_column_indices_;
  vector<int> string_column_indices_;
};

}  // namespace gbdt

#endif  // TSV_DATA_STORE_H_
