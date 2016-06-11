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

#ifndef TSV_BLOCK_H_
#define TSV_BLOCK_H_

#include <string>
#include <vector>

#include "src/base/base.h"

namespace gbdt {

// BlockReader reads specified float columns and string columns from a block of TSV>
class TSVBlock {
public:
  TSVBlock(const string& tsv,
           const vector<int>& float_column_indices,
           const vector<int>& string_column_indices,
           bool skip_header);

  const vector<vector<float>>& float_columns() const {
    return float_columns_;
  }
  const vector<vector<string>>& string_columns() const {
    return string_columns_;
  }

private:
  void ReadTSV(const string& tsv,
               const vector<int>& float_column_indices,
               const vector<int>& string_column_indices,
               bool skip_header);

  vector<vector<float>> float_columns_;
  vector<vector<string>> string_columns_;
};

}  // namespace gbdt

#endif  // TSV_BLOCK_H_
