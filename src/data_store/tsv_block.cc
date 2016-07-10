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

#include "tsv_block.h"

#include <fstream>
#include <string>
#include <vector>

#include "external/cppformat/format.h"
#include "src/base/base.h"
#include "src/utils/utils.h"
#include "src/utils/stopwatch.h"

namespace gbdt {

unordered_set<string> TSVBlock::kValidNaNValues_ = {"NAN", "nan", "NaN", "?", "_", "-", "*"};

TSVBlock::TSVBlock(const string& tsv,
                   const vector<int>& float_column_indices,
                   const vector<int>& string_column_indices,
                   bool skip_header) {
  status_ = ReadTSV(tsv, float_column_indices, string_column_indices, skip_header);
}

Status TSVBlock::ReadTSV(const string& tsv,
                       const vector<int>& float_column_indices,
                       const vector<int>& string_column_indices,
                       bool skip_header) {
  if (!FileExists(tsv)) {
    return Status(error::NOT_FOUND,
                  fmt::format("TSV {0} does not exit.", tsv));
  }
  StopWatch stopwatch;
  stopwatch.Start();

  float_columns_.clear();
  float_columns_.resize(float_column_indices.size());
  string_columns_.clear();
  string_columns_.resize(string_column_indices.size());

  std::ifstream in(tsv);
  if (skip_header) {
    ReadLine(in);
  }

  int num_rows = 0;
  while (!in.eof()) {
    string line = ReadLine(in);
    if (line.empty()) continue;
    auto row = strings::split(line, "\t");
    ++num_rows;
    // Load String Columns.
    for (int i = 0; i < string_column_indices.size(); ++i) {
      int index = string_column_indices[i];
      if (index >= row.size()) {
        return Status(error::OUT_OF_RANGE,
                      fmt::format("{0} has only {1} columns while we are accessing column#{2} at row#{3}",
                                  line, row.size(), index, num_rows));
      }
      string_columns_[i].push_back(row[index]);
    }

    // Load Float Columns.
    for (int i = 0; i < float_column_indices.size(); ++i) {
      int index = float_column_indices[i];
      if (index >= row.size()) {
        return Status(error::OUT_OF_RANGE,
                      fmt::format("{0} has only {1} columns while we are accessing column#{2} at row#{3}",
                                  line, row.size(), index, num_rows));
      }
      float v;
      if (strings::StringCast(row[index], &v)) {
        float_columns_[i].push_back(v);
      } else {
        if (kValidNaNValues_.find(row[index]) == kValidNaNValues_.end()) {
          return Status(error::INVALID_ARGUMENT,
                        fmt::format("Invalid input at row {0} column {1}: {2}",
                                    num_rows,
                                    index + 1,
                                    row[index]));
        }
        float_columns_[i].push_back(NAN);
      }
    }
  }
  stopwatch.End();
  LOG(INFO) << "Loaded " << float_columns_.size() << " float columns"
            << " and " << string_columns_.size() << " string columns, each with "
            << num_rows << " rows from " << tsv
            << " in " << StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs());
  return Status::OK();
}

}  // namespace gbdt
