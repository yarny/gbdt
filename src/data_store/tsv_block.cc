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

#include "src/base/base.h"
#include "src/utils/utils.h"
#include "src/utils/stopwatch.h"

namespace gbdt {

TSVBlock::TSVBlock(const string& tsv,
                   const vector<int>& float_column_indices,
                   const vector<int>& string_column_indices,
                   bool skip_header) {
  ReadTSV(tsv, float_column_indices, string_column_indices, skip_header);
}

void TSVBlock::ReadTSV(const string& tsv,
                       const vector<int>& float_column_indices,
                       const vector<int>& string_column_indices,
                       bool skip_header) {
  CHECK(FileExists(tsv)) << "TSV " << tsv << " does not exist.";
  StopWatch stopwatch;
  stopwatch.Start();

  float_columns_.clear();
  float_columns_.resize(float_column_indices.size());
  string_columns_.clear();
  string_columns_.resize(string_column_indices.size());

  std::ifstream in(tsv);
  if (skip_header) {
    string line;
    std::getline(in, line);
  }

  int num_rows = 0;
  while (!in.eof()) {
    string line;
    std::getline(in, line);
    if (line.empty()) continue;
    auto row = strings::split(line, "\t");
    ++num_rows;
    // Load String Columns.
    for (int i = 0; i < string_column_indices.size(); ++i) {
      int index = string_column_indices[i];
      CHECK_LT(index, row.size())
          << line << " has only " << row.size() << " columns "
          << " while we are accessing columm #" << index;
      string_columns_[i].push_back(row[index]);
    }

    // Load Float Columns.
    for (int i = 0; i < float_column_indices.size(); ++i) {
      int index = float_column_indices[i];
      CHECK_LT(index, row.size())
          << line << " has only " << row.size() << " columns "
          << " while we are accessing columm #" << index;
      float v;
      if (strings::StringCast(row[index], &v)) {
        float_columns_[i].push_back(v);
      } else {
        CHECK(row[index] == "nan" ||
              row[index] == "NAN" ||
              row[index] == "_" ||
              row[index] == "?" ||
              row[index] == "-" ||
              row[index] == "*")
            << "Invalid input at row " << num_rows << " column " << index + 1 << ": " << row[index];
        float_columns_[i].push_back(NAN);
      }
    }
  }
  stopwatch.End();
  LOG(INFO) << "Loaded " << float_columns_.size() << " float columns"
            << " and " << string_columns_.size() << " string columns, each with "
            << num_rows << " rows from " << tsv
            << " in " << StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs());
}

}  // namespace gbdt
