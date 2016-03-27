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

#include "flatfiles_data_store.h"

#include <fstream>

#include "column.h"
#include "src/utils/utils.h"

namespace gbdt {

FlatfilesDataStore::FlatfilesDataStore(const string& flatfiles_dir)
    : flatfiles_dirs_(vector<string>({flatfiles_dir})) {
}

FlatfilesDataStore::FlatfilesDataStore(const vector<string>& flatfiles_dirs)
    : flatfiles_dirs_(flatfiles_dirs) {
}

string FlatfilesDataStore::FindFlatfile(const string& column_name) const {
   for (string flatfiles_dir : flatfiles_dirs_) {
    string flatfile = flatfiles_dir + "/" + column_name;
    if (FileExists(flatfile)) {
      return flatfile;
    }
  }
   return "";
}

bool FlatfilesDataStore::LoadColumn(const string& column_name) {
  string flatfile = FindFlatfile(column_name);
  if (flatfile.empty())  {
    LOG(ERROR) << "Failed to find " << column_name << " in "
               << strings::JoinStrings(flatfiles_dirs_, ",");
    return false;
  }

  std::ifstream in(flatfile);
  // Read the first line to get the type.
  string column_type;
  std::getline(in, column_type);
  // Reset the stream.
  in.seekg(0);

  unique_ptr<Column> column;
  if (column_type == "# dtype=strings") {
    // Read as strings.
    column = LoadStringColumn(in, column_name);
  } else if (column_type == "# dtype=raw_floats") {
    // Read as raw floats.
    column = LoadFloatColumn(in, column_name, false);
  } else if (column_type == "# dtype=binned_floats") {
    column = LoadFloatColumn(in, column_name, true);
  } else {
    LOG(ERROR) << "Unknown flatfile type: " << column_type ;
    return false;
  }

  if (column == nullptr) {
    LOG(ERROR) << "Failed to create the column.";
    return false;
  }

  lock_guard<mutex> lock(mutex_);
  if (num_rows() > 0 && num_rows() != column->size()) {
    LOG(ERROR) << "Row size consistency check failed for " << column_name
               << "(old " << num_rows() << " vs. " << "new " << column->size();
    return false;
  }
  column_map_[column_name] = std::move(column);

  return true;
}

unique_ptr<Column> FlatfilesDataStore::LoadStringColumn(ifstream& in, const string& column_name) {
  vector<string> raw_strings;
  while (!in.eof()) {
    string line;
    std::getline(in, line);
    if (strings::HasPrefix(line, "#") || (line.empty() && !in.good())) {
      continue;
    }
    raw_strings.push_back(line);
  }

  return Column::CreateStringColumn(column_name, raw_strings);
}

unique_ptr<Column> FlatfilesDataStore::LoadFloatColumn(ifstream& in,
                                                       const string& column_name,
                                                       bool binned) {
  vector<float> raw_floats;
  while (!in.eof()) {
    string line;
    std::getline(in, line);
    if (strings::HasPrefix(line, "#") || (line.empty() && !in.good())) {
      continue;
    }
    float v;
    if (strings::StringCast(line, &v)) {
      raw_floats.push_back(v);
    } else {
      raw_floats.push_back(NAN);
    }
  }

  return binned ? Column::CreateBinnedFloatColumn(column_name, raw_floats) :
      Column::CreateRawFloatColumn(column_name, std::move(raw_floats));
}

const Column* FlatfilesDataStore::GetColumn(const string& column_name) {
  bool column_exists = false;
  {
    lock_guard<mutex> lock(mutex_);
    column_exists = column_map_.find(column_name) != column_map_.end();
  }

  if (!column_exists) {
    if (!LoadColumn(column_name)) {
      LOG(ERROR) << "Failed to load " << column_name << " from "
                 << strings::JoinStrings(flatfiles_dirs_, ",");
      return nullptr;
    }
  }

  {
    lock_guard<mutex> lock(mutex_);
    return column_map_[column_name].get();
  }
}

}  // namespace gbdt
