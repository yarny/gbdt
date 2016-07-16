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

#include "data_store.h"

#include <glog/logging.h>

#include "column.h"
#include "external/cppformat/format.h"

namespace gbdt {

DataStore::~DataStore() {
}

const RawFloatColumn* DataStore::GetRawFloatColumn(const string& column_name) {
  const auto* column = GetColumn(column_name);
  if (!column) {
    LOG(ERROR) << "Failed to load column " << column_name << " from data store.";
    return nullptr;
  }
  if (column->type() != Column::kRawFloatColumn) {
    LOG(ERROR) << column_name << " is NOT a RawFloatColumn.";
    return nullptr;
  }
  return static_cast<const RawFloatColumn*>(column);
}

const BinnedFloatColumn* DataStore::GetBinnedFloatColumn(const string& column_name) {
  const auto* column = GetColumn(column_name);
  if (!column) {
    LOG(ERROR) << "Failed to load column " << column_name << " from data store.";
    return nullptr;
  }
  if (column->type() != Column::kBinnedFloatColumn) {
    LOG(ERROR) << column_name << " is NOT a BinnedFloatColumn.";
    return nullptr;
  }
  return static_cast<const BinnedFloatColumn*>(column);
}

const StringColumn* DataStore::GetStringColumn(const string& column_name) {
  const auto* column = GetColumn(column_name);
  if (!column) {
    LOG(ERROR) << "Failed to load column " << column_name << " from data store.";
    return nullptr;
  }
  if (column->type() != Column::kStringColumn) {
    LOG(ERROR) << column_name << " is NOT a StringColumn.";
    return nullptr;
  }
  return static_cast<const StringColumn*>(column);
}

uint DataStore::num_rows() const {
  return column_map_.empty() ? 0 : column_map_.begin()->second->size();
}

uint DataStore::num_cols() const {
  return column_map_.size();
}

uint CountColumn(const unordered_map<string, unique_ptr<Column>>& column_map,
                 Column::ColumnType type) {
  int count = 0;
  for (const auto& p : column_map) {
    if (p.second->type() == type) {
      ++count;
    }
  }

  return count;
}
vector<const BinnedFloatColumn*> DataStore::GetBinnedFloatColumns() const {
  vector<const BinnedFloatColumn*> columns;
  auto type = Column::kBinnedFloatColumn;
  for (const auto& p : column_map_) {
      if (p.second->type() == type) {
        columns.emplace_back(static_cast<const BinnedFloatColumn*>(p.second.get()));
    }
  }
  return columns;
}

vector<const RawFloatColumn*> DataStore::GetRawFloatColumns() const {
  vector<const RawFloatColumn*> columns;
  auto type = Column::kRawFloatColumn;
  for (const auto& p : column_map_) {
      if (p.second->type() == type) {
        columns.emplace_back(static_cast<const RawFloatColumn*>(p.second.get()));
    }
  }
  return columns;
}

vector<const StringColumn*> DataStore::GetStringColumns() const {
  vector<const StringColumn*> columns;
  auto type = Column::kStringColumn;
  for (const auto& p : column_map_) {
      if (p.second->type() == type) {
        columns.emplace_back(static_cast<const StringColumn*>(p.second.get()));
    }
  }
  return columns;
}

const Column* DataStore::GetColumn(const string& column_name) {
  auto it = column_map_.find(column_name);
  return it == column_map_.end() ? nullptr : it->second.get();
}

string DataStore::Description() const {
  return fmt::format("DataStore with {0} binned float, {1} raw float and "
                     "{2} string columns, each with {3} rows.",
                     GetBinnedFloatColumns().size(),
                     GetRawFloatColumns().size(),
                     GetStringColumns().size(),
                     num_rows());
}

}  // namespace gbdt
