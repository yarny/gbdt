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

#include "mem_data_store.h"

#include <memory>
#include <string>
#include <utility>

namespace gbdt {

bool MemDataStore::AddColumn(const string& column_name, unique_ptr<Column>&& column) {
  auto it = column_map_.find(column_name);
  if (it != column_map_.end()) {
    return false;
  }
  if (num_rows() > 0 && num_rows() != column->size()) {
    LOG(ERROR) << "Row size consistency check failed for " << column_name
               << "(old " << num_rows() << " vs. " << "new " << column->size();
    return false;
  }
  
  column_map_[column_name] = std::move(column);
  return true;
}

}  // namespace gbdt
