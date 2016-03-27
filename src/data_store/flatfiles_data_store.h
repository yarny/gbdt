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

#ifndef FLATFILES_DATA_STORE_H_
#define FLATFILES_DATA_STORE_H_

#include <mutex>
#include <string>
#include <vector>

#include "column.h"
#include "data_store.h"
#include "src/base/base.h"

namespace gbdt {

// FlatfilesDataStore is an implmentation of DataStore where the data are stored
// in a directory of flatfiles. The data are loaded in a lazy way.
class FlatfilesDataStore : public DataStore {
public:
  FlatfilesDataStore(const string& flatfiles_dir);
  FlatfilesDataStore(const vector<string>& flatfiles_dirs);
  // The function load the data lazily. Although it is not declared as const,
  // but the function is thread-safe since the write-access to the data
  // is lock guarded.
  const Column* GetColumn(const string& column_name) override;

private:
  bool LoadColumn(const string& column_name);
  unique_ptr<Column> LoadStringColumn(ifstream& in, const string& column_name);
  unique_ptr<Column> LoadFloatColumn(ifstream& in, const string& column_name, bool binned);
  string FindFlatfile(const string& column_name) const;

  vector<string> flatfiles_dirs_;

  mutex mutex_;
};

}  // namespace gbdt

#endif  // FLATFILES_DATA_STORE_H_
