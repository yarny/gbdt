// Copyright 2015 Jiang Chen. All Rights Reserved.
// Author: Jiang Chen <criver@gmail.com>

#ifndef MEM_DATA_STORE_H_
#define MEM_DATA_STORE_H_

#include <unordered_map>

#include "data_store.h"
#include "src/base/base.h"

namespace gbdt {

class Column;

class MemDataStore : public DataStore {
public:
  bool AddColumn(const string& column_name, unique_ptr<Column>&& column);
};

}  // namespace gbdt

#endif  // MEM_DATA_STORE_H_
