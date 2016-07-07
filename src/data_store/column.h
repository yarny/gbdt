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

#ifndef COLUMN_H_
#define COLUMN_H_

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/base/base.h"

namespace gbdt {

class Column {
public:
  enum ColumnType {
    kStringColumn = 0,
    kBinnedFloatColumn = 1,
    kRawFloatColumn = 2,
  };

  virtual ~Column() {}

  static unique_ptr<Column>
      CreateStringColumn(const string& name, const vector<string>& raw_strings);
  static unique_ptr<Column>
      CreateBinnedFloatColumn(const string& name, const vector<float>& raw_floats,
                              uint num_bins=30000);
  static unique_ptr<Column>
      CreateRawFloatColumn(const string& name, vector<float>&& raw_floats);

  ColumnType type() const;
  const string& name() const;
  virtual uint size() const = 0;

protected:
  Column(const string& name, ColumnType type);

private:
  string name_;
  ColumnType type_;
};

// Integerized Column is a column where the values are represented by integers.
// They can be either StringColumn or BinnedFloatColumn.
class IntegerizedColumn : public Column {
 public:
  // An interface to access integer vector of different bits.
  class IntegerCol {
   public:
    virtual uint operator [] (uint) const = 0;
    inline bool missing(uint i) const {
      return (*this)[i] == 0;
    }
    virtual uint size() const = 0;
  };

  virtual ~IntegerizedColumn() {}

  uint size() const;
  virtual uint max_int() const = 0;
  virtual void Finalize();
  inline const IntegerCol& col() const {
    return *col_;
  }

 protected:
  IntegerizedColumn(const string& name, ColumnType type) : Column(name, type) {}

  unique_ptr<IntegerCol> col_;
  // Depending on the number of unique string, the strings are either
  // encoded as 8 bit, 16 bit or 32 bit. The maximum we support is 32 bit.
  vector<uint8> col_8_;
  vector<uint16> col_16_;
  vector<uint> col_32_;
};

// StringColumn.
// __missing__ is reserved to represent missing.
class StringColumn : public IntegerizedColumn {
public:
  StringColumn(const string& name);
  virtual ~StringColumn();

  const string& get_row_string(uint i) const;
  inline uint max_int() const override {
    return map_to_strings_.size();
  }
  inline const string& get_cat_string(uint cat_index) const {
    return map_to_strings_[cat_index];
  }
  inline bool get_cat_index(const string& cat, uint* cat_index) const {
    auto it = map_to_indices_.find(cat);
    if (it != map_to_indices_.end()) {
      (*cat_index) = it->second;
      return true;
    }
    return false;
  }

  void Add(const vector<string>* raw_strings);
  void Finalize() override;

protected:
  StringColumn(const string& name,
               vector<string>&& map_to_strings,
               unordered_map<string, uint>&& map_to_indices)
;
  // The first entry is reserved for "__missing__".
  vector<string> map_to_strings_ = { "__missing__" };
  unordered_map<string, uint> map_to_indices_ = { {"__missing__", 0} };
};

// BinnedFloatColumn.
// Float NAN is use to represent missing.
class BinnedFloatColumn : public IntegerizedColumn {
 public:
  BinnedFloatColumn(const string& name, int num_bins=30000);
  virtual ~BinnedFloatColumn();

  inline float get_row_max(uint i) const {
    return get_bin_max((*col_)[i]);
  }
  inline float get_row_min(uint i) const {
    return get_bin_min((*col_)[i]);
  }
  inline float get_bin_max(uint bin_index) const {
    return bin_maxs_[bin_index];
  }
  inline float get_bin_min(uint bin_index) const {
    return bin_mins_[bin_index];
  }

  // max_int is num_bins + 1. All values exceeding the max upper bound are
  // put in the bin #bins_.size().
  inline uint max_int() const override {
    return bin_maxs_.size();
  }

  void BuildBins();
  void Add(const vector<float>* raw_floats);
  void Finalize() override;

 private:
  void AddBinnedVec(const vector<float>& raw_floats);

  int num_bins_ = 30000;
  // Hold the raw floats before bins is built.
  vector<float> buffer_;

  bool finalized_ = false;

  // bin_max to bin index map.
  map<float, uint> bin_map_;
  // The first bin is NaN representing missing the last bin is always
  // numeric_limits<float>::max(). The bins are represented as [bin_min, bin_max].
  vector<float> bin_maxs_;
  vector<float> bin_mins_;
};

// Simply holds a vector of floats.
class RawFloatColumn : public Column {
public:
  RawFloatColumn(const string& name);
  inline float operator [](uint i) const {
    return raw_floats_[i];
  }
  inline uint size() const override {
    return raw_floats_.size();
  }
  void Add(const vector<float>* raw_floats);
  void Finalize();

protected:
  vector<float> raw_floats_;
};

}  // namespace gbdt

#endif  // COLUMN_H_
