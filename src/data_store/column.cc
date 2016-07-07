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

#include "column.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace gbdt {

using namespace std::placeholders;

namespace {

const long kMaxUInt8 = 256;
const long kMaxUInt16 = 65536;

void UniformBinning(const map<float, uint>& histograms, uint bin_capacity, vector<float>* bins) {
  uint running_sum = 0;
  float upper_limit = NAN;
  for (auto p : histograms) {
    running_sum += p.second;
    upper_limit = p.first;

    if (running_sum >= bin_capacity) {
      bins->push_back(upper_limit);
      running_sum = 0;
    }
  }

  if (running_sum > 0) {
    bins->push_back(upper_limit);
  }
}

template <class INT> vector<INT> ConvertIntVector(const vector<uint> col32) {
  vector<INT> col(col32.size());
  for (uint i = 0; i < col32.size(); ++i) {
    col[i] = static_cast<INT>(col32[i]);
  }
  return col;
}

class IntegerCol8 : public IntegerizedColumn::IntegerCol {
 public:
  IntegerCol8(const vector<uint8>* col) : col_(col) {
  }
  inline uint operator[] (uint i) const override {
    return (*col_)[i];
  }
  inline uint size() const override {
    return col_->size();
  }

 private:
  const vector<uint8>* col_ = nullptr;
};

class IntegerCol16 : public IntegerizedColumn::IntegerCol {
 public:
  IntegerCol16(const vector<uint16>* col) : col_(col) {
  }
  inline uint operator[] (uint i) const override {
    return (*col_)[i];
  }
  inline uint size() const override {
    return col_->size();
  }

 private:
  const vector<uint16>* col_ = nullptr;
};

class IntegerCol32 : public IntegerizedColumn::IntegerCol {
 public:
  IntegerCol32(const vector<uint32>* col) : col_(col) {
  }
  inline uint operator[] (uint i) const override {
    return (*col_)[i];
  }
  inline uint size() const override {
    return col_->size();
  }

 private:
  const vector<uint32>* col_ = nullptr;
};

}  // namespace


Column::Column(const string& name, ColumnType type) : name_(name), type_(type) {
}

Column::ColumnType Column::type() const {
  return type_;
}

const string& Column::name() const {
  return name_;
}

void IntegerizedColumn::Finalize() {
  col_8_.shrink_to_fit();
  col_16_.shrink_to_fit();
  col_32_.shrink_to_fit();

  col_.reset(new IntegerCol32(&col_32_));

  if (col_8_.size() > 0) {
    col_.reset(new IntegerCol8(&col_8_));
  } else if (col_16_.size() > 0) {
    col_.reset(new IntegerCol16(&col_16_));
  }
}

uint IntegerizedColumn::size() const {
  return col().size();
}

StringColumn::StringColumn(const string& name)
    : IntegerizedColumn(name, Column::kStringColumn) {
}

StringColumn::~StringColumn() {
}

const string& StringColumn::get_row_string(uint i) const {
  return map_to_strings_[col()[i]];
}

void StringColumn::Add(const vector<string>* raw_strings) {
  for (const auto& s : *raw_strings) {
    auto it = map_to_indices_.find(s);
    if (it == map_to_indices_.end()) {
      // New string.
      map_to_indices_[s] = map_to_strings_.size();
      col_32_.push_back(map_to_strings_.size());
      map_to_strings_.push_back(s);
    } else {
      // Seen string.
      col_32_.push_back(it->second);
    }
  }
}

void StringColumn::Finalize() {
  // According the number of unique values, convert the vector to
  // 8bit or 16bit col.
  if (max_int() <= kMaxUInt8) {
    col_8_ = ConvertIntVector<uint8>(col_32_);
    col_32_.clear();
  } else if (max_int() <= kMaxUInt16) {
    col_16_ = ConvertIntVector<uint16>(col_32_);
    col_32_.clear();
  }
  IntegerizedColumn::Finalize();
}

BinnedFloatColumn::BinnedFloatColumn(const string& name, int num_bins)
    : IntegerizedColumn(name, Column::kBinnedFloatColumn), num_bins_(num_bins) {
}

BinnedFloatColumn::~BinnedFloatColumn() {
}

template <typename INT> void AddBinnedVecHelper(const vector<float>& raw_floats,
                                                const map<float, uint>& bin_map,
                                                vector<INT>* col,
                                                vector<float>* bin_mins) {
  col->reserve(col->size() + raw_floats.size());
  for (auto v : raw_floats) {
    // NAN represents missing and has index 0.
    if (isnan(v)) {
      col->push_back(0);
    } else {
      auto it = bin_map.lower_bound(v);
      CHECK(it != bin_map.end()) << "This should not happen because the last bin is the max float value.";
      int bin_id = it->second;
      col->push_back(bin_id);
      (*bin_mins)[bin_id] = min((*bin_mins)[bin_id], v);
    }
  }
}

void BinnedFloatColumn::AddBinnedVec(const vector<float>& raw_floats) {
  if (max_int() <= kMaxUInt8) {
    AddBinnedVecHelper<uint8>(raw_floats, bin_map_, &col_8_, &bin_mins_);
  } else if (max_int() <= kMaxUInt16) {
    AddBinnedVecHelper<uint16>(raw_floats, bin_map_, &col_16_, &bin_mins_);
  } else {
    AddBinnedVecHelper<uint32>(raw_floats, bin_map_, &col_32_, &bin_mins_);
  }
}

void BinnedFloatColumn::Add(const vector<float>* raw_floats) {
  CHECK(!finalized_) << "Cannot run Add when finalized.";
  if (bin_maxs_.size() == 0) {
    // Stage 0: build binning.
    buffer_.reserve(buffer_.size() + raw_floats->size());
    for (const float v : *raw_floats) {
      buffer_.push_back(v);
    }
    if (buffer_.size() > 100 * num_bins_) {
      BuildBins();
    }
  } else {
    AddBinnedVec(*raw_floats);
  }
}

void BinnedFloatColumn::Finalize() {
  if (bin_maxs_.size() == 0) {
    BuildBins();
  }
  bin_map_.clear();
  IntegerizedColumn::Finalize();
  finalized_ = true;
}

void BinnedFloatColumn::BuildBins() {
  CHECK(!finalized_) << "Cannot run BuildBins when finalized.";
  const auto& raw_floats = buffer_;
  // Create a histogram of the float values. NaN is treated as missing values and are
  // excluded from the histograms.
  map<float, uint> histograms;
  for (auto v : raw_floats) {
    if (!isnan(v)) {
      ++histograms[v];
    }
  }
  uint bin_capacity = max(static_cast<unsigned long>(1), raw_floats.size() / num_bins_);

  // Put NaN at the beginning of the bins.
  bin_maxs_.push_back(NAN);
  int left_over = raw_floats.size();
  // First, any single value with counts > average bin capacity is put in their own bins.
  auto it = histograms.begin();
  while (it != histograms.end()) {
    if (it->second >= bin_capacity) {
      bin_maxs_.push_back(it->first);
      left_over -= it->second;
      it = histograms.erase(it);
    } else {
      ++it;
    }
  }
  // Use uniform binning for the rest.
  uint left_over_capacity =
      max(static_cast<unsigned long>(1), left_over / (num_bins_ - bin_maxs_.size()));
  UniformBinning(histograms, left_over_capacity, &bin_maxs_);
  sort(bin_maxs_.begin() + 1, bin_maxs_.end());
  bin_maxs_.push_back(numeric_limits<float>::max());

  for (int i = 1; i < bin_maxs_.size(); ++i) {
    bin_map_[bin_maxs_[i]] = i;
  }

  bin_maxs_.shrink_to_fit();
  // Initialize bin_mins_ to be bins_maxs_.
  bin_mins_ = bin_maxs_;
  AddBinnedVec(buffer_);
  buffer_.clear();
  buffer_.shrink_to_fit();
}

unique_ptr<Column> Column::CreateStringColumn(
    const string& name, const vector<string>& raw_strings) {
  auto* column = new StringColumn(name);
  column->Add(&raw_strings);
  column->Finalize();
  return unique_ptr<Column>(column);
}

unique_ptr<Column> Column::CreateBinnedFloatColumn(
    const string& name, const vector<float>& raw_floats, uint num_bins) {
  auto* column = new BinnedFloatColumn(name, num_bins);
  column->Add(&raw_floats);
  column->Finalize();
  return unique_ptr<Column>(column);
}

unique_ptr<Column> Column::CreateRawFloatColumn(
    const string& name, vector<float>&& raw_floats) {
  auto* column = new RawFloatColumn(name);
  column->Add(&raw_floats);
  column->Finalize();

  return unique_ptr<Column>(column);
}

RawFloatColumn::RawFloatColumn(const string& name) : Column(name, Column::kRawFloatColumn) {
}

void RawFloatColumn::Add(const vector<float>* raw_floats) {
  raw_floats_.reserve(raw_floats_.size() + raw_floats->size());
  raw_floats_.insert(raw_floats_.end(), raw_floats->begin(), raw_floats->end());
}

void RawFloatColumn::Finalize() {
  raw_floats_.shrink_to_fit();
}

}  // namespace gbdt
