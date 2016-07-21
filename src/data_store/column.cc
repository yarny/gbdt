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

#include "external/cppformat/format.h"

namespace gbdt {

using namespace std::placeholders;

namespace {

const long kMaxUInt8 = 256;
const long kMaxUInt16 = 65536;

void UniformBinning(const map<float, uint>& histograms, uint bucket_capacity, vector<float>* buckets) {
  uint running_sum = 0;
  float upper_limit = NAN;
  for (auto p : histograms) {
    running_sum += p.second;
    upper_limit = p.first;

    if (running_sum >= bucket_capacity) {
      buckets->push_back(upper_limit);
      running_sum = 0;
    }
  }

  if (running_sum > 0) {
    buckets->push_back(upper_limit);
  }
}

template <class INT> vector<INT> ConvertIntVector(const vector<uint>& col32) {
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
  finalized_ = true;
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
  if (!status_.ok()) return;
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
  if (!status_.ok()) return;
  if (finalized_) {
    status_ = Status(error::FAILED_PRECONDITION, "Cannot run Add after finalized.");
    return;
  }
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

BucketizedFloatColumn::BucketizedFloatColumn(const string& name, int num_buckets)
    : IntegerizedColumn(name, Column::kBucketizedFloatColumn), num_buckets_(num_buckets) {
}

BucketizedFloatColumn::~BucketizedFloatColumn() {
}

template <typename INT> Status AddBucketizedVecHelper(const vector<float>& raw_floats,
                                                      const map<float, uint>& bucket_map,
                                                      vector<INT>* col,
                                                      vector<float>* bucket_mins) {
  col->reserve(col->size() + raw_floats.size());
  for (auto v : raw_floats) {
    // NAN represents missing and has index 0.
    if (isnan(v)) {
      col->push_back(0);
    } else {
      auto it = bucket_map.lower_bound(v);
      if (it == bucket_map.end()) {
        return Status(error::OUT_OF_RANGE,
                      fmt::format(
                          "This should not happen because the last bucket is the max float value. "
                          "Value ({0})", v));
      }
      int bucket_id = it->second;
      col->push_back(bucket_id);
      (*bucket_mins)[bucket_id] = min((*bucket_mins)[bucket_id], v);
    }
  }
  return Status::OK;
}

Status BucketizedFloatColumn::AddBucketizedVec(const vector<float>& raw_floats) {
  Status status;
  if (max_int() <= kMaxUInt8) {
    return AddBucketizedVecHelper<uint8>(raw_floats, bucket_map_, &col_8_, &bucket_mins_);
  } else if (max_int() <= kMaxUInt16) {
    return AddBucketizedVecHelper<uint16>(raw_floats, bucket_map_, &col_16_, &bucket_mins_);
  } else {
    return AddBucketizedVecHelper<uint32>(raw_floats, bucket_map_, &col_32_, &bucket_mins_);
  }
  return Status::OK;
}

void BucketizedFloatColumn::Add(const vector<float>* raw_floats) {
  if (!status_.ok()) return;
  if (finalized_) {
    status_ = Status(error::FAILED_PRECONDITION, "Cannot run Add after finalized.");
    return;
  }
  if (bucket_maxs_.size() == 0) {
    // Stage 0: build buckets.
    buffer_.reserve(buffer_.size() + raw_floats->size());
    for (const float v : *raw_floats) {
      buffer_.push_back(v);
    }
    if (buffer_.size() > 100 * num_buckets_) {
      BuildBuckets();
    }
  } else {
    status_ = AddBucketizedVec(*raw_floats);
  }
}

void BucketizedFloatColumn::Finalize() {
  if (!status_.ok()) return;
  if (finalized_) {
    status_ = Status(error::FAILED_PRECONDITION, "Cannot run Add after finalized.");
    return;
  }
  if (bucket_maxs_.size() == 0) {
    BuildBuckets();
    if (!status_.ok()) return;
  }
  bucket_map_.clear();
  IntegerizedColumn::Finalize();
}

void BucketizedFloatColumn::BuildBuckets() {
  if (!status_.ok()) return;
  if (finalized_) {
    status_ = Status(error::FAILED_PRECONDITION, "Cannot run BuildBuckets after it is finalized");
    return;
  }
  const auto& raw_floats = buffer_;
  // Create a histogram of the float values. NaN is treated as missing values and are
  // excluded from the histograms.
  map<float, uint> histograms;
  for (auto v : raw_floats) {
    if (!isnan(v)) {
      ++histograms[v];
    }
  }
  uint bucket_capacity = max(static_cast<unsigned long>(1), raw_floats.size() / num_buckets_);

  // Put NaN at the beginning of the buckets.
  bucket_maxs_.push_back(NAN);
  int left_over = raw_floats.size();
  // First, any single value with counts > average bucket capacity is put in their own buckets.
  auto it = histograms.begin();
  while (it != histograms.end()) {
    if (it->second >= bucket_capacity) {
      bucket_maxs_.push_back(it->first);
      left_over -= it->second;
      it = histograms.erase(it);
    } else {
      ++it;
    }
  }
  // Use uniform binning for the rest.
  uint left_over_capacity =
      max(static_cast<unsigned long>(1), left_over / (num_buckets_ - bucket_maxs_.size()));
  UniformBinning(histograms, left_over_capacity, &bucket_maxs_);
  sort(bucket_maxs_.begin() + 1, bucket_maxs_.end());
  bucket_maxs_.push_back(numeric_limits<float>::max());

  for (int i = 1; i < bucket_maxs_.size(); ++i) {
    bucket_map_[bucket_maxs_[i]] = i;
  }

  bucket_maxs_.shrink_to_fit();
  // Initialize bucket_mins_ to be buckets_maxs_.
  bucket_mins_ = bucket_maxs_;
  status_ = AddBucketizedVec(buffer_);
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

unique_ptr<Column> Column::CreateBucketizedFloatColumn(
    const string& name, const vector<float>& raw_floats, uint num_buckets) {
  auto* column = new BucketizedFloatColumn(name, num_buckets);
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
  if (!status_.ok()) return;
  raw_floats_.reserve(raw_floats_.size() + raw_floats->size());
  raw_floats_.insert(raw_floats_.end(), raw_floats->begin(), raw_floats->end());
}

void RawFloatColumn::Finalize() {
  if (!status_.ok()) return;
  raw_floats_.shrink_to_fit();
}

const vector<float>& RawFloatColumn::raw_floats() const {
  return raw_floats_;
}

}  // namespace gbdt
