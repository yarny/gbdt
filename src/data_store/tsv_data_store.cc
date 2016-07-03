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

#include "tsv_data_store.h"

#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "column.h"
#include "external/cppformat/format.h"
#include "tsv_block.h"
#include "src/proto/config.pb.h"
#include "src/utils/threadpool.h"
#include "src/utils/stopwatch.h"
#include "src/utils/utils.h"

DECLARE_int32(num_threads);

namespace gbdt {

namespace {

string ReadFirstLine(const string& tsv) {
  std::ifstream in(tsv);
  return ReadLine(in);
}

Status MaybeFindFirstNotOK(const unordered_map<string, unique_ptr<Column>>& column_map) {
  for (const auto&p : column_map) {
    if (!p.second->status().ok()) {
      return p.second->status();
    }
  }
  return Status::OK();
}

}  // namespace

TSVDataStore::TSVDataStore(const vector<string>& tsvs, const DataConfig& config) {
  status_ = LoadTSVs(tsvs, config);
}

Status TSVDataStore::LoadTSVs(const vector<string>& tsvs, const DataConfig& config) {
  Status status;
  if (tsvs.size() <= 0) {
    return Status(error::INVALID_ARGUMENT, "There should be at least 1 tsvs.");
  }
  StopWatch stopwatch;
  stopwatch.Start();
  status = SetupColumns(tsvs[0], config);
  if (!status.ok()) {
    return status;
  }

  vector<promise<TSVBlock*>> blocks(tsvs.size());
  ThreadPool pool(FLAGS_num_threads);
  for (int i = 0; i < tsvs.size(); ++i) {
    pool.Enqueue([this, &block=blocks[i], tsv=tsvs[i], skip_header=(i==0)] {
        block.set_value(new TSVBlock(tsv, float_column_indices_, string_column_indices_, skip_header));
      });
  }

  for (int i = 0; i < blocks.size(); ++i) {
    auto block_future = blocks[i].get_future();
    block_future.wait();
    unique_ptr<TSVBlock> block(block_future.get());
    if (!block->status().ok()) {
      return block->status();
    }
    status = ProcessBlock(block.get());
    if (!status.ok()) return status;
    LOG(INFO) << "Processed block " << tsvs[i] << ".";
  }

  status = Finalize();
  if (!status.ok()) return status;
  stopwatch.End();
  LOG(INFO) << "Finished loading tsvs in "
            << StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs());

  return Status::OK();
}

Status TSVDataStore::ProcessBlock(const TSVBlock* block) {
  {
    ThreadPool pool(FLAGS_num_threads);
    for (auto& p : binned_float_columns_) {
      pool.Enqueue([&] { p.first->Add(&block->float_columns()[p.second]); });
    }
    for (auto& p : raw_float_columns_) {
      pool.Enqueue([&] { p.first->Add(&block->float_columns()[p.second]); });
    }
    for (auto& p : string_columns_) {
      pool.Enqueue([&] { p.first->Add(&block->string_columns()[p.second]); });
    }
  }

  return MaybeFindFirstNotOK(column_map_);
}

Status TSVDataStore::Finalize() {
  {
    ThreadPool pool(FLAGS_num_threads);
    for (auto& p : binned_float_columns_) {
      pool.Enqueue([&] { p.first->Finalize(); });
    }
    for (auto& p : string_columns_) {
      pool.Enqueue([&] { p.first->Finalize(); });
    }
    for (auto& p : raw_float_columns_) {
      pool.Enqueue([&] { p.first->Finalize(); });
    }
  }
  return MaybeFindFirstNotOK(column_map_);
}

Status TSVDataStore::SetupColumns(const string& first_tsv, const DataConfig& config) {
  // Read header from first tsv.
  if (!FileExists(first_tsv)) {
    return Status(error::NOT_FOUND, fmt::format("TSV {0} does not exit.", first_tsv));
  }
  vector<string> headers = strings::split(ReadFirstLine(first_tsv), "\t");

  unordered_map<string, int> map_from_header_to_index;
  for (int i = 0; i < headers.size(); ++i) {
    TrimWhiteSpace(&headers[i]);
    map_from_header_to_index[headers[i]] = i;
  }

  // Add float features as binned float columns.
  for (const string& header : config.float_feature()) {
    auto it = map_from_header_to_index.find(header);
    if (it == map_from_header_to_index.end()) {
      return Status(error::NOT_FOUND,
                    fmt::format("Failed to find column {0} in {1}.", header, first_tsv));
    }
    column_map_[header].reset(new BinnedFloatColumn(header));
    binned_float_columns_.push_back(
        make_pair(static_cast<BinnedFloatColumn*>(column_map_[header].get()),
                  float_column_indices_.size()));
    float_column_indices_.push_back(it->second);
  }

  // Add additional float columns as raw float columns.
  for (const string& header : config.additional_float_column()) {
    auto it = map_from_header_to_index.find(header);
    if (it == map_from_header_to_index.end()) {
      return Status(error::NOT_FOUND,
                    fmt::format("Failed to find column {0} in {1}", header, first_tsv));
    }
    column_map_[header].reset(new RawFloatColumn(header));
    raw_float_columns_.push_back(
        make_pair(static_cast<RawFloatColumn*>(column_map_[header].get()),
                  float_column_indices_.size()));
    float_column_indices_.push_back(it->second);
  }

  // Add categorical features as string columns.
  for (const string& header : config.categorical_feature()) {
    auto it = map_from_header_to_index.find(header);
    if (it == map_from_header_to_index.end()) {
      return Status(error::NOT_FOUND, "Failed to find column " + header + " in " + first_tsv);
    }
    column_map_[header].reset(new StringColumn(header));
    string_columns_.push_back(
        make_pair(static_cast<StringColumn*>(column_map_[header].get()),
                  string_column_indices_.size()));
    string_column_indices_.push_back(it->second);
  }

  // Add additional string columns.
  for (const string& header : config.additional_string_column()) {
    auto it = map_from_header_to_index.find(header);
    if (it == map_from_header_to_index.end()) {
      return Status(error::NOT_FOUND, "Failed to find column " + header + " in " + first_tsv);
    }
    column_map_[header].reset(new StringColumn(header));
    string_columns_.push_back(
        make_pair(static_cast<StringColumn*>(column_map_[header].get()),
                  string_column_indices_.size()));
    string_column_indices_.push_back(it->second);
  }

  return Status::OK();
}

}  // namespace gbdt
