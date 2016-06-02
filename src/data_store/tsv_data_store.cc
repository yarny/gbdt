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
#include "tsv_block.h"
#include "src/proto/config.pb.h"
#include "src/utils/threadpool.h"
#include "src/utils/stopwatch.h"
#include "src/utils/utils.h"

DECLARE_int32(num_threads);

namespace gbdt {

TSVDataStore::TSVDataStore(const string& header_file,
                           const vector<string>& tsvs,
                           const TSVDataConfig& config) {
  LoadTSVs(header_file, tsvs, config);
}

void TSVDataStore::LoadTSVs(const string& header_file,
                            const vector<string>& tsvs,
                            const TSVDataConfig& config) {
  StopWatch stopwatch;
  stopwatch.Start();
  SetupColumns(header_file, config);

  vector<promise<TSVBlock*>> blocks(tsvs.size());
  ThreadPool pool(FLAGS_num_threads);
  for (int i = 0; i < tsvs.size(); ++i) {
    pool.Enqueue([this, &block=blocks[i], tsv=tsvs[i]] {
        CHECK(FileExists(tsv)) << "TSV " << tsv << " does not exist.";
        block.set_value(new TSVBlock(tsv, float_column_indices_, string_column_indices_));
      });
  }

  for (int i = 0; i < blocks.size(); ++i) {
    auto block_future = blocks[i].get_future();
    block_future.wait();
    unique_ptr<TSVBlock> block(block_future.get());
    ProcessBlock(block.get());
    LOG(INFO) << "Processed block " << tsvs[i] << ".";
  }

  Finalize();
  stopwatch.End();
  LOG(INFO) << "Finished loading tsvs in "
            << StopWatch::MSecsToFormattedString(stopwatch.ElapsedTimeInMSecs());
}

void TSVDataStore::ProcessBlock(const TSVBlock* block) {
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

void TSVDataStore::Finalize() {
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

void TSVDataStore::SetupColumns(const string& header_file,
                                const TSVDataConfig& config) {
  // Read header file.
  vector<string> headers = strings::split(ReadFileToStringOrDie(header_file), "\t");
  unordered_map<string, int> map_from_header_to_index;
  for (int i = 0; i < headers.size(); ++i) {
    TrimWhiteSpace(&headers[i]);
    map_from_header_to_index[headers[i]] = i;
  }

  for (const string& header : config.binned_float_column()) {
    auto it = map_from_header_to_index.find(header);
    CHECK(it != map_from_header_to_index.end()) << "Failed to find " << header << " in header file.";
    column_map_[header].reset(new BinnedFloatColumn(header));
    binned_float_columns_.push_back(
        make_pair(static_cast<BinnedFloatColumn*>(column_map_[header].get()),
                  float_column_indices_.size()));
    float_column_indices_.push_back(it->second);
  }

  for (const string& header : config.raw_float_column()) {
    auto it = map_from_header_to_index.find(header);
    CHECK(it != map_from_header_to_index.end()) << "Failed to find " << header << " in header file.";
    column_map_[header].reset(new RawFloatColumn(header));
    raw_float_columns_.push_back(
        make_pair(static_cast<RawFloatColumn*>(column_map_[header].get()),
                  float_column_indices_.size()));
    float_column_indices_.push_back(it->second);
  }

  for (const string& header : config.string_column()) {
        auto it = map_from_header_to_index.find(header);
    CHECK(it != map_from_header_to_index.end()) << "Failed to find " << header << " in header file.";
    column_map_[header].reset(new StringColumn(header));
    string_columns_.push_back(
        make_pair(static_cast<StringColumn*>(column_map_[header].get()),
                  string_column_indices_.size()));
    string_column_indices_.push_back(it->second);
  }
}

}  // namespace gbdt
