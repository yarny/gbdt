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

#ifndef LOSS_FUNC_PAIRWISE_H_
#define LOSS_FUNC_PAIRWISE_H_

#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "loss_func.h"
#include "loss_func_math.h"
#include "src/base/base.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class RawFloatColumn;
class DataStore;

// Base class for pairwise loss funcs.
class Pairwise : public LossFunc {
 public:
  typedef std::function<LossFuncData(double delta_target, double delta_func)> PairwiseLossFunc;
  Pairwise(const LossFuncConfig& config, PairwiseLossFunc loss_func);

  virtual bool Init(DataStore* data_store, const vector<float>& w) override;
  virtual void ComputeFunctionalGradientsAndHessians(const vector<double>& f,
                                                     double* c,
                                                     vector<GradientData>* gradient_data_vec,
                                                     string* progress) override;

 protected:
  // The following function provides custom interface for adding custom pair weights.
  // This weights can be used to implement listwise loss functions like LambdaMart.
  virtual function<double(const pair<uint, uint>&)> GeneratePairWeightingFunc(
      const vector<uint>& group, const vector<double>& f);
  const RawFloatColumn* target_column_ = nullptr;

 private:
  class Group {
   public:
    Group(vector<uint>&& group, const RawFloatColumn* target_column,
          std::mt19937* generator);

    // NOTE: Positive and negative are relative in this setting. Each pair has a positive and
    // a negative but each item can be positive in one pair but negative in another.

    // Randomly sample a pair from the group.
    pair<uint, uint> SamplePair() const;
    const vector<uint>& group() const {
      return group_;
    }

    uint num_pairs() const {
      return num_pairs_;
    }

   private:
    vector<uint> group_;
    std::mt19937* generator_;
    uint num_pairs_ = 0;

    // The following data structure is used to map pair index to the actual
    // pair. Each entry represent a target block (instances with the
    // same target value), the key is the total accumulated up to this
    // target block. The value is (num_instances_in_block, start_of_negative) pair.
    map<uint, pair<uint, uint>> pair_map_;
  };

  string PrepareProgressMessage(double loss);

  vector<Group> groups_;
  // Division of [1, group_size] into slices to help multithreading.
  vector<pair<uint, uint>> slices_;
  double initial_loss_ = -1;
  LossFuncConfig config_;
  static unique_ptr<std::mt19937> generator_;
  const vector<float>* w_;

  PairwiseLossFunc loss_func_;
};


}  // namespace gbdt

#endif  // LOSS_FUNC_PAIRWISE_H_
