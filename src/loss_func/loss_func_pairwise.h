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

#include "group.h"
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
  // delta_target is always positive since we only generates pairs where the first has larger target
  // value.
  typedef std::function<LossFuncData(double delta_target, double delta_func)> PairwiseLossFunc;
  Pairwise(const LossFuncConfig& config, PairwiseLossFunc loss_func);

  virtual Status Init(int num_rows, FloatVector w, FloatVector y, DataStore* data_store) override;
  virtual void ComputeFunctionalGradientsAndHessians(const vector<double>& f,
                                                     double* c,
                                                     vector<GradientData>* gradient_data_vec,
                                                     string* progress) override;

 protected:
  // The following function provides custom interface for adding custom pair weights.
  // This weights can be used to implement listwise loss functions like LambdaMart.
  virtual function<double(const pair<uint, uint>&)> GeneratePairWeightingFunc(
      const vector<uint>& group, const vector<double>& f);

  string PrepareProgressMessage(double loss);
  LossFuncConfig config_;
  FloatVector w_;
  FloatVector y_;

 private:
  vector<Group> groups_;
  // Division of [1, group_size] into slices to help multithreading.
  vector<pair<uint, uint>> slices_;
  double initial_loss_ = -1;

  PairwiseLossFunc loss_func_;
};


}  // namespace gbdt

#endif  // LOSS_FUNC_PAIRWISE_H_
