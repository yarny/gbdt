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

// Base class for pairwise loss funcs.
class Pairwise : public LossFunc {
 public:
  // delta_target is always positive since we only generates pairs where the first has larger target
  // value.
  Pairwise(const Config& config, bool rerank, PointwiseLossFunc loss_func);

  virtual Status Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* group_column) override;
  virtual void ComputeFunctionalGradientsAndHessians(const vector<double>& f,
                                                     double* c,
                                                     vector<GradientData>* gradient_data_vec,
                                                     string* progress) override;

 protected:
  // The following function provides custom interface for adding custom pair weights.
  // This weights can be used to implement listwise loss functions like LambdaMart.
  virtual function<double(const pair<uint, uint>&)> PairWeightingFunc(const Group& group) const;

  string PrepareProgressMessage(double loss);

 private:
  vector<Group> groups_;

  // Division of [0, group_size) into slices to help multithreading.
  vector<pair<uint, uint>> slices_;

  double initial_loss_ = -1;
  uint64 min_num_pairs_ = 1;
  FloatVector w_;
  FloatVector y_;
  double pair_sampling_rate_;
  double pair_sampling_probability_;
  bool pair_weight_by_delta_target_;
  bool equal_group_weight_;
  // If true, the algorithm will rerank each group every iteration.
  bool rerank_ = false;
  PointwiseLossFunc loss_func_;
};


}  // namespace gbdt

#endif  // LOSS_FUNC_PAIRWISE_H_
