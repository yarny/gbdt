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

#ifndef LOSS_FUNC_H_
#define LOSS_FUNC_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/base/base.h"

namespace gbdt {

class LossFuncConfig;
class DataStore;

class LossFunc {
 public:
  struct Data {
    Data(double loss_arg, double g_arg, double h_arg) : loss(loss_arg), g(g_arg), h(h_arg) {}
    double loss;
    double g;
    double h;
  };

  virtual ~LossFunc() {}

  virtual bool Init(DataStore* data_store, const vector<float>& sample_weights) = 0;

  // We don't need to output the constant to make the algorithm work, but outputting a constant
  // which won't be scaled down by shrinkage helps the algorithm converge faster.
  virtual void ComputeFunctionalGradientsAndHessians(const vector<double>& f,
                                                     double* c,
                                                     vector<double>* g,
                                                     vector<double>* h,
                                                     string* progress) = 0;
};

}  // namespace gbdt

#endif  // LOSS_FUNC_H_
