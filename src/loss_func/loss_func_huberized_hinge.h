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

#ifndef LOSS_FUNC_HUBERIZED_HINGE_H_
#define LOSS_FUNC_HUBERIZED_HINGE_H_

#include "loss_func_pointwise.h"
#include "src/proto/config.pb.h"

namespace gbdt {

class DataStore;

// Huberized Hinge Loss (https://en.wikipedia.org/wiki/Hinge_loss) is a variation of Hinge loss that
// are smooth hence easier to optimize.
// Huberized Hinge = 1/2 - y *f         if yf <= 0
//                 = 1/2 (1 - y *f)^2   if 0 < y*f < 1
//                 = 0                  if y*f >= 1
// y is {-1, 1} valued.
class HuberizedHinge : public Pointwise {
 public:
  HuberizedHinge(const LossFuncConfig& config);

  Status Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* unused_group_column) override;

 private:
  LossFuncConfig config_;
};

}  // namespace gbdt

#endif  // LOSS_FUNC_HUBERIZED_HINGE_H_
