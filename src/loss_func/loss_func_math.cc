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

#include "loss_func_math.h"

#include <cmath>

#include "loss_func.h"

namespace gbdt {

LossFunc::Data ComputeMSE(double y, double f) {
  return LossFunc::Data((y - f) * (y -f),
                         y - f,
                         1.0);
}

LossFunc::Data ComputeLogLoss(double y, double f) {
  double e = exp(-y * f);
  return LossFunc::Data(log(1 + e),
                        y * e / (1 + e),
                        e / ((1 + e) * (1 + e)));
};

LossFunc::Data ComputeHuberizedHinge(double y, double f) {
  double e = y * f;
  if (e >= 1) {
    // Margin is greater than 1.0. Hinge loss is 0.
    return LossFunc::Data(0.0, 0.0, 0.0);
  } else if (e >= 0) {
    return LossFunc::Data(0.5 * (1 - e) * (1 - e), (1 - e) * y, 1);
  } else {
    return LossFunc::Data(0.5 - e, y, 0);
  }
}

}  // namespace gbdt
