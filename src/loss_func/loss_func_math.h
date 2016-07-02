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

#ifndef LOSS_FUNC_MATH_H_
#define LOSS_FUNC_MATH_H_

#include "loss_func.h"

namespace gbdt {

LossFuncData ComputeMSE(double y, double f);
LossFuncData ComputeHuberizedHinge(double y, double f);
LossFuncData ComputeLogLoss(double y, double f);
LossFuncData ComputeSquaredHinge(double y, double f);

}  // namespace gbdt

#endif  // LOSS_FUNC_MATH_H_
