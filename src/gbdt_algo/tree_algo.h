/*
 * Copyright 2016 Jiang Chen <criver@gmail.com>
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

#ifndef TREE_ALGO_H_
#define TREE_ALGO_H_

#include <vector>

#include "src/base/base.h"

namespace gbdt {

class Column;
struct GradientData;
class SamplingConfig;
class Config;
class TreeNode;

// Given gradients and weights, fit trees to minimize mse.
// It subsamples the examples and features according to the sampling_config.
TreeNode FitTreeToGradients(FloatVector w,
                            const vector<GradientData>& gradient_data_vec,
                            const vector<const Column*>& features,
                            const Config& config);

}  // namespace gbdt

#endif  // GBDT_ALGO_H_
