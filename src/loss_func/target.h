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

#ifndef TARGET_H_
#define TARGET_H_

#include <vector>

#include "src/base/base.h"

namespace gbdt {

class LossFuncConfig;
class DataStore;

// Converts the target column to {-1, 1} binary targets.
bool ComputeBinaryTargets(DataStore* data_store, const LossFuncConfig& config,
                          vector<float>* targets);

}  // namespace gbdt

#endif  // TARGET_H_
