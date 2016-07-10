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

#ifndef EVALUATION_H_
#define EVALUATION_H_

#include <string>

#include "src/base/base.h"
#include "src/utils/status.h"

namespace gbdt {

class Config;
class DataStore;
class Forest;

// Evaluates forest on data and outputs score files.
bool EvaluateForest(DataStore* data_store,
                    const Config& config,
                    const Forest& forest,
                    const string& output_dir);

Status EvaluateForest(DataStore* data_store,
                      const Forest& forest,
                      vector<double>* scores);

}  // namespace gbdt

#endif  // EVALUATION_H_
