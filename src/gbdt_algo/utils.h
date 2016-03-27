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

#ifndef GBDT_ALGO_UTILS_H_
#define GBDT_ALGO_UTILS_H_

#include <unordered_set>
#include <vector>

#include "src/base/base.h"

namespace gbdt {

class Column;
class DataStore;
class Forest;

vector<const Column*> LoadFeaturesOrDie(unordered_set<string>& feature_names,
                                        DataStore* data_store);
vector<pair<string, double>> ComputeFeatureImportance(const Forest& forest);
unordered_set<string> CollectAllFeatures(const Forest& forest);

}  // namespace gbdt

#endif  // GBDT_ALGO_UTILS_H_
