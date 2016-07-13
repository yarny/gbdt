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

// Gradient Boosting Decision Tree (GBDT) Algorithms
// http://statweb.stanford.edu/~jhf/ftp/stobst.pdf

#ifndef GBDT_ALGO_H_
#define GBDT_ALGO_H_

#include <memory>

#include "src/base/base.h"

namespace gbdt {

class Config;
class DataStore;
class Forest;
class LossFunc;
class TreeConfig;

// GBDT algorithnm
// Gradient Boosting Decision Tree (GBDT) Algorithm.
// The GBDT algorithm contains two configurable building blocks:
//
//  * LossFunc: Computes functional gradients and also has the option of changing example
//    weights.
//  * TreeFitter (FitTreeToGradients): Given the gradients and examples weights, build
//    a tree to minimize weighted mse of the gradients and weights.
//
// The generic framework allows us to implement a variety of algorithms by writing custom
// loss function.
//
// Inputs:
// * Config contains a collection of configurations including generate training params, loss
//   function, and tree building params.
// * DataStore is a container of data. It is not a const variable because some of our data
//   can be loaded in a lazy fashion.

Status TrainGBDT(DataStore* data_store,
                 const unordered_set<string>& feature_names,
                 FloatVector w,
                 FloatVector y,
                 LossFunc* loss_func,
                 const Config& config,
                 const Forest* base_forest,
                 Forest* forest);

}  // namespace gbdt

#endif  // GBDT_ALGO_H_
