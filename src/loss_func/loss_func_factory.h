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

#ifndef LOSS_FUNC_FACTORY_H_
#define LOSS_FUNC_FACTORY_H_

#include <memory>
#include <functional>
#include <string>
#include <unordered_map>

#include "loss_func.h"

namespace gbdt {

class Config;
class LossFuncFactory {
public:
  typedef std::function<LossFunc*(const Config&)> Creator;
  static unique_ptr<LossFunc> CreateLossFunc(const Config& config);
  static vector<string> LossFuncs();

private:
  static unordered_map<string, Creator> loss_func_creator_map_;
};

}  // namespace gbdt

#endif  // LOSS_FUNC_FACTORY_H_
