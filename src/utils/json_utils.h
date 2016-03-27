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

#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver.h>
#include <memory>
#include <string>

#include "src/base/base.h"

class JsonUtils {
 public:
  JsonUtils();

  static string ToJsonOrDie(const google::protobuf::Message& message);
  static bool FromJson(const string& json, google::protobuf::Message* message);

 private:
  static std::unique_ptr<google::protobuf::util::TypeResolver> resolver_;
};
