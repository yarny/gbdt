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

#ifndef STATUS_H_
#define STATUS_H_

#include <string>

#include "src/base/base.h"

namespace gbdt {

namespace error {
  enum ErrorCode {
    OK = 1,
    NOT_FOUND = 2,
    OUT_OF_RANGE = 3,
    INVALID_ARGUMENT = 4,
    INVALID_OPERATION = 5,
  };
}  // error

// A simply status class that carries error code and error messages.
class Status {
 public:
  Status(error::ErrorCode code, const string& msg);
  Status() {};
  static Status OK();
  bool ok() const;
  const string& error_msg() const;
  error::ErrorCode error_code() const;

 private:
   error::ErrorCode code_ = error::OK;
   string msg_;
};

}  // namespace gbdt

#endif  // STATUS_H_
