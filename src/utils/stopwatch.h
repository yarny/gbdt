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

#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <chrono>
#include <ctime>
#include <string>

// Measures elapsed time between Start() and End()
class StopWatch {
 public:
  StopWatch() {}
  void Start();
  void End();
  double ElapsedTimeInMSecs() const;

  // Converts elapsed time in msecs into string formatted as %w%d%h%m%s%ms
  static std::string MSecsToFormattedString(double msecs);

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
  std::chrono::time_point<std::chrono::system_clock> end_;
};


#endif  // STOPWATCH_H_
