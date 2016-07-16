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

#include "stopwatch.h"

#include "external/cppformat/format.h"

#include "utils.h"

void StopWatch::Start() {
  start_ = std::chrono::system_clock::now();;
}

void StopWatch::End() {
  end_ = std::chrono::system_clock::now();;
}

double StopWatch::ElapsedTimeInMSecs() const {
  std::chrono::duration<double, std::milli> diff = end_ - start_;
  return diff.count();
}

string StopWatch::MSecsToFormattedString(double ms) {
  if (ms < 1000) {
    return fmt::format("{0}ms", int(ms));
  }

  double seconds = ms / 1000.0;
  int mins = int(seconds / 60);
  seconds -= 60 * mins;
  int hrs = mins / 60;
  mins -= hrs * 60;
  int days = hrs / 24;
  hrs -= 24 * days;

  return ((days == 0 ? "" : fmt::format("{0}d", days)) +
          (hrs == 0 ? "" : fmt::format("{0}h", hrs)) +
          (mins == 0 ? "" : fmt::format("{0}m", mins)) +
          (seconds < 0.05 ? "" : fmt::format("{0:.1f}s",seconds)));
}
