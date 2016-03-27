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
  const uint num_levels = 6;
  const uint time_level[] = {1000, 60, 60, 24, 7};
  const string time_unit_names[] = {"ms", "s", "m", "h", "d", "w"};
  vector<uint> time(num_levels, 0);
  
  uint total_units = (uint) ms;
    
  for (uint i = 0; i < num_levels-1 && total_units > 0; ++i) {
    time[i] = total_units % time_level[i];
    total_units = total_units / time_level[i];
  }
  time[num_levels-1] = total_units;
  
  vector<string> formatted_str;
  for(uint i = 0; i < num_levels; ++i) {
    if (time[i] > 0) {
      formatted_str.insert(formatted_str.begin(),
			   fmt::format("{0}{1}",
				       time[i],
				       time_unit_names[i]));
    }
  }

  if (formatted_str.size() == 0)
    return "0ms";
  
  return strings::JoinStrings(formatted_str, "");
}
