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

#include "gtest/gtest.h"

namespace gbdt {

TEST(StopWatchTest, StopWatchTest) {
  std::string str = StopWatch::MSecsToFormattedString(1203121.0);
  EXPECT_EQ(str, "20m3s121ms");
  str = StopWatch::MSecsToFormattedString(604800021.0);
  EXPECT_EQ(str, "1w21ms");
}

}  // namespace gbdt
