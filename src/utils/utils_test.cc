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

#include "utils.h"
#include "utils_inl.h"

#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace gbdt {

TEST(SplitStringTest, SplitString) {
  EXPECT_EQ(vector<string>({"1", "2", "3", "4", "5", ""}),
            strings::split("1,2,3,4,5,", ","));
}

TEST(StringCastTest, StringCast) {
  float v;
  EXPECT_TRUE(strings::StringCast("1.5", &v));
  EXPECT_EQ(1.5, v);
  EXPECT_TRUE(strings::StringCast("-1.5", &v));
  EXPECT_EQ(-1.5, v);
  EXPECT_FALSE(strings::StringCast("foo", &v));
}

TEST(JoinStringsTest, JoinStrings) {
  vector<string> test1 = {"hello", "world", "foo", "bar"};
  EXPECT_EQ("hello,world,foo,bar", strings::JoinStrings(test1, ","));
  EXPECT_EQ("hello|world|foo|bar", strings::JoinStrings(test1, "|"));
  set<string> test2 = {"hello", "world", "foo", "bar"};
  EXPECT_EQ("bar,foo,hello,world", strings::JoinStrings(test2, ","));
  EXPECT_EQ("bar|foo|hello|world", strings::JoinStrings(test2, "|"));
}

TEST(HasPrefixSuffixTest, HasPrefixSuffix) {
  EXPECT_TRUE(strings::HasPrefix("foobar", "foo"));
  EXPECT_FALSE(strings::HasPrefix("foobar", "foo2"));
  EXPECT_TRUE(strings::HasSuffix("foobar", "bar"));
  EXPECT_FALSE(strings::HasSuffix("foobar", "bar2"));
}

}  // namespace gbdt
