// Copyright 2010 Jiang Chen. All Rights Reserved.
// Author: Jiang Chen <criver@gmail.com>

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
