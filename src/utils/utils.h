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

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "utils_inl.h"
#include "vector_slice.h"

namespace strings {

vector<string> split(const string& s, const string& delim);
bool StringCast(const string& s, float* v);
bool HasPrefix(const string& s, const string& prefix);
bool HasSuffix(const string& s, const string& suffix);

}  // namespace string

// approximate float equality function
inline bool ApproximatelyEqual(float a, float b,
			       float relative_tolerance = 1e-6) {
  if (a == 0 && b == 0)
    return true;
  float e = fabs(a-b)/fabs(max(a,b));
  return e < relative_tolerance;
}

string ReadFileToStringOrDie(const string& file);
void WriteStringToFile(const string& content, const string& file);

vector<uint> VectorSliceToVector(VectorSlice<uint> slice);

bool FileExists(const string& file);

inline void LTrimWhiteSpace(string *s) {
  s->erase(s->begin(), find_if(s->begin(), s->end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
}

inline void RTrimWhiteSpace(string *s) {
  s->erase(find_if(s->rbegin(), s->rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
           s->end());
}

inline void TrimWhiteSpace(string *s) {
  LTrimWhiteSpace(s);
  RTrimWhiteSpace(s);
}

#endif  // UTILS_H_
