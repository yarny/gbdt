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

#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <streambuf>
#include <string>

string ReadFileToStringOrDie(const string& file) {
  std::ifstream t(file);
  string str;
  t.seekg(0, std::ios::end);
  if (t.tellg() <= 0) {
    LOG(FATAL) << "Failed to read " << file
               << ". Please check its existence and permission.";
  }
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(t)),
             std::istreambuf_iterator<char>());
  return str;
}

void WriteStringToFile(const string& content, const string& file) {
  std::ofstream out(file);
  CHECK(!out.fail()) << "Failed to open " << file << " for writing.";
  out << content;
  out.close();
}

vector<uint> VectorSliceToVector(VectorSlice<uint> slice) {
  vector<uint> vec;
  vec.reserve(slice.size());
  for (auto i : slice) {
    vec.push_back(i);
  }
  return vec;
}

bool FileExists(const string& file) {
  return ifstream(file).good();
}

namespace strings {

void SplitStringUsing(const string& s,
                      const string& delim,
                      vector<string>* words) {
  words->clear();
  if (s.size() == 0)
    return;
  int pos = 0;
  uint start_pos = 0;
  do {
    pos = s.find(delim, start_pos);
    if (pos >= 0) {
      words->push_back(s.substr(start_pos, pos - start_pos));
      start_pos = pos + delim.size();
    }
  } while(pos >= 0);
  if (start_pos <= s.size())
    words->push_back(s.substr(start_pos));
}

vector<string> split(const string& s, const string& delim) {
  vector<string> words;
  SplitStringUsing(s, delim, &words);
  return words;
}

bool StringCast(const string& s, float* v) {
    char* pEnd;
    *v = strtof(s.c_str(), &pEnd);
    return *pEnd == '\0' && s.size() != 0;
}

bool HasPrefix(const string& s, const string& prefix) {
  return std::mismatch(prefix.begin(), prefix.end(), s.begin()).first == prefix.end();
}

bool HasSuffix(const string& s, const string& suffix) {
  return std::mismatch(suffix.rbegin(), suffix.rend(), s.rbegin()).first == suffix.rend();
}

}  // namespace strings

string ReadLine(istream& is) {
  string line;
  std::getline(is, line);
  line.erase(line.find_last_not_of("\n\r") + 1);
  return line;
}
