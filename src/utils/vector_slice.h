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

#ifndef VECTOR_SLICE_H_
#define VECTOR_SLICE_H_

#include <vector>

using std::vector;

// VectorSlice is designed to handle slices of a vector. It doesn't own memory.
template <typename T>
class VectorSlice {
public:
  VectorSlice(typename vector<T>::iterator begin,
              typename vector<T>::iterator end) : begin_(begin), end_(end) {}
  VectorSlice(vector<T>& vec, int start, int length)
    : begin_(vec.begin() + start), end_(vec.begin() + start + length) {}
  VectorSlice(VectorSlice<T>& vec, int start, int length)
    : begin_(vec.begin() + start), end_(vec.begin() + start + length) {}
  VectorSlice(vector<T>& vec) : begin_(vec.begin()), end_(vec.end()) {}

  inline typename vector<T>::iterator begin() {
    return begin_;
  }
  inline typename vector<T>::iterator end() {
    return end_;
  }
  inline typename vector<T>::const_iterator begin() const {
    return begin_;
  }
  inline typename vector<T>::const_iterator end() const {
    return end_;
  }
  inline T& operator [](int i) {
    return *(begin_ + i);
  }
  inline const T& operator [](int i) const {
    return *(begin_ + i);
  }
  inline int size() const {
    return end_ - begin_;
  }

private:
  typename vector<T>::iterator begin_;
  typename vector<T>::iterator end_;
};

#endif  // VECTOR_SLICE_H_
