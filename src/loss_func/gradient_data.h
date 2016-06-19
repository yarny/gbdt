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

#ifndef GRADIENT_DATA_H_
#define GRADIENT_DATA_H_

namespace gbdt {

struct GradientData {
  GradientData() {}
  GradientData(double g_arg, double h_arg) : g(g_arg), h(h_arg) {}
  inline GradientData operator - (const GradientData& data) const {
    GradientData res;
    res.g = g - data.g;
    res.h = h - data.h;
    return res;
  }
  inline GradientData operator + (const GradientData& data) const {
    GradientData res;
    res.g = g + data.g;
    res.h = h + data.h;
    return res;
  }
  inline void operator += (const GradientData& data) {
    g += data.g;
    h += data.h;
  }
  inline GradientData operator * (float w) const {
    return GradientData(w * g, w * h);
  }

  // Energy = g^2 / (h + lambda), where lambda is the L2 regularization factor.
  inline double Energy(double lambda) const {
    double h_smoothed = h + lambda;
    return h_smoothed == 0 ? 0 : g * g / h_smoothed;
  }

  // Score = g / (h + lambda), where lambda is the L2 regularization factor.
  inline double Score(double lambda) const {
    double h_smoothed = h + lambda;
    return h_smoothed == 0 ? 0 : g / h_smoothed;
  }

  double g = 0.0;
  double h = 0.0;
};

inline GradientData operator * (float w, const GradientData& data) {
  return GradientData(w * data.g, w * data.h);
}

// Contains GradientData and the loss.
struct LossFuncData {
  LossFuncData() {}
  LossFuncData(double loss_arg, double g_arg, double h_arg) : loss(loss_arg), gradient_data(g_arg, h_arg) {}

  double loss = 0;
  GradientData gradient_data;
};

}  // namespace gbdt

#endif  // GRADIENT_DATA_H_
