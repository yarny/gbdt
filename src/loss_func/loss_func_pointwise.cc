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

#include "loss_func_pointwise.h"

#include <functional>
#include <numeric>
#include <vector>

#include "external/cppformat/format.h"

#include "src/base/base.h"
#include "src/utils/subsampling.h"
#include "src/utils/threadpool.h"

DECLARE_int32(num_threads);

namespace gbdt {

// TODO(criver): reiterate on the selection of the constants here.
const double kConvergenceThreshold = 1e-4;
const int kMaxIterations = 10;

Pointwise::Pointwise(PointwiseLossFunc loss_func) : loss_func_(loss_func) {
}

Status Pointwise::Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* unused_group_column) {
  w_ = w;
  y_ = y;
  weight_sum_ = 0;
  for (int i = 0; i < num_rows; ++i) {
    weight_sum_ += w(i);
  }
  slices_ = Subsampling::DivideSamples(num_rows, FLAGS_num_threads * 5);
  return Status::OK;
}

void Pointwise::ComputeFunctionalGradientsAndHessians(const vector<double>& f,
                                                      double* c,
                                                      vector<GradientData>* gradient_data_vec,
                                                      string* progress) {
  // Resize g and h if they haven't be resized yet.
  if (gradient_data_vec->size() != f.size()) {
    gradient_data_vec->resize(f.size());
  }

  int k = 0;
  *c = 0;
  double delta_c = 0;
  LossFuncData total;
  do {
    vector<LossFuncData> totals(slices_.size());
    {
      ThreadPool pool(FLAGS_num_threads);
      for (int j = 0; j < slices_.size(); ++j) {
          pool.Enqueue([&, this, &slice=slices_[j], &total=totals[j]](){
            for (int i = slice.first; i < slice.second; ++i) {
              double f_current = f[i] + *c;
              auto data = loss_func_(y_(i), f_current);
              auto w = w_(i);
              auto& gradient_data = (*gradient_data_vec)[i];
              gradient_data.g = std::get<1>(data);
              gradient_data.h = std::get<2>(data);
              total.loss += w * std::get<0>(data);
              total.gradient_data += w * gradient_data;
            }
          });
      }
    }
    total = std::accumulate(totals.begin(), totals.end(), LossFuncData());
    delta_c = total.gradient_data.Score(0);
    *c += delta_c;
    ++k;
  } while (fabs(delta_c) > kConvergenceThreshold && k < kMaxIterations);

  if (progress) {
    *progress = PrepareProgressMessage(total.loss / weight_sum_);
  }
}

string Pointwise::PrepareProgressMessage(double loss) {
  // Prepare for the progress message.
  if (initial_loss_ < 0) {
    initial_loss_ = loss;
  }

  double relative_reduction = initial_loss_ == 0.0 ? 0.0 : (initial_loss_ - loss) / initial_loss_;
  return fmt::format("loss={0},reduced={1:.2f}%", loss, relative_reduction * 100.0);
}

}  // namespace gbdt
