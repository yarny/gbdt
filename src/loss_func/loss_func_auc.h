#ifndef LOSS_FUNC_AUC_H_
#define LOSS_FUNC_AUC_H_

#include <functional>

#include "loss_func_math.h"
#include "loss_func_pairwise.h"

namespace gbdt {

class LossFuncConfig;

// AUC: \sum_(\forall pairs) max(0, f_n + 1.0 - f_p). We huberize the hinge loss so we
// can get hessian out of it.
class AUC : public Pairwise {
 public:
  AUC(const LossFuncConfig& config)
      : Pairwise(config,
                 [] (double delta_target, double delta_func) {
                   return ComputeHuberizedHinge(1, delta_func); }) {}
};

}  // namespace

#endif  // LOSS_FUNC_PAIRWISE_LOGLOSS_H_
