#ifndef LOSS_FUNC_PAIRWISE_LOGLOSS_H_
#define LOSS_FUNC_PAIRWISE_LOGLOSS_H_

#include <functional>

#include "loss_func_math.h"
#include "loss_func_pairwise.h"

namespace gbdt {

class LossFuncConfig;

// PairwiseLogloss: \sum_(\forall pairs) log(1+exp(fn - fp)).
class PairwiseLogLoss : public Pairwise {
 public:
  PairwiseLogLoss(const LossFuncConfig& config)
      : Pairwise(config,
                 [](double delta_target,
                    double delta_func) {
                   return ComputeLogLoss(1, delta_func); }) {}
};

}  // namespace

#endif  // LOSS_FUNC_PAIRWISE_LOGLOSS_H_
