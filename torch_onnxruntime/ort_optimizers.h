#pragma once

#include "ort_tensor.h"
#include "ort_aten.h"
#include "ort_log.h"

namespace torch_ort {
namespace eager {
namespace optimizers {

  using namespace at;

  std::vector<Tensor>& ort_SGD(
    std::vector<Tensor>& parameters,
    const float lr
  );

  Tensor& ort_Adam(
    const float lr,
    const float alpha,
    const float beta,
    const float lambda,
    const float eps,
    const int64_t weight_decay_mode,
    const int64_t do_bias_correction,
    Tensor& self,
    Tensor& step,
    Tensor& exp_avg,
    Tensor& exp_avg_sq);
}
}
}