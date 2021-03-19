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
}
}
}