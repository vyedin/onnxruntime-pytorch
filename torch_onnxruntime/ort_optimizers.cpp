#include "ort_optimizers.h"

namespace torch_ort {
namespace eager {
namespace optimizers {

  using namespace at;

  static Tensor& ort_SGD(
    Tensor& self,
    const Tensor& eta) {
    if (!self.grad().defined()) {
      return self;
    }

    auto& invoker = GetORTInvoker(self.device());
    auto ort_in_eta = create_ort_value(invoker, eta); 
    auto ort_in_self = create_ort_value(invoker, self);
    auto ort_in_grad = create_ort_value(invoker, self.grad());
    
    std::vector<OrtValue> ort_out {ort_in_self};
    
    auto status = invoker.Invoke(
      "SGDOptimizer", {
        ort_in_eta,
        ort_in_self, 
        ort_in_grad
      }, ort_out, nullptr, onnxruntime::kMSDomain);
    
    if (!status.IsOK())
      throw std::runtime_error(
        "ORT return failure status:" + status.ErrorMessage());
    
    return self;    
  }

  std::vector<Tensor>& ort_SGD(
    std::vector<Tensor>& parameters,
    const float lr)  {

    const auto eta = torch::full({1}, lr);
    for (auto& param : parameters) 
    {
      ort_SGD(param, eta);
    }

    return parameters;
  }
}
}
}