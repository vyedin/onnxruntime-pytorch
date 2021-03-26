#include "ort_optimizers.h"
#include <onnx/defs/attr_proto_util.h>

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
    Tensor& exp_avg_sq) {
    
    if (!self.grad().defined()) {
      return self;
    }

    auto& invoker = GetORTInvoker(self.device());
    auto ort_in_eta = create_ort_value(invoker, lr); 
    auto ort_in_self = create_ort_value(invoker, self);
    auto ort_in_grad = create_ort_value(invoker, self.grad());
    auto ort_in_exp_avg = create_ort_value(invoker, exp_avg);
    auto ort_in_exp_avg_sq = create_ort_value(invoker, exp_avg_sq);
    auto ort_in_step = create_ort_value(invoker, step);

    std::vector<OrtValue> ort_out;
    ort_out.push_back(ort_in_step);
    ort_out.push_back(ort_in_exp_avg);
    ort_out.push_back(ort_in_exp_avg_sq);
    ort_out.push_back(ort_in_self);

    onnxruntime::NodeAttributes attributes;
    attributes["alpha"] = ::ONNX_NAMESPACE::MakeAttribute("alpha", alpha);
    attributes["beta"] = ::ONNX_NAMESPACE::MakeAttribute("beta", beta);
    attributes["lambda"] = ::ONNX_NAMESPACE::MakeAttribute("lambda", lambda);
    attributes["epsilon"] = ::ONNX_NAMESPACE::MakeAttribute("epsilon", eps);
    attributes["do_bias_correction"] = ::ONNX_NAMESPACE::MakeAttribute("do_bias_correction", do_bias_correction);
    attributes["weight_decay_mode"] = ::ONNX_NAMESPACE::MakeAttribute("weight_decay_mode", weight_decay_mode);

    auto status = invoker.Invoke(
      "AdamOptimizer", {
        ort_in_eta,
        ort_in_step,
        ort_in_self, 
        ort_in_grad,
        ort_in_exp_avg,
        ort_in_exp_avg_sq
      }, ort_out, &attributes, onnxruntime::kMSDomain);
    
    if (!status.IsOK())
      throw std::runtime_error(
        "ORT return failure status:" + status.ErrorMessage());
    
    return self;
  }
}
}
}