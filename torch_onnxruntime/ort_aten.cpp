// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_aten.h"
#include "ort_tensor.h"

namespace torch_ort::eager {

#pragma region Helpers

namespace {
  inline bool is_device_supported(at::DeviceType type) {
    return type == at::kORT || type == at::kCPU;
  }

  inline void assert_tensor_supported(const at::Tensor& tensor) {
    if (tensor.is_sparse()) {
      throw std::runtime_error("ORT copy: sparse not supported");
    }

    if (tensor.is_quantized()) {
      throw std::runtime_error("ORT copy: quantized not supported");
    }

    if (!is_device_supported(tensor.device().type())) {
      throw std::runtime_error("ORT copy: device not supported");
    }
  }
}

const at::Tensor get_at_tensor_from_ort_tensor(
  OrtValue&& ot,
  const at::TensorOptions& options) {
  return at::Tensor(c10::make_intrusive<ORTTensorImpl>(
    std::move(ot),
    options));
}

const OrtValue get_ort_tensor_from_at_tensor(const at::Tensor& tensor) {
  assert_tensor_supported(tensor);
  auto* impl = dynamic_cast<ORTTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl) {
    return impl->tensor();
  }
  OrtValue ort_tensor;
  CreateMLValue(
    tensor.data_ptr(),
    get_ort_scalar_type_from_aten(tensor.scalar_type()),
    tensor.sizes().vec(),
    &ort_tensor);
  return ort_tensor;
}

const OrtValue get_ort_tensor_from_at_scalar(const at::Scalar& scalar, 
                                             onnxruntime::ORTInvoker& invoker) {
  //TODO: support more types
  float val = scalar.toFloat();
  OrtValue ort_val;
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    get_ort_scalar_type_from_aten(at::kFloat),
    {},
    &ort_val);
  //TODO: use EP's data transfer to copy the data into that tensor
  auto* ort_tensor = ort_val.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<float>({val}, *ort_tensor);
  return ort_val;
}

const onnxruntime::MLDataType get_ort_scalar_type_from_aten(
  at::ScalarType dtype) {
  switch (dtype){
    case at::kFloat:
      return onnxruntime::DataTypeImpl::GetType<float>();
    case at::kDouble:
      return onnxruntime::DataTypeImpl::GetType<double>();
    case at::kHalf:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
    case at::kBFloat16:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>();
    case at::kInt:
      return onnxruntime::DataTypeImpl::GetType<int>();
    case at::kShort:
      return onnxruntime::DataTypeImpl::GetType<int16_t>();
    case at::kLong:
      return onnxruntime::DataTypeImpl::GetType<int64_t>();
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

#pragma endregion

#pragma region Hand-Implemented ATen Ops

at::Tensor aten_empty_memory_format(
  at::IntArrayRef size,
  // *
  const at::TensorOptions& options, 
  c10::optional<at::MemoryFormat> memory_format) {
  ORT_LOG_FN(size, options, memory_format);

  // TODO: validate options and memory format
  // TODO: figure out how to get the correct element type.
  OrtValue ot;
  auto& invoker = GetORTInvoker(options.device());
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    get_ort_scalar_type_from_aten(at::kFloat),
    size.vec(),
    &ot);

  return get_at_tensor_from_ort_tensor(
    std::move(ot),
    options);
}

at::Tensor aten_empty_strided(
  at::IntArrayRef size,
  at::IntArrayRef stride,
  // *
  c10::optional<at::ScalarType> dtype_opt,
  c10::optional<at::Layout> layout_opt,
  c10::optional<at::Device> device_opt,
  c10::optional<bool> pin_memory_opt) {
  ORT_LOG_FN(stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // TODO: handle stride
  // TODO: how to handle type conversion
  OrtValue ot;
  assert(device_opt.has_value());
  // TODO: how to support layout
  assert(!layout_opt.has_value());
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);
  auto& invoker = GetORTInvoker(*device_opt);
  CreateMLValue(
    invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
    get_ort_scalar_type_from_aten(dtype),
    size.vec(),
    &ot);
  return get_at_tensor_from_ort_tensor(
    std::move(ot),
    at::device(*device_opt).dtype(dtype));
}

at::Tensor aten_reshape(at::Tensor const& self, at::IntArrayRef shape) {
  ORT_LOG_FN(self, shape);

  return get_at_tensor_from_ort_tensor(
    reshape_copy(
      GetORTInvoker(self.device()),
      get_ort_tensor_from_at_tensor(self),
      shape.vec()),
    self.options());
}

at::Tensor aten_view(const at::Tensor& self, at::IntArrayRef size) {
  ORT_LOG_FN(self, size);

  return get_at_tensor_from_ort_tensor(
    reshape_copy(
      GetORTInvoker(self.device()),
      get_ort_tensor_from_at_tensor(self),
      at::infer_size(
        size,
        self.numel())),
    self.options());
}

at::Tensor& aten_copy_(
  at::Tensor& self,
  const at::Tensor& src,
  bool non_blocking) {
  ORT_LOG_FN(self, src, non_blocking);

  assert_tensor_supported(self);
  assert_tensor_supported(src);

  auto& invoker = GetORTInvoker(self.device().type() == at::kORT
    ? self.device()
    : src.device());
  const auto ort_src = get_ort_tensor_from_at_tensor(src);
  auto ort_self = get_ort_tensor_from_at_tensor(self);

  copy(invoker, ort_src, ort_self);

  return self;
}

at::Tensor aten_threshold_backward(
    const at::Tensor& grad_output, 
    const at::Tensor& self, 
    at::Scalar threshold){
  ORT_LOG_FN(grad_output, self, threshold);
  
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_in_dY = get_ort_tensor_from_at_tensor(grad_output);
  auto ort_in_self = get_ort_tensor_from_at_tensor(self);
  std::vector<OrtValue> ort_out(1);
  
  auto status = invoker.Invoke(
    "ReluGrad", {
      std::move(ort_in_dY), 
      std::move(ort_in_self)
    }, ort_out, nullptr, onnxruntime::kMSDomain);
  
  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status:" + status.ErrorMessage());
  
  OrtValue ort_result = ort_out[0];
  return get_at_tensor_from_ort_tensor(
    std::move(ort_result),
    self.options());
}

at::Tensor aten_zeros_like(
  const at::Tensor& self, 
  // *, 
  c10::optional<at::ScalarType> dtype, 
  c10::optional<at::Layout> layout, 
  c10::optional<at::Device> device, 
  c10::optional<bool> pin_memory, 
  c10::optional<at::MemoryFormat> memory_format){
  
  auto& invoker = GetORTInvoker(self.device());

  auto ort_in_self = get_ort_tensor_from_at_tensor(self);
  auto& ort_in_self_tensor = ort_in_self.Get<onnxruntime::Tensor>();    
  auto& shape = ort_in_self_tensor.Shape();

  OrtValue output;
  //todo: avoid the copy on this small shape vector;
  auto element_type = ort_in_self_tensor.DataType();
  CreateMLValue(invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                element_type, shape.GetDims(), &output);
  auto* output_tensor = output.GetMutable<onnxruntime::Tensor>();
  memset(output_tensor->MutableDataRaw(element_type), 0, element_type->Size() * shape.Size());
  return get_at_tensor_from_ort_tensor(
    std::move(output),
    self.options());
}

at::Tensor aten_sum_dim_IntList(
  const at::Tensor& self, 
  at::IntArrayRef dim, 
  bool keepdim, 
  // *, 
  c10::optional<at::ScalarType> dtype) {
  ORT_LOG_FN(self, dim, keepdim, dtype);
  
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_in_self = get_ort_tensor_from_at_tensor(self);
  OrtValue dim_ort_value;
  std::vector<int64_t> dim_vector;
  dim_vector.assign(dim.begin(), dim.end());
  //todo: avoid the copy on this small vector;
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  CreateMLValue(invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                element_type, {(int64_t)dim.size(),}, &dim_ort_value);
  auto* ort_dim_tensor = dim_ort_value.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<int64_t>(dim_vector, *ort_dim_tensor);
  
  std::vector<OrtValue> ort_out(1);
  
  auto status = invoker.Invoke(
    "ReduceSum", {
      std::move(ort_in_self), 
      std::move(dim_ort_value)
    }, ort_out, nullptr);
  
  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status:" + status.ErrorMessage());
  
  OrtValue ort_result = ort_out[0];
  return get_at_tensor_from_ort_tensor(
    std::move(ort_result),
    self.options());
}

at::Tensor& aten_zero_(at::Tensor& self){
  auto& invoker = GetORTInvoker(self.device());
  auto ort_in_self = get_ort_tensor_from_at_tensor(self);
  OrtValue flag_val;
  //construct a constant tensor
  auto element_type = onnxruntime::DataTypeImpl::GetType<int64_t>();
  CreateMLValue(invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault),
                element_type, {}, &flag_val);
  auto* ort_flag_tensor = flag_val.GetMutable<onnxruntime::Tensor>();
  CopyVectorToTensor<int64_t>({1}, *ort_flag_tensor);

  std::vector<OrtValue> ort_out;
  ort_out.push_back(ort_in_self);

  auto status = invoker.Invoke(
    "ZeroGradient", {
      std::move(ort_in_self), 
      std::move(flag_val)
    }, ort_out, nullptr, onnxruntime::kMSDomain);

  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status:" + status.ErrorMessage());
  
  return self;
}

at::Tensor& aten_add__Tensor(
    at::Tensor& self, 
    const at::Tensor& other, 
    // *, 
    at::Scalar alpha) {

  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_in_self = get_ort_tensor_from_at_tensor(self);
  auto ort_in_other = get_ort_tensor_from_at_tensor(other);
  auto ort_alpha = get_ort_tensor_from_at_scalar(alpha, invoker);

  std::vector<OrtValue> ort_tmp(1);

  auto status = invoker.Invoke(
    "Mul", {
      std::move(ort_in_other), 
      std::move(ort_alpha)
    }, ort_tmp, nullptr);

  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status: Mul: " + status.ErrorMessage());

  std::vector<OrtValue> ort_out;
  ort_out.push_back(ort_in_self);

  status = invoker.Invoke(
    "Add", {
      std::move(ort_in_self), 
      std::move(ort_tmp[0])
    }, ort_out, nullptr);

  if (!status.IsOK())
    throw std::runtime_error(
      "ORT return failure status: Add" + status.ErrorMessage());

  return self;
}

#pragma endregion

} // namespace torch_ort::eager