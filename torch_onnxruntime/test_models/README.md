# State of the Ops

This file documents the ONNX ops and tries to find their torch counterparts.

Note: ops marked with * contain c10::optional params.

## ArgMax and ArgMin*

[Torch](https://pytorch.org/docs/stable/generated/torch.argmax.html?highlight=argmax#torch.argmax) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax)

[Torch](https://pytorch.org/docs/stable/generated/torch.argmin.html?highlight=argmax#torch.argmin) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin)

`'aten::argmax': ArgMax('self', axis='dim', keepdims='keepdim', select_last_index=0),`

`'aten::argmin': ArgMin('self', axis='dim', keepdims='keepdim', select_last_index=0)`

If axis and dim are the same thing, which I think they are, this maps easily. There is a variant of torch.argmax that just takes an input tensor and nothing else, so we will need to account for that.

Mapping the above results in the following error:

`onnxruntime-pytorch/torch_onnxruntime/ort_aten.h:60:28: note: candidate function not viable: no known conversion from 'c10::optional<int64_t>' (aka 'optional<long long>') to 'at::Scalar' for 2nd argument`

## AveragePool

[Torch](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html?highlight=avgpool#torch.nn.AvgPool1d) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool)

Has 1D, 2D, and 3D variants on the torch side, they appear to all the the same on the ONNX side. Here is an example mapping:

`'aten::avg_pool1d': AveragePool('self', auto_pad='"NOTSET"', ceil_mode='ceil_mode', count_include_pad='count_include_pad', kernel_shape='kernel_size', pads='padding', strides='stride')`

Here is the resulting error:

`opgen.generator.FunctionGenerationError: Unsure how how to map ONNX op "AveragePool" attribute "kernel_shape" of type "<unsupported:INTS>" to a call to create_ort_attribute. Please teach generator.py. (torch: aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor)`

Note also that for whatever reason the documented torch op is `torch.nn.AvgPool...` but in RegistrationDeclarations.h it has an underscore (`avg_pool...`).

## BatchNormalization

[Torch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html?highlight=batchnorm#torch.nn.BatchNorm1d) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#batchnormalization)

This looks like it probably maps, but has multiple outputs. Torch also has 1D, 2D, and 3D variants that will need to be mapped, but it looks like ONNX op can handle this.

## BitShift

Torch | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#bitshift)

We got this mostly working mapped to `__rshift__` and `__lshift__` but it turns out ONNX wants unsigned integer tensors and pytorch does not have support for this, so will require additional work. Note these are not documented in torch docs but are in RegistrationDeclarations.h.

`'aten::__rshift__.Scalar': BitShift('self', 'other', direction='"RIGHT"'),`

`'aten::__rshift__.Tensor': BitShift('self', 'other', direction='"RIGHT"'),`

`'aten::__lshift__.Scalar': BitShift('self', 'other', direction='"LEFT"'),`

`'aten::__lshift__.Tensor': BitShift('self', 'other', direction='"LEFT"')`

## Cast

Torch | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#cast)

Closest match is https://pytorch.org/docs/stable/tensors.html?highlight=type#torch.Tensor.type but need more info

## Clip*

[Torch](https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#clip)

`'aten::clip': Clip('self', 'min', 'max'),`

`'aten::clamp': Clip('self', 'min', 'max')`

This maps to both Clip and Clamp on the torch side (torch.clip appears to be an alias for torch.clamp).

The maping is simple but does result in the following error:

`torch_onnxruntime/ort_aten.h:23:16: note: candidate function not viable: no known conversion from 'c10::optional<Scalar>' to 'const at::Scalar' for 2nd argument`

## Compress

Torch | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#compress)

This is equivalent to numpy.compress, and it does not look like there is a comparable function in torch except possibly using `torch.from_numpy`

## Concat and ConcatFromSequence

[Torch](https://pytorch.org/docs/stable/generated/torch.cat.html?highlight=cat#torch.cat) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#concat)

`'aten::cat': ConcatFromSequence('tensors', axis='dim')`

One of these maps to torch.cat, bit it’s unclear which one. RegistrationDeclarations.h says the first param is a TensorList, but the web documentation says it is a sequence of tensors. On the ONNX side, Concat takes a list while ConcatFromSequence (obviously) takes a sequence.

The mapping above results in the following error:

`onnxruntime-pytorch/torch_onnxruntime/ort_aten.g.cpp:519:41: error: no member named 'device' in 'c10::ArrayRef<at::Tensor>'`

## Constant and ConstantOfShape

Torch | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#constant)

I think this is just torch.tensor, nothing to map here.

## Conv, ConvInteger, and ConvTranspose

[Torch](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html?highlight=conv1d#torch.nn.Conv1d) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv)

Torch has 1D, 2D, and 3D variants of conv. I believe everything falls back on this torch op, which is not documented:

`Tensor convolution(const Tensor & input, const Tensor & weight, const c10::optional<Tensor>& bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups); // {"schema": "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor", "dispatch": "False", "default": "False"}`

Here is the same in ORT:

`'aten::convolution': Conv('input', 'weight', 'bias', auto_pad="'NOTSET'", dilations='dilation', group='groups', pads='padding', strides='stride')`

You are probably wondering - why is `auto_pad` a string that literally says the value is not set!? I do not know why, it just is.

## CumSum

[Torch](https://pytorch.org/docs/stable/generated/torch.cumsum.html?highlight=cumsum#torch.cumsum) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#cumsum)

`'aten::cumsum': CumSum('self', axis='dim', exclusive=0, reverse=0)`

This mapping actually compiles, the the following test fails:

`a = torch.tensor([[1, 2, 3]], dtype=torch.float).to(device)`

`b = torch.cumsum(a, dim=0) # should be [1, 3, 6]`

`print(b.cpu())`

The above code returns the following error:

`RuntimeError: ORT return failure status:This is an invalid model. Type Error: Type 'tensor(float)' of input parameter (I1) of operator (CumSum) in node (node1) is invalid.`

## DepthToSpace

[Torch](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pixel_shuffle) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#depthtospace)

`'aten::pixel_shuffle': DepthToSpace('self', blocksize='upscale_factor')`

Here's a good explainer of how these map: https://www.fatalerrors.org/a/0dh31Do.html

This one should work out of the box, but of course does not. The above mapping compiles, but the tests fail with a Pytorch RuntimeError:

`a = torch.randn(1, 9, 4, 4).to(device)`

`b = torch.nn.functional.pixel_shuffle(a, 3)`

`print(b.size().cpu()) #should return torch.Size([1, 1, 12, 12])`

`RuntimeError: The size of tensor a (3) must match the size of tensor b (48) at non-singleton dimension 5`

## DequantizeLinear

[Torch](https://pytorch.org/docs/stable/generated/torch.dequantize.html) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#dequantizelinear)

This is a Tensorflow function, but I can't find a direct match on the PyTorch side. Maybe a question for the ONNX folks.

## Dropout

[Torch](https://pytorch.org/docs/stable/nn.functional.html#dropout-functions) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout)

Dropout is mappable between torch and ORT, but it depends on aten::bernoulli_.float, which is not easily mappable between torch and ORT. This is catalogued in [PR 23](https://github.com/microsoft/onnxruntime-pytorch/pull/23).

### dropout variants

`Tensor feature_dropout(const Tensor & input, double p, bool train);`

`Tensor alpha_dropout(const Tensor & input, double p, bool train);`

`Tensor feature_alpha_dropout(const Tensor & input, double p, bool train);`

These do not have ORT equivalents (and sadly are different operations, so adjusting the inputs won't help in this case)

