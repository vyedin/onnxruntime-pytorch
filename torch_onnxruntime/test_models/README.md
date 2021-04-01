# State of the Ops

## dropout
_Tensor dropout(const Tensor & input, double p, bool train);_
[Torch](https://pytorch.org/docs/stable/nn.functional.html#dropout-functions) | [ORT](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout)

Dropout is mappable between torch and ORT, but it depends on aten::bernoulli_.float, which is not easily mappable between torch and ORT. This is catalogued in [PR 23](https://github.com/microsoft/onnxruntime-pytorch/pull/23).

### dropout variants
_Tensor feature_dropout(const Tensor & input, double p, bool train);_
_Tensor alpha_dropout(const Tensor & input, double p, bool train);_
_Tensor feature_alpha_dropout(const Tensor & input, double p, bool train);_

These do not have ORT equivalents (and sadly are different operations, so adjusting the inputs won't help in this case)


