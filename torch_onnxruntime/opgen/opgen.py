#!/usr/bin/env python3
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from copy import deepcopy

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  SignatureOnly as SignatureOnly, \
  MakeFallthrough as MakeFallthrough

from opgen.onnxops import *

kMSDomain = 'onnxruntime::kMSDomain'

class ReluGrad(ONNXOp):
  def __init__(self, dY, X):
    super().__init__('ReluGrad', 1, dY, X)
    self.domain = kMSDomain

ops = {
  # Hand-Implemented Ops
  'aten::empty.memory_format': SignatureOnly(),
  'aten::empty_strided': SignatureOnly(),
  'aten::zeros_like': SignatureOnly(),
  'aten::zero_': SignatureOnly(),
  'aten::copy_': SignatureOnly(),
  'aten::reshape': SignatureOnly(),
  'aten::view': SignatureOnly(),

  'aten::addmm': Gemm('mat1', 'mat2', 'self', alpha='alpha', beta='beta'),
  'aten::t': Transpose('self'),
  'aten::mm': MatMul('self', 'mat2'),

  'aten::sum.dim_IntList': ReduceSum('self', 'dim', keepdims='keepdim'),
  'aten::threshold_backward': ReluGrad('grad_output', 'self'),

  'aten::fmod.Scalar': Mod('self', 'other', fmod=1),
  'aten::fmod.Tensor': Mod('self', 'other', fmod=1),

  'aten::softshrink': Shrink('self', bias='lambd', lambd='lambd'), #yes, bias is set to 'lambd'
  'aten::hardshrink': Shrink('self', bias=0, lambd='lambd'),

  # 'aten::argmax': ArgMax('self', axis='dim', keepdims='keepdim', select_last_index=0),
  # 'aten::argmin': ArgMin('self', axis='dim', keepdims='keepdim', select_last_index=0)
  # onnxruntime-pytorch/torch_onnxruntime/ort_aten.h:60:28: note: candidate function not viable: no known conversion from 'c10::optional<int64_t>' (aka 'optional<long long>') to 'at::Scalar' for 2nd argument
  
  # 'aten::__rshift__.Scalar': BitShift('self', 'other', direction='"RIGHT"'),
  # 'aten::__rshift__.Tensor': BitShift('self', 'other', direction='"RIGHT"')
  # 'aten::__lshift__.Scalar': BitShift('self', 'other', direction='"LEFT"'),
  # 'aten::__lshift__.Tensor': BitShift('self', 'other', direction='"LEFT"')

  # 'aten::clip': Clip('self', 'min', 'max'),
  # 'aten::clamp': Clip('self', 'min', 'max'),
  # torch_onnxruntime/ort_aten.h:23:16: note: candidate function not viable: no known conversion from 'c10::optional<Scalar>' to 'const at::Scalar' for 2nd argument
 
  # 'aten::cat': ConcatFromSequence('tensors', axis='dim') #not sure if Concat or ConcatFromSequence here
  # onnxruntime-pytorch/torch_onnxruntime/ort_aten.g.cpp:519:41: error: no member named 'device' in 'c10::ArrayRef<at::Tensor>'

  'aten::convolution': Conv('input', 'weight', 'bias', dilations='dilation', group='groups', pads='padding', strides='stride'),
 }


for binary_op, onnx_op in {
  'add': Add('self', Mul('alpha', 'other')),
  'sub': Sub('self', Mul('alpha', 'other')),
  'mul': Mul('self', 'other'),
  'div': Div('self', 'other')}.items():
  for dtype in ['Tensor', 'Scalar']:
    for variant in ['', '_']:
      ops[f'aten::{binary_op}{variant}.{dtype}'] = deepcopy(onnx_op)

for unary_op in [
  'abs','acos','acosh', 'asinh', 'atanh', 'asin', 'atan', 'ceil', 'cos',
  'cosh', 'erf', 'exp', 'floor', 'isnan', 'log', 'reciprocal', 'neg', 'round',
  'relu', 'selu', 'sigmoid', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'nonzero',
  'sign', 'min', 'max', 'hardsigmoid', 'isinf', 'det']:
  aten_name = f'aten::{unary_op}'
  onnx_op = onnx_ops[unary_op]('self')
  ops[aten_name] = onnx_op
  # produce the in-place variant as well for ops that support it
  if unary_op not in ['isnan', 'nonzero', 'min', 'max', 'isinf', 'det']:
    ops[f'{aten_name}_'] = onnx_op

ortgen = ORTGen(ops)

import os
import sys

from opgen.parser import cpp_create_from_file as CPPParser
from opgen.writer import SourceWriter as SourceWriter

regdecs_path = os.path.realpath(os.path.join(
  os.path.dirname(__file__),
  '..',
  '..',
  'build',
  'aten',
  'src',
  'ATen',
  'RegistrationDeclarations.h'))
print(regdecs_path)
output = sys.stdout
if len(sys.argv) >= 2:
  output = open(sys.argv[1], 'wt')

with CPPParser(regdecs_path) as parser, SourceWriter(output) as writer:
  ortgen.run(parser, writer)