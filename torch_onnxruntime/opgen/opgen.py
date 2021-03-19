#!/usr/bin/env python3

from opgen.generator import \
  ORTGen as ORTGen, \
  ONNXOp as ONNXOp, \
  SignatureOnly as SignatureOnly

class Add(ONNXOp):
  def __init__(self, a, b): super().__init__('Add', 1, a, b)

class Sub(ONNXOp):
  def __init__(self, a, b): super().__init__('Sub', 1, a, b)

class Mul(ONNXOp):
  def __init__(self, a, b): super().__init__('Mul', 1, a, b)

class Gemm(ONNXOp):
  def __init__(self, a, b, c, Alpha=None, Beta=None, TransA=None, TransB=None):
    super().__init__('Gemm', 1, a, b, c, Alpha=Alpha, Beta=Beta, TransA=TransA, TransB=TransB)

class Transpose(ONNXOp):
  def __init__(self, data, perms=None): super().__init__('Transpose', 1, data, perms=perms)

class Relu(ONNXOp):
  def __init__(self, x): super().__init__('Relu', 1, x)

class MatMul(ONNXOp):
  def __init__(self, a, b): super().__init__('MatMul', 1, a, b)

ortgen = ORTGen({
  'aten::empty.memory_format': SignatureOnly(),
  'aten::empty_strided': SignatureOnly(),
  'aten::copy_': SignatureOnly(),
  'aten::reshape': SignatureOnly(),
  'aten::view': SignatureOnly(),
  'aten::add.Tensor': Add('self', Mul('alpha', 'other')),
  'aten::sub.Tensor': Sub('self', Mul('alpha', 'other')),
  'aten::mul.Tensor': Mul('self', 'other'),
  'aten::addmm': Gemm('mat1', 'mat2', 'self', Alpha='alpha', Beta='beta'),
  'aten::t': Transpose('self'),
  'aten::relu': Relu('self'),
  'aten::mm': MatMul('self', 'mat2'),
  'aten::sum.dim_IntList': SignatureOnly(),
  'aten::threshold_backward': SignatureOnly(),
  'aten::zeros_like': SignatureOnly(),
  'aten::add_.Tensor': SignatureOnly(),
  'aten::zero_': SignatureOnly()
}, function_name_prefix='ort_op_aten_')

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