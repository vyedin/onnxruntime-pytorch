#!/bin/bash

time (
	USE_ORT=1 \
	USE_CUDA=0 \
	USE_MKLDNN=0 \
	USE_OPENMP=0 \
	MACOSX_DEPLOYMENT_TARGET=10.9 \
	CC=clang \
	CXX=clang++ \
	python setup.py install
)

