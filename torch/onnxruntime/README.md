# ONNX Runtime Eager Mode Support

## Dependencies & Environment

```bash
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasse pkg-config libuv
```

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```

## Build

Run the following script (macOS) from the root of the repository:

```bash
./ort_eager_build.sh
```