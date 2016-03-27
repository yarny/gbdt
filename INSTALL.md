# Installation Instructions

The package can be compiled in both Linux and OSX platforms.
It depends on * [bazel](bazel.io), gflags, glogs, gperf, protobuf3*.
We include a convenient script to set up the dependencies. The binary
built by this package is static linked. Therefore, the external dependencies
are built in static model.

1. Install [`bazel`](bazel.io)
2. Run `setup_dependencies/setup.sh`.
3. Add the following to .bazelrc
```sh
build -copt -I"/usr/local/include"
build -linkopt -L"/usr/local/lib"
```
4. Run `bazel build -c opt src:gbdt` to build the binary.
