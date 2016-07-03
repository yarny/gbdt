# Installation Instructions

The package can be compiled in both Linux and OSX platforms.
It depends on *[bazel](bazel.io), gflags, glogs, gperf, protobuf3*.
Per limitation of bazel, for linux, please use ubuntu 14 and above to build.
The built binary can be used in lower version linux machine.
We include a convenient script to set up the dependencies.

1. Install [bazel](bazel.io)
2. Run `setup_dependencies/setup.sh`.
3. Run `bazel build -c opt src:gbdt` to build the binary.
4. Find your binary at `bazel-bin/src/gbdt`.