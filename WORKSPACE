load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/google/protobuf.git",
    tag = "v3.6.1",
)

bind(
    name = "cppformat-lib",
    actual = "@cppformat//:format"
)

new_git_repository(
    name = "cppformat",
    remote = "https://github.com/cppformat/cppformat.git",
    tag = "1.1.0",
    build_file = "//third_party:cppformat.BUILD",
)

bind(
    name = "re2-lib",
    actual = "@re2//:re2"
)

git_repository(
    name = "re2",
    remote = "https://github.com/google/re2.git",
    tag = "2015-11-01",
)

git_repository(
    name = "com_google_googletest",
    commit = "d850e144710e330070b756c009749dc7a7302301",
    remote = "https://github.com/google/googletest.git",
)

bind(
  name = "gtest_main",
  actual = "@com_google_googletest//:gtest_main",
)

bind(
    name = "pybind11-lib",
    actual = "@pybind11//:pybind11"
)

git_repository(
    name = "com_github_glog_glog",
    remote = "https://github.com/google/glog.git",
    commit = "5d46e1bcfc92bf06a9ca3b3f1c5bb1dc024d9ecd"
)

bind(
    name = "glog",
    actual = "@com_github_glog_glog//:glog",
)

git_repository(
    name = "com_github_gflags_gflags",
    remote = "https://github.com/gflags/gflags.git",
    commit = "1005485222e8b0feff822c5723ddcaa5abadc01a"
)

bind(
    name = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)

bind(
    name = "protobuf",
    actual = "@com_google_protobuf//:protobuf"
)

new_git_repository(
    name = "pybind11",
    remote = "https://github.com/pybind/pybind11.git",
    tag = "v1.8.1",
    build_file = "//third_party:pybind11.BUILD",
)

load("//third_party/py:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)