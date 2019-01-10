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

new_git_repository(
  name = "googletest",
  remote = "https://github.com/google/googletest.git",
  tag = "release-1.8.0",
  build_file = "//third_party:googletest.BUILD",
)

bind(
  name = "gtest_main",
  actual = "@googletest//:gtest_main",
)

bind(
    name = "pybind11-lib",
    actual = "@pybind11//:pybind11"
)

new_git_repository(
    name = "pybind11",
    remote = "https://github.com/pybind/pybind11.git",
    tag = "v1.8.1",
    build_file = "//third_party:pybind11.BUILD",
)

load("//third_party:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)