bind(
    name = "cppformat-lib",
    actual = "@cppformat//:format"
)

new_git_repository(
    name = "cppformat",
    remote = "https://github.com/cppformat/cppformat.git",
    tag = "1.1.0",
    build_file = "cppformat.BUILD",
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
  build_file = "googletest.BUILD",
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
    build_file = "pybind11.BUILD",
)