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

new_http_archive(
  name = "gmock_archive",
  url = "https://googlemock.googlecode.com/files/gmock-1.7.0.zip",
  sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
  build_file = "gmock.BUILD",
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

bind(
  name = "gtest",
  actual = "@gmock_archive//:gtest",
)

bind(
  name = "gtest_main",
  actual = "@gmock_archive//:gtest_main",
)