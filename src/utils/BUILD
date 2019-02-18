package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [
    "-lprotobuf",
    "-lglog",
    "-lgflags",
    "-pthread",
    "-ltcmalloc_minimal",
    "-lm",
]

C_TEST_OPTS = [
    "-Iexternal/googletest/googletest/include",
    "-Iexternal/googletest/googlemock/include",
]

cc_library(
    name = "stopwatch",
    srcs = ["stopwatch.cc"],
    hdrs = ["stopwatch.h"],
    deps = [
        ":utils",
        "//external:cppformat-lib",
    ],
)

cc_test(
    name = "stopwatch_test",
    srcs = ["stopwatch_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    deps = [
        ":stopwatch",
        "//external:gtest_main",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "utils.h",
        "utils_inl.h",
    ],
    deps = [
        ":vector_slice",
        "//external:glog",
        "//external:protobuf",
        "//src/base",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    deps = [
        ":utils",
        "//external:gtest_main",
    ],
)

cc_library(
    name = "vector_slice",
    hdrs = ["vector_slice.h"],
)

cc_test(
    name = "vector_slice_test",
    srcs = ["vector_slice_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    deps = [
        ":vector_slice",
        "//external:gtest_main",
    ],
)

cc_library(
    name = "threadpool",
    srcs = ["threadpool.cc"],
    hdrs = ["threadpool.h"],
)

cc_library(
    name = "json_utils",
    srcs = ["json_utils.cc"],
    hdrs = ["json_utils.h"],
    deps = [
        "//external:protobuf",
        "//src/base",
    ],
)

cc_test(
    name = "json_utils_test",
    srcs = ["json_utils_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":json_utils",
        "//external:gtest_main",
        "//src/proto:tree_cc_proto",
    ],
)

cc_library(
    name = "subsampling",
    srcs = ["subsampling.cc"],
    hdrs = ["subsampling.h"],
    deps = [
        ":utils",
    ],
)

cc_test(
    name = "subsampling_test",
    srcs = ["subsampling_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    deps = [
        ":subsampling",
        ":utils",
        "//external:gtest_main",
    ],
)
