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
    name = "column",
    srcs = ["column.cc"],
    hdrs = ["column.h"],
    deps = [
        "//external:cppformat-lib",
        "//src/base",
    ],
)

cc_test(
    name = "column_test",
    srcs = ["column_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":column",
        "//external:gtest_main",
        "//src/base",
    ],
)

cc_library(
    name = "data_store",
    srcs = ["data_store.cc"],
    hdrs = ["data_store.h"],
    deps = [
        ":column",
        "//external:cppformat-lib",
        "//src/base",
    ],
)

cc_test(
    name = "data_store_test",
    srcs = ["data_store_test.cc"],
    copts = C_TEST_OPTS,
    data = [":flatfiles_data_store_testdata"],
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":column",
        ":data_store",
        ":flatfiles_data_store",
        "//external:gtest_main",
    ],
)

cc_library(
    name = "flatfiles_data_store",
    srcs = ["flatfiles_data_store.cc"],
    hdrs = ["flatfiles_data_store.h"],
    deps = [
        ":data_store",
        "//src/base",
        "//src/utils",
    ],
)

cc_test(
    name = "flatfiles_data_store_test",
    srcs = ["flatfiles_data_store_test.cc"],
    copts = C_TEST_OPTS,
    data = [":flatfiles_data_store_testdata"],
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":column",
        ":flatfiles_data_store",
        "//external:gtest_main",
    ],
)

filegroup(
    name = "flatfiles_data_store_testdata",
    srcs = [
        "testdata/flatfiles_data_store_test",
    ],
)

cc_library(
    name = "tsv_block",
    srcs = ["tsv_block.cc"],
    hdrs = ["tsv_block.h"],
    deps = [
        "//external:cppformat-lib",
        "//src/base",
        "//src/utils",
        "//src/utils:stopwatch",
    ],
)

filegroup(
    name = "tsv_data_store_testdata",
    srcs = [
        "testdata/tsv_data_store_test",
    ],
)

cc_test(
    name = "tsv_block_test",
    srcs = ["tsv_block_test.cc"],
    copts = C_TEST_OPTS,
    data = [":tsv_data_store_testdata"],
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":tsv_block",
        "//external:gtest_main",
    ],
)

cc_library(
    name = "tsv_data_store",
    srcs = ["tsv_data_store.cc"],
    hdrs = ["tsv_data_store.h"],
    deps = [
        ":column",
        ":data_store",
        ":tsv_block",
        "//external:cppformat-lib",
        "//src:flags",
        "//src/base",
        "//src/proto:config_cc_proto",
        "//src/utils",
        "//src/utils:stopwatch",
        "//src/utils:threadpool",
    ],
)

cc_test(
    name = "tsv_data_store_test",
    srcs = ["tsv_data_store_test.cc"],
    copts = C_TEST_OPTS,
    data = [":tsv_data_store_testdata"],
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":column",
        ":tsv_data_store",
        "//external:gtest_main",
        "//src/proto:config_cc_proto",
    ],
)
