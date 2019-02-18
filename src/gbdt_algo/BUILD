package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [
    "-lprotobuf",
    "-lgflags",
    "-lglog",
    "-pthread",
    "-ltcmalloc_minimal",
    "-lm",
]

C_TEST_OPTS = []

cc_library(
    name = "compute_tree_scores",
    srcs = ["compute_tree_scores.cc"],
    hdrs = ["compute_tree_scores.h"],
    deps = [
        ":split_algo",
        "//src:flags",
        "//src/base",
        "//src/data_store",
        "//src/data_store:column",
        "//src/proto:tree_cc_proto",
        "//src/utils:subsampling",
        "//src/utils:threadpool",
        "//src/utils:vector_slice",
    ],
)

cc_test(
    name = "compute_tree_scores_test",
    srcs = ["compute_tree_scores_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":compute_tree_scores",
        "//external:gtest_main",
        "//src/data_store",
        "//src/data_store:column",
    ],
)

cc_library(
    name = "evaluation",
    srcs = ["evaluation.cc"],
    hdrs = ["evaluation.h"],
    deps = [
        ":compute_tree_scores",
        ":split_algo",
        ":utils",
        "//external:cppformat-lib",
        "//src/base",
        "//src/data_store",
        "//src/proto:tree_cc_proto",
        "//src/utils",
    ],
)

cc_library(
    name = "split_algo",
    srcs = ["split_algo.cc"],
    hdrs = ["split_algo.h"],
    deps = [
        "//src/base",
        "//src/data_store:column",
        "//src/loss_func:gradient_data",
        "//src/proto:config_cc_proto",
        "//src/proto:tree_cc_proto",
        "//src/utils",
        "//src/utils:vector_slice",
    ],
)

cc_test(
    name = "split_algo_test",
    srcs = ["split_algo_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":split_algo",
        "//external:gtest_main",
        "//src/proto:config_cc_proto",
    ],
)

cc_library(
    name = "gbdt_algo",
    srcs = ["gbdt_algo.cc"],
    hdrs = ["gbdt_algo.h"],
    deps = [
        ":compute_tree_scores",
        ":split_algo",
        ":tree_algo",
        ":utils",
        "//external:cppformat-lib",
        "//src:flags",
        "//src/base",
        "//src/data_store",
        "//src/data_store:flatfiles_data_store",
        "//src/loss_func",
        "//src/loss_func:loss_func_factory",
        "//src/proto:config_cc_proto",
        "//src/proto:tree_cc_proto",
        "//src/utils",
        "//src/utils:stopwatch",
        "//src/utils:subsampling",
        "//src/utils:threadpool",
        "//src/utils:vector_slice",
    ],
)

cc_test(
    name = "gbdt_algo_test",
    srcs = ["gbdt_algo_test.cc"],
    copts = C_TEST_OPTS,
    data = [":gbdt_algo_testdata"],
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":gbdt_algo",
        ":utils",
        "//external:gtest_main",
        "//src/loss_func",
        "//src/loss_func:loss_func_factory",
        "//src/proto:config_cc_proto",
        "//src/proto:tree_cc_proto",
        "//src/utils",
    ],
)

filegroup(
    name = "gbdt_algo_testdata",
    srcs = [
        "testdata/gbdt_algo_test",
    ],
)

cc_library(
    name = "tree_algo",
    srcs = ["tree_algo.cc"],
    hdrs = ["tree_algo.h"],
    deps = [
        ":split_algo",
        "//external:cppformat-lib",
        "//src:flags",
        "//src/base",
        "//src/data_store:column",
        "//src/proto:config_cc_proto",
        "//src/proto:tree_cc_proto",
        "//src/utils",
        "//src/utils:stopwatch",
        "//src/utils:subsampling",
        "//src/utils:threadpool",
    ],
)

cc_test(
    name = "tree_algo_test",
    srcs = ["tree_algo_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":tree_algo",
        "//external:gtest_main",
        "//src/loss_func:gradient_data",
        "//src/proto:config_cc_proto",
        "//src/proto:tree_cc_proto",
        "//src/utils:subsampling",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//external:cppformat-lib",
        "//src:flags",
        "//src/base",
        "//src/data_store",
        "//src/proto:config_cc_proto",
        "//src/proto:tree_cc_proto",
        "//src/utils",
        "//src/utils:json_utils",
        "//src/utils:stopwatch",
        "//src/utils:threadpool",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":utils",
        "//external:gtest_main",
        "//src/proto:tree_cc_proto",
    ],
)
