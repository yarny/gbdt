package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [ "-lprotobuf", "-lgflags", "-lglog", "-pthread", "-ltcmalloc_minimal", "-lm" ]
C_TEST_OPTS = [ "-Iexternal/googletest/googletest/include",
                "-Iexternal/googletest/googlemock/include"]

cc_library(
    name = "compute_tree_scores",
    srcs = [ "compute_tree_scores.cc" ],
    hdrs = [ "compute_tree_scores.h" ],
    deps = [
        "//src:flags",
        "//src/base:base",
	"//src/data_store:column",
        "//src/data_store:data_store",
        "//src/proto:tree_proto_cc",
        "//src/utils:subsampling",
	"//src/utils:threadpool",
        "//src/utils:vector_slice",
        ":split_algo",
    ],
)

cc_test(
    name = "compute_tree_scores_test",
    srcs = [ "compute_tree_scores_test.cc" ],
    deps = [
        "//external:gtest_main",
	"//src/data_store:column",
        "//src/data_store:data_store",
	":compute_tree_scores",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1
)

cc_library(
    name = "evaluation",
    srcs = [ "evaluation.cc" ],
    hdrs = [ "evaluation.h" ],
    deps = [
        "//external:cppformat-lib",
	"//src/base:base",
	"//src/data_store:data_store",
	"//src/proto:tree_proto_cc",
	"//src/utils:utils",
	":compute_tree_scores",
	":split_algo",
	":utils",
    ],
)

cc_library(
    name = "split_algo",
    srcs = [ "split_algo.cc" ],
    hdrs = [ "split_algo.h" ],
    deps = [
         "//src/base:base",
         "//src/data_store:column",
         "//src/loss_func:gradient_data",
         "//src/proto:config_proto_cc",
         "//src/proto:tree_proto_cc",
	 "//src/utils:vector_slice",
	 "//src/utils:utils",
    ],
)

cc_test(
    name = "split_algo_test",
    srcs = [ "split_algo_test.cc" ],
    deps = [
        "//external:gtest_main",
	"//src/proto:config_proto_cc",
	":split_algo",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1
)

cc_library(
    name = "gbdt_algo",
    srcs = [ "gbdt_algo.cc" ],
    hdrs = [ "gbdt_algo.h" ],
    deps = [
        "//external:cppformat-lib",
	"//src:flags",
        "//src/base:base",
	"//src/data_store:data_store",
	"//src/data_store:flatfiles_data_store",
	"//src/loss_func:loss_func",
	"//src/loss_func:loss_func_factory",
	"//src/proto:config_proto_cc",
	"//src/proto:tree_proto_cc",
        "//src/utils:stopwatch",
        "//src/utils:subsampling",
	"//src/utils:threadpool",
	"//src/utils:utils",
	"//src/utils:vector_slice",
        ":compute_tree_scores",
        ":split_algo",
	":tree_algo",
	":utils",
    ],
)

cc_test(
    name = "gbdt_algo_test",
    srcs = [ "gbdt_algo_test.cc" ],
    deps = [
        "//external:gtest_main",
         "//src/loss_func:loss_func",
        "//src/loss_func:loss_func_factory",
	"//src/proto:config_proto_cc",
	"//src/proto:tree_proto_cc",
	"//src/utils:utils",
        ":gbdt_algo",
        ":utils",
    ],
    data = [ ":gbdt_algo_testdata" ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1
)

filegroup(
    name = "gbdt_algo_testdata",
    srcs = [
        "testdata/gbdt_algo_test"
	]
)

cc_library(
    name = "tree_algo",
    srcs = [ "tree_algo.cc" ],
    hdrs = [ "tree_algo.h" ],
    deps = [
	"//src:flags",
        "//src/base:base",
	"//src/data_store:column",
        "//external:cppformat-lib",
	"//src/proto:config_proto_cc",
	"//src/proto:tree_proto_cc",
        "//src/utils:stopwatch",
        "//src/utils:subsampling",
	"//src/utils:threadpool",
	"//src/utils:utils",
        ":split_algo",
    ],
)

cc_test(
    name = "tree_algo_test",
    srcs = [ "tree_algo_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/loss_func:gradient_data",
	"//src/proto:config_proto_cc",
        "//src/proto:tree_proto_cc",
        "//src/utils:subsampling",
	":tree_algo",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1
)

cc_library(
    name = "utils",
    srcs = [ "utils.cc" ],
    hdrs = [ "utils.h" ],
    deps = [
        "//external:cppformat-lib",
	"//src:flags",
        "//src/base:base",
        "//src/data_store:data_store",
        "//src/proto:config_proto_cc",
        "//src/proto:tree_proto_cc",
        "//src/utils:json_utils",
	"//src/utils:stopwatch",
	"//src/utils:threadpool",
	"//src/utils:utils",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [ "utils_test.cc" ],
    deps = [
        "//external:gtest_main",
	"//src/proto:tree_proto_cc",
	":utils",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1
)