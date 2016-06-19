package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [ "-lprotobuf", "-lglog", "-lgflags", "-lpthread", "-ltcmalloc_minimal", "-lm" ]
C_TEST_OPTS = [ "-Iexternal/googletest/googletest/include",
	        "-Iexternal/googletest/googlemock/include" ]

cc_library(
    name = "loss_func_logloss",
    srcs = [ "loss_func_logloss.cc" ],
    hdrs = [ "loss_func_logloss.h" ],
    deps = [
        "//external:cppformat-lib",
	"//src/utils:utils",
        "//src/base:base",
	"//src/data_store:data_store",
	":loss_func_math",
	":loss_func_pointwise",
	":target",
    ],
)

cc_test(
    name = "loss_func_logloss_test",
    srcs = [ "loss_func_logloss_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/proto:config_proto_cc",
        "//src/base:base",
	"//src/data_store:column",
	"//src/data_store:mem_data_store",
	":loss_func_logloss",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "loss_func_huberized_hinge",
    srcs = [ "loss_func_huberized_hinge.cc" ],
    hdrs = [ "loss_func_huberized_hinge.h" ],
    deps = [
        "//external:cppformat-lib",
	"//src/utils:utils",
        "//src/base:base",
	"//src/data_store:data_store",
	":loss_func_math",
	":loss_func_pointwise",
	":target",
    ],
)

cc_test(
    name = "loss_func_huberized_hinge_test",
    srcs = [ "loss_func_huberized_hinge_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/proto:config_proto_cc",
        "//src/base:base",
	"//src/data_store:column",
	"//src/data_store:mem_data_store",
	":loss_func_huberized_hinge",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "loss_func_mse",
    srcs = [ "loss_func_mse.cc" ],
    hdrs = [ "loss_func_mse.h" ],
    deps = [
        "//external:cppformat-lib",
	"//src/utils:utils",
        "//src/base:base",
	"//src/data_store:data_store",
	":loss_func_math",
	":loss_func_pointwise",
    ],
)

cc_test(
    name = "loss_func_mse_test",
    srcs = [ "loss_func_mse_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/proto:config_proto_cc",
        "//src/base:base",
	"//src/data_store:column",
	"//src/data_store:mem_data_store",
	":loss_func_mse",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "loss_func",
    hdrs = [ "loss_func.h" ],
    deps = [
        "//src/base:base",
        "//src/proto:config_proto_cc",
        "//src/data_store:data_store",
        ":gradient_data"
    ],
)

cc_library(
    name = "loss_func_factory",
    srcs = [ "loss_func_factory.cc" ],
    hdrs = [ "loss_func_factory.h" ],
    deps = [
	"//src/base:base",
        "//src/proto:config_proto_cc",
        ":loss_func",
	":loss_func_huberized_hinge",
	":loss_func_lambdamart",
	":loss_func_logloss",
	":loss_func_mse",
	":loss_func_pairwise",
    ],
)

cc_test(
    name = "loss_func_factory_test",
    srcs = [ "loss_func_factory_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/proto:config_proto_cc",
        ":loss_func_factory",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "target",
    srcs = [ "target.cc" ],
    hdrs = [ "target.h" ],
    deps = [
        "//src/proto:config_proto_cc",
        "//src/base:base",
	"//src/data_store:column",
	"//src/data_store:data_store",
    ],
)

cc_test(
   name = "target_test",
   srcs = [ "target_test.cc" ],
   deps = [
   	"//external:gtest_main",
	"//src/proto:config_proto_cc",
        "//src/data_store:mem_data_store",
	":target",
   ],
   copts = C_TEST_OPTS,
   linkopts = LINK_OPTS,
)

cc_library(
    name = "loss_func_pointwise",
    srcs = [ "loss_func_pointwise.cc" ],
    hdrs = [ "loss_func_pointwise.h" ],
    deps = [
        "//external:cppformat-lib",
        "//src/base:base",
        "//src/utils:subsampling",
        "//src/utils:threadpool",
        "//src:flags",
	":loss_func",
    ],
)

cc_test(
    name = "loss_func_pointwise_test",
    srcs = [ "loss_func_pointwise_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/base:base",
        ":loss_func_math",
	":loss_func_pointwise",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "loss_func_pairwise",
    srcs = [ "loss_func_pairwise.cc" ],
    hdrs = [ "loss_func_pairwise.h" ],
    deps = [
        "//external:cppformat-lib",
        "//src/base:base",
        "//src/data_store:data_store",
        ":loss_func",
        ":loss_func_math",
    ],
)

cc_test(
    name = "loss_func_pairwise_test",
    srcs = [ "loss_func_pairwise_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/proto:config_proto_cc",
        "//src/base:base",
	"//src/data_store:column",
	"//src/data_store:mem_data_store",
	":loss_func_math",
	":loss_func_pairwise",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "loss_func_lambdamart",
    srcs = [ "loss_func_lambdamart.cc" ],
    hdrs = [ "loss_func_lambdamart.h" ],
    deps = [
        "//external:cppformat-lib",
        "//src/base:base",
        "//src/data_store:data_store",
	":loss_func_math",
        ":loss_func_pairwise",
    ],
)

cc_test(
    name = "loss_func_lambdamart_test",
    srcs = [ "loss_func_lambdamart_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/proto:config_proto_cc",
        "//src/base:base",
	"//src/data_store:column",
	"//src/data_store:mem_data_store",
	":loss_func_lambdamart",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "loss_func_math",
    srcs = [ "loss_func_math.cc" ],
    hdrs = [ "loss_func_math.h" ],
    deps = [
        ":loss_func"
    ],
)

cc_library(
    name = "gradient_data",
    hdrs = [ "gradient_data.h" ],
    deps = [
    ],
)
