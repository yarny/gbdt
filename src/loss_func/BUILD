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
    name = "loss_func_logloss",
    srcs = ["loss_func_logloss.cc"],
    hdrs = ["loss_func_logloss.h"],
    deps = [
        ":loss_func_math",
        ":loss_func_pointwise",
        "//external:cppformat-lib",
        "//src/base",
        "//src/utils",
    ],
)

cc_test(
    name = "loss_func_logloss_test",
    srcs = ["loss_func_logloss_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":loss_func_logloss",
        "//external:gtest_main",
        "//src/base",
    ],
)

cc_library(
    name = "loss_func_huberized_hinge",
    srcs = ["loss_func_huberized_hinge.cc"],
    hdrs = ["loss_func_huberized_hinge.h"],
    deps = [
        ":loss_func_math",
        ":loss_func_pointwise",
        "//external:cppformat-lib",
        "//src/base",
        "//src/utils",
    ],
)

cc_test(
    name = "loss_func_huberized_hinge_test",
    srcs = ["loss_func_huberized_hinge_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":loss_func_huberized_hinge",
        "//external:gtest_main",
        "//src/base",
    ],
)

cc_library(
    name = "loss_func_mse",
    hdrs = ["loss_func_mse.h"],
    deps = [
        ":loss_func_math",
        ":loss_func_pointwise",
        "//external:cppformat-lib",
        "//src/base",
    ],
)

cc_test(
    name = "loss_func_mse_test",
    srcs = ["loss_func_mse_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":loss_func_mse",
        "//external:gtest_main",
        "//src/base",
    ],
)

cc_library(
    name = "loss_func",
    hdrs = ["loss_func.h"],
    deps = [
        ":gradient_data",
        "//src/base",
        "//src/proto:config_cc_proto",
    ],
)

cc_library(
    name = "loss_func_factory",
    srcs = ["loss_func_factory.cc"],
    hdrs = ["loss_func_factory.h"],
    deps = [
        ":loss_func",
        ":loss_func_auc",
        ":loss_func_gbrank",
        ":loss_func_huberized_hinge",
        ":loss_func_lambdamart",
        ":loss_func_logloss",
        ":loss_func_mse",
        ":loss_func_pairwise_logloss",
        "//src/base",
        "//src/proto:config_cc_proto",
    ],
)

cc_test(
    name = "loss_func_factory_test",
    srcs = ["loss_func_factory_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":loss_func_factory",
        "//external:gtest_main",
        "//src/proto:config_cc_proto",
    ],
)

cc_library(
    name = "loss_func_pointwise",
    srcs = ["loss_func_pointwise.cc"],
    hdrs = ["loss_func_pointwise.h"],
    deps = [
        ":loss_func",
        ":loss_func_math",
        "//external:cppformat-lib",
        "//src:flags",
        "//src/base",
        "//src/utils:subsampling",
        "//src/utils:threadpool",
    ],
)

cc_library(
    name = "group",
    srcs = ["group.cc"],
    hdrs = ["group.h"],
    deps = [
        "//external:glog",
        "//src/base",
        "//src/data_store:column",
    ],
)

cc_test(
    name = "group_test",
    srcs = ["group_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":group",
        "//external:gtest_main",
        "//src/base",
    ],
)

cc_library(
    name = "loss_func_pairwise",
    srcs = ["loss_func_pairwise.cc"],
    hdrs = ["loss_func_pairwise.h"],
    deps = [
        ":group",
        ":loss_func",
        ":loss_func_math",
        "//external:cppformat-lib",
        "//src:flags",
        "//src/base",
        "//src/data_store:column",
        "//src/utils:subsampling",
        "//src/utils:threadpool",
    ],
)

cc_test(
    name = "loss_func_pairwise_test",
    srcs = ["loss_func_pairwise_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":loss_func_math",
        ":loss_func_pairwise",
        ":loss_func_pairwise_logloss",
        "//external:gtest_main",
        "//src/base",
        "//src/data_store",
        "//src/data_store:column",
        "//src/proto:config_cc_proto",
    ],
)

cc_library(
    name = "loss_func_pairwise_logloss",
    hdrs = ["loss_func_pairwise_logloss.h"],
    deps = [
        ":loss_func_math",
        ":loss_func_pairwise",
        "//src/base",
        "//src/data_store:column",
    ],
)

cc_library(
    name = "loss_func_auc",
    hdrs = ["loss_func_auc.h"],
    deps = [
        ":loss_func_math",
        ":loss_func_pairwise",
        "//src/base",
    ],
)

cc_library(
    name = "loss_func_gbrank",
    hdrs = ["loss_func_gbrank.h"],
    deps = [
        ":loss_func_math",
        ":loss_func_pairwise",
        "//src/base",
    ],
)

cc_library(
    name = "loss_func_lambdamart",
    srcs = ["loss_func_lambdamart.cc"],
    hdrs = ["loss_func_lambdamart.h"],
    deps = [
        ":loss_func_math",
        ":loss_func_pairwise",
        "//external:cppformat-lib",
        "//src/base",
        "//src/data_store",
    ],
)

cc_test(
    name = "loss_func_lambdamart_test",
    srcs = ["loss_func_lambdamart_test.cc"],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":loss_func_lambdamart",
        "//external:gtest_main",
        "//src/base",
        "//src/data_store",
        "//src/data_store:column",
        "//src/proto:config_cc_proto",
    ],
)

cc_library(
    name = "loss_func_math",
    srcs = ["loss_func_math.cc"],
    hdrs = ["loss_func_math.h"],
    deps = [
        ":loss_func",
    ],
)

cc_library(
    name = "gradient_data",
    hdrs = ["gradient_data.h"],
    deps = [
    ],
)
