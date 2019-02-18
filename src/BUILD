package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [
    "-pthread",
    "-ltcmalloc_minimal",
    "-lm",
    "-L /usr/local/lib",
]

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    deps = [
        "//external:gflags",
    ],
)

cc_binary(
    name = "gbdt",
    srcs = ["main.cc"],
    linkopts = LINK_OPTS,
    linkstatic = 1,
    deps = [
        ":flags",
        "//external:glog",
        "//src/base",
        "//src/data_store",
        "//src/data_store:flatfiles_data_store",
        "//src/data_store:tsv_data_store",
        "//src/gbdt_algo",
        "//src/gbdt_algo:evaluation",
        "//src/loss_func",
        "//src/loss_func:loss_func_factory",
        "//src/proto:config_cc_proto",
        "//src/proto:tree_cc_proto",
        "//src/utils",
        "//src/utils:json_utils",
        "//src/utils:stopwatch",
        "//src/utils:subsampling",
    ],
)
