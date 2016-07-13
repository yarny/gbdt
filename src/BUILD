package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [ "-lgflags", "-lglog", "-lprotobuf", "-pthread", "-ltcmalloc_minimal", "-lm" ]

cc_library(
    name = "flags",
    srcs = [ "flags.cc" ],
)

cc_binary(
   name = "gbdt",
   srcs = [ "main.cc" ],
   deps = [
      "//src/base:base",
      "//src/data_store:data_store",
      "//src/data_store:flatfiles_data_store",
      "//src/data_store:tsv_data_store",
      "//src/gbdt_algo:evaluation",
      "//src/gbdt_algo:gbdt_algo",
      "//src/loss_func:loss_func",
      "//src/loss_func:loss_func_factory",
      "//src/proto:config_proto_cc",
      "//src/proto:tree_proto_cc",
      "//src/utils:stopwatch",
      "//src/utils:subsampling",
      "//src/utils:utils",
      "//src/utils:json_utils",
      ":flags",
   ],
   linkopts = LINK_OPTS,
   linkstatic = 1
)