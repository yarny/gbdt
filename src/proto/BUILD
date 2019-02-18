package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

proto_library(
    name = "tree_proto",
    srcs = ["tree.proto"],
)

proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
)

cc_proto_library(
    name = "tree_cc_proto",
    deps = [":tree_proto"],
)

cc_proto_library(
    name = "config_cc_proto",
    deps = [":config_proto"],
)
