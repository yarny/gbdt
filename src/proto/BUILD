package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

load("/build_rules/gen_proto", "proto_lib")

proto_lib(
  name = "tree_proto",
  src = "tree.proto",
)

proto_lib(
  name = "config_proto",
  src = "config.proto",
)
