package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])
LINK_OPTS = [ "-lgflags", "-lglog", "-lprotobuf", "-pthread", "-ltcmalloc_minimal", "-lm" ]

C_OPTS = select({":darwin": ["-I/System/Library/Frameworks/Python.framework/Headers" ],
                 ":darwin_x86_64": ["-I/System/Library/Frameworks/Python.framework/Headers" ],
                 "//conditions:default": ["-I/usr/include/python2.7"]})

cc_library(
   name = "gbdt_py_base",
   srcs = [ "gbdt_py_base.cc" ],
   hdrs = [ "gbdt_py_base.h" ],
   deps = [
       "//external:pybind11-lib",
       "//src/base:base",
   ],
   copts = C_OPTS,
)

cc_library(
   name = "gbdt_py",
   srcs = [ "gbdt_py.cc" ],
   deps = [
       "//external:pybind11-lib",
       ":datastore_py",
       ":forest_py",
       ":gbdt_py_base",
       ":train_gbdt_py",
   ],
   copts = C_OPTS,
)

cc_library(
   name = "datastore_py",
   srcs = [ "datastore_py.cc" ],
   hdrs = [ "datastore_py.h" ],
   deps = [
       "//external:cppformat-lib",
       "//external:pybind11-lib",
       "//src/data_store:data_store",
       "//src/data_store:tsv_data_store",
       "//src/proto:config_proto_cc",
       ":column_py",
       ":gbdt_py_base",
   ],
   copts = C_OPTS,
)

cc_library(
   name = "column_py",
   srcs = [ "column_py.cc" ],
   hdrs = [ "column_py.h" ],
   deps = [
       "//external:cppformat-lib",
       "//external:pybind11-lib",
       "//src/data_store:column",
       "//src/utils:utils",
       ":gbdt_py_base",
   ],
   copts = [ "-I/usr/include/python2.7",  ],
)


cc_library(
   name = "forest_py",
   srcs = [ "forest_py.cc" ],
   hdrs = [ "forest_py.h" ],
   deps = [
       "//external:pybind11-lib",
       "//src/gbdt_algo:evaluation",
       "//src/gbdt_algo:utils",
       "//src/proto:tree_proto_cc",
       "//src/utils:json_utils",
       "//src/utils:stopwatch",
       ":datastore_py",
       ":gbdt_py_base",
   ],
   copts = C_OPTS,
)

cc_binary(
   name = "libgbdt.so",
   linkstatic = 1,
   linkshared = 1,
   deps = [ ":gbdt_py" ],
   linkopts = LINK_OPTS,
)

cc_library(
   name = "train_gbdt_py",
   srcs = [ "train_gbdt_py.cc" ],
   hdrs = [ "train_gbdt_py.h" ],
   deps = [
       "//external:pybind11-lib",
       ":datastore_py",
       ":forest_py",
       ":gbdt_py_base",
       "//src:flags",
       "//src/gbdt_algo:gbdt_algo",
       "//src/gbdt_algo:utils",
       "//src/proto:config_proto_cc",
       "//src/utils:json_utils",
       "//src/utils:subsampling",
       "//src/utils:utils",
   ],
   copts = C_OPTS,
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_x86_64",
    values = {"cpu": "darwin_x86_64"},
    visibility = ["//visibility:public"]
)
