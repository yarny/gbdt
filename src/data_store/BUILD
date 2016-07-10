package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [ "-lprotobuf", "-lglog", "-lgflags", "-pthread", "-ltcmalloc_minimal", "-lm" ]
C_TEST_OPTS = [ "-Iexternal/googletest/googletest/include",
	        "-Iexternal/googletest/googlemock/include" ]

cc_library(
    name = "column",
    srcs = [ "column.cc" ],
    hdrs = [ "column.h" ],
    deps = [
        "//external:cppformat-lib",
        "//src/base:base",
    ],
)

cc_test(
    name = "column_test",
    srcs = [ "column_test.cc" ],
    deps = [
    	 "//external:gtest_main",
         "//src/base:base",
	 ":column"
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "data_store",
    hdrs = [ "data_store.h" ],
    srcs = [ "data_store.cc" ],
    deps = [
         "//external:cppformat-lib",
         "//src/base:base",
	 ":column"
    ],
)

cc_test(
   name = "data_store_test",
   srcs = [ "data_store_test.cc" ],
   deps = [
       	"//external:gtest_main",
	":column",
	":data_store",
        ":flatfiles_data_store",
	":mem_data_store",
   ],
   data = [ ":flatfiles_data_store_testdata" ],
   copts = C_TEST_OPTS,
   linkopts = LINK_OPTS,
)

cc_library(
    name = "mem_data_store",
    hdrs = [ "mem_data_store.h" ],
    srcs = [ "mem_data_store.cc" ],
    deps = [
         ":column",
	 ":data_store",
    ],
)

cc_library(
    name = "flatfiles_data_store",
    srcs = [ "flatfiles_data_store.cc" ],
    hdrs = [ "flatfiles_data_store.h" ],
    deps = [
        "//src/base:base",
        "//src/utils:utils",
	":data_store",
    ],
)

cc_test(
   name = "flatfiles_data_store_test",
   srcs = [ "flatfiles_data_store_test.cc" ],
   deps = [
       	"//external:gtest_main",
	":column",
        ":flatfiles_data_store",
   ],
   data = [ ":flatfiles_data_store_testdata" ],
   copts = C_TEST_OPTS,
   linkopts = LINK_OPTS,
)

filegroup(
    name = "flatfiles_data_store_testdata",
    srcs = [
        "testdata/flatfiles_data_store_test"
	]
)

cc_library(
    name = "tsv_block",
    srcs = [ "tsv_block.cc" ],
    hdrs = [ "tsv_block.h" ],
    deps = [
        "//external:cppformat-lib",
        "//src/base:base",
        "//src/utils:stopwatch",
        "//src/utils:utils",
    ],
)

filegroup(
    name = "tsv_data_store_testdata",
    srcs = [
        "testdata/tsv_data_store_test"
        ]
)

cc_test(
   name = "tsv_block_test",
   srcs = [ "tsv_block_test.cc" ],
   deps = [
        "//external:gtest_main",
        ":tsv_block",
   ],
   data = [ ":tsv_data_store_testdata" ],
   copts = C_TEST_OPTS,
   linkopts = LINK_OPTS,
)

cc_library(
    name = "tsv_data_store",
    srcs = [ "tsv_data_store.cc" ],
    hdrs = [ "tsv_data_store.h" ],
    deps = [
        "//external:cppformat-lib",
        "//src/base:base",
        "//src:flags",
        "//src/proto:config_proto_cc",
        "//src/utils:threadpool",
        "//src/utils:stopwatch",
        "//src/utils:utils",
        ":column",
        ":data_store",
        ":tsv_block"
    ],
)

cc_test(
   name = "tsv_data_store_test",
   srcs = [ "tsv_data_store_test.cc" ],
   deps = [
        "//external:gtest_main",
        "//src/proto:config_proto_cc",
        ":column",
        ":tsv_data_store",
   ],
   data = [ ":tsv_data_store_testdata" ],
   copts = C_TEST_OPTS,
   linkopts = LINK_OPTS,
)
