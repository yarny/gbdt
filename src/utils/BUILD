package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

LINK_OPTS = [ "-lprotobuf", "-lglog", "-lgflags", "-pthread", "-ltcmalloc_minimal", "-lm" ]
C_TEST_OPTS = [ "-Iexternal/googletest/googletest/include",
	        "-Iexternal/googletest/googlemock/include" ]
cc_library(
    name = "stopwatch",
    srcs = [ "stopwatch.cc" ],
    hdrs = [ "stopwatch.h" ],
    deps = [
       "//external:cppformat-lib",
       ":utils"
    ],
)

cc_test(
    name = "stopwatch_test",
    srcs = [ "stopwatch_test.cc" ],
    deps = [
       "//external:gtest_main",
       ":stopwatch",
    ],
    linkopts = LINK_OPTS,
    copts = C_TEST_OPTS,
)

cc_library(
    name = "utils",
    srcs = [ "utils.cc" ],
    hdrs = [ "utils.h", "utils_inl.h" ],
    deps = [
         "//src/base:base",
	 ":vector_slice",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [ "utils_test.cc" ],
    deps = [
        "//external:gtest_main",
	":utils",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)

cc_library(
    name = "vector_slice",
    hdrs = [ "vector_slice.h" ],
)

cc_test(
   name = "vector_slice_test",
   srcs = [ "vector_slice_test.cc" ],
   deps = [
   	"//external:gtest_main",
	":vector_slice",
   ],
   copts = C_TEST_OPTS,
   linkopts = LINK_OPTS,
)

cc_library(
   name = "threadpool",
   srcs = [ "threadpool.cc" ],
   hdrs = [ "threadpool.h" ],
)

cc_library(
   name = "json_utils",
   srcs = [ "json_utils.cc" ],
   hdrs = [ "json_utils.h" ],
   deps = [
        "//src/base:base"
   ]
)

cc_test(
    name = "json_utils_test",
    srcs = [ "json_utils_test.cc" ],
    deps = [
        "//external:gtest_main",
        "//src/proto:tree_cc_proto",
        ":json_utils",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
    linkstatic = 1
)

cc_library(
    name = "subsampling",
    srcs = [ "subsampling.cc" ],
    hdrs = [ "subsampling.h" ],
    deps = [
        ":utils",
    ],
)

cc_test(
    name = "subsampling_test",
    srcs = [ "subsampling_test.cc" ],
    deps = [
        "//external:gtest_main",
        ":utils",
        ":subsampling",
    ],
    copts = C_TEST_OPTS,
    linkopts = LINK_OPTS,
)