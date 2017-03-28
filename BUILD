# Description:
#   Bazel build file for Guetzli.

cc_library(
    name = "guetzli_lib",
    srcs = glob(
        [
            "guetzli/*.h",
            "guetzli/*.cc",
            "guetzli/*.inc",
        ],
        exclude = ["guetzli/guetzli.cc"],
    ),
    copts = [ "-Wno-sign-compare" ],
    deps = [
        "@butteraugli//:butteraugli_lib",
    ],
)

cc_binary(
    name = "guetzli",
    srcs = ["guetzli/guetzli.cc"],
    deps = [
        ":guetzli_lib",
        "@png_archive//:png",
    ],
)
