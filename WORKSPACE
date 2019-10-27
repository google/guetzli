# Description:
#   Bazel workspace file for Guetzli.

workspace(name = "guetzli")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "png_archive",
    build_file = "png.BUILD",
    sha256 = "a941dc09ca00148fe7aaf4ecdd6a67579c293678ed1e1cf633b5ffc02f4f8cf7",
    strip_prefix = "libpng-1.2.57",
    url = "http://github.com/glennrp/libpng/archive/v1.2.57.zip",
)

http_archive(
    name = "zlib_archive",
    build_file = "zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    url = "http://zlib.net/fossils/zlib-1.2.11.tar.gz",
)

local_repository(
    name = "butteraugli",
    path = "third_party/butteraugli/",
)
