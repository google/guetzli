# Description:
#   Bazel workspace file for Guetzli.

workspace(name = "guetzli")

new_http_archive(
    name = "png_archive",
    build_file = "png.BUILD",
    sha256 = "a941dc09ca00148fe7aaf4ecdd6a67579c293678ed1e1cf633b5ffc02f4f8cf7",
    strip_prefix = "libpng-1.2.57",
    url = "http://github.com/glennrp/libpng/archive/v1.2.57.zip",
)

new_http_archive(
    name = "zlib_archive",
    build_file = "zlib.BUILD",
    sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
    strip_prefix = "zlib-1.2.8",
    url = "http://zlib.net/zlib-1.2.8.tar.gz",
)

git_repository(
    name = "gflags_git",
    commit = "cce68f0c9c5d054017425e6e6fd54f696d36e8ee",
    remote = "https://github.com/gflags/gflags.git",
)

bind(
    name = "gflags",
    actual = "@gflags_git//:gflags",
)

local_repository(
    name = "butteraugli",
    path = "third_party/butteraugli/",
)
