# Description:
#   Bazel workspace file for Guetzli.

workspace(name = "guetzli")

new_http_archive(
    name = "png_archive",
    build_file = "png.BUILD",
    sha256 = "c35bcc6387495ee6e757507a68ba036d38ad05b415c2553b3debe2a57647a692",
    strip_prefix = "libpng-1.2.53",
    url = "http://github.com/glennrp/libpng/archive/v1.2.53.zip",
)

new_http_archive(
    name = "zlib_archive",
    build_file = "zlib.BUILD",
    sha256 = "73ab302ef31ed1e74895d2af56f52f5853f26b0370f3ef21954347acec5eaa21",
    strip_prefix = "zlib-1.2.9",
    url = "http://zlib.net/zlib-1.2.9.tar.gz",
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
