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
    sha256 = "8d7e9f698ce48787b6e1c67e6bff79e487303e66077e25cb9784ac8835978017",
    strip_prefix = "zlib-1.2.10",
    url = "http://zlib.net/zlib-1.2.10.tar.gz",
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
