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
    sha256 = "8d7e9f698ce48787b6e1c67e6bff79e487303e66077e25cb9784ac8835978017",
    strip_prefix = "zlib-1.2.10",
    url = "http://zlib.net/fossils/zlib-1.2.10.tar.gz",
)

local_repository(
    name = "butteraugli",
    path = "third_party/butteraugli/",
)
