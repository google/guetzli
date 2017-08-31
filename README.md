<p align="center"><img src="https://cloud.githubusercontent.com/assets/203457/24553916/1f3f88b6-162c-11e7-990a-731b2560f15c.png" alt="Guetzli" width="64"></p>

# Introduction

Guetzli is a JPEG encoder that aims for excellent compression density at high
visual quality. Guetzli-generated images are typically 20-30% smaller than
images of equivalent quality generated by [libjpeg](http://libjpeg.sourceforge.net/).
Guetzli generates only sequential (nonprogressive) JPEGs due to faster decompression
speeds they offer.

[![Build Status](https://travis-ci.org/google/guetzli.svg?branch=master)](https://travis-ci.org/google/guetzli)

# Building

## On POSIX systems

1.  Get a copy of the source code, either by cloning this repository, or by
    downloading an
    [archive](https://github.com/google/guetzli/archive/master.zip) and
    unpacking it.
2.  Install [libpng](http://www.libpng.org/pub/png/libpng.html).
    If using your operating system
    package manager, install development versions of the packages if the
    distinction exists.
    *   On Ubuntu, do `apt-get install libpng-dev`.
    *   On Fedora, do `dnf install libpng-devel`. 
    *   On Arch Linux, do `pacman -S libpng`.
    *   On Alpine Linux, do `apk add libpng-dev`.
3.  Run `make` and expect the binary to be created in `bin/Release/guetzli`.

## On Windows

1.  Get a copy of the source code, either by cloning this repository, or by
    downloading an
    [archive](https://github.com/google/guetzli/archive/master.zip) and
    unpacking it.
2.  Install [Visual Studio 2015](https://www.visualstudio.com) and
    [vcpkg](https://github.com/Microsoft/vcpkg)
3.  Install `libpng` using vcpkg: `.\vcpkg install libpng`.
4.  Cause the installed packages to be available system-wide: `.\vcpkg integrate
    install`. If you prefer not to do this, refer to [vcpkg's
    documentation](https://github.com/Microsoft/vcpkg/blob/master/docs/EXAMPLES.md#example-1-2).
5.  Open the Visual Studio project enclosed in the repository and build it.

## On macOS

To install using [Homebrew](https://brew.sh/):
1. Install [Homebrew](https://brew.sh/)
2. `brew install guetzli`

To install using the repository:
1.  Get a copy of the source code, either by cloning this repository, or by
    downloading an
    [archive](https://github.com/google/guetzli/archive/master.zip) and
    unpacking it.
2.  Install [Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/)
3.  Install `libpng`
    *   Using [Homebrew](https://brew.sh/): `brew install libpng`.
    *   Using [MacPorts](https://www.macports.org/): `port install libpng` (You may need to use `sudo`).
4.  Run the following command to build the binary in `bin/Release/guetzli`.
    *   If you installed using [Homebrew](https://brew.sh/) simply use `make`
    *   If you installed using [MacPorts](https://www.macports.org/) use `CFLAGS='-I/opt/local/include' LDFLAGS='-L/opt/local/lib' make`

## With Bazel

There's also a [Bazel](https://bazel.build) build configuration provided. If you
have Bazel installed, you can also compile Guetzli by running `bazel build -c opt //:guetzli`.

# Using

**Note:** Guetzli uses a large amount of memory. You should provide 300MB of
memory per 1MPix of the input image.

**Note:** Guetzli uses a significant amount of CPU time. You should count on
using about 1 minute of CPU per 1 MPix of input image.

**Note:** Guetzli assumes that input is in **sRGB profile** with a **gamma of
2.2**. Guetzli will ignore any color-profile metadata in the image.

To try out Guetzli you need to [build](#building) or
[download](https://github.com/google/guetzli/releases) the Guetzli binary. The
binary reads a PNG or JPEG image and creates an optimized JPEG image:

```bash
guetzli [--quality Q] [--verbose] original.png output.jpg
guetzli [--quality Q] [--verbose] original.jpg output.jpg
```

Note that Guetzli is designed to work on high quality images. You should always
prefer providing uncompressed input images (e.g. that haven't been already
compressed with any JPEG encoders, including Guetzli). While it will work on other
images too, results will be poorer. You can try compressing an enclosed [sample
high quality
image](https://github.com/google/guetzli/releases/download/v0/bees.png).

You can pass a `--quality Q` parameter to set quality in units equivalent to
libjpeg quality. You can also pass a `--verbose` flag to see a trace of encoding
attempts made.

Please note that JPEG images do not support alpha channel (transparency). If the
input is a PNG with an alpha channel, it will be overlaid on black background
before encoding.
