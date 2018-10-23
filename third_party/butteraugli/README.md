# butteraugli

> A tool for measuring perceived differences between images

## Introduction

Butteraugli is a project that estimates the psychovisual similarity of two
images. It gives a score for the images that is reliable in the domain of barely
noticeable differences. Butteraugli not only gives a scalar score, but also
computes a spatial map of the level of differences.

One of the main motivations for this project is the statistical differences in
location and density of different color receptors, particularly the low density
of blue cones in the fovea. Another motivation comes from more accurate modeling
of ganglion cells, particularly the frequency space inhibition.

## Use

Butteraugli can work as a quality metric for lossy image and video compression.
On our small test corpus butteraugli performs better than our implementations of
the reference methods, psnrhsv-m, ssim, and our yuv-color-space variant of ssim.
One possible use is to define the quality level setting used in a jpeg
compressor, or to compare two or more compression methods at the same level of
psychovisual differences.

Butteraugli is intended to be a research tool more than a practical tool for
choosing compression formats. We don't know how well butteraugli performs with
major deformations -- we have mostly tuned it within a small range of quality,
roughly corresponding to jpeg qualities 90 to 95.

## Interface

Only a C++ interface is provided. The interface takes two images and outputs a
map together with a scalar value defining the difference. The scalar value can
be compared to two reference values that divide the value space into three
experience classes: 'great', 'acceptable' and 'not acceptable'.

## Build instructions

Install [Bazel](http://bazel.build) by following the
[instructions](https://www.bazel.build/docs/install.html). Run `bazel build -c opt
//:butteraugli` in the directory that contains this README file to build the
[command-line utility](#cmdline-tool). If you want to use Butteraugli as a
library, depend on the `//:butteraugli_lib` target.

Alternatively, you can use the Makefile provided in the `butteraugli` directory,
after ensuring that [libpng](http://www.libpng.org/) and
[libjpeg](http://ijg.org/) are installed. On some systems you might need to also
install corresponding `-dev` packages.

The code is portable and also compiles on Windows after defining
`_CRT_SECURE_NO_WARNINGS` in the project settings.

## Command-line utility {#cmdline-tool}

Butteraugli, apart from the library, comes bundled with a comparison tool. The
comparison tool supports PNG and JPG images as inputs. To compare images, run:

```
butteraugli image1.{png|jpg} image2.{png|jpg}
```

The tool can also produce a heatmap of differences between images. The heatmap
will be output as a PNM image. To produce one, run:

```
butteraugli image1.{png|jpg} image2.{png|jpg} heatmap.pnm
```
