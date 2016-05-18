# Butteraugli — a tool for measuring differences between images

## Introduction

Butteraugli is a project that estimates the psychovisual similarity of two
images. It gives a score for the images that is reliable in the domain of
barely noticeable differences. Butteraugli not only gives a scalar score,
but also a spatial map of the level of differences.

One of the main motivations for this project is the statistical differences
in location and density of different color receptors, particularly the
low density of blue cones in the fovea. Another motivation comes from
more accurate modeling of ganglion cells, particularly the frequency
space inhibition. 

## Use

Butteraugli can work as a quality metric for lossy image and
video compression. On our small test corpus butteraugli performs
better than our implementations of the reference methods, psnrhsv-m,
ssim, and our yuv-color-space variant of ssim. One possible use is to
define the quality level setting used in a jpeg compression, or to
compare two or more compression methods at the same level of psychovisual
differences.

Butteraugli is intended to be a research tool more than a practical tool for
choosing compression formats. We don't know how well butteraugli performs with
major deformations -- we have mostly tuned it within a small range of quality,
roughly corresponding to jpeg qualities 90 to 95.

## Interface

Only a C++ interface is provided. The interface takes two images, gives out a
map and a scalar value defining the difference. The scalar value can be
compared to two reference values that divide the value space into three
experience classes: 'great', 'acceptable' and 'not acceptable'. 
