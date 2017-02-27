#!/usr/bin/env python

from __future__ import print_function
import os
import glob
import sys
import subprocess

BA_CMDLINE = './butteraugli {0} {1} {2}'
GUETZLI_CMDLINE = './guetzli -quality {2} {0} {1}'
OTHER_CMDLINE = sys.argv[3]

def run(cmdline):
  print('running {}'.format(cmdline), file=sys.stderr)
  return subprocess.check_output(cmdline, shell=True)

def size(filename):
  return os.stat(filename).st_size

def ba_distance(orig, compressed):
  return float(run(BA_CMDLINE.format(orig, compressed, compressed + ".diffmap.pnm")))

def handle_png(png):
  other_jpeg = png + "." + sys.argv[1] + ".other.jpg"
  guetzli_jpeg = png + "." + sys.argv[1] + ".guetzli.jpg"
  run(OTHER_CMDLINE.format(png, other_jpeg))
  other_distance = ba_distance(png, other_jpeg)
  left = 84.0
  right = 110.0
  while right - left > 0.05:
    q = (left + right) / 2
    run(GUETZLI_CMDLINE.format(png, guetzli_jpeg, q))
    guetzli_distance = ba_distance(png, guetzli_jpeg)
    if guetzli_distance < other_distance:
      right = q
    else:
      left = q
  run(GUETZLI_CMDLINE.format(png, guetzli_jpeg, right))
  guetzli_distance = ba_distance(png, guetzli_jpeg)
  assert guetzli_distance < other_distance
  return (size(guetzli_jpeg), size(other_jpeg))

pngs = glob.glob(sys.argv[2])
sizes = (0, 0)
for png in pngs:
  this_size = handle_png(png)
  print(png, this_size)
  sizes = (sizes[0] + this_size[0], sizes[1] + this_size[1])
print(sizes)
