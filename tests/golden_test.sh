#!/bin/bash

GUETZLI=${1:-bin/Release/guetzli}
INPUT_DIR_PNG=$HOME/.cache/guetzli-test-corpus
INPUT_DIR_JPG=$(mktemp -d)
OUTPUT_DIR=$(mktemp -d)

if [[ -d $INPUT_DIR_PNG ]]; then
  (cd $INPUT_DIR_PNG ; sha256sum -c ; exit $? ) < $(dirname $0)/png_checksums.txt || rm -r ${INPUT_DIR_PNG}
fi

if [[ ! -d $INPUT_DIR_PNG ]]; then
  mkdir -p $INPUT_DIR_PNG || exit 2
  curl https://storage.googleapis.com/test-image-corpus/test-corpus.tgz | tar -C $INPUT_DIR_PNG -zxf - || exit 2
fi

for i in $INPUT_DIR_PNG/*.png; do
  pngtopnm < $i | cjpeg -sample 1x1 -quality 100 > $INPUT_DIR_JPG/$(basename $i .png).jpg || exit 2
  pngtopnm < $i | cjpeg -sample 1x1 -progressive -quality 100 > $INPUT_DIR_JPG/$(basename $i .png)-progressive.jpg || exit 2
  pngtopnm < $i | cjpeg -sample 2x2,1x1,1x1 -quality 100 > $INPUT_DIR_JPG/$(basename $i .png)-420.jpg || exit 2
done

for i in $INPUT_DIR_PNG/*.png $INPUT_DIR_JPG/*.jpg; do
  echo $i $OUTPUT_DIR/$(basename $i).guetzli.jpg
done | xargs -L 1 -P $(getconf _NPROCESSORS_ONLN) -t $GUETZLI

if [[ -n "$UPDATE_GOLDEN" ]]; then
  (cd $OUTPUT_DIR ; sha256sum *) > $(dirname $0)/golden_checksums.txt
else
  (cd $OUTPUT_DIR ; sha256sum -c) < $(dirname $0)/golden_checksums.txt
fi

