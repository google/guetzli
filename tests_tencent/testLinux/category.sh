#!/bin/bash

src=$1

val=`imageinfo $src --width --height`
arr=($val)

result=$(awk -v w=${arr[0]} -v h=${arr[1]} 'BEGIN{print w*h }')

if [ $result -lt 120000  ]; then
    dist=${src/GPUTEST/GPUTEST_CATEGORY\/small}
elif [ $result -lt 480000  ]; then
    dist=${src/GPUTEST/GPUTEST_CATEGORY\/normal}
elif [ $result -lt 786432  ]; then
    dist=${src/GPUTEST/GPUTEST_CATEGORY\/middle}
else
    dist=${src/GPUTEST/GPUTEST_CATEGORY\/large}
fi

dir=${dist%/*}
if [ ! -d "$dir" ]; then
    mkdir -p $dir
fi
echo "Copy $src -> $dist"
cp $src $dist
