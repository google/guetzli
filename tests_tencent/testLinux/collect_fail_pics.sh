#!/bin/bash

in=$1

cat $in | while read line
do

log=(${line//,/ })

src=${log[0]}

if [ $src == "Src" ]; then
    continue
fi

suc=${log[3]}
rate=${log[6]}
ssim=${log[7]}
psnr=${log[8]}

#copy failed picture
if [ $suc == 1 ]; then
    echo "$src compress failed"
    cp $src ./fail_pics/
    continue
fi

#copy low quality picture
if [ $(echo "$ssim < 0.85"|bc) == 1 ] || [ $(echo "$psnr < 30.0"|bc) == 1 ]; then
    echo "$src is low quality picture ssim=$ssim psnr=$psnr"
    cp $src ./low_quality/
fi

#copy picture if it become larger
if [ $(echo "$rate > 1.0"|bc) == 1 ]; then
    echo "$src become larger after compress, rate=$rate"
    cp $src ./become_larger/
fi
done

tar cvf bug_pics.tar low_quality/ become_larger/ fail_pics/
