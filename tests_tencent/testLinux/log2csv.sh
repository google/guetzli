#!/bin/bash

in=$1
out=$2

cat $in | while read line
do

log=(${line//,/ })

src=${log[0]}
dist=${log[1]}
time=${log[2]}
suc=${log[3]}
s1=${log[4]}
s2=${log[5]}

val=`./calc_ssim_psnr -i $src -o $dist`
arr=(${val//,/ })

rate=$(awk -v before=$s1 -v after=$s2 'BEGIN{print after/before }')
ssim=${arr[1]}
psnr=${arr[2]}
     
echo "$src,$dist,$time,$suc,$s1,$s2,$rate,$ssim,$psnr" >> $out

#copy failed picture
if [ $suc == 1 ]; then
    cp $src ./fail_pics/
    continue
fi

#copy low quality picture
if [ $(echo "$ssim < 0.85"|bc) == 1 ] || [ $(echo "$psnr < 30"|bc) == 1 ]; then
    cp $src ./low_quality/
fi

#copy picture if it become larger
if [ $(echo "$rate > 1"|bc) == 1 ]; then
    cp $src ./become_larger/
fi
done
