#!/bin/bash

in=$1

echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./small.$in
echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./normal.$in
echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./middle.$in
echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./large.$in

cat $in | while read line
do

log=(${line//,/ })

src=${log[0]}
#src=${src/GPUTEST/GPUTEST_ORIGIN}

if [ $src == "Src"  ]; then
    continue
fi

val=`imageinfo $src --width --height`
arr=($val)

result=$(awk -v w=${arr[0]} -v h=${arr[1]} 'BEGIN{print w*h }')

if [ $result -lt 120000  ]; then
    echo $line >> small.$in
elif [ $result -lt 480000  ]; then
    echo $line >> normal.$in
elif [ $result -lt 786432  ]; then
    echo $line >> middle.$in
else
    echo $line >> large.$in
fi

done
