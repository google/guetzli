#!/bin/bash

src=$1
dist=${src/GPUTEST/GPUTEST_WEBP}

echo "Converting $src -> $dist"

dir=${dist%/*}
if [ ! -d "$dir" ]; then
    mkdir -p $dir
fi

suc=-1

# arg1=start, arg2=end, format: %s.%N  
function calcTimeCost() {  
    start=$1  
    end=$2  
     
    start_s=$(echo $start | cut -d '.' -f 1)  
    start_ns=$(echo $start | cut -d '.' -f 2)  
    end_s=$(echo $end | cut -d '.' -f 1)  
    end_ns=$(echo $end | cut -d '.' -f 2)  
  
  
    time=$(( ( 10#$end_s - 10#$start_s ) * 1000 + ( 10#$end_ns / 1000000 - 10#$start_ns / 1000000 ) ))  
    
    s1=`stat -c %s $src | tr -d '\n'`
    s2=`stat -c %s $dist | tr -d '\n'`

    val=`./calc_ssim_psnr -i $src -o $dist`
    arr=(${val//,/ })

    rate=$(awk -v before=$s1 -v after=$s2 'BEGIN{print after/before }')
    ssim=${arr[1]}
    psnr=${arr[2]}
     
    echo "$src,$dist,$time,$suc,$s1,$s2,$rate,$ssim,$psnr" >> webp.csv
}

start=$(date +%s.%N)
./cwebp $src -o $dist
suc=$?
end=$(date +%s.%N)
calcTimeCost $start $end
