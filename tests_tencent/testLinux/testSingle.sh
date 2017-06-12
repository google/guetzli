#!/bin/bash

# arg1=start, arg2=end, format: %s.%N  
function calcTimeCost() {  
    start=$1  
    end=$2  
     
    start_s=$(echo $start | cut -d '.' -f 1)  
    start_ns=$(echo $start | cut -d '.' -f 2)  
    end_s=$(echo $end | cut -d '.' -f 1)  
    end_ns=$(echo $end | cut -d '.' -f 2)  
  
  
    time=$(( ( 10#$end_s - 10#$start_s ) * 1000 + ( 10#$end_ns / 1000000 - 10#$start_ns / 1000000 ) ))  
     
     
    echo "cost: $time ms"  
}

echo "opencl"
start=$(date +%s.%N)
./guetzli --opencl 1.jpg 1.opencl.jpg
end=$(date +%s.%N)
calcTimeCost $start $end

echo "cuda"
start=$(date +%s.%N)
./guetzli --cuda 1.jpg 1.cuda.jpg
end=$(date +%s.%N)
calcTimeCost $start $end

echo "C"
start=$(date +%s.%N)
./guetzli 1.jpg 1.opencl.jpg
end=$(date +%s.%N)
calcTimeCost $start $end
