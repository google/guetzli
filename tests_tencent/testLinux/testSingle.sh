#!/bin/bash
file=$1
if [ "$file" == ""  ]; then
    file="1.jpg"
fi

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
./guetzli --opencl $file ${file%.*}.opencl.jpg
end=$(date +%s.%N)
calcTimeCost $start $end

echo "cuda"
start=$(date +%s.%N)
./guetzli --cuda $file ${file%.*}.cuda.jpg
end=$(date +%s.%N)
calcTimeCost $start $end

echo "C"
start=$(date +%s.%N)
./guetzli $file ${file%.*}.cout.jpg
end=$(date +%s.%N)
calcTimeCost $start $end
