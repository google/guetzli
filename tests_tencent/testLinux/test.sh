#!/bin/bash

src=$1
out_dir=$(date +%s)

mkdir $out_dir

if [ "$src" == ""  ]; then
    src="/data/ftp/data/GPUTEST"
fi

echo "Source Floder: $src"

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

echo "Guetzli CUDA"
echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./guetzli.cuda.csv
start=$(date +%s.%N)
find $src -name "*.jpg" -exec ./test_guetzli.sh {} \;
end=$(date +%s.%N)
calcTimeCost $start $end
cp ./guetzli.cuda.csv ./$out_dir

#echo "WebP"
#echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./webp.csv
#start=$(date +%s.%N)
#find $src -name "*.jpg" -exec ./test_webp.sh {} \;
#end=$(date +%s.%N)
#calcTimeCost $start $end
#cp ./webp.csv ./$out_dir

#echo "SharpP"
#echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./sharpp.csv
#start=$(date +%s.%N)
#find $src -name "*.jpg" -exec ./test_tpg.sh {} \;
#end=$(date +%s.%N)
#calcTimeCost $start $end
#cp ./sharpp.csv ./$out_dir

#echo "Guetzli OpenCl"
#echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./guetzli.cl.csv
#start=$(date +%s.%N)
#find $src -name "*.jpg" -exec ./test_guetzli_cl.sh {} \;
#end=$(date +%s.%N)
#calcTimeCost $start $end
#cp ./guetzli.cl.csv ./$out_dir

#echo "Guetzli C"
#echo "Src,Dist,Cost,Ret,Before,After,Rate,SSIM,PSNR" > ./guetzli.c.csv
#start=$(date +%s.%N)
#find $src -name "*.jpg" -exec ./test_guetzli_c.sh {} \;
#end=$(date +%s.%N)
#calcTimeCost $start $end
#cp ./guetzli.c.csv ./$out_dir
