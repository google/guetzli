@rem setupt windows var
call vcvars64.bat

nvcc -Xcompiler "/wd 4819" -I"./" -arch=compute_30 --fmad=false --machine 64 -G -g -ptx -o clguetzli\clguetzli.cu.ptx64 clguetzli\clguetzli.cu