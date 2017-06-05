@rem setupt windows var
call vcvars64.bat

@echo %1 --machine 64 or 32
@echo %2  -G 

nvcc -Xcompiler "/wd 4819" -I"./" -arch=compute_30 --machine %1 %2 -ptx -o clguetzli\clguetzli.cu.ptx%1  clguetzli\clguetzli.cu