@rem setupt windows var
call vcvars64.bat

@echo %1 --machine 64 or 32
@echo %2  -G 

set machine_num=%1%
set debug_opt=%2%

if "%machine_num%" == "" set machine_num=64

nvcc -Xcompiler "/wd 4819" -I"./" -arch=compute_30 -lineinfo -O3 --machine %machine_num% %debug_opt% -ptx -o clguetzli\clguetzli.cu.ptx%machine_num%  clguetzli\clguetzli.cu