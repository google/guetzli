@echo off
echo CPU Time
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%
guetzli.exe big.jpg big.cout.jpg
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%

echo OpenCL Time
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%
guetzli.exe --opencl big.jpg big.opencl.jpg
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%

echo CUDA Time
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%
guetzli.exe --cuda big.jpg big.cuda.jpg
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%