@echo off
echo CPU Time
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%
guetzli.exe big.jpg big.cpu.out.jpg
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%

echo OpenCL Time
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%
guetzli.exe --opencl big.jpg big.gpu.out.jpg
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%
