@echo off
set IMG_DIR=.\imgs
set exe="guetzli.exe"

@echo on

echo CUDA Time
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%

for /f "delims=\" %%i in ('dir /b /a-d /o-d "%IMG_DIR%\*.jpg" ') do (
%exe% --opencl %IMG_DIR%\%%i  .\out\%%i
)

echo CUDA Time
set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo %CURRENT_TIME%
