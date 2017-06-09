@echo off
call copy.bat

set IMG_DIR=.\imgs
set exe="guetzli.exe"

@echo on

set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo CUDA Time Start = %CURRENT_TIME%

for /f "delims=\" %%i in ('dir /b /a-d /o-d "%IMG_DIR%\*.jpg" ') do (
%exe% --opencl %IMG_DIR%\%%i  .\out\%%i
)

set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo CUDA Time End = %CURRENT_TIME%