@echo off
@call copy.bat

@set IMG_DIR=imgs_in
@set IMG_OUTDIR=imgs_out
@set IMG_EXT=*.jpg

@set exe="guetzli.exe"

setlocal enabledelayedexpansion

set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo CUDA Time Start = %CURRENT_TIME%

for /f "delims=" %%i in ('dir /b/a-d/s %IMG_DIR%\%IMG_EXT%') do (
set fileName=%%~nxi
set path=%%~dpi
set dest_path=!path:%IMG_DIR%=%IMG_OUTDIR%!

mkdir !dest_path! >nul 2>nul

rem echo "!path!!fileName!"
rem echo "!dest_path!!fileName!"
%exe% --opencl "!path!!fileName!"  "!dest_path!!fileName!"
)

set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo CUDA Time End = %CURRENT_TIME%