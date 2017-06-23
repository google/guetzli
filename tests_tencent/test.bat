@echo off
@call copy.bat

@set IMG_DIR=%1
@set IMG_OUTDIR=%2
@set q=%3
@set IMG_EXT=*.jpg

if "%IMG_DIR%"=="" (
	set IMG_DIR=imgs_in
)

if "%IMG_OUTDIR%"=="" (
	set IMG_OUTDIR=imgs_out
)

if "%q%"=="" (
	set q=95
)

@set exe="guetzli.exe"

setlocal enabledelayedexpansion

call :time2sec %time%
set time1=%ns%

for /f "delims=" %%i in ('dir /b/a-d/s %IMG_DIR%\%IMG_EXT%') do (
	set fileName=%%~nxi
	set path=%%~dpi
	set dest_path=!path:%IMG_DIR%=%IMG_OUTDIR%!
	set input_file=!path!!fileName!
	set output_file=!dest_path!!fileName!

	mkdir !dest_path! >nul 2>nul

rem	echo "!input_file!"
rem	echo "!output_file!"

	call :time2sec !time!
	set time_file1=!ns!

	%exe% --cuda --quality %q% "!input_file!"  "!output_file!"

	call :time2sec !time!
	set time_file2=!ns!

	calc_ssim_psnr -i "!input_file!" -o "!output_file!"

	set /a diff=!time_file2! - !time_file1!
	echo cuda quality %q% time = !diff!
	echo ---------------------------------------
)

call :time2sec %time%
set time2=%ns%
set /a  diff=%time2%  - %time1%
echo cuda quality %q% total time = %diff%

goto :eof
 
:time2sec
rem 将时间转换成秒数，保存到ns中
set tt=%1
set hh=%tt:~0,2%
set mm=%tt:~3,2%
set ss=%tt:~6,2%
set ms=%tt:~9,2%
set /a ns=((%hh%*60+%mm%)*60+%ss%)*1000+ms
goto :eof