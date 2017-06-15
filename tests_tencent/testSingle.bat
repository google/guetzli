@echo off
@call copy.bat

@set exe="guetzli.exe"
@set ns=0

@set file=%1
@set q=95

if "%file%"=="" (
	set file=timg
)

call :time2sec %time%
set time1=%ns%
%exe% --cuda --quality %q% %file%.jpg %file%.%q%.cuda.jpg
call :time2sec %time%
set time2=%ns%
set /a  diff=%time2%  - %time1%
calc_ssim_psnr -i %file%.jpg -o %file%.%q%.cuda.jpg
echo cuda quality %q% time = %diff%

call :time2sec %time%
set time1=%ns%
%exe% --opencl --quality %q% %file%.jpg %file%.%q%.opencl.jpg
call :time2sec %time%
set time2=%ns%
set /a  diff=%time2%  - %time1%
calc_ssim_psnr -i %file%.jpg -o %file%.%q%.opencl.jpg
echo opencl quality %q% time = %diff%

call :time2sec %time%
set time1=%ns%
%exe% --quality %q% %file%.jpg %file%.%q%.cout.jpg
call :time2sec %time%
set time2=%ns%
set /a  diff=%time2%  - %time1%
calc_ssim_psnr -i %file%.jpg -o %file%.%q%.cout.jpg
echo c quality %q% time = %diff%

goto :eof
 
:time2sec
rem 将时间转换成秒数，保存到ns中
set tt=%1
set hh=%tt:~0,2%
set mm=%tt:~3,2%
set ss=%tt:~6,2%
set /a ns=(%hh%*60+%mm%)*60+%ss%
goto :eof
