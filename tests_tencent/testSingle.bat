@echo off
@call copy.bat

@set exe="guetzli.exe"
@set ns=0

@set input_file=%1
@set q=%2

if "%input_file%"=="" (
	set input_file=1.jpg
)

if "%q%"=="" (
	set q=95
)

for %%i in ("%input_file%") do (
	set extName=%%~xi
	set fileName=%%~ni
	set path=%%~dpi
)

call :time2sec %time%
set time1=%ns%
set output_file=%path%%fileName%.cuda%extName%
	%exe% --cuda --quality %q% %input_file% %output_file%
call :time2sec %time%
set time2=%ns%
set /a  diff=%time2%  - %time1%
	calc_ssim_psnr -i %file%.jpg -o %output_file%
echo cuda quality %q% time = %diff%
echo ---------------------------------------

call :time2sec %time%
set time1=%ns%
set output_file=%path%%fileName%.opencl%extName%
	%exe% --opencl --quality %q% %input_file% %output_file%
call :time2sec %time%
set time2=%ns%
set /a  diff=%time2%  - %time1%
	calc_ssim_psnr -i %file%.jpg -o %output_file%
echo opencl quality %q% time = %diff%
echo ---------------------------------------

call :time2sec %time%
set time1=%ns%
set output_file=%path%%fileName%.cout%extName%
	%exe% --c --quality %q% %input_file% %output_file%
call :time2sec %time%
set time2=%ns%
set /a  diff=%time2%  - %time1%
	calc_ssim_psnr -i %file%.jpg -o %output_file%
echo c quality %q% time = %diff%
echo ---------------------------------------

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
