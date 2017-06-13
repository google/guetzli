@echo off
@call copy.bat

@set exe="guetzli.exe"

@set file=big
@set q=95

@rem set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
@rem echo CUDA Time Start = %CURRENT_TIME% --------------

@rem %exe% --cuda --quality %q% %file%.jpg %file%.cuda.jpg

@rem set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
@rem echo CUDA Time End   = %CURRENT_TIME% --------------

@rem set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
@rem echo Opencl Time Start = %CURRENT_TIME% ------------

@rem %exe% --opencl %file%.jpg %file%.opencl.jpg

@rem set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
@rem echo Opencl Time End   = %CURRENT_TIME% ------------

set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo C Time Start = %CURRENT_TIME% ------------

%exe% --quality %q% %file%.jpg %file%.cout.jpg

set CURRENT_TIME=%time:~0,2%:%time:~3,2%:%time:~6,2%
echo C Time End   = %CURRENT_TIME% ------------