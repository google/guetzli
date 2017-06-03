


@if "%1" == "" goto start
@setlocal
@set userinput=%1
@if not "%1"=="store" @if not "%1"=="8.1" @if not "%userinput:~0,3%"=="10." goto usage
@endlocal

:start
@call :GetVSCommonToolsDir
@if "%VS140COMNTOOLS%"=="" goto error_no_VS140COMNTOOLSDIR

@call "%VS140COMNTOOLS%VCVarsQueryRegistry.bat" No32bit 64bit %1 %2

@if "%VSINSTALLDIR%"=="" goto error_no_VSINSTALLDIR
@if "%VCINSTALLDIR%"=="" goto error_no_VCINSTALLDIR
@if "%FrameworkDir64%"=="" goto error_no_FrameworkDIR64
@if "%FrameworkVersion64%"=="" goto error_no_FrameworkVer64
@if "%Framework40Version%"=="" goto error_no_Framework40Version

@set FrameworkDir=%FrameworkDir64%
@set FrameworkVersion=%FrameworkVersion64%

@if not "%WindowsSDK_ExecutablePath_x64%" == "" @set PATH=%WindowsSDK_ExecutablePath_x64%;%PATH%

@rem
@rem Set Windows SDK include/lib path
@rem
@if not "%WindowsSdkDir%" == "" @set PATH=%WindowsSdkDir%bin\x64;%WindowsSdkDir%bin\x86;%PATH%
@if not "%WindowsSdkDir%" == "" @set INCLUDE=%WindowsSdkDir%include\%WindowsSDKVersion%shared;%WindowsSdkDir%include\%WindowsSDKVersion%um;%WindowsSdkDir%include\%WindowsSDKVersion%winrt;%INCLUDE%
@if not "%WindowsSdkDir%" == "" @set LIB=%WindowsSdkDir%lib\%WindowsSDKLibVersion%um\x64;%LIB%
@if not "%WindowsSdkDir%" == "" @set LIBPATH=%WindowsLibPath%;%ExtensionSDKDir%\Microsoft.VCLibs\14.0\References\CommonConfiguration\neutral;%LIBPATH%

@REM Set NETFXSDK include/lib path
@if not "%NETFXSDKDir%" == "" @set INCLUDE=%NETFXSDKDir%include\um;%INCLUDE%
@if not "%NETFXSDKDir%" == "" @set LIB=%NETFXSDKDir%lib\um\x64;%LIB%

@rem
@rem Set UniversalCRT include/lib path, the default is the latest installed version.
@rem
@if not "%UCRTVersion%" == "" @set INCLUDE=%UniversalCRTSdkDir%include\%UCRTVersion%\ucrt;%INCLUDE%
@if not "%UCRTVersion%" == "" @set LIB=%UniversalCRTSdkDir%lib\%UCRTVersion%\ucrt\x64;%LIB%

@rem PATH
@rem ----
@if exist "%VSINSTALLDIR%Team Tools\Performance Tools\x64" @set PATH=%VSINSTALLDIR%Team Tools\Performance Tools\x64;%VSINSTALLDIR%Team Tools\Performance Tools;%PATH%

@if exist "%ProgramFiles%\HTML Help Workshop" set PATH=%ProgramFiles%\HTML Help Workshop;%PATH%
@if exist "%ProgramFiles(x86)%\HTML Help Workshop" set PATH=%ProgramFiles(x86)%\HTML Help Workshop;%PATH%
@if exist "%VSINSTALLDIR%Common7\Tools" set PATH=%VSINSTALLDIR%Common7\Tools;%PATH%
@if exist "%VSINSTALLDIR%Common7\IDE" set PATH=%VSINSTALLDIR%Common7\IDE;%PATH%
@if exist "%VCINSTALLDIR%VCPackages" set PATH=%VCINSTALLDIR%VCPackages;%PATH%
@if exist "%FrameworkDir%\%Framework40Version%" set PATH=%FrameworkDir%\%Framework40Version%;%PATH%
@if exist "%FrameworkDir%\%FrameworkVersion%" set PATH=%FrameworkDir%\%FrameworkVersion%;%PATH%
@if exist "%VCINSTALLDIR%BIN\amd64" set PATH=%VCINSTALLDIR%BIN\amd64;%PATH%

@rem Add path to MSBuild Binaries
@if exist "%ProgramFiles%\MSBuild\14.0\bin\amd64" set PATH=%ProgramFiles%\MSBuild\14.0\bin\amd64;%PATH%
@if exist "%ProgramFiles(x86)%\MSBuild\14.0\bin\amd64" set PATH=%ProgramFiles(x86)%\MSBuild\14.0\bin\amd64;%PATH%

@if exist "%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\TestWindow" @set PATH=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\TestWindow;%PATH%

@rem INCLUDE
@rem -------
@if exist "%VCINSTALLDIR%ATLMFC\INCLUDE" set INCLUDE=%VCINSTALLDIR%ATLMFC\INCLUDE;%INCLUDE%
@if exist "%VCINSTALLDIR%INCLUDE" set INCLUDE=%VCINSTALLDIR%INCLUDE;%INCLUDE%

@rem LIB
@rem ---
@if "%1" == "store" goto setstorelib
@if exist "%VCINSTALLDIR%ATLMFC\LIB\amd64" set LIB=%VCINSTALLDIR%ATLMFC\LIB\amd64;%LIB%
@if exist "%VCINSTALLDIR%LIB\amd64" set LIB=%VCINSTALLDIR%LIB\amd64;%LIB%
@goto setlibpath
:setstorelib
@if exist "%VCINSTALLDIR%LIB\store\amd64" set LIB=%VCINSTALLDIR%LIB\store\amd64;%LIB%

:setlibpath
@rem LIBPATH
@rem -------
@if "%1" == "store" goto setstorelibpath
@if exist "%VCINSTALLDIR%ATLMFC\LIB\amd64" set LIBPATH=%VCINSTALLDIR%ATLMFC\LIB\amd64;%LIBPATH%
@if exist "%VCINSTALLDIR%LIB\amd64" set LIBPATH=%VCINSTALLDIR%LIB\amd64;%LIBPATH%
@goto appendlibpath
:setstorelibpath
@if exist "%VCINSTALLDIR%LIB\store\amd64" set LIBPATH=%VCINSTALLDIR%LIB\store\amd64;%VCINSTALLDIR%LIB\store\references;%LIBPATH%
:appendlibpath
@if exist "%FrameworkDir%\%Framework40Version%" set LIBPATH=%FrameworkDir%\%Framework40Version%;%LIBPATH%
@if exist "%FrameworkDir%\%FrameworkVersion%" set LIBPATH=%FrameworkDir%\%FrameworkVersion%;%LIBPATH%

@set Platform=X64
@set CommandPromptType=Native

@goto end

@REM -----------------------------------------------------------------------
:GetVSCommonToolsDir
@set VS140COMNTOOLS=
@call :GetVSCommonToolsDirHelper32 HKLM > nul 2>&1
@if errorlevel 1 call :GetVSCommonToolsDirHelper32 HKCU > nul 2>&1
@if errorlevel 1 call :GetVSCommonToolsDirHelper64  HKLM > nul 2>&1
@if errorlevel 1 call :GetVSCommonToolsDirHelper64  HKCU > nul 2>&1
@exit /B 0

:GetVSCommonToolsDirHelper32
@for /F "tokens=1,2*" %%i in ('reg query "%1\SOFTWARE\Microsoft\VisualStudio\SxS\VS7" /v "14.0"') DO (
	@if "%%i"=="14.0" (
		@SET VS140COMNTOOLS=%%k
	)
)
@if "%VS140COMNTOOLS%"=="" exit /B 1
@SET VS140COMNTOOLS=%VS140COMNTOOLS%Common7\Tools\
@exit /B 0

:GetVSCommonToolsDirHelper64
@for /F "tokens=1,2*" %%i in ('reg query "%1\SOFTWARE\Wow6432Node\Microsoft\VisualStudio\SxS\VS7" /v "14.0"') DO (
	@if "%%i"=="14.0" (
		@SET VS140COMNTOOLS=%%k
	)
)
@if "%VS140COMNTOOLS%"=="" exit /B 1
@SET VS140COMNTOOLS=%VS140COMNTOOLS%Common7\Tools\
@exit /B 0

@REM -----------------------------------------------------------------------
:error_no_VS140COMNTOOLSDIR
@echo ERROR: Cannot determine the location of the VS Common Tools folder.
@goto end

:error_no_VSINSTALLDIR
@echo ERROR: Cannot determine the location of the VS installation.
@goto end

:error_no_VCINSTALLDIR
@echo ERROR: Cannot determine the location of the VC installation.
@goto end

:error_no_FrameworkDIR64
@echo ERROR: Cannot determine the location of the .NET Framework 64bit installation.
@goto end

:error_no_FrameworkVer64
@echo ERROR: Cannot determine the version of the .NET Framework 64bit installation.
@goto end

:error_no_Framework40Version
@echo ERROR: Cannot determine the .NET Framework 4.0 version.
@goto end

:usage
echo Error in script usage. The correct usage is:
echo     %0
echo   or
echo     %0 store
echo   or
echo     %0 10.0.10240.0
echo   or
echo     %0 store 10.0.10240.0

:end

