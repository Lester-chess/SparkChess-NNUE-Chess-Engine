@echo off
setlocal EnableDelayedExpansion

echo ===================================================
echo   Compiling SparkChess 3.0 (C++20 MSVC x64)
echo ===================================================

where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            set "VS_PATH=%%i"
        )
    )
    if defined VS_PATH (
        if exist "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat" (
            echo Initializing MSVC from: !VS_PATH!
            call "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
        )
    )
)

where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not find MSVC cl.exe.
    echo Please make sure Visual Studio C++ Desktop Development is installed.
    pause
    exit /b 1
)

echo Compiling sparkchess.exe with AVX2 optimizations...
cl.exe /nologo /std:c++20 /O2 /Oi /Ot /GL /Gy /MP /utf-8 /EHsc /DNDEBUG /arch:AVX2 /DUSE_AVX2 main.cpp /link /NOLOGO /LTCG /OPT:REF /OPT:ICF /STACK:16777216 /OUT:"sparkchess.exe"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================
    echo   [SUCCESS] Compiled successfully: sparkchess.exe
    echo ===================================================
    echo.
    echo Testing engine startup:
    echo isready | sparkchess.exe
) else (
    echo.
    echo [ERROR] Compilation failed.
)

pause
