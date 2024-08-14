@echo off
REM Navigate to the directory containing the virtual environment
cd /d C:\path\to\your\project

REM Check if the virtual environment exists
if not exist oly_env\Scripts\activate.bat (
    echo The virtual environment 'oly_env' does not exist.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment 'oly_env'...
call oly_env\Scripts\activate.bat

REM Inform the user that the virtual environment has been activated
echo Virtual environment 'oly_env' has been activated.


cmd /k