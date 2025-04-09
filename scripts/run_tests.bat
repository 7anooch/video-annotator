@echo off
REM Run tests for the Video Annotator project

REM Change to the project root directory
cd /d "%~dp0\.."

REM Run the tests
python -m tests.run_tests %*

REM Exit with the same status as the tests
exit /b %ERRORLEVEL%
