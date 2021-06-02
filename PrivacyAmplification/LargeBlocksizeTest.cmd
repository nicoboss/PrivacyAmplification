@echo OFF
for /F "tokens=2 delims==; " %%I in ('wmic Process call create "%CD%\bin\Release\PrivacyAmplificationCuda.exe"^,"%~dp0." ^| find "ProcessId"') do call :execute %%I
echo Exiting with %exitcode%
EXIT /B %exitcode%

:execute
set pid=%~n1
.\bin\Release\LargeBlocksizeExample.exe
set exitcode=%errorlevel%
echo LargeBlocksizeExample exited with %exitcode%
echo PID of PrivacyAmplification is %pid%
taskkill /PID %pid%
