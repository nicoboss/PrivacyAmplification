:start
for %%f in (glsl\*.comp) do (
    glsl\glslangValidator.exe -V glsl\%%~nf.comp -o bin\Release\%%~nf.spv
)
pause
goto start
