:start
for %%f in (glsl\*.comp) do (
    glsl\glslangValidator.exe -V glsl\%%~nf.comp -o SPIRV\%%~nf.spv
)
pause
goto start
