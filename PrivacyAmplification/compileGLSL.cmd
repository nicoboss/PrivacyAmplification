@echo off
if not exist SPIRV mkdir SPIRV
for %%f in (glsl\*.comp) do (
    if not "%%~nf" == "toBinaryArray" glsl\glslangValidator.exe -V glsl\%%~nf.comp -o SPIRV\%%~nf.spv
)
glsl\glslangValidator.exe -V glsl\toBinaryArray.comp -o SPIRV\toBinaryArray.spv --define-macro XOR_WITH_KEY_REST=TRUE
glsl\glslangValidator.exe -V glsl\toBinaryArray.comp -o SPIRV\toBinaryArrayNoXOR.spv --define-macro XOR_WITH_KEY_REST=FALSE
