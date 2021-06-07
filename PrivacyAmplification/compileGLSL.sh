#!/bin/bash
mkdir -p SPIRV
for f in glsl/*.comp; do
    if [ "$f" != "toBinaryArray" ]; then
        f_no_ext=$(echo "$f" | cut -f 1 -d '.')
        f_base=$(echo "$f_no_ext" | cut -f 2 -d '/')
        glslangValidator -V $f -o SPIRV/$f_base.spv
    fi
done
glslangValidator -V glsl/toBinaryArray.comp -o SPIRV/toBinaryArray.spv --define-macro XOR_WITH_KEY_REST=TRUE
glslangValidator -V glsl/toBinaryArray.comp -o SPIRV/toBinaryArrayNoXOR.spv --define-macro XOR_WITH_KEY_REST=FALSE
