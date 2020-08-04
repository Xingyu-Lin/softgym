rm -rf softgym_files
mkdir softgym_files

# only in Pyflex
cp PyFlex/bindings/CMakeLists.txt softgym_files/
cp PyFlex/bindings/pyflex.cpp softgym_files/
cp -r PyFlex/bindings/softgym_scenes softgym_files/
cp -r PyFlex/bindings/utils softgym_files/

# diff
cp PyFlex/bindings/helpers.h softgym_files/
cp PyFlex/bindings/opengl/shadersGL.cpp softgym_files/
cp PyFlex/bindings/scenes.h softgym_files/
cp PyFlex/external/SDL2-2.0.4/lib/x64/libSDL2.a softgym_files/
cp PyFlex/external/SDL2-2.0.4/lib/x64/libSDL2.la softgym_files/
cp PyFlex/external/SDL2-2.0.4/lib/x64/libSDL2main.a softgym_files/
cp PyFlex/external/SDL2-2.0.4/lib/x64/libSDL2_test.a softgym_files/
#cp PyFlex/lib/linux64/NvFlexDebugCUDA_x64.a softgym_files/
#cp PyFlex/lib/linux64/NvFlexDeviceDebug_x64.a softgym_files/
#cp PyFlex/lib/linux64/NvFlexDeviceRelease_x64.a softgym_files/
#cp PyFlex/lib/linux64/NvFlexExtDebugCUDA_x64.a softgym_files/
#cp PyFlex/lib/linux64/NvFlexExtReleaseCUDA_x64.a softgym_files/
#cp PyFlex/lib/linux64/NvFlexReleaseCUDA_x64.a softgym_files/

# only in pyflex
cp PyFlex/external/glad/include/glad/glad_egl.h softgym_files/
cp PyFlex/external/glad/src/glad_egl.c softgym_files/
cp PyFlex/external/SDL2-2.0.4/lib/x64/libSDL2-2.0.so.0 softgym_files/
#rm -rf softgym_files/build/pyflex*.so softgym_files/build/CMakeCache.txt softgym_files/build/CMakeFiles
