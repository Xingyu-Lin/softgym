# clone FleX repo
git clone https://github.com/NVIDIAGameWorks/FleX.git
mv FleX PyFlex
# copy files in softgym_files back to PyFlex
cd PyFlex
cp -r demo/ bindings/
cd ..
cp -r softgym_files/build PyFlex/bindings/
cp softgym_files/CMakeLists.txt PyFlex/bindings/
cp softgym_files/pyflex.cpp PyFlex/bindings/
cp -r softgym_files/softgym_scenes PyFlex/bindings/
cp -r softgym_files/utils PyFlex/bindings/
cp softgym_files/helpers.h PyFlex/bindings/
cp softgym_files/shadersGL.cpp PyFlex/bindings/opengl/
cp softgym_files/scenes.h PyFlex/bindings/
cp softgym_files/libSDL2.a PyFlex/external/SDL2-2.0.4/lib/x64/
cp softgym_files/libSDL2.la PyFlex/external/SDL2-2.0.4/lib/x64/
cp softgym_files/libSDL2main.a PyFlex/external/SDL2-2.0.4/lib/x64/
cp softgym_files/libSDL2_test.a PyFlex/external/SDL2-2.0.4/lib/x64/
#cp softgym_files/NvFlexDebugCUDA_x64.a PyFlex/lib/linux64/
#cp softgym_files/NvFlexDeviceDebug_x64.a PyFlex/lib/linux64/
#cp softgym_files/NvFlexDeviceRelease_x64.a PyFlex/lib/linux64/
#cp softgym_files/NvFlexExtDebugCUDA_x64.a PyFlex/lib/linux64/
#cp softgym_files/NvFlexExtReleaseCUDA_x64.a PyFlex/lib/linux64/
#cp softgym_files/NvFlexReleaseCUDA_x64.a PyFlex/lib/linux64/
cp softgym_files/glad_egl.h PyFlex/external/glad/include/glad/
cp softgym_files/glad_egl.c PyFlex/external/glad/src/
cp softgym_files/libSDL2-2.0.so.0 PyFlex/external/SDL2-2.0.4/lib/x64/
