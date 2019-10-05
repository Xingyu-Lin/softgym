cmake_minimum_required(VERSION 3.5)
project(FleX_examples)

# set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA 9.0 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MMD -m64 -Wall -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -fPIC -fpermissive -fno-strict-aliasing -O3")

set(ROOT /home/yunzhu/Documents/PyFleX)
set(EXAMPLE_BIN example)

set(SOURCES ${SOURCES} ${ROOT}/examples/main.cpp)
set(SOURCES ${SOURCES} ${ROOT}/examples/imgui.cpp)
set(SOURCES ${SOURCES} ${ROOT}/examples/shadersDemoContext.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/aabbtree.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/core.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/extrude.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/maths.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/mesh.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/perlin.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/pfm.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/platform.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/png.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/sdf.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/tga.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/voxelize.cpp)
set(SOURCES ${SOURCES} ${ROOT}/examples/opengl/imguiRenderGL.cpp)
set(SOURCES ${SOURCES} ${ROOT}/examples/opengl/shader.cpp)
set(SOURCES ${SOURCES} ${ROOT}/examples/opengl/shadersGL.cpp)

set(FLEXHEADERS ${ROOT})
set(FLEXLIB ${ROOT} ${ROOT}/lib/linux64 ${ROOT}/examples ${ROOT}/external/SDL2-2.0.4/lib/x64)

add_executable(${EXAMPLE_BIN} ${SOURCES})
target_include_directories(${EXAMPLE_BIN} PRIVATE ${FLEXHEADERS})
target_include_directories(${EXAMPLE_BIN} PRIVATE ${FLEXLIB})
target_link_libraries(${EXAMPLE_BIN} ${ROOT}/lib/linux64/NvFlexExtReleaseCUDA_x64.a)
target_link_libraries(${EXAMPLE_BIN} ${ROOT}/lib/linux64/NvFlexReleaseCUDA_x64.a)
target_link_libraries(${EXAMPLE_BIN} ${ROOT}/external/SDL2-2.0.4/lib/x64/libSDL2.a)
target_link_libraries(${EXAMPLE_BIN} ${ROOT}/external/SDL2-2.0.4/lib/x64/libSDL2main.a)
target_link_libraries(${EXAMPLE_BIN} ${CUDA_LIBRARIES})

