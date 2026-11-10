# RVC-UI — Qt6 Widgets front-end for the RVC-GGML toolchain.
#
# Build:
#   cmake -S rvc-ui -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# The UI locates the CLI tools (rvc-sep, RVC-GGML) and the models/ tree at
# runtime: RVC_TOOLS_DIR / RVC_ROOT environment variables, or next to / above
# the executable.

cmake_minimum_required(VERSION 3.20)

project(RVC-UI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(rvc-ui
    src/main.cpp
    src/main_window.cpp
    src/song_cover_page.cpp
    src/audio_utils.cpp
)
target_include_directories(rvc-ui PRIVATE src include)
target_link_libraries(rvc-ui PRIVATE Qt6::Widgets)

if(MSVC)
    target_compile_options(rvc-ui PRIVATE /W3)
endif()

# ---------------------------------------------------------------- tests
option(RVC_UI_BUILD_TESTS "build rvc-ui tests" ON)
if(RVC_UI_BUILD_TESTS)
    enable_testing()
    add_executable(resample_test tests/resample_test.cpp src/audio_utils.cpp)
    target_include_directories(resample_test PRIVATE src include)
    add_test(NAME resample COMMAND resample_test)
endif()
