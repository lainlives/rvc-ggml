# RVC-UI — Qt6 Widgets front-end for the RVC-GGML toolchain.
#
# Build:
#   cmake -S rvc-ui -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Runtime paths: env vars RVC_TOOLS_DIR / RVC_ROOT override the build-time
# defaults (RVC_UI_DEFAULT_TOOLS_DIR / RVC_UI_DEFAULT_ROOT), which override
# auto-detection (next to / above the executable).

cmake_minimum_required(VERSION 3.20)

project(RVC-UI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# ---------------------------------------------------------------- miniaudio (WAV/FLAC/MP3/Ogg Vorbis decode)
set(RVC_UI_MINIAUDIO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/miniaudio"
    CACHE PATH "miniaudio source directory")
if(NOT EXISTS "${RVC_UI_MINIAUDIO_DIR}/miniaudio.h")
    message(FATAL_ERROR "miniaudio not found at ${RVC_UI_MINIAUDIO_DIR}; run: git submodule update --init third_party/miniaudio")
endif()

# ---------------------------------------------------------------- baked-in default paths
set(RVC_UI_DEFAULT_ROOT "" CACHE PATH "default RVC_ROOT baked into the binary (used when the env var is unset)")
set(RVC_UI_DEFAULT_TOOLS_DIR "" CACHE PATH "default RVC_TOOLS_DIR baked into the binary (used when the env var is unset)")

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(rvc-ui
    src/main.cpp
    src/main_window.cpp
    src/song_cover_page.cpp
    src/audio_utils.cpp
    src/miniaudio_impl.cpp
)
target_include_directories(rvc-ui PRIVATE
    src include
    "${RVC_UI_MINIAUDIO_DIR}"
    "${RVC_UI_MINIAUDIO_DIR}/extras"
)
target_link_libraries(rvc-ui PRIVATE Qt6::Widgets)

if(RVC_UI_DEFAULT_ROOT)
    target_compile_definitions(rvc-ui PRIVATE RVC_UI_DEFAULT_ROOT="${RVC_UI_DEFAULT_ROOT}")
endif()
if(RVC_UI_DEFAULT_TOOLS_DIR)
    target_compile_definitions(rvc-ui PRIVATE RVC_UI_DEFAULT_TOOLS_DIR="${RVC_UI_DEFAULT_TOOLS_DIR}")
endif()

if(MSVC)
    target_compile_options(rvc-ui PRIVATE /W3)
endif()

# ---------------------------------------------------------------- tests
option(RVC_UI_BUILD_TESTS "build rvc-ui tests" ON)
if(RVC_UI_BUILD_TESTS)
    enable_testing()
    add_executable(resample_test tests/resample_test.cpp src/audio_utils.cpp src/miniaudio_impl.cpp)
    target_include_directories(resample_test PRIVATE
        src include
        "${RVC_UI_MINIAUDIO_DIR}"
        "${RVC_UI_MINIAUDIO_DIR}/extras"
    )
    add_test(NAME resample COMMAND resample_test)

    add_executable(normalize_test tests/normalize_test.cpp src/audio_utils.cpp src/miniaudio_impl.cpp)
    target_include_directories(normalize_test PRIVATE
        src include
        "${RVC_UI_MINIAUDIO_DIR}"
        "${RVC_UI_MINIAUDIO_DIR}/extras"
    )
    add_test(NAME normalize COMMAND normalize_test)
endif()
