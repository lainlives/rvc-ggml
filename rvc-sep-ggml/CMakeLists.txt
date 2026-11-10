# RVC-SeP-GGML — Roformer source separation, adapted to the RVC-GGML monorepo.
#
# Upstream: BSRoformer.cpp (MIT, Copyright (c) 2026 沉默の金 <cmzj@cmzj.org>),
# adapted copy at work/Roformer.cpp; WAV I/O replaced (dr_wav instead of libav).
#
# Build (standalone):
#   cmake -S rvc-sep-ggml -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
# cmake --build build
# When added from rvc-train-ggml, the ggml target provided by rvc-ggml is reused.

cmake_minimum_required(VERSION 3.20)

project(RVC-SeP-GGML VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# ================================================================
# Options
# ================================================================
option(RVC_SEP_VULKAN "use ggml Vulkan backend" ON)
option(RVC_SEP_CUDA   "use ggml CUDA backend"   OFF)
set(RVC_SEP_GGML_DIR "" CACHE PATH "local ggml source dir (highest priority)")

# ================================================================
# ggml dependency resolution (mirrors rvc-ggml; when this project is
# added from rvc-train-ggml, the existing ggml target is reused).
# Note: the RVC N-split patch is not applied here — roformer inference
# does not hit the long-audio VITS matmul shapes.
# ================================================================
if(NOT TARGET ggml)
    set(GGML_VULKAN ${RVC_SEP_VULKAN} CACHE BOOL "ggml: use Vulkan" FORCE)
    set(GGML_CUDA   ${RVC_SEP_CUDA}   CACHE BOOL "ggml: use CUDA"   FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "static ggml" FORCE)

    if(RVC_SEP_GGML_DIR AND EXISTS "${RVC_SEP_GGML_DIR}/CMakeLists.txt")
        message(STATUS "[sep] using local ggml: ${RVC_SEP_GGML_DIR}")
        add_subdirectory("${RVC_SEP_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
    else()
        find_package(ggml QUIET CONFIG)
        if(ggml_FOUND)
            message(STATUS "[sep] using installed ggml package")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master/CMakeLists.txt")
            message(STATUS "[sep] using sibling ggml: ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master")
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/CMakeLists.txt")
            message(STATUS "[sep] using sibling ggml: ${CMAKE_CURRENT_SOURCE_DIR}/../ggml")
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
        else()
            message(STATUS "[sep] local ggml not found, fetching official v0.23.0")
            if(POLICY CMP0169)
                cmake_policy(SET CMP0169 OLD)
            endif()
            include(FetchContent)
            FetchContent_Declare(ggml
                URL https://codeload.github.com/ggml-org/ggml/tar.gz/refs/tags/v0.23.0
                DOWNLOAD_EXTRACT_TIMESTAMP ON
            )
            FetchContent_GetProperties(ggml)
            if(NOT ggml_POPULATED)
                FetchContent_Populate(ggml)
                add_subdirectory("${ggml_SOURCE_DIR}" "${ggml_BINARY_DIR}" EXCLUDE_FROM_ALL)
            endif()
        endif()
    endif()
endif()

# ================================================================
# Core library + CLI
# ================================================================
find_package(Threads REQUIRED)

add_library(rvc_sep_core STATIC
    src/model.cpp
    src/inference.cpp
    src/audio.cpp
)
target_include_directories(rvc_sep_core PUBLIC include src)
target_link_libraries(rvc_sep_core PUBLIC ggml Threads::Threads)

if(MSVC)
    target_compile_options(rvc_sep_core PRIVATE /W3)
endif()

add_executable(rvc-sep cli/main.cpp)
target_link_libraries(rvc-sep PRIVATE rvc_sep_core)

if(MSVC)
    target_compile_options(rvc-sep PRIVATE /W3)
endif()
