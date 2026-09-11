# Omatts.cpp — Single-file C++ TTS runtime using ONNX Runtime
# Based on PocketTTS.cpp — https://github.com/VolgaGerm/PocketTTS.cpp
#
# Build:
#   cmake -B .build -DCMAKE_BUILD_TYPE=Release
#   cmake --build .build -j$(nproc)
#
# Clean:
#   rm -rf .build
#
# Run:
#   ./omatts -v dhh -o output.wav "Hello world"

cmake_minimum_required(VERSION 3.28)
project(omatts VERSION 1.0.0 LANGUAGES CXX C)

# URL of the install script, shown when models are missing.
# Override: cmake -B .build -DOMATTS_INSTALL_URL=https://raw.githubusercontent.com/<user>/omatts/master/install.sh
set(OMATTS_INSTALL_URL "https://raw.githubusercontent.com/OWNER/omatts/master/install.sh" CACHE STRING "Install script URL for the missing-models error")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# =============================================================================
# Dependencies
# =============================================================================
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ONNX Runtime (pre-built) - https://github.com/microsoft/onnxruntime/releases
set(ONNXRUNTIME_VERSION "1.23.2")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(ONNXRUNTIME_ARCH "x64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(ONNXRUNTIME_ARCH "aarch64")
    endif()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
endif()

FetchContent_Declare(onnxruntime
    URL ${ONNXRUNTIME_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)

# Statically bundle ONNX Runtime into the binaries (single self-contained file, Linux only).
# macOS/Windows keep the shared library copied next to the output.
option(BUNDLE_ONNXRUNTIME "Statically link ONNX Runtime (single self-contained binary, Linux)" ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
elseif(BUNDLE_ONNXRUNTIME AND EXISTS ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.a)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.a)
else()
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

if(BUNDLE_ONNXRUNTIME AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
        AND NOT ONNXRUNTIME_LIB MATCHES "\\.a$")
    # No static archive in official ORT releases — embed the shared lib into the
    # executable instead and dlopen it at runtime (single self-contained file).
    set(ONNXRUNTIME_EMBED ON)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/ort_payload.o
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_LIB} ${CMAKE_BINARY_DIR}/ort_payload.so
        COMMAND ${CMAKE_LINKER} -r -b binary -o ort_payload.o ort_payload.so
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${ONNXRUNTIME_LIB}
        COMMENT "Embedding ONNX Runtime (${ONNXRUNTIME_LIB}) into binary"
    )
    add_custom_target(ort_payload DEPENDS ${CMAKE_BINARY_DIR}/ort_payload.o)
else()
    set(ONNXRUNTIME_EMBED OFF)
endif()

# SentencePiece (static library only) - https://github.com/google/sentencepiece/releases
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)

# Patch SentencePiece for GCC 14 compatibility and CMake deprecation warning
file(WRITE ${CMAKE_BINARY_DIR}/patch_sentencepiece.cmake "
file(READ \"\${SOURCE_DIR}/src/sentencepiece_processor.h\" CONTENT)
if(NOT CONTENT MATCHES \"#include <cstdint>\")
    string(REPLACE \"#include <string>\" \"#include <string>\\n#include <cstdint>\" CONTENT \"\${CONTENT}\")
    file(WRITE \"\${SOURCE_DIR}/src/sentencepiece_processor.h\" \"\${CONTENT}\")
    message(STATUS \"Patched sentencepiece_processor.h for GCC 14 compatibility\")
endif()
file(READ \"\${SOURCE_DIR}/CMakeLists.txt\" CMLCONTENT)
string(REGEX REPLACE \"cmake_minimum_required\\\\(VERSION [0-9.]+\\\\)\" \"cmake_minimum_required(VERSION 3.14)\" CMLCONTENT \"\${CMLCONTENT}\")
file(WRITE \"\${SOURCE_DIR}/CMakeLists.txt\" \"\${CMLCONTENT}\")
")

FetchContent_Declare(sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.1
    GIT_SHALLOW TRUE
    PATCH_COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR> -P ${CMAKE_BINARY_DIR}/patch_sentencepiece.cmake
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(sentencepiece)
target_compile_options(sentencepiece-static PRIVATE -w)

set(SENTENCEPIECE_INCLUDE_DIRS "${sentencepiece_SOURCE_DIR}/src")

# dr_libs (header-only) - https://github.com/mackron/dr_libs
FetchContent_Declare(dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dr_libs)

# =============================================================================
# Targets
# =============================================================================

option(BUILD_SHARED_LIB "Build libomatts.so for FFI (Deno, Python, etc.)" OFF)

# Executable (CLI + HTTP server)
add_executable(omatts omatts.cpp)

target_compile_definitions(omatts PRIVATE OMATTS_INSTALL_URL="${OMATTS_INSTALL_URL}")

if(ONNXRUNTIME_EMBED)
    add_dependencies(omatts ort_payload)
    target_compile_definitions(omatts PRIVATE PTT_BUNDLE_ORT)
    target_sources(omatts PRIVATE ${CMAKE_BINARY_DIR}/ort_payload.o)
endif()

target_include_directories(omatts PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${SENTENCEPIECE_INCLUDE_DIRS}
    ${dr_libs_SOURCE_DIR}
)

target_link_libraries(omatts PRIVATE
    $<$<NOT:$<BOOL:${ONNXRUNTIME_EMBED}>>:${ONNXRUNTIME_LIB}>
    sentencepiece-static
)

set_target_properties(omatts PROPERTIES
    BUILD_RPATH "$ORIGIN:$ORIGIN/../lib:/usr/local/lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:/usr/local/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(omatts PRIVATE pthread dl)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(omatts PRIVATE ws2_32)
endif()

# Copy ONNX runtime shared library next to outputs (skipped when statically bundled)
if(NOT ONNXRUNTIME_STATIC)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_custom_command(TARGET omatts POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll
                $<TARGET_FILE_DIR:omatts>
        )
    else()
        add_custom_command(TARGET omatts POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_LIB} $<TARGET_FILE_DIR:omatts>
            COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.so $<TARGET_FILE_DIR:omatts>/libonnxruntime.so.1
        )
    endif()
endif()

# Shared library (for FFI from Deno/Python)
# Build with: cmake -B .build -DBUILD_SHARED_LIB=ON
if(BUILD_SHARED_LIB)
    add_library(omatts-lib SHARED omatts.cpp)
    set_target_properties(omatts-lib PROPERTIES
        OUTPUT_NAME "omatts"
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

    target_compile_definitions(omatts-lib PRIVATE PTT_SHARED_LIB)

    target_link_libraries(omatts-lib PRIVATE
        ${ONNXRUNTIME_LIB}
        sentencepiece-static
    )

    target_include_directories(omatts-lib PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${SENTENCEPIECE_INCLUDE_DIRS}
        ${dr_libs_SOURCE_DIR}
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(omatts-lib PRIVATE pthread dl)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_link_libraries(omatts-lib PRIVATE ws2_32)
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        add_custom_command(TARGET omatts-lib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll
                $<TARGET_FILE_DIR:omatts-lib>
        )
    else()
        add_custom_command(TARGET omatts-lib POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_LIB} $<TARGET_FILE_DIR:omatts-lib>
        )
        find_program(PATCHELF patchelf)
        if(PATCHELF)
            add_custom_command(TARGET omatts-lib POST_BUILD
                COMMAND ${PATCHELF} --replace-needed libonnxruntime.so.1 libonnxruntime.so $<TARGET_FILE:omatts-lib>
                COMMENT "Patching libonnxruntime SONAME dependency"
            )
        endif()
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== Omatts ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_VERSION}")
message(STATUS "SentencePiece: v0.2.1")
if(BUILD_SHARED_LIB)
    message(STATUS "Targets: omatts (CLI), libomatts.so (FFI)")
else()
    message(STATUS "Targets: omatts (CLI)")
endif()
message(STATUS "=====================")
message(STATUS "")
