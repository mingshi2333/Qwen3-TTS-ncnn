cmake_minimum_required(VERSION 3.18)
project(qwen3-tts-ncnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ncnn: pass -Dncnn_DIR=<prefix>/lib64/cmake/ncnn (or lib/cmake/ncnn) for a
# local build; otherwise a pinned revision is fetched and built. A system ncnn
# is deliberately NOT auto-detected: distro packages are typically too old for
# the SDPA kv-cache / RotaryEmbed layers and fail only at runtime.
if(DEFINED ncnn_DIR AND ncnn_DIR)
  find_package(ncnn REQUIRED)
endif()
if(NOT ncnn_FOUND)
  include(FetchContent)
  set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
  set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(ncnn
    GIT_REPOSITORY https://github.com/Tencent/ncnn.git
    GIT_TAG 244f30c8)  # needs SDPA kv_cache (>= #6405) — keep pinned
  FetchContent_MakeAvailable(ncnn)
endif()

add_library(qwen3tts STATIC
  src/net_utils.cpp
  src/tts_pipeline.cpp
  src/qwen_bpe.cpp
)
target_include_directories(qwen3tts PUBLIC src)
target_link_libraries(qwen3tts PUBLIC ncnn)

add_executable(smoke_talker examples/smoke_talker.cpp)
target_link_libraries(smoke_talker PRIVATE qwen3tts)

add_executable(e2e_codes examples/e2e_codes.cpp)
target_link_libraries(e2e_codes PRIVATE qwen3tts)

add_executable(test_tokenizer examples/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE qwen3tts)

add_executable(tts_cli examples/tts_cli.cpp)
target_link_libraries(tts_cli PRIVATE qwen3tts)

add_executable(make_xvec examples/make_xvec.cpp)
target_link_libraries(make_xvec PRIVATE qwen3tts)

add_executable(test_sampler examples/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE qwen3tts)

add_executable(e2e_icl examples/e2e_icl.cpp)
target_link_libraries(e2e_icl PRIVATE qwen3tts)

add_executable(bench examples/bench.cpp)
target_link_libraries(bench PRIVATE qwen3tts)

add_executable(vk_probe examples/vk_probe.cpp)
target_link_libraries(vk_probe PRIVATE qwen3tts)

add_executable(e2e_cv examples/e2e_cv.cpp)
target_link_libraries(e2e_cv PRIVATE qwen3tts)

if(MSVC)
  target_compile_options(qwen3tts PUBLIC /utf-8)
  target_compile_definitions(qwen3tts PUBLIC NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

enable_testing()
add_test(NAME tokenizer COMMAND test_tokenizer ${CMAKE_SOURCE_DIR}/models/tokenizer.txt ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_test.bin)
add_test(NAME sampler   COMMAND test_sampler   ${CMAKE_SOURCE_DIR}/tests/data/sampler_test.bin)
# model-dependent parity gate, only registered when converted nets are present
if(EXISTS ${CMAKE_SOURCE_DIR}/models/talker_decoder.ncnn.param)
  add_test(NAME e2e_greedy COMMAND e2e_codes ${CMAKE_SOURCE_DIR}/models ${CMAKE_SOURCE_DIR}/tests/data 2055)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/models-cv/talker_decoder.ncnn.param)
  add_test(NAME e2e_customvoice COMMAND e2e_cv ${CMAKE_SOURCE_DIR}/models-cv ${CMAKE_SOURCE_DIR}/tests/data 3066 2055)
endif()
