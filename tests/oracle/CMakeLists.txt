cmake_minimum_required(VERSION 3.10)
project(nam_oracle CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(NAMCORE "${CMAKE_CURRENT_SOURCE_DIR}/NeuralAmpModelerCore")
set(NAM_DEPS_PATH "${NAMCORE}/Dependencies")

file(GLOB NAM_SOURCES_TOP "${NAMCORE}/NAM/*.cpp" "${NAMCORE}/NAM/*.c")
file(GLOB NAM_SOURCES_SUB "${NAMCORE}/NAM/*/*.cpp")
set(NAM_SOURCES ${NAM_SOURCES_TOP} ${NAM_SOURCES_SUB})

add_executable(oracle oracle.cpp ${NAM_SOURCES})
target_include_directories(oracle SYSTEM PRIVATE
  "${NAMCORE}"
  "${NAM_DEPS_PATH}/eigen"
  "${NAM_DEPS_PATH}/nlohmann"
  "${NAM_DEPS_PATH}/AudioDSPTools/dsp")

# Use the generic WaveNet (not the a2_fast specialization) so the oracle's
# semantics are exactly the engine we port.
# (NAM_ENABLE_A2_FAST is left undefined -> generic path.)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(oracle PRIVATE stdc++fs)
endif()
target_compile_features(oracle PRIVATE cxx_std_20)
