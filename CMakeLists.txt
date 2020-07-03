cmake_minimum_required(VERSION 3.0.0)
project(guetzli C CXX)

include(CheckCXXCompilerFlag)

find_package(PNG REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}" "${PNG_INCLUDE_DIRS}" "third_party/butteraugli")

CHECK_CXX_COMPILER_FLAG("-std=c++11" GUETZLI_COMPILER_HAS_C11_FLAG)
if (GUETZLI_COMPILER_HAS_C11_FLAG)
  list(APPEND CMAKE_CXX_FLAGS "-std=c++11")
endif()

add_subdirectory(third_party)
add_subdirectory(guetzli)
