cmake_minimum_required(VERSION 3.25)
project(Ejemplo01)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include(FetchContent)
include(cmake/CPM.cmake)

#https://github.com/fmtlib/fmt.git
CPMAddPackage("gh:fmtlib/fmt#10.0.0")

#https://github.com/abdes/cryptopp-cmake.git
CPMAddPackage("gh:abdes/cryptopp-cmake#v8.8.0")

FetchContent_MakeAvailable(fmt)

add_executable(Ejemplo01 main_serial.cpp)
target_link_libraries(Ejemplo01 fmt::fmt cryptopp)

add_executable(Ej2 main_omp.cpp)
target_link_libraries(Ej2 fmt::fmt cryptopp)