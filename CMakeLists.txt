cmake_minimum_required(VERSION 3.18.4)

project(CPPANN)

option(MAIN "Build driver (main.cpp)" OFF)
option(TESTS "Build tests (tests.cpp)" ON)

if(MAIN)
    add_executable(main src/main.cpp src/Layer.cpp src/Network.cpp)
endif()

if(TESTS)
    add_subdirectory(googletest)
    include_directories(googletest/googletest/include src)
    add_executable(test test/tests.cpp)
    target_link_libraries(test PRIVATE gtest)
endif()