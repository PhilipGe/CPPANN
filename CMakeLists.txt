cmake_minimum_required(VERSION 3.18.4)

project(CPPANN)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_library(
    Network
    include/Layer.hpp
    include/Network.hpp
    include/Properties.hpp
    include/Trainer.hpp
    src/Layer.cpp
    src/Network.cpp
    src/Trainer.cpp
)

add_library(
    File-Storage
    FileStorage/NetworkSaver.cpp
    FileStorage/NetworkSaver.hpp
)

add_library(
    supporting
    MNIST_Processing/support/MNISTTester.cpp
    MNIST_Processing/support/MNISTTester.hpp
    MNIST_Processing/support/MNISTTrainerV2.cpp
    MNIST_Processing/support/MNISTTrainerV2.hpp
)


target_include_directories(Network PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(File-Storage PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/FileStorage")
target_include_directories(File-Storage PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/MNIST_Processing/support")

add_executable(MNIST MNIST_Processing/TestTrainer.cpp)

target_link_libraries(MNIST PRIVATE sqlite3 pthread Network File-Storage supporting)

# add_subdirectory(googletest)
# include_directories(googletest/googletest/include src test)
# add_executable(basicFunctionsTests test/basicFunctionsTests.cpp)
# target_link_libraries(basicFunctionsTests PRIVATE gtest)

# include_directories(googletest/googletest/include src FileStorage)
# add_executable(trainingTests test/trainingTests.cpp src/Trainer.cpp)
# target_link_libraries(trainingTests PRIVATE gtest sqlite3)

# option(MAIN "Build driver (main.cpp)" ON)
# option(TEST "Build basicTests (basicFunctionsTests.cpp)" ON)
# option(BASIC_FUNCTION_TESTS "Build basicTests (basicFunctionsTests.cpp)" ON)
# option(TRAINING_TESTS "Build trainingTests (trainingTests.cpp)" ON)

# if(MAIN)
#     add_executable(main src/main.cpp src/Layer.cpp src/Network.cpp src/Trainer.cpp)
# endif()

# if(TEST)
#     add_subdirectory(googletest)
#     include_directories(googletest/googletest/include src test)
#     if(BASIC_FUNCTION_TESTS)
#         add_executable(basicFunctionsTests test/basicFunctionsTests.cpp)
#         target_link_libraries(basicFunctionsTests PRIVATE gtest)
#     endif()
#     if(TRAINING_TESTS)
#         include_directories(googletest/googletest/include src FileStorage)
#         add_executable(test test/trainingTests.cpp src/Trainer.cpp)
#         target_link_libraries(test PRIVATE gtest sqlite3)
#     endif()
# endif()