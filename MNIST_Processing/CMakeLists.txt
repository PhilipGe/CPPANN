add_library(
    support
    "${CMAKE_CURRENT_SOURCE_DIR}/support/MNISTTester.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/support/MNISTTester.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/support/MNISTTrainerV2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/support/MNISTTrainerV2.hpp"
)

add_executable(MNISTTraining TestTrainer.cpp)

target_include_directories(support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/support")
target_link_libraries(MNISTTraining PRIVATE sqlite3 pthread Network File-Storage support)