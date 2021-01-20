cmake_minimum_required(VERSION 3.18.4)

project(CPPANN)

set(SOURCES ../src/Network.cpp     ../src/Layer.cpp     ../src/Trainer.cpp     ../FileStorage/NetworkSaver.cpp
            ../include/Network.hpp ../include/Layer.hpp ../include/Trainer.hpp ../FileStorage/NetworkSaver.hpp            
)

add_executable(MNISTTraining TestTrainer.cpp ../src)
target_include_directories(MNISTTraining PRIVATE ../include  FileStorage support)
target_link_libraries(MNISTTraining PRIVATE sqlite3 pthread)
