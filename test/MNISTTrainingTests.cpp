#include <gtest/gtest.h>
#include "../MNIST_Processing/MNISTTrainerV2.cpp"
#include <iostream>
#include <string>
#include <eigen/Eigen/Dense>
#include <fstream>
#include "../include/Network.hpp"
#include "../FileStorage/NetworkSaver.cpp"
#include "MNISTTrainerV2.hpp"
#include <chrono>
#include <thread>         // std::thread
#include <pthread.h>         // std::thread

TEST (trials, trainerTest){

    string saveToDatabaseAddress = "MNIST_Processing/MNISTTrialFour";

    MNISTTrainerV2::TrainOnMNIST(saveToDatabaseAddress);

}