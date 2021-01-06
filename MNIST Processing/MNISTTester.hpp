#pragma once
#include <iostream>
#include "MNISTTrainerV2.hpp"
#include <eigen/Eigen/Dense>
#include "../include/Network.hpp"
#include "../FileStorage/NetworkSaver.cpp"
#include <sqlite3.h>
#include <string>

class MNISTTester{

    public:
        static MatrixXd * TestImagesOnDatabase(string databaseAddress, string imagesAddress, string labelsAddress, int numberOfImagesToRead, bool printImages = false);
};