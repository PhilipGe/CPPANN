#include <iostream>
#include "support/MNISTTrainerV2.hpp"
#include "support/MNISTTester.hpp"
#include <eigen/Eigen/Dense>
#include "../include/Network.hpp"
#include "../FileStorage/NetworkSaver.hpp"
#include <sqlite3.h>
#include <string>
#include <random>


using namespace std;
// using namespace Eigen;



int main(){
    
    string saveToDatabaseAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/MNISTTrialNines";
    // string imageAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TestImages/t10k-images-idx3-ubyte";
    // string labelsAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TestImages/t10k-labels-idx1-ubyte";
    
    MNISTTrainerV2::TrainOnMNIST(saveToDatabaseAddress);
    // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/FinalNetwork.db",imageAddress,labelsAddress,100,true);
    
    return 0;
}