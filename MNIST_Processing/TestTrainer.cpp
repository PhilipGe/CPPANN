#include <iostream>
#include "support/MNISTTrainerV2.hpp"
#include <eigen/Eigen/Dense>
#include "../include/Network.hpp"
#include "../FileStorage/NetworkSaver.cpp"
#include <sqlite3.h>
#include <string>


using namespace std;
// using namespace Eigen;

int main(){
    
    string saveToDatabaseAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/MNISTTrialEight";
    
    MNISTTrainerV2::TrainOnMNIST(saveToDatabaseAddress);
    
    return 0;
}