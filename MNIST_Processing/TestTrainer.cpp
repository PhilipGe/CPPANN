#include <iostream>
#include "MNISTTrainerV2.hpp"
#include <eigen/Eigen/Dense>
#include "../include/Network.hpp"
#include "../FileStorage/NetworkSaver.cpp"
#include <sqlite3.h>
#include <string>


using namespace std;
// using namespace Eigen;

int main(){
    
    string saveToDatabaseAddress = "MNISTTrialFour";

    MNISTTrainerV2::TrainOnMNIST(saveToDatabaseAddress);
    
    return 0;
}