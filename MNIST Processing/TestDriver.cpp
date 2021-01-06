#include <iostream>
#include "MNISTTrainerV2.hpp"
#include <eigen/Eigen/Dense>
#include "../include/Network.hpp"
#include "../FileStorage/NetworkSaver.cpp"
#include <sqlite3.h>
#include <string>
#include "MNISTTester.hpp"

using namespace std;
// using namespace Eigen;

int main(){

    double numberOfImagesToRead = 60000;
    string postTrainingDatabase = "MNISTTrialThree/FinalNetwork.db";
    string imagesAddress = "Images/train-images-idx3-ubyte";
    string labelsAddress = "Images/train-labels-idx1-ubyte";

    MatrixXd * three = MNISTTester::TestImagesOnDatabase(postTrainingDatabase,imagesAddress,labelsAddress,numberOfImagesToRead);

    fstream labelFileStream = MNISTTrainerV2::initiateLabelFileStream(labelsAddress);
    int * labels = MNISTTrainerV2::readLabelsIntoArray(&labelFileStream, numberOfImagesToRead);

    double count = 0;
    int worngCount = 0;

    for(int i = 0;i < numberOfImagesToRead;i++){
        if(labels[i] == abs(round(three[i](0,0)))) 
            count++;
        else{
            if(worngCount<1) worngCount = i;
        }
    }

    cout<<"Percent Correct: "<<100*count/numberOfImagesToRead<<"% | "<<worngCount<<endl;

    return 0;
}