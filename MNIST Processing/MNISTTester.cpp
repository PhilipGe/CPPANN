#include <iostream>
#include "MNISTTrainerV2.hpp"
#include <eigen/Eigen/Dense>
#include "../include/Network.hpp"
#include "../FileStorage/NetworkSaver.cpp"
#include <sqlite3.h>
#include <string>
#include "MNISTTester.hpp"

MatrixXd * MNISTTester::TestImagesOnDatabase(string databaseAddress, string imagesAddress, string labelsAddress, int numberOfImagesToRead, bool printImages){
    //(1) Set up image reading
    fstream imageFileStream = MNISTTrainerV2::initiateImageFileStream(imagesAddress);
    MatrixXd image;
    char charImage[784];

    //(2) Set up label reading
    fstream labelFileStream = MNISTTrainerV2::initiateLabelFileStream(labelsAddress);
    int label;

    //(3) Get network that is going to be tested
    Network * net = NetworkSaver::NetworkGetter(databaseAddress.c_str());

    //(4) Set up testing parameters
    int NUMBEROFIMAGESTOREAD = numberOfImagesToRead;
    MatrixXd output;

    //(5) Set up return array
    MatrixXd * returnArray= new MatrixXd[NUMBEROFIMAGESTOREAD];
    
    //(6) Train Network
    for(int i = 0;i < NUMBEROFIMAGESTOREAD;i++){
        MNISTTrainerV2::readOneImageToCharArray(&imageFileStream, charImage);
        image = MNISTTrainerV2::convertCharArrayToImageMatrix(charImage)/255; //The 255 is there to normalize the data
        label = MNISTTrainerV2::readOneLabel(&labelFileStream);
        if(printImages) MNISTTrainerV2::printImage(charImage, label);

        output = net->feedForward(image);
        returnArray[i] = output;

        //cout<<"Desired: "<<label<<" | Actual: "<<output<<endl;
    }

    imageFileStream.close();
    labelFileStream.close();

    return returnArray;
}