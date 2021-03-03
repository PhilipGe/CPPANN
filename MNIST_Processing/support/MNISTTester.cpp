#include <iostream>
#include "MNISTTrainerV2.hpp"
#include <eigen/Eigen/Dense>
#include "../../include/Network.hpp"
#include "../../FileStorage/NetworkSaver.cpp"
#include <sqlite3.h>
#include <string>
#include "MNISTTester.hpp"

int outputValue(MatrixXd outputMatrix){

    double maxC = outputMatrix(0,0);
    int result;

    for(int x = 0;x < outputMatrix.rows();x++){
        if(outputMatrix(x,0) >= maxC){
            maxC = outputMatrix(x,0);
            result = x;
        }
    }

    return result;
}

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
    
    int correctCount = 0;
    double cost = 0;
    //(6) Train Network
    for(int i = 0;i < NUMBEROFIMAGESTOREAD;i++){
        MNISTTrainerV2::readOneImageToCharArray(&imageFileStream, charImage);
        image = MNISTTrainerV2::convertCharArrayToImageMatrix(charImage)/255; //The 255 is there to normalize the data
        label = MNISTTrainerV2::readOneLabel(&labelFileStream);
        if(printImages) MNISTTrainerV2::printImage(charImage, label);

        output = net->feedForward(image);
        returnArray[i] = output;

        MatrixXd loss = output-MNISTTrainerV2::getOutputMatrix(label);

        cost += (loss.array()*loss.array()).sum()/((double)NUMBEROFIMAGESTOREAD);

        // cout<<label<<" | "<<outputValue(output)<<endl;
        if(label == outputValue(output))
            correctCount++;
        // cout<<"Desired: "<<label<<" | Actual: "<<outputValue(output)<<endl;
        // cout<<output<<endl;
    }

    cout << ((double)correctCount)/((double)numberOfImagesToRead)<<endl;
    cout<<"Cost: "<<cost<<endl;

    imageFileStream.close();
    labelFileStream.close();

    return returnArray;
}