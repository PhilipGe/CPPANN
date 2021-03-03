#pragma once
#include <string>
#include <eigen/Eigen/Dense>
#include <fstream>
#include "../../include/Properties.hpp"
#include <chrono>

using namespace std;
using namespace Eigen;

class MNISTTrainerV2{

    public:
        static int PICTURESIZE;
        static int IMAGEHEADERSIZE;
        static int LABELHEADERSIZE;

        static MatrixXd readOneImageToMatrix(fstream * imageFileStream);
        static char * readOneImageToCharArray(fstream * imageFileStream, char * buffer);
        static MatrixXd convertCharArrayToImageMatrix(char * buffer);

        static int readOneLabel(fstream * labelFileStream);
        static int * readLabelsIntoArray(fstream * labelFileStream, int numberOfLabelsToRead);

        static MatrixXd quickReadFirstImageToMatrix();
        static void quickDrawFirstImage();

        static fstream initiateImageFileStream(string imageFileAddress);
        static fstream initiateLabelFileStream(string labelFileAddress);

        static void printImage(char * buffer, int label = -1);
        static void printTimeSignature(chrono::steady_clock::time_point current, chrono::steady_clock::time_point begin, int iterationNumber, int numberOfIterations);

        static void TrainOnMNIST(string saveTo, string getFrom = "", int count = 0, int batchSize = 1, double learningRate = Properties::learningSpeed, bool constrain = false);
        static MatrixXd getOutputMatrix(int output, int numberOfDifferentOutputs = 10);
};