#include <iostream>
#include <string>
#include <eigen/Eigen/Dense>
#include <fstream>
#include "../../include/Network.hpp"
#include "../../FileStorage/NetworkSaver.hpp"
#include "MNISTTrainerV2.hpp"
#include <chrono>
#include <thread>         // std::thread
#include <pthread.h>         // std::thread

using namespace std;
using namespace Eigen;

int MNISTTrainerV2::PICTURESIZE = 784;
int MNISTTrainerV2::IMAGEHEADERSIZE = 16;
int MNISTTrainerV2::LABELHEADERSIZE = 8;

MatrixXd MNISTTrainerV2::quickReadFirstImageToMatrix(){
    string imageFileAddress = "Images/train-images-idx3-ubyte";
    fstream imageFileStream(imageFileAddress, ios::in|ios::binary);

    imageFileStream.seekg(0,ios::beg);
    char header[IMAGEHEADERSIZE];
    imageFileStream.read(header, IMAGEHEADERSIZE);

    MatrixXd image;

    image = readOneImageToMatrix(&imageFileStream);

    imageFileStream.close();

    return image;
}

void MNISTTrainerV2::quickDrawFirstImage(){
    string imageFileAddress = "Images/train-images-idx3-ubyte";
    fstream imageFileStream(imageFileAddress, ios::in|ios::binary);

    imageFileStream.seekg(0, ios::beg);
    char header[IMAGEHEADERSIZE];
    imageFileStream.read(header, IMAGEHEADERSIZE);

    char image[PICTURESIZE];

    readOneImageToCharArray(&imageFileStream,image);
    
    imageFileStream.close();

    printImage(image);

}

MatrixXd MNISTTrainerV2::readOneImageToMatrix(fstream * imageFileStream){
    char buffer[PICTURESIZE];

    MatrixXd image(PICTURESIZE,1);

    imageFileStream->read(buffer, PICTURESIZE);

    for(int pixel = 0; pixel < 784;pixel++){
        image(pixel) = (int)buffer[pixel];
    }
    
    return image;
}

MatrixXd MNISTTrainerV2::convertCharArrayToImageMatrix(char * buffer){

    MatrixXd image(PICTURESIZE,1);

    for(int pixel = 0; pixel < 784;pixel++){
        image(pixel) = (int)buffer[pixel];
    }

    return image;
}

char * MNISTTrainerV2::readOneImageToCharArray(fstream * imageFileStream, char * buffer){
    char c;
    if(imageFileStream->get(c))
        imageFileStream->seekg(-1,ios::cur);
    else
        cout<<"EndOfImageFile"<<endl;
    imageFileStream->read(buffer, PICTURESIZE);
    return buffer;              
}

int MNISTTrainerV2::readOneLabel(fstream * labelFileStream){
    char c;
    if(!labelFileStream->get(c))
        cout<<"EndOfLabelFile"<<endl;

    return c;
}

int * MNISTTrainerV2::readLabelsIntoArray(fstream * labelFileStream, int numberOfLabelsToRead){
    char c;
    int * buffer = new int[numberOfLabelsToRead];

    for(int i = 0;i < numberOfLabelsToRead;i++){
        labelFileStream->get(c);
        buffer[i] = c;
    }

    return buffer;
}

void MNISTTrainerV2::printImage(char * buffer, int label){
    unsigned char picture[28][28];
    unsigned char currentByte;

    int row = 0;
    int col = -1;

    int startRead;
    int endRead;

    //(4) read the current image (p) into the array
        //determine the start and end bytes
        //read picture from startRead to endRead (byte locations) into picture array
        for(int i = 0;i < 784;i++){
            currentByte = buffer[i];
            if(row<28){
                if(col < 27){
                    col++;
                }else{
                    col = 0;
                    row++;
                }
                //cout<<row<<" "<<col<<endl;
                if(row < 28 && col <28)
                    picture[row][col] = currentByte;
            }
        }

    //(5) print information gathered onto command line
        string line = "";
        string displacement = "";
        int length;

        //draw image from picture array
        for(int i = 0;i < 29;i++){
            for(int x = 0;x < 28;x++){
                if(i < 28){
                    if(picture[i][x] != 0){
                        length = to_string((int)(picture[i][x])).length();

                        for(int r = 0;r < 4-length;r++)
                            displacement += " ";
                        line += to_string((int)(picture[i][x])) + displacement;
                        displacement = "";
                    }else{
                        line += "    ";
                    }
                }else
                    line += "----";
            }

            if(i < 28){
                line += "|";
                if(label != -1)
                    line += to_string(label);
                if(i<10)
                    cout<<i<<":  "<<line<<endl;
                else
                    cout<<i<<": "<<line<<endl;
            }else{
                cout<<"    "<<line<<endl;
            }
            line = "";
        }

}

fstream MNISTTrainerV2::initiateImageFileStream(string imageFileAddress){
    fstream imageFileStream(imageFileAddress, ios::in|ios::binary);

    imageFileStream.seekg(0,ios::beg);
    char header[IMAGEHEADERSIZE];
    imageFileStream.read(header, IMAGEHEADERSIZE);

    return imageFileStream;
}

fstream MNISTTrainerV2::initiateLabelFileStream(string labelFileAddress){
    char header[LABELHEADERSIZE];

    fstream labelFileStream(labelFileAddress, ios::in|ios::binary);
    
    labelFileStream.seekg(0,ios::beg);
    labelFileStream.read(header, LABELHEADERSIZE);

    return labelFileStream;
}

void MNISTTrainerV2::printTimeSignature(chrono::steady_clock::time_point current, chrono::steady_clock::time_point begin, int iterationNumber, int totalNumberOfIterations){
    chrono::steady_clock::time_point previous = current;
    current = chrono::steady_clock::now();

    string secondsSinceLastPercentage = to_string(chrono::duration_cast<chrono::seconds>(current-previous).count());
    string timeSignatureMinutes = to_string(chrono::duration_cast<chrono::minutes>(current-begin).count()) + ":";
    string timeSignatureSeconds = to_string(chrono::duration_cast<chrono::seconds>(current-begin).count()%60);
    if(timeSignatureSeconds.length() < 2)
        timeSignatureSeconds = "0" + timeSignatureSeconds;

    cout<<(((double)iterationNumber)/((double)totalNumberOfIterations))*100.0<<"% Trained | " << secondsSinceLastPercentage << "s | " << timeSignatureMinutes + timeSignatureSeconds <<" Minutes Elapsed"<<endl;
}

void backpropogate(Network * net, MatrixXd testInput, MatrixXd desiredOutput, vector<MatrixXd> * derivativeVector){
    net->backpropogate(testInput, desiredOutput, derivativeVector);
}

vector<MatrixXd> vectorAverage(vector<MatrixXd> * d1,vector<MatrixXd> * d2,vector<MatrixXd> * d3,vector<MatrixXd> * d4,vector<MatrixXd> * d5){
    vector<MatrixXd> da;
    MatrixXd average;
    int lengh = d1[0].size()-1;

    cout<<"BRUV"<<endl;

    cout<<d1[0].size()<<endl;
    cout<<d2[0].size()<<endl;
    cout<<d3[0].size()<<endl;
    cout<<d4[0].size()<<endl;
    cout<<d5[0].size()<<endl;

    cout<<"Columns:"<<endl;
    cout<<d1[0][lengh].cols()<<endl;
    cout<<d2[0][lengh].cols()<<endl;
    cout<<d3[0][lengh].cols()<<endl;
    cout<<d4[0][lengh].cols()<<endl;
    cout<<d5[0][lengh].cols()<<endl;

    cout<<"Rows:"<<endl;
    cout<<d1[0][lengh].rows()<<endl;
    cout<<d2[0][lengh].rows()<<endl;
    cout<<d3[0][lengh].rows()<<endl;
    cout<<d4[0][lengh].rows()<<endl;
    cout<<d5[0][lengh].rows()<<endl;

    for(int i = lengh;i >= 0;i--){
        cout<<i<<endl;
        average = (d1[0][i]+d2[0][i]+d3[0][i]+d4[0][i]+d5[0][i])/5.0;
        da.insert(da.begin(),average);
    }

    return da;
}


void MNISTTrainerV2::TrainOnMNIST(string saveToDirectory, string getFrom, int count, int batchSize, double learningRate){

    //(1) Set up image reading
    fstream imageFileStream = initiateImageFileStream("Images/train-images-idx3-ubyte");
    int imagePointerBegginningPosition = imageFileStream.tellg();
    int currentImagePointerPosition = imagePointerBegginningPosition;
    MatrixXd image;

    //(2) Set up label reading
    fstream labelFileStream = initiateLabelFileStream("Images/train-labels-idx1-ubyte");
    int labelPointerBegginningPosition = labelFileStream.tellg();
    int currentLabelPointerPosition = labelPointerBegginningPosition;
    int label;

    //(3) Set up network for training
    Network * net;

    if(getFrom != "")
        net = NetworkSaver::NetworkGetter(getFrom.c_str());
    else{
        net = new Network();
        cout<<"Creating Netowrk"<<endl;
    }

    //(4) Set up training parameters
    int numberOfEpochs = 5;
    int iterationsPerBatch = 20;

    int NUMBEROFIMAGES = 60000;
    int NUMBEROFBATCHES = NUMBEROFIMAGES/batchSize;

    if(NUMBEROFIMAGES%batchSize != 0) throw invalid_argument("Batch size is not compatible with the 60000 images");

    //(4a) Set up time signature parameters    
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    chrono::steady_clock::time_point current = begin;
    int totalNumberOfIterations = NUMBEROFBATCHES*numberOfEpochs;
    int iterationCount = 0;

    //(5) Set up vectors of Matrices to store the derivatives calculated in seperate threads
    vector<MatrixXd> * d1Holder;
    vector<MatrixXd> * d2Holder;
    vector<MatrixXd> * d3Holder;
    vector<MatrixXd> * d4Holder;
    vector<MatrixXd> * d5Holder;
    batchSize = 5;

    //(5) Train Network
    for(int epoch = 1;epoch<=numberOfEpochs;epoch++){
        
        //(5a) set filestream pointer positions to first images
        imageFileStream.seekg(imagePointerBegginningPosition, ios::beg);
        labelFileStream.seekg(labelPointerBegginningPosition, ios::beg);

        for(int batchNumber = 0;batchNumber < NUMBEROFBATCHES;batchNumber++){
            //for(int batchIteration = 0;batchIteration < iterationsPerBatch; batchIteration++){

                d1Holder = new vector<MatrixXd>[1];
                d2Holder = new vector<MatrixXd>[1];
                d3Holder = new vector<MatrixXd>[1];
                d4Holder = new vector<MatrixXd>[1];
                d5Holder = new vector<MatrixXd>[1];
                
                cout<<"Threads initiated"<<endl;
                //thread ONE
                image = readOneImageToMatrix(&imageFileStream)/255.0; //The 255 is there to normalize the data
                label = readOneLabel(&labelFileStream);
                thread t1(backpropogate,net,image,MatrixXd::Constant(1,1,label), d1Holder);

                //thread TWO
                image = readOneImageToMatrix(&imageFileStream)/255.0; //The 255 is there to normalize the data
                label = readOneLabel(&labelFileStream);
                thread t2(backpropogate,net,image,MatrixXd::Constant(1,1,label), d2Holder);

                //thread THREE
                image = readOneImageToMatrix(&imageFileStream)/255.0; //The 255 is there to normalize the data
                label = readOneLabel(&labelFileStream);
                thread t3(backpropogate,net,image,MatrixXd::Constant(1,1,label), d3Holder);

                //thread FOUR
                image = readOneImageToMatrix(&imageFileStream)/255.0; //The 255 is there to normalize the data
                label = readOneLabel(&labelFileStream);
                thread t4(backpropogate,net,image,MatrixXd::Constant(1,1,label), d4Holder);

                //thread FIVE
                image = readOneImageToMatrix(&imageFileStream)/255.0; //The 255 is there to normalize the data
                label = readOneLabel(&labelFileStream);
                thread t5(backpropogate,net,image,MatrixXd::Constant(1,1,label), d5Holder);

                t1.join();
                t2.join();
                t3.join();
                t4.join();
                t5.join();

                cout<<"Joined"<<endl;

                vector<MatrixXd> dA = vectorAverage(d1Holder,d2Holder,d3Holder,d4Holder,d5Holder);

                net->updateWeights(dA);
                
                // currentImagePointerPosition = imagePointerBegginningPosition + batchNumber*batchSize*PICTURESIZE;
                // currentLabelPointerPosition = labelPointerBegginningPosition + batchNumber*batchSize;

                // //set pointers back to the beginning of the batch
                // imageFileStream.seekg(currentImagePointerPosition, ios::beg);
                // labelFileStream.seekg(currentLabelPointerPosition, ios::beg);
            //}

            iterationCount++;

            //(5b) Print the percentage and time elapsed of the training process every 3 batches
            if((iterationCount) % 3 == 0){
                printTimeSignature(current, begin, iterationCount, totalNumberOfIterations);
                current = chrono::steady_clock::now();
            }
        }
        
        //(5c) Save the network every 1 epochs
        if((epoch) % 1 == 0){
            NetworkSaver::SaveNetwork((saveToDirectory + "/Network" + to_string(count) + ".db").c_str(), net,true);
            count++;
        }
    }
    
    NetworkSaver::SaveNetwork((saveToDirectory + "/FinalNetwork.db").c_str(), net, true);
}