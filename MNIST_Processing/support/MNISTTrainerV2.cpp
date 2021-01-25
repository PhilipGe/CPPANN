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
#include <future>
#include <eigen/Eigen/Core>

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
    if(labelFileStream.fail()){
        cout<<"FAILED"<<endl;
        throw 32;
    }

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

vector<MatrixXd> backpropogate(Network * net, MatrixXd testInput, double desiredOutput){
    MatrixXd dOInMatrix = MatrixXd::Constant(1,1,desiredOutput);
    return net->backpropogate(testInput, dOInMatrix);
}

vector<MatrixXd> vectorAverage(vector<MatrixXd> d1,vector<MatrixXd> d2,vector<MatrixXd> d3,vector<MatrixXd> d4,vector<MatrixXd> d5){
    vector<MatrixXd> da;
    MatrixXd average;
    int lengh = d1.size()-1;

    for(int i = lengh;i >= 0;i--){
        average = (d1[i]+d2[i]+d3[i]+d4[i]+d5[i])/5.0;
        da.insert(da.begin(),average);
    }

    return da;
}

vector<MatrixXd> futureAverage(vector<future<vector<MatrixXd>>> &futureVector, vector<MatrixXd> weightsVector){ 
    vector<MatrixXd> result;
    vector<MatrixXd> tempDerivatives;// = futureVector[0].get();
    vector<MatrixXd> averageDerivatives;

    int batchSize = futureVector.size();


    for(int x = 0;x < weightsVector.size();x++){
        weightsVector[x] = weightsVector[x]-weightsVector[x]; //resets the values to 0 but maintains the correct dimensions
    }

    for(int i = 1;i < weightsVector.size();i++){      //makes a derivative holder vector that has the same dimensions as the weights vector minus the input layer "weights"
        averageDerivatives.push_back(weightsVector[i]);
    }

    for(int i = 0;i < batchSize;i++){
        tempDerivatives = futureVector[i].get();
        for(int layer = 0;layer < averageDerivatives.size();layer++){
            averageDerivatives[layer] = averageDerivatives[layer] + tempDerivatives[layer];
        }
    }

    for(int layer = 0;layer >= averageDerivatives.size();layer++){
        averageDerivatives[layer] = averageDerivatives[layer]/batchSize;
        for(int y = 0;y < averageDerivatives[layer].rows();y++){
            for(int x = 0;x < averageDerivatives[layer].cols();x++){
                if(isnan(averageDerivatives[layer](y,x)))
                    cout<<"NAN!"<<endl;
            }
        }
    }
    
    

    return averageDerivatives;
}

void MNISTTrainerV2::TrainOnMNIST(string saveToDirectory, string getFrom, int count, int batchSize, double learningRate){

    initParallel();

    //(1) Set up image reading
    fstream imageFileStream = initiateImageFileStream("/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/Images/train-images-idx3-ubyte");
    imageFileStream.seekg(0, ios::beg);
    int imagePointerBegginningPosition = imageFileStream.tellg();
    int currentImagePointerPosition = imagePointerBegginningPosition;
    MatrixXd image;

    //(2) Set up label reading
    fstream labelFileStream = initiateLabelFileStream("/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/Images/train-labels-idx1-ubyte");
    int labelPointerBegginningPosition = labelFileStream.tellg();
    int currentLabelPointerPosition = labelPointerBegginningPosition;
    int label;

    //(3) Set up network for training
    Network * net;

    if(getFrom != "")
        net = NetworkSaver::NetworkGetter(getFrom.c_str());
    else{
        net = new Network();
        cout<<"Creating Network"<<endl;
        NetworkSaver::SaveNetwork((saveToDirectory + "/Network" + to_string(count) + ".db").c_str(), net);
        count++;
    }

    //(4) Set up training parameters
    int numberOfEpochs = 5;
    batchSize = 100;

    int NUMBEROFIMAGES = 60000;
    int NUMBEROFBATCHES = NUMBEROFIMAGES/batchSize;

    if(NUMBEROFIMAGES%batchSize != 0) throw invalid_argument("Batch size is not compatible with the 60000 images");

    //(4a) Set up time signature parameters    
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    chrono::steady_clock::time_point current = begin;
    int totalNumberOfIterations = NUMBEROFBATCHES*numberOfEpochs;
    int iterationCount = 0;

    //initialize variables for more efficient async
    vector<MatrixXd> initialWeights;
    for(int i = 0; i< net->network.size();i++){
        initialWeights.push_back(net->network[i].weights);
    }

    vector<Network*> netVec;
    vector<future<vector<MatrixXd>>> fVec;

    for(int i = 0;i < batchSize;i++){
        netVec.push_back(new Network(*net));
    }

    for(int i = 0;i < batchSize;i++){
        image = readOneImageToMatrix(&imageFileStream)/255.0; //The 255 is there to normalize the data
        label = readOneLabel(&labelFileStream);
        fVec.push_back(async(launch::async, backpropogate, netVec[i], image,label));
    }

    //(5) Train Network
    for(int epoch = 1;epoch<=numberOfEpochs;epoch++){   
        
        //(5a) set filestream pointer positions to first images
        imageFileStream.seekg(0, ios::beg);
        labelFileStream.seekg(0, ios::beg);

        for(int batchNumber = 0;batchNumber < 20;batchNumber++){

            //initiate backpropogation for each image in the batch
            for(int i = 0;i < batchSize;i++){
                image = readOneImageToMatrix(&imageFileStream)/255.0; //The 255 is there to normalize the data
                label = readOneLabel(&labelFileStream);
                fVec[i] = async(launch::async, backpropogate, netVec[i], image,label);
            }

            //get the average of the derivatives calculated
            vector<MatrixXd> dA = futureAverage(fVec, initialWeights);       

            for(int layer = 0;layer < dA.size();layer++){
                cout<<dA[layer]<<endl;
            }        
            
            //update the main network's weights with the derivative
            net->updateWeights(dA);

            //update the weights of the temporary networks that exist to hold the outputs of each thread (not the optimal way to do this but will take a while to fix)
            for(int i = 0;i < batchSize;i++){
                for(int layer = 1;layer<net->network.size();layer++){
                    netVec[i]->network[layer].weights = net->network[layer].weights;
                }
            }

            iterationCount++;

            // (5b) Print the percentage and time elapsed of the training process every 3 batches
            if((iterationCount) % 3 == 0){
                printTimeSignature(current, begin, iterationCount, totalNumberOfIterations);
                current = chrono::steady_clock::now();
                NetworkSaver::SaveNetwork((saveToDirectory + "/Network" + to_string(count) + ".db").c_str(), net, true,true);
                count++;
            }
        }
        
        //(5c) Save the network every 1 epochs
        if((epoch) % 1 == 0){
            NetworkSaver::SaveNetwork((saveToDirectory + "/Network" + to_string(count) + ".db").c_str(), net);
            count++;
        }
    }

    imageFileStream.close();
    labelFileStream.close();
    
    NetworkSaver::SaveNetwork((saveToDirectory + "/FinalNetwork.db").c_str(), net);
}