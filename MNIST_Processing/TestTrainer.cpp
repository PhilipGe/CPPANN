#include <iostream>
#include "support/MNISTTrainerV2.hpp"
#include "support/MNISTTester.hpp"


using namespace std;

int main(){

    bool train = false;
    bool test = true;
    
    string saveToDatabaseAddress1 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/One";
    string saveToDatabaseAddress2 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Two";
    string saveToDatabaseAddress3 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Three";
    string saveToDatabaseAddress4 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Four";
    string saveToDatabaseAddress5 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Five";
    string saveToDatabaseAddress6 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Six";
    string saveToDatabaseAddress7 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Seven";
    string saveToDatabaseAddress8 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Eight";
    string saveToDatabaseAddress9 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Nine";
    string saveToDatabaseAddress10 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Ten";
    string saveToDatabaseAddress11 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Eleven";
    string saveToDatabaseAddress12 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Twelve";
    string saveToDatabaseAddress13 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Thirteen";
    string saveToDatabaseAddress14 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TrainingTests/MNISTTrialTwentyNine/Fourteen";

    string saveToDatabaseAddress30 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/MNISTTrialTwentySeven/Two";
    string saveToDatabaseAddress = saveToDatabaseAddress7;
    // string saveToDatabaseAddress3 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/MNISTTrialTwentySix";
    string saveToDatabaseAddress19 = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/MNISTTrialNineteen";
    string imageAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/Images/train-images-idx3-ubyte";
    string labelsAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/Images/train-labels-idx1-ubyte";
    string testImageAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TestImages/t10k-images-idx3-ubyte";
    string testLabelsAddress = "/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/TestImages/t10k-labels-idx1-ubyte";
    // testImageAddress = imageAddress;
    // testLabelsAddress = labelsAddress;
    
    //NETWORK 38 glitched (Trial19)

    // Network *net0 = NetworkSaver::NetworkGetter((saveToDatabaseAddress4+"/Network0.db").c_str());
    // Network *net1 = NetworkSaver::NetworkGetter((saveToDatabaseAddress4+"/Network1.db").c_str());
    // Network *net2 = NetworkSaver::NetworkGetter((saveToDatabaseAddress4+"/Network2.db").c_str());

    // net2->printNetworkWeights();

    // Network *net17 = NetworkSaver::NetworkGetter((saveToDatabaseAddress17+"/FinalNetwork.db").c_str());

    // // net10->printNetworkWeights();
    // // cout<<net37->network[1].weights<<endl;
    // cout<<net17->network[1].weights<<endl;
    //net38->printNetworkWeights();
    //-0.657199  0.114092 -0.413585  0.242141 -0.366913  0.248355  -0.82845  0.197113 -0.770209  0.923681 -0.163225  0.516229  0.817738  0.993598 -0.413044  0.039539  0.891505  0.921767  0.095012  -0.02124 -0.567417 -0.268339 -0.370736 -0.851909 -0.871716  0.383813   0.68978  0.964701  0.507595  0.399675  0.839585  0.002514

    if(train){
        // string networkAddress = saveToDatabaseAddress7 + "/Network73.db"; 
        MNISTTrainerV2::TrainOnMNIST(saveToDatabaseAddress30);//,networkAddress,7074);
        // MNISTTrainerV2::TrainOnMNIST(saveToDatabaseAddress2,"",0,500,0.01,true);
    }
    // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network1.db",imageAddress,labelsAddress,50);
    if(test){
        int numberOfImagesToRead = 10000;
        MNISTTester::TestImagesOnDatabase("/home/philip/Desktop/Projects/CPPANN/MNIST_Processing/MNISTTrialTwentySeven/Two/FinalNetwork.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network1.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network2.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network3.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network4.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network50.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network60.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network70.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network80.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network90.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network100.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network110.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network120.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network130.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network140.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network150.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network160.db",testImageAddress,testLabelsAddress,numberOfImagesToRead);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress+"/Network70.db",testImageAddress,testLabelsAddress,10000);
        // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress2+"/FinalNetwork.db",imageAddress,labelsAddress,1000);
    }
    // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress17+"/Network676.db",imageAddress,labelsAddress,50);
    // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress17+"/Network680.db",imageAddress,labelsAddress,50);
    // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress17+"/Network683.db",imageAddress,labelsAddress,50);
    // MNISTTester::TestImagesOnDatabase(saveToDatabaseAddress17+"/Network686.db",imageAddress,labelsAddress,50);
    
    return 0;
}