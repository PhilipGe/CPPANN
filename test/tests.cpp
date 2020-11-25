#include <gtest/gtest.h>
#include <eigen/Eigen/Dense>
#include "../src/Layer.cpp"
#include "../src/Network.cpp"

TEST (network, feedForward){

    Network *net = new Network();

    MatrixXd inputs(3,1);
    inputs << 0.5,-0.5,0;

    int numberOfHiddenLayers = 3;
    int nodesPerLayer = 3;
    int numberOfInputs = 9;
    int learningSpeed = 1;

    vector<MatrixXd> testWeights;

    MatrixXd layerOneWeights(3,3);
    layerOneWeights << 1,2,3,4,5,6,7,8,9;
    MatrixXd layerTwoWeights(3,3);
    layerTwoWeights << 9,8,7,6,5,4,3,2,1;
    MatrixXd layerThreeWeights(3,3);
    layerThreeWeights << -1,2,-3,4,-5,6,-7,8,-9;
    MatrixXd layerFourWeights(1,3);
    layerFourWeights << 4,-5,6;

    testWeights.push_back(layerOneWeights);
    testWeights.push_back(layerTwoWeights);
    testWeights.push_back(layerThreeWeights);
    testWeights.push_back(layerFourWeights);
    
    //intializes network's test weights
    for(int i = 0;i < net->network.size()-1;i++){
        MatrixXd& currentLayerWeights = net->network[i+1].weights;

        if(currentLayerWeights.cols() != testWeights[i].cols() || currentLayerWeights.rows() != testWeights[i].rows()){
            cout<< "Layer " << i+1 << " test weight matrix proportions do not match initial network's layer" << i+1 << "weight matrix proportions"<<endl;
            cout<< "Automatic proportions: COLUMNS - " << currentLayerWeights.cols() << " ROWS - " << currentLayerWeights.rows()<<endl;
            cout<< "Given proportions: COLUMNS - " << testWeights[i].cols() << " ROWS - " << testWeights[i].rows()<<endl;
            ASSERT_TRUE(false);
        }
        net->network[i+1].weights = testWeights[i];
    }

    //feed inputs through network
    double actualOutput = net->feedForward(inputs);

    //check outputs of network
    int error = 100000;
    vector<MatrixXd> outputs;

    
    MatrixXd layerOneOutputs(3,1);
    layerOneOutputs << 0.377541, 0.377541, 0.377541;
    MatrixXd layerTwoOutputs(3,1);
    layerTwoOutputs << 0.999884, 0.99654, 0.905957;
    MatrixXd layerThreeOutputs(3,1);
    layerThreeOutputs << 0.15127, 0.988486, 0.000760722;
    MatrixXd layerFourOutputs(1,1);
    layerFourOutputs << 0.0129608;

    outputs.push_back(inputs);
    outputs.push_back(layerOneOutputs);
    outputs.push_back(layerTwoOutputs);
    outputs.push_back(layerThreeOutputs);
    outputs.push_back(layerFourOutputs);


    for(int i = 0;i < net->network.size();i++){
        double approxDifference = round((net->network[i].outputs-outputs[i]).sum()*error);

        ASSERT_EQ(0, approxDifference);
    }

    //check the outputs of each layer
    double correctOutput = round(0.0129608*error);
    actualOutput = round(actualOutput*error);
    ASSERT_EQ(correctOutput/error,actualOutput/error);
}

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}