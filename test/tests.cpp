#include <gtest/gtest.h>
#include <eigen/Eigen/Dense>
#include "../src/Layer.cpp"
#include "../src/Network.cpp"
#include "TestInitialization.h"

TEST (network, feedForward){

    //initializes a test network through static functions in TestInitialization.h
    TestInitialization::createTestNetwork();
    Network *net = TestInitialization::net;

    MatrixXd inputs(3,1);
    inputs << 0.5,-0.5,0;

    //feed inputs through network
    double actualOutput = net->feedForward(inputs);

    //check outputs of network
    int error = 100000;
    vector<MatrixXd> outputs;

    //correct outputs
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

    //checks outputs of each node to correct outputs
    for(int i = 0;i < net->network.size();i++){
        double approxDifference = round((net->network[i].outputs-outputs[i]).sum()*error);

        ASSERT_EQ(0, approxDifference);
    }

    //check the final output
    double correctOutput = round(0.0129608*error);
    actualOutput = round(actualOutput*error);
    ASSERT_EQ(correctOutput/error,actualOutput/error);
}

TEST (network, backpropogation){
    
}

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}