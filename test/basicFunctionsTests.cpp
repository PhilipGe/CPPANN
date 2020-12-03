#include <gtest/gtest.h>
#include <eigen/Eigen/Dense>
#include "../src/Layer.cpp"
#include "../src/Network.cpp"
#include "basicFunctionsTestInitialization.hpp"
#include <limits>
#include <eigen/Eigen/Dense>
#include "../include/Trainer.hpp"

typedef std::numeric_limits< double > dbl;

TEST (network, checkFeedForwardOutputs){

    /*PROPERTIES:
    public:
        static const int numberOfHiddenLayers = 3;
        static const int nodesPerLayer = 3;
        static const int numberOfInputs = 3;
        static constexpr double learningSpeed = 0.01;
    */

    //initializes a test network through static functions in TestInitialization.h
    TestInitialization::createTestNetwork();
    Network *net = TestInitialization::net;

    MatrixXd inputs(3,1);
    inputs << 0.5,-0.5,0;

    //feed inputs through network
    double actualOutput = net->feedForward(inputs);

    //check outputs of network
    double error = 0.00000000000000000001;
    vector<MatrixXd> outputs;

    //correct outputs
    MatrixXd layerOneOutputs(3,1);
    layerOneOutputs << 0.37754066879814541, 0.37754066879814541, 0.37754066879814541;
    MatrixXd layerTwoOutputs(3,1);
    layerTwoOutputs << 0.99988390388751403, 0.99654030803906346, 0.90595736581985653;
    MatrixXd layerThreeOutputs(3,1);
    layerThreeOutputs << 0.15126992421132374, 0.98848562983469246, 0.00076072231080266943;
    MatrixXd layerFourOutputs(1,1);
    layerFourOutputs << -4.3327841184633513;

    outputs.push_back(inputs);
    outputs.push_back(layerOneOutputs);
    outputs.push_back(layerTwoOutputs);
    outputs.push_back(layerThreeOutputs);
    outputs.push_back(layerFourOutputs);
    cout.precision(dbl::max_digits10);

    //checks outputs of each node to correct outputs
    for(int i = 0;i < net->network.size();i++){
        EXPECT_NEAR(net->network[i].outputs.sum(), outputs[i].sum(),error);
    }

    //check the final output (rechecks)
    EXPECT_NEAR(net->network[net->network.size()-1].outputs.sum(),outputs[net->network.size()-1].sum(),error);
}

TEST (network, checkErrors){
    Network *net = TestInitialization::net;
    double error = 0.00000001;

    MatrixXd inputs(3,1);
    inputs << 0.5,-0.5,0;

    double desiredOutput = 1;
    net->backpropogate(inputs, desiredOutput);
    
    double approximationError = 0.00000000000000000001;
    vector<MatrixXd> errors;

    //correct outputs
    MatrixXd inputLayerErrors(3,1);
    inputLayerErrors << 0, 0, 0;
    MatrixXd layerOneErrors(3,1);
    layerOneErrors << 0.00343655955155263, 0.018192919611868353, 0.023371981511584004;
    MatrixXd layerTwoError(3,1);
    layerTwoError << 0.0006093096371321019, 0.0051610525438491543, 0.016576182126409357;
    MatrixXd layerThreeError(3,1);
    layerThreeError << -2.7447231563807537, 0.29996362486397055, -0.024322271596046743;
    MatrixXd layerFourError(1,1);
    layerFourError<<-5.3327841184633513;

    errors.push_back(inputLayerErrors);
    errors.push_back(layerOneErrors);
    errors.push_back(layerTwoError);
    errors.push_back(layerThreeError);
    errors.push_back(layerFourError);

    //check errors of network
    EXPECT_NEAR(net->network[net->network.size()-1].lastLayerErrors.diagonal().sum(),errors[net->network.size()-1].sum(),approximationError);

    for(int i = net->network.size()-2;i <= 0;i--){
        EXPECT_NEAR(net->network[i].hiddenLayerErrors.diagonal().sum(), errors[i].sum(),approximationError);
    }    
}


int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}