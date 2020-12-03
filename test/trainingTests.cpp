#include <gtest/gtest.h>
#include <eigen/Eigen/Dense>
#include "../src/Layer.cpp"
#include "../src/Network.cpp"
#include "basicFunctionsTestInitialization.hpp"
#include <limits>
#include <eigen/Eigen/Dense>
#include "../include/Trainer.hpp"

TEST (trials, one){

    srand(100);
    Network *net = new Network();

    MatrixXd inputs1(3,1);
    inputs1 << 0, 0, 0;
    MatrixXd inputs2(3,1);
    inputs2 << 0, 0, 1;
    MatrixXd inputs3(3,1);
    inputs3 << 0, 1, 0;
    MatrixXd inputs4(3,1);
    inputs4 << 0, 1, 1;
    MatrixXd inputs5(3,1);
    inputs5 << 1, 0, 0;
    MatrixXd inputs6(3,1);
    inputs6 << 1, 0, 1;
    MatrixXd inputs7(3,1);
    inputs7 << 1, 1, 0;
    MatrixXd inputs8(3,1);
    inputs8 << 1, 1, 1;

    vector<MatrixXd> inputs;
    inputs.push_back(inputs1);
    inputs.push_back(inputs2);
    inputs.push_back(inputs3);
    inputs.push_back(inputs4);
    inputs.push_back(inputs5);
    inputs.push_back(inputs6);
    inputs.push_back(inputs7);
    inputs.push_back(inputs8);

    vector<double> desiredOutputs {1,1,-1,-1,-1,-1,1,1};

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<net->feedForward(inputs[i])<<" Desired: "<<desiredOutputs[i]<<endl;
    }

    Trainer::train(net,inputs,desiredOutputs,50000);

    cout<<endl;

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<net->feedForward(inputs[i])<<" Desired: "<<desiredOutputs[i]<<endl;
    }
}

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}