#include <iostream>
#include "../include/Layer.hpp"
#include "../include/Network.hpp"
#include "../include/Trainer.hpp"
#include <chrono> 
#include <math.h>

using namespace std;

/*
0   1   2   3...m
------------------
0
0   0   0   0...
0   0   0   0...0
0   0   0   0...
0

*/

int main(){
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
    
    

    vector<MatrixXd> desiredOutputs;

    for(int i = 0; i<16;i++){
        desiredOutputs.push_back(MatrixXd::Constant(16,1,0));
        desiredOutputs[i](3,0) = 1;
    }

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<net->feedForward(inputs[i]).array().round().abs().transpose()<<" Desired: "<<desiredOutputs[i].transpose()<<endl;
    }

    Trainer::train(net,inputs,desiredOutputs,50000,true);

    // desiredOutputs.push_back();
    // // inputs.push_back(inputs8);

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<net->feedForward(inputs[i]).array().round().abs().transpose()<<" Desired: "<<desiredOutputs[i].transpose()<<endl;
    }
    
    return 0;
}