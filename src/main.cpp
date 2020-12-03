#include <iostream>
#include "../include/Layer.hpp"
#include "../include/Network.hpp"
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
    Network *net = new Network();

    MatrixXd inputs(3,1);

    inputs << -0.5,0.5,1;

    net->printNetworkWeights();

    srand(1000);
    for(int i = 0;i<10000;i++){
        net->calculateErrors(inputs, 5);
    }

    net->printNetworkWeights();
    double output = net->feedForward(inputs);
    cout<<output<<endl;

    return 0;
}