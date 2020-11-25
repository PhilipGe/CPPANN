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

    MatrixXd inputs(4,1);

    inputs << 1,-1,2,5;

    net->feedForward(inputs);
    net->printNetworkWeights();

    srand(1000);
    for(int i = 0;i<10000;i++){
        net->calculateErrors(-0.0324);
    }

    net->printNetworkWeights();
    double output = net->feedForward(inputs);
    cout<<output<<endl;

    return 0;
}