#include <iostream>
#include "../include/Layer.hpp"
#include "../include/Network.hpp"
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
    //Layer *lay = new Layer(3,4);
    Network *net = new Network(3,3,5,true); 

    MatrixXd in = MatrixXd::Constant(5,1,1);
    cout<<"Inputs: \n"<<in<<endl<<endl;
    
    net->printNetworkWeights();

    double out = net->feedForward(in, true);

    cout<<out<<endl;

    return 0; 
}