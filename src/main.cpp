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
    
    
    MatrixXd inputs = MatrixXd::Constant(5,1,1);

    Layer ** layers;
    layers = new Layer*[5];

    MatrixXd ** integers;
    integers = new MatrixXd*[5];

    integers[0] = new MatrixXd;
    integers[1] = new MatrixXd;
    integers[2] = new MatrixXd;
    integers[3] = new MatrixXd;
    integers[4] = new MatrixXd;

    cout<<"Integer length: "<<sizeof(integers)<<"/"<<sizeof(integers[1])<<endl;

    cout<<"Layers: "<<sizeof(layers)/sizeof(layers[0])<<endl;
    cout<<"Num of Layers: "<<Layer::numOfLayers<<endl;

    Network *network = new Network(3,3,5);
    network->network[0]->feedForward(inputs);
    cout<<"First Layer Outputs: \n"<<network->network[0]->outputs<<endl<<endl;

    network->network[1]->feedForward(network->network[0]->outputs);
    cout<<"Second Layer Outputs: \n"<<network->network[1]->outputs<<endl<<endl;

    network->network[2]->feedForward(network->network[1]->outputs);
    cout<<"Third Layer Outputs: \n"<<network->network[2]->outputs<<endl<<endl;

    MatrixXd output = network->feedForward(inputs);

    cout<<endl<<output<<endl;

    return 0; 
}