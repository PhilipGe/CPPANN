#include <iostream>
#include "../include/Network.hpp"
#include "../include/Layer.hpp"

/*
0   1   2   3...m
------------------
0
0   0   0   0...
0   0   0   0...0
0   0   0   0...
0

*/


Network::Network(int numberOfHiddenLayers, int nodesPerLayer, int numberOfInputs){
    network = new Layer*[numberOfHiddenLayers+1]; 
    cout<<sizeof(network)/sizeof(network[0])<<endl;

    //initializes hidden layers 1-(m-1)
    for(int i = 1;i < numberOfHiddenLayers;i++){
        network[i] = new Layer(nodesPerLayer,nodesPerLayer);
    }

    //initializes first hidden layer
    network[0] = new Layer(nodesPerLayer,numberOfInputs);

    //initializes output layer
    network[numberOfHiddenLayers] = new Layer(1,nodesPerLayer);

}

MatrixXd Network::feedForward(MatrixXd inputs){

    //feeds forward inputs through first layer which defines output matrix of that layer
    network[0]->feedForward(inputs);

    //feeds forward outputs of first layer through the network (accesses previous layer's outputs and feeds them into current layer, repeats until the end)
    for(int i = 1;i < numberOfHiddenlayers+1;i++){
        network[i]->feedForward(network[i-1]->outputs);
    }

    //returns output value of last layer output matrix (1x1 matrix)
    return network[numberOfHiddenlayers+3]->outputs;
}