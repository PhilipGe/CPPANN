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

/*
INTIALIZATION
________________________________________________________________________________________________________
*/

//after you costruct the network, look into whether or not having a vector of pointers vs a vector of actual functions is better
Network::Network(int numberOfHiddenLayers, int nodesPerLayer, int numberOfInputs){

    this->numberOfHiddenlayers = numberOfHiddenLayers;
    this-> nodesPerLayer = nodesPerLayer;
    this-> numberOfInputs = numberOfInputs;
    
    //initializes first hidden layer: each node has as many weights as there are inputs
    network.push_back(*(new Layer(nodesPerLayer,numberOfInputs)));

    //initializes hidden layers 2-(m-1): each node has as many weights as there are nodes per hidden layer
    for(int i = 1;i < numberOfHiddenLayers;i++){
        network.push_back(*(new Layer(nodesPerLayer,nodesPerLayer)));
    }

    //initializes output layer: single node layer that has as many weights as there are nodes per hidden layer
    network.push_back(*(new Layer(1,nodesPerLayer)));

}


/*
FEED FORWARD
____________
*/

double Network::feedForward(MatrixXd inputs, bool print){

    if(inputs.rows() != network[0].weights.cols()){
        throw invalid_argument("Input matrix size does not match network's input dimensions");
    }

    //feeds forward inputs through first layer which defines output matrix of that layer
    network[0].feedForward(inputs);
    if (print) cout<<"Layer 0 outputs: \n"<<network[0].outputs<<endl<<endl;

    //feeds forward outputs of first layer through the network (accesses previous layer's outputs and feeds them into current layer, repeats until the end)
    for(int i = 1;i < numberOfHiddenlayers+1;i++){
        network[i].feedForward(network[i-1].outputs);
        if (print) cout<<"Layer "<<i<<" outputs: \n"<<network[i].outputs<<endl<<endl;
    }

    //returns output value of last layer output matrix (1x1 matrix)
    return network[numberOfHiddenlayers].outputs(0,0);
}


/*
BACKPROPOGATION
_____________________________________________________________________________

*/





/*
DEBUGGING
_______________________________________________________________________________
*/

//Test initialization with predefined (not random) weights
Network::Network(int numberOfHiddenLayers, int nodesPerLayer, int numberOfInputs, bool test){

    this->numberOfHiddenlayers = numberOfHiddenLayers;
    this-> nodesPerLayer = nodesPerLayer;
    this-> numberOfInputs = numberOfInputs;
    
    //initializes first hidden layer: each node has as many weights as there are inputs - each weight is equal to 1
    network.push_back(*(new Layer(nodesPerLayer,numberOfInputs, true, 1)));

    //initializes hidden layers 2-(m-1): each node has as many weights as there are nodes per hidden layer -  each weight is equal to i+1
    for(int i = 1;i < numberOfHiddenLayers;i++){
        network.push_back(*(new Layer(nodesPerLayer,nodesPerLayer, true, i+1)));
    }

    //initializes output layer: single node layer that has as many weights as there are nodes per hidden layer - each layer is equal to the number of that layer
    network.push_back(*(new Layer(1,nodesPerLayer,true,numberOfHiddenLayers+1)));

}

//prints network's weights
void Network::printNetworkWeights(){
    cout<<"Row: Node in this layer"<< endl <<"Column: Weight for node in previous layer"<<endl<<endl;
    cout<<"Layer " << 0 <<" Weights:\n"<<network[0].weights<<endl<<endl;

    for(unsigned int i = 1;i < network.size();i++){
        cout<<"Layer " << i <<" Weights:\n"<<network[i].weights<<endl<<endl;
    }

    cout<<"Network size: \n"<<network.size()<<" layers"<<endl<<endl;
    cout<<"Number of hidden layers: \n"<<numberOfHiddenlayers<<" layers"<<endl<<endl;
}