#include <iostream>
#include "../include/Layer.hpp"
#include "../include/Network.hpp"


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
Network::Network(){
    
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
    for(int i = 1;i < numberOfHiddenLayers+1;i++){
        network[i].feedForward(network[i-1].outputs);
        if (print) cout<<"Layer "<<i<<" outputs: \n"<<network[i].outputs<<endl<<endl;
    }

    //returns output value of last layer output matrix (1x1 matrix)
    return network[numberOfHiddenLayers].outputs(0,0);
}


/*
BACKPROPOGATION
_____________________________________________________________________________

*/
void Network::calculateErrors(double desiredOutput){

    //calculate the last node's error
    double lastLayerError = network[network.size()-1].outputs(0,0)-desiredOutput;
    cout<<lastLayerError<<endl;

    network[network.size()-1].lastLayerErrors.diagonal() << lastLayerError;
    cout<<network[network.size()-1].lastLayerErrors.diagonal()<<endl;

    //initializes the last hidden layer errors
    cout<<network[network.size()-2].hiddenLayerErrors.diagonal()<<endl;

    //double sigmoidDerivative = output*(1-output);
    network[network.size()-2].hiddenLayerErrors.diagonal() <<  network[network.size()-1].weights.transpose()*network[network.size()-1].lastLayerErrors.diagonal();
    cout<<network[network.size()-2].hiddenLayerErrors.diagonal()<<endl;

    /*
        0
        0
        0
        0

        1
        
        Formula for calculating error of layer l:
        network[l+1].weights.transpose()*network[l+1].lastLayerErrors.diagonal();
    */

    //calculate the hidden nodes' errors
    for(int currentLayer = network.size()-2;currentLayer >= 1;currentLayer--){

        //MAKE SURE THIS WORKS - NOT TESTED
        network[currentLayer-1].hiddenLayerErrors.diagonal() << network[currentLayer].hiddenLayerErrors.diagonal()*network[currentLayer].weights;

    }
}

//This method can efficiently calculate the average derivatives of an entire layer, however, it will not calculate the individual derivatives of that layer
void Network::calculateNonIndividualDerivatives(){
    
}

void Network::updateWeights(){

}

/*
DEBUGGING
_______________________________________________________________________________
*/

//Test initialization with predefined (not random) weights
Network::Network(bool test){

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
    cout<<"Number of hidden layers: \n"<<numberOfHiddenLayers<<" layers"<<endl<<endl;
}