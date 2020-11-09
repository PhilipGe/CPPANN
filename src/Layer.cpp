#include <iostream>
#include "../include/Layer.hpp"
#include <math.h>

int Layer::numOfLayers = 0;

//Creates a layer with a parameterized amount of nodes and parameterized (unless not specified) amount of weights per node
Layer::Layer(int thisLayerNodes, int previousLayerNodes) {
    this->thisLayerNodes = thisLayerNodes;
    this->previousLayerNodes = previousLayerNodes != 0 ? previousLayerNodes : thisLayerNodes;

    weights = MatrixXd::Random(thisLayerNodes, previousLayerNodes);

}

//the number of outputs is equal to the number of nodes in the previous layer
//this function returns the outputs of this layer based on the outputs of the previous layer
    //it does so by taking the product
MatrixXd Layer::feedForward(MatrixXd ioutputs){
    productSums = weights*ioutputs;

    outputs = sigmoid(productSums);
    return outputs;
}

//the sigmoid function compresses the outputs of the product sums into a value between 0 and 1
double Layer::sigmoid(double x){
    return 1/(1 + exp(-x));
}

//any matrix that is being sigmoided is a vertical vector of the product sums (ps) whose length is the number of nodes in the current layer
//that means we can loop through it without checking its dimensions
MatrixXd Layer::sigmoid(MatrixXd ps){
    MatrixXd out(thisLayerNodes,1);
    for(int i = 0;i < thisLayerNodes;i++){
        out(i,0) = sigmoid(ps(i,0));
    }

    return out;
}

//default constructor constructs a layers that acts as a palceholder in the initial array of layers (network[]) in Network.cpp 
//it is later replaced by the other constructor Layer(int, int, int)
Layer::Layer(){
    Layer::numOfLayers += 1;
}