#include <iostream>
#include "../include/Layer.hpp"
#include <math.h>


//Creates a layer with a parameterized amount of nodes and parameterized (unless not specified) amount of weights per node
Layer::Layer(int thisLayerNodes, int previousLayerNodes, bool test, int weightConstant) {
    this->thisLayerNodes = thisLayerNodes;
    this->previousLayerNodes = previousLayerNodes;

    //the rows represent this layer's nodes. The columns represent the previous layer's nodes    
    if(!test){
        weights = MatrixXd::Random(thisLayerNodes, previousLayerNodes);
        derivatives = MatrixXd::Constant(thisLayerNodes, previousLayerNodes,0);
        outputs = MatrixXd::Constant(thisLayerNodes, 1,0);
    }else{
        weights = MatrixXd::Constant(thisLayerNodes, previousLayerNodes, weightConstant);
    }
}

//the number of outputs is equal to the number of nodes in the previous layer
//this function returns the outputs of this layer based on the outputs of the previous layer
    //it does so by taking the product
MatrixXd Layer::feedForward(MatrixXd previous_outputs, bool lastLayer){
    productSums = weights*previous_outputs;

    if(!lastLayer){
        outputs = sigmoid(productSums);
    }else{
        outputs = productSums;
    }

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

//BACKPROPOGATION
