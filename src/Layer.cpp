#include <iostream>
#include "../include/Layer.h"


//Creates a layer with a thisLayerNodes amount of nodes and previousLayerNodes amount of weights per node
Layer::Layer(int thisLayerNodes, int previousLayerNodes) {
    this->thisLayerNodes = thisLayerNodes;
    this->previousLayerNodes = previousLayerNodes != 0 ? previousLayerNodes : thisLayerNodes;

    weights = MatrixXd::Random(thisLayerNodes, previousLayerNodes);

}

MatrixXd Layer::feedForward(MatrixXd outputs){
    
    return MatrixXd(1,thisLayerNodes);
}