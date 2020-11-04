#pragma once
#include <eigen/Eigen/Dense>

using namespace Eigen;
using namespace std;

class Layer{
    private:
        MatrixXd weights;
        int thisLayerNodes;
        int previousLayerNodes;
    
    public:
        Layer(int numberOfNodesInThisLayer, int numberOfNodesInPreviousLaver = 0);
        MatrixXd feedForward(MatrixXd outputs);
};