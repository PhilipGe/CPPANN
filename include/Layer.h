#pragma once
#include <eigen/Eigen/Dense>

using namespace Eigen;
using namespace std;

class Layer{
    private:
        
        int thisLayerNodes;
        int previousLayerNodes;
    
    public:
        MatrixXd weights;
        Layer(int numberOfNodesInThisLayer, int numberOfNodesInPreviousLaver = 0);
        MatrixXd feedForward(MatrixXd outputs);
};