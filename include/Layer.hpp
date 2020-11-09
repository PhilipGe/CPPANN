#pragma once
#include <eigen/Eigen/Dense>

using namespace Eigen;
using namespace std;

class Layer{
    private:
        
        //for initialization
        int thisLayerNodes;
        int previousLayerNodes;

        //for feed forward
        
        double sigmoid(double x);
        MatrixXd sigmoid(MatrixXd mat);
    
    public:
        //for feed forward
        MatrixXd outputs;
        
        MatrixXd productSums;
        MatrixXd weights;
        Layer(int numberOfNodesInThisLayer, int numberOfNodesInPreviousLaver = 0);
        Layer();
        MatrixXd feedForward(MatrixXd outputs);

        static int numOfLayers;
};