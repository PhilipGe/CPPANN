#pragma once
#include <eigen/Eigen/Dense>
#include "../include/Properties.hpp"

using namespace Eigen;
using namespace std;

class Layer{
    private:
        
        //INITIALIZATION
        int thisLayerNodes;
        int previousLayerNodes;

        //FEED FORWARD
        double sigmoid(double x);
        MatrixXd sigmoid(MatrixXd mat);

        //BACKPROPOGATION
        double sigmoidDerivative(double x);
        
    
    public:
        //FEED FORWARD
        Layer(int thisLayerNodes, int previousLayerNodes, bool test = false, int weightConstant = 0);

        //FEED FORWARD
        MatrixXd feedForward(MatrixXd, bool lastLayer = false);
        MatrixXd outputs;
        MatrixXd productSums;
        MatrixXd weights;
        MatrixXd momentum;

        //BACKPROPOGATION
        DiagonalMatrix<double,Properties::numberOfOutputs> lastLayerErrors;
        DiagonalMatrix<double,Properties::nodesPerLayer> hiddenLayerErrors;
        MatrixXd derivatives;

        MatrixXd getNodesWeightsMagnitudes();

        void constrainWeights(double divideBy);
};