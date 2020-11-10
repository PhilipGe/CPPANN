#pragma once
#include <eigen/Eigen/Dense>

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
        double sigmoidDerivative(int x);
        
    
    public:
        //FEED FORWARD
        Layer(int thisLayerNodes, int previousLayerNodes, bool test = false, int weightConstant = 0);

        //FEED FORWARD
        MatrixXd feedForward(MatrixXd);
        MatrixXd outputs;
        MatrixXd productSums;
        MatrixXd weights;

        //BACKPROPOGATION
        MatrixXd errors;
        MatrixXd derivatives;
        
};