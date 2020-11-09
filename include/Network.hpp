#include <iostream>
#include <eigen/Eigen/Dense>
#include "../include/Layer.hpp"

using namespace Eigen;
using namespace std;

class Network{
    private:
        int numberOfHiddenlayers;
        int nodesPerLayer;
        int numberOfInputs;
    public:
        Layer ** network;
        Network(int numberOfHiddenLayers, int nodesPerLayer, int numberOfInputs);
        MatrixXd feedForward(MatrixXd inputs);
};
