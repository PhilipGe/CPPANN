#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
#include "../include/Layer.hpp"

using namespace Eigen;
using namespace std;

class Network{
    private:
        int numberOfHiddenlayers;
        int nodesPerLayer;
        int numberOfInputs;
        int learningSpeed;
    public:
        //INITIALIZATION
        vector<Layer> network;
        Network(int numberOfHiddenLayers, int nodesPerLayer, int numberOfInputs);
        Network(int numberOfHiddenLayers, int nodesPerLayer, int numberOfInputs, bool test);

        //FEED FORWARD
        double feedForward(MatrixXd inputs, bool print = false);

        //BACKPROPOGATION
        void calculateErrors(double desiredOutput);
        void calculateDerivatives();
        void updateWeights();

        //DEBUGGING
        void printNetworkWeights();
};

