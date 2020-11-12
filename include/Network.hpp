#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
#include "../include/Layer.hpp"

using namespace Eigen;
using namespace std;

class Network{
    private:
        
        int learningSpeed;


    public:
        //NETWORK PROPERTIES
        const int numberOfHiddenLayers = 3;
        const int nodesPerLayer = 3;
        const int numberOfInputs = 5;
        const int learningSpeed = 0.1;

        //INITIALIZATION 
        vector<Layer> network;
        Network();
        Network(bool test);

        //FEED FORWARD
        double feedForward(MatrixXd inputs, bool print = false);

        //BACKPROPOGATION
        void calculateErrors(double desiredOutput);
        void calculateNonIndividualDerivatives();
        void updateWeights();

        //DEBUGGING
        void printNetworkWeights();
};

