#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
#include "../include/Properties.hpp"
#include "../include/Layer.hpp"


using namespace Eigen;
using namespace std;

class Network{
    private:

    public:
        //NETWORK PROPERTIES
        int numberOfHiddenLayers = Properties::numberOfHiddenLayers;
        int nodesPerLayer = Properties::nodesPerLayer;
        int numberOfInputs = Properties::numberOfInputs;
        int learningSpeed = Properties::learningSpeed;

        //INITIALIZATION 
        vector<Layer> network;
        Network();
        Network(bool test);

        //FEED FORWARD
        double feedForward(MatrixXd inputs, bool print = false);

        //BACKPROPOGATION
        MatrixXd matrixOfOnes = MatrixXd::Constant(nodesPerLayer,1,1);
        
        void calculateErrors(double desiredOutput);
        void calculateNonIndividualDerivatives();
        void updateWeights();

        //DEBUGGING
        void printNetworkWeights();
};

