#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
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
        double learningSpeed = Properties::learningSpeed;

        //INITIALIZATION 
        vector<Layer> network;
        Network();
        Network(bool test);

        //FEED FORWARD
        double feedForward(MatrixXd inputs, bool print = false);

        //BACKPROPOGATION
        MatrixXd verticalMatrixOfOnes = MatrixXd::Constant(nodesPerLayer,1,1);
        MatrixXd horizontalMatrixOfOnes = MatrixXd::Constant(1,nodesPerLayer,1);
        
        void calculateErrors(MatrixXd testInput, double desiredOutput);
        void calculateNonIndividualDerivatives();
        void updateWeights();

        //DEBUGGING
        void printNetworkWeights();
        void printNetworkOutputs();
        void printNetworkErrors();
};

