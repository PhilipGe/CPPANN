#include <iostream>
#include <vector>
#include <eigen/Eigen/Dense>
#include "Network.hpp"
#include "Properties.hpp"

class Trainer{

    public:
    static void train(Network * net, vector<MatrixXd> testInputs, vector<double> desiredOutputs, int numberOfIterations, bool track = false);   

};