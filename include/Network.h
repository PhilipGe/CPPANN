#include <iostream>
#include <eigen/Eigen/Dense>

using namespace Eigen;
using namespace std;

class Network{
    private:
        int numberOfHiddenlayers;
        int nodesPerLayer;
        int numberOfInputs;
    public:
        Network(int numberOfHiddenLayers, int nodesPerLayer, int numberOfInputs);
        double feedForward();
};
