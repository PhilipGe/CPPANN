#pragma once

class TestInitialization{

    public:
    
    static Network *net;

    static Network * createTestNetwork();
};

Network * TestInitialization::net = new Network();

Network * TestInitialization::createTestNetwork(){
    int numberOfHiddenLayers = 3;
    int nodesPerLayer = 3;
    int numberOfInputs = 9;
    double learningSpeed = 1;

    vector<MatrixXd> testWeights;

    MatrixXd layerOneWeights(3,3);
    layerOneWeights << 1,2,3,4,5,6,7,8,9;
    MatrixXd layerTwoWeights(3,3);
    layerTwoWeights << 9,8,7,6,5,4,3,2,1;
    MatrixXd layerThreeWeights(3,3);
    layerThreeWeights << -1,2,-3,4,-5,6,-7,8,-9;
    MatrixXd layerFourWeights(1,3);
    layerFourWeights << 4,-5,6;

    testWeights.push_back(layerOneWeights);
    testWeights.push_back(layerTwoWeights);
    testWeights.push_back(layerThreeWeights);
    testWeights.push_back(layerFourWeights);
    
    //intializes network's test weights
    for(int i = 0;i < net->network.size()-1;i++){
        MatrixXd& currentLayerWeights = net->network[i+1].weights;

        if(currentLayerWeights.cols() != testWeights[i].cols() || currentLayerWeights.rows() != testWeights[i].rows()){
            cout<< "Layer " << i+1 << " test weight matrix proportions do not match initial network's layer " << i+1 << " weight matrix proportions"<<endl;
            cout<< "Automatic proportions: COLUMNS - " << currentLayerWeights.cols() << " ROWS - " << currentLayerWeights.rows()<<endl;
            cout<< "    Given proportions: COLUMNS - " << testWeights[i].cols() << " ROWS - " << testWeights[i].rows()<<endl;
            throw invalid_argument("Invalid Argument");
        }
        net->network[i+1].weights = testWeights[i];
    }

    return net;
}