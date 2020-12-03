#include <iostream>
#include "../include/Layer.hpp"
#include "../include/Network.hpp"
#include "../include/Properties.hpp"



/*
0   1   2   3...m
------------------
0
0   0   0   0...
0   0   0   0...0
0   0   0   0...
0

*/

/*
INTIALIZATION
________________________________________________________________________________________________________
*/

//after you costruct the network, look into whether or not having a vector of pointers vs a vector of actual functions is better
Network::Network(){
    
    //initializes the input layer (the only attributes it contains are the inputs)
    network.push_back(*(new Layer(numberOfInputs,0)));

    //initializes first hidden layer: each node has as many weights as there are inputs
    network.push_back(*(new Layer(nodesPerLayer,numberOfInputs)));

    //initializes hidden layers 2-(m-1): each node has as many weights as there are nodes per hidden layer
    for(int i = 1;i < numberOfHiddenLayers;i++){
        network.push_back(*(new Layer(nodesPerLayer,nodesPerLayer)));
    }

    //initializes output layer: single node layer that has as many weights as there are nodes per hidden layer
    network.push_back(*(new Layer(1,nodesPerLayer)));

}


/*
FEED FORWARD
____________
*/

double Network::feedForward(MatrixXd inputs, bool print){

    if(inputs.rows() != network[1].weights.cols()){
        throw invalid_argument("NETWORK.CPP: FEEDFORWARD -- Input matrix size does not match network's input dimensions");
    }

    //feeds forward inputs through first layer which defines output matrix of that layer
    network[0].outputs = inputs;
    if (print) cout<<"Layer 0 outputs: \n"<<network[0].outputs<<endl<<endl;

    //feeds forward outputs of first layer through the network (accesses previous layer's outputs and feeds them into current layer, repeats until the last HIDDEN layer)
    for(int i = 1;i < numberOfHiddenLayers+1;i++){
        network[i].feedForward(network[i-1].outputs);
        if (print) cout<<"Layer "<<i<<" outputs: \n"<<network[i].outputs<<endl<<endl;
    }

    network[network.size()-1].feedForward(network[network.size()-2].outputs, true);

    //returns output value of last layer output matrix (1x1 matrix)
    return network[network.size()-1].outputs(0,0);
}


/*
BACKPROPOGATION
_____________________________________________________________________________

*/
void Network::calculateErrors(MatrixXd testInput, double desiredOutput){

    feedForward(testInput);

    //calculate the OUTPUT LAYER error
    double lastLayerError = network[network.size()-1].outputs(0,0)-desiredOutput;

    //input output node's error into Diagonal Matrix that stores that error
    network[network.size()-1].lastLayerErrors.diagonal() << lastLayerError;

    //Calculates derivatives of the weights in that node based on that error
    network[network.size()-1].derivatives = lastLayerError*network[network.size()-2].outputs.transpose();

    //Updates the weights of the output node
    network[network.size()-1].weights -= network[network.size()-1].derivatives * learningSpeed;

    //initializes the LAST HIDDEN LAYER errors by
    //(1) Calculating the sigmoid derivative of all of the outputs of the last hidden layer
    MatrixXd sigmoidDerivative = (verticalMatrixOfOnes - network[network.size()-2].outputs).cwiseProduct(network[network.size()-2].outputs);

    //(2) Multiplying the weights stored in the last layer by the diagonal matrix of the errors of the last layer
    MatrixXd summation = network[network.size()-1].lastLayerErrors*network[network.size()-1].weights;

    //(3) Multiplying the sigmoid derivative by the summation and inputing it into the diagonal of the error matrix of the last hidden layer 
    network[network.size()-2].hiddenLayerErrors.diagonal()<<sigmoidDerivative.cwiseProduct(summation.transpose());
    
    //(4) Now that the errors of this layer are calculated, we can calculate the derivative by multiplying the diagonal of the error 
    //    by the outputs of the previous layer
    network[network.size()-2].derivatives = network[network.size()-2].hiddenLayerErrors.diagonal()*network[network.size()-3].outputs.transpose();

    //(5) Now that we have the derivative matrix, we can multiply the matrix by the scalar learning speed and subtract the dreivative matrix 
    //    from the weight matrix
    network[network.size()-2].weights -= network[network.size()-2].derivatives * learningSpeed;

    //calculate the hidden nodes' errors
    for(int currentLayer = network.size()-3;currentLayer >= 1;currentLayer--){

        //cout<<"1"<<endl;
        //(1) Calculating the sigmoid derivative of all of the outputs of the current layer
        sigmoidDerivative = (verticalMatrixOfOnes - network[currentLayer].outputs).cwiseProduct(network[currentLayer].outputs);

        //cout<<"2"<<endl;
        //(2a) Multiplying the weights stored in the next layer by the diagonal matrix of the errors of the next layer
        MatrixXd temp;
        temp = network[currentLayer+1].hiddenLayerErrors*network[currentLayer+1].weights;
        //(2b) Getting the sums of the rows and putting it into a seperate matrix
        MatrixXd summation(nodesPerLayer,1);
        summation << temp.row(0).sum(),temp.row(1).sum(),temp.row(2).sum();

        //cout<<"3"<<endl;
        //(3) Multiplying the sigmoid derivative by the summation and inputing it into the diagonal of the error matrix of the current layer 
        MatrixXd errors(nodesPerLayer, 1);
        errors << sigmoidDerivative.cwiseProduct(summation);
        network[currentLayer].hiddenLayerErrors.diagonal()<< errors;

        //cout<<"4"<<endl;
        //(4) Now that the errors are calculated, we can multiply the error vector by the transposed output vector to create a matrix
        //    that had the same dimensions as the weight matrix. Each value will be a derivative to its corresponding weight
        network[currentLayer].derivatives = errors*network[currentLayer-1].outputs.transpose();

        //cout<<"5"<<endl;
        //(5) Once we have the derivatives, we can multiply them by the scalar learning rate and subract them from the weights
        network[currentLayer].weights -= network[currentLayer].derivatives * learningSpeed;
    }

}

//This method can efficiently calculate the average derivatives of an entire layer, however, it will not calculate the individual derivatives of that layer
void Network::calculateNonIndividualDerivatives(){
    
}

void Network::updateWeights(){

}

/*
DEBUGGING
_______________________________________________________________________________
*/

//Test initialization with predefined (not random) weights
Network::Network(bool test){

    //initializes first hidden layer: each node has as many weights as there are inputs - each weight is equal to 1
    network.push_back(*(new Layer(nodesPerLayer,numberOfInputs, true, 1)));

    //initializes hidden layers 2-(m-1): each node has as many weights as there are nodes per hidden layer -  each weight is equal to i+1
    for(int i = 1;i < numberOfHiddenLayers;i++){
        network.push_back(*(new Layer(nodesPerLayer,nodesPerLayer, true, i+1)));
    }

    //initializes output layer: single node layer that has as many weights as there are nodes per hidden layer - each layer is equal to the number of that layer
    network.push_back(*(new Layer(1,nodesPerLayer,true,numberOfHiddenLayers+1)));

}

//prints network's weights
void Network::printNetworkWeights(){
    cout<<"Row: Node in this layer"<< endl <<"Column: Weight for node in previous layer"<<endl<<endl;
    cout<<"Layer " << 0 <<" Weights:\n"<<network[0].weights<<endl<<endl;

    for(unsigned int i = 1;i < network.size();i++){
        cout<<"Layer " << i <<" Weights:\n"<<network[i].weights<<endl<<endl;
    }

    cout<<"Network size: \n"<<network.size()<<" layers"<<endl<<endl;
    cout<<"Number of hidden layers: \n"<<numberOfHiddenLayers<<" layers"<<endl<<endl;
}

//print network's outputs
void Network::printNetworkOutputs(){
    for(unsigned int i = 0;i < network.size();i++){
        cout<<"Layer " << i << " Outputs:\n"<<network[i].outputs<<endl<<endl;
    }

    cout<<"Network size: \n"<<network.size()<<" layers"<<endl<<endl;
    cout<<"Number of hidden layers: \n"<<numberOfHiddenLayers<<" layers"<<endl<<endl;
}

//print network's errors
void Network::printNetworkErrors(){
    for(unsigned int i = 0;i < network.size()-1;i++){
        cout<<"Layer " << i << " Errors:\n"<<network[i].hiddenLayerErrors.diagonal()<<endl<<endl;
    }

    cout<<"Layer "<<network.size()-1<<" Errors:\n"<<network[network.size()-1].lastLayerErrors.diagonal()<<endl<<endl;

    cout<<"Network size: \n"<<network.size()<<" layers"<<endl<<endl;
    cout<<"Number of hidden layers: \n"<<numberOfHiddenLayers<<" layers"<<endl<<endl;
}