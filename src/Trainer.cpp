#include "../include/Trainer.hpp"
#include <string>


void Trainer::train(Network * net, vector<MatrixXd> testInputs, vector<MatrixXd> desiredOutputs, int numberOfIterations, bool track){
        
    //Will need to optimize later by putting together this function and another one
    vector<MatrixXd> cumulativeDerivatives;
    vector<MatrixXd> currentDerivatives;
    double numberOfTestInputs = testInputs.size();

    if(testInputs.size() != desiredOutputs.size()){
        throw invalid_argument("Test input vector and desired output vector sizes do not match");
    }

    for(int i = 0;i < desiredOutputs.size()-1;i++){
        if((desiredOutputs[i].rows() != net->network[net->network.size()-1].outputs.rows()) && (desiredOutputs[i].cols() != net->network[net->network.size()-1].outputs.cols())){
            cout<<"Network Output Rows and Cols (as defined in Properties): "<<net->network[net->network.size()-1].outputs.cols()<<" "<<net->network[net->network.size()-1].outputs.rows()<<endl;
            cout<<"                Given Desired Output Size Rows and Cols:"<<desiredOutputs[i].cols()<<" "<<desiredOutputs[i].rows()<<endl;
            throw invalid_argument("The 'desiredOutput' matrix " + to_string(i) + " does not match the expected number of outputs in the network");
        }
    }

    for(int iterationNumber = 0;iterationNumber < numberOfIterations;iterationNumber++){        
        for(int currentInput = 0;currentInput < numberOfTestInputs;currentInput++){
            currentDerivatives = net->backpropogate(testInputs[currentInput],desiredOutputs[currentInput]);

            if(currentInput == 0){
                for(int i = 0;i < currentDerivatives.size();i++){         //i = 0 because the first layer's values for derivatives are not part of currentDerivative
                    cumulativeDerivatives.push_back(currentDerivatives[i]/numberOfTestInputs);
                }
            }else{
                for(int i = 0;i < currentDerivatives.size();i++){         //i = 0 because the first layer's values for derivatives are not part of currentDerivative
                    cumulativeDerivatives[i] += currentDerivatives[i]/numberOfTestInputs;
                }
            }
        }
        net->updateWeights(cumulativeDerivatives);
        cumulativeDerivatives.clear();

        if(track){
            if(iterationNumber%(numberOfIterations/10)==0){
                cout<<"Trained "<< iterationNumber/(numberOfIterations/100) <<"%"<<endl;
            }
        }
    }
}