#include "../include/Trainer.hpp"


void Trainer::train(Network * net, vector<MatrixXd> testInputs, vector<double> desiredOutputs, int numberOfIterations, bool track){
        
    //Will need to optimize later by putting together this function and another one
    vector<MatrixXd> cumulativeDerivatives;
    vector<MatrixXd> currentDerivatives;
    double numberOfTestInputs = testInputs.size();

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