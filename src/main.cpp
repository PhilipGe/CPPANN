#include <iostream>
#include "../include/Layer.hpp"
#include "../include/Network.hpp"
#include <math.h>

using namespace std;

/*
0   1   2   3...m
------------------
0
0   0   0   0...
0   0   0   0...0
0   0   0   0...
0

*/

int main(){
    Network *net = new Network();

    MatrixXd inputs = MatrixXd::Constant(5,1,1);

    net->feedForward(inputs,true);

    //net->calculateErrors(1.0);

    MatrixXd mat = MatrixXd::Constant(3,3,1);
    MatrixXd mat2 = MatrixXd::Random(3,3);

    cout<<mat2<<endl<<endl<<mat2-mat<<endl<<endl;


    // MatrixXd weights(2,3);

    // weights << 1,2,3,4,5,6;

    // DiagonalMatrix<double,2> diag;

    // diag.diagonal() << 1,2;

    // cout<<diag.diagonal().transpose()<<endl;
    // cout<<weights<<endl;
    // cout<<diag.diagonal().transpose()*weights<<endl;

    return 0;
}