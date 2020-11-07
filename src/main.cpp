#include <iostream>
#include "../include/Layer.h"
#include <math.h>

using namespace std;

int main(){

    Layer *layer = new Layer(3,3);

    layer->weights = MatrixXd::Random(3,4);

    cout<<endl<<"weights:"<<endl<<layer->weights<<endl<<endl;
    /*
        [ 1 1 1 1
          1 1 1 1
          1 1 1 1 ]
    */

    
    MatrixXd outputs(4,1);
    outputs << 2,1,1,3;
    cout<<"outputs:"<<endl<<outputs<<endl<<endl;
    /*
        [ 2
          1
          1 
          1 ]
    */

    MatrixXd out = layer->feedForward(outputs);

    cout<<"Product Sums: \n"<<layer->productSums<<endl;
    cout<<"\nOutputs: \n"<<out<<endl;

    return 0; 
}