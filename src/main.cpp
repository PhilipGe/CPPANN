#include <iostream>
#include "../include/Layer.h"

using namespace std;

int main(){

    Layer *layer = new Layer(3,3);

    cout<<layer->weights<<endl;
    return 0; 
}