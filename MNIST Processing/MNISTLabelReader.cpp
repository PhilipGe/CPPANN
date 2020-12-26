#include <iostream>
#include <fstream>
#include <bitset>


//first 8 bytes:
//first 4 bytes are the magic number (2049)
//next  4 bytes are the number of labels (60000)
//the next byte is the next image number

using namespace std;
using bitform = bitset<8>;

int main(){

    string fileAddress = "Images/train-labels-idx1-ubyte";
    fstream newFile(fileAddress, ios::in|ios::binary);

    int numberOfLabels = 40;
    int bytesRead = 8+numberOfLabels;

    char buffer[bytesRead];
    newFile.seekg(0,ios::beg);
    
    newFile.read(buffer,bytesRead);

    for(int i = 8;i < numberOfLabels;i++){
        cout<<i<<" "<<int(buffer[i])<<endl;
    }

    return 0;
}