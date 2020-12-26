#include <iostream>
#include <fstream>
#include <istream>
#include <string>

#include <fstream>
#include <iterator>
#include <vector>

#include <bitset>
#include <limits>


using namespace std;

using byte = unsigned char;
const size_t bits_per_byte = numeric_limits<byte>::digits;
using bits_in_byte = bitset<bits_per_byte>;

std::string read_bits( const char* path_to_file )
{
    std::string bitstring ;
    std::ifstream file( path_to_file, std::ios::binary ) ; // open in binary mode

    char c ;
    int count ;
    while( file.get(c) && count <= 5){ // read byte by byte
        bitstring += bits_in_byte( byte(c) ).to_string() ; // append as string of '0' '1'
        count ++;
    }

    return bitstring ;
}

std::string read_bytes( const char* path_to_file )
{
    std::string bitstring ;
    std::ifstream file( path_to_file ) ; // open in binary mode

    char c ;
    int count ;
    while( file.get(c) && count <= 5){ // read byte by byte
        bitstring += byte(c) ; // append as string of '0' '1'
        count ++;
    }

    return bitstring ;
}

string getFunction(char* c, string filePath){

    std::ifstream file;
    file.open(filePath.c_str(), ios::binary);

    file.get(c, 50);
    
    return string(c);    
}

int main(){

    ifstream myfile;
    const string inputString = "../../../MNIST Images/Training Images/train-images-idx3-ubyte";
    // myfile.open(inputString, ios::binary);

    // int count = 0;
    // char line[50];

    // myfile.seekg(0, ios::end);
    // size_t length = myfile.tellg();  

    // myfile.read(line, 50);

    // while(count != length){
    //     cout<<bits_in_byte(byte(line[count]))<<" ";
    //     count++;
    //     if(count >= 5){
    //         break;
    //     }
    // }
    // cout<<endl;

    string bitString = read_bits(inputString.c_str());
    string bitString2 = read_bytes(inputString.c_str());

    char cs[50];
    string bitString3 = getFunction(cs, inputString);

    cout<<bitString<<endl;
    cout<<bitString3<<endl;
    
    return 0;
}

//http://www.cplusplus.com/forum/general/119145/