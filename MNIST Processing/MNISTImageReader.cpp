#include <iostream>
#include <fstream>
#include <bitset>
#include <SFML/Graphics.hpp>
#include <string>

using namespace std;
using bitform = bitset<8>;

int main(){


    //first 16 bytes:
    //first 4 bytes are the magic number (2051)
    //next  4 bytes are the number of images (60000)
    //next  4 bytes are the number of rows (28)
    //next  4 bytes are the number of columns (28)
    
    //next 784 bytes are the first image
    //next 784 bytes are the second image
    //... x 59998
    //next 784 bytes are the last image

    //(1) save file into array
    string fileAddress = "Images/train-images-idx3-ubyte";
    fstream newFile(fileAddress, ios::in|ios::binary);

    // newFile.seekg(0,ios::end);
    // int end = newFile.tellg();
    //(2) determine the number of bytes to read
    int NUMBEROFPICTURESREAD = 20;
    int PICTURESIZE = 784;
    int HEADERSIZE = 16;

    const int bytesRead = HEADERSIZE + NUMBEROFPICTURESREAD*PICTURESIZE;
    char buffer[bytesRead];

    newFile.seekg(0,ios::beg);
    newFile.read(buffer, bytesRead);

    //(3) set up array to store single image information
    unsigned char picture[28][28];
    unsigned char currentByte;

    int row = 0;
    int col = -1;

    int startRead;
    int endRead;

    sf::Image img;

    for(int p = 0;p < NUMBEROFPICTURESREAD;p++){

    //(4) read the current image (p) into the array
        //determine the start and end bytes
        startRead = bytesRead-(PICTURESIZE*(NUMBEROFPICTURESREAD-p));
        endRead = bytesRead-(PICTURESIZE*(NUMBEROFPICTURESREAD-(p+1)));

        //read picture from startRead to endRead (byte locations) into picture array
        cout<<"Picture: "<<p<<" \nStart byte: "<<startRead<<" | End byte: "<<endRead<<endl;
        for(int i = startRead;i < endRead;i++){
            currentByte = buffer[i];
            if(row<28){
                if(col < 27){
                    col++;
                }else{
                    col = 0;
                    row++;
                }
                //cout<<row<<" "<<col<<endl;
                if(row < 28 && col <28)
                    picture[row][col] = currentByte;
            }
        }

    //(5) print information gathered onto command line
        string line = "";
        string displacement = "";
        int length;

        //draw image from picture array
        for(int i = 0;i < 28;i++){
            for(int x = 0;x < 28;x++){
                if(picture[i][x] != 0){
                    length = to_string((int)(picture[i][x])).length();

                    for(int r = 0;r < 4-length;r++)
                        displacement += " ";
                    line += to_string((int)(picture[i][x])) + displacement;
                    displacement = "";
                }else{
                    line += "    ";
                }
            }
            if(i<10)
                cout<<i<<":  "<<line<<endl;
            else
                cout<<i<<": "<<line<<endl;
            line = "";
        }

    //(6) draw images into files in "drawnImages"
        uint8_t pixels[784*4];
        int index = 0;

        for(int x = 0;x < 28;x++){
            for(int y = 0;y < 28;y++){

                for(int i = 0;i<3;i++){
                    index++;
                    pixels[index] = picture[x][y];
                    cout<<index<<endl;
                }
                index ++;
                pixels[index] = 255;
                
            }
        }

        // for(int i = 0;i<784*4;i++){
        //     cout<<int(pixels[i])<<endl;
        // }
        img.create(28,28,pixels);
        cout<<img.saveToFile("drawnImages/image" + to_string(p) + ".png")<<endl;

        row = 0;
        col = -1;
    }

    newFile.close();

    //(2) write text to file
    

    //TODO: 
    //read binary in idx image file: create an array of the image file containing the bytes of the file in order
    
    //split binary file data into a vector<vector<int>> that holds the images' pixel values

    //read binary in idx label file
    //split binary file data into vector<int> with the values for each index of the image vactor corresponding to the value of the index of the label vector
    
    //draw image data into image files and input that into seperate folders (0-9) corresponding to their labels
    
    return 0;
}