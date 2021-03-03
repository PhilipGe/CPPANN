#include <eigen/Eigen/Dense>
#include "MNISTTrainerV2.hpp"
#include <fstream>

using namespace Eigen;
using namespace std;

class ImageHolder{

    public:

    MatrixXd image;
    MatrixXd output;

    ImageHolder(MatrixXd img, double value){
        image = img;
        output = MNISTTrainerV2::getOutputMatrix(value);
    }

};