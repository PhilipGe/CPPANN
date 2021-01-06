#include <gtest/gtest.h>
#include <eigen/Eigen/Dense>
#include "../src/Layer.cpp"
#include "../src/Network.cpp"
#include "basicFunctionsTestInitialization.hpp"
#include <limits>
#include <eigen/Eigen/Dense>
#include "../include/Trainer.hpp"
#include "../FileStorage/NetworkSaver.cpp"

TEST (trials, DISABLED_one){

    /*Properties:
    public:
        static const int numberOfInputs = 3;
        static const int numberOfOutputs = 2;
    */
   
    srand(100);
    Network *net = new Network();

    MatrixXd inputs1(3,1);
    inputs1 << 0, 0, 0;
    MatrixXd inputs2(3,1);
    inputs2 << 0, 0, 1;
    MatrixXd inputs3(3,1);
    inputs3 << 0, 1, 0;
    MatrixXd inputs4(3,1);
    inputs4 << 0, 1, 1;
    MatrixXd inputs5(3,1);
    inputs5 << 1, 0, 0;
    MatrixXd inputs6(3,1);
    inputs6 << 1, 0, 1;
    MatrixXd inputs7(3,1);
    inputs7 << 1, 1, 0;
    MatrixXd inputs8(3,1);
    inputs8 << 1, 1, 1;

    vector<MatrixXd> inputs;
    inputs.push_back(inputs1);
    inputs.push_back(inputs2);
    inputs.push_back(inputs3);
    inputs.push_back(inputs4);
    inputs.push_back(inputs5);
    inputs.push_back(inputs6);
    inputs.push_back(inputs7);
    inputs.push_back(inputs8);

    vector<MatrixXd> desiredOutputs;
    MatrixXd outputs1(2,1);
    outputs1<< 1, 0;
    MatrixXd outputs2(2,1);
    outputs2<< 1, 0;
    MatrixXd outputs3(2,1);
    outputs3<< 0, 1;
    MatrixXd outputs4(2,1);
    outputs4<< 0, 1;
    MatrixXd outputs5(2,1);
    outputs5<< 0, 1;
    MatrixXd outputs6(2,1);
    outputs6<< 0, 1;
    MatrixXd outputs7(2,1);
    outputs7<< 1, 0;
    MatrixXd outputs8(2,1);
    outputs8<< 1, 0;

    desiredOutputs.push_back(outputs1);
    desiredOutputs.push_back(outputs2);
    desiredOutputs.push_back(outputs3);
    desiredOutputs.push_back(outputs4);
    desiredOutputs.push_back(outputs5);
    desiredOutputs.push_back(outputs6);
    desiredOutputs.push_back(outputs7);
    desiredOutputs.push_back(outputs8);
    
    Trainer::train(net,inputs,desiredOutputs,10000,true);

    for(int i =0;i<inputs.size();i++){
        // EXPECT_NEAR(desiredOutputs[i],net->feedForward(inputs[i]),0.1);
    }

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<net->feedForward(inputs[i]).transpose()<<" Desired: "<<desiredOutputs[i].transpose()<<endl;
    }
}

TEST (trials, DISABLED_two){

    /*Properties:
    public:
        static const int numberOfInputs = 3;
        static const int numberOfOutputs = 2;
    */
   
    srand(100);
    Network *net = new Network();

    MatrixXd inputs1(3,1);
    inputs1 << 0, 0, 0;
    MatrixXd inputs2(3,1);
    inputs2 << 0, 0, 1;
    MatrixXd inputs3(3,1);
    inputs3 << 0, 1, 0;
    MatrixXd inputs4(3,1);
    inputs4 << 0, 1, 1;
    MatrixXd inputs5(3,1);
    inputs5 << 1, 0, 0;
    MatrixXd inputs6(3,1);
    inputs6 << 1, 0, 1;
    MatrixXd inputs7(3,1);
    inputs7 << 1, 1, 0;
    MatrixXd inputs8(3,1);
    inputs8 << 1, 1, 1;

    vector<MatrixXd> inputs;
    inputs.push_back(inputs1);
    inputs.push_back(inputs2);
    inputs.push_back(inputs3);
    inputs.push_back(inputs4);
    inputs.push_back(inputs5);
    inputs.push_back(inputs6);
    inputs.push_back(inputs7);
    inputs.push_back(inputs8);

    vector<MatrixXd> desiredOutputs;
    MatrixXd outputs1(1,1);
    outputs1<< 2;
    MatrixXd outputs2(1,1);
    outputs2<< 2;
    MatrixXd outputs3(1,1);
    outputs3<< 0;
    MatrixXd outputs4(1,1);
    outputs4<< 0;
    MatrixXd outputs5(1,1);
    outputs5<< 0;
    MatrixXd outputs6(1,1);
    outputs6<< 0;
    MatrixXd outputs7(1,1);
    outputs7<< 2;
    MatrixXd outputs8(1,1);
    outputs8<< 2;

    desiredOutputs.push_back(outputs1);
    desiredOutputs.push_back(outputs2);
    desiredOutputs.push_back(outputs3);
    desiredOutputs.push_back(outputs4);
    desiredOutputs.push_back(outputs5);
    desiredOutputs.push_back(outputs6);
    desiredOutputs.push_back(outputs7);
    desiredOutputs.push_back(outputs8);
    
    Trainer::train(net,inputs,desiredOutputs,10000,true);

    for(int i =0;i<inputs.size();i++){
        // EXPECT_NEAR(desiredOutputs[i],net->feedForward(inputs[i]),0.1);
    }

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<net->feedForward(inputs[i]).transpose()<<" Desired: "<<desiredOutputs[i].transpose()<<endl;
    }
}

TEST (trials, DISABLED_binary){

    /*
    Properties:
    public: 
        static const int numberOfHiddenLayers = 3;
        static const int nodesPerLayer = 4;
        static const int numberOfInputs = 4;
        static constexpr double learningSpeed = 0.01;
    */


    Network *net = new Network();

    MatrixXd inputs0(4,1);
    inputs0 << 0, 0, 0, 0;
    MatrixXd inputs1(4,1);
    inputs1 << 0, 0, 0, 1;
    MatrixXd inputs2(4,1);
    inputs2 << 0, 0, 1, 0;
    MatrixXd inputs3(4,1);
    inputs3 << 0, 0, 1, 1;
    MatrixXd inputs4(4,1);
    inputs4 << 0, 1, 0, 0;
    MatrixXd inputs5(4,1);
    inputs5 << 0, 1, 0, 1;
    MatrixXd inputs6(4,1);
    inputs6 << 0, 1, 1, 0;
    MatrixXd inputs7(4,1);
    inputs7 << 0, 1, 1, 1;
    MatrixXd inputs8(4,1);
    inputs8 << 1, 0, 0, 0;
    MatrixXd inputs9(4,1);
    inputs9 << 1, 0, 0, 1;
    MatrixXd inputs10(4,1);
    inputs10 << 1, 0, 1, 0;
    MatrixXd inputs11(4,1);
    inputs11 << 1, 0, 1, 1;
    MatrixXd inputs12(4,1);
    inputs12 << 1, 1, 0, 0;
    MatrixXd inputs13(4,1);
    inputs13 << 1, 1, 0, 1;
    MatrixXd inputs14(4,1);
    inputs14 << 1, 1, 1, 0;
    MatrixXd inputs15(4,1);
    inputs15 << 1, 1, 1, 1;


    vector<MatrixXd> inputs;
    inputs.push_back(inputs0);
    inputs.push_back(inputs1);
    inputs.push_back(inputs2);
    inputs.push_back(inputs3);
    inputs.push_back(inputs4);
    inputs.push_back(inputs5);
    inputs.push_back(inputs6);
    inputs.push_back(inputs7);
    inputs.push_back(inputs8);
    inputs.push_back(inputs9);
    inputs.push_back(inputs10);
    inputs.push_back(inputs11);
    inputs.push_back(inputs12);
    inputs.push_back(inputs13);
    inputs.push_back(inputs14);
    inputs.push_back(inputs15); 

    vector<MatrixXd> desiredOutputs;
    MatrixXd sixteenZeros = MatrixXd::Constant(16,1,0);
    
    for(int i = 0;i < 16; i++){
        desiredOutputs.push_back(sixteenZeros);
        desiredOutputs[i](i,0) = 1;
        cout<<desiredOutputs[i].transpose()<<endl;
    }

    Trainer::train(net,inputs,desiredOutputs,10000,true);

    inputs.clear();
    inputs.push_back(inputs0);
    inputs.push_back(inputs1);
    inputs.push_back(inputs2);
    inputs.push_back(inputs3);
    inputs.push_back(inputs4);
    inputs.push_back(inputs5);
    inputs.push_back(inputs6);
    inputs.push_back(inputs7);
    inputs.push_back(inputs8);
    inputs.push_back(inputs9);
    inputs.push_back(inputs10);
    inputs.push_back(inputs11);
    inputs.push_back(inputs12);
    inputs.push_back(inputs13);
    inputs.push_back(inputs14);
    inputs.push_back(inputs15);

    vector<double> checkDesiredOutputs {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    
    for(int i =0;i<inputs.size();i++){
        // EXPECT_NEAR(net->feedForward(inputs[i]),(checkDesiredOutputs[i]/8.0)-1, 0.1);
    }

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<(net->feedForward(inputs[i])*100).array().round().abs().transpose()/100<<" Desired: "<<desiredOutputs[i].transpose()<<endl;
    }

}

TEST (trials, binary_and_saving_to_disk){

    /*
    Properties:
    public: 
        static const int numberOfHiddenLayers = 3;
        static const int nodesPerLayer = 4;
        static const int numberOfInputs = 4;
        static const int numberOfOutputs = 16;
        static constexpr double learningSpeed = 0.01;
    */


    Network *net = new Network();

    MatrixXd inputs0(4,1);
    inputs0 << 0, 0, 0, 0;
    MatrixXd inputs1(4,1);
    inputs1 << 0, 0, 0, 1;
    MatrixXd inputs2(4,1);
    inputs2 << 0, 0, 1, 0;
    MatrixXd inputs3(4,1);
    inputs3 << 0, 0, 1, 1;
    MatrixXd inputs4(4,1);
    inputs4 << 0, 1, 0, 0;
    MatrixXd inputs5(4,1);
    inputs5 << 0, 1, 0, 1;
    MatrixXd inputs6(4,1);
    inputs6 << 0, 1, 1, 0;
    MatrixXd inputs7(4,1);
    inputs7 << 0, 1, 1, 1;
    MatrixXd inputs8(4,1);
    inputs8 << 1, 0, 0, 0;
    MatrixXd inputs9(4,1);
    inputs9 << 1, 0, 0, 1;
    MatrixXd inputs10(4,1);
    inputs10 << 1, 0, 1, 0;
    MatrixXd inputs11(4,1);
    inputs11 << 1, 0, 1, 1;
    MatrixXd inputs12(4,1);
    inputs12 << 1, 1, 0, 0;
    MatrixXd inputs13(4,1);
    inputs13 << 1, 1, 0, 1;
    MatrixXd inputs14(4,1);
    inputs14 << 1, 1, 1, 0;
    MatrixXd inputs15(4,1);
    inputs15 << 1, 1, 1, 1;


    vector<MatrixXd> inputs;
    inputs.push_back(inputs0);
    inputs.push_back(inputs1);
    inputs.push_back(inputs2);
    inputs.push_back(inputs3);
    inputs.push_back(inputs4);
    inputs.push_back(inputs5);
    inputs.push_back(inputs6);
    inputs.push_back(inputs7);
    inputs.push_back(inputs8);
    inputs.push_back(inputs9);
    inputs.push_back(inputs10);
    inputs.push_back(inputs11);
    inputs.push_back(inputs12);
    inputs.push_back(inputs13);
    inputs.push_back(inputs14);
    inputs.push_back(inputs15); 

    vector<MatrixXd> desiredOutputs;
    MatrixXd sixteenZeros = MatrixXd::Constant(16,1,0);
    
    for(int i = 0;i < 16; i++){
        desiredOutputs.push_back(sixteenZeros);
        desiredOutputs[i](i,0) = 1;
        //cout<<desiredOutputs[i].transpose()<<endl;
    }

    Trainer::train(net,inputs,desiredOutputs,10000,true);

    inputs.clear();
    inputs.push_back(inputs0);
    inputs.push_back(inputs1);
    inputs.push_back(inputs2);
    inputs.push_back(inputs3);
    inputs.push_back(inputs4);
    inputs.push_back(inputs5);
    inputs.push_back(inputs6);
    inputs.push_back(inputs7);
    inputs.push_back(inputs8);
    inputs.push_back(inputs9);
    inputs.push_back(inputs10);
    inputs.push_back(inputs11);
    inputs.push_back(inputs12);
    inputs.push_back(inputs13);
    inputs.push_back(inputs14);
    inputs.push_back(inputs15);

    vector<double> checkDesiredOutputs {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    cout<<"Before Database:"<<endl;

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<(net->feedForward(inputs[i])*100).array().round().abs().transpose()/100<<" Desired: "<<desiredOutputs[i].transpose()<<endl;
    }

    remove("testTwoDB.db");
    NetworkSaver::SaveNetwork("testTwoDB.db", net);
    net = NetworkSaver::NetworkGetter("testTwoDB.db");

    cout<<endl<<"After Database:"<<endl;

    for(int i =0;i<inputs.size();i++){
        cout<<"Input: "<<inputs[i].transpose()<<" Output: "<<(net->feedForward(inputs[i])*100).array().round().abs().transpose()/100<<" Desired: "<<desiredOutputs[i].transpose()<<endl;
    }


}

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}