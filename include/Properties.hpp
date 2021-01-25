#pragma once

class Properties {

        public:
        static const int numberOfHiddenLayers = 2;
        static const int nodesPerLayer = 16;
        static const int numberOfInputs = 784;
        static constexpr double learningSpeed = 0.01;
        static const int numberOfOutputs = 10;
};