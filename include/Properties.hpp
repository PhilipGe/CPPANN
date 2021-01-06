#pragma once

class Properties {

        public:
        static const int numberOfHiddenLayers = 2;
        static const int nodesPerLayer = 200;
        static const int numberOfInputs = 784;
        static const int numberOfOutputs = 1;
        static constexpr double learningSpeed = 0.02;
};