#pragma once
#include <sqlite3.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include "../include/Properties.hpp"
#include "../include/Network.hpp"
#include "../include/Layer.hpp"

class NetworkSaver {
    public:
    static void SaveNetwork(const char* databaseAddress,  Network* network,  bool override = false, bool print = false, int iterationNumber = 0){

            //check to see if database exists already
            struct stat buffer;
            if((stat (databaseAddress, &buffer) == 0) && !override){
                string userInput;
                cout<<"A database at the location "<<databaseAddress<<" exists. Would you like to override that database? (Y/N)";
                cin>>userInput;
                if(userInput != "Y" || userInput != "y")
                    cout<<"Terminating Program"<<endl;
                    throw exception();
            }

            remove(databaseAddress);

            sqlite3* db;
            char* err;

            sqlite3_open(databaseAddress, &db);
            vector<Layer> layers = network->network;
            string stringOfWeights;
            string command;
            double currentNodeWeights[4];
            double currentWeight;
            int success;

            cout<<"Saving Network to "<<databaseAddress<<endl;
            for(int layer = 1;layer < layers.size();layer++){

                //create table for layer
                stringOfWeights = "(";
                for(int incomingNode = 0;incomingNode< layers[layer].weights.cols()-1;incomingNode++){
                    stringOfWeights += "node" + to_string(incomingNode) +" DOUBLE,";
                }


                stringOfWeights += "node" + to_string(layers[layer].weights.cols()-1) + " DOUBLE)";
                command = "CREATE TABLE LAYER" + to_string(layer) + stringOfWeights;
                if(print) cout<<command<<endl;
                success = sqlite3_exec(db, (command).c_str(), NULL, NULL, &err);

                if(success != SQLITE_OK){
                    cout<<"Error 1: "<<err<<endl; // << " | " <<command<<endl;
                    cout<<"Iteration Number: "<<iterationNumber<<endl;
                    for(int layer = 0;layer < network->network.size();layer++){
                            cout<<network->network[layer].weights.rows()<<" "<<network->network[layer].weights.cols()<<endl;
                       }
                    throw exception();
                }

                //fill table with nodes of the layer as rows and incoming nodes as the columns

                for(int node = 0; node < layers[layer].weights.rows();node++){
                    stringOfWeights = "(";
                    for(int incomingNode = 0;incomingNode < layers[layer].weights.cols()-1;incomingNode++){
                        currentWeight = layers[layer].weights(node,incomingNode);
                        stringOfWeights += to_string(currentWeight) + ", ";
                    }

                    currentWeight = layers[layer].weights(node, layers[layer].weights.cols()-1);
                    stringOfWeights += to_string(currentWeight) + ")";

                    command = "INSERT INTO LAYER"+to_string(layer)+" VALUES " + stringOfWeights;
                    if(print)cout<<command<<endl;
                    success = sqlite3_exec(db, (command).c_str(), NULL,NULL,&err);

                    if(success != SQLITE_OK){
                        cout<<"Error 2: "<<err<<endl;//<< " | " <<command<<endl;
                        cout<<"Iteration Number: "<<iterationNumber<<endl;
                        for(int layer = 0;layer < network->network.size();layer++){
                            cout<<network->network[layer].weights.rows()<<" "<<network->network[layer].weights.cols()<<endl;
                        }
                    throw exception();
                    }
                }
            }

            command = "CREATE TABLE PROPERTIES (numberOfHiddenLayers INT, nodesPerLayer INT, numberOfInputs INT, numberOfOutputs INT)";
            sqlite3_exec(db, (command).c_str(), NULL,NULL,&err);
            if(success != SQLITE_OK){
                        cout<<"Error 3: "<<err<<endl;//<< " | " <<command<<endl;
                        cout<<"Iteration Number: "<<iterationNumber<<endl;
                        for(int layer = 0;layer < network->network.size();layer++){
                            cout<<network->network[layer].weights.rows()<<" "<<network->network[layer].weights.cols()<<endl;
                        }
                        throw exception();
            }

            command = "INSERT INTO PROPERTIES VALUES (" + to_string(Properties::numberOfHiddenLayers) + ","+ to_string(Properties::nodesPerLayer) +","+ to_string(Properties::numberOfInputs) +","+ to_string(Properties::numberOfOutputs) +")";
            //cout<<command<<endl;
            sqlite3_exec(db, (command).c_str(), NULL,NULL,&err);
            if(success != SQLITE_OK){
                        cout<<"Error 4: "<<err<<endl;//<< " | " <<command<<endl;
                        cout<<"Iteration Number: "<<iterationNumber<<endl;
                        for(int layer = 0;layer < network->network.size();layer++){
                            cout<<network->network[layer].weights.rows()<<" "<<network->network[layer].weights.cols()<<endl;
                        }
                        throw exception();
            }
            
            sqlite3_close(db);
        }


    static Network * NetworkGetter(const char* databaseAddress){
            sqlite3* db;
            sqlite3_stmt * stmt;
            char* err;

            Network * net = new Network();

            cout<<"Getting Network from "<<databaseAddress<<endl;
            sqlite3_open(databaseAddress, &db);
            sqlite3_prepare_v2(db, "SELECT * FROM PROPERTIES", -1, &stmt,0);

            sqlite3_step(stmt);

            int numberOfHiddenLayers = sqlite3_column_int(stmt, 0);
            int nodesPerLayer = sqlite3_column_int(stmt, 1);
            int numberOfInputs = sqlite3_column_int(stmt, 2);
            int numberOfOutputs = sqlite3_column_int(stmt, 3);

            if(numberOfHiddenLayers != Properties::numberOfHiddenLayers)
                throw invalid_argument("The network in the database inputted into NetworkGetter does not match the template defined in Properties. " + to_string(numberOfHiddenLayers) +" != " + to_string(Properties::numberOfHiddenLayers));
            else if(nodesPerLayer != Properties::nodesPerLayer)
                throw invalid_argument("The network in the database inputted into NetworkGetter does not match the template defined in Properties. " + to_string(nodesPerLayer) +" != " + to_string(Properties::nodesPerLayer));
            else if(numberOfInputs != Properties::numberOfInputs)
                throw invalid_argument("The network in the database inputted into NetworkGetter does not match the template defined in Properties. " + to_string(numberOfInputs) +" != " + to_string(Properties::numberOfInputs));
            else if(numberOfOutputs != Properties::numberOfOutputs)
                throw invalid_argument("The network in the database inputted into NetworkGetter does not match the template defined in Properties. " + to_string(numberOfOutputs) +" != " + to_string(Properties::numberOfOutputs));


            //for first hidden layer
            sqlite3_prepare_v2(db, "SELECT * FROM LAYER1", -1, &stmt,0);
            for(int node = 0;node < nodesPerLayer;node++){
                sqlite3_step(stmt);
                for(int incomingNode = 0;incomingNode < numberOfInputs;incomingNode++){
                    net->network[1].weights(node,incomingNode) = sqlite3_column_double(stmt,incomingNode);
                }
            }

            //for deep layers
            for(int layer = 2;layer< numberOfHiddenLayers+1;layer++){
                sqlite3_prepare_v2(db, ("SELECT * FROM LAYER" + to_string(layer)).c_str(), -1, &stmt,0);
                for(int node = 0; node < nodesPerLayer;node++){
                    sqlite3_step(stmt);
                    for(int incomingNode = 0;incomingNode < nodesPerLayer;incomingNode++){
                        net->network[layer].weights(node,incomingNode) = sqlite3_column_double(stmt,incomingNode);
                    }
                }
            }

            //for last layer
            sqlite3_prepare_v2(db, ("SELECT * FROM LAYER" + to_string(numberOfHiddenLayers+1)).c_str(), -1, &stmt,0);
            for(int node = 0;node < numberOfOutputs;node++){
                sqlite3_step(stmt);
                for(int incomingNode = 0;incomingNode < nodesPerLayer; incomingNode++){
                    net->network[numberOfHiddenLayers+1].weights(node,incomingNode) = sqlite3_column_double(stmt,incomingNode);
                }
            }

            sqlite3_close(db);
            
            return net;
        }
};