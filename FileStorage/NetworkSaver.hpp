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
    static void SaveNetwork(const char* databaseAddress,  Network* network,  bool override = false, bool print = false);
    static Network * NetworkGetter(const char* databaseAddress);
};