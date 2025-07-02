// File: GEQDSK-generation.cpp 
// Purpose: Generates the final GEQDSK plasma file by finding the individual variable files and appending them
// Run Instructions: make clean && make GEQDSK-generation && ./GEQDSK-generation 

#include <fstream>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <cstdlib>
using namespace std;

// Create initial/unorganized G-EQDSK file
void GEQDSK_header(int nx, int ny)
{     
    // Create folder for GEQDSK sub-files and final output (if it doesn't exist already)
    system("mkdir -p GEQDSK"); 
 
    // Create the GEQDSK file for rectangular mesh plasma region
    ofstream file("GEQDSK/GEQDSK.txt");

    // Build and insert header
    time_t t = time(nullptr);
    tm *lt = localtime(&t);
    char date_str[11];
    strftime(date_str, sizeof(date_str), "%d/%m/%Y", lt);

    string prefix = "MFEM       ";
    string rest = "        # 0  0ms              ";

    string line = prefix + string(date_str) + rest;

    if (line.size() < 48) line.resize(48, ' ');

    int i1 = 3;

    file << left << setw(48) << line
         << right << setw(4) << i1
         << setw(4) << nx
         << setw(4) << ny
         << endl;

    file.close();
}

int main(){
    // Build header
    int nx, ny; 
    ifstream file("GEQDSK/GEQDSK_nx_ny.txt");
    file >> nx >> ny; 
    file.close();
    GEQDSK_header(nx, ny);

    return 0; 
}

