#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

int main(){
    
    std::vector<double> psiVal;
    std:: string line;

    std::ifstream ReadFile("interpolated.gf");

    int line_count = 0;
    double value;

    while (std::getline (ReadFile, line)) {

        line_count++; 

        if (line_count < 5) {
            continue;
        }

        if (line.empty()) {
            continue;
        }
        
        value = std::stod(line);

        psiVal.push_back(value);
    }

    ReadFile.close();

    std::sort(psiVal.begin(), psiVal.end()); 


    //UNFORMATTED PSI VALUES
    std::ofstream file("GEQDSK/GEQDSK_psi.txt");
    
    int val_count = 0;
    for (const auto& val : psiVal) {

        //Convert to scientific notation
        int exponent = (int)std::floor(std::log10(std::abs(val)));
        double decimal = val / std::pow(10, exponent);

        decimal = decimal/10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            file << std::fixed << std::setprecision(8) << " " << decimal << "E";
        } else {
            file << std::fixed << std::setprecision(8) << decimal << "E";
        }

        if (exponent >= 0) {
            file << "+" << std::setfill('0') << std::setw(2) << exponent;
        } else {
            file << exponent;
        }

        val_count++;

        if (val_count >= 5) {
            file << std::endl;
            val_count = 0;
        }
    }

    file.close();

    return 0;
}
