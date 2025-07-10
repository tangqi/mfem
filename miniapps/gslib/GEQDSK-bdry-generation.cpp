#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <string>

void readCSV() {
    std::vector<double> r_data;
    std::vector<double> z_data;
    int nbdry = 0;

    std::ifstream file("../tds-gs/ParaView/gs/Cycle000000/contour_points.csv");

    if (!file.is_open()) {
        std::cerr <<"Failed to open file." << std::endl;
        return;
    }

    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        if (line_count == 0) {
            line_count++;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;

        double r_temp;
        double z_temp;
        
        int cell_count = 0;
        while (std::getline(ss, cell, ',')) {
            
            if (cell_count == 1) {
                r_temp = std::stod(cell);
            }

            if (cell_count == 2) {
                z_temp = std::stod(cell);
            }

            if (cell_count == 3) {
                //contour intersection @ (4.95502, -3.56733)
                if (r_temp >= 4.95502 && z_temp >= -3.56733) {
                    r_data.push_back(r_temp);
                    z_data.push_back(z_temp);
                    nbdry++;
                }
            }
            cell_count++;
        }
        line_count++;
    }

    file.close();

    std::ofstream NewFile("GEQDSK/GEQDSK_nbdry_rbdry_zbdry.txt");

    NewFile << nbdry << std::endl;
    
    int val_count = 0;

    for (int i = 0; i < nbdry; i++) {

        //Convert to scientific notation
        int exponent = (int)std::floor(std::log10(std::abs(r_data[i])));
        double decimal = r_data[i] / std::pow(10, exponent);

        decimal = decimal/10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            NewFile << std::fixed << std::setprecision(8) << " " << decimal << "E";
        } else {
            NewFile << std::fixed << std::setprecision(8) << decimal << "E";
        }

        if (exponent >= 0) {
            NewFile << "+" << std::setfill('0') << std::setw(2) << exponent;
        } else {
            NewFile << "-" << std::setfill('0') << std::setw(2) << std::abs(exponent);        
        }

        val_count++;

        if (val_count < 5) {
            int exponent = (int)std::floor(std::log10(std::abs(z_data[i])));
            double decimal = z_data[i] / std::pow(10, exponent);

            decimal = decimal/10.0;
            exponent = exponent + 1;

            if (decimal >= 0) {
                NewFile << std::fixed << std::setprecision(8) << " " << decimal << "E";
            } else {
                NewFile << std::fixed << std::setprecision(8) << decimal << "E";
            }

            if (exponent >= 0) {
                NewFile << "+" << std::setfill('0') << std::setw(2) << exponent;
            } else {
                NewFile << "-" << std::setfill('0') << std::setw(2) << std::abs(exponent);        
            }

            val_count++;
        }

        if (val_count >= 5) {
            NewFile << std::endl;
            val_count = 0;
        }
    }

    NewFile.close();

}

int main() {
    readCSV();
    return 0;
}