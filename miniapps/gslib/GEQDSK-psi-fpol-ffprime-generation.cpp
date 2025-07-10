#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

std::vector<double> psiSort(){
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
    return psiVal;
}

std::vector<double> fpolCalc(double alpha, double psi_x, double f_x, const std::vector<double> psiVal) {
    //Calculate fpol values
    std::vector<double> fpol;
    double fpol_val;

    for (const auto& val : psiVal) {
        fpol_val = f_x + alpha * (val - psi_x);
        fpol.push_back(fpol_val);
    }

    return fpol;
}

void ffprimeCalc(double alpha, const std::vector<double>fpol) {
    std::vector<double> ffprime;
    double ffprime_val;

    for (const auto& val : fpol) {
        ffprime_val = alpha * val;
        ffprime.push_back(ffprime_val);
    }

    //Generate ffprime file

    std::ofstream file("GEQDSK/GEQDSK_ffprime.txt");
    
    int val_count = 0;

    for (const auto& val : ffprime) {

        //Convert to scientific notation
        int exponent = (int)std::floor(std::log10(std::abs(val)));
        double decimal = val / std::pow(10, exponent);

        decimal = decimal/10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            file << std::fixed << std::setprecision(9) << " " << decimal << "E";
        } else {
            file << std::fixed << std::setprecision(9) << decimal << "E";
        }

        if (exponent >= 0) {
            file << "+" << std::setfill('0') << std::setw(2) << exponent;
        } else {
            file << "-" << std::setfill('0') << std::setw(2) << std::abs(exponent);        
        }

        val_count++;

        if (val_count >= 5) {
            file << std::endl;
            val_count = 0;
        }
    }

    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0) {
        file << std::endl;
    }

    file.close();

}

void fpolFormat(const std::vector<double> fpol) {
    // Generate fpol file
    std::ofstream file("GEQDSK/GEQDSK_fpol.txt");

    int val_count = 0;

    for (const auto& val : fpol) {
        // Convert to scientific notation
        int exponent = (int)std::floor(std::log10(std::abs(val)));
        double decimal = val / std::pow(10, exponent);

        decimal = decimal / 10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            file << std::fixed << std::setprecision(9) << " " << decimal << "E";
        } else {
            file << std::fixed << std::setprecision(9) << decimal << "E";
        }

        if (exponent >= 0) {
            file << "+" << std::setfill('0') << std::setw(2) << exponent;
        } else {
            file << "-" << std::setfill('0') << std::setw(2) << std::abs(exponent);
        }

        val_count++;

        if (val_count >= 5) {
            file << std::endl;
            val_count = 0;
        }
    }

    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0) {
        file << std::endl;
    }

    file.close();
}


void psiFormat(const std::vector<double> psiVal) {
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
            file << std::fixed << std::setprecision(9) << " " << decimal << "E";
        } else {
            file << std::fixed << std::setprecision(9) << decimal << "E";
        }

        if (exponent >= 0) {
            file << "+" << std::setfill('0') << std::setw(2) << exponent;
        } else {
            file << "-" << std::setfill('0') << std::setw(2) << std::abs(exponent);        
        }

        val_count++;

        if (val_count >= 5) {
            file << std::endl;
            val_count = 0;
        }
    }

    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0) {
        file << std::endl;
    }
    
    file.close();
}

int main(){

    //Extract alpha, psi_x, f_x values
    std::ifstream ReadFile("GEQDSK/GEQDSK_alpha_psi_x_f_x.txt");
    std:: string line;
    int line_count = 0;
    double alpha;
    double psi_x;
    double f_x;

    while (std::getline (ReadFile, line)) {
        if (line_count == 0) {
            alpha = std::stod(line);
        } else if (line_count == 1) {
            psi_x = std::stod(line);
        } else if (line_count == 2) {
            f_x = std::stod(line);
        } else {
            continue;
        }
        line_count++;
    }
    ReadFile.close();

    
    std::vector<double> psi = psiSort();
    std::vector<double> fpol = fpolCalc(alpha, psi_x, f_x, psi);
    ffprimeCalc(alpha, fpol);
    fpolFormat(fpol);
    psiFormat(psi);

    return 0;
}
