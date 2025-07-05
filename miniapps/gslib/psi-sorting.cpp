#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

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
    
    for (const auto& val : psiVal) {
        file << val << std::endl;
    }

    file.close();

    return 0;
}
