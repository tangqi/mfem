// Calculate safety factor q(psi) as a line integral.

// Steps:
// 0) Load in pre-developed R and Z coordinates from text file and preprocess into 2D arrays. This text file contains the R and Z values for level sets of psi.
// 1) Choose the level set(s) to compute the line integral for.
// 2) Calculate the integrand for each (R_i, Z_i).
// 3) Compute segment lengths dl.
// 4) Compute the average value of the integrand over each segment.
// 5) Approximate the integral as a sum.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

// Function to compute dot product (integral approximation is effectively taking the dot product)
double dot_product(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Vectors must be the same size to compute dot product.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * y[i];
    }

    return sum;
}


int main() {

    // Step 0): Load in pre-developed R and Z coordinates from text file and preprocess into 2D arrays.

    // Read in radial_points.txt file
    std::ifstream file("./Contours/all_contour_radial_points.txt");

    if (!file.is_open()) {
        std::cerr << "Failed to open contours file." << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line); // Read header row

    // Count number of r/z column pairs we have, and also save header values as psi.
    std::istringstream headerStream(line);
    std::string header;
    int rCount = 0, zCount = 0;

    std::vector<double> psi;  // 1D psi array for the level sets of psi

    while (std::getline(headerStream, header, '\t')) {

        // Count r/z columns
        if (header[0] == 'r') rCount++;
        else if (header[0] == 'z') zCount++;

        // Extract header values from r for psi array
        if (header[0] == 'r') {
            std::string psi_val = header.substr(2);
            psi.push_back(std::stod(psi_val));
        }
    }

    // Raise error if the number of r values does not equal the number of z values
    if (rCount != zCount) {
        std::cerr << "Header format error: Unequal number of 'r' and 'z' columns.\n";
        std::cerr << "Found " << rCount << " r-columns and " << zCount << " z-columns.\n";
        return 1;
    }

    // Construct 2D arrays, one for R and one for Z
    std::vector<std::vector<double>> R;
    std::vector<std::vector<double>> Z;

    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string value;
        std::vector<double> rRow, zRow;

        for (size_t i = 0; i < rCount; ++i) {
            std::getline(lineStream, value, '\t');
            rRow.push_back(value == "nan" ? NAN : std::stod(value));

            std::getline(lineStream, value, '\t');
            zRow.push_back(value == "nan" ? NAN : std::stod(value));
        }

        R.push_back(rRow);
        Z.push_back(zRow);
    }

    // Remove the last column in R and Z, which corresponds to the magnetic axis (values are NAN)
    for (auto& row : R) {
        if (!row.empty()) {
            row.pop_back();
        }
    }

    for (auto& row : Z) {
        if (!row.empty()) {
            row.pop_back();
        }
    }

    if (!psi.empty()) {
        psi.pop_back();
    }

    file.close();

    // Verify that R and Z are the same shape
    size_t rRows = R.size();
    size_t rCols = (rRows > 0) ? R[0].size() : 0;
    size_t zRows = Z.size();
    size_t zCols = (zRows > 0) ? Z[0].size() : 0;
    if (rRows != zRows || rCols != zCols){
        std::cerr << "R matrix shape ( " << rRows << "," << rCols << " ) "
        << "shape does not equal Z matrix shape ( " << zRows << "," << zCols << " )\n";
        return 1;
    }

    // DEBUG: print psi level set values
    std::cout << "psi: ";
    for (double val : psi) std::cout << val << ' ';
    std::cout << '\n';

    // Step 1): Choose the level set(s) to compute the line integral for.

    std::vector<double> qpsi;  // Store q(psi) for each level set of psi
    for (size_t j = 0; j < R[0].size(); ++j) {

        std::vector<double> r_col, z_col;
        for (size_t row = 0; row < R.size(); ++row) {
            r_col.push_back(R[row][j]);
            z_col.push_back(Z[row][j]);
        }

        // Step 2): Calculate the integrand for each (R_i, Z_i).

        // f(psi) (Taylor state equilibrium)
        double alpha = 0.144526;
        double psi_x = 1.28864;
        double f_x = -32.86;
        double f = f_x + alpha * (psi[j] - psi_x);

        // // Gradient of psi
        // size_t numLevels = psi.size();
        // size_t M = r_col.size();
        // gradPsi.reserve(M);
        // for (size_t i = 0; i < M; ++i) {
        //     double dpsi, dR, dZ
        // }

        // Integrand f(psi) / R^2
        std::vector<double> integrand;
        for(size_t i = 0; i < r_col.size(); ++i) {
            double denom = r_col[i] * r_col[i];
            if (denom != 0.0) {
                integrand.push_back(f / denom);
            }
            else {
                throw std::runtime_error("Division by zero encountered when computing the integrand.");
            }
        }

        // Step 3): Compute segment lengths dl.
        std::vector<double> dl;
        size_t N = r_col.size();
        for (size_t i = 0; i < N - 1; ++i) {
            double delta_r = r_col[i + 1] - r_col[i];
            double delta_z = z_col[i + 1] - z_col[i];
            double dist = std::sqrt(delta_r * delta_r + delta_z * delta_z);
            dl.push_back(dist);
        }

        // Close loop
        {
            double delta_r = r_col[0] - r_col[N - 1];
            double delta_z = z_col[0] - z_col[N - 1];
            double dist = std::sqrt(delta_r * delta_r + delta_z * delta_z);
            dl.push_back(dist);
        }

        // Step 4): Compute the average value of the integrand over each segment.
        std::vector<double> avg_integrand;
        N = integrand.size();
        for (size_t i = 0; i < N; ++i) {
            size_t next = (i + 1) % N;
            double avg = 0.5 * (integrand[i] + integrand[next]);
            avg_integrand.push_back(avg);
        }

        // Step 5): Approximate the integral as a sum.
        double q = (1 / (2 * 3.14159265358979323846)) * dot_product(avg_integrand, dl);

        // Store q value
        qpsi.push_back(q);
    }

    // DEBUG: print q(psi)
    std::cout << '\n';
    std::cout << "q(psi): ";
    for(size_t j = 0; j < qpsi.size(); ++j) {
        std::cout << qpsi[j] << ' ';
    }
    std::cout << '\n';

    // Save output as qpsi.txt, in Fortran format (5e16.9)
    std::ofstream outfile("./GEQDSK/GEQDSK_qpsi.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open GEQDSK/GEQDSK_qpsi.txt \n";
        return 1;
    }

    outfile << std::uppercase << std::scientific << std::setprecision(9) << std::showpos;

    const int fieldWidth = 14;

    for (size_t i = 0; i < qpsi.size(); ++i) {
        outfile << std::setw(fieldWidth) << qpsi[i];
        if ((i + 1) % 5 == 0) outfile << '\n';  // Newline after every 5 entries
    }

    outfile.close();

    return 0;
}
