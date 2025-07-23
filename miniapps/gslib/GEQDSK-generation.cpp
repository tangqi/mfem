// File: GEQDSK-generation.cpp 
// Purpose: Generates the final GEQDSK plasma file by finding the individual variable files and appending them
// Workflow: The respective MFEM files generate "GEQDSK_[variable_names]" files that store the values of the needed variables. GEQDDSK-generation.cpp extracts the values from those files, generates other needed variables, and compiles the final GEQDSK.txt file. 
// File un instructions: Run a triangular mesh file (Ex: sh run_3_taylor.sh)
//                       Run 2D-cartesian-mesh.cpp (make clean && make 2d-cartesian-mesh && srun -n 1 ./2d-cartesian-mesh) 
// Note: Running 2d-cartesian-mesh.cpp automatically runs GEQDSK-generation.cpp. if want to run independently, execute: make GEQDSK-generation && ./GEQDSK-generation 

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <regex>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace mfem;
using namespace std;

// Find simagx coordinates: rmagx, zmagx
void find_rmagx_zmagx(const string& rect_gf_file, const string& rect_mesh_file, float simagx, float &rmagx, float &zmagx)
{
    // Open .gf file for rectangular mesh and find the index of the closest psi to simagx
    ifstream file_solution(rect_gf_file); 
    if (!file_solution.is_open()){
        cerr << "File can't be opened: " << rect_gf_file << "\n" << endl;   
    }

    string line; 
    int counter = 1, closest_psi_idx;
    float prev_diff = 1000, current_diff, closest_psi;

    for(int i = 0; i < 4 && getline(file_solution, line); ++i); //Skips first 4 lines
    while(getline(file_solution, line)){
        if(line.empty()) continue;

        float new_psi = stof(line);
        current_diff = abs(simagx - new_psi); 
       
        if(current_diff < prev_diff){
            prev_diff = current_diff; 
            closest_psi = new_psi;
            closest_psi_idx = counter;  
        }
        
        ++counter;
    }

    printf("simagx = %.5f\n", simagx);
    printf("Closest psi val = %.5f\n", closest_psi);
    printf("index: %d\n", closest_psi_idx); 

    file_solution.close(); 

    // Use closest_psi_idx to get rmagx and zmagx
    ifstream file_mesh(rect_mesh_file);
    if (!file_mesh.is_open()){
        cerr << "File can't be opened: " << rect_mesh_file << "\n" << endl;   
    } 

    string line1; 
    int counter1 = 0;
    bool found_title = false;

    while (getline(file_mesh, line1))
    {
        if (line1 == "vertices")
        {
            printf("vertices line found\n");
            getline(file_mesh, line1); // Skip # of vertices line
            getline(file_mesh, line1); // Skip dimension line
            found_title = true; 
            counter1 = 0;
        }

        if (found_title == true && counter1 == closest_psi_idx){
            istringstream iss(line1);
            iss >> rmagx >> zmagx;   
            
            // printf("rmagx: = %.5f\n", rmagx);
            // printf("zmagx: %.5f\n", zmagx); 
            break;   
        }
        counter1 = counter1 + 1;
    }
    file_mesh.close(); 
}
    
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

// Format and append Section 1
void append_section1(float rdim, float zdim, float rcentr, float rleft, float zmid, float rmagx, float zmagx, float simagx, float sibdry, float bcentr, float cpasma)
{
    ofstream file("GEQDSK/GEQDSK.txt", ios::app);
    if (!file.is_open()) {
        cerr << "Could not open GEQDSK.txt" << endl;
        return;
    }

    // Set scientific format, width = 16, precision = 9
    file << uppercase << scientific << setprecision(9);
    file << " " << endl;

    // 1st line
    file << setw(16) << rdim
         << setw(16) << zdim
         << setw(16) << rcentr
         << setw(16) << rleft
         << setw(16) << zmid << endl;

    // 2nd line
    file << setw(16) << rmagx
         << setw(16) << zmagx
         << setw(16) << simagx
         << setw(16) << sibdry
         << setw(16) << bcentr << endl;

    // 3rd line
    file << setw(16) << cpasma
         << setw(16) << simagx
         << setw(16) << 0.0
         << setw(16) << rmagx
         << setw(16) << 0.0 << endl;

    // 4th line
    file << setw(16) << zmagx
         << setw(16) << 0.0
         << setw(16) << sibdry
         << setw(16) << 0.0
         << setw(16) << 0.0 << endl;

    file.close();
}

//Sort psi values from interpolated.gf
vector<double> psiSort(){
    vector<double> psiVal;
     string line;

    ifstream ReadFile("interpolated.gf");

    int line_count = 0;
    double value;

    while (getline (ReadFile, line)) {

        line_count++; 

        if (line_count < 5) {
            continue;
        }

        if (line.empty()) {
            continue;
        }
        
        value = stod(line);

        psiVal.push_back(value);
    }

    ReadFile.close();

    sort(psiVal.begin(), psiVal.end()); 
    return psiVal;
}


//Calculate fpol values
vector<double> fpolCalc(double alpha, double psi_x, double f_x, const vector<double> psiVal) {
    //Calculate fpol values
    vector<double> fpol;
    double fpol_val;

    for (const auto& val : psiVal) {
        fpol_val = f_x + alpha * (val - psi_x);
        fpol.push_back(fpol_val);
    }

    return fpol;
}

//Generation ffprime file
void ffprimeCalc(double alpha, const vector<double>fpol) {
    vector<double> ffprime;
    double ffprime_val;

    for (const auto& val : fpol) {
        ffprime_val = alpha * val;
        ffprime.push_back(ffprime_val);
    }

    //Generate ffprime file

    ofstream file("GEQDSK/GEQDSK_ffprime.txt");
    
    int val_count = 0;

    for (const auto& val : ffprime) {

        //Convert to scientific notation
        int exponent = (int)floor(log10(abs(val)));
        double decimal = val / pow(10, exponent);

        decimal = decimal/10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            file << fixed << setprecision(9) << " " << decimal << "E";
        } else {
            file << fixed << setprecision(9) << decimal << "E";
        }

        if (exponent >= 0) {
            file << "+" << setfill('0') << setw(2) << exponent;
        } else {
            file << "-" << setfill('0') << setw(2) << abs(exponent);        
        }

        val_count++;

        if (val_count >= 5) {
            file << endl;
            val_count = 0;
        }
    }

    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0) {
        file << endl;
    }

    file.close();

}

//Generate fpol file 
void fpolFormat(const vector<double> fpol) {
    // Generate fpol file
    ofstream file("GEQDSK/GEQDSK_fpol.txt");

    int val_count = 0;

    for (const auto& val : fpol) {
        // Convert to scientific notation
        int exponent = (int)floor(log10(abs(val)));
        double decimal = val / pow(10, exponent);

        decimal = decimal / 10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            file << fixed << setprecision(9) << " " << decimal << "E";
        } else {
            file << fixed << setprecision(9) << decimal << "E";
        }

        if (exponent >= 0) {
            file << "+" << setfill('0') << setw(2) << exponent;
        } else {
            file << "-" << setfill('0') << setw(2) << abs(exponent);
        }

        val_count++;

        if (val_count >= 5) {
            file << endl;
            val_count = 0;
        }
    }

    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0) {
        file << endl;
    }

    file.close();
}

//Generation psi file
void psiFormat(const vector<double> psiVal) {
    //UNFORMATTED PSI VALUES
    ofstream file("GEQDSK/GEQDSK_psi.txt");
    
    int val_count = 0;
    for (const auto& val : psiVal) {

        //Convert to scientific notation
        int exponent = (int)floor(log10(abs(val)));
        double decimal = val / pow(10, exponent);

        decimal = decimal/10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            file << fixed << setprecision(9) << " " << decimal << "E";
        } else {
            file << fixed << setprecision(9) << decimal << "E";
        }

        if (exponent >= 0) {
            file << "+" << setfill('0') << setw(2) << exponent;
        } else {
            file << "-" << setfill('0') << setw(2) << abs(exponent);        
        }

        val_count++;

        if (val_count >= 5) {
            file << endl;
            val_count = 0;
        }
    }

    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0) {
        file << endl;
    }
    
    file.close();
}

// Generate GEQDSK_pres.txt
void generate_pres(){
    ifstream infile("GEQDSK/GEQDSK_fpol.txt");
    ofstream outfile("GEQDSK/GEQDSK_pres.txt");

    double dummy;
    int count = 0;

    // Read value by value (assuming file only has Fortran-formatted values)
    while (infile >> dummy) {
        // Force format: width 16, scientific, 9 decimals, uppercase E
        outfile << setw(16) << setprecision(9)
                << uppercase << scientific << 0.0;
        count++;

        // Newline every 5 values
        if (count % 5 == 0) {
            outfile << '\n';
        }
    }

    // Add newline if last line is incomplete
    if (count % 5 != 0) {
        outfile << '\n';
    }

    infile.close();
    outfile.close();
}

// Generate GEQDSK_pprime.txt
void generate_pprime(){
    ifstream infile("GEQDSK/GEQDSK_pres.txt");
    ofstream outfile("GEQDSK/GEQDSK_pprime.txt");

    outfile << infile.rdbuf();

    infile.close();
    outfile.close(); 
}

// Append subfiles to GEQDSK.txt
void append_to_GEQDSK_txt(const string& infile) {
    ifstream source(infile, ios::binary);         // File to append
    ofstream dest("GEQDSK/GEQDSK.txt", ios::app | ios::binary); // File to append *to*

    dest << "\n"; 
    dest << source.rdbuf(); // Append the entire content

    source.close();
    dest.close();
}


vector<double> ExtractContourLine(const Mesh &mesh, const GridFunction &u, double level)
{
    MFEM_VERIFY(mesh.Dimension() == 2, "Only 2D meshes are supported.");
    MFEM_VERIFY(u.FESpace()->GetVDim() == 1, "Only scalar fields are supported.");

    const FiniteElementSpace &fes = *u.FESpace();
    const GridFunction *nodes = mesh.GetNodes();
    MFEM_VERIFY(nodes, "Mesh must be in nodal form (use high-order mesh).");

    const int nedges = mesh.GetNEdges();
    const int dim = 2;

    vector<double> rbdry_zbdry;

    //ofstream out("contour_check.txt");

    for (int e = 0; e < nedges; ++e)
    {
        // Get the vertex indices of this edge
        Array<int> ev;
        mesh.GetEdgeVertices(e, ev);

        // Get coordinates of endpoints
        const double *coords_i = mesh.GetVertex(ev[0]);
        const double *coords_j = mesh.GetVertex(ev[1]);

        Array<int> dofs_i, dofs_j;

        // Evaluate u at the vertices
        fes.GetVertexDofs(ev[0], dofs_i);
        fes.GetVertexDofs(ev[1], dofs_j);
        double ui = u(dofs_i[0]);
        double uj = u(dofs_j[0]);
        // Check if contour level crosses this edge
        if ((ui - level) * (uj - level) < 0.0)
        {
            //cout<<ui<<" "<<uj<<" ";
            // Linear interpolation to find contour crossing
            double alpha = (level - ui) / (uj - ui);
            Vector pt(dim);
            for (int d = 0; d < dim; d++)
            {
                pt[d] = coords_i[d] + alpha * (coords_j[d] - coords_i[d]);
            }
            
            //Specific to mesh, cannot be used universally
            if (pt[1] >= -3.56733) {
                rbdry_zbdry.push_back(pt[0]);
                rbdry_zbdry.push_back(pt[1]);

                //out << pt[1] << " " << pt[0] << "\n";
            }
        }
    }

    //out.close();
    return rbdry_zbdry;
}


// Format and append rbdry & zbdry
int append_rbdry_zbdry(const vector<double>& rbdry_zbdry) {
    int val_count = 0;

    ofstream outfile("GEQDSK/GEQDSK.txt", ios::app);
    if (!outfile.is_open()) {
        cerr << "Could not open GEQDSK/GEQDSK.txt" << endl;
        return val_count;
    }

    outfile << '\n';

    for (double val : rbdry_zbdry) {
        int exponent = (int)floor(log10(abs(val)));
        double decimal = val / pow(10, exponent);

        decimal = decimal/10.0;
        exponent = exponent + 1;

        if (decimal >= 0) {
            outfile << fixed << setprecision(9) << " " << decimal << "E";
        } else {
            outfile << fixed << setprecision(9) << decimal << "E";
        }

        if (exponent >= 0) {
            outfile << "+" << setfill('0') << setw(2) << exponent;
        } else {
            outfile << "-" << setfill('0') << setw(2) << abs(exponent);        
        }

        val_count++;

        if (val_count >= 5) {
            outfile << endl;
            val_count = 0;
        }
    }
    outfile.close();
    return val_count;
}

// Format and append rlim and zlim
void append_rlim_zlim(const int& num, const vector<float>& rlim_zlim) {
    ofstream outfile("GEQDSK/GEQDSK.txt", ios::app);
    if (!outfile.is_open()) {
        cerr << "Could not open GEQDSK/GEQDSK.txt" << endl;
        return;
    }

    outfile << uppercase << scientific << setprecision(9);

    int count = num;
    for (float val : rlim_zlim) {
        outfile << setw(16) << val;
        count++;
        if (count >= 5) {
            outfile << endl;
            count = 0;
        }
    }

    if (count > 0) {
        outfile << "\n"; // Final newline if not divisible by 5
    }

    outfile.close();
}


int main(){
    // Known constants
    float rcentr = 6.200000286e+00; // [meter] Reference value of R 
    float bcentr = -5.300000000e+00; // [tesla] Vacuum toroidal magnetic field at rcentr
    int nlim = 56; // Number of points in the limiter grid, value gotten from tds-gs/data/seperated_file.data
    double alpha = 0.144526;
    double psi_x = 1.28864;
    double f_x = -32.86;
    
    vector<float> rlim_zlim = {
        6.267000e+00, -3.046000e+00, 7.283000e+00, -2.257000e+00,
        7.899000e+00, -1.342000e+00, 8.306000e+00, -4.210000e-01,
        8.395000e+00,  6.330000e-01, 8.270000e+00,  1.681000e+00,
        7.904000e+00,  2.464000e+00, 7.400000e+00,  3.179000e+00,
        6.587000e+00,  3.894000e+00, 5.753000e+00,  4.532000e+00,
        4.904000e+00,  4.712000e+00, 4.311000e+00,  4.324000e+00,
        4.126000e+00,  3.582000e+00, 4.076000e+00,  2.566000e+00,
        4.046000e+00,  1.549000e+00, 4.046000e+00,  5.330000e-01,
        4.067000e+00, -4.840000e-01, 4.097000e+00, -1.500000e+00,
        4.178000e+00, -2.506000e+00, 3.957900e+00, -2.538400e+00,
        4.003400e+00, -2.538400e+00, 4.174200e+00, -2.567400e+00,
        4.325700e+00, -2.651400e+00, 4.440800e+00, -2.780800e+00,
        4.506600e+00, -2.941000e+00, 4.515700e+00, -3.113900e+00,
        4.467000e+00, -3.280100e+00, 4.406400e+00, -3.404300e+00,
        4.406200e+00, -3.404800e+00, 4.377300e+00, -3.479900e+00,
        4.311500e+00, -3.614800e+00, 4.245700e+00, -3.749700e+00,
        4.179900e+00, -3.884700e+00, 4.491800e+00, -3.909200e+00,
        4.568700e+00, -3.827600e+00, 4.645600e+00, -3.746000e+00,
        4.821500e+00, -3.709000e+00, 4.998200e+00, -3.741400e+00,
        5.149600e+00, -3.838200e+00, 5.252900e+00, -3.985200e+00,
        5.262800e+00, -4.124400e+00, 5.272700e+00, -4.263600e+00,
        5.565000e+00, -4.555900e+00, 5.565000e+00, -4.402600e+00,
        5.565000e+00, -4.249400e+00, 5.565000e+00, -4.096200e+00,
        5.572000e+00, -3.996100e+00, 5.572000e+00, -3.995600e+00,
        5.572000e+00, -3.896000e+00, 5.572000e+00, -3.895000e+00,
        5.600800e+00, -3.702400e+00, 5.684200e+00, -3.526500e+00,
        5.815000e+00, -3.382300e+00, 5.982100e+00, -3.282200e+00,
        6.171000e+00, -3.235000e+00, 6.365500e+00, -3.244600e+00}; // r and z coordinates of limiter grid, value gotten from tds-gs/data/seperated_file.data   

    // Extract values from files
    int nx, ny; 
    float rdim, zdim, rleft, zmid, simagx, sibdry, cpasma, rmagx, zmagx; 
    
    ifstream file1("GEQDSK/GEQDSK_nx_ny_rdim_zdim_rleft_zmid.txt");
    file1 >> nx >> ny >> rdim >> zdim >> rleft >> zmid; 
    file1.close();

    ifstream file2("GEQDSK/GEQDSK_simagx_sibdry_cpasma.txt");
    file2 >> simagx >> sibdry >> cpasma; 
    file2.close();

    //Extract Boundary Points to Find nbdry

    Mesh my_mesh("../tds-gs/meshes/mesh_refine.mesh");
    my_mesh.EnsureNodes();
    FiniteElementCollection *fec = new H1_FECollection(1, my_mesh.Dimension()); // P1 elements
    FiniteElementSpace fespace(&my_mesh, fec);

    // Create GridFunction
    ifstream ifs("../tds-gs/gf/final_model2_pc5_cyc1_it5.gf");
    GridFunction lgf(&my_mesh, ifs);

    // Now call your function
    vector<double> rbdry_zbdry = ExtractContourLine(my_mesh, lgf, sibdry);

    //Find and print nbdry
    int nbdry = rbdry_zbdry.size() / 2;  
    printf("nbdry: %i\n", nbdry); 

    ofstream file3("GEQDSK/GEQDSK_nbdry_nlim.txt");
    file3 << nbdry << "    " << nlim << '\n';
    file3.close();

    
    find_rmagx_zmagx("interpolated.gf", "my_new.mesh", simagx, rmagx, zmagx);
    printf("rmagx: = %.5f\n", rmagx);
    printf("zmagx: %.5f\n", zmagx); 
    
    // Generate needed values
    vector<double> psi = psiSort();
    vector<double> fpol = fpolCalc(alpha, psi_x, f_x, psi);
    ffprimeCalc(alpha, fpol);
    fpolFormat(fpol);
    psiFormat(psi);
    generate_pres();
    generate_pprime();

    // Build final GEQDSK.txt file
    ofstream clear_file("GEQDSK/GEQDSK.txt", ios::trunc); // clears the file
    clear_file.close();
    
    GEQDSK_header(nx, ny);
    append_section1(rdim, zdim, rcentr, rleft, zmid, rmagx, zmagx, simagx, sibdry, bcentr, cpasma);
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_fpol.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_pres.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_ffprime.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_pprime.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_psi.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_nbdry_nlim.txt");

    // append_to_GEQDSK_txt("GEQDSK/GEQDSK_qpsi.txt");
    // generate a correct file format for nbdry and nlim
    // append_to_GEQDSK_txt("GEQDSK/GEQDSK_nbdry_nlim.txt");
    // append_to_GEQDSK_txt("GEQDSK/GEQDSK_qpsi.txt"); 
    
    int count = append_rbdry_zbdry(rbdry_zbdry);
    append_rlim_zlim(count, rlim_zlim );

    return 0; 
}

