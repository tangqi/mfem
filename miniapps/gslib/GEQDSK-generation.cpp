// File: GEQDSK-generation.cpp 
// Purpose: Generates the final GEQDSK plasma file by finding the individual variable files and appending them
// Workflow: The respective MFEM files generate "GEQDSK_[variable_names]" files that store the values of the needed variables. GEQDDSK-generation.cpp extracts the values from those files, generates other needed variables, and compiles the final GEQDSK.txt file. 
// File un instructions: Run a triangular mesh file (Ex: sh run_3_taylor.sh)
//                       Run 2D-cartesian-mesh.cpp (make clean && make 2d-cartesian-mesh && srun -n 1 ./2d-cartesian-mesh) 
// Note: Running 2d-cartesian-mesh.cpp automatically runs GEQDSK-generation.cpp. if want to run independently, execute: make GEQDSK-generation && ./GEQDSK-generation 

#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <cstdlib>
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


int main(){
    // Known constants
    float rcentr = +6.200000286e+00; // [meter] Reference value of R 
    float bcentr = -5.300000000e+00; // [tesla] Vacuum toroidal magnetic field at rcentr
     
    // Extract values from files
    int nx, ny; 
    float rdim, zdim, rleft, zmid, simagx, sibdry, cpasma, rmagx, zmagx; 
    
    ifstream file1("GEQDSK/GEQDSK_nx_ny_rdim_zdim_rleft_zmid.txt");
    file1 >> nx >> ny >> rdim >> zdim >> rleft >> zmid; 
    file1.close();

    ifstream file2("GEQDSK/GEQDSK_simagx_sibdry_cpasma.txt");
    file2 >> simagx >> sibdry >> cpasma; 
    file2.close();
    
    find_rmagx_zmagx("interpolated.gf", "my_new.mesh", simagx, rmagx, zmagx);
    printf("rmagx: = %.5f\n", rmagx);
    printf("zmagx: %.5f\n", zmagx); 
    
    // Build final GEQDSK.txt file
    GEQDSK_header(nx, ny);
    append_section1(rdim, zdim, rcentr, rleft, zmid, rmagx, zmagx, simagx, sibdry, bcentr, cpasma);

    return 0; 
}

