// To Do:
//        1. Generate & append qpsi

// File                  : GEQDSK-generation.cpp 
// Purpose               : Generates the final GEQDSK plasma file for ITER tokamak by finding/generating the individual parameter files and appending them
// Workflow              : The respective MFEM files generate "GEQDSK_[variable_names]" files that store the values of the needed variables. GEQDDSK-generation.cpp extracts the values from those files, generates other needed variables, and compiles the final GEQDSK.txt file. 
// File run instructions : Run a triangular mesh file (Ex: sh run_3_taylor.sh)
//                         Run 2D-cartesian-mesh.cpp (make 2d-cartesian-mesh && srun -n 1 ./2d-cartesian-mesh) 
//                         Run GEQDSK-generation.cpp (make GEQDSK-generation && ./GEQDSK-generation) 
// Reference             : https://github.com/Sruthifeb14/OSPO_VSIP_2025_MFEM_Internship                             


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
#include <sys/stat.h>
#include <ctype.h>
using namespace mfem;
using namespace std;
  
// Import file and store values in a vector 
vector<double> fileToVector(const string &filename){
    vector<double> vec;
    string line;
    ifstream ReadFile(filename);
    int line_count = 0;

    double value;
    while (getline (ReadFile, line)){
        line_count++; 
        if (line.empty()) {
            continue;
        }
        if (!isdigit(line[0]) && (line[0] != '-')){
            continue;
        }
        if (line[0] == '-' && (line.length() < 2 || !isdigit(line[1]))){
            continue;
        }
        value = stod(line);
        vec.push_back(value);
    }
    ReadFile.close();
    return vec;
}

// Returns a linearly spaced vector given min and max values
vector<double> linspace(double min, double max, int n) {
    vector<double> result;
    if (n <= 1){
        result.push_back(min);
        return result;
    }
    double step = (max - min) / (n - 1);
    for (int i = 0; i < n; ++i){
        result.push_back(min + i * step);
    }
    return result;
}

// Generates a "contour_line_" file with the contour line points of a given psi value, provided its mesh and solution files
void extract_contour_line(const Mesh &mesh, const GridFunction &u, double level,
                        const string &filename){
    MFEM_VERIFY(mesh.Dimension() == 2, "Only 2D meshes are supported.");
    MFEM_VERIFY(u.FESpace()->GetVDim() == 1, "Only scalar fields are supported.");
    const FiniteElementSpace &fes = *u.FESpace();
    const GridFunction *nodes = mesh.GetNodes();
    MFEM_VERIFY(nodes, "Mesh must be in nodal form (use high-order mesh).");
 
    ofstream NewFile(filename);
    const int nedges = mesh.GetNEdges();
    const int dim = 2;
    for (int e = 0; e < nedges; ++e){
        Array<int> ev;
        mesh.GetEdgeVertices(e, ev);
        const double *coords_i = mesh.GetVertex(ev[0]);
        const double *coords_j = mesh.GetVertex(ev[1]);
        Array<int> dofs_i, dofs_j;
        fes.GetVertexDofs(ev[0], dofs_i);
        fes.GetVertexDofs(ev[1], dofs_j);
        double ui = u(dofs_i[0]);
        double uj = u(dofs_j[0]);
        if ((ui - level) * (uj - level) < 0.0){
            double alpha = (level - ui) / (uj - ui);
            Vector pt(dim);
            for (int d = 0; d < dim; d++){
                pt[d] = coords_i[d] + alpha * (coords_j[d] - coords_i[d]);
            }
            if (pt[1] >= -3.56733) {
                NewFile << pt[0] << " " << pt[1] << "\n";
            }
        }
    }
    NewFile.close();
}

// Reads a "contour_line_" file and stores its coordinates in a vector of (x, y) pairs
vector<pair<double,double>> read_data(const string &filename) {
    vector<pair<double,double>> data;
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return data;
    }
    string line;
    while (getline(infile, line)) {
        istringstream iss(line);
        double x, y;
        if (!(iss >> x >> y)) {
            cerr << "Skipping malformed line: " << line << endl;
            continue;
        }
        data.emplace_back(x, y);
    }
    return data;
}

// Returns the radial points given a contour, magnetic axis coordinates, and the desired angles at which the radial points should be found
vector<pair<double,double>> extract_radial_points(double rmagx, double zmagx,
                                     const vector<pair<double,double>> &points,
                                     const vector<double> &angles_deg) {
    vector<pair<double,double>> selected_points;
    for (double angle_deg : angles_deg) {
        double angle_rad = angle_deg * M_PI / 180.0;
        double dx = cos(angle_rad);
        double dy = sin(angle_rad);

        double best_perp_dist = numeric_limits<double>::infinity();
        pair<double,double> best_point = {numeric_limits<double>::quiet_NaN(), 
                                          numeric_limits<double>::quiet_NaN()};

        for (const auto &pt : points) {
            double vec_x = pt.first - rmagx;
            double vec_y = pt.second - zmagx;

            // Projection of vector onto direction vector
            double proj_len = vec_x * dx + vec_y * dy;
            if (proj_len <= 0.0) continue; 

            // Compute perpendicular distance to the ray
            double proj_x = proj_len * dx;
            double proj_y = proj_len * dy;
            double perp_x = vec_x - proj_x;
            double perp_y = vec_y - proj_y;
            double perp_dist = sqrt(perp_x * perp_x + perp_y * perp_y);

            if (perp_dist < best_perp_dist) {
                best_perp_dist = perp_dist;
                best_point = pt;
            }
        }
        selected_points.push_back(best_point);
    }
    return selected_points;
}

// Generates a range of psi values and for each psi, generates "contour_line_" files for each psi, extracts radial points from each "contour_line_", and writes those points to "all_contour_radial_points.txt" 
void ProcessContours(double sibdry, double simagx,
                     double rmagx, double zmagx, int n, const string& mesh_path, const string& gf_path, 
                     const vector<double>& angles_deg, const string& output_dir = "Contours") {
    // Generate psi values
    vector<double> values = linspace(sibdry, simagx, n);

    // Load mesh and grid function
    Mesh my_mesh(mesh_path.c_str());
    my_mesh.EnsureNodes();

    FiniteElementCollection* fec = new H1_FECollection(1, my_mesh.Dimension());
    FiniteElementSpace fespace(&my_mesh, fec);

    ifstream ifs(gf_path);
    GridFunction lgf(&my_mesh, ifs);

    // Ensure output folder exists
    string mkdir_cmd = "mkdir -p " + output_dir;
    std::system(mkdir_cmd.c_str());

    // Extract contour lines and write to files
    vector<string> filenames;
    for (double val : values) {
        ostringstream filename;
        filename << output_dir << "/contour_line_" << fixed << setprecision(5) << val << ".txt";
        filenames.push_back(filename.str());
        extract_contour_line(my_mesh, lgf, val, filename.str());
    }

    // Extract radial points from contours
    vector<vector<pair<double, double>>> all_radial_pts;
    for (const auto& file : filenames) {
        vector<pair<double, double>> contour = read_data(file);
        vector<pair<double, double>> radial_pts = extract_radial_points(rmagx, zmagx, contour, angles_deg);
        all_radial_pts.push_back(radial_pts);
    }

    // Write all radial points to a single output file
    ofstream out(output_dir + "/all_contour_radial_points.txt");
    for (size_t j = 0; j < filenames.size(); ++j) {
        string base = filenames[j];
        size_t start = base.find("line_");
        size_t end = base.find(".txt");
        string id = (start != string::npos && end != string::npos && end > start + 5)
                        ? base.substr(start + 5, end - (start + 5))
                        : to_string(j);

        out << "r_" << id << "\t" << "z_" << id;
        if (j != filenames.size() - 1) out << "\t";
    }
    out << "\n";

    out << fixed << setprecision(6);
    for (size_t i = 0; i < angles_deg.size(); ++i) {
        for (size_t j = 0; j < all_radial_pts.size(); ++j) {
            out << all_radial_pts[j][i].first << "\t" << all_radial_pts[j][i].second;
            if (j != all_radial_pts.size() - 1) out << "\t";
        }
        out << "\n";
    }
    out.close();
    delete fec;
}

// GEQDSK folder and file setup
void GEQDSK_header(int nx, int ny){     
    // Create folder for GEQDSK files and final output (if it doesn't exist already)
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

// Calculate fpol values
vector<double> fpol_calc(double alpha, double psi_x, double f_x, const vector<double> psiVal){
    vector<double> fpol;
    double fpol_val;

    for (const auto& val : psiVal) {
        fpol_val = f_x + alpha * (val - psi_x);
        fpol.push_back(fpol_val);
    }
    return fpol;
}

// Generate GEQDSK_fpol.txt
void fpol_format(const vector<double> fpol){
    ofstream file("GEQDSK/GEQDSK_fpol.txt");
    file << uppercase << scientific << setprecision(9);

    int val_count = 0;
    for (const auto& val : fpol) {
        // Convert to scientific notation
        file << setw(16) << val;

        val_count++;
        if (val_count >= 5){
            file << endl;
            val_count = 0;
        }
    }
    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0){
        file << "\n";
    }
    file.close();
}

// Compute ffprime and generate GEQDSK_ffprime.txt
void ffprime_calc(double alpha, const vector<double>fpol){
    vector<double> ffprime;
    double ffprime_val;

    for (const auto& val : fpol){
        ffprime_val = alpha * val;
        ffprime.push_back(ffprime_val);
    }

    ofstream file("GEQDSK/GEQDSK_ffprime.txt");
    file << uppercase << scientific << setprecision(9);

    int val_count = 0;
    for (const auto& val : ffprime){
        // Convert to scientific notation
        file << setw(16) << val;
        val_count++;
        if (val_count >= 5){
            file << endl;
            val_count = 0;
        }
    }
    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0){
        file << endl;
    }
    file.close();
}

// Generate GEQDSK_psi.txt
void psiFormat(const vector<double> psiVal){
    ofstream file("GEQDSK/GEQDSK_psi.txt");
    file << uppercase << scientific << setprecision(9);
    
    int val_count = 0;
    for (const auto& val : psiVal){
        //Convert to scientific notation
        file << setw(16) << val;

        val_count++;
        if (val_count >= 5) {
            file << endl;
            val_count = 0;
        }
    }
    // Ensure the file ends with a newline (even if val_count == 0)
    if (val_count != 0) {
        file << "\n";
    }
    
    file.close();
}

// Generate GEQDSK_pres.txt
void generate_pres(int nx){
    ifstream infile("GEQDSK/GEQDSK_fpol.txt");
    ofstream outfile("GEQDSK/GEQDSK_pres.txt");

    int count = 0;

    // Read value by value (assuming file only has Fortran-formatted values)
    while (count < nx) {
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

// Find simagx coordinates: rmagx, zmagx
void find_rmagx_zmagx(const string& rect_gf_file, const string& rect_mesh_file, float simagx, float &rmagx, float &zmagx){
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

    file_solution.close(); 

    // Use closest_psi_idx to get rmagx and zmagx
    ifstream file_mesh(rect_mesh_file);
    if (!file_mesh.is_open()){
        cerr << "File can't be opened: " << rect_mesh_file << "\n" << endl;   
    } 

    string line1; 
    int counter1 = 0;
    bool found_title = false;

    while (getline(file_mesh, line1)){
        if (line1 == "vertices"){
            getline(file_mesh, line1); // Skip # of vertices line
            getline(file_mesh, line1); // Skip dimension line
            found_title = true; 
            counter1 = 0;
        }

        if (found_title == true && counter1 == closest_psi_idx){
            istringstream iss(line1);
            iss >> rmagx >> zmagx;            
            break;   
        }
        counter1 = counter1 + 1;
    }
    file_mesh.close(); 
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

    file << setw(16) << rdim << setw(16) << zdim << setw(16) << rcentr
        << setw(16) << rleft << setw(16) << zmid << endl;
    file << setw(16) << rmagx << setw(16) << zmagx << setw(16) << simagx
        << setw(16) << sibdry << setw(16) << bcentr << endl;
    file << setw(16) << cpasma << setw(16) << simagx << setw(16) << 0.0
        << setw(16) << rmagx << setw(16) << 0.0 << endl;
    file << setw(16) << zmagx << setw(16) << 0.0 << setw(16) << sibdry
        << setw(16) << 0.0 << setw(16) << 0.0 << endl;

    file.close();
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

// Format and append rbdry & zbdry to GEQDSK.txt
int append_rbdry_zbdry(const vector<double>& rbdry_zbdry) {
    int val_count = 0;

    ofstream outfile("GEQDSK/GEQDSK.txt", ios::app);
    if (!outfile.is_open()) {
        cerr << "Could not open GEQDSK/GEQDSK.txt" << endl;
        return val_count;
    }

    outfile << uppercase << scientific << setprecision(9);
    outfile << '\n';

    for (double val : rbdry_zbdry) {
        outfile << setw(16) << val;

        val_count++;
        if (val_count >= 5) {
            outfile << endl;
            val_count = 0;
        }
    }
    outfile.close();
    return val_count;
}

// Format and append rlim and zlim to GEQDSK.txt
void append_rlim_zlim(const int& num, const vector<double>& rlim_zlim) {
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

// Generate GEQDSK_rlim_zlim.txt
vector<double> generate_rlim_zlim(){
    vector<double> rlim_zlim; 
    ifstream infile("../tds-gs/data/separated_file.data");

    if (!infile.is_open()) {
        cerr << "Error: Could not open separated_file.dat." << endl;
        return rlim_zlim;
    }

    bool in_rlim_zlim = false;
    string line;
    while (getline(infile, line)) {
        if (in_rlim_zlim == false) {
            if (line == "# rlim(i),zlim(i)") {
                in_rlim_zlim = true;
            }
        } else {
            istringstream iss(line);
            double val;
            while (iss >> val) {
                if (val == 0) {
                    continue;
                }
                rlim_zlim.push_back(val);
            }
        }        
    }

    ofstream outFile("GEQDSK/GEQDSK_rlim_zlim.txt");
    outFile << uppercase << scientific << setprecision(9);

    for (const auto &e : rlim_zlim) {
        outFile << setw(16) << e << "\n";
    }

    outFile.close();
    return rlim_zlim;
}


int main(){
    // Delete previous files in "Contours" folder
    struct stat info;
    if (stat("Contours", &info) == 0 && (info.st_mode & S_IFDIR)) {
        std::system("rm -f Contours/*");
    }
    
    // Known constants
    float rcentr = 6.200000286e+00; // [meter] Reference value of R 
    float bcentr = -5.300000000e+00; // [tesla] Vacuum toroidal magnetic field at rcentr
    int nlim = 56; // Number of points in the limiter grid, value gotten from tds-gs/data/seperated_file.data

    // Extract values from files
    int nx, ny; 
    float rdim, zdim, rleft, zmid, simagx, sibdry, cpasma, rmagx, zmagx;
    double alpha, psi_x, f_x;
    
    ifstream file1("GEQDSK/GEQDSK_nx_ny_rdim_zdim_rleft_zmid.txt");
    file1 >> nx >> ny >> rdim >> zdim >> rleft >> zmid; 
    file1.close();

    ifstream file2("GEQDSK/GEQDSK_simagx_sibdry_cpasma.txt");
    file2 >> simagx >> sibdry >> cpasma; 
    file2.close();

    ifstream file3("GEQDSK/GEQDSK_alpha_f_x_psi_x.txt");
    file3 >> alpha >> f_x >> psi_x;
    file3.close();

    // Define element space and grid function to find nbdry
    Mesh my_mesh("../tds-gs/meshes/mesh_refine.mesh");
    my_mesh.EnsureNodes();
    FiniteElementCollection *fec = new H1_FECollection(1, my_mesh.Dimension()); // P1 elements
    FiniteElementSpace fespace(&my_mesh, fec);

    ifstream ifs("../tds-gs/gf/final_model2_pc5_cyc1_it5.gf");
    GridFunction lgf(&my_mesh, ifs);

    // Generate boundary values: rbdry and zbdry
    extract_contour_line(my_mesh, lgf, sibdry, "GEQDSK_rbdry_zbdry.txt");
    vector<double> rbdry_zbdry = fileToVector("GEQDSK_rbdry_zbdry.txt");

    // Generate GEQDSK_nbdry_nlim.txt 
    int nbdry = rbdry_zbdry.size() / 2;  

    ofstream file4("GEQDSK/GEQDSK_nbdry_nlim.txt");
    file4 << nbdry << "    " << nlim << '\n';
    file4.close();
    
    // Generate needed values
    find_rmagx_zmagx("interpolated.gf", "my_new.mesh", simagx, rmagx, zmagx);
    vector<double> psi = fileToVector("interpolated.gf");
    vector<double> equalSpacePsi = linspace(simagx, sibdry, nx);
    vector<double> fpol = fpol_calc(alpha, psi_x, f_x, equalSpacePsi);
    ffprime_calc(alpha, fpol);
    fpol_format(fpol);
    psiFormat(psi);
    generate_pres(nx);
    generate_pprime();

    vector<double> rlim_zlim = generate_rlim_zlim();
    int count = append_rbdry_zbdry(rbdry_zbdry);
    
    // qpsi calcs ----------------------------------------------------------------------------
    
    // Find radially aligned points for nx number of contours
    vector<double> angles_deg = {0, 45, 90, 135, 180, 225, 270, 315};
    ProcessContours(sibdry, simagx, rmagx, zmagx, nx+1,
                    "../tds-gs/meshes/mesh_refine.mesh",
                    "../tds-gs/gf/final_model2_pc5_cyc1_it5.gf", angles_deg);


    // Build final GEQDSK.txt file ----------------------------------------------------------------
    
    // Clear the existing files
    ofstream clear_file("GEQDSK/GEQDSK.txt", ios::trunc);
    clear_file.close();
    
    // Build GEQDSK.txt
    GEQDSK_header(nx, ny);
    append_section1(rdim, zdim, rcentr, rleft, zmid, rmagx, zmagx, simagx, sibdry, bcentr, cpasma);
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_fpol.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_pres.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_ffprime.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_pprime.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_psi.txt");
    // append_to_GEQDSK_txt("GEQDSK/GEQDSK_qpsi.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_nbdry_nlim.txt");
    append_rlim_zlim(count, rlim_zlim);

    return 0; 
}

