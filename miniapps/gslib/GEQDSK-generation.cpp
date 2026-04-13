// File                  : GEQDSK-generation.cpp 
// Purpose               : Generates the final GEQDSK plasma file for ITER tokamak by finding/generating the individual parameter files and appending them
// Workflow              : The respective MFEM files generate "GEQDSK_[variable_names]" files that store the values of the needed variables. GEQDDSK-generation.cpp extracts the values from those files, generates other needed variables, and compiles the final GEQDSK.txt file. 
// File run instructions : Run a triangular mesh file (Ex: sh run_3_taylor.sh)
//                         Run 2D-cartesian-mesh.cpp (make 2d-cartesian-mesh && srun -n 1 ./2d-cartesian-mesh) 
//                         Run GEQDSK-g-generation.cpp (make GEQDSK-q-generation && ./GEQDSK-q-generation)
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

// Reads files to store values in a vector (used for psi and nbdry_zbdry) 
vector<double> fileToVector(const string &filename) {
    vector<double> vec;
    string line;
    ifstream ReadFile(filename);
    int line_count = 0;

    if (!ReadFile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return vec;
    }

    while (getline(ReadFile, line)) {
        line_count++;
        if (line.empty()) continue;

        // Skip lines where first non-space char is not a digit or '-'
        string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t")); // remove leading spaces
        if (!isdigit(trimmed[0]) && (trimmed[0] != '-')) continue;
        if (trimmed[0] == '-' && (trimmed.length() < 2 || !isdigit(trimmed[1]))) continue;

        stringstream ss(trimmed);
        double value;
        while (ss >> value) {  // handles 1, 2, or more numbers per line
            vec.push_back(value);
        }
    }

    ReadFile.close();
    return vec;
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
vector<pair<double,double>> readContourLineFile(const string &filename) {
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
void generateAllContourRadialPoints(double sibdry, double simagx,
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
        vector<pair<double, double>> contour = readContourLineFile(file);
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

// Generate GEQDSK_Section_1
void generate_section_1(float rdim, float zdim, float rcentr, float rleft, float zmid, float rmagx, float zmagx, float simagx, float sibdry, float bcentr, float cpasma){
    ofstream file("GEQDSK/GEQDSK_section_1.txt");

    // Set scientific format, width = 16, precision = 9
    file << uppercase << scientific << setprecision(9);

    file << setw(16) << rdim << setw(16) << zdim << setw(16) << rcentr
        << setw(16) << rleft << setw(16) << zmid << endl;
    file << setw(16) << rmagx << setw(16) << zmagx << setw(16) << simagx
        << setw(16) << sibdry << setw(16) << bcentr << endl;
    file << setw(16) << cpasma << setw(16) << simagx << setw(16) << 0.0
        << setw(16) << rmagx << setw(16) << 0.0 << endl;
    file << setw(16) << zmagx << setw(16) << 0.0 << setw(16) << sibdry
        << setw(16) << 0.0 << setw(16) << 0.0;

    file.close();
}

// Compute & generate GEQDSK_fpol.txt
vector<double> generate_fpol(double alpha, double psi_x, double f_x, const vector<double>& psiVal) {
    vector<double> fpol;
    fpol.reserve(psiVal.size());

    for (const auto& val : psiVal) {
        fpol.push_back(f_x + alpha * (val - psi_x));
    }

    ofstream file("GEQDSK/GEQDSK_fpol.txt");
    if (!file.is_open()) {
        cerr << "Error opening file for writing: GEQDSK/GEQDSK_fpol.txt\n";
        // Still return the vector even if writing failed
        return fpol;
    }

    file << uppercase << scientific << setprecision(9);

    int val_count = 0;
    size_t total = fpol.size();
    for (size_t i = 0; i < total; ++i) {
        file << setw(16) << fpol[i];
        val_count++;
        if (val_count == 5 && i != total - 1) {
            file << '\n';
            val_count = 0;
        }
    }
    file.close();
    return fpol;
}

// Generate GEQDSK_pres.txt
void generate_pres(int nx){
    ifstream infile("GEQDSK/GEQDSK_fpol.txt");
    ofstream outfile("GEQDSK/GEQDSK_pres.txt");

    int count = 0;

    while (count < nx) {
        outfile << setw(16) << setprecision(9)
                << uppercase << scientific << 0.0;
        count++;

        // Newline every 5 values, except after last
        if (count % 5 == 0 && count != nx) {
            outfile << '\n';
        }
    }
    infile.close();
    outfile.close();
}

// Compute & generate GEQDSK_ffprime.txt
void generate_ffprime(double alpha, const vector<double>& fpol){
    vector<double> ffprime;
    double ffprime_val;

    for (const auto& val : fpol){
        ffprime_val = alpha * val;
        ffprime.push_back(ffprime_val);
    }

    ofstream file("GEQDSK/GEQDSK_ffprime.txt");
    file << uppercase << scientific << setprecision(9);

    int val_count = 0;
    size_t total = ffprime.size();
    for (size_t i = 0; i < total; ++i){
        file << setw(16) << ffprime[i];
        val_count++;

        // Add newline every 5 values except after last one
        if (val_count == 5 && i != total - 1){
            file << endl;
            val_count = 0;
        }
    }
    file.close();
}

// Generate GEQDSK_pprime.txt
void generate_pprime(){
    ifstream infile("GEQDSK/GEQDSK_pres.txt");
    ofstream outfile("GEQDSK/GEQDSK_pprime.txt");

    outfile << infile.rdbuf();

    infile.close();
    outfile.close(); 
}

// Generate GEQDSK_psi.txt
void generate_psi(const vector<double>& psiVal){
    ofstream file("GEQDSK/GEQDSK_psi.txt");
    file << uppercase << scientific << setprecision(9);

    int val_count = 0;
    size_t total = psiVal.size();
    for (size_t i = 0; i < total; ++i){
        file << setw(16) << psiVal[i];
        val_count++;

        // Add newline every 5 values, except after the last value
        if (val_count == 5 && i != total - 1) {
            file << endl;
            val_count = 0;
        }
    }
    file.close();
}

// Generate GEQDSK_nbdry_nlim.txt
void generate_nbdry_nlim(int nbdry, int nlim) {
    ofstream file4("GEQDSK/GEQDSK_nbdry_nlim.txt");
    file4 << setw(5) << nbdry << setw(5) << nlim;
    file4.close();
}

// Generates GEQDSK_rbdry_zbdry.txt
int generate_rbdry_zbdry(const std::string& mesh_path, const std::string& gf_path, double sibdry) {
    // Load mesh and setup FE space
    Mesh my_mesh(mesh_path.c_str());
    my_mesh.EnsureNodes();

    FiniteElementCollection* fec = new H1_FECollection(1, my_mesh.Dimension()); // P1 elements
    FiniteElementSpace fespace(&my_mesh, fec);

    ifstream ifs(gf_path);
    if (!ifs.is_open()) {
        cerr << "Error opening GF file: " << gf_path << endl;
        delete fec;
        return 0;
    }
    GridFunction lgf(&my_mesh, ifs);

    // Extract contour line to unformatted file
    const std::string unformatted_file = "GEQDSK/GEQDSK_rbdry_zbdry_unformatted.txt";
    extract_contour_line(my_mesh, lgf, sibdry, unformatted_file);

    // Read unformatted data
    std::vector<double> rbdry_zbdry = fileToVector(unformatted_file);

    // Write formatted data directly here:
    const std::string formatted_file = "GEQDSK/GEQDSK_rbdry_zbdry.txt";
    std::ofstream outFile(formatted_file);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open " << formatted_file << " for writing.\n";
        delete fec;
        return 0;
    }

    outFile << std::uppercase << std::scientific << std::setprecision(9);

    int count = 0;
    for (double val : rbdry_zbdry) {
        outFile << std::setw(16) << val;
        if (++count % 5 == 0) outFile << '\n';
    }
    outFile.close();

    // Delete unformatted file
    remove(unformatted_file.c_str());

    delete fec;
    return static_cast<int>(rbdry_zbdry.size() / 2);
}

// Generate GEQDSK_rlim_zlim.txt
int generate_rlim_zlim(){
    vector<double> rlim_zlim; 
    ifstream infile("../tds-gs/data/separated_file.data");

    if (!infile.is_open()) {
        cerr << "Error: Could not open separated_file.dat." << endl;
    }

    bool in_rlim_zlim = false;
    string line;
    while (getline(infile, line)) {
        if (!in_rlim_zlim) {
            if (line == "# rlim(i),zlim(i)") {
                in_rlim_zlim = true;
            }
        } else {
            // Stop reading if line is empty or looks like end of section
            if (line.empty() || line[0] == '#') break;  

            istringstream iss(line);
            double val;
            while (iss >> val) {
                if (val == 0) continue;  // your logic to skip zeros
                rlim_zlim.push_back(val);
            }
        }        
    }

    infile.close();

    ofstream outFile("GEQDSK/GEQDSK_rlim_zlim.txt");
    if (!outFile.is_open()) {
        cerr << "Error opening output file\n";
    }

    outFile << uppercase << scientific << setprecision(9);

    int count = 0;
    for (const auto &e : rlim_zlim) {
        outFile << setw(16) << e;
        count++;
        if (count % 5 == 0) {
            outFile << '\n';
        }
    }
    outFile.close();
    return (rlim_zlim.size())/2;
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
         << setw(4) << ny;
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


int main(){
    // Delete Previous Files ----------------------------------------------------------------    
    // Delete previous files in "Contours" folder
    struct stat info;
    if (stat("Contours", &info) == 0 && (info.st_mode & S_IFDIR)) {
        std::system("rm -f Contours/*");
    }

    // Gather & Generate Needed Parameters ----------------------------------------------------------------
    // Known constants
    float rcentr = 6.200000286e+00; // [meter] Reference value of R 
    float bcentr = -5.300000000e+00; // [tesla] Vacuum toroidal magnetic field at rcentr

    // Extract values from files
    int nx, ny; 
    float rdim, zdim, rleft, zmid, simagx, sibdry, cpasma, rmagx, zmagx;
    double alpha, psi_x, f_x;
    
    ifstream file1("GEQDSK/GEQDSK_nx_ny_rdim_zdim_rleft_zmid.txt");
    file1 >> nx >> ny >> rdim >> zdim >> rleft >> zmid; 
    file1.close();
    // remove("GEQDSK/GEQDSK_nx_ny_rdim_zdim_rleft_zmid.txt");

    ifstream file2("GEQDSK/GEQDSK_simagx_sibdry_cpasma.txt");
    file2 >> simagx >> sibdry >> cpasma; 
    file2.close();
    // remove("GEQDSK/GEQDSK_simagx_sibdry_cpasma.txt");

    ifstream file3("GEQDSK/GEQDSK_alpha_f_x_psi_x.txt");
    file3 >> alpha >> f_x >> psi_x;
    file3.close();
    // remove("GEQDSK/GEQDSK_alpha_f_x_psi_x.txt");
   
    // Generate needed files    
    find_rmagx_zmagx("interpolated.gf", "my_new.mesh", simagx, rmagx, zmagx);
    generate_section_1(rdim, zdim, rcentr, rleft, zmid, rmagx, zmagx, simagx, sibdry, bcentr, cpasma);

    vector<double> psi = fileToVector("interpolated.gf");
    vector<double> equalSpacePsi = linspace(simagx, sibdry, nx);
    vector<double> fpol = generate_fpol(alpha, psi_x, f_x, equalSpacePsi);
   
    generate_pres(nx);
    generate_pprime();
    generate_ffprime(alpha, fpol);
    generate_pprime();
    generate_psi(psi);
    
    int nbdry = generate_rbdry_zbdry("../tds-gs/meshes/mesh_refine.mesh", "../tds-gs/gf/final_model2_pc5_cyc1_it5.gf", sibdry);
    int nlim = generate_rlim_zlim();
    generate_nbdry_nlim(nbdry, nlim); 
    
    // qpsi calcs ----------------------------------------------------------------------------
    // Find radially aligned points for nx number of contours
    vector<double> angles_deg = {0, 45, 90, 135, 180, 225, 270, 315};
    generateAllContourRadialPoints(sibdry, simagx, rmagx, zmagx, nx+1,
                    "../tds-gs/meshes/mesh_refine.mesh",
                    "../tds-gs/gf/final_model2_pc5_cyc1_it5.gf", angles_deg);

    // Build final GEQDSK.txt file ----------------------------------------------------------------
    GEQDSK_header(nx, ny);
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_section_1.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_fpol.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_pres.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_ffprime.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_pprime.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_psi.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_qpsi.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_nbdry_nlim.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_rbdry_zbdry.txt");
    append_to_GEQDSK_txt("GEQDSK/GEQDSK_rlim_zlim.txt");

    return 0; 
}

