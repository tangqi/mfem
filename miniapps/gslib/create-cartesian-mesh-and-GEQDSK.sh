# File             : create-cartesian-mesh-and-GEQDSK.sh
# Purpose          : Automatically run tds-gs/run_3_taylor.sh, 2d-cartesian-mesh.cpp, and GEQDSK-generation.cpp to generate a GEQDSK.txt file for a specified quadrilateral mesh.   
# Run Instructions : ./create-cartesian-mesh-and-GEQDSK.sh
# Improvements     : Can add user input for nx, ny, and r and z dimensions for cartesian grid
#                    Can add functionality to enable other run cases instead of run_3_taylor.sh
# Note             : *Takes a couple minutes to run completely

set -e
start_time=$(date +%s)

source ~/.bashrc

# Step 0: Go to tds-gs directory and run run_3_taylor.sh
echo "Entering tds-gs and running run_3_taylor.sh..."
pushd ../tds-gs > /dev/null
bash run_3_taylor.sh > /dev/null
popd > /dev/null

# Step 1: Build 2d-cartesian-mesh
echo "Compiling 2d-cartesian-mesh.cpp..."
make 2d-cartesian-mesh > /dev/null

# Step 2: Run 2d-cartesian-mesh
echo "Running 2d-cartesian-mesh..."
srun -n 1 ./2d-cartesian-mesh > /dev/null

# Step 3: Build GEQDSK-q-generation
echo "Compiling GEQDSK-q-generation.cpp..."
make GEQDSK-q-generation > /dev/null

# Step 5: Run GEQDSK-q-generation
echo "Running GEQDSK-q-generation..."
./GEQDSK-q-generation

# Step 4: Build GEQDSK-generation
echo "Compiling GEQDSK-generation.cpp..."
make GEQDSK-generation > /dev/null

# Step 5: Run GEQDSK-generation
echo "Running GEQDSK-generation..."
./GEQDSK-generation


end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Total time to run script: $elapsed seconds"