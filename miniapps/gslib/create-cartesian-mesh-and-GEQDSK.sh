#!/bin/bash
# File: create-cartesian-mesh-and-GEQDSK.sh

set -e  # optional: stop immediately on any error

# Load environment
source ~/.bashrc

# Save the original directory
ORIG_DIR=$(pwd)

# Step 0: Go to tds-gs directory and run run_3_taylor.sh
echo "Entering tds-gs and running run_3_taylor.sh..."
cd ../tds-gs
bash run_3_taylor.sh
cd "$ORIG_DIR"   # Return to original dir

# Step 1: Build 2d-cartesian-mesh
echo "Compiling 2d-cartesian-mesh.cpp..."
make 2d-cartesian-mesh

# Step 2: Run 2d-cartesian-mesh
echo "Running 2d-cartesian-mesh..."
./2d-cartesian-mesh

# Step 3: Build GEQDSK-generation
echo "Compiling GEQDSK-generation.cpp..."
make GEQDSK-generation

# Step 4: Run GEQDSK-generation
echo "Running GEQDSK-generation..."
./GEQDSK-generation

echo "✅ All steps completed successfully."
