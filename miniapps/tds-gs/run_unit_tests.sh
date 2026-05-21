# Runs unit tests in test.cpp for our Grad-Shafranov solver.

# Set arguments in main to perform unit tests. All other parameters are set to their default settings.
do_test=1  # When do_test == 1, main.cpp is set to run unit tests as opposed to running the GS solver.
do_initial=0

srun -n 1 ./main \
    -t $do_test \
    --initial $do_initial
