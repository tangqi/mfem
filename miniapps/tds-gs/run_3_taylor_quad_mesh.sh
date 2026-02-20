# coefficient of ff' term
alpha=1.46579418e-01  # This is the tri-mesh solve value
# alpha=-.2

# coefficient of p' term
beta=0.0

# unused?
gamma=0.0

# model
# 1: ff' defined from fpol data
# 2: Taylor equilibrium
# 3: ff' defined from ff' data
model=2

# plasma current
Ip=1.5e+7

R0=2.4
rho_gamma=16
mu=12.5663706144e-7

# mesh_file="meshes/iter_gen.msh"
mesh_file="meshes/iter_gen_quad.msh"
data_file="data/separated_file.data"

refinement_factor=1
amr_frac_in=0.08
amr_frac_out=0.3
max_levels=8
max_dofs=100000

do_test=0
do_manufactured_solution=0
do_initial=0

# linear solver parameters
light_tol=1e-8
max_krylov_iter=10000
krylov_tol=1e-4
max_newton_iter=1
newton_tol=1e-6

# number of control points on plasma
N_control=100
do_control=1
weight_coils=1e-12
weight_solenoids=1e-12
weight_obj=1.0
optimize_alpha=1

# objective function
# 0: sum_k (psi_k - psi_0) ^ 2
# 1: sum_k (psi_N_k - 1) ^ 2
# 2: sum_k (psi_k - psi_x) ^ 2
obj_option=1

alpha_in=1.61803398875
gamma_in=1.0
pc_option=5
amg_cycle_type=1
amg_num_sweeps_a=1
amg_num_sweeps_b=1
amg_max_iter=5

# poloidal flux coils  # These are tri-mesh solve values
c6=-1.715e+06
c7=3.871e+06
c8=3.369e+06
c9=-2.728e+04
c10=1.618e+07
c11=-2.453e+07

# center solenoids  # These are tri-mesh solve values
c1=-1.459e+07
c2=3.377e+07
c3=1.824e+07
c4=2.105e+07
c5=-3.116e+06

# # poloidal flux coils
# c6=-4.552585e+06
# c7=3.180596e+06
# c8=5.678096e+06
# c9=3.825538e+06
# c10=1.066498e+07
# c11=-2.094771e+07

# # center solenoids
# c1=1.143284e+03
# c2=-2.478694e+04
# c3=-3.022037e+04
# c4=-2.205664e+04
# c5=-2.848113e+03

ur_coeff=1.0

# ./../gslib/field-interp -m1 initial/initial_mesh_g3.mesh \
#                        -m2 $mesh_file \
#                        -s1 initial/initial_guess_g3.gf \
#                        -r $refinement_factor \
#                        -no-vis

# ./../gslib/field-interp -m1 meshes/mesh_refine.mesh \
#                        -m2 $mesh_file \
#                        -s1 gf/final_model2_pc5_cyc1_it5.gf \
#                        -r $refinement_factor \
# #                        -o 2 \
#                        -no-vis

# lldb -- main.o \
srun -n 1 ./main \
    -m $mesh_file \
    -o 1 \
    -d $data_file \
    -g $refinement_factor \
    -t $do_test \
    --model $model \
    --alpha $alpha \
    --beta $beta \
    --gamma $gamma \
    --plasma_current $Ip \
    --N_control $N_control \
    --mu $mu \
    --r_zero $R0 \
    --rho_gamma $rho_gamma \
    --do_manufactured_solution $do_manufactured_solution \
    --max_krylov_iter $max_krylov_iter \
    --max_newton_iter $max_newton_iter \
    --krylov_tol $krylov_tol \
    --newton_tol $newton_tol \
    --c1 $c1 \
    --c2 $c2 \
    --c3 $c3 \
    --c4 $c4 \
    --c5 $c5 \
    --c6 $c6 \
    --c7 $c7 \
    --c8 $c8 \
    --c9 $c9 \
    --c10 $c10 \
    --c11 $c11 \
    --ur_coeff $ur_coeff \
    --do_control $do_control \
    --weight_coils $weight_coils \
    --weight_solenoids $weight_solenoids \
    --initial $do_initial \
    --weight_obj $weight_obj \
    --obj_option $obj_option \
    --optimize_alpha $optimize_alpha \
    --pc_option $pc_option \
    --max_levels $max_levels \
    --max_dofs $max_dofs \
    --light_tol $light_tol \
    --alpha_in $alpha_in \
    --gamma_in $gamma_in \
    --amg_cycle_type $amg_cycle_type \
    --amg_num_sweeps_a $amg_num_sweeps_a \
    --amg_num_sweeps_b $amg_num_sweeps_b \
    --amg_max_iter $amg_max_iter \
    --amr_frac_in $amr_frac_in \
    --amr_frac_out $amr_frac_out

# ./../gslib/field-interp -m1 mesh.mesh -m2 meshes/geqdsk.msh -s1 final.gf -no-vis
