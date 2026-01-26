// Old preconditioners code. Consists of various preconditioners, mostly unsuccessful. Currently unused, here for reference purposes.

int ind_x, ind_p;

// If we have a non-symmetric block matrix
if (PC_option == 0) {  // PC_option: preconditioner option
    ind_x = 0;
    ind_p = 1;
}

// If we have a symmetric block matrix
else {
    ind_x = 1;
    ind_p = 0;
}

/*
    Form block system
    dx1 = [dy; dp]
    dx2 = [da; dl]

    c1 = [b1; b3 + F H^{-1} b_2]
    c2 = [b4; b5]
*/
BlockSystem.SetBlock(0, ind_x, AMat);
BlockSystem.SetBlock(0, ind_p, BMat);
BlockSystem.SetBlock(1, ind_x, BTMat);
BlockSystem.SetBlock(1, ind_p, CMat);

// Write contents of matrices to text files in CSR format using the WriteSparseMatrixToFile function
FILE *fp_spy;
char filename_spy[60];
sprintf(filename_spy, "spys/spy_model%d_amr%d.txt", model->get_model_choice(), it_amr);
fp_spy = fopen(filename_spy, "w");
fprintf(fp_spy, "AMat\n");
WriteSparseMatrixToFile(fp_spy, AMat);
fprintf(fp_spy, "\nBMat\n");
WriteSparseMatrixToFile(fp_spy, BMat);
fprintf(fp_spy, "\nCMat\n");
WriteSparseMatrixToFile(fp_spy, CMat);

// Define RHS of the block linear system (equation 5.3 of paper)
// RHS = [c1, c2]
// c1 = b1 - Cy b4 / Ca
// c2 = b3 + mu F H^{-1} b_2 - Ba b5 / Ca
rhs = 0;
add(1.0, b1, -b4 / Ca, Cy, rhs.GetBlock(0));  // c1
MuFinvH->Mult(b2, rhs.GetBlock(1));  // c2 (this line and subsequent lines)
rhs.GetBlock(1) += b3;
add(1.0, rhs.GetBlock(1), - b5 / Ca, Ba, rhs.GetBlock(1));
rhs.GetBlock(1) *= scale;

// Configure the FGMRES solver
FGMRESSolver solver;
solver.SetAbsTol(1e-12);
solver.SetRelTol(eta);
solver.SetMaxIter(max_krylov_iter);
solver.SetOperator(BlockSystem);
solver.SetKDim(kdim);
solver.SetPrintLevel(-1);

// Initialize solution guess dx to zero
BlockVector dx(row_offsets);
dx = 0.0;

double dalpha, dlv;

if (PC_option == 0) {
    /*
    non symmetric system, block AMG
    */

    Solver *inv_BT, *inv_B;

    // HypreParMatrix * B_Hypre = ConvertToHypre(BMat);
    // HypreParMatrix * BT_Hypre = ConvertToHypre(BTMat);
    HypreParMatrix * B_Hypre = ConvertToHypre(ScaleByT);
    HypreParMatrix * BT_Hypre = ConvertToHypre(ScaleBy);

    HypreBoomerAMG *B_AMG = new HypreBoomerAMG(*B_Hypre);
    HypreBoomerAMG *BT_AMG = new HypreBoomerAMG(*BT_Hypre);

    B_AMG->SetPrintLevel(0);
    B_AMG->SetCycleType(amg_cycle_type);
    B_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    B_AMG->SetMaxIter(amg_max_iter);

    BT_AMG->SetPrintLevel(0);
    BT_AMG->SetCycleType(amg_cycle_type);
    BT_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BT_AMG->SetMaxIter(amg_max_iter);

    inv_B = B_AMG;
    inv_BT = BT_AMG;

    BlockDiagonalPreconditioner BlockPrec(row_offsets);
    BlockPrec.SetDiagonalBlock(0, inv_B);
    BlockPrec.SetDiagonalBlock(1, inv_BT);
    solver.SetPreconditioner(BlockPrec);

    solver.Mult(rhs, dx);
    printf("BlockOperator.BlockSystem iterations:            %d\n", solver.GetNumIterations());
    fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
}

else if (PC_option == 1) {
    /*
    non symmetric system, block AMG, schur complement
    */

    // form approximation to schur complement
    double denom = Ca;
    for (int j = 0; j < pv.Size(); ++j) {
    denom -= Ba(j) * Cy(j) / By(j, j);
    }
    SparseMatrix *Mapp;
    Mapp = new SparseMatrix(pv.Size(), pv.Size());
    for (int j = 0; j < pv.Size(); ++j) {
    for (int k = 0; k < pv.Size(); ++k) {
        if (j == k) {
        Mapp->Set(j, k, 1.0 / By(j, j) - Ba(j) * Cy(k) / (denom * By(j, j) * By(k, k)));
        } else {
        if (Ba(j) * Cy(k) != 0.0) {
            Mapp->Set(j, k, - Ba(j) * Cy(k) / (denom * By(j, j) * By(k, k)));
        }
        }
    }
    }
    Mapp->Finalize();
    // B - A Mapp C
    SparseMatrix *MC = Mult(*Mapp, *CMat);
    SparseMatrix *AMC = Mult(*AMat, *MC);
    SparseMatrix *SC = Add(1.0, *BMat, -1.0, *AMC);

    Solver *inv_SC, *inv_BT;

    HypreParMatrix * SC_Hypre = ConvertToHypre(SC);
    HypreParMatrix * BT_Hypre = ConvertToHypre(BTMat);

    HypreBoomerAMG *SC_AMG = new HypreBoomerAMG(*SC_Hypre);
    HypreBoomerAMG *BT_AMG = new HypreBoomerAMG(*BT_Hypre);

    SC_AMG->SetPrintLevel(0);
    SC_AMG->SetCycleType(amg_cycle_type);
    SC_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    SC_AMG->SetMaxIter(amg_max_iter);

    BT_AMG->SetPrintLevel(0);
    BT_AMG->SetCycleType(amg_cycle_type);
    BT_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BT_AMG->SetMaxIter(amg_max_iter);

    inv_SC = SC_AMG;
    inv_BT = BT_AMG;

    BlockDiagonalPreconditioner BlockPrec(row_offsets);
    BlockPrec.SetDiagonalBlock(0, inv_SC);
    BlockPrec.SetDiagonalBlock(1, inv_BT);
    solver.SetPreconditioner(BlockPrec);
    
    solver.Mult(rhs, dx);
    printf("BlockOperator.BlockSystem iterations:            %d\n", solver.GetNumIterations());
    fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());      
}

else if (PC_option == 2){
    /*
    Non Symmetric System, AMG on full block system
    */
    Solver *inv_BlockMatrix;
    HypreParMatrix * AMat_Hypre = ConvertToHypre(AMat);
    HypreParMatrix * BMat_Hypre = ConvertToHypre(BMat);
    HypreParMatrix * BTMat_Hypre = ConvertToHypre(BTMat);
    HypreParMatrix * CMat_Hypre = ConvertToHypre(CMat);

    Array2D<HypreParMatrix *> Block(2, 2);
    Block(0, 0) = BMat_Hypre;
    Block(0, 1) = AMat_Hypre;
    Block(1, 0) = CMat_Hypre;
    Block(1, 1) = BTMat_Hypre;

    HypreParMatrix * BlockMatrix_Hypre = HypreParMatrixFromBlocks(Block);
    HypreBoomerAMG *BlockMatrix_AMG = new HypreBoomerAMG(*BlockMatrix_Hypre);

    BlockMatrix_AMG->SetPrintLevel(0);
    BlockMatrix_AMG->SetCycleType(amg_cycle_type);
    BlockMatrix_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BlockMatrix_AMG->SetMaxIter(amg_max_iter);

    inv_BlockMatrix = BlockMatrix_AMG;
    solver.SetPreconditioner(*inv_BlockMatrix);

    solver.Mult(rhs, dx);
}

else if (PC_option == 3){
    /*
    Non Symmetric System, AMG on partial full block system
    */
    Solver *inv_BlockMatrix;
    HypreParMatrix * AMat_Hypre = ConvertToHypre(AMat);
    HypreParMatrix * BMat_Hypre = ConvertToHypre(BMat);
    HypreParMatrix * BTMat_Hypre = ConvertToHypre(BTMat);

    Array2D<HypreParMatrix *> Block(2, 2);
    Block(0, 0) = BMat_Hypre;
    Block(0, 1) = AMat_Hypre;
    Block(1, 1) = BTMat_Hypre;

    HypreParMatrix * BlockMatrix_Hypre = HypreParMatrixFromBlocks(Block);
    HypreBoomerAMG *BlockMatrix_AMG = new HypreBoomerAMG(*BlockMatrix_Hypre);

    BlockMatrix_AMG->SetPrintLevel(0);
    BlockMatrix_AMG->SetCycleType(amg_cycle_type);
    BlockMatrix_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BlockMatrix_AMG->SetMaxIter(amg_max_iter);

    inv_BlockMatrix = BlockMatrix_AMG;
    solver.SetPreconditioner(*inv_BlockMatrix);

    solver.Mult(rhs, dx);
    fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
}

else if (PC_option == 4){
    /*
    Non Symmetric System, stepped approach
    */

    Solver *inv_BT, *inv_B;

    HypreParMatrix * B_Hypre = ConvertToHypre(BMat);
    HypreParMatrix * BT_Hypre = ConvertToHypre(BTMat);

    HypreBoomerAMG *B_AMG = new HypreBoomerAMG(*B_Hypre);
    HypreBoomerAMG *BT_AMG = new HypreBoomerAMG(*BT_Hypre);

    B_AMG->SetPrintLevel(0);
    B_AMG->SetCycleType(amg_cycle_type);
    B_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    B_AMG->SetMaxIter(amg_max_iter);

    BT_AMG->SetPrintLevel(0);
    BT_AMG->SetCycleType(amg_cycle_type);
    BT_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BT_AMG->SetMaxIter(amg_max_iter);

    inv_B = B_AMG;
    inv_BT = BT_AMG;

    // B - A (BT)^{-1} C
    SchurComplement SC(BTMat, CMat, AMat, BMat, inv_BT, light_tol);
    SchurComplementInverse SCinv(&SC, inv_B, light_tol);

    GMRESSolver bsolver;
    bsolver.SetAbsTol(1e-16);
    bsolver.SetRelTol(light_tol);
    bsolver.SetMaxIter(max_krylov_iter);
    bsolver.SetOperator(*BTMat);
    bsolver.SetKDim(kdim);
    bsolver.SetPrintLevel(-1);
    bsolver.SetPreconditioner(*inv_BT);
    
    BlockDiagonalPreconditioner BlockPrec(row_offsets);
    BlockPrec.SetDiagonalBlock(0, &SCinv);
    BlockPrec.SetDiagonalBlock(1, &bsolver);
    solver.SetPreconditioner(BlockPrec);

    solver.Mult(rhs, dx);
    printf("BlockOperator.BlockSystem iterations:            %d\n", solver.GetNumIterations());
    printf("SchurComplement.SC average iterations:           %.2f\n", SC.GetAvgIterations());
    printf("SchurComplementInverse.SCinv average iterations: %.2f\n", SCinv.GetAvgIterations());
    printf("bsolver average iterations:                      %.2f\n", ((double) bsolver.GetNumIterations()));
    fprintf(fp, "amr=%d newton=%d iters=%d amgTot=%f\n", it_amr, i, solver.GetNumIterations(), solver.GetNumIterations() * (SC.GetAvgIterations() * SCinv.GetAvgIterations() + ((double) bsolver.GetNumIterations())));
}

else if (PC_option == 5) {

    // upper triangular AMG
    Solver *inv_BT, *inv_B;

    // HypreParMatrix * B_Hypre = ConvertToHypre(BMat);
    // HypreParMatrix * BT_Hypre = ConvertToHypre(BTMat);
    HypreParMatrix * B_Hypre = ConvertToHypre(ScaleByT);
    HypreParMatrix * BT_Hypre = ConvertToHypre(ScaleBy);

    HypreBoomerAMG *B_AMG = new HypreBoomerAMG(*B_Hypre);
    HypreBoomerAMG *BT_AMG = new HypreBoomerAMG(*BT_Hypre);

    B_AMG->SetPrintLevel(0);
    B_AMG->SetCycleType(amg_cycle_type);
    B_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    B_AMG->SetMaxIter(amg_max_iter);

    BT_AMG->SetPrintLevel(0);
    BT_AMG->SetCycleType(amg_cycle_type);
    BT_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BT_AMG->SetMaxIter(amg_max_iter);

    inv_B = B_AMG;
    inv_BT = BT_AMG;
    
    SchurPC SCPC(AMat, CMat, inv_B, inv_BT, &Ba, &Cy, Ca, 1);
    solver.SetPreconditioner(SCPC);
    solver.Mult(rhs, dx);
    printf("BlockOperator.BlockSystem iterations:            %d\n", solver.GetNumIterations());
    fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
}

else if (PC_option == 6) {
    // lower triangular AMG

    Solver *inv_BT, *inv_B;

    // HypreParMatrix * B_Hypre = ConvertToHypre(BMat);
    // HypreParMatrix * BT_Hypre = ConvertToHypre(BTMat);
    HypreParMatrix * B_Hypre = ConvertToHypre(ScaleByT);
    HypreParMatrix * BT_Hypre = ConvertToHypre(ScaleBy);

    HypreBoomerAMG *B_AMG = new HypreBoomerAMG(*B_Hypre);
    HypreBoomerAMG *BT_AMG = new HypreBoomerAMG(*BT_Hypre);

    B_AMG->SetPrintLevel(0);
    B_AMG->SetCycleType(amg_cycle_type);
    B_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    B_AMG->SetMaxIter(amg_max_iter);

    BT_AMG->SetPrintLevel(0);
    BT_AMG->SetCycleType(amg_cycle_type);
    BT_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BT_AMG->SetMaxIter(amg_max_iter);

    inv_B = B_AMG;
    inv_BT = BT_AMG;
    
    SchurPC SCPC(AMat, CMat, inv_B, inv_BT, &Ba, &Cy, Ca, 2);
    solver.SetPreconditioner(SCPC);
    solver.Mult(rhs, dx);
    printf("BlockOperator.BlockSystem iterations:            %d\n", solver.GetNumIterations());
    fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
}

else if (PC_option == 7) {
    // block diagonal woodbury

    Solver *inv_BT, *inv_B;

    HypreParMatrix * B_Hypre = ConvertToHypre(ScaleByT);
    HypreParMatrix * BT_Hypre = ConvertToHypre(ScaleBy);

    HypreBoomerAMG *B_AMG = new HypreBoomerAMG(*B_Hypre);
    HypreBoomerAMG *BT_AMG = new HypreBoomerAMG(*BT_Hypre);
    
    B_AMG->SetPrintLevel(0);
    B_AMG->SetCycleType(amg_cycle_type);
    B_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    B_AMG->SetMaxIter(amg_max_iter);

    BT_AMG->SetPrintLevel(0);
    BT_AMG->SetCycleType(amg_cycle_type);
    BT_AMG->SetCycleNumSweeps(amg_num_sweeps_a, amg_num_sweeps_b);
    BT_AMG->SetMaxIter(amg_max_iter);

    inv_B = B_AMG;
    inv_BT = BT_AMG;
    
    SchurPC SCPC(AMat, CMat, inv_B, inv_BT, &Ba, &Cy, Ca, 3);
    solver.SetPreconditioner(SCPC);
    solver.Mult(rhs, dx);
    printf("BlockOperator.BlockSystem iterations:            %d\n", solver.GetNumIterations());
    fprintf(fp, "amr=%d newton=%d iters=%d\n", it_amr, i, solver.GetNumIterations());
}

else if (PC_option == -1) {
    /*
    Symmetric System, Schur Complement
    */

    // *** compute SC_op = C - B^T invApaI B *** //
    double alpha_ = .001;
    Vector diag(pv.Size());
    AMat->GetDiag(diag);
    SparseMatrix *ApaI_app;
    ApaI_app = new SparseMatrix(pv.Size(), pv.Size());
    SparseMatrix *invApaI;
    invApaI = new SparseMatrix(pv.Size(), pv.Size());
    SparseMatrix *aI;
    aI = new SparseMatrix(pv.Size(), pv.Size());
    for (int j = 0; j < pv.Size(); ++j) {
    invApaI->Set(j, j, 1.0 / (alpha_ + diag(j)));
    ApaI_app->Set(j, j, (alpha_ + diag(j)));
    aI->Set(j, j, alpha_);
    }
    aI->Finalize();
    invApaI->Finalize();
    SparseMatrix *ApaI = Add(1.0, *AMat, 1.0, *aI);
    SparseMatrix *op1 = Mult(*invApaI, *BMat);
    SparseMatrix *op2 = Mult(*BTMat, *op1);
    SparseMatrix *SC_op = Add(1.0, *CMat, -1.0, *op2);

    Solver *inv_ApaI, *inv_SC;

    //https://hypre.readthedocs.io/en/latest/api-sol-parcsr.html
    // set up AMG for Schur complement
    HypreParMatrix * SC_Hypre = ConvertToHypre(SC_op);
    HypreBoomerAMG *SC_AMG = new HypreBoomerAMG(*SC_Hypre);
    SC_AMG->SetPrintLevel(0);
    SC_AMG->SetCycleType(1);
    SC_AMG->SetCycleNumSweeps(1, 1);
    SC_AMG->SetMaxIter(10);
    inv_SC = SC_AMG;

    // Gauss Seidel for A + alpha I
    GSSmoother ojs(*ApaI, 0, 2);
    inv_ApaI = &ojs;

    // Operators for SC and inverse of SC
    SchurComplement SC(ApaI, BMat, BTMat, CMat, inv_ApaI, light_tol);
    SchurComplementInverse SCinv(&SC, inv_SC, light_tol);

    // solver for A + alpha I
    CGSolver ApaIinv;
    ApaIinv.SetAbsTol(1e-16);
    ApaIinv.SetRelTol(krylov_tol);
    ApaIinv.SetMaxIter(max_krylov_iter);
    ApaIinv.SetOperator(*ApaI);
    ApaIinv.SetPreconditioner(ojs);
    ApaIinv.SetPrintLevel(0);

    // define preconditioner
    BlockDiagonalPreconditioner BlockPrec(row_offsets);
    BlockPrec.SetDiagonalBlock(0, &ApaIinv);
    BlockPrec.SetDiagonalBlock(1, &SCinv);
    solver.SetPreconditioner(BlockPrec);

    solver.Mult(rhs, dx);
    printf("ApaIinv average iterations:                        %d\n", ApaIinv.GetNumIterations());
    printf("SchurComplement.SC average iterations:           %.2f\n", SC.GetAvgIterations());
    printf("SchurComplementInverse.SCinv average iterations: %.2f\n", SCinv.GetAvgIterations());
    printf("BlockOperator.BlockSystem iterations:            %d\n", solver.GetNumIterations());
}

else {
    // TODO!!!
}
