// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include <stdio.h>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    std::cout<<"Testing Sidre data files"<<std::endl;
    //Set up a small mesh and a couple of grid function on that mesh
    Mesh *mesh = new Mesh(2, 3, Element::QUADRILATERAL, 0, 2.0, 3.0);
    FiniteElementCollection *fec = new LinearFECollection;
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
    GridFunction *u = new GridFunction(fespace);
    GridFunction *v = new GridFunction(fespace);

    int N = u->Size();
    for (int i = 0; i < N; ++i)
    {
        (*u)(i) = double(i);
        (*v)(i) = double(N - i - 1);
    }

    int intOrder = 3;

    QuadratureSpace *qspace = new QuadratureSpace(mesh, intOrder);
    QuadratureFunction *qs = new QuadratureFunction(qspace, 1);
    QuadratureFunction *qv = new QuadratureFunction(qspace, 2);

    int Nq = qs->Size();
    for (int i = 0; i < Nq; ++i)
    {
        (*qs)(i) = double(i);
        (*qv)(2*i+0) = double(i);
        (*qv)(2*i+1) = double(Nq - i - 1);
    }

    SidreDataCollection dc("base", mesh);
    dc.RegisterField("u", u);
    dc.RegisterField("v", v);
    dc.RegisterQField("qs", qs);
    dc.RegisterQField("qv", qv);
    dc.SetCycle(5);
    dc.SetTime(8.0);
    dc.Save();

    SidreDataCollection dc_new("base");
    dc_new.Load(dc.GetCycle());
    Mesh* mesh_new = dc_new.GetMesh();

    if(mesh_new == nullptr) {
        std::cout << "Mesh not returned..." << std::endl;
        // return -1;
    }

    int order = dc_new.GetQFieldOrder("qs");
    double* data = dc_new.GetQFieldData("qs");

    std::cout << "Order " << order << "data[0]" << data[0];

    QuadratureSpace qspace_new(mesh, order);

    QuadratureFunction qs_new(&qspace_new, data, 1);

    qs_new.Print();


    delete u; delete v; delete fespace;
    delete qs; delete qv; delete qspace;
    delete mesh;

    return 0;
}
