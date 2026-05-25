#ifndef GS_IO_UTILS_HPP
#define GS_IO_UTILS_HPP

#include "mfem.hpp"
#include <utility>

// Save a Mesh while forcing MFEM to emit the legacy v1.0/v1.2 header.
//
// MFEM 4.9 emits "MFEM mesh v1.3" whenever attribute_sets or
// bdr_attribute_sets are non-empty (mesh/mesh.cpp Printer). Older GLVis
// binaries (the X11/OGL1 build still on this cluster -- SDL2/glm are
// unavailable so a newer GLVis cannot be built) only know v1.0/v1.2 and
// abort on v1.3 with "Unknown input mesh format". The integer attribute
// IDs themselves live in a separate section unchanged across versions,
// so legacy GLVis still color-codes elements correctly -- only the
// name->{int} map from Gmsh's $PhysicalNames is dropped on disk. The
// in-memory mesh is unchanged (sets are swapped out and back).
inline void SaveMeshLegacyFormat(mfem::Mesh &mesh, const char *fname)
{
   mfem::ArraysByName<int> saved_attr;
   mfem::ArraysByName<int> saved_bdr_attr;
   std::swap(mesh.attribute_sets.attr_sets,     saved_attr);
   std::swap(mesh.bdr_attribute_sets.attr_sets, saved_bdr_attr);
   try
   {
      mesh.Save(fname);
   }
   catch (...)
   {
      std::swap(mesh.attribute_sets.attr_sets,     saved_attr);
      std::swap(mesh.bdr_attribute_sets.attr_sets, saved_bdr_attr);
      throw;
   }
   std::swap(mesh.attribute_sets.attr_sets,     saved_attr);
   std::swap(mesh.bdr_attribute_sets.attr_sets, saved_bdr_attr);
}

#endif
