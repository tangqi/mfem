# Porting the TDS Grad-Shafranov solver to MFEM 4.9

## Context

The GS solver was originally a fork of **MFEM 4.5.3**. Adding cut-cell / cut-volume
integration (MFEM example `ex38`) requires `MomentFittingIntRules` and the `real_t`
type, both introduced in **MFEM 4.7**. The solver was therefore rebased onto
**MFEM 4.9** (latest stable, released 2025-12-11), which contains the cut-cell
machinery. Integrating cut-cell support into the GS solver itself is future work;
this document covers the port that makes it possible.

**Tree layout** (created during the port):
```
~/Desktop/MFEM/MFEM-4.9-parallel/
├── mfem-gs/        clone of MFEM v4.9 (branch gs_cut_cell), with miniapps/tds-gs/
├── hypre -> hypre-2.26.0
├── metis-4.0 -> metis-4.0.3
└── gslib/
```
The original `~/Desktop/MFEM/MFEM-4.8-parallel/mfem-gs` (MFEM 4.5.3) is untouched
and remains the rollback baseline.

---

## Status

| Step | Status |
|------|--------|
| 1. Set up MFEM 4.9 tree + TPLs + copy tds-gs miniapp | ✅ Done |
| 2. Build MFEM 4.9 with LAPACK / MPI / GSLIB | ✅ Done |
| 3. Build & run `ex38` (cut-cell health check) | ✅ Done |
| 4. Port tds-gs miniapp to the 4.9 API | ✅ Done |
| 5. Fix `std::ios_failure` crash — Part A (gate debug output) | ✅ Done |
| 5. Fix `std::ios_failure` crash — Part B (non-fatal `SafeSave`) | ⬜ Pending |
| 6. Verification: full run-script suite vs. 4.5.3 baseline | ⬜ Pending |
| 7. Fix GLVis "Unknown input mesh format: v1.3" via `SaveMeshLegacyFormat` | ✅ Done |

---

## Completed work

### Steps 1–3 — MFEM 4.9 in place

- Cloned MFEM `v4.9` into `MFEM-4.9-parallel/mfem-gs`, branch `gs_cut_cell`.
- Copied the TPLs (`hypre-2.26.0`, `metis-4.0.3`, `gslib`) alongside it; they are
  compatible with 4.9, no rebuild needed.
- Extracted `miniapps/tds-gs/` from the 4.5.3 fork's `gs_cut_cell` branch.
- Built MFEM 4.9 and confirmed `ex38` builds and runs.

**Build commands** (HPC, from `MFEM-4.9-parallel/mfem-gs`):
```bash
module purge
module load gcc/12.3.0 mvapich2/2.3.7-1 openblas/0.3.23
make config MFEM_USE_MPI=YES MFEM_USE_METIS_5=NO MFEM_USE_GSLIB=YES \
            MFEM_USE_LAPACK=YES MFEM_DEBUG=YES \
            MFEM_MPIEXEC=srun MFEM_MPIEXEC_NP=-n \
            LAPACK_LIB="-L$OPENBLAS_ROOT/lib -lopenblas -Wl,-rpath,$OPENBLAS_ROOT/lib"
make -j8
```
Build the miniapp from `miniapps/tds-gs/` with `make -j1` after loading the same
modules.

### Step 4 — tds-gs ported to the 4.9 API

The miniapp needed exactly **one** source change: MFEM 4.9 removed the public
`Refinement::ref_type` field. In `amr.cpp` (`RegionalThresholdRefiner::ApplyRef`),
`ref.ref_type = aniso_flags[ref.index];` became `ref.SetType(aniso_flags[ref.index]);`
— matching upstream's own `ThresholdRefiner` (`mesh/mesh_operators.cpp`).
Everything else compiled unchanged (`real_t` defaults to `double`, so the custom
`Coefficient`/`Operator`/`Solver` subclass signatures still match).

### Step 5 Part A — crash fix: gate debug output

**Root cause of the `std::ios_failure` / `basic_ios::clear: iostream error` crash:**
MFEM 4.9's `ofgzstream` (used by every `GridFunction::Save` / `Mesh::Save` /
`DataCollection::Save`) calls `exceptions(std::ios_base::badbit)`
(`general/zstr.hpp`). Any failed open or short write now **throws**; nothing catches
it, so the process aborts. MFEM 4.5.3's streams had no exceptions enabled, so the
same failure was silently swallowed. The GS solver writes a large volume of
per-iteration debug `.gf` / VisIt / mesh files, so on 4.9 any I/O hiccup is fatal.

**Fix (Part A):** a `--debug-output` flag (default `0` = off) gates the
per-iteration debug/visualization writes.

- `gs.hpp` — `int debug_output = 0;` added to `GSProblemConfig`.
- `main.cpp` — `-dbg` / `--debug-output` CLI option.
- `sys_operator.hpp` / `.cpp` — `debug_output` member + `set_debug_output()` setter;
  gates the four debug `.Save()` blocks in `NonlinearEquationRes`.
- `gs.cpp` — `debug_output` threaded into `SolveControlProblem` and
  `SolveFixedBoundaryProblem`; gates the per-iteration mesh snapshot, `eq_res`
  save, `WritePerIterationDiagnostics` VisIt frame, and the `xtmp`/`res` writes.
  Final/once-per-run output (final solution, GEQDSK, final ParaView) is **not**
  gated.
- All 15 `run_*.sh` scripts now pass `--debug-output 1` as the last `./main`
  argument (set to `0` for a fast/quiet run).

Builds cleanly; `./main --help` shows the flag.

---

## Remaining work

### Step 5 Part B — make stream errors non-fatal (not yet done)

Even with debug output off, a few essential writes remain; a failure in one should
warn, not abort. Plan:

1. Add a helper in a new `io_utils.hpp` (included by `gs.cpp` and `sys_operator.cpp`):
   ```cpp
   inline void SafeSave(const char *what, const std::function<void()> &save_fn)
   {
      try { save_fn(); }
      catch (const std::exception &e)
      {
         std::cerr << "WARNING: failed to write '" << what << "': " << e.what()
                   << " -- continuing without it." << std::endl;
      }
   }
   ```
2. Route all `.Save()` / `DataCollection::Save()` calls in `gs.cpp`,
   `sys_operator.cpp` (and optionally `build-field.cpp` / `compute-J.cpp`) through
   it, e.g. `SafeSave("gf/f.gf", [&]{ f_.Save("gf/f.gf"); });`.

Part A + Part B together: most writes skipped, and any survivor degrades to a
console warning instead of a crash.

### Step 7 — GLVis "Unknown input mesh format: v1.3"

**Symptom:** Running `glvis -m meshes/mesh_refine.mesh -g gf/<sol>.gf` against output
from the 4.9-ported solver aborted with
```
MFEM abort: Unknown input mesh format: MFEM mesh v1.3
  ... in function: void mfem::Mesh::Loader(std::istream&, int, std::string)
  ... in file: mesh/mesh.cpp:4110
```

**Root cause:** Two-part.
1. **MFEM 4.9 writes `v1.3` whenever the mesh carries named attribute sets.**
   `Mesh::Printer` (`mesh/mesh.cpp:~12024`) emits the header based on
   `attribute_sets.SetsExist() || bdr_attribute_sets.SetsExist()`. The Gmsh
   reader in 4.9 (`mesh/mesh_readers.cpp:~2808`) auto-converts `$PhysicalNames`
   in `.msh` files into named attribute sets. Every solver input mesh in
   `meshes/*.msh` has physical names, so every `Mesh::Save` we do produces v1.3.
2. **The installed GLVis is built against MFEM 4.5.** It lives at
   `~/Desktop/MFEM/MFEM-parallel/glvis/glvis`, linked to the sibling MFEM 4.5
   tree. MFEM 4.5's `Mesh::Loader` only knows `v1.0`/`v1.2` and aborts on `v1.3`.

**Why we didn't rebuild GLVis:** GLVis 3.5+ requires SDL2 + glm + GLEW.
None of those are installed on this PACE cluster (no module, no system RPM,
no spack package). The existing GLVis binary works because it's the pre-3.5
X11/OGL1 build, which upstream GLVis dropped years ago. Installing SDL2 from
source (with X11 dev headers) on top of all the other TPLs would be a
sideshow. The existing binary is fine for visualization — we just need MFEM
4.9 to write a file it can read.

**Fix:** New header [io_utils.hpp](io_utils.hpp) provides
`SaveMeshLegacyFormat(Mesh&, const char*)`. It temporarily swaps the
`attribute_sets.attr_sets` and `bdr_attribute_sets.attr_sets` maps with
empty ones, calls `Mesh::Save`, then swaps them back (exception-safe).
With both maps empty, `Mesh::Printer` falls back to `v1.0` (serial) /
`v1.2` (parallel), which the existing GLVis reads natively. The integer
attribute IDs themselves live in a separate section unchanged across
formats, so element coloring in GLVis is identical — only the
human-readable `name → {int}` map from Gmsh's `$PhysicalNames` is dropped
on disk. The in-memory mesh is unchanged. No source code in the solver
reads the named map (grepped `attribute_set` / `AttributeSet` across
`miniapps/tds-gs/*.{cpp,hpp}` — zero hits), so there is no runtime impact.

**Patched call sites in [gs.cpp](gs.cpp):**
| Old | New |
|-----|-----|
| `mesh->Save(name_mesh)` (debug-gated AMR snapshot, ~L829) | `SaveMeshLegacyFormat(*mesh, name_mesh)` |
| `mesh->Save("meshes/mesh_refine.mesh")` (~L1135) | `SaveMeshLegacyFormat(*mesh, "meshes/mesh_refine.mesh")` |
| `mesh.Save("meshes/mesh.mesh")` (~L1442) | `SaveMeshLegacyFormat(mesh, "meshes/mesh.mesh")` |
| `mesh.Save("meshes/initial.mesh")` (~L1447) | `SaveMeshLegacyFormat(mesh, "meshes/initial.mesh")` |
| `mesh.Save(name_mesh_out)` (~L1499) | `SaveMeshLegacyFormat(mesh, name_mesh_out)` |

Original lines are kept as `//`-prefixed comments next to the new ones, so
the diff against the 4.5.3 baseline stays legible during ongoing verification.

**Verified:**
- Re-ran `run_3_taylor_quad_mesh.sh` after rebuilding `main`. Solver completed
  normally (`time elapsed: 46.6 s`; saddle/magnetic-axis locations within 1e-4
  of the prior 4.9 run).
- `head -1 meshes/mesh_refine.mesh` now reports `MFEM NC mesh v1.0`
  (previously `MFEM mesh v1.3`).
- `glvis -m meshes/mesh_refine.mesh -g gf/final_model2_pc5_cyc1_it5.gf` no
  longer aborts on the mesh format — it gets past the loader and stops only
  at `Can't connect to display!` in non-X11 shells (expected). Will render
  normally under X11 forwarding.

**Follow-ups (intentionally deferred):**
- `test.cpp` (lines 43/88/258/312) and `build-field.cpp:406` also save meshes
  and would benefit from the same helper if their output is ever passed to
  the old GLVis. Not urgent — neither produces files the user is currently
  visualizing.
- Stale pre-fix `meshes/initial.mesh` was not regenerated (this run had
  `do_initial=false`); re-running with `do_initial=true` will refresh it.

### Step 6 — verification (not yet done)

On a Slurm compute node, modules loaded, from `miniapps/tds-gs/`:
1. `bash run_unit_tests.sh`
2. Physics runs: `run_1_ffp.sh`, `run_2_fpol.sh`, `run_3_taylor.sh`,
   `run_3_taylor_amr.sh`, `run_4_lb.sh`, plus the `*_quad_mesh.sh` variants.
3. Compare Newton iteration counts and final ψ field / saddle-point locations
   against the MFEM 4.5.3 baseline (`MFEM-4.8-parallel/mfem-gs`) — expect agreement
   within solver tolerance.

---

## Notes for whoever continues this

- A HPC home-directory quota exhaustion contributed to the original crashes
  (writes failing for lack of space). Keep disk headroom; `~/scratch` has a
  separate, larger quota for bulk output.
- `--debug-output 0` is the safe default until Part B lands; `--debug-output 1`
  re-enables the `gf/*.gf` / VisIt debug files but is still crash-prone on any I/O
  failure until `SafeSave` is in place.
- The 3 minor core-MFEM patches from the old 4.5.3 fork (`fem/fe/fe_base.cpp`,
  `fem/lininteg.cpp`, `linalg/hypre.hpp`) were intentionally **not** re-applied;
  revisit only if verification reveals a regression.
