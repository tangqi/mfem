# Cut-Cell Plasma Current (I_p) — Validation Module

## Context

This codebase is an MFEM-based free-boundary Grad-Shafranov solver. The plasma
region is currently identified by a **BFS** over a vertex adjacency map
(`compute_plasma_points` in `plasma_model.cpp:458-661`): vertices satisfying
`psi_ma <= psi <= psi_x` and connected to the magnetic axis are collected into
`set<int> plasma_inds`. The total plasma current is then
`I_p = -integral of g(psi,r)` over the plasma region, where the region is
approximated by **whole elements** all of whose vertices lie in `plasma_inds`
(a staircase approximation — boundary elements straddling the separatrix either
count in full or not at all).

The goal is to eventually replace the BFS detector with an MFEM **cut-cell**
(moment-fitting) approach (cf. `examples/ex38.cpp`), which resolves the
separatrix at sub-element resolution. As a first step, we want a module that
recomputes `I_p` with the cut-cell method and prints it side-by-side with the
BFS value during a solver run. The two values should agree closely on a
converged solution; this validates that the cut-cell machinery is wired up
correctly before it is used to replace the detector.

Decisions (confirmed with user):
- **Invocation:** integrated into the solver run — printed each Newton iteration.
- **Region scope:** hybrid — cut-volume integration, but limited to limiter
  elements that touch the BFS-connected region (excludes disconnected
  private-flux pockets so the two `I_p` values are comparable).

## Summary of all changes (as built)

Authoritative record of every change made. The narrative sections further down
(original implementation, "Post-run assessment" round 2, "Performance" round 3)
keep the diagnostic rationale.

### New files

- **`miniapps/tds-gs/cut_cell_current.hpp`** — public API
  `compute_plasma_current_cutcell(psi, psi_x, psi_ma, model, fespace, attr_lim,
  plasma_inds, ok, int_order=-1, ls_order=2)`.
- **`miniapps/tds-gs/cut_cell_current.cpp`** — three pieces:
  - `PsiLevelSetCoefficient` — level set `phi = psi_x - psi`.
  - `PlasmaSourceIntegrand` — gate-free port of `NonlinearGridCoefficient::Eval`
    option==1 (duplicated physics; cross-referenced in `plasma_model.cpp`).
  - `compute_plasma_current_cutcell` — quad-only geometry guard; per-element
    loop over limiter elements touching `plasma_inds`; `MomentFittingIntRules`
    cut-volume rule; returns `-Ip`. When `int_order < 0` it defaults to
    `int_order = 3` (set in round-3 phase 1).

### Modified files — tds-gs miniapp

- **`miniapps/tds-gs/sys_operator.cpp`** — added `#include "cut_cell_current.hpp"`
  and `#include <cmath>`; added the print-only `[I_p comparison]` block in
  `NonlinearEquationRes()` immediately after `plasma_current *= -1.0;`. It prints
  BFS `I_p`, cut-cell `I_p` (both divided by `mu`), and the normalised relative
  difference, once per Newton iteration. Solver behaviour is unchanged —
  diagnostic output only. (The block was first placed in the unused
  `get_plasma_current(GridFunction&,double&)` overload, which the solver never
  calls, then moved here; `get_mu()` is the in-class accessor.)
- **`miniapps/tds-gs/plasma_model.cpp`** — comment-only edit at the option==1
  block (~line 269) warning that the formula is duplicated in
  `cut_cell_current.cpp::PlasmaSourceIntegrand`. No code/behaviour change.
- **`miniapps/tds-gs/makefile`** — `cut_cell_current.cpp` appended to
  `GS_COMMON_SRC`.

### Modified files — MFEM library (`fem/intrules_cut.hpp`, `intrules_cut.cpp`)

Three independent patches (the repo is a full MFEM source tree, so `libmfem`
is rebuilt from it). Both files carry a "LOCAL MODIFICATIONS" notice at the
top.

- **Patch A — element-local level-set projection** (round 2). In
  `ComputeSurfaceWeights2D` (~708) and `ComputeVolumeWeights2D` (~957): replaced
  the whole-mesh `FiniteElementSpace` + `ProjectCoefficient` of the level set
  with an element-local `FiniteElement::Project`. Numerically identical; removes
  an O(mesh-size)-per-cut-element cost (~0.5 s in practice — minor).
- **Patch B — cache the Gram-Schmidt orthonormalisation** (round 3 phase 2).
  `OrthoBasis2D` rewritten: the modified Gram-Schmidt coefficients depend only
  on `(Order, nBasis, elem_geom)`, so they are computed once, cached in a
  function-local `static std::map`, and replayed per quadrature point. Added
  `<map>`, `<vector>`, `<utility>`, `<tuple>` includes. `mGSStep` is now unused
  but left in place. Numerically identical output. **Reversal instructions:
  see the round-3 / phase-2 section below.**
- **Patch C — triangle support.** Added a `Geometry::Type elem_geom` member
  to `MomentFittingIntRules` (set in `InitSurface`) and parameterised the 2D
  moment-fitting path by it: `OrthoBasis2D`'s reference quadrature, and the
  reference-edge outward normals + reference-edge length used by the
  edge-RHS blocks in `ComputeSurfaceWeights2D` / `ComputeVolumeWeights2D`,
  are now selected from `elem_geom`. SQUARE behaviour is bit-identical to
  before; TRIANGLE adds the 3 reference-triangle edges with a `sqrt(2)`
  factor on the hypotenuse to absorb the non-unit reference edge length.
  The geometry guard in `cut_cell_current.cpp` was relaxed to accept
  SQUARE or TRIANGLE. See "Triangle support and refinement study" below
  for the validation runs.

### Performance (run_3_taylor_quad_mesh.sh, refinement 0)

| Stage | runtime |
|---|---|
| baseline, no cut-cell | ~30 s |
| cut-cell, `int_order=6`, no patches | 1262 s |
| + Patch A (element-local projection) | 1252 s |
| + phase 1 (`int_order` 6 → 3) | 65 s |
| + Patch B (Gram-Schmidt cache) | 7.7 s |

Refinement-level 1 with all changes: ~31 s — comparable to baseline without
cut-cell.

### Verification status

- Builds clean (`libmfem` + miniapp), no warnings.
- Patch B confirmed numerically neutral: cut-cell `I_p` values in `run_3`
  (pre-Patch-B) and `run_4` (post-Patch-B) are byte-identical
  (`1.43243e7, 1.26443e7, 1.24031e7, 1.24119e7, 1.24119e7`) — the cache refactor
  changed only speed.
- Cut-cell `I_p` sits ~17 % below BFS on the coarsest mesh, consistent with BFS
  whole-element staircase over-counting.
- **Outstanding:** (1) the refinement study — confirm the BFS-vs-cut-cell gap
  shrinks with `h` across refinement 0/1/2 (timings are known, the gap-vs-`h`
  numbers are not yet recorded); (2) the constant-1 area sanity check.
- The stale `int_order` doc comment in `cut_cell_current.hpp` has been
  corrected (`-1 -> default of 3`).

## Key facts established during exploration

- `I_p` is computed in `SysOperator::get_plasma_current()`
  (`sys_operator.cpp:445-524`): `compute_plasma_points` → `NonlinearGridCoefficient`
  option=1 → `DomainLFIntegrator` in a `LinearForm` → `plasma_term(ones)`, negated.
- The integrand `g(psi,r)` is `NonlinearGridCoefficient::Eval` option==1
  (`plasma_model.cpp:275-291`). `Eval` has two geometric gates at the top:
  `attr_lim` (line 202-204, keep it) and the `plasma_inds` whole-element gate
  (lines 206-215, must be bypassed for cut-cell).
- Cut-cell machinery: `MomentFittingIntRules` in `fem/intrules_cut.{hpp,cpp}`,
  guarded by `MFEM_USE_LAPACK` (enabled in this build). API:
  `GetVolumeIntegrationRule(ElementTransformation&, IntegrationRule&)` builds a
  quadrature rule for the sub-region `{phi > 0}` of an element.
- **Constraint:** the 2D moment-fitting basis (`OrthoBasis2D`/`DivFreeBasis2D`,
  `intrules_cut.cpp:1450+`) hardcodes `Geometry::SQUARE` — **quad meshes only**.
  The default `meshes/iter_gen_fixed.mesh` is triangular; quad meshes
  (`meshes/iter_gen_quad.msh`) exist and are used by `run_*_quad_mesh.sh`.
- `compute_plasma_points` BFS inserts a one-vertex halo of just-outside
  vertices into `plasma_inds` as well — so "element touches `plasma_inds`" is a
  safe, slightly generous connectivity filter; truly disconnected pockets are
  never reached by the BFS and are correctly excluded.
- Build is via `miniapps/tds-gs/makefile`; new `.cpp` files go in `GS_COMMON_SRC`.

## Implementation

**Scope note.** This is a print-only validation: the solver continues to use
the BFS-computed `Plasma_Current` for its residual/Jacobians, unchanged. The
cut-cell value is a pure side computation, printed each Newton iteration
alongside the BFS value and their difference. **No existing file's behavior
changes** — `plasma_model.hpp/.cpp` are left entirely untouched (per design
decision: the gate-free integrand is reimplemented in the new module rather
than added as a flag on `NonlinearGridCoefficient`).

### 1. New module: `cut_cell_current.hpp` / `cut_cell_current.cpp`

`miniapps/tds-gs/cut_cell_current.hpp` — public API:

```cpp
double compute_plasma_current_cutcell(const GridFunction &psi,
                                      double psi_x, double psi_ma,
                                      PlasmaModelBase *model,
                                      FiniteElementSpace *fespace,
                                      int attr_lim,
                                      const std::set<int> &plasma_inds,
                                      bool &ok,
                                      int int_order = -1,   // -1 -> default of 3
                                      int ls_order  = 2);
```

`miniapps/tds-gs/cut_cell_current.cpp` contains three pieces:

**(a) Level-set coefficient** — a small `Coefficient` subclass
`PsiLevelSetCoefficient` evaluating `phi = psi_x - psi(x)` (so `phi > 0` ⇔
inside the separatrix). No separate GridFunction needed.

**(b) Gate-free integrand coefficient** — a `Coefficient` subclass
`PlasmaSourceIntegrand` that reimplements the option=1 plasma-source formula
**without** the `attr_lim` / `plasma_inds` gates. It is a direct port of
`NonlinearGridCoefficient::Eval` option==1 (`plasma_model.cpp:269-291`, plus the
`model_choice` switch block at `:229-266` and the `psi_N`/`ri` setup at
`:217-227`). It holds a `PlasmaModelBase*`, the `psi` GridFunction, and
`psi_ma`/`psi_x`; in `Eval` it computes `ri` via `T.Transform`, `psi_N` via
`normalized_psi`, and returns the same expression. The geometric restriction is
now supplied entirely by the cut-volume rule + element-loop filters.
*Maintenance note (required):* this duplicates physics from `plasma_model.cpp`.
Add cross-referencing comments in **both** places so the formula is kept in sync
if the source model changes:
- In `cut_cell_current.cpp`, a comment on `PlasmaSourceIntegrand` pointing to
  `NonlinearGridCoefficient::Eval` option==1 (`plasma_model.cpp:269-291`).
- In `plasma_model.cpp`, a comment at the option==1 block (~line 269) warning
  that this formula is duplicated in `cut_cell_current.cpp::PlasmaSourceIntegrand`
  and that any change to the source model must be mirrored there. This is the
  **only** edit to `plasma_model.cpp` — a comment, no code/behavior change.

**(c) The driver `compute_plasma_current_cutcell`:**
- **Guard:** if `MFEM_USE_LAPACK` is undefined, or any element is not
  `Geometry::SQUARE`, set `ok=false`, print a skip message, return `NAN`.
- `MomentFittingIntRules mf_ir(int_order, phi, ls_order);`
- For each element `e`: skip unless `attr(e) == attr_lim` **and** at least one
  vertex of `e` is in `plasma_inds` (hybrid connectivity filter).
- `mf_ir.GetVolumeIntegrationRule(*T, vir);` then accumulate
  `Ip += ip.weight * T->Weight() * integrand.Eval(*T, ip)` over the rule's
  points (mirrors ex38's `SubdomainLFIntegrator` weight convention).
- Return `-Ip` (matches the existing `Plasma_Current *= -1.0`).
- **Weight-convention check during bring-up:** integrate the constant `1` over a
  fully-interior element and confirm it equals the physical element area; drop
  the `T->Weight()` factor only if moment-fitting already returns physical
  weights. (ex38 includes the factor; the plan follows ex38.)

### 2. Invoke the comparison in `get_plasma_current()`

`miniapps/tds-gs/sys_operator.cpp` — add `#include "cut_cell_current.hpp"`;
after line 521 (`Plasma_Current *= -1.0;`), call
`compute_plasma_current_cutcell(x, val_x, val_ma, model, fespace, attr_lim,
plasma_inds_, ok)` and print three lines: BFS `I_p`, cut-cell `I_p`, relative
difference (or a "skipped" message when `ok` is false). Everything needed
(`model`, `val_ma`, `val_x`, `x`, `fespace`, `attr_lim`, `plasma_inds_`) is
already in scope here. The return value of `get_plasma_current()` is
**unchanged** (still the BFS `Plasma_Current`); only print statements are added.
`get_plasma_current()` is called once per Newton iteration (from `gs.cpp:906`),
so the comparison prints per iteration as desired.

> As built, the comparison block lives in `NonlinearEquationRes()` (not the
> unused `get_plasma_current(GridFunction&,double&)` overload) — see round 2.

### 3. Build

`miniapps/tds-gs/makefile:11-12` — append `cut_cell_current.cpp` to
`GS_COMMON_SRC`. `GS_COMMON_OBJ` and the `%.o` rule handle the rest.

## Files

- NEW `miniapps/tds-gs/cut_cell_current.hpp`
- NEW `miniapps/tds-gs/cut_cell_current.cpp`
- `miniapps/tds-gs/sys_operator.cpp` — include + print-only comparison call
  (return value unchanged)
- `miniapps/tds-gs/makefile` — add source to `GS_COMMON_SRC`
- `miniapps/tds-gs/plasma_model.cpp` — **comment-only** edit at the option==1
  block (~line 269) warning that the formula is duplicated in
  `cut_cell_current.cpp`; no code/behavior change. `plasma_model.hpp` unchanged.
- `fem/intrules_cut.cpp` — Patch A and Patch B (see "Summary of all changes").
- Read-only reference: `examples/ex38.cpp`, `fem/intrules_cut.hpp`

## Verification

1. **Build:** `make` in `miniapps/tds-gs/`; confirm `cut_cell_current.o`
   compiles and `main` links.
2. **Run on a quad mesh** (the default triangular mesh is skipped by the
   geometry guard): `./run_3_taylor_quad_mesh.sh` and/or
   `./run_2_fpol_quad_mesh.sh`. For the cleanest first check, run with
   `max_amr_levels = 0` (conforming mesh — avoids hanging-node effects).
3. **Expected output:** each Newton iteration prints the three `[I_p comparison]`
   lines. On a converged solution the BFS and cut-cell values should agree to a
   **relative difference of ~1e-2 or better**; the cut-cell value is the more
   accurate one (no staircase over/under-count of boundary elements).
4. **Refinement check:** run with `refinement_factor` 1, 2, 4 and confirm the
   BFS-vs-cut-cell gap shrinks (BFS staircase error is O(h)) — this is the
   strongest evidence the cut-cell rule is correct.
5. **Sanity baseline:** temporarily integrate the constant `1` over `{phi>0}`
   and compare to a hand estimate of the plasma cross-sectional area to confirm
   the weight convention before trusting the `g`-weighted result.

## Optional follow-up: supporting triangular meshes

The default GS mesh (`iter_gen_fixed.mesh`) is triangular. Start with the quad
mesh as above; to extend cut-cell `I_p` to triangle meshes later, here is what
is involved.

**Why triangles fail today.** In `fem/intrules_cut.cpp`, the 2D moment-fitting
path is *almost* geometry-agnostic already: `ComputeSurfaceWeights2D` and
`ComputeVolumeWeights2D` loop over `me->GetNEdges()` (3 for a triangle, 4 for a
quad) and `InitSurface`/`InitVolume` fetch the base quadrature via
`irs.Get(Tr.GetGeometryType(), ...)`. The **only** hardcoded quad assumption in
the 2D path is the Gram-Schmidt orthonormalization domain — `Geometry::SQUARE`
in `OrthoBasis2D` (and, in the original code, `mGSStep`). The divergence-free
basis dimension `nBasis` is a polynomial-space count and is shape-independent.

**Recommended approach — small `libmfem` patch.** This repo is a full MFEM
source tree, so the library can be rebuilt.

1. In `MomentFittingIntRules` (`fem/intrules_cut.hpp`), add a member
   `Geometry::Type elem_geom;` (alongside the existing `dim`).
2. In `InitSurface` (`intrules_cut.cpp` ~line 202) set
   `elem_geom = Tr.GetGeometryType();`.
3. Replace the `IntRules.Get(Geometry::SQUARE, 2*Order+1)` calls in the 2D
   orthonormalization with `IntRules.Get(elem_geom, 2*Order+1)`. (Note: after
   Patch B, `OrthoBasis2D` has a single such call and `mGSStep` is unused.)
4. Rebuild `libmfem`, then rebuild the `tds-gs` miniapp.
5. Remove (or relax) the `Geometry::SQUARE` guard in `cut_cell_current.cpp` so
   triangles are no longer skipped; keep a guard that still rejects any
   non-`SQUARE`/non-`TRIANGLE` 2D geometry.

This is a localized change. Gram-Schmidt over the reference triangle instead of
the square only changes the *conditioning* of the SVD solve, not the span of the
fitted polynomial space — so triangle results should be correct, but the patch
must be validated (Verification step below).

**Validation for triangles.** Repeat the constant-`1` area test (Verification
step 5) on a triangular element and confirm the cut rule reproduces the physical
triangle area; then run the full `I_p` comparison on `iter_gen_fixed.mesh` and
confirm agreement with the BFS value at the same level seen on the quad mesh.

**Zero-library-change interim:** simply run the whole solver on the existing
quad mesh (`run_*_quad_mesh.sh`) — this is not "triangle support" but needs no
patch. Avoid the alternatives of tri→quad mesh conversion or per-triangle
sub-quad meshing: both add interpolation error or substantial module code for no
benefit over the small library patch above.

## Risks

- **Triangle meshes unsupported** — handled cleanly by the `Geometry::SQUARE`
  guard (prints skip, returns `NAN`); use a quad mesh.
- **Non-conforming quad AMR** — hanging nodes may cause small level-set
  inconsistencies at T-junctions; validate first with `max_amr_levels = 0`.
- **Performance** — see the round-2 / round-3 sections below; the headline cost
  was inefficiency inside `MomentFittingIntRules`, addressed by lowering
  `int_order` and by two library patches.
- **`ls_order` vs `psi` order** — keep `ls_order >= order`; default `ls_order=2`
  is safe for the default `order=1`.

## Post-run assessment & performance patch (round 2)

> **Superseded by round 3 below.** The round-2 patch (element-local level-set
> projection) was correct and is kept, but it removed only ~0.5 s — the
> whole-mesh `ProjectCoefficient` was a minor cost, not the bottleneck. The
> real bottleneck is identified in round 3.

First end-to-end run: `run_3_taylor_quad_mesh.sh` (Taylor model, quad mesh,
AMR off, uniform refinement 0), logged to `run_3_taylor_quad_mesh_cut_cell.log`.

### Findings

1. **Wiring fixed.** The comparison block was initially placed in the dead
   function `SysOperator::get_plasma_current(GridFunction&, double&)`, which the
   solver never calls. It was moved into `NonlinearEquationRes()` (right after
   `plasma_current *= -1.0;`), which runs once per Newton iteration. The
   `[I_p comparison]` lines now print. (Already applied.)

2. **Numbers are correct — not a bug.** `relative diff` is normalised:
   `relative diff = |I_p_cutcell - I_p_BFS| / max(1, |I_p_BFS|)`. Converged
   iteration: `|15.597 - 18.8496| / 18.8496 = 0.1726`, matching the printed
   `0.172555`. (The raw printed integral values are now divided by `mu` in the
   `sys_operator.cpp` comparison block, matching the solver's
   `plasma_current = 1.5e7` line — already applied.) The cut-cell value sits
   ~17% below BFS on the coarsest mesh — plausibly whole-element staircase
   over-counting plus, for the Taylor model, the ungated `switch_taylor` term
   being integrated over whole boundary elements by BFS. Confirm via the
   refinement study below.

3. **Runtime 1262 s vs ~30 s baseline — a real performance bug, in the MFEM
   library.** `MomentFittingIntRules`, for every *cut* element, builds a
   whole-mesh `FiniteElementSpace` and runs a whole-mesh `ProjectCoefficient`
   of the level set, then uses only the current element's dofs
   (`fem/intrules_cut.cpp:710-713` in `ComputeSurfaceWeights2D`, and
   `:959-962` in `ComputeVolumeWeights2D`). With a GridFunction-backed level
   set that is ~2x10^8 `psi.GetValue` calls across the solve (~mesh-size x
   cut-elements x 2 x Newton-iters).

### The fix: patch `fem/intrules_cut.cpp` for element-local projection

At both spots (`ComputeSurfaceWeights2D` ~708-734 and `ComputeVolumeWeights2D`
~957-983), replace the whole-mesh projection:

```cpp
H1_FECollection fec(lsOrder, 2);
FiniteElementSpace fes(const_cast<Mesh*>(Tr.mesh), &fec);
GridFunction LevelSet(&fes);
LevelSet.ProjectCoefficient(*LvlSet);
mesh->GetElementTransformation(elem, &Trafo);
const FiniteElement* fe = fes.GetFE(elem);
...
Array<int> dofs;
fes.GetElementDofs(elem, dofs);
...
gradi *= LevelSet(dofs[dof]);
```

with an element-local projection:

```cpp
H1_FECollection fec(lsOrder, 2);
mesh->GetElementTransformation(elem, &Trafo);
const FiniteElement* fe = fec.FiniteElementForGeometry(Tr.GetGeometryType());
Vector LevelSet(fe->GetDof());
fe->Project(*LvlSet, Trafo, LevelSet);   // projects coeff onto THIS element only
...
gradi *= LevelSet(dof);                  // local dof index; drop the dofs array
```

`FiniteElement::Project(Coefficient&, ElementTransformation&, Vector&)` is the
standard element-local projection. The result is numerically identical to the
old code (same projected level set on the current element) — only the wasted
whole-mesh work is removed. Cost drops from O(mesh-size) to O(1) per cut
element. Then rebuild `libmfem` and the `tds-gs` miniapp.

Scope: patch the 2D routines only (this is a 2D solver). `ComputeSurfaceWeights3D`
/ `ComputeVolumeWeights3D` have the same pattern but are out of scope.

### Verification (round 2)

1. Rebuild `libmfem` (after the `intrules_cut.cpp` patch) and the miniapp.
2. Re-run `run_3_taylor_quad_mesh.sh`; total runtime should fall back to the
   ~30-60 s range, with `[I_p comparison]` lines still printed every Newton
   iteration.
3. **Refinement study** — re-run with `refinement_factor` 1, 2, 3 and confirm
   the BFS-vs-cut-cell gap shrinks roughly with `h`. This is the real check
   that the cut-cell integration is correct; the ~17% gap at refinement 0
   should drop substantially.
4. **Constant-1 area check** — as a one-off, integrate the constant `1` over
   `{phi>0}` with the cut rule and compare to a hand estimate of the plasma
   cross-sectional area, to rule out a weight-convention error.

## Performance, round 3 — the actual bottleneck

Second run (`run_3_taylor_quad_mesh_cut_cell_run_2.log`) was still 1252 s
(vs 1262 s before): the round-2 patch was verified in the binary
(`intrules_cut.o` → `libmfem.a` → `main` timestamps all in order) and the
cut-cell numbers were byte-identical, confirming it changed nothing material.

### Root cause (verified by reading the moment-fitting hot path)

`MomentFittingIntRules::ComputeSurfaceWeights2D` calls `OrthoBasis2D` **once per
quadrature point** of every cut element (`intrules_cut.cpp:738`). `OrthoBasis2D`
(`:1452`) recomputed the **entire modified Gram-Schmidt orthonormalization from
scratch** on every call — but that orthonormalization depends only on `Order`
and the reference square, *not* on the quadrature point or the element. And
`mGSStep` (`:1494`) heap-allocated tiny `Vector(2)` objects inside its innermost
loops (millions of `new[]`/`delete[]` per cut element).

This is amplified steeply by **`int_order`**: the module originally set
`int_order = 2*order + 4 = 6`. Moment-fitting cost grows very fast with order;
order 6 is gross overkill for validating a smooth integral (`ex38` defaults to
order 2).

### Fix — phase 1: lower `int_order` — APPLIED

In `cut_cell_current.cpp`, the `if (int_order < 0)` block now sets a modest
fixed value:

```cpp
// Moment-fitting cost scales steeply with this order; ~3 is ample for
// validating a smooth source integral (ex38 uses 2). Raising it is expensive
// until the OrthoBasis2D library inefficiency (phase 2) is fixed.
int_order = 3;
```

One-line change, miniapp rebuild only. Brought the run from 1252 s to 65 s.
The cut-cell `I_p` values shift slightly versus order 6 (a lower-order but
still accurate rule) — expected, not a regression.

### Fix — phase 2: patch `OrthoBasis2D` in `fem/intrules_cut.cpp` — APPLIED

Phase 1 alone brought the run to 65 s (`run_3_taylor_quad_mesh_cut_cell_run_3.log`),
which is acceptable. Phase 2 was then applied to speed things up further (to
7.7 s); it can be reverted (see below) if not worthwhile.

**What was done.** `OrthoBasis2D` (`fem/intrules_cut.cpp`) was rewritten to
compute the modified Gram-Schmidt coefficients **once per `(Order, nBasis)`**,
cache them in a function-local `static std::map`, and on every call only
evaluate the raw `DivFreeBasis2D(ip,...)` and *replay* the cached row
operations. The compute-once block uses the exact same nested loops and
arithmetic as the original `OrthoBasis2D` + `mGSStep`, so the output `shape` is
numerically identical to the original per-point recomputation. `mGSStep` is now
unused but **left in place** (no warning; simplifies reversal). Three includes
(`<map>`, `<vector>`, `<utility>`) were added near the top of the file.
`libmfem` and the miniapp were rebuilt.

**Verification:** the cut-cell `I_p` values stayed byte-identical to the
phase-1 run (`run_3` vs `run_4`: 1.43243e7, 1.26443e7, 1.24031e7, 1.24119e7,
1.24119e7); only runtime changed.

**How to revert phase 2** (independent of the round-2 element-local-projection
patch — do not touch that):
1. In `fem/intrules_cut.cpp`, replace the rewritten `OrthoBasis2D` body with the
   original:
   ```cpp
   void MomentFittingIntRules::OrthoBasis2D(const IntegrationPoint& ip,
                                            DenseMatrix& shape)
   {
      const IntegrationRule *ir_ = &IntRules.Get(Geometry::SQUARE, 2*Order+1);
      shape.SetSize(nBasis, 2);
      // evaluate basis in the point
      DenseMatrix preshape(nBasis, 2);
      DivFreeBasis2D(ip, shape);
      // evaluate basis for quadrature points
      DenseTensor shapeMFN(nBasis, 2, ir_->GetNPoints());
      for (int p = 0; p < ir_->GetNPoints(); p++)
      {
         DenseMatrix shapeN(nBasis, 2);
         DivFreeBasis2D(ir_->IntPoint(p), shapeN);
         for (int i = 0; i < nBasis; i++)
            for (int j = 0; j < 2; j++)
            {
               shapeMFN(i, j, p) = shapeN(i, j);
            }
      }
      // do modified Gram-Schmidt orthogonalization
      for (int count = 1; count < nBasis; count++)
      {
         mGSStep(shape, shapeMFN, count);
      }
   }
   ```
2. Optionally remove the added `#include <map>`, `<vector>`, `<utility>` lines
   (harmless to leave).
3. Rebuild `libmfem` (`make` at repo root) and relink the miniapp
   (`rm miniapps/tds-gs/main && make` in `miniapps/tds-gs/`).

### Verification (round 3)

1. Rebuild the miniapp (and `libmfem` for phase 2).
2. Re-run `run_3_taylor_quad_mesh.sh`; runtime is ~7.7 s at refinement 0 and
   ~31 s at refinement 1, with `[I_p comparison]` lines printed each Newton
   iteration.
3. Cut-cell `I_p` values are byte-identical before/after phase 2, confirming the
   cache refactor changed only speed.

## AMR / non-conforming mesh compatibility

`run_3_taylor_quad_mesh.sh` with `max_amr_levels=1` ran successfully, but
that pass did not refine the plasma boundary, so the cut-cell path was not
stressed on hanging nodes near the separatrix. The analysis below was done
before the validation runs.

**Analysis: the cut-cell approach should work on non-conforming AMR meshes.**

- *Cut-volume rule is element-local.* `MomentFittingIntRules::GetVolumeIntegrationRule`
  builds the rule from `T.mesh`, `T.ElementNo`, `T.GetGeometryType()`, edge
  bisection, and `LvlSet->Eval(T, ip)`. None of that consults neighbors or
  cares whether an edge is non-conforming. A refined child quad is still
  `Geometry::SQUARE` with a normal `IsoparametricTransformation` — the cut
  rule is built per element exactly as on a conforming mesh.
- *Level set is C0 across T-junctions.* `PsiLevelSetCoefficient::Eval` returns
  `psi_x - psi.GetValue(T, ip, 0)`. The H1 conforming prolongation constrains
  slave (hanging) DOFs to interpolate the master coarse-side trace, so `psi`
  is continuous across the hanging edge. The solver updates `psi` via the
  usual MFEM Newton path before `NonlinearEquationRes` runs, and the BFS
  code in `compute_plasma_points` already relies on the same synchronisation
  via `fespace->GetConformingProlongation()`.
- *Hybrid connectivity filter works.* `mesh->GetElementVertices(e, verts)`
  returns the element's vertices (master or hanging) by index, and
  `plasma_inds` is a `set<int>` of vertex indices populated by BFS over
  `vertex_map` — the same indexing scheme. The filter works identically on
  AMR meshes.
- *Patches A, B, C* are all per-element / order-only — they are unaffected
  by mesh non-conformity.
- *Solver-side slave-DOF caveat that does NOT bite us.* The existing solver
  has special handling for the case where `ind_ma` / `ind_x` (the O-/X-point
  DOFs) land on a slave vertex — see the `snap_to_master` logic in
  `SysOperator::NonlinearEquationRes`. That is a *Jacobian-construction*
  concern (the slave column needs the prolongation applied correctly). The
  cut-cell module only computes a scalar integral and does not touch any
  Jacobian column, so the snap logic is not needed on its path.

If a run does crash or the numbers go off, the most likely culprit (and the
only one not bullet-proofed above) is a stale `psi` slave-DOF state —
fixable by ensuring `psi.SetFromTrueDofs(...)` (or the equivalent
prolongation step) runs before the cut-cell call. That has not been needed
so far.

## Triangle support and refinement study — results

Patch C (above) was implemented to add triangle-mesh support to the 2D
moment-fitting path. Validation runs (both Taylor model, `Ip=1.5e7`,
`max_newton_iter=8`, `do_control=1`):

| run | mesh | refinement_factor | max_amr_levels | runtime |
|---|---|---|---|---|
| `run_3_taylor_quad_mesh.sh` | `iter_gen_quad.msh` | 1 | 1 | 46.7 s |
| `run_3_taylor.sh`           | `iter_gen_fixed.mesh` | 1 | 4 | 311.6 s |

**Quad regression / non-conforming AMR check.** BFS converges to `1.5e7`,
cut-cell to `~1.338e7`; relative diff stays at `~10.5–10.8 %` across all
Newton iterations — including after AMR triggers at iter 8 (VSize `5921`
→ `6281`, `TrueVSize 6179` < VSize, i.e. hanging nodes are present). No
crash, no drift in the cut-cell value when hanging nodes appear. This is
the first confirmed run of cut-cell on a non-conforming mesh and it
behaves as the analysis above predicted.

**Triangle refinement study.** With `max_amr_levels=4`, AMR walks the mesh
through 5 sizes during the solve; the relative diff falls monotonically:

| VSize | BFS I_p | cut-cell I_p | relative diff |
|---|---|---|---|
| 1850  | 1.500e7 | 1.256e7 | 16.26 % |
| 2041  | 1.500e7 | 1.256e7 | 16.27 % |
| 3490  | 1.500e7 | 1.306e7 | 12.95 % |
| 7992  | 1.500e7 | 1.385e7 |  7.68 % |
| 19248 | 1.500e7 | 1.438e7 |  4.16 % |

The gap shrinks ~4× as DOFs grow ~10× (i.e. h ~ 3.2× smaller) — slightly
faster than `O(h)`, consistent with AMR concentrating refinement on the
separatrix band where the staircase error lives. This is the strongest
evidence that the cut-cell rule is mathematically correct on triangles: a
wrong hypotenuse `sqrt(2)` factor or wrong reference-edge normal would
amplify bias on finer meshes (more hypotenuses cutting the separatrix), not
reduce it. The monotone convergence to BFS rules that out.

Other observations:
- First triangle iteration has `BFS = -2.22e7`, `cut-cell = -1.85e7` — both
  negative on the unconverged initial guess. Sign- and magnitude-consistency
  confirms both methods integrate the same integrand and respond to `psi`
  the same way far from convergence.
- The coarse-tri gap (`~16 %`) and the coarse-quad gap (`~17 %`) are
  consistent at comparable separatrix h — exactly what a
  geometry-independent rule should produce.

## Option A — multi-model validation: results

All four `PlasmaSourceIntegrand` `model_choice` cases were exercised on
both triangular and quadrilateral meshes (eight runs total). All ran
end-to-end with no crashes and no `cut-cell skipped` messages.

### Important caveat on refinement variation

The eight runs used **different `refinement_factor` and `max_amr_levels`**
settings — they were the existing scripts as-shipped, not a controlled
study. Final VSize spans 1850 → 19248, so absolute rel-diff numbers are not
directly comparable across runs. Worse, the starting `iter_gen_quad` mesh
has roughly **twice** the element count of `iter_gen_fixed`, so even
matched-refinement settings would leave quad runs at ~2× the DOF count of
tri runs. Cross-mesh-within-model rel-diff comparisons should be read as
"consistent same order of magnitude" only — not as a true tri-vs-quad test
at matched h. A controlled re-run (planned: `refinement_factor=1`,
`max_amr_levels=1` for all 8) is in flight.

The cross-MODEL comparison still holds despite this. The rel-diff spread
across models is four orders of magnitude (`6e-5` → `0.1`); refinement
level shifts any single number by at most ~1 order of magnitude. The
**ordering** Model 4 ≪ Models 1/3 ≪ Model 2 is therefore robust under any
reasonable refinement.

### Run inventory and converged values

Script ↔ `model_choice` mapping (verified against the CLI flags in each
log's header; the script *names* describe the origin of `ff'`, not the
`model_choice` number):

| run | script CLI `--model` | switch-block role |
|---|---|---|
| run_1 ffp     | 3 | `switch_ff   = 1` (15MA ITER baseline ff' from ff' data) |
| run_2 fpol    | 1 | `switch_beta = 1` (Luxon-Brown variant, ff' from fpol data) |
| run_3 taylor  | 2 | `switch_taylor = -1` (Taylor state) |
| run_4 lb      | 4 | `switch_lb   = 1` (Luxon-Brown) |

(An earlier draft of this doc had run_1/run_2 swapped — corrected here.)

Converged values per run (note the refinement caveat above):

| run | mesh | final VSize | BFS I_p | cut-cell I_p | rel diff | runtime |
|---|---|---|---|---|---|---|
| run_1 ffp    | tri  | 1850  | 1.500e7 | 1.484e7 | 1.08 %  | 12 s  |
| run_1 ffp    | quad | 5921  | 1.500e7 | 1.474e7 | 1.75 %  | 214 s |
| run_2 fpol   | tri  | 8310  | 1.500e7 | 1.493e7 | 0.50 %  | 274 s |
| run_2 fpol   | quad | 6470  | 1.505e7 | 1.482e7 | 1.53 %  | 262 s |
| run_3 taylor | tri  | 19248 | 1.500e7 | 1.438e7 | 4.16 %  | 312 s |
| run_3 taylor | quad | 6179  | 1.500e7 | 1.338e7 | 10.77 % | 47 s  |
| run_4 lb     | tri  | 3836  | 1.500e7 | 1.500e7 | 0.006 % | 32 s  |
| run_4 lb     | quad | 8483  | 1.508e7 | 1.508e7 | 0.015 % | 569 s |

### Interpretation

The relative-diff spread tracks one thing: **whether the integrand vanishes
at `ψ_N = 1`** (the separatrix).

- *Model 4 (Luxon-Brown), rel diff < 0.02 %.* All `switch_lb`-gated terms
  are inside the `in01 = (psi_N > 0) && (psi_N < 1)` ternary; at `ψ_N = 1`
  the `in01` gate is false and the contribution vanishes. Combined with
  `S_p_prime`/`S_ff_prime` going to zero at `ψ_N = 1`, the integrand is
  essentially zero on the separatrix.
- *Model 1 (switch_beta) and Model 3 (switch_ff), rel diff ~0.5–2 %.* Both
  multiply by `model->f_bar_prime` / `S_ff_prime`, which go smoothly to
  zero at `ψ_N = 1`. Boundary integrand is small but non-zero.
- *Model 2 (Taylor), rel diff 4–11 %.* The Taylor term
  `switch_taylor * α * (-f_x + α * (ψ_x - ψ_val))` is **not** gated by
  `ψ_N ∈ (0,1)` and does **not** vanish at the separatrix — at `ψ = ψ_x`
  it evaluates to `-α · f_x`. BFS integrates this non-vanishing boundary
  value over whole boundary elements; cut-cell integrates it only over the
  actual `{ψ < ψ_x}` sub-region. The difference is large precisely because
  the integrand is largest where the two methods disagree.

This is the cleanest empirical confirmation that "the surface-Gateaux term
of Eq. 3.10 matters when the integrand does not vanish on the separatrix" —
exactly the prediction that motivated the next step (Gateaux cut-cell
diagnostic, plan section below). Taylor is the model where that prediction
has the most bite. Models 1, 3, 4 give that diagnostic a clean near-zero
baseline to calibrate against.

### Controlled-refinement re-runs (`refinement_factor=1`, `max_amr_levels=1`)

To remove the refinement confound noted above, all 8 runs were re-executed
with **identical** mesh-refinement settings (`refinement_factor=1`,
`max_amr_levels=1`). The starting meshes still differ by ~3× in VSize
(quad 5921 vs tri 1850), so this is "same refinement knob settings", not
"matched effective h" — but it isolates model-effect within mesh-type much
more cleanly than the previous as-shipped scripts did.

| run | mesh | VSize (pre→post AMR) | BFS I_p | cut-cell I_p | rel diff | runtime | clean convergence? |
|---|---|---|---|---|---|---|---|
| run_1 ffp (m3) | tri  | 1850 → 2078 | 1.500e7 | 1.484e7 | **1.07 %**            | 19 s  | yes |
| run_1 ffp      | quad | 5921 → 6524 | 1.501e7 | 1.476e7 | 1.66 % (osc.)         | 518 s | **no — X-point oscillation** |
| run_2 fpol (m1)| tri  | 1850 → 2066 | 1.500e7 | 1.484e7 | **1.06 %**            | 64 s  | yes |
| run_2 fpol     | quad | 5921 → 6470 | 1.505e7 | 1.482e7 | 1.53 % (osc.)         | 262 s | **no — X-point oscillation** |
| run_3 taylor(m2)| tri | 1850 → 2041 | 1.500e7 | 1.256e7 | **16.27 %**           | 16 s  | yes |
| run_3 taylor   | quad | 5921 → 6179 | 1.500e7 | 1.338e7 | **10.77 %**           | 47 s  | yes |
| run_4 lb (m4)  | tri  | 1850 → 2108 | 1.500e7 | 1.499e7 | **4.36e-4**           | 17 s  | yes |
| run_4 lb       | quad | 5921 → 6204 | 1.504e7 | 1.504e7 | 7e-5 (osc.); 1.78e-4 (pre-AMR) | 149 s | **no — X-point oscillation** |

**Key takeaways from controlled refinement:**

1. **Four-OoM model ordering is preserved on the tri mesh at matched
   VSize** (~2050): Model 4 `4.36e-4` ≪ Models 1, 3 `~1.06 %` ≪ Model 2
   `16.27 %`. This is now a clean cross-model comparison at controlled
   refinement on a single geometry, which the earlier as-shipped runs
   could not provide.

2. **Taylor tri jumped from 4.16 % → 16.27 %** because the controlled
   re-run uses only 1 AMR pass (VSize 2041) instead of the 4 in the
   original `run_3_taylor.sh` (VSize 19248). The new point sits exactly on
   the earlier h-refinement curve (`16.26 % @ 1850` → `4.16 % @ 19248`),
   which strongly supports the "cut-cell − BFS = O(h) staircase error"
   reading.

3. **New finding: a BFS-detector X-point oscillation on the quad mesh for
   non-Taylor models.** Post-AMR, the solver finds two close saddles
   (e.g. `val=10.62` and `val=10.71` for model 3) and flips between them
   every Newton iteration; ψ_x, α, and BFS I_p oscillate in lock-step; the
   Newton residual stalls at ~10⁻² and the loop hits `max_newton_iter`
   instead of converging. **This is a BFS-detector issue, not a cut-cell
   issue** — the cut-cell value tracks BFS reliably through the
   oscillation. Model 2 (Taylor) quad converges cleanly because its
   un-gated `switch_taylor` term sharpens the X-point's local profile and
   the saddle selection is unambiguous. Worth flagging for Option D: when
   cut-cell starts driving the residual, the X-point sensitivity will be
   different and may smooth this out.

4. **Cross-mesh-within-model is now cleaner for the converged cases.** For
   Model 2 (the only model where both mesh types converged): tri rel diff
   `16.27 %` at VSize 2041 vs quad `10.77 %` at VSize 6179. Quad has ~3×
   more DOFs (1.75× smaller h) and ~2/3 the rel diff, consistent with
   `O(h)` staircase. For Model 4, both meshes give rel diff well under
   0.1 %, which is what the integrand-vanishes-at-separatrix argument
   predicts.

5. **The Option-B calibration baseline is reinforced.** Model 4 stays in
   the `10⁻⁴` range at controlled refinement, so the upcoming cut-cell
   Jacobian diagnostic should print a near-zero surface-Gateaux
   contribution on Model 4 (right-sign sanity check). Model 2 (Taylor) at
   `16 %` at coarse h is the positive-signal model: the cut-surface term
   should show large agreement with the BFS-vs-cut-cell volume diff there.

## Next step: Gateaux cut-cell diagnostic — execution plan

The Option A multi-model results above motivate extending the I_p
diagnostic pattern to the Jacobian. The plan recorded here mirrors the
working version in
`~/.claude/plans/this-codebase-implements-an-shimmying-unicorn.md`.

### Goal

Build a cut-cell `B_y` alongside the existing BFS `B_y` in
`SysOperator::NonlinearEquationRes`. Print Frobenius and max-abs
differences each Newton iteration. **Do not use the cut-cell value in the
solve.** Split the reported diff into "volume-only" and
"volume + cut-surface" pieces so the new Eq. 3.10 boundary-motion term is
isolated. The cut-volume code produced here is exactly what the eventual
solver-replacement step will reuse.

### The math: what the cut-surface term is

`R = diff_op·ψ − coil_term − plasma_term` and
`plasma_term = ∫_{Ω_p(ψ)} g·v dr dz`. With `Ω_p(ψ) = { ψ < ψ_x(ψ) }`, the
Gateaux semiderivative is

```
δ(plasma_term)[u, v] = ∫_{Ω_p} (∂g/∂ψ) · u · v dr dz                   (volume)
                    + ∫_{∂Ω_p} g · v · (u[ind_x] − u) / |∇ψ| ds         (surface)
```

The volume term is what `NonlinearGridCoefficient` options 2, 3, 4 already
implement (over the BFS whole-element region). The surface term is what
BFS implicitly drops to zero. Since `By = diff_op − ∂plasma_term/∂ψ`, the
surface contribution to `By` is:

- `By` bilinear: `+ ∫_{∂Ω_p} (g / |∇ψ|) · φ_j · φ_i ds`
- `By` column at `ind_x`: `− ∫_{∂Ω_p} g · φ_i / |∇ψ| ds` (per row i)
- (no surface contribution at `ind_ma` — the level set is `ψ = ψ_x`, not
  `ψ = ψ_ma`)

### New module

- `miniapps/tds-gs/gateaux_cutcell.hpp` — public API:
  ```cpp
  void compute_jacobian_diff_cutcell(const GridFunction &psi,
                                     double psi_x, double psi_ma,
                                     int ind_ma, int ind_x,
                                     PlasmaModelBase *model,
                                     FiniteElementSpace *fespace,
                                     int attr_lim,
                                     const std::set<int> &plasma_inds,
                                     const SparseMatrix &option2_mat_bfs,
                                     const Vector &option3_vec_bfs,
                                     const Vector &option4_vec_bfs,
                                     double &fro_diff_vol_only,
                                     double &fro_diff_full,
                                     double &maxabs_diff_full,
                                     double &fro_plasma_source_bfs,
                                     bool &ok,
                                     int int_order = -1,
                                     int ls_order  = 2);
  ```
- `miniapps/tds-gs/gateaux_cutcell.cpp` containing:
  - **Coefficient ports** (all gate-free, cross-referenced to
    `plasma_model.cpp` like `PlasmaSourceIntegrand`):
      * `GateauxOpt2Coef` — option==2 (the `psi_N_multiplier × 1/(psi_bdp -
        psi_max) - switch_taylor × α²/ri` block at plasma_model.cpp:333).
      * `GateauxOpt3Coef` — option==3 (ψ_ma column integrand).
      * `GateauxOpt4Coef` — option==4 (ψ_x column integrand).
      * `GBoundaryCoef` — `g(ψ_val=ψ_x, r) / |∇ψ|` for the cut-surface
        term. Reuses the option==1 formula at `ψ_N = 1`; obtains `|∇ψ|`
        via `psi.GetGradient(T, grad)`; clamps `|∇ψ|` by `max(·, ε·h_local)`
        to keep the X-point regular.
  - **Per-element driver**: get `vir` (cut-volume) and `sir` (cut-surface)
    via `MomentFittingIntRules`; assemble local element matrix
    contributions for the four pieces (option-2 bilinear, option-3/4
    columns, surface bilinear, surface ind_x column); accumulate into a
    global `SparseMatrix`.
  - **Sign convention** (mirrors the existing BFS assembly:
    `By = diff_op − option2_matrix − option3_at_ind_ma_col −
    option4_at_ind_x_col`):
      * option-2 cut-volume bilinear: enters with `−`
      * option-3 cut-volume column at `ind_ma`: enters with `−`
      * option-4 cut-volume column at `ind_x`:  enters with `−`
      * **new** cut-surface bilinear: enters with `+`
      * **new** cut-surface column at `ind_x`:  enters with `−`
  - **Surface-rule weight convention** (verified against ex38's
    `SurfaceLFIntegrator` + `SIntegrationRule`): per quadrature point of
    the cut-surface rule, the total physical-surface measure factor is
    `sir.IntPoint(ip).weight * surf_w[ip] * T.Weight()`, where
    `surf_w` is the vector returned by
    `mf_ir.GetSurfaceWeights(T, sir, surf_w)` (the surface Jacobian
    factor). The cut-volume rule does NOT need a separate `surf_w` step —
    its `ip.weight * T.Weight()` convention is the same as the I_p driver.

### Simplification: skip diff_op and Dirichlet from the diff entirely

`By_bfs` and any cut-cell `By_cut` share the same `diff_op` and the same
Dirichlet row elimination. Their **difference** is therefore independent of
both — it depends only on the plasma-source contributions. The diagnostic
can just compute the difference of the plasma-source matrices directly
(option-2 mass-like, options 3/4 columns at ind_ma/ind_x, plus the surface
terms), saving the boundary-vdofs / diff_op plumbing.

### Hook in `NonlinearEquationRes`

After the existing `By` is built, invoke the diagnostic and print:

```
[B_y comparison] BFS plasma-source Fro : <fro_plasma_source_bfs>
[B_y comparison] cut-cell diff (vol)   : <fro_diff_vol_only> (relative <…>)
[B_y comparison] cut-cell diff (vol+s) : <fro_diff_full>     (relative <…>)
[B_y comparison] cut-cell max-abs diff : <maxabs_diff_full>
```

Free the cut-cell matrices before returning — the solver continues with
the BFS `By` unchanged.

### Verification

1. Build clean.
2. Re-run `run_3_taylor_quad_mesh.sh` (Model 2 — expect surface term to be
   non-zero) and a Model-4 run (expect near-zero surface contribution as
   the natural calibration check).
3. **Interpretation matrix:**
   * `fro_diff_vol_only` small and `fro_diff_full` small → BFS Jacobian
     already accurate.
   * `fro_diff_vol_only` large, `fro_diff_full` ≪ `fro_diff_vol_only` → the
     cut-surface term reconciles BFS and cut-cell; full solver replacement
     needs it.
   * `fro_diff_full` ≳ `fro_diff_vol_only` → likely sign error in the
     cut-surface contribution; revisit the derivation above.
4. Refinement check: rerun with finer mesh / more AMR levels; both diffs
   should shrink with h (mirroring the I_p convergence study).

### Risks

- `|∇ψ| → 0` at the X-point. Mitigation: clamp `|∇ψ|` from below by
  `ε·h_local`.
- Sign conventions on the surface contribution — verify empirically via
  the interpretation matrix above.
- Model-dependent surface integrand: `g(ψ=ψ_x, r)` vanishes for models
  where the source function vanishes at `ψ_N = 1`. Model 2 (Taylor) is the
  natural positive test; Models 1, 3, 4 are the negative calibration.

## Option B — runtime crash diagnosis (`DenseMatrixSVD::Eval() : info = 19`)

### Symptom

First end-to-end run of the Gateaux diagnostic on `run_3_taylor.sh` (tri
mesh, VSize=1850) aborts immediately after the first
`[DEBUGGING: NonlinearEquationRes]` line — *before* any `[B_y comparison]`
line is printed. The error is:

```
DenseMatrixSVD::Eval() : info = 19
application called MPI_Abort(MPI_COMM_WORLD, 1) - process 0
```

(The trailing `No space left on device (28)` is incidental — the MPI abort
handler tried to write a stack/log to a full scratch directory after the
SVD failure had already triggered `MFEM_ABORT`.)

LAPACK's `dgesvd` info > 0 means the underlying bidiagonal QR did not
converge in the allotted iterations — almost always because the input
matrix contains uninitialized / garbage values that yield no meaningful
singular structure.

### Root cause: `MomentFittingIntRules` Order/nBasis mismatch

`MomentFittingIntRules` keeps `Order` and `nBasis` as instance state, and
the two interact through `InitVolume`'s order-juggling
(`fem/intrules_cut.cpp:263-268`):

```cpp
void MomentFittingIntRules::InitVolume(int order, ...) {
   order++;
   InitSurface(order, levelset, lsO, Tr);  // sets Order = order = original+1
                                           // and nBasis = 2*(Order+1)+Order*(Order+1)/2
                                           //            = 20 for the case below
   Order--;                                // Order back to original = 3
   ...
}
```

After `InitVolume(int_order=3, ...)`: `Order = 3`, but `nBasis = 20`
(formula evaluated when Order was 4).

Inside `GetVolumeIntegrationRule`, this is fine because the wrapper does
`Order++; GetSurfaceIntegrationRule(...); Order--` — the surface code runs
with Order=4 matching nBasis=20.

But the Gateaux diagnostic calls `GetSurfaceIntegrationRule(T, sir)`
**directly** on the same instance, *after* `GetVolumeIntegrationRule` has
returned. At that point Order is back to 3, but `nBasis` is still 20.
`ComputeSurfaceWeights2D` then calls `DivFreeBasis2D` which does
`shape.SetSize(nBasis=20, 2)` and fills only `2*(Order+1) + Order*(Order+1)/2
= 14` rows. The last 6 rows of `shape` are uninitialised heap memory →
`OrthoBasis2D`'s Gram-Schmidt operates on garbage → the moment-fitting
SVD matrix is ill-conditioned / NaN → `dgesvd` info=19 → `MFEM_ABORT`.

The existing cut-cell `I_p` diagnostic never tripped this because it only
calls `GetVolumeIntegrationRule`; the surface code is always reached via
the wrapper's Order++ envelope with the matched Order=4 / nBasis=20 pair.

### Fix: use two `MomentFittingIntRules` instances

In `compute_jacobian_diff_cutcell` (`gateaux_cutcell.cpp`), replace the
single `mf_ir` instance with two:

```cpp
MomentFittingIntRules mf_ir_vol(int_order, phi, ls_order);   // GetVolumeIntegrationRule only
MomentFittingIntRules mf_ir_surf(int_order, phi, ls_order);  // GetSurfaceIntegrationRule + GetSurfaceWeights only
```

Each instance owns its own Order/nBasis state.
`mf_ir_vol.GetVolumeIntegrationRule(T, vir)` initialises and uses
`(Order=3, nBasis=20)` consistently (the Order++ wrapper still applies
inside that call). `mf_ir_surf.GetSurfaceIntegrationRule(T, sir)` on a
fresh instance triggers `InitSurface(Order=3, ...)` directly, which sets
`nBasis = 2*4 + 3*4/2 = 14` to match Order=3. No mismatch.
`mf_ir_surf.GetSurfaceWeights(T, sir, surf_w)` then runs on the same
surface-initialised instance.

This is a small, localised change to one file and does not require
re-touching `fem/intrules_cut.cpp`.

### Optional follow-up

The Order/nBasis mismatch in `MomentFittingIntRules` is a library design
hazard — any external code that calls `GetSurfaceIntegrationRule` directly
on an instance previously used for volume work will hit the same crash. A
clean library fix would be to either:
(a) re-init `Order` and `nBasis` consistently inside
   `GetSurfaceIntegrationRule` when the cached state doesn't match the
   current Order, or
(b) record `(Order, nBasis)` as a pair invariant and detect mismatch with
   a clear error message.

Recorded here for later; not part of the immediate fix.

### Verification (planned, not yet executed)

1. Apply the two-instance edit to `gateaux_cutcell.cpp`.
2. Rebuild the miniapp (`make` in `miniapps/tds-gs/`; no library rebuild
   needed).
3. Re-run `run_3_taylor.sh`. The diagnostic should print the four
   `[B_y comparison]` lines per Newton iteration. Expected: large diffs on
   Taylor; the vol+surf diff should be smaller than the vol-only diff if
   the surface-Gateaux term is doing genuine work.
4. Re-run a Model 4 case (`run_4_lb.sh`) as the calibration baseline.
   Both diffs should be near zero (mirroring the I_p result).
5. The HPC `No space left on device (28)` warning is a node-level cleanup
   issue independent of the diagnostic; re-running after node cleanup
   should not surface that message.
