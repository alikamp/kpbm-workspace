# KPBM -- A Field-Local Shear-Triggered Relaxation-Time Stabilizer for SRT-BGK LBM

A 3D Lattice Boltzmann (D3Q19) fluid solver featuring the Kinetic-Pressure
Ballooning Model (KPBM): a field-local collision stabilizer that extends the
numerical stability of Single-Relaxation-Time (SRT) BGK regimes.

Written in Python with Numba compilation (`@njit`, `prange`, `fastmath`) for
CPU acceleration on transient 3D wake simulations. No GPU required.

**Author:** Alika M. Parks -- Independent Researcher -- alikamp@gmail.com
**License:** MIT

---

## Why a workstation solver

KPBM's point is accessibility. By keeping SRT-BGK stable where it would
otherwise diverge, it lets coarse-grid, higher-Re transient 3D runs complete on
a single CPU node -- no GPU, no cluster.

Paired with a conservation-preserving compression layer
([conserved-field-compression](https://github.com/alikamp/conserved-field-compression)),
which cuts the distribution-function memory traffic a bandwidth-bound LBM is
limited by, the target is *resolved-enough CFD on commodity hardware*.

The lever there is memory capacity, not FLOPs: compressing the distribution
functions lets a grid that would not otherwise fit in RAM run on a single node.
Any speedup figure depends entirely on the baseline chosen -- a run that fits in
memory, one that swaps, or a cluster -- and on holding accuracy equal between
the two sides. No such figure is quoted here; see Open Questions Section 4 for the
experiment that would produce one.

## Technical Overview

SRT-BGK LBM becomes numerically unstable at high Reynolds number, where
localized high-shear velocity gradients drive the distribution functions out of
range. The established fixes are Multi-Relaxation-Time (MRT) transforms,
cumulant operators, or selective/artificial-viscosity filters.

KPBM is a field-local modification of the relaxation time $\tau$:

$$\tau_{\text{local}} = \tau_0 \left(1 + \alpha \cdot \frac{\text{shear} \cdot |\mathbf{u}|^2}{U_{\infty}^3}\right)$$

It belongs to the selective-viscosity family of stabilizers. What is specific to
it is the functional form: the product of local shear and velocity magnitude
squared, normalized by the cube of the inlet velocity.

### Key characteristics

- **Shear-and-speed-triggered:** the added dissipation concentrates where shear
  and velocity magnitude are both high -- the wake regions where SRT-BGK
  instabilities tend to originate. It activates least in calm regions.
- **Selective, not free:** because the prior scales with
  $\text{shear} \cdot |\mathbf{u}|^2$, calm laminar regions stay near baseline
  $\tau_0$, so damping is not applied globally. Where the prior *does* activate,
  it adds dissipation, and that dissipation has a measurable cost in accuracy
  (see the $\alpha$-sweep in Validation Section 2). $\alpha$ is effectively a
  dissipation dial.
- **No GPU dependency:** parallelized across CPU cores via Numba JIT, no CUDA
  toolchain -- suited to numerical prototyping without GPU hardware.

### Solver mechanics, and three corrections

The KPBM prior is unchanged from the original formulation. The surrounding
solver mechanics in `kpbm_core_3d.py` carry three fixes over the originally
deployed version, which is why drag figures here differ from any earlier ones:

1. **Force via full momentum exchange.** Halfway bounce-back transfers
   momentum $(f_{\text{in}} + f_{\text{bounced}}) \mathbf{c}_i$ to the solid;
   summed over solid links on the fluid side this is $2 f_{\text{in}}
   \mathbf{c}_i$. The original summed a single population, so its forces were
   wrong.
2. **Streaming into a separate buffer.** Streaming writes to `f_stream` and
   collision reads from it, so bounce-back and force never alias the array
   being written. The original read and wrote overlapping arrays mid-step.
3. **Force returned as a coefficient.** `run_sphere()` normalizes to
   $C_d = 2F / (\rho U^2 A)$ with $A = \pi (D/2)^2$, rather than reporting raw
   lattice units.

## Validation

### 1. Base solver accuracy -- sphere drag at Re = 300

With KPBM inactive ($\alpha = 0$), the momentum-exchange force formulation
converges toward the reference literature value ($C_d \approx 0.65$) as the
sphere is better resolved:

| Grid resolution | $C_d$ ($\alpha = 0$) | Trend |
|---|---|---|
| 9 cells / D  | 1.08 | coarse baseline |
| 15 cells / D | 0.83 | monotonic approach to reference |

The residual gap is geometry discretization on coarse grids, not solver error.
The base physics and force calculation check out.

### 2. Dissipation tuning -- $\alpha$-sweep at Re = 300

Response of the KPBM prior at fixed resolution (9 cells / D), everything held
constant except $\alpha$:

| $\alpha$ | Mean $C_d$ | $C_d$ std. dev. | Effect |
|---|---|---|---|
| 0.00 | 1.078 | 0.028 | unmodified SRT-BGK baseline |
| 0.10 | 1.025 | 0.022 | added dissipation |
| 0.25 | 0.979 | 0.018 | more dissipation |
| 0.50 | 0.937 | 0.016 | strong dissipation |

Drag and fluctuation amplitude both decrease monotonically with $\alpha$. This
is the signature of a viscosity-based stabilizer: the prior stabilizes by adding
dissipation, and that dissipation lowers drag even here at Re = 300, where the
flow is stable at $\alpha = 0$ and needs no stabilizing.

So this sweep measures the prior's *cost* on a flow that did not require it  --
not its benefit on one that would otherwise diverge. For a case that does
require it, see Section 3.

### 3. KPBM under in-loop compression

The clearest case for the prior so far is compressed-state LBM, where the
compressor's error acts as a perturbation injected every timestep.

**2D D2Q9 cylinder, Re = 10** (stable BGK baseline): uncorrected in-loop
compression always diverges. A per-node mass and momentum restore keeps it
stable, tracking the uncompressed solution to ~3% at modest compression.

**3D D3Q19 sphere, Re = 200** (ZFP, in-loop): the per-node restore alone only
delays the crash at loose tolerances. With KPBM active:

| ZFP tolerance | $\alpha = 0$ (plain SRT-BGK) | KPBM $\alpha = 0.5$ |
|---|---|---|
| 3e-4 | diverges at step 2106 | stable, 3.74% from uncompressed run |
| 1e-3 | diverges at step 883  | stable, 4.86% from uncompressed run |

Why the two fit: compression error concentrates where gradients are steep, and
the KPBM prior activates on $\text{shear} \cdot |\mathbf{u}|^2$ -- so the added
dissipation lands in the same cells where the injected error does.

**The control, which bounds the claim.** A uniform viscosity increase of the
same *average* magnitude also rescues both cases, at 4.46% error. So KPBM is
not uniquely necessary here -- it is the more accurate way to spend the same
dissipation budget, placing it selectively rather than globally. That is a
narrower claim than "required," and it is the one the data supports.

Note the reference in this section is an uncompressed run of this same solver,
not a literature value. It is a self-consistency result, not an accuracy
validation.

## Open questions

This repo is a testbed for the KPBM prior. The experiments that would turn it
from a working stabilizer into a characterized method:

1. **High-Re survival (Re >= 1500).** Find grid/Re configurations where pure
   SRT-BGK ($\alpha = 0$) diverges, then show $\alpha > 0$ both survives *and*
   lands on a literature $C_d/C_l$. That is the line between "doesn't crash"
   and "correct," and it is the regime KPBM is meant for. Still open -- Section 3 shows
   survival where BGK diverges, but the instability there is injected
   compressor error at Re = 200, not Reynolds number, and it is measured
   against this solver's own uncompressed run.
2. **Accuracy penalty vs. MRT.** Compare KPBM against a standard MRT closure at
   matched stability -- Strouhal consistency ($St \approx 0.20$) and drag -- to
   quantify how much accuracy each method gives up. Still open. The control in
   Section 3 was uniform viscosity, which KPBM beats; MRT is the stronger baseline and
   the one that settles whether the field-local form earns its complexity.
3. **High-resolution $\alpha$-sweep.** Repeat Section 2 at 20-30 cells / D to separate
   grid artifacts from the physical dissipation trend.
4. **Accuracy-matched cost of the compressed path.** Section 3 shows the compressed run
   is cheaper and 3.7-4.9% less accurate. A speedup figure needs both sides at
   equal accuracy -- a finer grid or tighter tolerance on the compressed side,
   timed against an uncompressed baseline that fits in memory. Until that is
   run, the claim is capacity (a grid that fits that otherwise would not), not
   speed.

## Project structure

```
kpbm_core_3d.py        # D3Q19 LBM kernel, KPBM field-local tau, sphere driver
kpbm_validation_3d.py  # validation harness: sphere drag vs literature, alpha-sweep
validated_lbm.py       # 2D D2Q9 cylinder baseline, Strouhal via FFT
LICENSE                # MIT
```

## Getting started

```bash
pip install numpy numba scipy matplotlib
```

**The validation harness -- this is what produced the tables in Validation
Sections 1 and 2:**

```bash
python kpbm_validation_3d.py
```

Runs a sphere at Re = 300 with $\alpha = 0$ and reports $C_d$ against the
literature value ($C_d \approx 0.65$, Roos & Willmarth 1971), then sweeps
$\alpha$. Expect a resolution offset on coarse grids -- see Section 1.

**A single quick check of the kernel alone:**

```bash
python kpbm_core_3d.py
```

Sphere at Re = 300, $\alpha = 0$ on a 120x60x60 lattice for 2000 steps.

**2D cylinder baseline:**

```bash
python validated_lbm.py
```

Frozen geometry (D = 40, 800x400 domain, 10% blockage), reporting drag and
Strouhal number from an FFT of the lift history.

To drive KPBM yourself, call `run_sphere()` from `kpbm_core_3d.py` with
`kpbm_alpha > 0`:

```python
from kpbm_core_3d import run_sphere

r = run_sphere(Re=300, kpbm_alpha=0.5,
               Lx=120, Ly=60, Lz=60, U_inf=0.05, N_steps=2000)
print(r["Cd"], r["Cd_std"], r["status"])
```

`run_sphere()` returns time-averaged $C_d$ and $C_l$ with standard deviations
plus the full drag and lift histories, and takes an optional outflow sponge
(`use_sponge`, `sponge_thickness`, `sponge_max`) and an `update_callback` for
progress reporting.

## If you build on this

Free to take under MIT. If you run the high-Re validation or the MRT comparison
and learn something either way, I'd like to hear how it went.
