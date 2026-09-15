[KPBM_README (1).md](https://github.com/user-attachments/files/32264483/KPBM_README.1.md)
# KPBM — A Field-Local Shear-Triggered Relaxation-Time Stabilizer for SRT-BGK LBM

A 3D Lattice Boltzmann (D3Q19) fluid solver featuring the Kinetic-Pressure
Ballooning Model (KPBM): a field-local collision stabilizer that extends the
numerical stability of Single-Relaxation-Time (SRT) BGK regimes.

Written in Python with Numba compilation (`@njit`, `prange`, `fastmath`) for
CPU acceleration on transient 3D wake simulations. No GPU required.

Author: Alika M. Parks — Independent Researcher — alikamp@gmail.com
License: MIT

## Technical Overview

SRT-BGK LBM becomes numerically unstable at high Reynolds number, where
localized high-shear velocity gradients drive the distribution functions out of
range. The established fixes are Multi-Relaxation-Time (MRT) transforms,
cumulant operators, or selective/artificial-viscosity filters.

KPBM is a field-local modification of the relaxation time $\tau$:

$$\tau_{\text{local}} = \tau_0 \left(1 + \alpha \cdot \frac{\text{shear} \cdot |\mathbf{u}|^2}{U_{\infty}^3}\right)$$

It belongs to the selective-viscosity family of stabilizers. What is specific
to it is the functional form: the product of local shear and velocity magnitude
squared, normalized by the cube of the inlet velocity.

### Key characteristics

- **Shear-and-speed-triggered:** the added dissipation concentrates where shear
  and velocity magnitude are both high — the wake regions where SRT-BGK
  instabilities tend to originate. It activates least in calm regions.
- **Selective, not free:** because the prior scales with
  $\text{shear} \cdot |\mathbf{u}|^2$, calm laminar regions stay near baseline
  $\tau_0$, so damping is not applied globally. Where the prior *does* activate,
  it adds dissipation and that dissipation has a measurable cost in accuracy
  (see the $\alpha$-sweep below). $\alpha$ is effectively a dissipation dial.
- **No GPU dependency:** parallelized across CPU cores via Numba JIT, no CUDA
  toolchain — suited to numerical prototyping without GPU hardware.

## Validation

### 1. Base solver accuracy — sphere drag at Re = 300

With KPBM inactive ($\alpha = 0$), the momentum-exchange force formulation
converges toward the reference literature value ($C_d \approx 0.65$) as the
sphere is better resolved:

| Grid resolution | $C_d$ ($\alpha=0$) | Trend |
|---|---|---|
| 9 cells / D  | 1.08 | coarse baseline |
| 15 cells / D | 0.83 | monotonic approach to reference |

The residual gap is geometry discretization on coarse grids, not solver error.
The base physics and force calculation check out.

### 2. Dissipation tuning — $\alpha$-sweep at Re = 300

Response of the KPBM prior at fixed resolution (9 cells / D), everything held
constant except $\alpha$:

| $\alpha$ | Mean $C_d$ | $C_d$ std. dev. | Effect |
|---|---|---|---|
| 0.00 | 1.078 | 0.028 | unmodified SRT-BGK baseline |
| 0.10 | 1.025 | 0.022 | added dissipation |
| 0.25 | 0.979 | 0.018 | more dissipation |
| 0.50 | 0.937 | 0.016 | strong dissipation |

Drag and fluctuation amplitude both decrease monotonically with $\alpha$. This
is the signature of a viscosity-based stabilizer: the prior stabilizes by
adding dissipation, and that dissipation lowers drag even here at Re = 300,
where the flow is stable at $\alpha = 0$ and needs no stabilizing. So this sweep
measures the prior's *cost* on a flow that did not require it — not its benefit
on one that would otherwise diverge. That benefit is an open question below.

## Open questions

This repo is a testbed for the KPBM prior. The experiments that would turn it
from a working stabilizer into a characterized method:

1. **High-Re survival (Re ≥ 1500).** Find grid/Re configurations where pure
   SRT-BGK ($\alpha = 0$) diverges, then show $\alpha > 0$ both survives *and*
   lands on a literature $C_d/C_l$. That is the line between "doesn't crash" and
   "correct," and it is the regime KPBM is meant for.
2. **Accuracy penalty vs. MRT.** Compare KPBM against a standard MRT closure at
   matched stability — Strouhal consistency ($St \approx 0.20$) and drag — to
   quantify how much accuracy each method gives up.
3. **High-resolution $\alpha$-sweep.** Repeat the sweep at 20–30 cells / D to
   separate grid artifacts from the physical dissipation trend.

## Project structure

```
kpbm_core_3d.py        # D3Q19 LBM kernel with KPBM tau field-local adjustment
kpbm_validation_3d.py  # multi-grid validation harness + alpha-sweep
validated_lbm.py       # 2D D2Q9 baseline verification solver
results/               # simulation output (JSON histories, figures)
LICENSE                # MIT
```

## Getting started

```bash
pip install numpy numba scipy matplotlib
python kpbm_validation_3d.py
```

Runs the sphere validation at Re = 300, reports $C_d$ at $\alpha = 0$ against
the reference, then sweeps $\alpha$ (set `RUN_SWEEP = True` in `main()`).
Grid configs (`SMALL` / `MEDIUM` / `VALID`) trade speed for resolution.

## If you build on this

Free to take under MIT. If you run the high-Re validation or the MRT comparison
and learn something either way, I'd like to hear how it went.
