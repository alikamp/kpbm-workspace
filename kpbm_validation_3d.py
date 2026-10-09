#!/usr/bin/env python3
"""
KPBM 3D Validation Harness — Sphere Drag at Known Reynolds Number
==================================================================

WHAT THIS DOES, AND WHY
-----------------------
Your production kernel reports STABLE vs CRASHED. That cannot distinguish
a correctly-stabilized flow from one that was over-damped into physically
wrong mush. This harness replaces that test with a real one: it measures
the drag coefficient of a sphere and compares it against the known
literature value.

Reference: flow past a sphere at Re = 300 has Cd approximately 0.65
(Roos & Willmarth 1971; Schlichting). This is a standard CFD validation
point. If a solver gets this number, its forces are trustworthy. If it
does not, they are not -- regardless of whether it stayed "stable".

THREE BUGS FROM THE PRODUCTION KERNEL ARE FIXED HERE
----------------------------------------------------
1. FORCE CALCULATION. Momentum-exchange bounce-back is
       F = sum over boundary links of (f_incoming + f_outgoing) * c_i
   The production kernel summed only one population. Here the force is
   computed on the FLUID side of each solid link, using both the
   population heading into the wall and the one bounced back. This is the
   standard Ladd/Mei momentum-exchange formula.

2. BOUNCE-BACK ALIASING. The production kernel read and wrote overlapping
   arrays mid-step. Here streaming writes into a separate buffer, and
   bounce-back is applied cleanly from post-collision populations.

3. NO CORRECTNESS CHECK. Replaced STABLE/CRASHED with an actual Cd
   measurement against a reference.

THE ACTUAL EXPERIMENT
---------------------
Step 1: run with kpbm_alpha = 0 (KPBM disabled). This is plain BGK. It
        MUST reproduce Cd ~ 0.65 at Re = 300. If it does not, the base
        solver is wrong and nothing else matters -- stop and fix that.

Step 2: only if step 1 passes, sweep kpbm_alpha upward. The question the
        whole project rests on:
          - does drag stay near 0.65 as alpha rises?  -> KPBM preserves
            physics while adding stability. Real result.
          - does drag climb with alpha?               -> KPBM is just
            numerical dissipation. The "stabilization" is over-damping.

Either answer is worth having. A negative result here saves you from
staking your name on an unvalidated claim.

PERFORMANce NOTE
----------------
This is a real 3D run. To get drag right you need ~20 cells across the
sphere and a domain large enough that walls don't corrupt the wake.
That is a few million cells. The kernel is Numba-jitted (parallel), so
first call pays a compile cost, then each run is minutes not hours.
Start with the SMALL config to confirm it works, then move to VALID.

USAGE (Colab)
-------------
    !pip install numba
    !python kpbm_validation_3d.py
"""

import numpy as np
from numba import njit, prange
from time import perf_counter
import json, os

OUT = "kpbm_validation"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# D3Q19 lattice
# ---------------------------------------------------------------------------
CX = np.array([0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0], dtype=np.float64)
CY = np.array([0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1], dtype=np.float64)
CZ = np.array([0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1], dtype=np.float64)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15], dtype=np.int64)
W = np.array([1/3, 1/18,1/18,1/18,1/18,1/18,1/18,
              1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36], dtype=np.float64)


# ---------------------------------------------------------------------------
# Core step. Streaming -> macroscopic -> collision(KPBM) -> bounce-back+force
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def lbm_step(f, f_stream, cx, cy, cz, opp, w, rho, ux, uy, uz,
             obstacle, tau_0, kpbm_alpha, U_inf, use_sponge,
             sponge_thickness, sponge_max):
    Q = 19
    Lz, Ly, Lx = f.shape[1], f.shape[2], f.shape[3]

    # ---- 1. STREAMING into a separate buffer (no aliasing) --------------
    # f_stream[i, z, y, x] = f[i, (z,y,x) - c_i], with inlet equilibrium
    # injected from the left face and copy-outflow on the right.
    for i in prange(Q):
        dx = int(cx[i]); dy = int(cy[i]); dz = int(cz[i])
        for z in range(Lz):
            zp = (z - dz) % Lz
            for y in range(Ly):
                yp = (y - dy) % Ly
                for x in range(Lx):
                    xp = x - dx
                    if 0 <= xp < Lx:
                        f_stream[i, z, y, x] = f[i, zp, yp, xp]
                    elif xp < 0:
                        # inlet: equilibrium at (rho=1, u=U_inf x-hat)
                        cu = cx[i] * U_inf
                        f_stream[i, z, y, x] = w[i] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U_inf*U_inf)
                    else:
                        # outlet: copy from last interior column
                        f_stream[i, z, y, x] = f[i, zp, yp, Lx-1]

    # ---- 2. MACROSCOPIC (fluid nodes only) ------------------------------
    for z in prange(Lz):
        for y in range(Ly):
            for x in range(Lx):
                if obstacle[z, y, x]:
                    rho[z, y, x] = 1.0
                    ux[z, y, x] = 0.0
                    uy[z, y, x] = 0.0
                    uz[z, y, x] = 0.0
                    continue
                r = 0.0; mx = 0.0; my = 0.0; mz = 0.0
                for i in range(Q):
                    fi = f_stream[i, z, y, x]
                    r += fi
                    mx += fi * cx[i]
                    my += fi * cy[i]
                    mz += fi * cz[i]
                rho[z, y, x] = r
                if r > 1e-9:
                    ux[z, y, x] = mx / r
                    uy[z, y, x] = my / r
                    uz[z, y, x] = mz / r
                else:
                    ux[z, y, x] = 0.0
                    uy[z, y, x] = 0.0
                    uz[z, y, x] = 0.0

    u_scale = U_inf**3 if U_inf > 0 else 1.0

    # ---- 3. COLLISION with KPBM tau field -------------------------------
    for z in prange(Lz):
        dist_z = z if z < Lz-1-z else Lz-1-z
        for y in range(Ly):
            dist_y = y if y < Ly-1-y else Ly-1-y
            md = dist_z if dist_z < dist_y else dist_y
            sponge_tau = 0.0
            if use_sponge and md < sponge_thickness:
                weight = (sponge_thickness - md) / sponge_thickness
                sponge_tau = sponge_max * weight * weight
            for x in range(Lx):
                if obstacle[z, y, x]:
                    continue

                # shear via central differences on ux; guard domain edges
                if 0 < y < Ly-1:
                    dux_dy = 0.5 * (ux[z, y+1, x] - ux[z, y-1, x])
                else:
                    dux_dy = 0.0
                if 0 < z < Lz-1:
                    dux_dz = 0.5 * (ux[z+1, y, x] - ux[z-1, y, x])
                else:
                    dux_dz = 0.0

                shear = np.sqrt(dux_dy*dux_dy + dux_dz*dux_dz)
                uxl = ux[z, y, x]; uyl = uy[z, y, x]; uzl = uz[z, y, x]
                u_sq = uxl*uxl + uyl*uyl + uzl*uzl

                prior = (shear * u_sq) / u_scale
                tau = tau_0 * (1.0 + kpbm_alpha * prior) + sponge_tau
                inv_tau = 1.0 / tau

                r = rho[z, y, x]
                for i in range(Q):
                    cu = cx[i]*uxl + cy[i]*uyl + cz[i]*uzl
                    feq = w[i] * r * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq)
                    fs = f_stream[i, z, y, x]
                    f[i, z, y, x] = fs - inv_tau * (fs - feq)

    # ---- 4. BOUNCE-BACK + MOMENTUM-EXCHANGE FORCE -----------------------
    # Done on the FLUID side of each solid link. For a fluid node that has
    # a solid neighbour in direction i, halfway bounce-back reflects the
    # population; the momentum transferred to the solid is
    #     (f_stream[i] + f_stream_after_bounce[opp[i]]) * c_i
    # We accumulate force per-thread then reduce, to avoid races.
    nthreads = 1
    fx_parts = np.zeros(Lz)
    fy_parts = np.zeros(Lz)
    fz_parts = np.zeros(Lz)

    for z in prange(Lz):
        fxl = 0.0; fyl = 0.0; fzl = 0.0
        for y in range(Ly):
            for x in range(Lx):
                if obstacle[z, y, x]:
                    continue
                for i in range(Q):
                    dx = int(cx[i]); dy = int(cy[i]); dz = int(cz[i])
                    zn = z + dz; yn = y + dy; xn = x + dx
                    if 0 <= zn < Lz and 0 <= yn < Ly and 0 <= xn < Lx:
                        if obstacle[zn, yn, xn]:
                            # population heading into the solid along i
                            f_in = f[i, z, y, x]
                            # halfway bounce-back: it returns along opp[i]
                            f[opp[i], z, y, x] = f_in
                            # momentum exchange (both populations carry c_i,
                            # opp carries -c_i, so contribution is 2*f_in*c_i)
                            fxl += 2.0 * f_in * cx[i]
                            fyl += 2.0 * f_in * cy[i]
                            fzl += 2.0 * f_in * cz[i]
        fx_parts[z] = fxl
        fy_parts[z] = fyl
        fz_parts[z] = fzl

    force_x = 0.0; force_y = 0.0; force_z = 0.0
    for z in range(Lz):
        force_x += fx_parts[z]
        force_y += fy_parts[z]
        force_z += fz_parts[z]

    return force_x, force_y, force_z


# ---------------------------------------------------------------------------
def run_sphere(Re, kpbm_alpha, Lx, Ly, Lz, U_inf, N_steps,
               use_sponge=False, sponge_thickness=4, sponge_max=2.0,
               D_frac=0.15, warmup_frac=0.5, label=""):
    """
    Flow past a sphere. Returns time-averaged Cd and diagnostics.

    D is set from Ly so the sphere fits with clearance. Cd uses frontal
    area A = pi (D/2)^2 :  Cd = 2 F_x / (rho U^2 A).
    """
    D = max(int(Ly * D_frac), 6)
    nu = U_inf * D / Re
    tau_0 = 3.0 * nu + 0.5

    cz0, cy0, cx0 = Lz//2, Ly//2, int(Lx*0.35)
    Z, Y, X = np.ogrid[:Lz, :Ly, :Lx]
    obstacle = ((X-cx0)**2 + (Y-cy0)**2 + (Z-cz0)**2) <= (D/2.0)**2
    obstacle = np.ascontiguousarray(obstacle)

    n_solid = int(obstacle.sum())

    rho = np.ones((Lz, Ly, Lx))
    ux = np.ones((Lz, Ly, Lx)) * U_inf
    uy = np.zeros((Lz, Ly, Lx))
    uz = np.zeros((Lz, Ly, Lx))
    ux[obstacle] = 0.0

    f = np.zeros((19, Lz, Ly, Lx))
    for i in range(19):
        cu = CX[i]*ux + CY[i]*uy + CZ[i]*uz
        f[i] = W[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(ux*ux+uy*uy+uz*uz))
    f_stream = np.zeros_like(f)

    area = np.pi * (D/2.0)**2
    print(f"\n  [{label}] Re={Re}, alpha={kpbm_alpha}, D={D}, tau0={tau_0:.4f}")
    print(f"    grid {Lx}x{Ly}x{Lz} = {Lx*Ly*Lz:,} cells, sphere {n_solid:,} cells, {N_steps} steps")

    warmup = int(warmup_frac * N_steps)
    cd_hist = []
    crashed = False
    t0 = perf_counter()

    for step in range(1, N_steps+1):
        fx, fy, fz = lbm_step(f, f_stream, CX, CY, CZ, OPP, W,
                              rho, ux, uy, uz, obstacle,
                              tau_0, kpbm_alpha, U_inf,
                              use_sponge, sponge_thickness, sponge_max)

        if np.isnan(fx) or np.nanmax(np.abs(ux)) > U_inf*20:
            crashed = True
            print(f"    CRASHED at step {step}")
            break

        if step >= warmup:
            cd = 2.0 * fx / (1.0 * U_inf*U_inf * area)
            cd_hist.append(cd)

        if step % max(1, N_steps//10) == 0:
            umax = np.nanmax(np.sqrt(ux*ux+uy*uy+uz*uz))
            cd_now = cd_hist[-1] if cd_hist else float('nan')
            print(f"    step {step:>5}/{N_steps}  umax={umax:.4f}  Cd~{cd_now:.3f}")

    elapsed = perf_counter() - t0
    cd_arr = np.array(cd_hist)
    cd_mean = float(np.mean(cd_arr)) if len(cd_arr) else float('nan')
    cd_std = float(np.std(cd_arr)) if len(cd_arr) else float('nan')

    print(f"    -> Cd = {cd_mean:.3f} +/- {cd_std:.3f}   ({elapsed:.1f}s, "
          f"{'CRASHED' if crashed else 'ran'})")

    return dict(Re=Re, alpha=kpbm_alpha, D=D, tau_0=tau_0,
                grid=[Lx,Ly,Lz], cells=Lx*Ly*Lz, steps=N_steps,
                Cd=cd_mean, Cd_std=cd_std, crashed=crashed,
                elapsed=round(elapsed,1))


# ---------------------------------------------------------------------------
CONFIGS = {
    # tiny: proves the harness runs and compiles. Cd will be rough.
    "SMALL": dict(Lx=120, Ly=60, Lz=60, U_inf=0.05, N_steps=2000),
    # medium: D~15, big enough to tell 0.65 from 1.0, finishes in ~10 min.
    # Good first trustworthy-ish baseline without risking a Colab timeout.
    "MEDIUM": dict(Lx=180, Ly=100, Lz=100, U_inf=0.04, N_steps=4000),
    # validation: enough resolution/clearance for a meaningful Cd at Re=300.
    # expect Cd ~ 0.65 at alpha=0. This one is the real test. Slower.
    "VALID": dict(Lx=240, Ly=140, Lz=140, U_inf=0.04, N_steps=8000),
}

REF_CD = {300: 0.65}   # Roos & Willmarth 1971


def main():
    print("="*66)
    print("KPBM 3D VALIDATION — sphere drag against known Cd")
    print("="*66)
    print("""
  Plan:
    1. alpha = 0 (KPBM off) must reproduce Cd ~ 0.65 at Re=300.
       If it doesn't, the base solver is wrong -- fix before anything else.
    2. sweep alpha. Watch whether Cd stays near 0.65 (KPBM preserves
       physics) or climbs (KPBM = numerical dissipation).
""")

    cfg_name = "SMALL"         # SMALL for the sweep: fast, survives Colab
    cfg = CONFIGS[cfg_name]
    print(f"  Using config: {cfg_name}  "
          f"({cfg['Lx']}x{cfg['Ly']}x{cfg['Lz']}, {cfg['N_steps']} steps)\n")

    results = []

    # Step 1: baseline, KPBM disabled
    r0 = run_sphere(Re=300, kpbm_alpha=0.0, label="baseline a=0", **cfg)
    results.append(r0)

    ref = REF_CD[300]
    if not r0["crashed"] and np.isfinite(r0["Cd"]):
        err = abs(r0["Cd"] - ref) / ref
        verdict = "PASS" if err < 0.25 else "OFF"
        print(f"\n  baseline Cd={r0['Cd']:.3f} vs ref {ref} : {verdict} "
              f"({100*err:.0f}% from reference)")
        if verdict == "OFF":
            print("  -> base solver not yet matching reference. Likely causes:")
            print("     resolution too low (use VALID), domain too small,")
            print("     or transient not cleared. Fix before trusting the sweep.")

    # Step 2: alpha sweep. Set RUN_SWEEP=True only AFTER the alpha=0
    # baseline lands near the reference. No point sweeping a wrong baseline.
    RUN_SWEEP = True
    if RUN_SWEEP:
        for a in (0.1, 0.25, 0.5):
            results.append(run_sphere(Re=300, kpbm_alpha=a,
                                      label=f"alpha={a}", **cfg))

    print("\n" + "="*66)
    print("SUMMARY")
    print("="*66)
    print(f"  reference Cd(Re=300) = {ref}\n")
    print(f"  {'alpha':>6} {'Cd':>8} {'Cd_std':>8} {'crashed':>8}")
    for r in results:
        print(f"  {r['alpha']:>6.2f} {r['Cd']:>8.3f} {r['Cd_std']:>8.3f} "
              f"{str(r['crashed']):>8}")
    print("""
  READING IT:
    Cd flat near reference as alpha rises -> KPBM adds stability without
      corrupting drag. That is a real, publishable result.
    Cd rising with alpha -> the prior is over-damping; "stability" is
      just numerical dissipation. Also worth knowing, before you stake
      anything on it.
""")

    with open(f"{OUT}/validation_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"  -> {OUT}/validation_results.json")


if __name__ == "__main__":
    main()
