#!/usr/bin/env python3
"""
Validated D2Q9 LBM — Cylinder Flow
====================================

Based on the canonical formulation from:
  Krüger et al. "The Lattice Boltzmann Method" (Springer, 2017)

Key fixes from previous attempts:
1. Correct ordering: COLLIDE -> STREAM -> BOUNCE-BACK -> BC
2. Force via momentum exchange: computed AFTER streaming, using
   pre-streaming distributions stored at boundary
3. Proper Zou-He inlet with density correction
4. Small initial perturbation to seed wake instability

Frozen geometry: D=40, domain 800x400, blockage 10%

Usage on Colab:
  !python validated_lbm.py

Author: Alika M. Parks- transfered from orig kpbm repo on 9-19-26
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks
from time import perf_counter
import json, os

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# D2Q9 LATTICE
# ══════════════════════════════════════════════════════════════════════════════

#           0  1  2  3  4  5  6  7  8
cx = np.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1])
cy = np.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1])
w  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
cs2 = 1.0 / 3.0  # speed of sound squared


# ══════════════════════════════════════════════════════════════════════════════
# FROZEN GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

D = 40
R = D // 2
NX = 800
NY = 400
CX_CYL = NX // 5      # cylinder at x = NX/5
CY_CYL = NY // 2      # centered vertically
U0 = 0.04             # reference velocity

# Build masks
obstacle = np.zeros((NY, NX), dtype=bool)
for j in range(NY):
    for i in range(NX):
        if (i - CX_CYL)**2 + (j - CY_CYL)**2 <= R**2:
            obstacle[j, i] = True

# Top/bottom walls
wall_tb = np.zeros((NY, NX), dtype=bool)
wall_tb[0, :] = True
wall_tb[-1, :] = True

all_solid = obstacle | wall_tb

print(f"Geometry: {NX}x{NY}, D={D}, blockage={D/NY:.0%}")
print(f"Cylinder at ({CX_CYL}, {CY_CYL}), R={R}")

# Precompute boundary links for force computation
# A boundary link connects a fluid node to a solid (cylinder) node
bnd_fluid_j = []
bnd_fluid_i = []
bnd_dir = []  # direction index pointing from fluid INTO solid

for j in range(NY):
    for i in range(NX):
        if obstacle[j, i]:
            continue
        if all_solid[j, i]:
            continue
        for k in range(1, 9):
            jn = j + cy[k]
            in_ = i + cx[k]
            if 0 <= jn < NY and 0 <= in_ < NX and obstacle[jn, in_]:
                bnd_fluid_j.append(j)
                bnd_fluid_i.append(i)
                bnd_dir.append(k)

bnd_fj = np.array(bnd_fluid_j)
bnd_fi = np.array(bnd_fluid_i)
bnd_dk = np.array(bnd_dir)
bnd_ok = opp[bnd_dk]

print(f"Boundary links: {len(bnd_fj)}")


def equilibrium(rho, ux, uy):
    """Compute equilibrium distribution for all 9 directions."""
    feq = np.zeros((9, NY, NX))
    usq = ux * ux + uy * uy
    for i in range(9):
        cu = cx[i] * ux + cy[i] * uy
        feq[i] = w[i] * rho * (1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * usq / cs2)
    return feq


def run_case(Re, N_steps, label=""):
    """Run cylinder flow at given Re. Returns physics report."""
    
    # Derive tau
    nu = U0 * D / Re
    tau = nu / cs2 + 0.5
    omega_lbm = 1.0 / tau
    
    print(f"\n{'='*60}")
    print(f"[{label}] Re={Re}, tau={tau:.4f}, nu={nu:.6f}, omega={omega_lbm:.4f}")
    print(f"  {NX}x{NY}, {N_steps} steps")
    print(f"{'='*60}")
    
    # Initialize: equilibrium at uniform flow + small perturbation
    rho = np.ones((NY, NX))
    ux = np.ones((NY, NX)) * U0
    uy = np.zeros((NY, NX))
    
    # Small perturbation to break symmetry and seed instability
    np.random.seed(42)
    uy += 0.001 * U0 * np.random.randn(NY, NX)
    
    # Zero velocity in solid
    ux[all_solid] = 0
    uy[all_solid] = 0
    
    f = equilibrium(rho, ux, uy)
    
    # Inlet profile (parabolic)
    y_phys = np.arange(NY)
    u_inlet = U0 * 4.0 * y_phys * (NY - 1 - y_phys) / ((NY - 1)**2)
    u_inlet[0] = 0
    u_inlet[-1] = 0
    
    # Storage
    Cd_hist = []
    Cl_hist = []
    probe_uy = []
    
    probe_j = CY_CYL + D // 2  # 0.5D off center
    probe_i = CX_CYL + 4 * D   # 4D downstream
    if probe_i >= NX:
        probe_i = int(0.7 * NX)
    
    transient = int(0.3 * N_steps)
    
    t0 = perf_counter()
    
    for step in range(N_steps):
        
        # ── 1. MACROSCOPIC ────────────────────────────────────────
        rho = np.sum(f, axis=0)
        rho = np.maximum(rho, 1e-10)
        ux = np.sum(f * cx[:, None, None], axis=0) / rho
        uy = np.sum(f * cy[:, None, None], axis=0) / rho
        
        # ── 2. COLLISION (BGK) ────────────────────────────────────
        feq = equilibrium(rho, ux, uy)
        f_post = f - omega_lbm * (f - feq)  # post-collision
        
        # ── 3. STREAMING ──────────────────────────────────────────
        f_str = np.zeros_like(f_post)
        for i in range(9):
            f_str[i] = np.roll(np.roll(f_post[i], cx[i], axis=1), cy[i], axis=0)
        
        # ── 4. BOUNCE-BACK at obstacles ───────────────────────────
        # For each solid node, the incoming distribution is reflected
        for i in range(9):
            f_str[i][obstacle] = f_post[opp[i]][obstacle]
        
        # ── 5. BOUNCE-BACK at top/bottom walls ────────────────────
        for i in range(9):
            f_str[i][wall_tb] = f_post[opp[i]][wall_tb]
        
        # ── 6. INLET BC (Zou-He) ──────────────────────────────────
        # Known: ux = u_inlet, uy = 0
        # Unknown: rho, f[1], f[5], f[8]
        rho_in = (1.0 / (1.0 - u_inlet)) * (
            f_str[0, :, 0] + f_str[2, :, 0] + f_str[4, :, 0] +
            2.0 * (f_str[3, :, 0] + f_str[6, :, 0] + f_str[7, :, 0])
        )
        
        f_str[1, :, 0] = f_str[3, :, 0] + (2.0/3.0) * rho_in * u_inlet
        f_str[5, :, 0] = f_str[7, :, 0] + (1.0/6.0) * rho_in * u_inlet \
                          + 0.5 * (f_str[4, :, 0] - f_str[2, :, 0])
        f_str[8, :, 0] = f_str[6, :, 0] + (1.0/6.0) * rho_in * u_inlet \
                          - 0.5 * (f_str[4, :, 0] - f_str[2, :, 0])
        
        # ── 7. OUTLET BC (zero gradient) ──────────────────────────
        f_str[:, :, -1] = f_str[:, :, -2]
        
        # ── 8. UPDATE ─────────────────────────────────────────────
        f = f_str
        
        # ── 9. MEASUREMENTS ───────────────────────────────────────
        if step >= transient and step % 5 == 0:
            # Force via momentum exchange
            # F = sum_links (f_post[k] + f_str[opp[k]]) * c[k]
            # where k points from fluid into solid
            dp = f_post[bnd_dk, bnd_fj, bnd_fi] + f_str[bnd_ok, bnd_fj, bnd_fi]
            Fx = float(np.sum(cx[bnd_dk] * dp))
            Fy = float(np.sum(cy[bnd_dk] * dp))
            
            Cd = 2.0 * Fx / (rho.mean() * U0**2 * D)
            Cl = 2.0 * Fy / (rho.mean() * U0**2 * D)
            Cd_hist.append(Cd)
            Cl_hist.append(Cl)
        
        if step % 2 == 0:
            probe_uy.append(uy[probe_j, probe_i])
        
        if step % 3000 == 0:
            max_u = np.sqrt(ux**2 + uy**2).max()
            cd_now = Cd_hist[-1] if Cd_hist else 0
            cl_now = Cl_hist[-1] if Cl_hist else 0
            print(f"  step {step:>6}/{N_steps}, |u|_max={max_u:.5f}, "
                  f"Cd={cd_now:.3f}, Cl={cl_now:.4f}")
    
    elapsed = perf_counter() - t0
    
    # ── EXTRACT OBSERVABLES ───────────────────────────────────────────
    Cd_arr = np.array(Cd_hist)
    Cl_arr = np.array(Cl_hist)
    
    # Steady-state: last 50%
    if len(Cd_arr) > 20:
        ss = len(Cd_arr) // 2
        Cd_ss = Cd_arr[ss:]
        Cl_ss = Cl_arr[ss:]
    else:
        Cd_ss = Cd_arr
        Cl_ss = Cl_arr
    
    Cd_mean = float(np.mean(Cd_ss)) if len(Cd_ss) > 0 else 0
    Cd_std = float(np.std(Cd_ss)) if len(Cd_ss) > 0 else 0
    Cl_amp = float(np.std(Cl_ss) * np.sqrt(2)) if len(Cl_ss) > 0 else 0
    
    # Strouhal from probe
    probe = np.array(probe_uy)
    St = 0.0
    shedding = False
    
    if len(probe) > 200:
        p = probe[len(probe)//2:]  # second half
        p = p - p.mean()
        
        if np.std(p) > 1e-7:
            win = np.hanning(len(p))
            sig = p * win
            N_fft = len(sig)
            dt_s = 2
            freqs = np.arange(N_fft//2) / (N_fft * dt_s)
            spectrum = np.abs(fft(sig))[:N_fft//2]
            
            skip = max(5, N_fft // 50)
            peaks, props = find_peaks(spectrum[skip:],
                                      height=0.15 * np.max(spectrum[skip:]),
                                      distance=N_fft // 20)
            if len(peaks) > 0:
                best = peaks[np.argmax(props['peak_heights'])] + skip
                St = float(freqs[best] * D / U0)
                if 0.05 < St < 0.5:
                    shedding = True
                else:
                    St = 0
    
    # Vorticity
    omega_field = np.zeros((NY, NX))
    omega_field[1:-1, 1:-1] = (
        (uy[1:-1, 2:] - uy[1:-1, :-2]) / 2 -
        (ux[2:, 1:-1] - ux[:-2, 1:-1]) / 2
    )
    omega_field[all_solid] = np.nan
    
    print(f"\n  RESULT: Cd={Cd_mean:.4f}±{Cd_std:.4f}, St={St:.4f}, "
          f"shedding={shedding}, Cl_amp={Cl_amp:.4f}, time={elapsed:.1f}s")
    
    return {
        "Re": Re, "tau": round(tau, 4), "nu": round(nu, 6),
        "Cd": round(Cd_mean, 4), "Cd_std": round(Cd_std, 4),
        "St": round(St, 4), "Cl_amp": round(Cl_amp, 4),
        "shedding": shedding, "elapsed": round(elapsed, 1),
        "steps": N_steps,
        "_omega": omega_field, "_Cd": Cd_arr, "_Cl": Cl_arr,
        "_probe": np.array(probe_uy),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUN CI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    REFS = {
        36:  {"shedding": False, "Cd": [0.9, 2.0], "St": None},
        100: {"shedding": True,  "Cd": [1.2, 1.6], "St": [0.15, 0.18]},
    }
    
    print("\n" + "#" * 60)
    print("# CFD CI — VALIDATED LBM — FROZEN GEOMETRY")
    print("#" * 60)
    
    failures = []
    all_results = {}
    
    # Stage A: Regime
    r36 = run_case(36, 6000, "Re36")
    all_results["Re36"] = r36
    if r36["shedding"]:
        failures.append(f"A1: spurious shedding at Re=36 (St={r36['St']})")
    
    r100 = run_case(100, 18000, "Re100")
    all_results["Re100"] = r100
    if not r100["shedding"]:
        failures.append("A2: no shedding at Re=100")
    
    # Stage B: Accuracy
    ref = REFS[100]
    if r100["St"] > 0 and ref["St"]:
        if not (ref["St"][0] <= r100["St"] <= ref["St"][1]):
            failures.append(f"B: St={r100['St']} outside {ref['St']}")
    if r100["Cd"] > 0:
        if not (ref["Cd"][0] <= r100["Cd"] <= ref["Cd"][1]):
            failures.append(f"B: Cd={r100['Cd']} outside {ref['Cd']}")
    
    # Report
    status = "PASS" if not failures else "FAIL"
    print(f"\n{'='*60}")
    print(f"CI STATUS: {status}")
    if failures:
        for f_msg in failures:
            print(f"  ✗ {f_msg}")
    print(f"\n  Re36:  Cd={r36['Cd']:.4f}, St={r36['St']:.4f}, shedding={r36['shedding']}")
    print(f"  Re100: Cd={r100['Cd']:.4f}, St={r100['St']:.4f}, shedding={r100['shedding']}")
    print(f"{'='*60}")
    
    # Figures
    for name, r in all_results.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 6))
        
        omega = r['_omega']
        vmax = np.nanpercentile(np.abs(omega), 95)
        if vmax > 0:
            axes[0].imshow(omega, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          aspect='auto', origin='lower')
        axes[0].set_title(f"Re={r['Re']} | Cd={r['Cd']:.3f} | St={r['St']:.3f} | "
                         f"shedding={r['shedding']}", fontweight='bold')
        axes[0].set_ylabel('y')
        
        if len(r['_Cd']) > 0:
            axes[1].plot(r['_Cd'], 'b-', lw=0.5, label='Cd')
            axes[1].set_ylabel('Cd')
            ax2 = axes[1].twinx()
            if len(r['_Cl']) > 0:
                ax2.plot(r['_Cl'], 'r-', lw=0.5, alpha=0.5, label='Cl')
                ax2.set_ylabel('Cl', color='r')
            axes[1].legend(loc='upper left')
        axes[1].set_xlabel('Sample')
        
        plt.tight_layout()
        plt.savefig(f'{OUT}/{name}.png')
        plt.close()
        print(f"  -> {OUT}/{name}.png")
    
    # Save JSON
    report = {
        "status": status,
        "failures": failures,
        "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                    for k, v in all_results.items()}
    }
    with open(f"{OUT}/ci_report.json", "w") as fout:
        json.dump(report, fout, indent=2)
    print(f"  -> {OUT}/ci_report.json")
