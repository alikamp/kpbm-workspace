#!/usr/bin/env python3
"""
KPBM Core — 3D D3Q19 Lattice Boltzmann solver with field-local tau adaptation
=============================================================================

CPU solver, Numba-compiled. Incompressible viscous flow past a sphere,
returning time-resolved drag and lift.

This is the corrected core. Three fixes over the original deployed version,
all in the mechanics of the solver -- the KPBM tau-adaptation itself is
unchanged:

  1. FORCE via full momentum exchange. Halfway bounce-back transfers
     momentum (f_incoming + f_bounced)*c_i to the solid; summed over solid
     links on the fluid side this is 2*f_in*c_i. The original summed a
     single population, so its forces were wrong.

  2. STREAMING into a separate buffer. Streaming writes to f_stream and
     collision reads from it; bounce-back and force never alias the array
     being written. The original read and wrote overlapping arrays mid-step.

  3. FORCE returned as a coefficient. run_sphere() normalizes to
     Cd = 2 F / (rho U^2 A), A = pi (D/2)^2, instead of raw lattice units.

The KPBM prior is exactly as before:
    prior   = (shear * |u|^2) / U_inf^3
    tau     = tau_0 * (1 + alpha * prior) + sponge_tau

Author: Alika M. Parks
License: MIT
"""

import numpy as np
from numba import njit, prange

# --- D3Q19 lattice ---------------------------------------------------------
CX = np.array([0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0], dtype=np.float64)
CY = np.array([0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1], dtype=np.float64)
CZ = np.array([0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1], dtype=np.float64)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15], dtype=np.int64)
W = np.array([1/3, 1/18,1/18,1/18,1/18,1/18,1/18,
              1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36], dtype=np.float64)


@njit(parallel=True, fastmath=True, cache=True)
def lbm_step(f, f_stream, cx, cy, cz, opp, w, rho, ux, uy, uz,
             obstacle, tau_0, kpbm_alpha, U_inf, use_sponge,
             sponge_thickness, sponge_max):
    """One LBM step: stream -> macro -> collide(KPBM) -> bounce-back + force.

    Returns (force_x, force_y, force_z) on the obstacle in lattice units.
    """
    Q = 19
    Lz, Ly, Lx = f.shape[1], f.shape[2], f.shape[3]

    # --- 1. STREAMING into a separate buffer (no aliasing) ---------------
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
                        # inlet: equilibrium at rho=1, u = U_inf x-hat
                        cu = cx[i] * U_inf
                        f_stream[i, z, y, x] = w[i] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*U_inf*U_inf)
                    else:
                        # outlet: copy last interior column
                        f_stream[i, z, y, x] = f[i, zp, yp, Lx-1]

    # --- 2. MACROSCOPIC (fluid nodes) ------------------------------------
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

    # --- 3. COLLISION with KPBM tau field --------------------------------
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

    # --- 4. BOUNCE-BACK + MOMENTUM-EXCHANGE FORCE ------------------------
    # For each fluid node with a solid neighbour in direction i: halfway
    # bounce-back reflects the population, and the momentum handed to the
    # solid is 2 * f_in * c_i. Accumulate per-z then reduce (no races).
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
                            f_in = f[i, z, y, x]
                            f[opp[i], z, y, x] = f_in           # bounce-back
                            fxl += 2.0 * f_in * cx[i]           # momentum exchange
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


def run_sphere(Re, kpbm_alpha, Lx, Ly, Lz, U_inf, N_steps,
               use_sponge=False, sponge_thickness=4, sponge_max=2.0,
               D_frac=0.15, warmup_frac=0.5, sample_every=1,
               update_callback=None):
    """
    Flow past a sphere. Returns time-averaged Cd/Cl plus histories.

    Cd = 2 F_x / (rho U^2 A),  Cl = 2 F_y / (rho U^2 A),  A = pi (D/2)^2.
    Drag/lift are normalized coefficients, not raw lattice-unit forces.
    """
    D = max(int(Ly * D_frac), 6)
    nu = U_inf * D / Re
    tau_0 = 3.0 * nu + 0.5

    cz0, cy0, cx0 = Lz//2, Ly//2, int(Lx*0.35)
    Z, Y, X = np.ogrid[:Lz, :Ly, :Lx]
    obstacle = np.ascontiguousarray(
        ((X-cx0)**2 + (Y-cy0)**2 + (Z-cz0)**2) <= (D/2.0)**2)

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
    norm = 1.0 / (1.0 * U_inf*U_inf * area)   # 2/(...) applied below
    warmup = int(warmup_frac * N_steps)

    cd_hist = []
    cl_hist = []
    crashed = False
    crash_step = N_steps

    for step in range(1, N_steps+1):
        fx, fy, fz = lbm_step(f, f_stream, CX, CY, CZ, OPP, W,
                              rho, ux, uy, uz, obstacle,
                              tau_0, kpbm_alpha, U_inf,
                              use_sponge, sponge_thickness, sponge_max)

        if np.isnan(fx) or np.nanmax(np.abs(ux)) > U_inf*20:
            crashed = True
            crash_step = step
            break

        cd = 2.0 * fx * norm
        cl = 2.0 * fy * norm
        cd_hist.append(cd)
        cl_hist.append(cl)

        if update_callback is not None and step % 50 == 0:
            update_callback(step, cd_hist, cl_hist)

    cd_arr = np.array(cd_hist)
    cl_arr = np.array(cl_hist)
    ss = cd_arr[warmup:] if len(cd_arr) > warmup else cd_arr
    sl = cl_arr[warmup:] if len(cl_arr) > warmup else cl_arr

    return {
        "status": "CRASHED" if crashed else "STABLE",
        "crash_step": crash_step,
        "Re": Re, "alpha": kpbm_alpha, "D": D, "tau_0": tau_0,
        "grid": [Lx, Ly, Lz], "cells": Lx*Ly*Lz, "steps": N_steps,
        "Cd": float(np.mean(ss)) if len(ss) else float("nan"),
        "Cd_std": float(np.std(ss)) if len(ss) else float("nan"),
        "Cl": float(np.mean(sl)) if len(sl) else float("nan"),
        "Cl_std": float(np.std(sl)) if len(sl) else float("nan"),
        "drag": cd_hist, "lift": cl_hist,
    }


if __name__ == "__main__":
    # quick self-test: sphere at Re=300, KPBM off
    r = run_sphere(Re=300, kpbm_alpha=0.0,
                   Lx=120, Ly=60, Lz=60, U_inf=0.05, N_steps=2000)
    print(f"status={r['status']}  Cd={r['Cd']:.3f} +/- {r['Cd_std']:.3f}  "
          f"(literature Cd(Re=300) ~ 0.65; expect resolution offset at D={r['D']})")
