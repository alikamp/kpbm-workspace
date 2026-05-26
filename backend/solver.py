import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def lbm_core_3d_matrix(f, f_post_stream, cx, cy, cz, opp, w, rho, u_x, u_y, u_z,
                       obstacle, tau_0, kpbm_alpha, U_inf, use_sponge, sponge_thickness=4, sponge_max=2.0):
    Lz, Ly, Lx = f.shape[1], f.shape[2], f.shape[3]

    # 1. Streaming Stage
    for i in prange(19):
        dx, dy, dz = cx[i], cy[i], cz[i]
        for z in range(Lz):
            zp = (z - dz) % Lz
            for y in range(Ly):
                yp = (y - dy) % Ly
                for x in range(Lx):
                    xp = x - dx
                    if xp >= 0 and xp < Lx:
                        f_post_stream[i, z, y, x] = f[i, zp, yp, xp]
                    elif xp < 0:
                        cu = dx * U_inf
                        f_post_stream[i, z, y, x] = w[i] * 1.0 * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*(U_inf**2))
                    else:
                        f_post_stream[i, z, y, x] = f[i, zp, yp, Lx - 1]

    # 2. Macro Metrics & Bounce-Back Tracking
    force_x, force_y = 0.0, 0.0
    for z in prange(Lz):
        for y in range(Ly):
            for x in range(Lx):
                if obstacle[z, y, x]:
                    for i in range(19):
                        f_curr = f_post_stream[opp[i], z, y, x]
                        f[i, z, y, x] = f_curr
                        force_x += f_curr * cx[i]
                        force_y += f_curr * cy[i]
                    rho[z, y, x] = 1.0
                    u_x[z, y, x] = 0.0
                    u_y[z, y, x] = 0.0
                    u_z[z, y, x] = 0.0
                else:
                    r, ux, uy, uz = 0.0, 0.0, 0.0, 0.0
                    for i in range(19):
                        r += f_post_stream[i, z, y, x]
                        ux += f_post_stream[i, z, y, x] * cx[i]
                        uy += f_post_stream[i, z, y, x] * cy[i]
                        uz += f_post_stream[i, z, y, x] * cz[i]
                    rho[z, y, x] = r
                    u_x[z, y, x] = ux / r if r > 1e-6 else 0.0
                    u_y[z, y, x] = uy / r if r > 1e-6 else 0.0
                    u_z[z, y, x] = uz / r if r > 1e-6 else 0.0

    u_scale_factor = U_inf**3 if U_inf > 0 else 1.0

    # 3. Collision Loop with Non-Linear KPBM Prior
    for z in prange(1, Lz - 1):
        dist_z = min(z, Lz - 1 - z)
        for y in range(1, Ly - 1):
            dist_y = min(y, Ly - 1 - y)
            min_dist = min(dist_z, dist_y)

            sponge_tau = 0.0
            if use_sponge and (min_dist < sponge_thickness):
                weight = (sponge_thickness - min_dist) / sponge_thickness
                sponge_tau = sponge_max * (weight ** 2)

            for x in range(1, Lx - 1):
                if obstacle[z, y, x]:
                    continue

                dux_dy = 0.5 * (u_x[z, y+1, x] - u_x[z, y-1, x])
                dux_dz = 0.5 * (u_x[z+1, y, x] - u_x[z-1, y, x])

                shear_mag = np.sqrt(dux_dy**2 + dux_dz**2)
                u_mag_sq = u_x[z, y, x]**2 + u_y[z, y, x]**2 + u_z[z, y, x]**2

                normalized_prior = (shear_mag * u_mag_sq) / u_scale_factor
                tau_field = tau_0 * (1.0 + kpbm_alpha * normalized_prior) + sponge_tau

                r = rho[z, y, x]
                ux, uy, uz = u_x[z, y, x], u_y[z, y, x], u_z[z, y, x]

                for i in range(19):
                    cu = cx[i]*ux + cy[i]*uy + cz[i]*uz
                    feq = w[i] * r * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*u_mag_sq)
                    f[i, z, y, x] = f_post_stream[i, z, y, x] - (f_post_stream[i, z, y, x] - feq) / tau_field

    # 4. Guard-Cell Boundary Sync
    for z in prange(Lz):
        for y in range(Ly):
            if not obstacle[z, y, 0]:
                u_sq = U_inf**2
                for i in range(19):
                    if cx[i] >= 0:
                        cu = cx[i] * U_inf
                        f[i, z, y, 0] = w[i] * 1.0 * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*u_sq)
            for i in range(19):
                f[i, z, y, Lx - 1] = f[i, z, y, Lx - 2]

    return force_x, force_y


def execute_solver_task(use_sponge, kpbm_alpha, Re_target, Lx, Ly, Lz, U_inf, N_steps, update_callback=None):
    w = np.array([1/3, 1/18,1/18,1/18,1/18,1/18,1/18, 1/36,1/36,1/36,1/36,1/36,1/36, 1/36,1/36,1/36,1/36,1/36,1/36])
    cx = np.array([0,  1,-1, 0, 0, 0, 0,  1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0])
    cy = np.array([0,  0, 0, 1,-1, 0, 0,  1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1])
    cz = np.array([0,  0, 0, 0, 0, 1,-1,  0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1])
    opp = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])

    D = int(Ly * 0.15) or 1
    nu = (U_inf * D) / Re_target
    tau_0 = 3.0 * nu + 0.5

    rho = np.ones((Lz, Ly, Lx))
    u_x = np.ones((Lz, Ly, Lx)) * U_inf
    u_y, u_z = np.zeros((Lz, Ly, Lx)), np.zeros((Lz, Ly, Lx))

    cz_obj, cy_obj, cx_obj = int(Lz*0.5), int(Ly*0.5), int(Lx*0.35)
    Z, Y, X = np.ogrid[:Lz, :Ly, :Lx]
    obstacle = (X - cx_obj)**2 + (Y - cy_obj)**2 + (Z - cz_obj)**2 <= (D/2.0)**2

    f = np.zeros((19, Lz, Ly, Lx))
    f_post_stream = np.zeros((19, Lz, Ly, Lx))
    for i in range(19):
        cu = cx[i]*u_x + cy[i]*u_y + cz[i]*u_z
        f[i] = w[i] * rho * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*(u_x**2))

    drag_history = []
    lift_history = []

    for step in range(1, N_steps + 1):
        fx, fy = lbm_core_3d_matrix(f, f_post_stream, cx, cy, cz, opp, w, rho, u_x, u_y, u_z,
                                   obstacle, tau_0, kpbm_alpha, U_inf, use_sponge)

        if np.isnan(fx) or np.abs(np.max(u_x)) > (U_inf * 15.0):
            return {"status": "CRASHED", "step": step, "drag": drag_history, "lift": lift_history}

        drag_history.append(float(fx))
        lift_history.append(float(fy))

        if update_callback and step % 50 == 0:
            update_callback(step, drag_history, lift_history)

    return {"status": "STABLE", "step": N_steps, "drag": drag_history, "lift": lift_history}
