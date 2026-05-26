import os
from datetime import datetime, timezone
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import uuid
from solver import execute_solver_task

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class SimulationRequest(BaseModel):
    license_code: str
    kpbm_alpha: float = 0.25
    use_sponge: bool = True
    Re_target: float = 1500.0
    Lx: int = 100
    Ly: int = 40
    Lz: int = 40
    U_inf: float = 0.04
    N_steps: int = 1500


def verify_license(license_code: str) -> bool:
    result = supabase.table("licenses").select("*").eq("license_code", license_code).execute()

    if not result.data:
        return False

    license = result.data[0]
    now = datetime.now(timezone.utc)
    expires_at = datetime.fromisoformat(license["expires_at"].replace("Z", "+00:00"))

    if license["status"] != "ACTIVE" or now > expires_at:
        supabase.table("licenses").update({"status": "EXPIRED"}).eq("license_code", license_code).execute()
        return False

    return True


def async_simulation_worker(job_id: str, req: SimulationRequest):
    def db_progress_logger(current_step: int, drag: list, lift: list):
        percentage = int((current_step / req.N_steps) * 100)
        supabase.table("jobs").update({
            "progress": percentage,
            "current_step": current_step,
            "drag_history": drag,
            "lift_history": lift,
            "log_stream": f"Step {current_step}/{req.N_steps} — Drag: {drag[-1]:.4f}, Lift: {lift[-1]:.4f}"
        }).eq("id", job_id).execute()

    try:
        res = execute_solver_task(
            use_sponge=req.use_sponge,
            kpbm_alpha=req.kpbm_alpha,
            Re_target=req.Re_target,
            Lx=req.Lx, Ly=req.Ly, Lz=req.Lz,
            U_inf=req.U_inf, N_steps=req.N_steps,
            update_callback=db_progress_logger
        )

        supabase.table("jobs").update({
            "status": res["status"],
            "progress": 100,
            "current_step": res["step"],
            "drag_history": res["drag"],
            "lift_history": res["lift"],
            "log_stream": f"Execution terminated — State: {res['status']} at step {res['step']}."
        }).eq("id", job_id).execute()

    except Exception as e:
        supabase.table("jobs").update({
            "status": "CRASHED",
            "log_stream": f"System error: {str(e)}"
        }).eq("id", job_id).execute()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/simulate")
async def start_simulation(payload: SimulationRequest, background_tasks: BackgroundTasks):
    if not verify_license(payload.license_code):
        raise HTTPException(status_code=403, detail="Invalid or expired license token.")

    job_id = str(uuid.uuid4())

    supabase.table("jobs").insert({
        "id": job_id,
        "status": "RUNNING",
        "progress": 0,
        "current_step": 0,
        "drag_history": [],
        "lift_history": [],
        "log_stream": "Initializing matrices and invoking parallel Numba kernel compilation..."
    }).execute()

    background_tasks.add_task(async_simulation_worker, job_id, payload)

    return {"job_id": job_id, "status": "ACCEPTED"}
