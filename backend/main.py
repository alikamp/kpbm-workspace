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
    try:
        result = supabase.table("licenses").select("*").eq("license_code", license_code).execute()

        if not result.data:
            return False

        lic = result.data[0]
        now = datetime.now(timezone.utc)

        expires_raw = lic.get("expires_at", "")
        if not expires_raw:
            return False

        # Handle multiple timestamp formats from Supabase
        expires_raw = expires_raw.replace("Z", "+00:00")
        if "+" not in expires_raw and expires_raw.count("-") == 2:
            expires_raw += "+00:00"
        expires_at = datetime.fromisoformat(expires_raw)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if lic.get("status") != "ACTIVE" or now > expires_at:
            try:
                supabase.table("licenses").update({"status": "EXPIRED"}).eq("license_code", license_code).execute()
            except Exception:
                pass
            return False

        return True
    except Exception as e:
        print(f"License verification error: {e}")
        return False


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


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.get("/debug-db")
async def debug_db():
    try:
        result = supabase.table("licenses").select("*").limit(1).execute()
        return {"status": "ok", "data": result.data}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/simulate")
async def start_simulation(payload: SimulationRequest, background_tasks: BackgroundTasks):
    try:
        licensed = verify_license(payload.license_code)
        if not licensed:
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation start error: {str(e)}")
