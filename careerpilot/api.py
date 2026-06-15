from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .service import run_careerpilot


app = FastAPI(
    title="CareerPilot-LangGraph API",
    description="A LangGraph-based career preparation agent service.",
    version="0.1.0",
)


class CareerRequest(BaseModel):
    resume_text: str
    jd_text: str
    target_role: str = "大模型/Agent 工程实习生"
    offline: bool = False


class CareerResponse(BaseModel):
    final_report: str
    match_score: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=CareerResponse)
def analyze(req: CareerRequest):
    result = run_careerpilot(
        resume_text=req.resume_text,
        jd_text=req.jd_text,
        target_role=req.target_role,
        offline=req.offline,
    )

    return CareerResponse(
        final_report=result.get("final_report", ""),
        match_score=result.get("match_report", {}).get("score"),
    )