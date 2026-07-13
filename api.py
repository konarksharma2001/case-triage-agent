"""
REST API for investigation review and human approval.
"""
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import storage

app = FastAPI(title="Case Triage Agent")

@app.get("/")
def root():
    return {"message": "Case Triage AI Agent API"}

class DecisionRequest(BaseModel):
    decision: Literal["approve","reject","override"]
    reviewer: str
    override_verdict: str | None = None


@app.get("/investigations")
def investigations():

    return storage.list_pending()


@app.get("/investigations/{investigation_id}")
def investigation(investigation_id: str):

    inv = storage.get_investigation(investigation_id)

    if inv is None:
        raise HTTPException(404, "Investigation not found")

    return inv


@app.post("/investigations/{investigation_id}/decision")
def approve(investigation_id: str,
            request: DecisionRequest):
    
    if (request.decision != "override" and request.override_verdict is not None):
        raise HTTPException(400,"override_verdict can only be supplied with decision='override'",)

    try:
        return storage.record_decision(
            investigation_id,
            request.decision,
            request.reviewer,
            request.override_verdict,
        )

    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/audit")
def audit():

    return storage.get_audit_log()