"""
api.py

FastAPI wrapper for the SMB risk assessment workflow.
Exposes the LangGraph workflow as a REST API so it can be called
from a frontend, MCP server, or any HTTP client.

Endpoints:
    POST /assess          → run a full risk assessment
    GET  /status/{app_id} → check status of a running assessment
    GET  /report/{app_id} → retrieve completed report
    GET  /health          → health check

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.application import (
    SMBLoanApplication,
    LoanPurpose,
    BusinessStructure,
    OwnerDemographics,
)
from models.assessment import RiskAssessmentOutput
from models.audit import AuditResult
from graph.state import RiskAssessmentState
from graph.workflow import build_dev_graph, build_prod_graph

load_dotenv()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SMB Risk Assessment Agent",
    description=(
        "Autonomous multi-agent SMB loan risk assessment system. "
        "Applies Basel III capital standards and CFPB fair lending compliance "
        "via a self-auditing critic loop."
    ),
    version="1.0.0",
)

# Allow frontend and MCP clients to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for assessment results (replace with DB in production)
assessment_store: dict = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AssessmentRequest(BaseModel):
    """
    The body of a POST /assess request.
    Accepts the full loan application data.
    """
    application: SMBLoanApplication
    use_prod_db: bool = True   # if True, uses PostgreSQL checkpointer


class AssessmentStatus(BaseModel):
    """Status response for a running or completed assessment."""
    application_id: str
    status: str                          # "pending" | "running" | "complete" | "escalated"
    submitted_at: str
    completed_at: Optional[str] = None
    risk_category: Optional[str] = None
    recommended_action: Optional[str] = None
    escalated_to_human: bool = False


class AssessmentResponse(BaseModel):
    """Full response including report, assessment, and audit."""
    application_id: str
    status: str
    final_report: Optional[str] = None
    assessment: Optional[dict] = None
    audit: Optional[dict] = None
    escalated_to_human: bool = False
    retry_count: int = 0
    message_count: int = 0


# ---------------------------------------------------------------------------
# Background task — runs the workflow asynchronously
# ---------------------------------------------------------------------------

def run_assessment(application: SMBLoanApplication, use_prod_db: bool):
    """
    Runs the full LangGraph workflow in the background.
    Stores the result in assessment_store keyed by application_id.
    """
    app_id = application.application_id

    # Mark as running
    assessment_store[app_id]["status"] = "running"

    try:
        # Build graph
        graph = build_prod_graph() if use_prod_db else build_dev_graph()

        # Initial state
        initial_state = RiskAssessmentState(application=application)

        # Config with thread_id for checkpointing, and finding in Langgraph flows
        config = {"configurable": {"thread_id": app_id}}

        # Run the workflow
        for step_update in graph.stream(initial_state, config=config):
            node_name = list(step_update.keys())[0]
            assessment_store[app_id]["last_node"] = node_name

        # Get final state
        final = graph.get_state(config).values

        # Store results
        assessment_store[app_id].update({
            "status": "escalated" if final.get("escalated_to_human") else "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "final_report": final.get("final_report"),
            "assessment": final.get("assessment").model_dump() if final.get("assessment") else None,
            "audit": final.get("audit").model_dump() if final.get("audit") else None,
            "escalated_to_human": final.get("escalated_to_human", False),
            "retry_count": final.get("retry_count", 0),
            "message_count": len(final.get("messages", [])),
        })

    except Exception as e:
        import traceback
        print(f"WORKFLOW ERROR: {str(e)}")
        print(traceback.format_exc())
        assessment_store[app_id].update({
        "status": "error",
        "error": str(e),
        "completed_at": datetime.now(timezone.utc).isoformat(),

        })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.post("/assess", response_model=AssessmentStatus)
async def assess_loan(request: AssessmentRequest, background_tasks: BackgroundTasks):
    """
    Submit a loan application for risk assessment.
    Runs the full LangGraph workflow asynchronously.
    Returns immediately with a status — poll /status/{app_id} for results.
    """
    app_id = request.application.application_id

    # Check if already running
    if app_id in assessment_store and assessment_store[app_id]["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail=f"Assessment for {app_id} is already running"
        )

    # Initialize store entry
    submitted_at = datetime.now(timezone.utc).isoformat()
    assessment_store[app_id] = {
        "application_id": app_id,
        "status": "pending",
        "submitted_at": submitted_at,
        "completed_at": None,
        "final_report": None,
        "assessment": None,
        "audit": None,
        "escalated_to_human": False,
        "retry_count": 0,
        "message_count": 0,
    }

    # Run workflow in background
    background_tasks.add_task(
        run_assessment,
        request.application,
        request.use_prod_db
    )

    return AssessmentStatus(
        application_id=app_id,
        status="pending",
        submitted_at=submitted_at,
    )


@app.get("/status/{app_id}", response_model=AssessmentStatus)
def get_status(app_id: str):
    """
    Check the status of a running or completed assessment.
    Poll this endpoint until status is 'complete' or 'escalated'.
    """
    if app_id not in assessment_store:
        raise HTTPException(
            status_code=404,
            detail=f"No assessment found for application {app_id}"
        )

    entry = assessment_store[app_id]
    assessment = entry.get("assessment")

    return AssessmentStatus(
        application_id=app_id,
        status=entry["status"],
        submitted_at=entry["submitted_at"],
        completed_at=entry.get("completed_at"),
        risk_category=assessment["risk_category"] if assessment else None,
        recommended_action=assessment["recommended_action"] if assessment else None,
        escalated_to_human=entry.get("escalated_to_human", False),
    )


@app.get("/report/{app_id}", response_model=AssessmentResponse)
def get_report(app_id: str):
    """
    Retrieve the full completed assessment report.
    Returns 404 if not found, 202 if still running.
    """
    if app_id not in assessment_store:
        raise HTTPException(
            status_code=404,
            detail=f"No assessment found for application {app_id}"
        )

    entry = assessment_store[app_id]

    if entry["status"] in ("pending", "running"):
        raise HTTPException(
            status_code=202,
            detail=f"Assessment for {app_id} is still {entry['status']}"
        )

    return AssessmentResponse(
        application_id=app_id,
        status=entry["status"],
        final_report=entry.get("final_report"),
        assessment=entry.get("assessment"),
        audit=entry.get("audit"),
        escalated_to_human=entry.get("escalated_to_human", False),
        retry_count=entry.get("retry_count", 0),
        message_count=entry.get("message_count", 0),
    )


@app.get("/scenarios")
def list_scenarios():
    """
    Returns the available test scenarios for the API playground.
    Useful for frontend dropdowns and demos.
    """
    return {
        "scenarios": [
            {
                "name": "moderate",
                "business": "Sunshine Bakery LLC",
                "description": "Moderate risk — approve with conditions",
                "application_id": "APP-2024-001",
            },
            {
                "name": "strong",
                "business": "Coastal Logistics Partners",
                "description": "Strong financials — clean approval",
                "application_id": "APP-2024-002",
            },
            {
                "name": "decline",
                "business": "QuickFix Auto Repair",
                "description": "Critical risk — decline",
                "application_id": "APP-2024-003",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
