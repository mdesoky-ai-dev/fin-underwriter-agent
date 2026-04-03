"""
agents/ingestion_agent.py

Node 1: Data Ingestion Agent
----------------------------
Receives raw application data, validates it through the Pydantic schema,
computes derived ratios (DSCR, current ratio, debt-to-revenue), and loads
the validated application into the shared graph state.

This node does NOT call an LLM — it is a deterministic data pipeline step.
Its job is to guarantee that by the time the assessment agent runs, the
data is clean, typed, and complete.

LangSmith will trace this as the first step in every workflow run.
"""

import structlog
from datetime import datetime, timezone

from graph.state import RiskAssessmentState, AgentMessage
from models.application import SMBLoanApplication

log = structlog.get_logger()


def ingestion_node(state: RiskAssessmentState) -> dict:
    """
    LangGraph node function.

    Receives the current state, returns a PARTIAL UPDATE DICT.
    LangGraph merges this dict into the state — we only return
    fields we want to change, not the entire state object.
    """
    log.info("ingestion_node.start", application_id=state.application.application_id if state.application else "unknown")

    # If application is already loaded and valid, pass through
    if state.application is None:
        error_msg = "No application data found in state. Cannot proceed."
        log.error("ingestion_node.no_application")
        return {
            "messages": [
                AgentMessage(
                    role="system",
                    content=f"ERROR: {error_msg}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            ],
            "workflow_complete": True,
        }

    app = state.application

    # Log the computed ratios so LangSmith shows them in the trace
    log.info(
        "ingestion_node.ratios_computed",
        application_id=app.application_id,
        dscr=app.dscr,
        current_ratio=app.current_ratio,
        debt_to_revenue=app.debt_to_revenue,
        requested_amount=app.requested_amount,
        annual_revenue=app.annual_revenue,
    )

    # Confirm the ECOA firewall — demographics excluded from credit fields
    credit_fields = app.credit_decision_fields()
    ecoa_firewall_confirmed = "owner_demographics" not in credit_fields

    summary = (
        f"Application {app.application_id} ingested successfully. "
        f"Business: {app.business_name} | "
        f"Requested: ${app.requested_amount:,.0f} | "
        f"DSCR: {app.dscr:.2f} | "
        f"Current ratio: {app.current_ratio:.2f} | "
        f"Debt/Revenue: {app.debt_to_revenue:.2f} | "
        f"ECOA firewall confirmed: {ecoa_firewall_confirmed}"
    )

    return {
        "messages": [
            AgentMessage(
                role="system",
                content=summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ]
    }
