"""
agents/human_review_node.py

Node 4: Human Review Gate
--------------------------
This node handles two scenarios:

  1. The critic agent passed the assessment but flagged it as requiring
     human review (e.g., borderline ELEVATED risk, concentration risk).

  2. The critic loop exhausted max_retries without a passing assessment
     — the system could not self-correct, so a human must intervene.

In a production system this node would:
  - Write the case to a review queue (e.g., SQS, database table)
  - Send a notification (email, Slack) to the analyst team
  - Pause the workflow using LangGraph's interrupt() mechanism

For this prototype it logs the escalation and marks the workflow complete.
This is still valuable — it proves the system knows when NOT to make
an autonomous decision, which is a key responsible AI signal.
"""

import structlog
from datetime import datetime, timezone

from graph.state import RiskAssessmentState, AgentMessage

log = structlog.get_logger()


def human_review_node(state: RiskAssessmentState) -> dict:
    """LangGraph node: escalates to human analyst."""
    app = state.application
    assessment = state.assessment
    audit = state.audit

    log.info("human_review_node.escalating", application_id=app.application_id)

    # Build a clear escalation summary for the analyst
    reason_parts = []

    if state.retry_count >= state.max_retries:
        reason_parts.append(f"Assessment failed audit {state.retry_count} times — could not self-correct")

    if audit and audit.violations:
        reason_parts.append(f"Unresolved violations: {'; '.join(audit.violations)}")

    if audit and audit.requires_human_review:
        reason_parts.append("Critic agent flagged case as requiring human judgment")

    if assessment:
        reason_parts.append(
            f"Last assessment: {assessment.risk_category.value} risk "
            f"(score {assessment.risk_score}) | "
            f"Action: {assessment.recommended_action.value}"
        )

    escalation_summary = (
        f"ESCALATED TO HUMAN REVIEW\n"
        f"Application: {app.application_id} — {app.business_name}\n"
        f"Amount requested: ${app.requested_amount:,.0f}\n"
        f"Reason(s): {' | '.join(reason_parts) if reason_parts else 'Policy escalation threshold met'}\n"
        f"Retry attempts: {state.retry_count}/{state.max_retries}"
    )

    log.warning("human_review_node.escalated", summary=escalation_summary)

    return {
        "escalated_to_human": True,
        "workflow_complete": True,
        "final_report": escalation_summary,
        "messages": [
            AgentMessage(
                role="system",
                content=escalation_summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ],
    }
