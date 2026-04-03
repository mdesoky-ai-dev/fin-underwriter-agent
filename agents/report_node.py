"""
agents/report_node.py

Node 5: Final Report Generator
--------------------------------
Assembles the completed risk assessment and audit results into a
clean, structured final report. This is the terminal node for
successful (non-escalated) workflows.

In production this would render to PDF, write to a database, or
trigger a loan origination system API. For this prototype it
produces a formatted text report and marks the workflow complete.
"""

import structlog
from datetime import datetime, timezone

from graph.state import RiskAssessmentState, AgentMessage
from models.assessment import RecommendedAction

log = structlog.get_logger()


def report_node(state: RiskAssessmentState) -> dict:
    """LangGraph node: generates the final decision report."""
    app = state.application
    assessment = state.assessment
    audit = state.audit

    log.info("report_node.generating", application_id=app.application_id)

    m = assessment.basel_metrics
    action_label = {
        RecommendedAction.APPROVE: "✅ APPROVED",
        RecommendedAction.APPROVE_WITH_CONDITIONS: "✅ APPROVED WITH CONDITIONS",
        RecommendedAction.DECLINE: "❌ DECLINED",
        RecommendedAction.ESCALATE_TO_HUMAN: "⚠️  ESCALATED",
    }.get(assessment.recommended_action, assessment.recommended_action.value)

    report_lines = [
        "=" * 60,
        "SMB WORKING CAPITAL LOAN — RISK ASSESSMENT REPORT",
        "=" * 60,
        f"Application ID  : {app.application_id}",
        f"Business        : {app.business_name}",
        f"Generated       : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Audit retries   : {state.retry_count}",
        "",
        "DECISION",
        "-" * 40,
        f"  {action_label}",
        f"  Risk category : {assessment.risk_category.value.upper()}",
        f"  Risk score    : {assessment.risk_score}/100",
        f"  Confidence    : {assessment.confidence:.0%}",
        "",
        "BASEL III METRICS",
        "-" * 40,
        f"  PD  (Prob. of Default)    : {m.pd:.2%}",
        f"  LGD (Loss Given Default)  : {m.lgd:.2%}",
        f"  EAD (Exposure at Default) : ${m.ead:>12,.2f}",
        f"  Risk Weight               : {m.risk_weight:.2f}",
        f"  Expected Loss             : ${m.expected_loss:>12,.2f}",
        f"  Risk-Weighted Assets      : ${m.rwa:>12,.2f}",
        f"  Capital Requirement (8%)  : ${m.capital_requirement:>12,.2f}",
        "",
        "RISK ANALYSIS",
        "-" * 40,
        f"  {assessment.rationale}",
        "",
        "KEY RISK FACTORS",
        "-" * 40,
    ]

    for i, factor in enumerate(assessment.key_risk_factors, 1):
        report_lines.append(f"  {i}. {factor}")

    if assessment.mitigating_factors:
        report_lines += ["", "MITIGATING FACTORS", "-" * 40]
        for i, factor in enumerate(assessment.mitigating_factors, 1):
            report_lines.append(f"  {i}. {factor}")

    if assessment.conditions:
        report_lines += ["", "CONDITIONS FOR APPROVAL", "-" * 40]
        for i, cond in enumerate(assessment.conditions, 1):
            report_lines.append(f"  {i}. {cond}")

    if assessment.suggested_loan_amount:
        report_lines.append(f"\n  Counter-offer amount: ${assessment.suggested_loan_amount:,.0f}")

    report_lines += [
        "",
        "COMPLIANCE AUDIT",
        "-" * 40,
        f"  Audit verdict  : {audit.verdict.value.upper()}",
        f"  Basel III      : {'PASS' if audit.basel_audit.passes else 'FAIL'}",
        f"  CFPB / ECOA    : {'PASS' if audit.cfpb_audit.passes else 'FAIL'}",
        f"  Adverse action notice required: {audit.cfpb_audit.adverse_action_notice_required}",
        f"  Violations     : {len(audit.violations)}",
    ]

    if audit.violations:
        for v in audit.violations:
            report_lines.append(f"    - {v}")

    report_lines += ["", "=" * 60]

    final_report = "\n".join(report_lines)
    log.info("report_node.complete", application_id=app.application_id)

    return {
        "final_report": final_report,
        "workflow_complete": True,
        "messages": [
            AgentMessage(
                role="system",
                content=f"Final report generated for {app.application_id}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ],
    }
