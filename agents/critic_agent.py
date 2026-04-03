"""
agents/critic_agent.py

Node 3: Critic (Audit) Agent
-----------------------------
Uses Amazon Nova Pro (via AWS Bedrock) to audit the assessment agent's
output against two regulatory frameworks:

  Basel III — quantitative consistency checks:
    - Is PD within bounds for the assigned risk category?
    - Is the risk weight correctly applied (0.75 vs 1.0)?
    - Is EL = PD × LGD × EAD (within rounding tolerance)?
    - Does capital requirement meet the 8% CET1 floor?

  CFPB ECOA / Section 1071 — fair lending checks:
    - Were demographics excluded from the credit decision?
    - Is adverse action notice required (decline or conditional)?
    - Is Section 1071 demographic record complete?
    - Does the rationale contain any protected-class language?

Nova Pro is used here (vs Claude) for cost efficiency — the audit task
is more structured and bounded than the original assessment, so a
less expensive model is appropriate. This is a deliberate architecture
decision you should be able to explain in interviews.

The critic node writes `route_decision` to the state:
  'retry'        → loops back to assessment_node with feedback
  'human_review' → escalates to the human review gate
  'complete'     → proceeds to final report
"""

import json
import os
import math
import structlog
from datetime import datetime, timezone

import boto3

from graph.state import RiskAssessmentState, AgentMessage
from models.audit import AuditResult, AuditVerdict, BaselIIIAudit, CFPBAudit
from models.assessment import RiskCategory

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Risk category PD bounds for Basel III consistency check
# ---------------------------------------------------------------------------

PD_BOUNDS = {
    RiskCategory.LOW:      (0.0,  0.02),
    RiskCategory.MODERATE: (0.02, 0.05),
    RiskCategory.ELEVATED: (0.05, 0.10),
    RiskCategory.HIGH:     (0.10, 0.20),
    RiskCategory.CRITICAL: (0.20, 1.0),
}

ECOA_PROTECTED_TERMS = [
    "race", "ethnicity", "sex", "gender", "religion", "national origin",
    "color", "marital status", "age", "public assistance", "veteran",
]

SYSTEM_PROMPT = """You are a regulatory compliance auditor specializing in 
Basel III credit risk standards and CFPB fair lending regulations (ECOA, 
Reg B, Section 1071).

Your job is to audit a completed SMB loan risk assessment and return a 
structured JSON compliance report.

You must be thorough and flag any inconsistencies, even minor ones.
Your response must be valid JSON only. No preamble or explanation outside the JSON.

Return ONLY this JSON structure:
{
  "application_id": "<id>",
  "verdict": "pass|fail|requires_human_review",
  "basel_audit": {
    "pd_within_category_bounds": <bool>,
    "risk_weight_correctly_applied": <bool>,
    "el_calculation_consistent": <bool>,
    "capital_requirement_meets_floor": <bool>,
    "concentration_risk_flag": <bool>,
    "notes": ["<note if any>"]
  },
  "cfpb_audit": {
    "demographic_data_excluded_from_decision": <bool>,
    "adverse_action_notice_required": <bool>,
    "section_1071_record_complete": <bool>,
    "disparate_treatment_flag": <bool>,
    "reg_b_timing_compliant": <bool>,
    "notes": ["<note if any>"]
  },
  "violations": ["<specific violation description>"],
  "feedback_to_assessment_agent": "<actionable feedback if verdict=fail, else null>",
  "requires_human_review": <bool>
}"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def critic_node(state: RiskAssessmentState) -> dict:
    """
    LangGraph node: audits the assessment against Basel III + CFPB rules.
    Returns partial state update dict including route_decision.
    """
    
    # Guard — if assessment failed, route directly to human review
    if state.assessment is None:
        log.warning("critic_node.no_assessment", application_id=state.application.application_id)
        return {
            "route_decision": "human_review",
            "escalated_to_human": True,
            "messages": [
                AgentMessage(
                    role="critic_agent",
                    content="No assessment available to audit — routing to human review",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            ]
        }

    assessment = state.assessment    
    app = state.application

    log.info("critic_node.start", application_id=app.application_id, retry=state.retry_count)

    # Run deterministic pre-checks before calling the LLM.
    # These are math checks — no LLM needed.
    pre_checks = _run_deterministic_checks(assessment, app)

    # Build the audit prompt
    user_prompt = _build_audit_prompt(assessment, app, pre_checks)

    # Call Nova Pro via Bedrock
    try:
        raw_response = _call_bedrock(
            model_id=os.getenv("BEDROCK_AUDIT_MODEL", "amazon.nova-pro-v1:0"),
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        log.error("critic_node.bedrock_error", error=str(e))
        # If the auditor itself fails, escalate to human
        return {
            "messages": [
                AgentMessage(
                    role="critic_agent",
                    content=f"ERROR: Audit agent failed to call Bedrock: {str(e)}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            ],
            "route_decision": "human_review",
            "escalated_to_human": True,
        }

    # Parse and validate audit result
    try:
        raw_json = _extract_json(raw_response)
        audit = AuditResult(**raw_json)
        log.info(
            "critic_node.success",
            verdict=audit.verdict,
            violations=len(audit.violations),
            requires_human=audit.requires_human_review,
        )
    except Exception as e:
        log.error("critic_node.validation_error", error=str(e))
        # Malformed audit output → escalate rather than infinite loop
        return {
            "messages": [
                AgentMessage(
                    role="critic_agent",
                    content=f"Audit schema validation failed: {str(e)}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            ],
            "route_decision": "human_review",
            "escalated_to_human": True,
        }

    # Determine routing
    route = _determine_route(audit, state.retry_count, state.max_retries)

    log.info("critic_node.routing", route=route, retry_count=state.retry_count)

    return {
        "audit": audit,
        "retry_count": state.retry_count + (1 if route == "retry" else 0),
        "escalated_to_human": route == "human_review",
        "messages": [
            AgentMessage(
                role="critic_agent",
                content=(
                    f"Audit verdict: {audit.verdict.value.upper()} | "
                    f"Violations: {len(audit.violations)} | "
                    f"Route: {route} | "
                    f"Basel: {'✓' if audit.basel_audit.passes else '✗'} | "
                    f"CFPB: {'✓' if audit.cfpb_audit.passes else '✗'}"
                    + (f" | Feedback: {audit.feedback_to_assessment_agent[:80]}..."
                       if audit.feedback_to_assessment_agent else "")
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ],
        "route_decision": route,
    }


# ---------------------------------------------------------------------------
# Deterministic pre-checks (no LLM needed)
# ---------------------------------------------------------------------------

def _run_deterministic_checks(assessment, app) -> dict:
    """
    Math-based checks we can verify without an LLM.
    Results are passed to the audit prompt so Nova doesn't have to re-derive them.
    """
    m = assessment.basel_metrics
    bounds = PD_BOUNDS[assessment.risk_category]

    # EL tolerance check (floating point safe)
    expected_el = m.pd * m.lgd * m.ead
    el_consistent = math.isclose(m.expected_loss, expected_el, rel_tol=0.01)

    # Capital floor check
    expected_cap = m.rwa * 0.08
    cap_meets_floor = m.capital_requirement >= (expected_cap * 0.99)

    # Risk weight rule: 0.75 for SMB revolving under $1M, else 1.0
    is_revolving_smb = app.requested_amount < 1_000_000
    correct_weight = 0.75 if is_revolving_smb else 1.0
    weight_correct = math.isclose(m.risk_weight, correct_weight, abs_tol=0.01)

    # Protected class scan in rationale
    rationale_lower = assessment.rationale.lower()
    protected_terms_found = [t for t in ECOA_PROTECTED_TERMS if t in rationale_lower]

    return {
        "pd_in_bounds": bounds[0] <= m.pd <= bounds[1],
        "pd_bounds": bounds,
        "el_consistent": el_consistent,
        "el_computed": round(expected_el, 2),
        "el_reported": m.expected_loss,
        "capital_meets_floor": cap_meets_floor,
        "capital_computed": round(expected_cap, 2),
        "capital_reported": m.capital_requirement,
        "risk_weight_correct": weight_correct,
        "correct_weight": correct_weight,
        "reported_weight": m.risk_weight,
        "protected_terms_in_rationale": protected_terms_found,
        "demographics_in_assessment": hasattr(assessment, "owner_demographics"),
    }


def _build_audit_prompt(assessment, app, pre_checks: dict) -> str:
    return f"""Audit this SMB loan risk assessment for Basel III and CFPB compliance.

ASSESSMENT TO AUDIT:
{assessment.model_dump_json(indent=2)}

APPLICATION CONTEXT:
- Application ID: {app.application_id}
- Requested amount: ${app.requested_amount:,.0f}
- Demographics present in assessment: {pre_checks['demographics_in_assessment']}
- Has owner demographics on file: {app.owner_demographics is not None}
- Adverse action required: {assessment.recommended_action.value in ['decline', 'approve_with_conditions']}

PRE-COMPUTED DETERMINISTIC CHECKS (use these in your audit):
- PD {assessment.basel_metrics.pd:.4f} in bounds {pre_checks['pd_bounds']}: {pre_checks['pd_in_bounds']}
- EL computed: ${pre_checks['el_computed']:,.2f} | reported: ${pre_checks['el_reported']:,.2f} | consistent: {pre_checks['el_consistent']}
- Capital computed: ${pre_checks['capital_computed']:,.2f} | reported: ${pre_checks['capital_reported']:,.2f} | meets floor: {pre_checks['capital_meets_floor']}
- Correct risk weight: {pre_checks['correct_weight']} | reported: {pre_checks['reported_weight']} | correct: {pre_checks['risk_weight_correct']}
- Protected class terms in rationale: {pre_checks['protected_terms_in_rationale'] or 'none found'}

Your audit must:
1. Use the pre-computed checks above for Basel III math verification
2. Verify CFPB compliance based on the application context
3. Set verdict to 'fail' if ANY hard check fails
4. Set verdict to 'requires_human_review' for borderline ELEVATED cases or concentration risk
5. Provide specific, actionable feedback_to_assessment_agent if verdict is 'fail'"""


def _determine_route(audit: AuditResult, retry_count: int, max_retries: int) -> str:
    """Determines the LangGraph routing decision."""
    if audit.verdict == AuditVerdict.PASS and not audit.requires_human_review:
        return "complete"
    if audit.verdict == AuditVerdict.REQUIRES_HUMAN_REVIEW or audit.requires_human_review:
        return "human_review"
    # FAIL — retry if attempts remain, else escalate
    if retry_count < max_retries:
        return "retry"
    return "human_review"


# ---------------------------------------------------------------------------
# Bedrock helpers
# ---------------------------------------------------------------------------

def _call_bedrock(model_id: str, system_prompt: str, user_prompt: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.1, "topP": 0.9},
    )
    return response["output"]["message"]["content"][0]["text"]


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(text)
