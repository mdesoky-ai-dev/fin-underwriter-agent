"""
agents/assessment_agent.py

Node 2: Risk Assessment Agent
------------------------------
Uses Claude 3.5 Sonnet (via AWS Bedrock) to analyze the SMB loan
application and produce a structured risk assessment.

Key design decisions:
  1. We pass ONLY credit_decision_fields() to the LLM — demographics
     are stripped before the prompt is built (ECOA firewall).
  2. We instruct the model to respond in JSON that matches our
     RiskAssessmentOutput schema exactly.
  3. Pydantic validates the JSON before it touches the state — if the
     LLM hallucinates a bad value, we catch it here, not downstream.
  4. If the critic agent previously failed this assessment, its
     feedback is injected into the prompt so the model can self-correct.

LangSmith traces every token of this call automatically via the
LANGCHAIN_TRACING_V2 environment variable.
"""

import json
import os
import structlog
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

from graph.state import RiskAssessmentState, AgentMessage
from models.assessment import (
    RiskAssessmentOutput,
    RiskCategory,
    RecommendedAction,
    BaselIIIMetrics,
)

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior credit risk analyst specializing in SMB 
(small and medium business) working capital lending. You apply Basel III 
Internal Ratings-Based methodology adapted for the Standardized Approach.

Your job is to analyze loan applications and return a structured JSON risk 
assessment. You must be precise, consistent, and conservative in your 
risk ratings. When in doubt, rate higher risk rather than lower.

CRITICAL RULES:
- Your response must be valid JSON only. No preamble, no explanation outside the JSON.
- risk_score must be consistent with risk_category:
    LOW: 1-20, MODERATE: 21-40, ELEVATED: 41-60, HIGH: 61-80, CRITICAL: 81-100
- PD (probability of default) must align with risk_category:
    LOW: <0.02, MODERATE: 0.02-0.05, ELEVATED: 0.05-0.10, HIGH: 0.10-0.20, CRITICAL: >0.20
- If action is approve_with_conditions, conditions list must not be empty.
- rationale must be at least 50 characters and cite specific data points.
- LGD for unsecured SMB working capital is typically 0.45-0.75.
- Risk weight: use 0.75 for qualifying revolving SMB exposures under $1M, 
  1.0 for all others (Basel III Standardized Approach Table 1).
"""

def build_assessment_prompt(app_data: dict, critic_feedback: str | None = None) -> str:
    """
    Builds the user-facing prompt. Injects critic feedback if this is a retry.
    """
    base_prompt = f"""Analyze this SMB working capital loan application and return a JSON risk assessment.

APPLICATION DATA:
{json.dumps(app_data, indent=2)}

KEY RATIOS (pre-computed):
- DSCR (Debt Service Coverage Ratio): {app_data.get('dscr', 'N/A')} 
  (Benchmark: >1.25 acceptable, >1.5 strong, <1.0 critical)
- Current Ratio: {app_data.get('current_ratio', 'N/A')}
  (Benchmark: >1.5 healthy, <1.0 liquidity risk)
- Debt-to-Revenue: {app_data.get('debt_to_revenue', 'N/A')}
  (Benchmark: <0.3 healthy, >0.5 elevated, >1.0 critical)

Return ONLY this JSON structure (no other text):
{{
  "application_id": "{app_data['application_id']}",
  "risk_category": "low|moderate|elevated|high|critical",
  "risk_score": <integer 1-100>,
  "recommended_action": "approve|approve_with_conditions|decline|escalate_to_human",
  "basel_metrics": {{
    "pd": <float 0-1>,
    "lgd": <float 0-1>,
    "ead": <float, equals requested_amount>,
    "risk_weight": <0.75 or 1.0>
  }},
  "rationale": "<minimum 50 char explanation citing specific data points>",
  "key_risk_factors": ["<factor 1>", "<factor 2>"],
  "mitigating_factors": ["<factor 1>"],
  "conditions": ["<condition if approve_with_conditions, else empty list>"],
  "suggested_loan_amount": <float or null>,
  "confidence": <float 0-1>
}}"""

    if critic_feedback:
        base_prompt += f"""

IMPORTANT — PREVIOUS ASSESSMENT WAS REJECTED BY THE AUDIT AGENT.
You must address these specific issues in your revised assessment:

{critic_feedback}

Do not repeat the same errors. Be more precise and ensure all values 
are internally consistent."""

    return base_prompt


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def assessment_node(state: RiskAssessmentState) -> dict:
    """
    LangGraph node: calls Claude 3.5 Sonnet to produce a risk assessment.
    Returns partial state update dict.
    """
    app = state.application
    log.info("assessment_node.start", application_id=app.application_id, retry=state.retry_count)

    # Get critic feedback if this is a retry
    critic_feedback = None
    if state.audit and state.audit.feedback_to_assessment_agent:
        critic_feedback = state.audit.feedback_to_assessment_agent
        log.info("assessment_node.retry_with_feedback", feedback=critic_feedback[:100])

    # Build prompt using ONLY credit decision fields (ECOA firewall)
    app_data = app.credit_decision_fields()
    user_prompt = build_assessment_prompt(app_data, critic_feedback)

    # Call Claude 3.5 Sonnet via Bedrock
    try:
        raw_response = _call_bedrock(
            model_id=os.getenv("BEDROCK_ASSESSMENT_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        log.error("assessment_node.bedrock_error", error=str(e))
        return {
            "messages": [
                AgentMessage(
                    role="assessment_agent",
                    content=f"ERROR calling Bedrock: {str(e)}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            ],
            "route_decision": "human_review",
        }

    # Parse and validate the JSON response with Pydantic
    try:
        raw_json = _extract_json(raw_response)
        assessment = RiskAssessmentOutput(**raw_json)
        log.info(
            "assessment_node.success",
            risk_category=assessment.risk_category,
            risk_score=assessment.risk_score,
            action=assessment.recommended_action,
            pd=assessment.basel_metrics.pd,
            el=assessment.basel_metrics.expected_loss,
        )
    except Exception as e:
        log.error("assessment_node.validation_error", error=str(e), raw=raw_response[:300])
        # If Pydantic rejects the output, treat it like a failed audit
        return {
            "messages": [
                AgentMessage(
                    role="assessment_agent",
                    content=f"Schema validation failed: {str(e)}\nRaw output: {raw_response[:500]}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            ],
            "retry_count": state.retry_count + 1,
            "route_decision": "retry" if state.retry_count < state.max_retries else "human_review",
        }

    return {
        "assessment": assessment,
        "messages": [
            AgentMessage(
                role="assessment_agent",
                content=(
                    f"Assessment complete: {assessment.risk_category.value.upper()} risk "
                    f"(score {assessment.risk_score}/100) | "
                    f"Action: {assessment.recommended_action.value} | "
                    f"PD: {assessment.basel_metrics.pd:.1%} | "
                    f"EL: ${assessment.basel_metrics.expected_loss:,.2f} | "
                    f"Confidence: {assessment.confidence:.0%}"
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ],
    }


# ---------------------------------------------------------------------------
# Bedrock helpers
# ---------------------------------------------------------------------------

def _call_bedrock(model_id: str, system_prompt: str, user_prompt: str) -> str:
    """Calls Claude via AWS Bedrock Converse API."""
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={
            "maxTokens": 2048,
            "temperature": 0.1,   # Low temp for consistent structured output
            "topP": 0.9,
        },
    )
    return response["output"]["message"]["content"][0]["text"]


def _extract_json(text: str) -> dict:
    """
    Extracts JSON from the model response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(text)
