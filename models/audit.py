"""
models/audit.py

Output schema for the critic (auditor) agent.
Checks the assessment against two regulatory frameworks:

  Basel III:
    - Is the PD within acceptable bounds for the assigned risk category?
    - Is the risk weight correctly applied per the Standardized Approach?
    - Does the capital requirement meet the 8% CET1 floor?
    - Is the EL calculation internally consistent?

  CFPB (ECOA / Section 1071 / Reg B):
    - Was demographic data excluded from the credit decision? (ECOA firewall)
    - If declined, is an adverse action notice required? (Reg B §202.9)
    - Is the Section 1071 data collection record complete?
    - Does the decision show any pattern consistent with disparate treatment?
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class AuditVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REQUIRES_HUMAN_REVIEW = "requires_human_review"


class BaselIIIAudit(BaseModel):
    """Checks the quantitative risk outputs against Basel III rules."""

    pd_within_category_bounds: bool = Field(
        ..., description="PD is consistent with the assigned risk category"
    )
    risk_weight_correctly_applied: bool = Field(
        ...,
        description=(
            "Risk weight matches Basel III Standardized Approach table "
            "(0.75 for qualifying SMB revolving, 1.0 otherwise)"
        )
    )
    el_calculation_consistent: bool = Field(
        ..., description="EL = PD × LGD × EAD within rounding tolerance"
    )
    capital_requirement_meets_floor: bool = Field(
        ..., description="capital_requirement >= RWA × 0.08"
    )
    concentration_risk_flag: bool = Field(
        False,
        description="True if single-borrower exposure exceeds 10% of notional portfolio"
    )

    notes: list[str] = Field(default_factory=list)

    @property
    def passes(self) -> bool:
        return all([
            self.pd_within_category_bounds,
            self.risk_weight_correctly_applied,
            self.el_calculation_consistent,
            self.capital_requirement_meets_floor,
        ])


class CFPBAudit(BaseModel):
    """Checks the decision for CFPB / ECOA / Reg B compliance."""

    demographic_data_excluded_from_decision: bool = Field(
        ...,
        description=(
            "Confirms owner_demographics were NOT present in the "
            "assessment agent's input (ECOA firewall check)"
        )
    )
    adverse_action_notice_required: bool = Field(
        ...,
        description=(
            "True if the decision is decline or conditional — "
            "lender must issue written adverse action notice per Reg B §202.9"
        )
    )
    section_1071_record_complete: bool = Field(
        ...,
        description="Required demographic fields present (even if applicant declined to state)"
    )
    disparate_treatment_flag: bool = Field(
        False,
        description=(
            "True if the rationale references any characteristic "
            "protected under ECOA (race, sex, religion, national origin, "
            "color, marital status, age, or public assistance status)"
        )
    )
    reg_b_timing_compliant: bool = Field(
        True,
        description="Decision notification within 30-day Reg B window (assumed true unless flagged)"
    )

    notes: list[str] = Field(default_factory=list)

    @property
    def passes(self) -> bool:
        return all([
            self.demographic_data_excluded_from_decision,
            self.section_1071_record_complete,
            not self.disparate_treatment_flag,
            self.reg_b_timing_compliant,
        ])


class AuditResult(BaseModel):
    """
    Full output of the critic agent.
    The LangGraph conditional edge reads `verdict` to decide whether to:
      - Exit to the human review gate (PASS / REQUIRES_HUMAN_REVIEW)
      - Loop back to the assessment agent with feedback (FAIL)
    """
    application_id: str
    verdict: AuditVerdict

    basel_audit: BaselIIIAudit
    cfpb_audit: CFPBAudit

    violations: list[str] = Field(
        default_factory=list,
        description="Specific rule violations found; empty list means clean"
    )
    feedback_to_assessment_agent: Optional[str] = Field(
        None,
        description=(
            "When verdict=FAIL, this message is injected back into the "
            "assessment agent's context so it can self-correct. "
            "Must be actionable and specific."
        )
    )
    requires_human_review: bool = Field(
        False,
        description="Escalate to human analyst regardless of verdict (e.g., borderline ELEVATED cases)"
    )

    model_used: str = Field(default="amazon-nova-pro", description="Audit model ID")

    model_config = {"protected_namespaces": ()}

    @model_validator(mode="after")
    def feedback_required_on_fail(self) -> "AuditResult":
        if self.verdict == AuditVerdict.FAIL and not self.feedback_to_assessment_agent:
            raise ValueError(
                "feedback_to_assessment_agent must be populated when verdict is FAIL "
                "so the assessment agent can self-correct"
            )
        return self

    @model_validator(mode="after")
    def sync_verdict_with_checks(self) -> "AuditResult":
        """
        Auto-populate violations list from sub-audit results.
        Ensures the LLM's verdict is consistent with its own audit findings.
        """
        computed_violations = []

        if not self.basel_audit.pd_within_category_bounds:
            computed_violations.append("Basel III: PD inconsistent with risk category")
        if not self.basel_audit.risk_weight_correctly_applied:
            computed_violations.append("Basel III: Incorrect risk weight applied")
        if not self.basel_audit.el_calculation_consistent:
            computed_violations.append("Basel III: EL calculation error (PD × LGD × EAD mismatch)")
        if not self.basel_audit.capital_requirement_meets_floor:
            computed_violations.append("Basel III: Capital requirement below 8% CET1 floor")
        if not self.cfpb_audit.demographic_data_excluded_from_decision:
            computed_violations.append("CFPB ECOA: Demographic data present in credit decision")
        if self.cfpb_audit.disparate_treatment_flag:
            computed_violations.append("CFPB ECOA: Potential disparate treatment detected in rationale")
        if not self.cfpb_audit.section_1071_record_complete:
            computed_violations.append("CFPB Section 1071: Incomplete demographic data record")

        # Merge with any violations the LLM explicitly listed
        all_violations = list(set(computed_violations + self.violations))
        self.violations = all_violations

        # Enforce verdict consistency
        has_hard_failures = not (self.basel_audit.passes and self.cfpb_audit.passes)
        if has_hard_failures and self.verdict == AuditVerdict.PASS:
            raise ValueError(
                f"Verdict is PASS but audit checks failed: {all_violations}. "
                "Verdict must be FAIL or REQUIRES_HUMAN_REVIEW."
            )
        return self
