"""
models/assessment.py

Output schema for the risk assessment agent.
Structured around Basel III's Internal Ratings-Based (IRB) approach
adapted for the Standardized Approach used by most SMB lenders.

Key Basel III concepts modeled here:
  - PD  : Probability of Default (12-month horizon)
  - LGD : Loss Given Default (% of EAD lost if borrower defaults)
  - EAD : Exposure at Default (outstanding balance at time of default)
  - EL  : Expected Loss = PD × LGD × EAD
  - RWA : Risk-Weighted Assets = EAD × Risk Weight
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class RiskCategory(str, Enum):
    LOW = "low"           # PD < 2%
    MODERATE = "moderate" # PD 2–5%
    ELEVATED = "elevated" # PD 5–10%
    HIGH = "high"         # PD 10–20%
    CRITICAL = "critical" # PD > 20%


class RecommendedAction(str, Enum):
    APPROVE = "approve"
    APPROVE_WITH_CONDITIONS = "approve_with_conditions"
    DECLINE = "decline"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class BaselIIIMetrics(BaseModel):
    """
    Core IRB-style metrics. For SMB lenders using the Standardized
    Approach, risk weights are prescribed by regulator rather than
    internally modeled — but PD/LGD are still best-practice inputs.
    """
    pd: float = Field(..., ge=0.0, le=1.0, description="Probability of Default, 0–1")
    lgd: float = Field(..., ge=0.0, le=1.0, description="Loss Given Default, 0–1")
    ead: float = Field(..., ge=0, description="Exposure at Default, USD")
    risk_weight: float = Field(
        ..., ge=0.0, le=1.5,
        description=(
            "Standardized risk weight per Basel III Table 1. "
            "Typically 0.75 for qualifying revolving SMB exposures, "
            "1.0 for other SMB exposures."
        )
    )
    expected_loss: float = Field(0.0, description="EL = PD × LGD × EAD (computed)")
    rwa: float = Field(0.0, description="Risk-Weighted Assets = EAD × risk_weight (computed)")
    capital_requirement: float = Field(
        0.0,
        description="Minimum regulatory capital = RWA × 0.08 (8% Basel III floor)"
    )

    def model_post_init(self, __context):
        self.expected_loss = round(self.pd * self.lgd * self.ead, 2)
        self.rwa = round(self.ead * self.risk_weight, 2)
        self.capital_requirement = round(self.rwa * 0.08, 2)


class RiskAssessmentOutput(BaseModel):
    """
    Full structured output from the assessment agent.
    Every field here is enforced by Pydantic before the critic agent
    ever sees it — malformed LLM output raises ValidationError, not
    a silent downstream failure.
    """
    application_id: str
    risk_category: RiskCategory
    risk_score: int = Field(..., ge=1, le=100, description="Internal composite score, 1=best 100=worst")
    recommended_action: RecommendedAction
    basel_metrics: BaselIIIMetrics

    # Narrative fields (required — not optional)
    rationale: str = Field(
        ..., min_length=50,
        description="Plain-language explanation of the risk drivers (min 50 chars)"
    )
    key_risk_factors: list[str] = Field(
        ..., min_length=1,
        description="Ordered list of top risk drivers, most significant first"
    )
    mitigating_factors: list[str] = Field(
        default_factory=list,
        description="Factors that reduce risk (collateral, strong cash flow, etc.)"
    )

    # Conditions (populated when action = APPROVE_WITH_CONDITIONS)
    conditions: list[str] = Field(
        default_factory=list,
        description="Required conditions if action is approve_with_conditions"
    )
    suggested_loan_amount: Optional[float] = Field(
        None, ge=0,
        description="Counter-offer amount if requested amount is too high"
    )

    # Metadata
    model_used: str = Field(default="claude-3-5-sonnet", description="Inference model ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model self-assessed confidence, 0–1")

    model_config = {"protected_namespaces": ()} #“Do NOT reserve any prefixes — allow any field names”

    @model_validator(mode="after")
    def conditions_required_when_conditional_approval(self) -> "RiskAssessmentOutput":
        if (
            self.recommended_action == RecommendedAction.APPROVE_WITH_CONDITIONS
            and not self.conditions
        ):
            raise ValueError(
                "conditions list must be non-empty when action is approve_with_conditions"
            )
        return self

    @model_validator(mode="after")
    def risk_score_aligns_with_category(self) -> "RiskAssessmentOutput":
        """Catch LLM hallucinations where score and category contradict each other."""
        thresholds = {
            RiskCategory.LOW: (1, 20),
            RiskCategory.MODERATE: (21, 40),
            RiskCategory.ELEVATED: (41, 60),
            RiskCategory.HIGH: (61, 80),
            RiskCategory.CRITICAL: (81, 100),
        }
        low, high = thresholds[self.risk_category]
        if not (low <= self.risk_score <= high):
            raise ValueError(
                f"risk_score {self.risk_score} is inconsistent with "
                f"risk_category '{self.risk_category}' (expected {low}–{high})"
            )
        return self
