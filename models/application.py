"""
models/application.py

Input schema for an SMB working capital loan application.
Captures both the financial data needed for Basel III risk-weighting
and the demographic data required by CFPB Section 1071.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LoanPurpose(str, Enum):
    WORKING_CAPITAL = "working_capital"
    INVENTORY = "inventory"
    EQUIPMENT = "equipment"
    PAYROLL_BRIDGE = "payroll_bridge"
    EXPANSION = "expansion"
    REFINANCE = "refinance"


class BusinessStructure(str, Enum):
    SOLE_PROPRIETOR = "sole_proprietor"
    PARTNERSHIP = "partnership"
    LLC = "llc"
    S_CORP = "s_corp"
    C_CORP = "c_corp"


class OwnerDemographics(BaseModel):
    """
    CFPB Section 1071 (Dodd-Frank) requires collection of owner
    demographic data for fair lending analysis. This data must be
    firewalled from the credit decision itself.
    """
    ethnicity: Optional[str] = Field(None, description="HMDA ethnicity category")
    race: Optional[str] = Field(None, description="HMDA race category")
    sex: Optional[str] = Field(None, description="HMDA sex category")
    veteran_status: Optional[bool] = None

    class Config:
        # Demographic fields must never be passed to the assessment agent.
        # The critic agent checks ECOA compliance after the fact.
        json_schema_extra = {
            "x-cfpb-section-1071": True,
            "x-exclude-from-credit-decision": True,
        }


class SMBLoanApplication(BaseModel):
    """
    Full application record. The assessment agent receives everything
    EXCEPT owner_demographics. The critic agent sees everything.
    """

    # --- Identifiers ---
    application_id: str = Field(..., description="Unique application reference")
    business_name: str
    naics_code: str = Field(..., description="6-digit NAICS industry code", min_length=6, max_length=6)
    business_structure: BusinessStructure
    years_in_operation: int = Field(..., ge=0, description="Years since business formation")
    state_of_incorporation: str = Field(..., min_length=2, max_length=2, description="Two-letter state code")

    # --- Loan request ---
    requested_amount: float = Field(..., gt=0, description="USD requested")
    loan_purpose: LoanPurpose
    requested_term_months: int = Field(..., ge=3, le=60)

    # --- Financial data (Basel III inputs) ---
    annual_revenue: float = Field(..., gt=0, description="Most recent 12-month revenue, USD")
    gross_profit_margin: float = Field(..., ge=0.0, le=1.0, description="Gross profit / revenue")
    current_assets: float = Field(..., ge=0)
    current_liabilities: float = Field(..., ge=0)
    total_debt: float = Field(..., ge=0, description="All outstanding debt obligations, USD")
    annual_debt_service: float = Field(..., ge=0, description="Annual principal + interest payments, USD")
    net_operating_income: float = Field(..., description="EBITDA proxy, USD; can be negative")
    business_credit_score: int = Field(..., ge=0, le=100, description="Dun & Bradstreet PAYDEX or equivalent 0-100")
    personal_credit_score: int = Field(..., ge=300, le=850, description="Owner FICO score")
    months_cash_runway: float = Field(..., ge=0, description="Current cash / monthly burn rate")
    times_nsfed_last_12mo: int = Field(0, ge=0, description="NSF / returned items in bank history")

    # --- Collateral ---
    collateral_value: float = Field(0.0, ge=0, description="USD fair market value of pledged collateral")
    collateral_type: Optional[str] = None

    # --- CFPB Section 1071 demographic data ---
    owner_demographics: Optional[OwnerDemographics] = None

    # --- Derived ratios (computed on instantiation) ---
    dscr: float = Field(0.0, description="Debt service coverage ratio (computed)")
    current_ratio: float = Field(0.0, description="Current assets / current liabilities (computed)")
    debt_to_revenue: float = Field(0.0, description="Total debt / annual revenue (computed)")

    def model_post_init(self, __context):
        """Compute financial ratios after validation."""
        if self.annual_debt_service > 0:
            self.dscr = round(self.net_operating_income / self.annual_debt_service, 4)
        if self.current_liabilities > 0:
            self.current_ratio = round(self.current_assets / self.current_liabilities, 4)
        if self.annual_revenue > 0:
            self.debt_to_revenue = round(self.total_debt / self.annual_revenue, 4)

    @field_validator("naics_code")
    @classmethod
    def validate_naics(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError("NAICS code must be 6 digits")
        return v

    def credit_decision_fields(self) -> dict:
        """
        Returns only the fields the assessment agent should see.
        Strips owner_demographics per ECOA firewall requirement.
        """
        data = self.model_dump(exclude={"owner_demographics"})
        return data


#Pydantic defines and validates a structured data model, enforcing types and constraints across related fields. Enums are used to restrict individual fields to predefined values. The Config class allows us to attach metadata to the schema, such as compliance flags. Field validators enforce domain-specific rules like ensuring NAICS codes are numeric. 
# Finally, we expose a sanitized version of the data using credit_decision_fields, which removes protected demographic information to enforce fair-lending constraints.