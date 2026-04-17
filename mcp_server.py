"""
mcp_server.py

MCP (Model Context Protocol) server that exposes the SMB risk assessment
workflow as tools any MCP-compatible AI client can call.

When connected to Claude, this allows a loan officer to type naturally:
  "Assess Sunshine Bakery for a $150K working capital loan..."
  → Claude calls assess_smb_loan() tool
  → your LangGraph workflow runs
  → Claude reads the result and responds conversationally

Tools exposed:
    assess_smb_loan     → submit a loan application for assessment
    get_assessment_status → check status of a running assessment
    get_assessment_report → retrieve completed report
    list_scenarios       → list available test scenarios

Run locally:
    python mcp_server.py

Connect to Claude Desktop:
    Add to claude_desktop_config.json:
    {
        "mcpServers": {
            "smb-risk-agent": {
                "command": "python",
                "args": ["path/to/mcp_server.py"]
            }
        }
    }

Connect to deployed server:
    Add your Render URL to Claude MCP settings
"""

import json
import time
import sys
import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="SMB Risk Assessment Agent",
    instructions=(
    "You are connected to an autonomous SMB loan risk assessment system. "
    "Use assess_smb_loan to evaluate loan applications against Basel III "
    "capital standards and CFPB fair lending regulations. "
    "After submitting: first call get_assessment_status until status is complete, "
    "then call get_assessment_report to retrieve the full results. "
    "Never call get_assessment_report before status shows complete."
    )
)

# Base URL of your FastAPI server
# Change to your Render URL when deployed
API_BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def assess_smb_loan(
    application_id: str,
    business_name: str,
    naics_code: str,
    business_structure: str,
    years_in_operation: int,
    state_of_incorporation: str,
    requested_amount: float,
    loan_purpose: str,
    requested_term_months: int,
    annual_revenue: float,
    gross_profit_margin: float,
    current_assets: float,
    current_liabilities: float,
    total_debt: float,
    annual_debt_service: float,
    net_operating_income: float,
    business_credit_score: int,
    personal_credit_score: int,
    months_cash_runway: float,
    times_nsfed_last_12mo: int,
    collateral_value: float,
    collateral_type: str,
) -> str:
    """
    Submit an SMB loan application for full risk assessment.

    After calling this tool:
    1. Call get_assessment_status(application_id) to check progress
    2. Keep calling get_assessment_status until status is 'complete'
    3. Then call get_assessment_report(application_id) for the full report

    The workflow typically takes 20-30 seconds to complete.

    Args:
        application_id: Unique identifier for this application (e.g. APP-2024-001)
        business_name: Legal name of the business
        naics_code: 6-digit NAICS industry code
        business_structure: One of: sole_proprietor, partnership, llc, s_corp, c_corp
        years_in_operation: Number of years since business formation
        state_of_incorporation: Two-letter state code (e.g. FL, TX)
        requested_amount: Loan amount requested in USD
        loan_purpose: One of: working_capital, inventory, equipment, payroll_bridge, expansion, refinance
        requested_term_months: Loan term in months (3-60)
        annual_revenue: Most recent 12-month revenue in USD
        gross_profit_margin: Gross profit divided by revenue (0.0 to 1.0)
        current_assets: Total current assets in USD
        current_liabilities: Total current liabilities in USD
        total_debt: All outstanding debt in USD
        annual_debt_service: Annual principal + interest payments in USD
        net_operating_income: EBITDA proxy in USD
        business_credit_score: Dun & Bradstreet PAYDEX score (0-100)
        personal_credit_score: Owner FICO score (300-850)
        months_cash_runway: Current cash divided by monthly burn rate
        times_nsfed_last_12mo: Number of NSF/returned items in past 12 months
        collateral_value: Fair market value of pledged collateral in USD
        collateral_type: Description of collateral (e.g. equipment, real_estate)
    """
    payload = {
        "application": {
            "application_id": application_id,
            "business_name": business_name,
            "naics_code": naics_code,
            "business_structure": business_structure,
            "years_in_operation": years_in_operation,
            "state_of_incorporation": state_of_incorporation,
            "requested_amount": requested_amount,
            "loan_purpose": loan_purpose,
            "requested_term_months": requested_term_months,
            "annual_revenue": annual_revenue,
            "gross_profit_margin": gross_profit_margin,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "total_debt": total_debt,
            "annual_debt_service": annual_debt_service,
            "net_operating_income": net_operating_income,
            "business_credit_score": business_credit_score,
            "personal_credit_score": personal_credit_score,
            "months_cash_runway": months_cash_runway,
            "times_nsfed_last_12mo": times_nsfed_last_12mo,
            "collateral_value": collateral_value,
            "collateral_type": collateral_type,
            "dscr": 0,
            "current_ratio": 0,
            "debt_to_revenue": 0,
        },
        "use_prod_db": True
    }

    try:
        response = httpx.post(
            f"{API_BASE_URL}/assess",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()

        return (
            f"Assessment submitted successfully.\n"
            f"Application ID: {result['application_id']}\n"
            f"Status: {result['status']}\n"
            f"Submitted at: {result['submitted_at']}\n\n"
            f"The workflow is running. Call get_assessment_report('{application_id}') "
            f"in about 20-30 seconds to retrieve the full results."
        )

    except httpx.HTTPError as e:
        return f"Error submitting assessment: {str(e)}"


@mcp.tool()
def get_assessment_status(application_id: str) -> str:
    """
    Check the status of a running or completed risk assessment.

    Args:
        application_id: The application ID used when submitting
    """
    try:
        response = httpx.get(
            f"{API_BASE_URL}/status/{application_id}",
            timeout=10.0
        )
        response.raise_for_status()
        result = response.json()

        status = result["status"]

        if status in ("pending", "running"):
            return (
                f"Assessment for {application_id} is still {status}. "
                f"Please wait a few more seconds and try again."
            )

        return (
            f"Assessment complete.\n"
            f"Application: {application_id}\n"
            f"Status: {status}\n"
            f"Risk category: {result.get('risk_category', 'N/A')}\n"
            f"Recommended action: {result.get('recommended_action', 'N/A')}\n"
            f"Escalated to human: {result.get('escalated_to_human', False)}\n\n"
            f"Call get_assessment_report('{application_id}') for the full report."
        )

    except httpx.HTTPError as e:
        return f"Error checking status: {str(e)}"


@mcp.tool()
def get_assessment_report(application_id: str) -> str:
    """
    Retrieve the full completed risk assessment report including
    Basel III metrics, risk analysis, conditions, and compliance audit.

    Args:
        application_id: The application ID used when submitting
    """
    try:
        response = httpx.get(
            f"{API_BASE_URL}/report/{application_id}",
            timeout=10.0
        )

        if response.status_code == 202:
            return (
                f"Assessment for {application_id} is still running. "
                f"Please wait a few more seconds and try again."
            )

        response.raise_for_status()
        result = response.json()

        if result.get("final_report"):
            return result["final_report"]

        return (
            f"Assessment {application_id} completed with status: {result['status']}\n"
            f"Escalated to human review: {result.get('escalated_to_human', False)}"
        )

    except httpx.HTTPError as e:
        return f"Error retrieving report: {str(e)}"


@mcp.tool()
def list_test_scenarios() -> str:
    """
    List the available test scenarios for demonstration purposes.
    Returns pre-built loan applications you can use to test the system.
    """
    try:
        response = httpx.get(
            f"{API_BASE_URL}/scenarios",
            timeout=10.0
        )
        response.raise_for_status()
        result = response.json()

        output = "Available test scenarios:\n\n"
        for scenario in result["scenarios"]:
            output += (
                f"Name: {scenario['name']}\n"
                f"Business: {scenario['business']}\n"
                f"Description: {scenario['description']}\n"
                f"Application ID: {scenario['application_id']}\n\n"
            )
        return output

    except httpx.HTTPError as e:
        return f"Error listing scenarios: {str(e)}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting SMB Risk Assessment MCP Server...", file=sys.stderr)
    print(f"Connected to API at: {API_BASE_URL}", file=sys.stderr)
    mcp.run(transport="stdio")
