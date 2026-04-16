"""
main.py

Entry point for the SMB Risk Assessment Agent.
Run this file to execute a full workflow end-to-end.

LangSmith tracing is enabled automatically when these env vars are set:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<your key>
  LANGCHAIN_PROJECT=smb-risk-agent

Every run will appear in your LangSmith dashboard at smith.langchain.com
showing: node execution order, inputs/outputs at each step, token usage,
latency, and the full message trace between agents.

Usage:
  python main.py                    # runs the default test case
  python main.py --scenario decline # runs a high-risk decline scenario
"""

import os
import argparse
import structlog
from dotenv import load_dotenv

load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

from graph.workflow import build_dev_graph, build_prod_graph
from graph.state import RiskAssessmentState
from models.application import (
    SMBLoanApplication, LoanPurpose, BusinessStructure, OwnerDemographics
)


# ---------------------------------------------------------------------------
# Test scenarios — realistic SMB loan applications
# ---------------------------------------------------------------------------

SCENARIOS = {

    "moderate": SMBLoanApplication(
        application_id="APP-2024-001",
        business_name="Sunshine Bakery LLC",
        naics_code="311811",
        business_structure=BusinessStructure.LLC,
        years_in_operation=4,
        state_of_incorporation="FL",
        requested_amount=150_000,
        loan_purpose=LoanPurpose.WORKING_CAPITAL,
        requested_term_months=24,
        annual_revenue=820_000,
        gross_profit_margin=0.42,
        current_assets=95_000,
        current_liabilities=62_000,
        total_debt=180_000,
        annual_debt_service=48_000,
        net_operating_income=71_000,
        business_credit_score=72,
        personal_credit_score=698,
        months_cash_runway=3.2,
        times_nsfed_last_12mo=1,
        collateral_value=80_000,
        collateral_type="equipment",
        owner_demographics=OwnerDemographics(
            ethnicity="Hispanic", race="White", sex="Female"
        ),
    ),

    "conditional": SMBLoanApplication(
        application_id="APP-2024-002",
        business_name="Coastal Logistics Partners",
        naics_code="488510",
        business_structure=BusinessStructure.S_CORP,
        years_in_operation=9,
        state_of_incorporation="GA",
        requested_amount=400_000,
        loan_purpose=LoanPurpose.WORKING_CAPITAL,
        requested_term_months=36,
        annual_revenue=3_200_000,
        gross_profit_margin=0.31,
        current_assets=680_000,
        current_liabilities=290_000,
        total_debt=520_000,
        annual_debt_service=110_000,
        net_operating_income=295_000,
        business_credit_score=88,
        personal_credit_score=762,
        months_cash_runway=8.5,
        times_nsfed_last_12mo=0,
        collateral_value=350_000,
        collateral_type="accounts_receivable",
        owner_demographics=OwnerDemographics(
            ethnicity="Not Hispanic", race="Black or African American", sex="Male"
        ),
    ),

    "decline": SMBLoanApplication(
        application_id="APP-2024-003",
        business_name="QuickFix Auto Repair",
        naics_code="811111",
        business_structure=BusinessStructure.SOLE_PROPRIETOR,
        years_in_operation=1,
        state_of_incorporation="TX",
        requested_amount=85_000,
        loan_purpose=LoanPurpose.PAYROLL_BRIDGE,
        requested_term_months=12,
        annual_revenue=210_000,
        gross_profit_margin=0.18,
        current_assets=12_000,
        current_liabilities=38_000,
        total_debt=95_000,
        annual_debt_service=42_000,
        net_operating_income=18_000,
        business_credit_score=31,
        personal_credit_score=589,
        months_cash_runway=0.8,
        times_nsfed_last_12mo=4,
        collateral_value=15_000,
        collateral_type="vehicle",
        owner_demographics=OwnerDemographics(
            ethnicity="Not Hispanic", race="White", sex="Male"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(scenario_name: str = "moderate", use_prod: bool = False):
    application = SCENARIOS.get(scenario_name)
    if not application:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Choose from: {list(SCENARIOS.keys())}")

    log.info(
        "workflow.starting",
        scenario=scenario_name,
        application_id=application.application_id,
        business=application.business_name,
        langsmith_project=os.getenv("LANGCHAIN_PROJECT", "not set"),
        tracing=os.getenv("LANGCHAIN_TRACING_V2", "false"),
    )

    # Build the graph (in-memory checkpointer for dev)
    app = build_prod_graph() if use_prod else build_dev_graph()

    # Initial state — application is pre-loaded
    initial_state = RiskAssessmentState(application=application)

    # Thread ID allows LangGraph to track this run in the checkpointer
    # and resume it if needed. Use the application_id for traceability.
    config = {
        "configurable": {
            "thread_id": application.application_id
        }
    }

    # Execute the full workflow
    # LangGraph streams state updates — we print each step as it completes
    print(f"\n{'='*60}")
    print(f"Running SMB Risk Assessment — {scenario_name.upper()} scenario")
    print(f"Application: {application.business_name} ({application.application_id})")
    print(f"{'='*60}\n")

    final_state = None
    for step_update in app.stream(initial_state, config=config):
        node_name = list(step_update.keys())[0]
        print(f"▶ Node completed: {node_name}")

        # Show agent messages as they come in
        if "messages" in step_update.get(node_name, {}):
            for msg in step_update[node_name]["messages"]:
                print(f"  [{msg.role}] {msg.content}")
        print()

        final_state = step_update

    # Print the final report
    state_values = app.get_state(config).values
    if state_values.get("final_report"):
        print(state_values["final_report"])

    print(f"\n{'='*60}")
    print(f"Workflow complete. Check LangSmith for full trace:")
    print(f"  https://smith.langchain.com → Project: {os.getenv('LANGCHAIN_PROJECT', 'smb-risk-agent')}")
    print(f"{'='*60}\n")

    return state_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SMB risk assessment workflow")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="moderate",
        help="Test scenario to run"
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Use PostgreSQL checkpointer instead of in-memory"
    )
    args = parser.parse_args()
    run(args.scenario, args.prod)
