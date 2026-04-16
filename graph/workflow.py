"""
graph/workflow.py

The LangGraph StateGraph — this is the orchestration layer that connects
all agents into a single executable workflow with:

  - Defined nodes (each agent function)
  - Defined edges (fixed transitions between nodes)
  - A conditional edge (the critic loop routing decision)
  - LangSmith tracing (automatic via environment variables)
  - PostgreSQL checkpointing (state persistence across restarts)

GRAPH STRUCTURE:

  [START]
     │
     ▼
  ingestion_node          ← validate & load application
     │
     ▼
  assessment_node         ← Claude 3.5 Sonnet: score risk
     │
     ▼
  critic_node             ← Nova Pro: audit vs Basel III + CFPB
     │
     ├── route="retry"       → back to assessment_node (with feedback)
     ├── route="human_review" → human_review_node
     └── route="complete"    → report_node
                                    │
                                   [END]
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from graph.state import RiskAssessmentState
from agents.ingestion_agent import ingestion_node
from agents.assessment_agent import assessment_node
from agents.critic_agent import critic_node
from agents.human_review_node import human_review_node
from agents.report_node import report_node

load_dotenv()


# ---------------------------------------------------------------------------
# Conditional edge function — reads route_decision from state
# ---------------------------------------------------------------------------

def route_after_critic(state: RiskAssessmentState) -> str:
    """
    This function is called by LangGraph after the critic node runs.
    It reads state.route_decision and returns the name of the NEXT NODE.

    LangGraph uses the returned string to look up the next node in the
    graph — it must exactly match a node name registered below.
    """
    route = state.route_decision

    if route == "retry":
        return "assessment_node"
    elif route == "human_review":
        return "human_review_node"
    else:
        return "report_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    Constructs and compiles the LangGraph StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.
                      Pass a PostgresSaver instance for production,
                      or MemorySaver for local development/testing.

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    # Initialize the graph with our shared state class
    graph = StateGraph(RiskAssessmentState)

    # --- Register nodes ---
    # Each string name maps to a callable (our agent functions).
    # LangGraph calls these functions with the current state and
    # merges the returned dict back into the state.
    graph.add_node("ingestion_node", ingestion_node)
    graph.add_node("assessment_node", assessment_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("human_review_node", human_review_node)
    graph.add_node("report_node", report_node)

    # --- Register fixed edges ---
    # These transitions always happen — no branching logic.
    graph.add_edge(START, "ingestion_node")
    graph.add_edge("ingestion_node", "assessment_node")
    graph.add_edge("assessment_node", "critic_node")

    # --- Register the conditional edge (the critic loop) ---
    # After critic_node runs, call route_after_critic(state) to
    # determine which node to go to next.
    # The dict maps possible return values → node names.
    graph.add_conditional_edges(
        "critic_node",                   # source node
        route_after_critic,              # routing function
        {
            "assessment_node":   "assessment_node",    # retry
            "human_review_node": "human_review_node",  # escalate
            "report_node":       "report_node",        # complete
        }
    )

    # Terminal edges — both paths end the workflow
    graph.add_edge("report_node", END)
    graph.add_edge("human_review_node", END)

    # Compile with optional checkpointer
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Convenience: build with in-memory checkpointer for local dev
# ---------------------------------------------------------------------------

def build_dev_graph():
    """
    Builds the graph with an in-memory checkpointer.
    Use this for local development and testing — no PostgreSQL needed.
    State is lost when the process exits but the workflow runs correctly.
    """
    from langgraph.checkpoint.memory import MemorySaver
    return build_graph(checkpointer=MemorySaver())

def build_prod_graph():
    from db.checkpointer import get_prod_checkpointer
    return build_graph(checkpointer=get_prod_checkpointer())
