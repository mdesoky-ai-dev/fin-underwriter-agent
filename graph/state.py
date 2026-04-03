"""
graph/state.py

The single shared state object that flows through every node in the graph.
LangGraph passes this state between agents — each node reads from it and
returns a partial update dict. The checkpointer serializes this to PostgreSQL
so the workflow can resume after a crash or restart.
"""

from typing import Optional, Annotated
from pydantic import BaseModel, Field
import operator

from models.application import SMBLoanApplication
from models.assessment import RiskAssessmentOutput
from models.audit import AuditResult


class AgentMessage(BaseModel):
    """A message in the inter-agent conversation log."""
    role: str          # "assessment_agent" | "critic_agent" | "system"
    content: str
    timestamp: Optional[str] = None


class RiskAssessmentState(BaseModel):
    """
    The full mutable state of one risk assessment workflow run.

    LangGraph merges partial update dicts returned by each node.
    The `messages` field uses `operator.add` as its reducer so messages
    accumulate rather than overwrite — this gives us a full audit trail
    of agent communications, visible in LangSmith.
    """

    # Core data
    application: Optional[SMBLoanApplication] = None
    assessment: Optional[RiskAssessmentOutput] = None
    audit: Optional[AuditResult] = None

    # Critic loop control
    retry_count: int = Field(default=0, description="Number of times assessment has been retried")
    max_retries: int = Field(default=3, description="Max critic loop iterations before hard escalation")

    # Inter-agent message log (append-only — drives LangSmith trace)
    # Annotated[list, operator.add] tells LangGraph to ADD new messages
    # rather than replace the whole list on each node update.
    messages: Annotated[list[AgentMessage], operator.add] = Field(default_factory=list)

    # Terminal outputs
    final_report: Optional[str] = None
    escalated_to_human: bool = False
    workflow_complete: bool = False

    # Routing signal written by the critic node, read by the conditional edge
    route_decision: Optional[str] = Field(
        None,
        description="Set by critic node: 'retry' | 'human_review' | 'complete'"
    )

    class Config:
        arbitrary_types_allowed = True
