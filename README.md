# SMB Working Capital Risk Assessment Agent

An autonomous multi-agent system that conducts financial risk assessments for small and medium business (SMB) working capital loans. The system uses a self-auditing critic loop to verify findings against Basel III capital standards and CFPB fair lending regulations before producing a final decision.

**Live Dashboard:** https://smb-risk-agent.netlify.app
**Live API:** https://fin-underwriter-agent.onrender.com
**API Docs:** https://fin-underwriter-agent.onrender.com/docs
**Health:** https://fin-underwriter-agent.onrender.com/health

---

## Architecture

```
[START]
   │
   ▼
ingestion_node          Validates application data, computes financial ratios,
                        enforces ECOA demographic firewall
   │
   ▼
assessment_node         Claude 3.7 Sonnet via AWS Bedrock
                        Produces risk score, PD/LGD/EAD, recommended action
   │
   ▼
critic_node             Amazon Nova Pro via AWS Bedrock
                        Audits assessment against Basel III + CFPB rules
   │
   ├── FAIL ──────────► assessment_node (with feedback, up to 3 retries)
   ├── REQUIRES REVIEW ► human_review_node
   └── PASS ───────────► report_node
                              │
                           [END]
```

---

## Key Features

**Autonomous critic loop** — the audit agent rejects and retries flawed assessments rather than passing bad output downstream. Failed audits inject specific feedback back into the assessment prompt so the model can self-correct.

**ECOA demographic firewall** — owner demographic data (race, ethnicity, sex) is collected for CFPB Section 1071 compliance but excluded from the credit decision. The critic verifies this firewall held on every run.

**Basel III Standardized Approach** — risk weights, expected loss, and capital requirements are computed and validated against regulatory thresholds. The critic catches any internal inconsistencies in the LLM's math.

**Pydantic schema validation** — every LLM output is validated against a strict schema before it touches the state. Score/category mismatches, missing conditions, and inconsistent calculations are caught at the boundary, not downstream.

**Full observability** — every node, token, and agent message is traced in LangSmith. Every state transition is checkpointed to PostgreSQL so workflows survive restarts.

**MCP integration** — the system is accessible as a Claude Desktop tool via MCP, allowing loan officers to run assessments through natural language conversation.

**REST API** — FastAPI wrapper with async workflow execution, status polling, and structured JSON responses. Deployed on Render.

**Loan officer dashboard** — financial terminal UI deployed on Netlify with three pre-loaded test scenarios and live Basel III metrics display.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Agent orchestration | LangGraph | Stateful cyclic workflow with conditional routing |
| Risk assessment LLM | Claude 3.7 Sonnet (AWS Bedrock) | Financial reasoning and risk scoring |
| Audit LLM | Amazon Nova Pro (AWS Bedrock) | Cost-efficient compliance verification |
| Data validation | Pydantic v2 | Schema enforcement and cross-field validation |
| State persistence | PostgreSQL + LangGraph checkpointer (Neon) | Crash-resilient workflow resumption |
| Observability | LangSmith | Full agent trace, token usage, latency per node |
| REST API | FastAPI + Uvicorn | Async HTTP wrapper, deployed on Render |
| MCP server | MCP SDK (FastMCP) | Claude Desktop tool integration |
| Frontend | HTML/CSS/JS | Loan officer dashboard, deployed on Netlify |
| Structured logging | structlog | JSON-formatted logs at every node |

---

## Regulatory Frameworks

**Basel III (Standardized Approach)**
- Probability of Default (PD) bounds per risk category
- Loss Given Default (LGD) estimation
- Risk weight: 0.75 for qualifying SMB revolving exposures under $1M, 1.0 otherwise
- Expected Loss = PD × LGD × EAD
- Capital Requirement = RWA × 8% (CET1 floor)

**CFPB / ECOA / Reg B**
- Section 1071 demographic data collection
- ECOA firewall — demographics excluded from credit decision
- Adverse action notice requirement (Reg B §202.9)
- Disparate treatment detection in rationale text

---

## Project Structure

```
fin-underwriter-agent/
│
├── agents/
│   ├── ingestion_agent.py      Data validation and ratio computation
│   ├── assessment_agent.py     Claude 3.7 Sonnet risk assessment
│   ├── critic_agent.py         Nova Pro compliance audit
│   ├── human_review_node.py    Escalation gate
│   └── report_node.py          Final report generation
│
├── graph/
│   ├── state.py                Shared RiskAssessmentState (LangGraph)
│   └── workflow.py             Graph construction and conditional routing
│
├── models/
│   ├── application.py          SMBLoanApplication Pydantic schema
│   ├── assessment.py           RiskAssessmentOutput + BaselIIIMetrics schemas
│   └── audit.py                AuditResult + BaselIIIAudit + CFPBAudit schemas
│
├── db/
│   └── checkpointer.py         PostgreSQL checkpointer (Neon)
│
├── agent-fe/
│   └── index.html              Loan officer dashboard (deployed on Netlify)
│
├── api.py                      FastAPI REST API (deployed on Render)
├── mcp_server.py               MCP server for Claude Desktop
├── main.py                     CLI entry point with three test scenarios
├── requirements-backend.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access (us-east-1)
- LangSmith account (free tier)
- PostgreSQL (optional — for production checkpointing)

### Installation

```bash
git clone https://github.com/mdesoky-ai-dev/fin-underwriter-agent.git
cd fin-underwriter-agent
pip install -r requirements-backend.txt
cp .env.example .env
```

### Configuration

Edit `.env` with your credentials:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
BEDROCK_ASSESSMENT_MODEL=us.anthropic.claude-3-7-sonnet-20250219-v1:0
BEDROCK_AUDIT_MODEL=amazon.nova-pro-v1:0
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=smb-risk-agent
DATABASE_URL=your_neon_connection_string
MAX_CRITIC_RETRIES=3
```

### Running

```bash
# CLI — moderate risk scenario (dev mode, in-memory checkpointer)
python main.py

# CLI — strong financials
python main.py --scenario strong

# CLI — critical risk decline
python main.py --scenario decline

# CLI — production mode with PostgreSQL checkpointer
python main.py --prod

# REST API
uvicorn api:app --reload --port 8000
# then open http://localhost:8000/docs

# MCP server (requires api.py running)
python mcp_server.py
```

---

## Sample Output

```
============================================================
SMB WORKING CAPITAL LOAN — RISK ASSESSMENT REPORT
============================================================
Application ID  : APP-2024-003
Business        : QuickFix Auto Repair
Generated       : 2026-04-03 17:01 UTC
Audit retries   : 0

DECISION
----------------------------------------
  ❌ DECLINED
  Risk category : CRITICAL
  Risk score    : 92/100
  Confidence    : 95%

BASEL III METRICS
----------------------------------------
  PD  (Prob. of Default)    : 28.00%
  LGD (Loss Given Default)  : 65.00%
  EAD (Exposure at Default) : $   85,000.00
  Risk Weight               : 0.75
  Expected Loss             : $   15,470.00
  Risk-Weighted Assets      : $   63,750.00
  Capital Requirement (8%)  : $    5,100.00

KEY RISK FACTORS
----------------------------------------
  1. DSCR of 0.43 — unable to service existing debt
  2. Current ratio of 0.32 — severe liquidity constraint
  3. Only 0.8 months cash runway
  4. 4 NSF events in last 12 months
  5. Requested loan ($85K) exceeds annual NOI ($18K) by 4.7x

COMPLIANCE AUDIT
----------------------------------------
  Audit verdict  : PASS
  Basel III      : PASS
  CFPB / ECOA    : PASS
  Violations     : 0
============================================================
```

---

## Design Decisions

**Why two models?** Claude 3.7 Sonnet handles the primary assessment because it requires complex financial reasoning across many variables. Amazon Nova Pro handles the audit because compliance checking is a more structured, bounded task — using a less expensive model here without sacrificing audit quality is a deliberate cost engineering decision.

**Why Pydantic before the critic?** LLM outputs are validated at the boundary of each node before they enter the state. This means the critic agent never sees malformed data — it only audits structurally valid assessments. Validation errors trigger retries independently of the compliance audit.

**Why LangGraph over a simple chain?** The critic loop requires cycles — a linear chain cannot express "retry if audit fails." LangGraph's stateful graph model with conditional edges makes the retry logic explicit, debuggable, and observable in LangSmith.

**Why PostgreSQL checkpointing?** Financial workflows must be resumable. If the system crashes between the assessment and audit nodes, the application should not be re-assessed from scratch — that could produce a different decision. Checkpointing guarantees exactly-once semantics per application.

**Why boto3 over LangChain's ChatBedrock?** Direct boto3 calls give explicit control over every API parameter and response format. In a regulated financial system, predictable behavior and full visibility into the request/response cycle is more important than provider portability.

**Why MCP?** MCP transforms the system from a developer tool into something any employee can access through natural language. A loan officer can request an assessment by describing the applicant in plain English — Claude handles data extraction and tool orchestration transparently.

---

## Roadmap

- [x] Multi-agent LangGraph workflow
- [x] Basel III + CFPB compliance audit with critic loop
- [x] Pydantic schema validation at every node boundary
- [x] PostgreSQL checkpointer (Neon) for production persistence
- [x] LangSmith observability
- [x] FastAPI REST API deployed on Render
- [x] MCP server for Claude Desktop integration
- [x] Loan officer dashboard deployed on Netlify
- [ ] Guardrails module — explicit Basel III + CFPB rule functions
- [ ] Batch processing for portfolio-level risk assessment
- [ ] Additional regulatory frameworks (FFIEC, CRA)
- [ ] Deploy MCP server publicly for web-based Claude integration
