# Higher Education Advisory Services — AI Agent Pipeline

An AI-powered quality analysis agent for higher education call centers, built on Databricks with Unity Catalog, LangGraph, and MLflow.

## What It Does

This project deploys an AI agent that can:

- **Find** audio recordings of student advisory calls (financial aid, admissions, enrollment, etc.)
- **Transcribe** calls using OpenAI Whisper large-v3 speech recognition
- **Analyze** transcripts with AI — sentiment analysis, topic extraction, intent classification, call categorization
- **Score** advisor performance against a weighted 5-criterion rubric using RAG
- **Report** on pipeline status across bronze/silver/gold layers

Interact with the agent through natural language:
- *"What audio files are available?"*
- *"Transcribe speaker 5"*
- *"Run a full quality analysis on this transcript"*
- *"What's the average rubric score for Financial Aid calls?"*

## Architecture

```
+------------------------------------------------------------------+
|                     DATA FLOW (Medallion)                         |
|                                                                   |
|   Audio Files (.wav)         UC Volume                            |
|        |                     /Volumes/.../audio/                   |
|        v                                                          |
|   +----------+                                                    |
|   |  BRONZE  |  Auto Loader -> file metadata                     |
|   +----+-----+                                                    |
|        v                                                          |
|   +----------+                                                    |
|   |  SILVER  |  Whisper large-v3 -> text transcriptions          |
|   +----+-----+                                                    |
|        v                                                          |
|   +----------+                                                    |
|   |   GOLD   |  LLM enrichment -> sentiment, topics,            |
|   |          |  call category, rubric scores (1-5)               |
|   +----+-----+                                                    |
|        v                                                          |
|   AI Agent Endpoint  <->  AI Playground / REST API / Genie       |
+------------------------------------------------------------------+
```

## Technology Stack

| Component | How It's Used |
|-----------|--------------|
| **Unity Catalog** | Stores all tables, functions, and the model under `chada_demos.higher_ed_advisory` |
| **Delta Tables** | Three tables: `bronze_audio_files`, `silver_transcriptions`, `gold_enriched_calls` |
| **UC Volumes** | Stores `.wav` audio files |
| **UC Functions** | 12 SQL functions the agent calls as tools |
| **ai_query()** | Calls Whisper (STT) and Llama (analysis) directly from SQL |
| **Model Serving** | Deploys the agent as a scalable REST API with scale-to-zero |
| **LangGraph** | Manages the agent's tool-calling loop |
| **MLflow** | Logs, versions, and deploys the agent model |
| **Auto Loader** | Incremental audio file metadata ingestion |

## Prerequisites

1. **Databricks Workspace** with Unity Catalog enabled
2. **Compute Cluster** — Single user access mode, DBR 15.4 LTS+
3. **SQL Warehouse** — for `ai_query()` serverless execution
4. **Model Serving Endpoints:**

| Endpoint | Model | Purpose |
|----------|-------|---------|
| `databricks-claude-3-7-sonnet` | Claude 3.7 Sonnet | Agent reasoning and tool orchestration |
| `databricks-meta-llama-3-3-70b-instruct` | Llama 3.3 70B | Sentiment, topic extraction, rubric scoring |
| `va_whisper_large_v3` | Whisper large-v3 | Audio speech-to-text |

5. **Audio Files** — `.wav` files in a UC Volume

## Quick Start

Run the three notebooks in order:

| Step | Notebook | Time | What It Does |
|------|----------|------|-------------|
| 1 | `01_setup.py` | ~3 min | Creates schema, tables, rubric data, and 12 SQL UC functions |
| 2 | `02_deploy.py` | ~15 min | Ingests audio metadata, packages agent, deploys as REST endpoint |
| 3 | `03_test.py` | ~5 min | Runs 40+ E2E tests (pre-deploy + post-deploy) |

### Step 1: Setup

```
01_setup.py
```
- Creates `chada_demos.higher_ed_advisory` schema
- Creates bronze, silver, gold Delta tables + `advisor_rubric` reference table
- Registers all 12 UC SQL functions

### Step 2: Deploy

```
02_deploy.py
```
- Runs Auto Loader for audio file metadata ingestion
- Packages the LangGraph agent with MLflow
- Deploys as a model serving endpoint
- Runs post-deployment validation

### Step 3: Test (Optional)

```
03_test.py
```
- Phase 1: Validates schemas, rubric data, UC functions, agent tool wiring
- Phase 2: Tests live endpoint — health check, tool invocation, data quality

## Using the Agent

### AI Playground (No Code)

1. Open **Playground** in the Databricks sidebar
2. Select endpoint: `higher_ed_advisory_agent`
3. Start chatting

### REST API

```python
import requests

url = f"{WORKSPACE_URL}/serving-endpoints/higher_ed_advisory_agent/invocations"
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}
payload = {
    "messages": [
        {"role": "user", "content": "Find and transcribe speaker 12, then run a full quality analysis."}
    ]
}
response = requests.post(url, json=payload, headers=headers)
```

### Databricks SDK

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
response = w.serving_endpoints.query(
    name="higher_ed_advisory_agent",
    messages=[{"role": "user", "content": "What audio files are available?"}],
)
```

### Genie Space (Business Analysts)

After calls are transcribed and enriched, create a Genie Space with the `gold_enriched_calls` and `advisor_rubric` tables for natural language querying.

## Agent Tools

### Discovery
| Tool | Description |
|------|------------|
| `find_audio_file(speaker_query)` | Find a specific speaker's audio file |
| `find_all_audio_files()` | List all `.wav` files in the Volume |

### Transcription
| Tool | Description |
|------|------------|
| `transcribe_and_save_to_silver(file_path)` | Transcribe one audio file with Whisper and save to silver |
| `process_all_audio_to_silver()` | Show transcription status: total/done/pending |

### Analysis
| Tool | Description |
|------|------------|
| `classify_call_category(transcription)` | Classify into 9 higher-ed categories |
| `analyze_call_sentiment(transcription)` | Sentiment label + confidence score |
| `extract_topics_and_intent(transcription)` | Key topics, intent, improvement areas |
| `assess_rubric_rag(transcription)` | Score advisor 1-5 on weighted rubric criteria |
| `enrich_single_call(transcription)` | Run all analysis tools in one call |

### Pipeline
| Tool | Description |
|------|------------|
| `enrich_silver_to_gold()` | Report silver vs gold enrichment status |

## Delta Table Schemas

### bronze_audio_files
| Column | Type | Description |
|--------|------|-------------|
| `filename` | STRING | Original filename (e.g., `Speaker_0005_00000.wav`) |
| `file_path` | STRING | Full Volume path |
| `file_size_bytes` | LONG | File size in bytes |
| `modified_time` | TIMESTAMP | Last modified in cloud storage |
| `ingested_at` | TIMESTAMP | Auto Loader ingestion timestamp |

### silver_transcriptions
| Column | Type | Description |
|--------|------|-------------|
| `filename` | STRING | Original audio filename |
| `file_path` | STRING | Full Volume path |
| `speaker_id` | STRING | Extracted speaker identifier |
| `transcription` | STRING | Full Whisper transcription |
| `word_count` | INT | Word count |
| `duration_hint` | STRING | `short` / `medium` / `long` |
| `transcribed_at` | TIMESTAMP | Transcription timestamp |

### gold_enriched_calls
| Column | Type | Description |
|--------|------|-------------|
| `sentiment` | STRING | Positive / Negative / Neutral / Mixed |
| `sentiment_confidence` | DOUBLE | Confidence 0.0–1.0 |
| `topics` | STRING | Comma-separated topics |
| `intent` | STRING | Primary caller intent |
| `call_category` | STRING | Financial Aid, Admissions, Enrollment, etc. |
| `rubric_score` | INT | Weighted advisor score 1–5 |
| `rubric_assessment` | STRING | Narrative assessment |
| `improvement_areas` | STRING | Suggested improvements |

### Advisor Rubric

| Criterion | Weight | Score 1 (Poor) | Score 5 (Excellent) |
|-----------|--------|----------------|---------------------|
| Greeting & Identification | 15% | No greeting | Warm greeting; confirms name, ID, reason |
| Active Listening | 20% | Interrupts; ignores | Paraphrases; clarifying questions |
| Accurate Information | 25% | Incorrect info | Fully accurate with citations |
| Empathy & Tone | 20% | Dismissive | Warm, empathetic, validates feelings |
| Resolution & Next Steps | 20% | No resolution | Full resolution with deadlines |

## Troubleshooting

**Endpoint deployment timed out** — Check **Serving > Events** tab. Delete and redeploy if stuck.

**PERMISSION_DENIED errors** — The serving endpoint's service principal needs UC grants:
```sql
GRANT USE CATALOG ON CATALOG chada_demos TO `<sp-id>`;
GRANT USE SCHEMA ON SCHEMA chada_demos.higher_ed_advisory TO `<sp-id>`;
GRANT EXECUTE ON SCHEMA chada_demos.higher_ed_advisory TO `<sp-id>`;
GRANT SELECT ON SCHEMA chada_demos.higher_ed_advisory TO `<sp-id>`;
```

**Redeploying after changes** — Use the "Redeploy Only" section at the bottom of `02_deploy.py` instead of re-running the full pipeline.

## Files

| File | Purpose |
|------|---------|
| `README.py` | Databricks notebook documentation |
| `01_setup.py` | Schema, tables, rubric, and 12 UC function registration |
| `02_deploy.py` | Full pipeline: ingest, package agent, deploy endpoint |
| `03_test.py` | 40+ E2E tests across pre-deploy and post-deploy phases |
| `agent.py` | LangGraph agent source code (Claude + 10 tools) |
