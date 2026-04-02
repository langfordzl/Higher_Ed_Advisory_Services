# Databricks notebook source
# MAGIC %md
# MAGIC # Higher Education Advisory Services — AI Agent Pipeline
# MAGIC
# MAGIC ## Complete Setup & User Guide
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### What This Project Does
# MAGIC
# MAGIC This project deploys an **AI-powered quality analysis agent** for a higher education call center.
# MAGIC The agent can:
# MAGIC
# MAGIC - **Find** audio recordings of student advisory calls (financial aid, admissions, enrollment, etc.)
# MAGIC - **Transcribe** calls using OpenAI Whisper large-v3 speech recognition
# MAGIC - **Analyze** transcripts with AI: sentiment analysis, topic extraction, intent classification, call categorization
# MAGIC - **Score** advisor performance against a weighted 5-criterion rubric using RAG (Retrieval-Augmented Generation)
# MAGIC - **Report** on pipeline status (how many files transcribed, how many enriched)
# MAGIC
# MAGIC You interact with the agent through **natural language** -- just ask it questions like:
# MAGIC - *"What audio files are available?"*
# MAGIC - *"Transcribe speaker 5"*
# MAGIC - *"Run a full quality analysis on this transcript: [paste text]"*
# MAGIC - *"What's the average rubric score for Financial Aid calls?"*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Architecture Overview
# MAGIC
# MAGIC ```
# MAGIC +-----------------------------------------------------------------+
# MAGIC |                        DATA FLOW (Medallion)                    |
# MAGIC |                                                                 |
# MAGIC |   Audio Files (.wav)          UC Volume                         |
# MAGIC |        |                      /Volumes/chada_demos/pubsec_demos |
# MAGIC |        v                      /audio/                           |
# MAGIC |   +----------+                                                  |
# MAGIC |   |  BRONZE   |  Auto Loader -> file metadata                  |
# MAGIC |   +----+-----+                                                  |
# MAGIC |        v                                                        |
# MAGIC |   +----------+                                                  |
# MAGIC |   |  SILVER   |  Whisper large-v3 -> text transcriptions       |
# MAGIC |   +----+-----+                                                  |
# MAGIC |        v                                                        |
# MAGIC |   +----------+                                                  |
# MAGIC |   |   GOLD    |  LLM enrichment -> sentiment, topics,          |
# MAGIC |   |           |  call category, rubric scores (1-5)            |
# MAGIC |   +----+-----+                                                  |
# MAGIC |        v                                                        |
# MAGIC |   AI Agent Endpoint  <->  AI Playground / REST API / Genie     |
# MAGIC +-----------------------------------------------------------------+
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Technology Stack
# MAGIC
# MAGIC | Component | What It Is | How It's Used Here |
# MAGIC |-----------|-----------|-------------------|
# MAGIC | **Unity Catalog** | Databricks' data governance layer | Stores all tables, functions, and the model under `chada_demos.higher_ed_advisory` |
# MAGIC | **Delta Tables** | Versioned, ACID-compliant data tables | Three tables: `bronze_audio_files`, `silver_transcriptions`, `gold_enriched_calls` |
# MAGIC | **UC Volumes** | Managed file storage in Unity Catalog | Stores the 301 `.wav` audio files |
# MAGIC | **UC Functions** | SQL functions registered in Unity Catalog | 12 functions (all SQL) that the agent calls as tools |
# MAGIC | **ai_query()** | Built-in Databricks SQL function | Calls AI model endpoints (Whisper for STT, Llama for analysis) directly from SQL |
# MAGIC | **Model Serving** | Hosts ML models as REST endpoints | Deploys the agent as a scalable REST API with scale-to-zero |
# MAGIC | **LangGraph** | Framework for building AI agents | Manages the agent's tool-calling loop |
# MAGIC | **MLflow** | ML lifecycle management | Logs, versions, and deploys the agent model |
# MAGIC | **Auto Loader** | Incremental file ingestion | Streams new audio file metadata into the bronze table |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC Before you begin, make sure you have:
# MAGIC
# MAGIC ### 1. A Databricks Workspace
# MAGIC You need access to a Databricks workspace with **Unity Catalog** enabled.
# MAGIC This project is configured for: `https://e2-demo-field-eng.cloud.databricks.com`
# MAGIC
# MAGIC ### 2. A Compute Cluster
# MAGIC You need a cluster to run the setup notebooks:
# MAGIC - **Access mode**: `Single user` (required for UC function registration)
# MAGIC - **Databricks Runtime**: `15.4 LTS` or newer
# MAGIC - **Node type**: `i3.xlarge` or similar (single node is fine)
# MAGIC
# MAGIC ### 3. A SQL Warehouse
# MAGIC Several UC functions use `ai_query()` which runs on serverless SQL compute.
# MAGIC Default warehouse ID: `4b9b953939869799`
# MAGIC
# MAGIC ### 4. Model Serving Endpoints (Already Available)
# MAGIC
# MAGIC | Endpoint | Model | Purpose |
# MAGIC |----------|-------|---------|
# MAGIC | `databricks-claude-3-7-sonnet` | Claude 3.7 Sonnet | Agent reasoning and tool orchestration |
# MAGIC | `databricks-meta-llama-3-3-70b-instruct` | Llama 3.3 70B | Sentiment analysis, topic extraction, rubric scoring |
# MAGIC | `va_whisper_large_v3` | Whisper large-v3 | Audio speech-to-text transcription |
# MAGIC
# MAGIC ### 5. Audio Files
# MAGIC The project expects `.wav` audio files in a Unity Catalog Volume:
# MAGIC `/Volumes/chada_demos/pubsec_demos/audio/` (301 pre-loaded recordings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step-by-Step Setup
# MAGIC
# MAGIC ### Quick Start (3 notebooks, run in order)
# MAGIC
# MAGIC | Step | Notebook | Time | What It Does |
# MAGIC |------|----------|------|-------------|
# MAGIC | **1** | `01_setup` | ~3 min | Creates the database, tables, rubric data, and registers all 12 SQL functions |
# MAGIC | **2** | `02_deploy` | ~15 min | Ingests audio metadata, packages the agent, deploys it as a live REST endpoint |
# MAGIC | **3** (optional) | `03_test` | ~5 min | Runs comprehensive E2E test suite (pre-deploy + post-deploy) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 1: Run `01_setup`
# MAGIC
# MAGIC **What this does:**
# MAGIC - Creates the Unity Catalog schema `chada_demos.higher_ed_advisory`
# MAGIC - Creates three Delta tables: `bronze_audio_files`, `silver_transcriptions`, `gold_enriched_calls`
# MAGIC - Creates the `advisor_rubric` reference table and seeds it with 5 scoring criteria
# MAGIC - Registers all 12 Unity Catalog SQL functions (the AI tools the agent will use)
# MAGIC
# MAGIC **How to run it:**
# MAGIC 1. Open the `01_setup` notebook
# MAGIC 2. Attach it to your cluster
# MAGIC 3. Click **Run All**
# MAGIC 4. You should see "Registered: [function_name]" for each of the 12 functions
# MAGIC
# MAGIC **Expected output (last cell):**
# MAGIC ```
# MAGIC All UC Functions in chada_demos.higher_ed_advisory:
# MAGIC   - chada_demos.higher_ed_advisory.find_audio_file
# MAGIC   - chada_demos.higher_ed_advisory.find_all_audio_files
# MAGIC   - chada_demos.higher_ed_advisory.read_audio_base64
# MAGIC   - chada_demos.higher_ed_advisory.transcribe_audio
# MAGIC   - chada_demos.higher_ed_advisory.classify_call_category
# MAGIC   - chada_demos.higher_ed_advisory.analyze_call_sentiment
# MAGIC   - chada_demos.higher_ed_advisory.extract_topics_and_intent
# MAGIC   - chada_demos.higher_ed_advisory.assess_rubric_rag
# MAGIC   - chada_demos.higher_ed_advisory.transcribe_and_save_to_silver
# MAGIC   - chada_demos.higher_ed_advisory.process_all_audio_to_silver
# MAGIC   - chada_demos.higher_ed_advisory.enrich_silver_to_gold
# MAGIC   - chada_demos.higher_ed_advisory.enrich_single_call
# MAGIC
# MAGIC All 12 functions registered successfully.
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 2: Run `02_deploy`
# MAGIC
# MAGIC **What this does (9 stages):**
# MAGIC 1. Creates a Volume for audio file storage
# MAGIC 2. Runs Auto Loader to ingest audio file metadata into the bronze table
# MAGIC 3. Writes `agent.py` -- the agent's code (LangGraph + Claude + 10 tools)
# MAGIC 4. Tests the agent locally on your cluster (pre-deploy smoke test)
# MAGIC 5. Logs the agent model to MLflow with all resource declarations
# MAGIC 6. Registers the model in Unity Catalog
# MAGIC 7. Deploys the agent as a serving endpoint (REST API)
# MAGIC 8. Runs post-deployment validation tests
# MAGIC 9. Sets up Vector Search index (optional)
# MAGIC
# MAGIC **How to run it:**
# MAGIC 1. Open the `02_deploy` notebook
# MAGIC 2. Attach it to the same cluster
# MAGIC 3. Click **Run All**
# MAGIC 4. **Be patient during Stages 7-8** -- endpoint deployment takes 5-10 minutes
# MAGIC
# MAGIC **Expected final output:**
# MAGIC ```
# MAGIC ==========================================================
# MAGIC   HIGHER EDUCATION ADVISORY SERVICES -- PIPELINE SUMMARY
# MAGIC ==========================================================
# MAGIC
# MAGIC   Catalog/Schema:  chada_demos.higher_ed_advisory
# MAGIC   Agent Model:     main.higher_ed_advisory.higher_ed_advisory_agent
# MAGIC   Endpoint:        higher_ed_advisory_agent
# MAGIC
# MAGIC   +----------+-----------+
# MAGIC   |  Layer   |  Records  |
# MAGIC   +----------+-----------+
# MAGIC   |  Bronze  |  301       |
# MAGIC   |  Silver  |  0         |
# MAGIC   |  Gold    |  0         |
# MAGIC   +----------+-----------+
# MAGIC ```
# MAGIC
# MAGIC > **Note:** Silver and Gold are 0 because transcription hasn't been run yet.
# MAGIC > Ask the agent to transcribe files after deployment.
# MAGIC
# MAGIC **Redeploy only?** If you're iterating on the agent (changing tools or system prompt),
# MAGIC skip to the "Redeploy Only" section at the bottom of `02_deploy` instead of re-running the full pipeline.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step 3 (Optional): Run `03_test`
# MAGIC
# MAGIC **What this does:**
# MAGIC - **Phase 1 (Pre-deploy):** Validates table schemas, rubric data, UC function registration (all 12), mock transformations, agent tool wiring (all 10 tools), direct SQL function tests
# MAGIC - **Phase 2 (Post-deploy):** Tests the live endpoint -- health check, tool invocation for every function, gold data quality
# MAGIC
# MAGIC **How to run it:**
# MAGIC 1. Open `03_test`
# MAGIC 2. Set the `endpoint_name` widget to: `higher_ed_advisory_agent`
# MAGIC 3. Click **Run All**
# MAGIC
# MAGIC **Expected output:**
# MAGIC ```
# MAGIC   Total:   40+
# MAGIC   Passed:  40+
# MAGIC   Failed:  0
# MAGIC   Pass Rate: 100.0%
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use the Agent
# MAGIC
# MAGIC Once the deploy notebook completes, your agent is live. Here are four ways to interact with it:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Option A: AI Playground (Easiest -- No Code Required)
# MAGIC
# MAGIC 1. In the left sidebar, click **Playground**
# MAGIC 2. In the **Endpoint** dropdown, select: `higher_ed_advisory_agent`
# MAGIC 3. Type a message and press Enter
# MAGIC
# MAGIC **Example prompts:**
# MAGIC
# MAGIC | What to ask | What happens |
# MAGIC |------------|-------------|
# MAGIC | "What audio files are available?" | Agent calls `find_all_audio_files` -> returns list of 301 .wav files |
# MAGIC | "Find the audio file for speaker 5" | Agent calls `find_audio_file` -> returns the file path |
# MAGIC | "Transcribe speaker 5" | Agent calls `find_audio_file` then `transcribe_and_save_to_silver` -> returns transcript |
# MAGIC | "Check the transcription pipeline status" | Agent calls `process_all_audio_to_silver` -> shows counts |
# MAGIC | "Classify this call: I'm calling about my application status..." | Agent calls `classify_call_category` -> returns "Admissions" |
# MAGIC | "Analyze the sentiment: I am really frustrated..." | Agent calls `analyze_call_sentiment` -> returns "Negative, 90% confidence" |
# MAGIC | "Score this call using the rubric: Good morning, this is Maria..." | Agent calls `assess_rubric_rag` -> returns 4/5 with criterion scores |
# MAGIC | "Run a full analysis on this transcript: [paste text]" | Agent calls `enrich_single_call` -> returns all analysis in one response |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Option B: REST API (For Developers)

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC import requests
# MAGIC
# MAGIC WORKSPACE_URL = "https://e2-demo-field-eng.cloud.databricks.com"
# MAGIC TOKEN = "<your-personal-access-token>"
# MAGIC
# MAGIC url = f"{WORKSPACE_URL}/serving-endpoints/higher_ed_advisory_agent/invocations"
# MAGIC headers = {
# MAGIC     "Authorization": f"Bearer {TOKEN}",
# MAGIC     "Content-Type": "application/json"
# MAGIC }
# MAGIC
# MAGIC payload = {
# MAGIC     "messages": [
# MAGIC         {
# MAGIC             "role": "user",
# MAGIC             "content": "Find and transcribe speaker 12, then run a full quality analysis."
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC
# MAGIC response = requests.post(url, json=payload, headers=headers)
# MAGIC result = response.json()
# MAGIC
# MAGIC for msg in result.get("messages", []):
# MAGIC     if msg.get("role") == "assistant":
# MAGIC         print(msg["content"])
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option C: Databricks SDK (For Python Scripts)

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC response = w.serving_endpoints.query(
# MAGIC     name="higher_ed_advisory_agent",
# MAGIC     messages=[
# MAGIC         {"role": "user", "content": "What audio files are available?"}
# MAGIC     ],
# MAGIC )
# MAGIC print(response)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option D: Genie Space (For Business Analysts)
# MAGIC
# MAGIC After calls are transcribed and enriched (gold table has data):
# MAGIC
# MAGIC 1. In the left sidebar, click **Genie**
# MAGIC 2. Click **New Genie Space**
# MAGIC 3. Add tables: `chada_demos.higher_ed_advisory.gold_enriched_calls` + `advisor_rubric`
# MAGIC 4. Example queries: "Average rubric score by category?", "Show negative sentiment calls"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Table Reference
# MAGIC
# MAGIC ### bronze_audio_files
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | `filename` | STRING | Original filename (e.g., `Speaker_0005_00000.wav`) |
# MAGIC | `file_path` | STRING | Full Volume path to the audio file |
# MAGIC | `file_size_bytes` | LONG | Size of the audio file in bytes |
# MAGIC | `modified_time` | TIMESTAMP | When the file was last modified in cloud storage |
# MAGIC | `ingested_at` | TIMESTAMP | When Auto Loader cataloged the file |
# MAGIC
# MAGIC ### silver_transcriptions
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | `filename` | STRING | Original audio filename |
# MAGIC | `file_path` | STRING | Full Volume path |
# MAGIC | `speaker_id` | STRING | Extracted speaker identifier |
# MAGIC | `transcription` | STRING | Full text transcription from Whisper |
# MAGIC | `word_count` | INT | Number of words |
# MAGIC | `duration_hint` | STRING | `short` (<100 words), `medium` (<500), or `long` (500+) |
# MAGIC | `transcribed_at` | TIMESTAMP | When the transcription was completed |
# MAGIC
# MAGIC ### gold_enriched_calls
# MAGIC | Column | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | `sentiment` | STRING | `Positive`, `Negative`, `Neutral`, or `Mixed` |
# MAGIC | `sentiment_confidence` | DOUBLE | AI confidence 0.0 to 1.0 |
# MAGIC | `topics` | STRING | JSON array of key topics |
# MAGIC | `intent` | STRING | Primary reason the student called |
# MAGIC | `call_category` | STRING | Financial Aid, Admissions, Enrollment, etc. |
# MAGIC | `rubric_score` | INT | Weighted advisor score 1-5 |
# MAGIC | `rubric_assessment` | STRING | Narrative assessment |
# MAGIC | `improvement_areas` | STRING | JSON array of improvement suggestions |
# MAGIC
# MAGIC ### advisor_rubric
# MAGIC | Criterion | Weight | Score 1 (Poor) | Score 5 (Excellent) |
# MAGIC |-----------|--------|----------------|---------------------|
# MAGIC | Greeting & Identification | 15% | No greeting | Warm greeting; confirms name, ID, reason |
# MAGIC | Active Listening | 20% | Interrupts; ignores | Paraphrases; clarifying questions |
# MAGIC | Accurate Information | 25% | Incorrect info | Fully accurate with citations |
# MAGIC | Empathy & Tone | 20% | Dismissive | Warm, empathetic, validates feelings |
# MAGIC | Resolution & Next Steps | 20% | No resolution | Full resolution with deadlines |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Tools Reference
# MAGIC
# MAGIC The agent has access to **10 tools** (backed by 12 UC SQL functions).
# MAGIC
# MAGIC ### Discovery & File Management
# MAGIC | Tool | What It Does | Example Prompt |
# MAGIC |------|-------------|----------------|
# MAGIC | `find_audio_file(speaker_query)` | Finds a specific speaker's audio file | "Find speaker 5" |
# MAGIC | `find_all_audio_files()` | Lists every .wav file in the Volume | "What files do we have?" |
# MAGIC
# MAGIC ### Transcription
# MAGIC | Tool | What It Does | Example Prompt |
# MAGIC |------|-------------|----------------|
# MAGIC | `transcribe_and_save_to_silver(file_path)` | Transcribes one audio file with Whisper | "Transcribe speaker 5" |
# MAGIC | `process_all_audio_to_silver()` | Shows transcription status: total/done/pending | "Check transcription status" |
# MAGIC
# MAGIC ### Analysis
# MAGIC | Tool | What It Does | Example Prompt |
# MAGIC |------|-------------|----------------|
# MAGIC | `classify_call_category(transcription)` | Classifies into 9 categories | "Classify this call: [text]" |
# MAGIC | `analyze_call_sentiment(transcription)` | Returns sentiment + confidence | "What's the sentiment of: [text]" |
# MAGIC | `extract_topics_and_intent(transcription)` | Extracts topics, intent, improvements | "Extract topics from: [text]" |
# MAGIC | `assess_rubric_rag(transcription)` | Scores advisor 1-5 on each criterion | "Score this call: [text]" |
# MAGIC | `enrich_single_call(transcription)` | Runs ALL of the above in one call | "Full analysis of: [text]" |
# MAGIC
# MAGIC ### Pipeline Status
# MAGIC | Tool | What It Does | Example Prompt |
# MAGIC |------|-------------|----------------|
# MAGIC | `enrich_silver_to_gold()` | Reports silver vs gold counts | "Enrichment pipeline status?" |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Reference
# MAGIC
# MAGIC | Notebook | Purpose | When to Run |
# MAGIC |----------|---------|------------|
# MAGIC | **README** (this file) | Documentation and setup guide | Read first |
# MAGIC | **01_setup** | Creates schema, tables, rubric, and all 12 SQL UC functions | Step 1 of initial setup |
# MAGIC | **02_deploy** | Full pipeline: ingest -> agent -> deploy -> validate (+ redeploy section) | Step 2 of initial setup |
# MAGIC | **03_test** | Comprehensive E2E test suite (40+ tests in 2 phases) | Optional, after any deployment |
# MAGIC | **agent.py** | Agent source code (auto-generated by 02_deploy) | Deployed automatically |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC ### "Endpoint deployment timed out"
# MAGIC - Go to **Serving** -> click `higher_ed_advisory_agent` -> check the **Events** tab
# MAGIC - If stuck, delete the endpoint in the UI and re-run the deployment cells in `02_deploy`
# MAGIC
# MAGIC ### "PERMISSION_DENIED" errors from the deployed agent
# MAGIC The serving endpoint runs under a **service principal** that needs access:
# MAGIC 1. Go to **Serving** -> click `higher_ed_advisory_agent` -> **AI Gateway** tab
# MAGIC 2. Find the Service Principal ID
# MAGIC 3. Grant permissions:
# MAGIC
# MAGIC ```sql
# MAGIC GRANT USE CATALOG ON CATALOG chada_demos TO `<sp-id>`;
# MAGIC GRANT USE SCHEMA ON SCHEMA chada_demos.higher_ed_advisory TO `<sp-id>`;
# MAGIC GRANT USE SCHEMA ON SCHEMA chada_demos.pubsec_demos TO `<sp-id>`;
# MAGIC GRANT READ VOLUME ON VOLUME chada_demos.pubsec_demos.audio TO `<sp-id>`;
# MAGIC GRANT EXECUTE ON SCHEMA chada_demos.higher_ed_advisory TO `<sp-id>`;
# MAGIC GRANT SELECT ON SCHEMA chada_demos.higher_ed_advisory TO `<sp-id>`;
# MAGIC ```
# MAGIC
# MAGIC ### Agent gives wrong or empty responses
# MAGIC - Check all 12 functions: `SHOW USER FUNCTIONS IN chada_demos.higher_ed_advisory`
# MAGIC - Check endpoint model version: **Serving** -> endpoint -> **Served entities**
# MAGIC - Check service principal permissions
# MAGIC
# MAGIC ### How to redeploy after making changes
# MAGIC 1. Edit the `agent_code` variable in `02_deploy`
# MAGIC 2. Skip to the **"Redeploy Only"** section at the bottom
# MAGIC 3. Run that cell -- it logs, registers, and updates the endpoint
# MAGIC 4. Wait ~5 minutes for the new version to deploy
# MAGIC 5. Run `03_test` to validate

# COMMAND ----------

# MAGIC %md
# MAGIC ## E2E Test Results (Verified 2026-02-27)
# MAGIC
# MAGIC All 10 agent tools tested against the live endpoint `higher_ed_advisory_agent` (model v3):
# MAGIC
# MAGIC | # | Tool | Test | Result |
# MAGIC |---|------|------|--------|
# MAGIC | 1 | `find_all_audio_files` | List all audio files | **PASS** -- 301 files |
# MAGIC | 2 | `enrich_single_call` | Full enrichment on transcript | **PASS** -- Sentiment: Neutral, Rubric: 4/5 |
# MAGIC | 3 | `find_audio_file` | Find Speaker 5 | **PASS** |
# MAGIC | 4 | `classify_call_category` | Admissions inquiry | **PASS** -- "Admissions" |
# MAGIC | 5 | `process_all_audio_to_silver` | Pipeline status | **PASS** -- 301 total |
# MAGIC | 6 | `enrich_silver_to_gold` | Enrichment status | **PASS** |
# MAGIC | 7 | `analyze_call_sentiment` | Frustrated caller | **PASS** -- Negative, 90% |
# MAGIC | 8 | `assess_rubric_rag` | Financial aid call | **PASS** -- 4/5 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Reference
# MAGIC
# MAGIC | Widget | Default Value | Description |
# MAGIC |--------|---------------|-------------|
# MAGIC | `catalog` | `chada_demos` | Unity Catalog name |
# MAGIC | `schema` | `higher_ed_advisory` | Schema within the catalog |
# MAGIC | `volume_path` | `/Volumes/chada_demos/pubsec_demos/audio` | Audio file storage |
# MAGIC | `warehouse_id` | `4b9b953939869799` | SQL Warehouse for `ai_query()` |
# MAGIC | `whisper_endpoint` | `va_whisper_large_v3` | Speech-to-text endpoint |
# MAGIC | `llm_endpoint` | `databricks-meta-llama-3-3-70b-instruct` | LLM for analysis |
# MAGIC | `agent_llm_endpoint` | `databricks-claude-3-7-sonnet` | LLM for agent reasoning |
# MAGIC | `embedding_endpoint` | `databricks-gte-large-en` | Embedding model for Vector Search |
# MAGIC | `vector_search_endpoint` | `one-env-shared-endpoint-1` | Vector Search endpoint |
# MAGIC | `endpoint_name` | `higher_ed_advisory_agent` | Deployed serving endpoint |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Project Links
# MAGIC
# MAGIC | Resource | URL |
# MAGIC |----------|-----|
# MAGIC | Workspace | https://e2-demo-field-eng.cloud.databricks.com |
# MAGIC | Agent Endpoint | https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/higher_ed_advisory_agent |
# MAGIC | Catalog Explorer | https://e2-demo-field-eng.cloud.databricks.com/explore/data/chada_demos/higher_ed_advisory |
# MAGIC | AI Playground | https://e2-demo-field-eng.cloud.databricks.com/ml/playground |
# MAGIC | This Folder | https://e2-demo-field-eng.cloud.databricks.com/workspace/Users/chad.ammirati@databricks.com/Higher_Ed_Advisory_Services |
