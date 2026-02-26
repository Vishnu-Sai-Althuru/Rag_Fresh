# AI/ML BOTP Training -- Tasks Repository

## Project Overview

This repository contains AI/ML engineering tasks completed as part of
the **BOTP Training Program**. The project demonstrates practical
implementation of modern AI system components including:

-   Prompt Engineering
-   Retrieval-Augmented Generation (RAG)
-   AI Agents
-   Memory Systems
-   Task Orchestration
-   Evaluation Pipelines
-   Cost Tracking & Batching
-   Observability Dashboards

The tasks are organized week-wise to reflect progressive learning and
implementation.

------------------------------------------------------------------------

## Project Structure

Tasks-main/
│
├── Week_1/
│   ├── Task_1 – Prompt Engineering
│   ├── Task_2 – RAG Pipeline
│   ├── Task_3 – Evaluation Framework
│   └── Task_4 – Batching & Cost Tracking
│
├── Week_2/
│   ├── Task_1 – AI Agent + RAG
│   ├── Task_2 – Memory Systems (Short/Long Memory)
│   ├── Task_3 – Task Orchestration
│   └── Task_4 – Monitoring Dashboard
│
├── Week_3/
│   ├── Task_1 – Advanced RAG + Vector DB
│   ├── Task_2 – LangGraph Agent
│   ├── Task_3 – Evaluation Pipeline
│   └── Task_4 – Metrics Dashboard
│
├── memory_db/
├── requirements.txt
└── test.py

------------------------------------------------------------------------

## Features

### ✅ Prompt Engineering
- Template-based prompt design
- Structured input handling

### ✅ RAG (Retrieval-Augmented Generation)
- PDF document ingestion
- Text splitting & embeddings
- Vector database storage (ChromaDB)
- Context-aware querying

### ✅ AI Agents
- Tool-based agent workflows
- LangGraph agent orchestration
- Multi-step reasoning

### ✅ Memory Systems
- Short-term memory
- Long-term persistent memory
- SQLite-based storage

### ✅ Task Orchestration
- Planner + Executor architecture
- Retry handling
- Execution pipelines

### ✅ Evaluation Framework
- Dataset evaluation
- Metrics calculation
- Performance scoring

### ✅ Cost Optimization
- Token usage tracking
- Request batching
- Caching layer implementation

### ✅ Observability
- Metrics dashboards
- Logging system
- Cost reporting

------------------------------------------------------------------------

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| AI Frameworks | LangChain, LangGraph |
| Vector DB | ChromaDB |
| LLM Integration | Custom LLM Client |
| Data Processing | NumPy, Pandas |
| Storage | SQLite |
| Evaluation | Custom Metrics Framework |
| Visualization | Dashboard Scripts |
| Environment | Virtual Environment (venv) |

------------------------------------------------------------------------

## Setup Instructions

### Clone Repository

git clone <repo-url>
cd Tasks-main

### Create Virtual Environment

python -m venv venv

## Activate environment:

Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

### Install Dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

## Usage

Run Example Task: python Week_2/Task_1/main.py

Run Evaluation: python Week_3/Task_3/run_eval.py

Launch Dashboard: python Week_3/Task_4/run_demo.py

------------------------------------------------------------------------

## Known Issues

Batch processing optimization under testing phase.

------------------------------------------------------------------------

## Author

Vishnu Sai

------------------------------------------------------------------------

## Submission Notes

This repository is submitted as proof of completion for BOTP AI/ML training tasks.
