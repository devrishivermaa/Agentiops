

---

# **AgenticOps: Distributed Multi-Agent Workflow for Document Processing**

AgenticOps is a scalable multi-agent system for large-scale PDF analysis. It uses a hierarchical mapper–reducer architecture powered by Ray, MongoDB, and LLM-driven agents to deliver fast, reliable, and consistent document understanding.


---
## System Architecture

<p align="center">
  <img src="Flowchart (1).jpg" width="90%">
</p>
<p align="center"><b>Distributed Multi-Agent Architecture</b></p>

## **Features**

* Multi-tier agent architecture with Master, SubMasters, Workers, and Residual Agent
* Parallel page-level processing with coordinated global context
* Automatic retries, lineage-based recovery, and checkpointing
* Dynamic resource allocation based on document complexity
* Real-time monitoring using an event-driven system
* Final JSON and PDF report generation with structured insights


---

## **System Architecture**

The system uses a distributed pipeline where:

* The Master Agent analyzes metadata and generates the execution plan
* SubMasters handle page-range processing and worker pools
* Workers perform LLM-based extraction
* The Residual Agent maintains global context and validates quality
* Merger Supervisor synthesizes final outputs




---

## **Agent Responsibilities**

### **Master Agent**

* Extracts PDF metadata
* Generates execution plan using Mistral
* Allocates SubMasters and manages approval workflow
* Persists planning data to MongoDB


### **SubMaster Agents**

* Spawn worker pools
* Handle page extraction and distribution
* Aggregate worker outputs at section level
* Report progress to orchestrator


### **Worker Agents**

* Process pages with LLM prompts using global context
* Extract entities, keywords, and summaries
* Use rate limiting and retry logic for stability


### **Residual Agent**

* Generates and distributes global context
* Validates output quality
* Handles anomaly detection and targeted retries


### **Merger Supervisor**

* Consolidates section-level outputs
* Resolves conflicts and unifies narrative
* Prepares final synthesis for reporting


---

## **Processing Pipeline**

AgenticOps follows an eight-stage workflow:

1. Metadata extraction
2. Master planning
3. SubMaster initialization
4. Worker pool creation
5. Parallel mapper execution
6. Quality validation
7. Hierarchical reduction
8. Report generation (JSON and PDF)


---

## **Fault Tolerance**

* Worker retries with exponential backoff
* SubMaster checkpoint recovery from MongoDB
* Backup Master failover
* Global rate limiter preventing overload
* Automatic replays using Ray lineage


---

## **Tech Stack**

* Ray Distributed Framework
* Mistral LLM
* MongoDB
* pdfplumber and PyPDF2
* ReportLab


---

## **Running the Pipeline**

### **Prerequisites**

* Python 3.10+
* Ray installed and running
* MongoDB instance
* Mistral API key

### **Start Processing**

```bash
npm run dev
python run_api.py
```

Outputs are stored as:

* `/output/report.json`
* `/output/report.pdf`

---

## **Contributors**

* Adhiraj Singh
* Dev Rishi Verma
* Nishant Raj


---


