# Pharmacovigilance Adverse Event Processing System - Complete Documentation

## 1. Project Overview

The **Pharmacovigilance Adverse Event Processing System** is an end-to-end solution designed to automate the intake, classification, and escalation of adverse event (AE) reports. It leverages Machine Learning (ML), Large Language Model (LLM) reasoning, and rule-based engines to ensure patient safety and regulatory compliance.

### Key Objectives
- **Automated Triage**: Classify AE reports as "Serious" vs. "Non-Serious" using a trained ML model.
- **Intelligent Extraction**: Extract drugs, symptoms, and other entities from unstructured text.
- **Risk Escalation**: Apply configurable rules to determine if a case requires immediate human intervention.
- **Medical Reasoning**: Use Phi-3 (LLM) to provide a "second opinion" and explain ML decisions.
- **Data Enrichment**: Integrate with FDA DailyMed and RxNorm for authoritative drug information.
- **Traceability**: Full audit logging of all processing steps and decisions.

---

## 2. System Architecture

The system follows a modern decoupled architecture:

### 2.1 Backend (`pharmacovigilance/app`)
- **Framework**: FastAPI (Python)
- **Role**: Core processing, API exposure, orchestration of services.
- **Key Services**:
    - `classifier.py`: Loads and runs the Logistic Regression ML model.
    - `entity_extractor.py`: Identifies drugs and symptoms.
    - `escalation_engine.py`: Evaluates risk based on ML score + keywords + drug warnings.
    - `llm_reasoning_analyzer.py`: Interface to Phi-3 for reasoning validation.
    - `rag_service.py`: Retrieval-Augmented Generation for processing PDF documents.
    - `audit_logger.py`: Logs every request and decision for compliance.

### 2.2 Frontend (`pharmacovigilance/pv-frontend`)
- **Framework**: Next.js 16 (React)
- **Styling**: Tailwind CSS
- **Role**: User interface for manual entry, PDF upload, and dashboard visualization.
- **Features**:
    - Real-time drug autocomplete (RxNorm).
    - Drag-and-drop PDF upload.
    - Visual risk indicators and confidence scores.
    - Detailed report generation and export.

### 2.3 Machine Learning (`pharmacovigilance/ml`)
- **Model**: Logistic Regression with TF-IDF Vectorization.
- **Training**: `train_classifier.py` script with stratified sampling and Langfuse experiment tracking.
- **Dataset**: FAERS (FDA Adverse Event Reporting System) data.

---

## 3. Key Features in Detail

### A. Core Processing Pipeline
1.  **Normalization**: Drug names are standardized using RxNorm.
2.  **Classification**: The ML model predicts `Serious` or `Non-Serious`.
3.  **Extraction**: Symptoms and conditions are extracted from the text.
4.  **Enrichment**: FDA DailyMed data (indications, warnings) is fetched.
5.  **Reasoning**: Phi-3 analyzes the case to support or challenge the ML prediction.
6.  **Escalation**: Rules determine if the case is Critical, High, Medium, or Low risk.

### B. RAG & PDF Processing
- Users can upload PDF medical records.
- The system extracts text (with OCR fallback options).
- Reports are auto-associated with detected drugs.
- Content is indexed for semantic search (RAG) to find similar cases.

### C. Observability
- **Langfuse Integration**: Tracks ML model performance, latency, and LLM reasoning steps.
- **Audit Logs**: Secure logs of who processed what, when, and the outcome.

---

## 4. Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **Git**

### 4.1 Backend Setup

1.  Navigate to the project root:
    ```bash
    cd pharmacovigilance
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Download NLTK data:
    ```bash
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
    ```

5.  Set up Environment Variables (create a `.env` file):
    ```env
    GROQ_API_KEY=your_groq_api_key
    LANGFUSE_PUBLIC_KEY=your_key
    LANGFUSE_SECRET_KEY=your_secret
    LANGFUSE_HOST=https://cloud.langfuse.com
    ```

6.  Train the ML Model (if not present):
    ```bash
    python ml/train_classifier.py
    ```

7.  Start the Backend Server:
    ```bash
    uvicorn app.main:app --reload
    ```
    *API will be available at `http://localhost:8000`*

### 4.2 Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd pv-frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the Development Server:
    ```bash
    npm run dev
    ```
    *App will be available at `http://localhost:3000`*

---

## 5. API Reference

### Processing
- `POST /api/process`: Process a manual text report.
- `POST /api/process-pdf`: Upload and process a PDF file.

### Authentication
- `POST /api/auth/signup`: Register a new user.
- `POST /api/auth/login`: Login and retrieve JWT token.
- `GET /api/auth/me`: Get current user details.

### Audit & System
- `GET /api/audit`: Retrieve paginated audit logs.
- `GET /api/drugs/suggest`: Autocomplete for drug names.
- `GET /api/health`: System health status (DB, ML model, LLM).
- `GET /api/model/info`: Details about the loaded ML model.

---

## 6. Directory Structure

```
pharmacovigilance/
├── app/                        # Backend Application
│   ├── services/               # Business Logic (ML, RAG, Auth, etc.)
│   ├── models/                 # Pydantic Schemas & DB Models
│   ├── database/               # Database Connection & Logic
│   └── main.py                 # FastAPI Entry Point
├── ml/                         # Machine Learning
│   ├── trained_models/         # Saved .pkl models
│   ├── train_classifier.py     # Training Script
│   └── experiments/            # Experiment Logs
├── pv-frontend/                # Next.js Frontend
│   ├── app/                    # App Router Pages
│   └── components/             # Reusable UI Components
├── data/                       # Dataset Storage
└── requirements.txt            # Python Dependencies
```

## 7. Troubleshooting

- **Model not loading**: Ensure you ran `python ml/train_classifier.py` and that `model.pkl` exists in `ml/`.
- **Frontend connection error**: Verify `NEXT_PUBLIC_API_BASE_URL` in frontend `.env` matches your backend URL.
- **LLM errors**: Check your `GROQ_API_KEY` validity.
