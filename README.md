# Pharmacovigilance Adverse Event Processing System

An end-to-end system for ingesting, classifying, and escalating adverse event reports.

## Features
- ML-based seriousness classification (Logistic Regression + TF-IDF)
- LLM-assisted entity extraction (Ollama/Qwen 2.5)
- Rule-based escalation engine
- Audit logging for regulatory compliance
- FastAPI backend + Streamlit frontend

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Train the classifier
python ml/train_classifier.py

# Start the API server
uvicorn app.main:app --reload

# Start the Streamlit frontend (in another terminal)
streamlit run frontend/streamlit_app.py
```

## Project Structure
```
pharmacovigilance/
├── app/                      # FastAPI application
│   ├── main.py              # API endpoints
│   ├── models/schemas.py    # Pydantic models
│   ├── services/            # Business logic
│   └── database/            # SQLite connection
├── ml/                       # ML model training
├── frontend/                 # Streamlit UI
├── data/                     # Dataset location
└── requirements.txt
```

## API Endpoints
- `POST /api/process` - Process single AE report
- `POST /api/batch` - Process multiple reports
- `GET /api/audit` - Retrieve audit logs
- `GET /api/health` - System health check
