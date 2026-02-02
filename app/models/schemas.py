"""
Pydantic Models / Schemas
Request and response models for the API.
ENHANCED: With Phi-3 Reasoning Integration
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# ============ Request Models ============

class ProcessReportRequest(BaseModel):
    """Request to process a single adverse event report."""
    drugname: str = Field(..., description="Name of the drug")
    adverse_event: str = Field(..., description="Description of the adverse event")
    
    class Config:
        json_schema_extra = {
            "example": {
                "drugname": "Aspirin",
                "adverse_event": "Patient experienced severe bleeding after taking medication"
            }
        }


class BatchProcessRequest(BaseModel):
    """Request to process multiple adverse event reports."""
    reports: List[ProcessReportRequest]


class AuditQueryRequest(BaseModel):
    """Request for querying audit logs."""
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)
    risk_level: Optional[str] = Field(default=None, description="Filter by risk level")
    escalated_only: bool = Field(default=False)
    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD format")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD format")


# ============ Response Models ============

class EntityExtractionResult(BaseModel):
    """Extracted entities from the report."""
    drug: str
    symptoms: List[str]
    extraction_method: str = Field(description="regex or rule-based")


class MLClassificationResult(BaseModel):
    """ML classification result."""
    prediction: str = Field(description="Serious or Non-Serious")
    serious_probability: float
    non_serious_probability: float
    confidence: float


class EscalationDecision(BaseModel):
    """Escalation decision details."""
    should_escalate: bool
    risk_level: str = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    final_score: float
    ml_probability: float
    keyword_score: float
    triggered_keywords: List[str]
    explanation: str


class SimilarEventResult(BaseModel):
    """Similar historical serious event."""
    report_id: str
    drugname: str
    adverse_event: str
    timestamp: Optional[str] = None
    risk_level: Optional[str] = None
    ml_probability: Optional[float] = None
    final_score: Optional[float] = None
    matched_symptoms: Optional[List[str]] = None


class DrugInfo(BaseModel):
    """FDA Drug Information from DailyMed"""
    source: str
    indications: str
    warnings: str
    adverse_reactions: str


# NEW: Phi-3 Reasoning Result Model
class Phi3ReasoningResult(BaseModel):
    """Phi-3 medical reasoning analysis result."""
    reasoning_alignment: str = Field(
        ..., 
        description="How Phi-3's reasoning relates to ML decision: SUPPORTS, CHALLENGES, or UNAVAILABLE"
    )
    reasoning: str = Field(
        ..., 
        description="Medical reasoning explanation from Phi-3"
    )
    key_factors: List[str] = Field(
        default_factory=list, 
        description="Medical factors considered by Phi-3"
    )
    reasoning_certainty: str = Field(
        ..., 
        description="Self-reported certainty level: HIGH, MEDIUM, LOW, or UNKNOWN"
    )


class ProcessReportResponse(BaseModel):
    """Response for a processed adverse event report."""
    report_id: str
    drugname: str
    adverse_event: str
    entities: EntityExtractionResult
    classification: MLClassificationResult
    escalation: EscalationDecision
    processing_time_ms: float
    explanation: str = Field(description="AI-generated clinical explanation")
    email_notification: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Email notification status for high-risk cases"
    )
    similar_events: List[SimilarEventResult] = Field(
        default_factory=list,
        description="Similar historical serious events (vector-based retrieval)"
    )
    drug_info: Optional[DrugInfo] = Field(
        None,
        description="FDA drug information from DailyMed"
    )
    
    # NEW: Phi-3 Reasoning Fields
    phi3_reasoning: Optional[Phi3ReasoningResult] = Field(
        None,
        description="Medical reasoning analysis from Phi-3 (if available)"
    )
    needs_human_review: Optional[bool] = Field(
        None,
        description="Flag indicating if human review is recommended"
    )
    review_reason: Optional[str] = Field(
        None,
        description="Reason why human review is recommended"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "RPT-20240121120000-ABC12345",
                "drugname": "Aspirin",
                "adverse_event": "Patient experienced severe bleeding",
                "entities": {
                    "drug": "Aspirin",
                    "symptoms": ["bleeding"],
                    "extraction_method": "regex"
                },
                "classification": {
                    "prediction": "Serious",
                    "serious_probability": 0.85,
                    "non_serious_probability": 0.15,
                    "confidence": 0.85
                },
                "escalation": {
                    "should_escalate": True,
                    "risk_level": "HIGH",
                    "final_score": 0.82,
                    "ml_probability": 0.85,
                    "keyword_score": 0.7,
                    "triggered_keywords": ["bleeding"],
                    "explanation": "High-risk clinical indicators and hospitalization detected."
                },
                "processing_time_ms": 125.5,
                "explanation": "Based on the ML analysis (Serious probability: 85.2%), combined with triggered regulatory keywords: bleeding, this case is classified as HIGH risk.",
                "email_notification": {
                    "attempted": True,
                    "sent": True,
                    "recipient": "safety@company.com",
                    "timestamp": "2024-01-21T12:00:00"
                },
                "similar_events": [
                    {
                        "report_id": "AE-20240115-XYZ",
                        "drugname": "Aspirin",
                        "adverse_event": "Patient developed gastrointestinal bleeding",
                        "timestamp": "2024-01-15T10:30:00",
                        "risk_level": "HIGH",
                        "ml_probability": 0.82,
                        "final_score": 0.78,
                        "matched_symptoms": ["bleeding"]
                    }
                ],
                "drug_info": {
                    "source": "DailyMed",
                    "indications": "Pain relief, fever reduction",
                    "warnings": "Risk of bleeding, gastrointestinal issues",
                    "adverse_reactions": "Bleeding, stomach upset, allergic reactions"
                },
                "phi3_reasoning": {
                    "reasoning_alignment": "SUPPORTS",
                    "reasoning": "The event meets FDA seriousness criteria including severe bleeding requiring medical intervention.",
                    "key_factors": [
                        "Severe bleeding - life-threatening",
                        "Medical intervention required",
                        "Meets FDA criteria for serious adverse event"
                    ],
                    "reasoning_certainty": "HIGH"
                },
                "needs_human_review": False,
                "review_reason": None
            }
        }


class BatchProcessResponse(BaseModel):
    """Response for batch processing."""
    total_processed: int
    total_escalated: int
    results: List[ProcessReportResponse]
    total_processing_time_ms: float


class AuditLogRecord(BaseModel):
    """Single audit log record."""
    id: int
    report_id: str
    timestamp: str
    drugname: str
    adverse_event: str
    ml_prediction: str
    ml_probability: float
    extracted_drug: str
    extracted_symptoms: str
    escalation_decision: str
    risk_level: str
    final_score: float
    triggered_keywords: str
    explanation: str
    processing_time_ms: float


class AuditLogResponse(BaseModel):
    """Response for audit log query."""
    total: int
    records: List[Dict]


class AuditSummaryResponse(BaseModel):
    """Summary statistics from audit log."""
    total_processed: int
    by_risk_level: Dict[str, int]
    escalation_rate: float
    avg_processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """System health check response."""
    status: str
    model_loaded: bool
    llm_available: bool
    database_connected: bool
    phi3_available: Optional[bool] = Field(None, description="Whether Phi-3 reasoning is available")
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Information about the loaded model."""
    loaded: bool
    model_type: Optional[str] = None
    classes: Optional[List[str]] = None
    n_features: Optional[str] = None
    phi3_reasoning: Optional[Dict[str, Any]] = None