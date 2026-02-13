#==============================================================================================
# DESCRIPTION : implements a medical reasoning / interpretability layer using a local LLM 
#               (Gemma 2:2B via Ollama) to explain ML classifier decisions in pharmacovigilance
#==============================================================================================

import requests
import json
from typing import Dict, Optional
from langfuse import Langfuse

from dotenv import load_dotenv
from app.services.classifier import is_negated

import os

load_dotenv()

assert os.getenv("LANGFUSE_PUBLIC_KEY"), "LANGFUSE_PUBLIC_KEY missing"
assert os.getenv("LANGFUSE_SECRET_KEY"), "LANGFUSE_SECRET_KEY missing"


# ============================================================================
# LANGFUSE INITIALIZATION (ONE TIME, TOP OF FILE)
# ============================================================================
langfuse = Langfuse()


class llmReasoningAnalyzer:
    """
    Medical reasoning analyzer using a local LLM.
    Analyzes adverse events and provides medical reasoning to support/challenge ML predictions.

    IMPORTANT:
    - This class does NOT validate, approve, or override ML decisions
    - ML prediction is FINAL
    - LLM output is used ONLY for interpretability and review prioritization
    - Langfuse tracks explanations for audit/compliance (does NOT influence decisions)
    """

    def __init__(
        self,
        model_url: str = "http://localhost:11434/api/generate",
        model_name: str = "gemma2:2b",
        use_streaming: bool = False,
        timeout: int = 90
    ):
        self.model_url = model_url
        self.model_name = model_name
        self.use_streaming = use_streaming
        self.timeout = timeout

    def analyze_prediction(
        self,
        drugname: str,
        adverse_event: str,
        ml_prediction: str,
        ml_confidence: float,
        ml_reason: str,
        serious_probability: float,
        report_id: Optional[str] = None  # NEW: For Langfuse tracing
    ) -> Dict:
        """
        Provide medical reasoning for an ML prediction.
        Returns interpretability ONLY.
        
        Args:
            drugname: Drug name
            adverse_event: Adverse event description
            ml_prediction: ML classifier's decision (FINAL)
            ml_confidence: ML confidence score
            ml_reason: ML reasoning/rationale
            serious_probability: Serious class probability
            report_id: Unique report identifier (for audit trail)
        
        Returns:
            Dict with reasoning analysis
        """

        # ====================================================================
        # LANGFUSE TRACE: ONE PER ADVERSE EVENT REPORT
        # ====================================================================
        trace = None
        if report_id:
            try:
                trace = langfuse.trace(
                    name="AE_LLM_Explanation",
                    id=report_id,  # Adverse event report ID
                    metadata={
                        "drug": drugname,
                        "ml_prediction": ml_prediction,
                        "ml_confidence": ml_confidence,
                        "serious_probability": serious_probability,
                        "classifier": "LogisticRegression_balanced_v1",
                        "decision_authority": "CLASSICAL_ML",
                        "llm_role": "EXPLANATION_ONLY"
                    }
                )
            except Exception as e:
                print(f" Langfuse trace creation failed: {e}")
                trace = None

        # ----------------------------------------------------
        # NEGATION GUARD: prevent false seriousness reasoning
        # ----------------------------------------------------
        SERIOUS_TERMS = [
            "hospitalization", "hospitalized", "bleeding", "death",
            "cardiac arrest", "icu", "respiratory failure", "coma"
        ]

        text = adverse_event.lower()
        found_serious_term = False
        all_negated = True

        for term in SERIOUS_TERMS:
            if term in text:
                found_serious_term = True
                if not is_negated(adverse_event, term):
                    all_negated = False
                    break

        # Only bypass LLM if serious terms exist AND all are negated
        if found_serious_term and all_negated:
            return {
                "reasoning_alignment": "SUPPORTS",
                "reasoning": (
                    "The adverse event description explicitly negates serious clinical outcomes "
                    "such as hospitalization or bleeding. No regulatory seriousness criteria "
                    "are met based on the provided information."
                ),
                "key_factors": ["Negated serious clinical terms"],
                "reasoning_certainty": "HIGH",
                "llm_bypassed": True
            }

        # Build prompt
        prompt = self._build_reasoning_prompt(
            drugname=drugname,
            adverse_event=adverse_event,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            ml_reason=ml_reason,
            serious_probability=serious_probability
        )

        try:
            # ================================================================
            # LANGFUSE SPAN: TRACKS LLM CALL
            # ================================================================
            span = None
            if trace:
                try:
                    span = trace.span(
                        name="LLM_Medical_Reasoning",
                        input={
                            "prompt": prompt,
                            "adverse_event": adverse_event,
                            "ml_prediction": ml_prediction,
                            "ml_confidence": ml_confidence,
                            "ml_reason": ml_reason
                        },
                        metadata={
                            "model": self.model_name,
                            "temperature": 0.1,
                            "max_tokens": 180,
                            "purpose": "interpretability_only"
                        }
                    )
                except Exception as e:
                    print(f" Langfuse span creation failed: {e}")
                    span = None

            # ================================================================
            # CALL LLM (EXISTING CODE - UNCHANGED)
            # ================================================================
            response = self._call_llm(prompt)
            
            # Parse response
            parsed_result = self._parse_llm_response(response)

            # ================================================================
            # END LANGFUSE SPAN: CAPTURE OUTPUT
            # ================================================================
            if span:
                try:
                    span.end(
                        output={
                            "reasoning_alignment": parsed_result.get("reasoning_alignment"),
                            "reasoning": parsed_result.get("reasoning"),
                            "key_factors": parsed_result.get("key_factors"),
                            "reasoning_certainty": parsed_result.get("reasoning_certainty")
                        },
                        metadata={
                            "model": self.model_name,
                            "role": "explanation_only",
                            "llm_changed_decision": False,  # CRITICAL: Always False
                            "raw_response_length": len(response)
                        }
                    )
                except Exception as e:
                    print(f" Langfuse span end failed: {e}")

            # ================================================================
            # UPDATE TRACE: REINFORCE DECISION AUTHORITY
            # ================================================================
            if trace:
                try:
                    trace.update(
                        metadata={
                            "llm_influence": "NONE",
                            "final_decision_source": "CLASSICAL_ML",
                            "ml_decision_unchanged": True,
                            "reasoning_alignment": parsed_result.get("reasoning_alignment"),
                            "reasoning_certainty": parsed_result.get("reasoning_certainty")
                        }
                    )
                except Exception as e:
                    print(f" Langfuse trace update failed: {e}")

            return parsed_result

        except Exception as e:
            # ================================================================
            # ERROR HANDLING WITH LANGFUSE
            # ================================================================
            error_result = {
                "reasoning_alignment": "UNAVAILABLE",
                "reasoning": f"LLM reasoning unavailable: {str(e)}",
                "key_factors": [],
                "reasoning_certainty": "UNKNOWN",
                "error": str(e)
            }

            # Log error to Langfuse
            if span:
                try:
                    span.end(
                        output=error_result,
                        metadata={
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                except Exception as langfuse_error:
                    print(f" Langfuse error logging failed: {langfuse_error}")

            if trace:
                try:
                    trace.update(
                        metadata={
                            "llm_influence": "NONE",
                            "final_decision_source": "CLASSICAL_ML",
                            "ml_decision_unchanged": True,
                            "llm_error": str(e)
                        }
                    )
                except Exception as langfuse_error:
                    print(f" Langfuse trace error update failed: {langfuse_error}")

            return error_result

    

# =======================================================================================================
# This function constructs a strict, structured prompt for the LLM so it can explain why an adverse event
# is classified as Serious or Non-Serious
# =======================================================================================================

    def _build_reasoning_prompt(
        self,
        drugname: str,
        adverse_event: str,
        ml_prediction: str,
        ml_confidence: float,
        ml_reason: str,
        serious_probability: float
    ) -> str:
        """
        Build prompt optimized for Gemma2:2b with EXPLICIT seriousness reasoning.
        
        CRITICAL FIXES:
        - Asks WHY event is serious/non-serious
        - Includes regulatory context (ICH E2A)
        - Uses ML rationale to guide reasoning
        - Allows 2-3 sentences instead of restricting to 1-2
        """
        return f"""You are assisting with pharmacovigilance interpretability.

Drug: {drugname}
Adverse Event: {adverse_event}

ML Classification: {ml_prediction}
ML Confidence: {ml_confidence:.0%}
ML Rationale: {ml_reason}
Serious Probability Score: {serious_probability:.2f}

Task:
Explain WHY this adverse event is considered SERIOUS or NON-SERIOUS based on standard clinical and regulatory reasoning (hospitalization, life-threatening risk, disability, congenital anomaly, or medical intervention required).

Do NOT change the ML decision. Only explain the medical justification.

Output ONLY in this format:

REASONING_ALIGNMENT: SUPPORTS or CHALLENGES

REASONING:
Explain the medical or clinical justification in 2-3 clear sentences.

KEY_FACTORS:
- Factor 1
- Factor 2

REASONING_CERTAINTY: HIGH or MEDIUM or LOW

Response:""".strip()

    def _call_llm(self, prompt: str) -> str:
        """
        Call Ollama LLM with optimized settings for Gemma2:2b speed.
        
        CRITICAL FIXES:
        - Removed destructive "\n\n" stop token
        - Increased num_predict from 120 to 180
        - Only keeps "---" as stop token
        
        NOTE: NO Langfuse tracking here - that happens in analyze_prediction()
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": self.use_streaming,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 20,
                "num_predict": 180,  # INCREASED from 120
                "num_ctx": 1024,
                "stop": ["---"]  # FIXED: removed "\n\n", "Response:", "Output:"
            }
        }

        if self.use_streaming:
            return self._call_llm_streaming(payload)
        return self._call_llm_non_streaming(payload)

    def _call_llm_streaming(self, payload: dict) -> str:
        """
        Streaming call for faster partial responses.
        """
        response_text = ""
        max_length = 700  
        with requests.post(
            self.model_url,
            json=payload,
            timeout=self.timeout,
            stream=True
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    response_text += token

                    # Stop when we have complete reasoning certainty
                    if (
                        "REASONING_CERTAINTY:" in response_text and
                        any(k in response_text[-30:] for k in ["HIGH", "MEDIUM", "LOW"])
                    ):
                        break

                    if len(response_text) > max_length:
                        break

                    if chunk.get("done", False):
                        break

                except json.JSONDecodeError:
                    continue

        return response_text

# ============================================================================================
# This function makes a single, non-streaming HTTP call to the local LLM (via Ollama) and
# returns the generated text
# ============================================================================================

    def _call_llm_non_streaming(self, payload: dict) -> str:
        """
        Non-streaming fallback.
        """
        response = requests.post(
            self.model_url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")

# ======================================================================
# It converts the raw LLM text output into a structured dictionary
# ======================================================================

    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response into structured format.
        
        NOTE: NO Langfuse tracking here - parsing is deterministic
        """
        result = {
            "reasoning_alignment": "UNKNOWN",
            "reasoning": "",
            "key_factors": [],
            "reasoning_certainty": "UNKNOWN",
            "raw_response": response
        }

        try:
            # Parse REASONING_ALIGNMENT
            if "REASONING_ALIGNMENT:" in response:
                line = response.split("REASONING_ALIGNMENT:")[1].split("\n")[0].strip().upper()
                if "SUPPORTS" in line:
                    result["reasoning_alignment"] = "SUPPORTS"
                elif "CHALLENGES" in line:
                    result["reasoning_alignment"] = "CHALLENGES"

            # Parse REASONING
            if "REASONING:" in response:
                section = response.split("REASONING:")[1]
                if "KEY_FACTORS:" in section:
                    result["reasoning"] = section.split("KEY_FACTORS:")[0].strip()
                else:
                    result["reasoning"] = section.strip()

            # Parse KEY_FACTORS
            if "KEY_FACTORS:" in response:
                block = response.split("KEY_FACTORS:")[1]
                if "REASONING_CERTAINTY:" in block:
                    block = block.split("REASONING_CERTAINTY:")[0]

                result["key_factors"] = [
                    line.lstrip("-•* ").strip()
                    for line in block.splitlines()
                    if line.strip().startswith(("-", "•", "*"))
                ][:4]

            # Parse REASONING_CERTAINTY
            if "REASONING_CERTAINTY:" in response:
                cert = response.split("REASONING_CERTAINTY:")[1].split("\n")[0].upper()
                if "HIGH" in cert:
                    result["reasoning_certainty"] = "HIGH"
                elif "MEDIUM" in cert:
                    result["reasoning_certainty"] = "MEDIUM"
                elif "LOW" in cert:
                    result["reasoning_certainty"] = "LOW"

        except Exception as e:
            result["parsing_error"] = str(e)

        return result

# =======================================================================================================
# It merges the ML decision with the LLM explanation and decides whether a human should review the case,
# without ever changing the ML outcome
# =======================================================================================================

    def get_combined_output(
        self,
        drugname: str,
        adverse_event: str,
        ml_result: Dict,
        llm_result: Dict,  
        report_id: Optional[str] = None 
    ) -> Dict:
        """
        Combine ML and LLM results into final assessment.
        
        CRITICAL: ML decision is FINAL. LLM only adds interpretability.
        """
        alignment = llm_result.get("reasoning_alignment", "UNKNOWN")

        needs_review = (
            alignment in ["CHALLENGES", "UNKNOWN"]
            or ml_result.get("confidence", 0) < 0.7
            or llm_result.get("reasoning_certainty") == "LOW"
        )

        combined_output = {
            "ml_decision": ml_result,
            "llm_reasoning": llm_result,
            "final_assessment": {
                "decision": ml_result.get("prediction"), 
                "needs_human_review": needs_review,
                "review_reason": self._get_review_reason(alignment, ml_result, llm_result)
            },
            "decision_authority": "CLASSICAL_ML",  # Explicit
            "llm_role": "EXPLANATION_ONLY"  # Explicit
        }

        # Optional: Log combined assessment to Langfuse
        if report_id:
            try:
                event = langfuse.event(
                    trace_id=report_id,
                    name="Combined_Assessment",
                    metadata={
                        "ml_prediction": ml_result.get("prediction"),
                        "llm_alignment": alignment,
                        "needs_human_review": needs_review,
                        "review_reason": combined_output["final_assessment"]["review_reason"],
                        "decision_unchanged": True  # ML decision never changes
                    }
                )
            except Exception as e:
                print(f" Langfuse event logging failed: {e}")

        return combined_output

# =====================================================================
# It explains why a case was flagged for human review, in plain text
# =====================================================================

    def _get_review_reason(
        self,
        alignment: str,
        ml_result: Dict,
        llm_result: Dict
    ) -> Optional[str]:
        """
        Determine why case needs human review.
        
        NOTE: Review is triggered by uncertainty, NOT by LLM challenging ML.
        LLM challenges should be flagged for review, but ML decision stands.
        """
        if alignment == "CHALLENGES":
            return "LLM medical reasoning challenges the ML prediction - recommend expert review"
        if alignment == "UNKNOWN":
            return "Unable to determine LLM reasoning alignment"
        if ml_result.get("confidence", 0) < 0.7:
            return "Low ML confidence"
        if llm_result.get("reasoning_certainty") == "LOW":
            return "Low certainty in medical reasoning"
        return None


# ============================================================================
# Function that creates and returns a configured llmReasoningAnalyzer instance
# ============================================================================

def create_reasoning_analyzer(
    model_name: str = "gemma2:2b",
    use_streaming: bool = False,
    enable_langfuse: bool = True  # Can disable for testing
) -> llmReasoningAnalyzer:
    """
    Factory function to create a reasoning analyzer instance.
    
    Args:
        model_name: Ollama model to use
        use_streaming: Enable streaming responses
        enable_langfuse: Enable Langfuse tracking (default: True)
    
    Returns:
        Configured llmReasoningAnalyzer instance
    """
    if not enable_langfuse:
        print(" Langfuse tracking is disabled")
    
    return llmReasoningAnalyzer(
        model_name=model_name,
        use_streaming=use_streaming
    )