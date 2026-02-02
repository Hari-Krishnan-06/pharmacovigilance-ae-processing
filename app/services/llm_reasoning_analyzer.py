"""
Medical Reasoning Service
Provides medical reasoning and analysis for ML classifier decisions.
Does NOT validate or approve - only provides interpretability.

NOTE:
This implementation uses Gemma 2:2B via Ollama for MAXIMUM SPEED.
The ML classifier remains FINAL and AUTHORITATIVE.

MODEL: gemma2:2b (ultra-fast, 5–15 second response time)
"""

import requests
import json
from typing import Dict, Optional


class llmReasoningAnalyzer:
    """
    Medical reasoning analyzer using a local LLM.
    Analyzes adverse events and provides medical reasoning to support/challenge ML predictions.

    IMPORTANT:
    - This class does NOT validate, approve, or override ML decisions
    - ML prediction is FINAL
    - LLM output is used ONLY for interpretability and review prioritization
    """

    def __init__(
        self,
        model_url: str = "http://localhost:11434/api/generate",
        model_name: str = "gemma2:2b",
        use_streaming: bool = True,
        timeout: int = 30
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
        serious_probability: float
    ) -> Dict:
        """
        Provide medical reasoning for an ML prediction.
        Returns interpretability ONLY.
        """

        prompt = self._build_reasoning_prompt(
            drugname=drugname,
            adverse_event=adverse_event,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
            ml_reason=ml_reason,
            serious_probability=serious_probability
        )

        try:
            response = self._call_llm(prompt)
            return self._parse_llm_response(response)

        except Exception as e:
            return {
                "reasoning_alignment": "UNAVAILABLE",
                "reasoning": f"LLM reasoning unavailable: {str(e)}",
                "key_factors": [],
                "reasoning_certainty": "UNKNOWN",
                "error": str(e)
            }

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
        Build ultra-concise prompt optimized for Gemma2:2b.
        """
        return f"""Analyze this adverse event briefly:

Drug: {drugname}
Event: {adverse_event}
ML Says: {ml_prediction} ({ml_confidence:.0%} confidence)

Output ONLY this format:

REASONING_ALIGNMENT: SUPPORTS or CHALLENGES

REASONING: 1-2 sentences.

KEY_FACTORS:
- Factor 1
- Factor 2

REASONING_CERTAINTY: HIGH or MEDIUM or LOW

Response:""".strip()

    def _call_llm(self, prompt: str) -> str:
        """
        Call Ollama LLM with optimized settings for Gemma2:2b speed.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": self.use_streaming,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 20,
                "num_predict": 120,
                "num_ctx": 1024,
                "stop": ["\n\n", "---", "Response:", "Output:"]
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
        max_length = 500

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

    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response into structured format.
        """
        result = {
            "reasoning_alignment": "UNKNOWN",
            "reasoning": "",
            "key_factors": [],
            "reasoning_certainty": "UNKNOWN",
            "raw_response": response
        }

        try:
            if "REASONING_ALIGNMENT:" in response:
                line = response.split("REASONING_ALIGNMENT:")[1].split("\n")[0].strip().upper()
                if "SUPPORTS" in line:
                    result["reasoning_alignment"] = "SUPPORTS"
                elif "CHALLENGES" in line:
                    result["reasoning_alignment"] = "CHALLENGES"

            if "REASONING:" in response:
                section = response.split("REASONING:")[1]
                if "KEY_FACTORS:" in section:
                    result["reasoning"] = section.split("KEY_FACTORS:")[0].strip()
                else:
                    result["reasoning"] = section.strip()

            if "KEY_FACTORS:" in response:
                block = response.split("KEY_FACTORS:")[1]
                if "REASONING_CERTAINTY:" in block:
                    block = block.split("REASONING_CERTAINTY:")[0]

                result["key_factors"] = [
                    line.lstrip("-•* ").strip()
                    for line in block.splitlines()
                    if line.strip().startswith(("-", "•", "*"))
                ][:4]

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

    def get_combined_output(
        self,
        drugname: str,
        adverse_event: str,
        ml_result: Dict,
        phi3_result: Dict
    ) -> Dict:
        """
        Combine ML and LLM results into final assessment.
        """
        alignment = phi3_result.get("reasoning_alignment", "UNKNOWN")

        needs_review = (
            alignment in ["CHALLENGES", "UNKNOWN"]
            or ml_result.get("confidence", 0) < 0.7
            or phi3_result.get("reasoning_certainty") == "LOW"
        )

        return {
            "ml_decision": ml_result,
            "llm_reasoning": phi3_result,
            "final_assessment": {
                "decision": ml_result.get("prediction"),
                "needs_human_review": needs_review,
                "review_reason": self._get_review_reason(alignment, ml_result, phi3_result)
            }
        }

    def _get_review_reason(
        self,
        alignment: str,
        ml_result: Dict,
        phi3_result: Dict
    ) -> Optional[str]:
        if alignment == "CHALLENGES":
            return "LLM medical reasoning challenges the ML prediction"
        if alignment == "UNKNOWN":
            return "Unable to determine LLM reasoning alignment"
        if ml_result.get("confidence", 0) < 0.7:
            return "Low ML confidence"
        if phi3_result.get("reasoning_certainty") == "LOW":
            return "Low certainty in medical reasoning"
        return None
