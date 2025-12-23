from typing import List, Dict, Any
from schemas import ShapFactor

def build_llm_prompt(
    risk_percent: float, 
    risk_level: str, 
    top_factors: List[ShapFactor], 
    rag_context: str
) -> List[Dict[str, str]]:

    system_prompt = (
        "You are a professional medical expert and cardiologist. "
        "Your task is to provide concise, direct, and actionable medical recommendations "
        "based on a patient's risk profile. "
        "Do NOT use conversational filler, questions, or disclaimers. "
        "Respond ONLY with the 2-3 sentence medical advice."
    )

    factors_text = ""
    for factor in top_factors:
        factors_text += f"- {factor.factor} (Value: {factor.value}): {factor.interpretation}\n"

    user_prompt = f"""CARDIOLOGY CONSULTATION REQUEST

PATIENT DATA:
- Risk Score: {risk_percent:.1f}%
- Risk Level: {risk_level}

KEY RISK FACTORS:
{factors_text}
ADDITIONAL CONTEXT:
{rag_context}

Required: Provide ONLY the direct medical recommendations in 2-3 sentences."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]