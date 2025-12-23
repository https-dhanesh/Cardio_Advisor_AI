import logging
import pandas as pd
import numpy as np
import torch
from typing import List

from ai_core import models 
from schemas import PatientData, RiskResponse, AdviceResponse, ShapFactor
from prompts import build_llm_prompt

logger = logging.getLogger("cardio-advisor")

def get_risk_prediction_and_explanation(patient_data: PatientData) -> RiskResponse:

    if not (models.xgb_loaded and models.shap_loaded):
        raise Exception("Fast models (XGB/SHAP) are not loaded.")

    feature_names = PatientData.Config.feature_names
    patient_df = pd.DataFrame([patient_data.model_dump()], columns=feature_names)

    risk_prob = models.xgb_model.predict_proba(patient_df)[0, 1]
    risk_percent = float(risk_prob * 100)

    if risk_percent >= 70:
        risk_level, risk_emoji = "HIGH", "🔴"
    elif risk_percent >= 30:
        risk_level, risk_emoji = "MEDIUM", "🟡"
    else:
        risk_level, risk_emoji = "LOW", "🟢"

    top_factors = []
    try:
 
        shap_values = models.explainer(patient_df)

        shap_values_flat = shap_values.values[0]
        
        shap_dict = dict(zip(feature_names, shap_values_flat))
        top_3_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        for factor, impact in top_3_factors:
            value = getattr(patient_data, factor)
            top_factors.append(ShapFactor(
                factor=factor,
                value=value,

                impact=float(impact), 
                interpretation=models.get_interpretation(factor, value)
            ))
            
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")

        for factor in feature_names[:3]:
            value = getattr(patient_data, factor)
            top_factors.append(ShapFactor(
                factor=factor,
                value=value,
                impact=0.0,
                interpretation="SHAP unavailable"
            ))

    return RiskResponse(
        risk_score=round(risk_percent, 1),
        risk_level=risk_level,
        risk_emoji=risk_emoji,
        top_factors=top_factors,
        shap_available=models.shap_loaded
    )

def generate_ai_advice(patient_data: PatientData) -> AdviceResponse:
 
    if not models.llm_loaded:
        raise Exception("Slow models (LLM) are not loaded.")

    risk_data = get_risk_prediction_and_explanation(patient_data)

    rag_context = ""
    rag_used = False
    if models.retriever:
        try:

            query = f"management for {risk_data.risk_level} cardiovascular risk"
            docs = models.retriever.invoke(query)
            if docs:
                rag_context = " ".join([doc.page_content for doc in docs])
                rag_used = bool(rag_context.strip())
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    prompt_dict = build_llm_prompt(
        risk_data.risk_score,
        risk_data.risk_level,
        risk_data.top_factors,
        rag_context
    )

    try:

        inputs = models.tokenizer.apply_chat_template(
            prompt_dict,
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to("cuda")


        with torch.no_grad():
            outputs = models.model.generate(
                inputs,
                max_new_tokens=200,
                eos_token_id=models.llm_eos_token_id, 
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
                pad_token_id=models.tokenizer.pad_token_id,
                repetition_penalty=1.1
            )

        full_response = models.tokenizer.decode(outputs[0], skip_special_tokens=False)

        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        response_part = full_response.split(assistant_marker)[-1]

        ai_advice = response_part.replace("<|eot_id|>", "").strip()
        ai_advice = ai_advice.replace(models.tokenizer.eos_token, "").strip()

        if not ai_advice or len(ai_advice.split()) < 5:
            logger.warning("LLM generated empty response, using fallback.")
            ai_advice = _get_enhanced_medical_fallback(risk_data)

    except Exception as e:
        logger.error(f"LLM GENERATION FAILED: {e}")
        ai_advice = _get_enhanced_medical_fallback(risk_data)
    
    return AdviceResponse(
        ai_advice=ai_advice,
        llm_available=models.llm_loaded,
        rag_available=models.rag_loaded,
        rag_used=rag_used
    )

def _get_enhanced_medical_fallback(risk_data: RiskResponse) -> str:

    if risk_data.risk_level == "HIGH":
        return f"High cardiac risk ({risk_data.risk_score:.1f}%) requiring immediate comprehensive evaluation. Urgent cardiology assessment essential for diagnostic testing and aggressive risk factor modification."
    elif risk_data.risk_level == "MEDIUM":
        return f"Moderate cardiac risk ({risk_data.risk_score:.1f}%) indicating structured management. Schedule cardiology consultation within 2-4 weeks for evaluation, lifestyle modifications, and potential statin therapy."
    else:
        return f"Low cardiac risk ({risk_data.risk_score:.1f}%) is reassuring. Continue preventive cardiovascular care with healthy lifestyle habits, regular activity, and annual risk assessment with primary care provider."