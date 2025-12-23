import os
import logging
import pandas as pd
import torch
import xgboost as xgb
import shap
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional, Any
from threading import Lock

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except Exception:
    Chroma = None
    SentenceTransformerEmbeddings = None

logger = logging.getLogger("cardio-advisor")

class AIModels:

    def __init__(self):
        self.PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.retriever = None
        self.model = None
        self.tokenizer = None
        self.llm_eos_token_id: Optional[int] = None
        self.xgb_loaded = False
        self.shap_loaded = False
        self.rag_loaded = False
        self.llm_loaded = False
  
        self.factor_interpretations = {
            "thal": {
                3: "normal blood flow", 4: "normal blood flow",
                5: "possible fixed defect", 6: "possible fixed defect",
                7: "reversible defect (ischemia)"
            },
            "ca": {
                0: "no significant CAD", 1: "mild single-vessel disease",
                2: "moderate multi-vessel disease",
                3: "severe multi-vessel disease", 4: "severe multi-vessel disease"
            },
            "cp": {
                0: "asymptomatic", 1: "atypical chest pain",
                2: "non-anginal chest pain", 3: "typical angina (CAD symptom)"
            },
            "exang": {0: "no exercise-induced angina", 1: "exercise-induced angina present"},
            "slope": {1: "upsloping ST", 2: "flat ST", 3: "downsloping ST (ischemia)"}
        }

    def get_interpretation(self, factor: str, value: Any) -> str:

        rules = self.factor_interpretations.get(factor)
        if rules:
            return rules.get(value, "clinical correlation needed")

        if factor == 'oldpeak':
            if value == 0: return "no ST depression"
            if value <= 1.0: return "minimal ST depression"
            if value <= 2.0: return "moderate ST depression"
            return "significant ST depression (ischemia)"

        return "clinical correlation needed"


    def load_fast_models(self):

        logger.info("Loading FAST models (XGBoost + SHAP)...")
        try:

            XGB_MODEL_PATH = os.path.join(self.PROJECT_PATH, "models", "xgb_model.json")
            if not os.path.exists(XGB_MODEL_PATH):
                XGB_MODEL_PATH = os.path.join(self.PROJECT_PATH, "..", "models", "xgb_model.json")

            if not os.path.exists(XGB_MODEL_PATH):
                raise FileNotFoundError(f"XGBoost model not found at {XGB_MODEL_PATH}")
            
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(XGB_MODEL_PATH)
            self.xgb_loaded = True
            logger.info(" XGBoost loaded successfully")

            self.explainer = shap.TreeExplainer(self.xgb_model)
            self.shap_loaded = True
            logger.info("SHAP TreeExplainer initialized successfully")
        
        except Exception as e:
            logger.error(f" Failed to load FAST models: {e}")
            self.xgb_loaded = False
            self.shap_loaded = False

    def load_slow_models(self):

        logger.info("Loading SLOW models (RAG + LLM)...")

        self._load_rag()

        self._load_llm()
        
        logger.info(" All SLOW models loaded!")

    def _load_rag(self):
        logger.info(" Loading RAG/Chroma vector database...")
        try:
            if Chroma is None or SentenceTransformerEmbeddings is None:
                raise ImportError("LangChain/Chroma not available.")
            
            VECTOR_DB_PATH = os.path.join(self.PROJECT_PATH, "models", "chroma_db")
            if not os.path.exists(VECTOR_DB_PATH):
                VECTOR_DB_PATH = os.path.join(self.PROJECT_PATH, "..", "models", "chroma_db")
            if not os.path.exists(VECTOR_DB_PATH):
                raise FileNotFoundError(f"Chroma directory not found at {VECTOR_DB_PATH}")

            embedding_model = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)
            self.retriever = vector_db.as_retriever(search_kwargs={"k": 2})
            self.rag_loaded = True
            logger.info("RAG/Chroma loaded successfully")
        
        except Exception as e:
            logger.warning(f" RAG initialization failed: {e}")
            self.rag_loaded = False

    def _load_llm(self):
        logger.info("Loading LLM (this may take a few minutes)...")
        try:
            BASE_MODEL_PATH = os.path.join(self.PROJECT_PATH, "models", "Meta-Llama-3-8B-Instruct")
            ADAPTER_PATH = os.path.join(self.PROJECT_PATH, "models", "Dhanesh-56/cardio-advisor-llama3-8b")

            if not (os.path.exists(BASE_MODEL_PATH) and os.path.exists(ADAPTER_PATH)):
                raise FileNotFoundError("LLM model folders not found.")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. LLM loading is skipped.")

            torch.cuda.empty_cache()
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                    "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )

            self.llm_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
            self.model = self.model.merge_and_unload()
            
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.llm_loaded = True
            logger.info("LLM loaded and ready")

        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self.llm_loaded = False

models = AIModels()