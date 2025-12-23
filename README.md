Cardio Advisor AI

Cardio Advisor AI is a clinical decision support system designed to assist medical professionals in assessing cardiovascular risk.

It utilizes a hybrid two-stage architecture:

Stage 1 (Predictive AI): Instantly calculates a risk score using XGBoost and explains the result using SHAP (Explainable AI).

Stage 2 (Generative AI): Asynchronously generates a personalized clinical recommendation using a fine-tuned Llama 3 8B model and RAG (Retrieval-Augmented Generation) to reference medical guidelines.

The system is designed to be non-blocking and resource-efficient. It creates an immediate response for the user interface while the heavy LLM processing happens in the background.

🚀 Features

Real-time Prediction: Sub-second risk assessment using XGBoost.

Explainability: Detailed breakdown of why a patient is at risk using SHAP values (e.g., "High Cholesterol contributed +12% to risk").

Medical RAG: Retrieves relevant clinical guidelines from a ChromaDB vector store based on the specific risk profile.

Custom LLM Adapter: Uses a LoRA adapter fine-tuned on synthetic medical data to enforce a professional, concise clinical tone.

CPU Optimization: Deployed using GGUF quantization and llama-cpp-python to run efficiently on standard CPU hardware (Dockerized).

📂 Project Structure

cardio_advisor/
├── app/                     # Main application source code
│   ├── main.py              # FastAPI entry point
│   ├── ai_core.py           # Model loading logic (XGB, SHAP, LLM)
│   ├── services.py          # Business logic & orchestration
│   ├── schemas.py           # Pydantic data models
│   └── prompts.py           # Prompt engineering templates
├── models/                  # Local artifacts (XGB json, ChromaDB)
├── notebooks/               # Research & Training notebooks (Colab)
├── Dockerfile               # Production build configuration
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation


🛠️ Installation & Setup

Prerequisites

Python 3.9+

Docker (optional, for containerization)

1. Clone the Repository

git clone [https://github.com/https-dhanesh/Cardio_Advisor_AI.git](https://github.com/https-dhanesh/Cardio_Advisor_AI.git)
cd cardio-advisor-ai


2. Install Dependencies

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install libraries
pip install -r requirements.txt


3. Run Locally

uvicorn app.main:app --reload


The API will be available at http://localhost:8000.

🐳 Docker Deployment

This project is optimized for deployment on Hugging Face Spaces or any container service.

# Build the image
docker build -t cardio-advisor .

# Run the container
docker run -p 7860:7860 cardio-advisor


API Usage Example

You can test the API using curl or Postman.

Endpoint: POST /predict

Sample Input (High Risk Patient)

{
  "age": 63,
  "sex": 1,
  "cp": 0,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}


Sample Output (JSON Response)

{
  "status": "success",
  "prediction": {
    "risk_score": 0.748,
    "risk_level": "High Risk",
    "probability_percent": "74.8%"
  },
  "explainability": {
    "top_factors": {
      "thal": "reversible defect (ischemia)",
      "oldpeak": "significant ST depression",
      "cp": "asymptomatic"
    }
  },
  "ai_advice": "High cardiac risk detected (74.8%). The patient exhibits significant ST depression (oldpeak: 2.3) and reversible defects on thalassemia scan, strongly suggesting ischemia. Immediate cardiology consultation is recommended for further evaluation, including a potential angiogram to assess coronary artery patency."
}


License

This project is licensed under the Apache 2.0 License.
