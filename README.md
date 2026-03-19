🩺 NeuroDoC

"The best clinical note is the one that writes itself."

NeuroDoC is an ambient AI clinical scribe built specifically for Indian hospitals. It listens to a doctor–patient conversation in Hindi, English, or Hinglish, separates who said what, corrects every drug name and diagnosis, and produces a complete clinical record — SOAP note, ICD-10 code, and five HL7 FHIR R4 resources — in seconds. No typing. No templates.
Have a look at the demo (add your demo link here)


🚀 Features

🎙️ 11-Stage Noise Reduction Pipeline
A hospital-grade ffmpeg DSP chain handles India 50 Hz power-line hum, HVAC, crowd noise, monitor beeps, and variable mic distances before a single byte reaches the ASR model.
🗣️ Two-Pass Whisper Transcription
Pass 1 auto-detects language. Pass 2 re-transcribes with a language lock and a 224-token Indian medical vocabulary prompt — dramatically improving accuracy for Hindi, English, and Hinglish.
👥 4-Stage Speaker Diarization
Automatically labels every segment as DOCTOR or PATIENT with no voice enrollment. Uses rule scoring → LLM contextual labeling → consistency smoothing → per-session voice fingerprinting.
✍️ 2-Pass Medical Correction Engine
Pass A runs 250+ compiled regex rules for drug names, lab tests, and Hindi symptom terms. Pass B uses a whole-transcript LLM call with speaker context to fix anything that slipped through.
📋 Full Clinical NLP
Extracts 18 structured entity fields, writes a complete SOAP note (minimum 20 clinical sentences), assigns a specific ICD-10 code, and formats every medication with generic name + dose + route + frequency + duration.
⚡ 5× FHIR R4 Resources
Generates and auto-pushes Patient, Encounter, Observation, Condition, and MedicationRequest resources to HAPI FHIR. Uses LOINC, SNOMED CT, RxNorm, and ICD-10 coding systems.
📄 PDF & JSON Reports
One-click download of a formatted clinical PDF or raw JSON from any session — current or historical.
🕓 Session History
Last 30 reports stored locally. Load, edit, or re-download any consultation at any time.

Show Image
<!-- Save your architecture diagram as assets/architecture.png -->

🛠️ Tech Stack

FastAPI — async Python backend
Groq API — Whisper-large-v3 (ASR) + LLaMA-3.3-70b-versatile (NLP / diarization)
ffmpeg — 11-stage audio DSP pipeline
aiosqlite / SQLite WAL — zero-config persistence, Postgres-upgradeable
Redis — production session store (in-memory fallback for development)
ReportLab — PDF clinical report generation
HL7 FHIR R4 / HAPI FHIR — structured health data export
Vanilla JS + Web Audio API — zero-dependency frontend, no build step


⚙️ Usage
Running the Backend

Clone the repository:

shgit clone https://github.com/YOUR-USERNAME/neurodoc.git
cd neurodoc

Install dependencies:

shpip install fastapi "uvicorn[standard]" groq httpx aiosqlite reportlab redis

Make sure ffmpeg is installed and on your PATH:

sh# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html

Create a .env file (or export directly):

envGROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

# Optional — defaults shown
MAX_AUDIO_MB=30
SESSION_TTL_H=12
RATE_LIMIT_RPM=15
FHIR_BASE_URL=http://hapi.fhir.org/baseR4
FHIR_PUSH=1
REDIS_URL=
DB_FILE=neurodoc.db
ALLOWED_ORIGINS=*

Start the server:

shuvicorn main:app --host 0.0.0.0 --port 8000
Open http://localhost:8000 in Chrome or Edge.

Running with Docker
shdocker build -t neurodoc .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_xxx neurodoc
<details>
<summary>Dockerfile</summary>
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" groq httpx aiosqlite reportlab redis
ENV GROQ_API_KEY=""
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
</details>

📡 API Reference
POST /transcribe
The core endpoint. Upload audio, receive a complete clinical record.
shcurl -X POST http://localhost:8000/transcribe \
  -F "file=@consultation.webm"
<details>
<summary>Response shape</summary>
```jsonc
{
  "session_id": "a3f9c2...",
  "whisper_language": "Hinglish",
  "transcript": "Patient presents with chest pain since 3 days...",
  "report": {
    "entities": {
      "name": "Ramesh Kumar",
      "age": "52 years",
      "icd": "I20.9",
      "icd_name": "Angina pectoris, unspecified",
      "symptoms": ["chest pain", "breathlessness on exertion"],
      "meds": ["Aspirin 150mg PO OD x 30 days"],
      "vitals": ["BP: 138/88 mmHg", "HR: 92 bpm"],
      "red_flags": ["Chest pain radiating to left arm"]
      // + 10 more fields
    },
    "soap": {
      "s": "Patient is a 52-year-old male presenting with...",
      "o": "On examination, BP is 138/88 mmHg...",
      "a": "Primary diagnosis is Angina Pectoris (ICD-10: I20.9)...",
      "p": "Investigations ordered: ECG, Troponin I..."
    },
    "dialogue": [
      { "speaker": "DOCTOR", "time": "0.0s", "text": "Kya problem hai?", "clinical_significance": "instruction" }
    ],
    "fhir": { "patient": {}, "encounter": {}, "observation": {}, "condition": {}, "medication_request": {} }
  }
}
```
</details>
Other Endpoints
MethodEndpointDescriptionGET/healthService status, model info, DB statsGET/download?session_id=&format=pdf|jsonDownload reportGET/consultationsList all consultations (paginated)GET/consultations/search?q=Search by name, ICD code, diagnosisPATCH/consultations/{id}/entitiesEdit any entity field post-transcriptionGET/fhir/Bundle/{id}Full FHIR transaction Bundle (ABDM-ready)POST/consultations/{id}/push-fhirRe-push FHIR to HAPI serverGET/search_patient?name=Search HAPI FHIR patient archive

⚠️ Clinical Disclaimer

NeuroDoC is a documentation assistance tool only and does not constitute medical advice.
All AI-generated clinical records must be reviewed and countersigned by a licensed physician before use in any clinical setting.


📄 License
MIT

<div align="center">
Built for Indian clinicians  ·  Powered by Groq  ·  FHIR by HL7
NeuroDoC — From voice to verified clinical record, in seconds.
⭐ Star this repo if NeuroDoC saved you from typing a consultation note.
</div>
