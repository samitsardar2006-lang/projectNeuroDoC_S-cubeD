✨ What Does It Do?
Drop in a raw audio recording of a doctor-patient consultation. Get back a complete clinical record — automatically.
InputOutput🎙️ Raw audio (any format)📋 Full SOAP note (AIIMS-style paragraphs)🌐 Hindi / English / Hinglish🏷️ Speaker-labeled dialogue (Doctor vs Patient)🏥 Noisy OPD environment💊 Medications with dose, route, frequency📱 Phone recording🔬 Tests ordered with clinical indications🧬 5 × FHIR R4 resources (ABDM-ready)📄 PDF + JSON download

🏗️ Architecture
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│  Browser (WebM/OGG) ──────────────────────────────► FastAPI Upload  │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AUDIO PIPELINE  (ffmpeg, 11 stages)              │
│                                                                      │
│  ① 50Hz hum removal  ② FFT spectral denoise  ③ Non-local means     │
│  ④ Click removal     ⑤ High-pass (80Hz)      ⑥ Low-pass (8kHz)    │
│  ⑦ Speech EQ +3dB    ⑧ Compressor/gate       ⑨ EBU R128 loudnorm  │
│  ⑩ Silence strip     ⑪ → 16kHz 16-bit PCM                         │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
┌─────────────────────┐   ┌────────────────────────┐
│  WHISPER PASS 1     │   │  WHISPER PASS 2         │
│  Auto language      │──►│  Language-locked        │
│  detection          │   │  (Hindi accuracy ++)    │
└─────────┬───────────┘   └──────────┬─────────────┘
          └───────────────┬──────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│              DIARIZATION ENGINE  (4 stages)                         │
│                                                                      │
│  Stage 1 ── Rule scoring     6 linguistic signal families           │
│  Stage 2 ── LLM labeling     7-rule system prompt, full context     │
│  Stage 3 ── Smoothing        Island fix + long-run breaking         │
│  Stage 4 ── Fingerprinting   Per-session exclusive vocabulary       │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│              CORRECTION ENGINE  (2 passes)                          │
│                                                                      │
│  Pass A ── 250+ regex rules   Drug names, diagnoses, dosing         │
│  Pass B ── LLM whole-transcript  Speaker-aware contextual fix       │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
┌─────────────────────┐   ┌────────────────────────┐
│  CLINICAL NLP       │   │  FHIR R4 BUILDER       │
│  LLaMA-3.3-70b      │   │  5 resources           │
│  SOAP + entities    │   │  HAPI push (async)     │
└─────────┬───────────┘   └──────────┬─────────────┘
          └───────────────┬──────────┘
                          ▼
              ┌───────────────────────┐
              │  SQLite (WAL mode)    │
              │  Session store        │
              │  PDF / JSON export    │
              └───────────────────────┘

⚡ Quickstart
bash# 1. Clone the repo
git clone https://github.com/your-org/neurodoc && cd neurodoc

# 2. Install dependencies
pip install fastapi groq httpx aiosqlite reportlab redis uvicorn python-multipart

# 3. (Required) Install ffmpeg
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu / Debian

# 4. Set your Groq API key
export GROQ_API_KEY=gsk_...

# 5. Launch
uvicorn main:app --host 0.0.0.0 --port 8000
Open http://localhost:8000 — tap Record, speak Hindi or English, tap Stop. Report is ready in ~15 seconds.

💡 Get a free Groq API key at console.groq.com — Whisper + LLaMA inference is free on the dev tier.


🔌 API Reference
Core endpoints
MethodEndpointDescriptionPOST/transcribeUpload audio → full clinical report JSONGET/download?session_id=&format=pdf|json — download reportGET/healthService health, model info, DB stats
Consultation history
MethodEndpointDescriptionGET/consultationsList all stored consultations (most recent first)GET/consultations/search?q=Search by patient name, ICD code, or diagnosisGET/consultations/{sid}/fullFull report JSON from DBGET/consultations/{sid}/fhirFHIR resources + push status audit trailPATCH/consultations/{sid}/entitiesEdit any entity field (name, meds, ICD…)GET/consultations/{sid}/entitiesFast entity-only lookup
FHIR & interoperability
MethodEndpointDescriptionGET/fhir/Bundle/{sid}ABDM-ready transaction Bundle (all 5 resources)POST/consultations/{sid}/push-fhirRe-push FHIR to HAPI serverGET/fhir/Patient/{id}Proxy: fetch Patient from HAPI FHIRGET/fhir/Condition?patient=Proxy: fetch Conditions from HAPI FHIRGET/search_patient?name=Search HAPI FHIR public sandbox by name

🧬 FHIR R4 Output
Each consultation generates 5 linked HL7 FHIR R4 resources, stored locally and pushed to HAPI FHIR:
Patient
  └── Encounter ─────────────────── (AMB · Ambulatory · SNOMED 11429006)
        ├── Observation             (LOINC vital-signs panel · 85353-1)
        ├── Condition               (ICD-10 coded · SNOMED severity · confirmed)
        └── MedicationRequest       (RxNorm · full dosage instructions · substitution)
The /fhir/Bundle/{sid} endpoint returns an ABDM-compatible transaction Bundle — suitable for direct submission to the Ayushman Bharat Digital Mission health data exchange.

⚙️ Configuration
VariableDefaultDescriptionGROQ_API_KEYrequiredGroq API key for Whisper + LLaMAMAX_AUDIO_MB30Maximum upload sizeSESSION_TTL_H12In-memory session expiry (hours)REDIS_URL(empty)Redis URL — enables production session storeFHIR_BASE_URLHAPI publicTarget FHIR server base URLFHIR_PUSH1Auto-push resources on transcription (0 to disable)DB_FILEneurodoc.dbSQLite database pathTMP_DIR/tmp/sp_audioTemp directory for audio processingRATE_LIMIT_RPM15API rate limit per IP (requests/minute)

🌐 Language Support
LanguageScriptWhisper AccuracyNotesHindiDevanagari★★★★★Two-pass with language lock, medical vocab promptEnglishLatin★★★★★Standard Whisper pipelineHinglishDevanagari + Latin★★★★☆Auto-detected by script analysisRomanised HindiLatin★★★★☆Labelled as Hindi, no Devanagari requiredUrdu(treated as Hindi)★★★☆☆Language-mapped to hi code

🛠️ Tech Stack
LayerTechnologyASRGroq Whisper-large-v3 — two-pass strategyLLMLLaMA-3.3-70b-versatile via GroqBackendFastAPI — fully asyncAudioffmpeg — 11-stage hospital-grade pipelineDatabaseaiosqlite — SQLite with WAL modeSession cacheRedis (optional) / in-memory with TTL evictionPDF exportReportLabFHIR sandboxHAPI FHIR R4 public serverFrontendVanilla JS — zero frameworks, zero build step

🗄️ Database Schema
Three tables persist every consultation for audit and recall:
sqlconsultations    -- one row per session (full report JSON, patient metadata)
fhir_resources   -- one row per FHIR resource (5 per session, with push status)
corrections      -- audit log of every phonetic + LLM correction made
WAL mode enables concurrent readers — suitable for clinic-scale multi-user deployment. Drop-in upgrade path to PostgreSQL via asyncpg with zero schema changes.

🔒 Security

Per-IP sliding-window rate limiting (default: 15 req/min)
Magic-byte audio validation + file size limits
UUIDs validated with strict regex before any DB query
Security headers on all responses (X-Frame-Options, X-Content-Type-Options, CSP)
CORS configurable via ALLOWED_ORIGINS environment variable
No PII logged — session IDs are opaque hex, audio deleted after processing


📁 Project Structure
neurodoc/
├── main.py           # Full backend — FastAPI app, all routes, pipeline
├── recorder.html     # Frontend — recording UI, SOAP viewer, FHIR explorer
├── neurodoc.db       # SQLite database (auto-created)
├── /tmp/sp_audio/    # Temp audio files (auto-deleted after processing)
└── /tmp/sp_reports/  # Generated PDF/JSON reports

🚀 Deployment
Docker (recommended)
dockerfileFROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY . .
RUN pip install fastapi groq httpx aiosqlite reportlab redis uvicorn python-multipart
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
bashdocker build -t neurodoc .
docker run -e GROQ_API_KEY=gsk_... -p 8000:8000 neurodoc
Production checklist

 Set REDIS_URL for persistent session store
 Set ALLOWED_ORIGINS to your frontend domain
 Mount a persistent volume for DB_FILE
 Set FHIR_BASE_URL to your ABDM/institutional FHIR endpoint
 Place behind nginx/Caddy with TLS


🏆 Hackathon Context
NeuroDoC was built to solve a real gap in Indian primary healthcare: 90% of OPD consultations are undocumented because manual note-taking is too slow. A doctor sees 60–100 patients a day; they cannot type while talking.
This system turns any phone into an ambient scribe. The doctor speaks normally. The record writes itself.
Key technical bets:

Groq's ultra-low-latency inference makes the 15-second turnaround possible
Two-pass Whisper solves the Hindi language-lock accuracy problem
The 4-stage diarization engine reliably separates doctor from patient even in noisy OPDs
FHIR R4 output makes it plug-in compatible with any ABDM-compliant EMR


📜 License
MIT — see LICENSE. Use freely, contribute back.

<div align="center">
Built for Indian healthcare · Hindi + English + Hinglish · FHIR R4 · ABDM-ready
If this helped you, ⭐ the repo.
</div>
