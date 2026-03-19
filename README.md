📁 Frontend — recorder.html

A single-file, production-grade UI built with:

Vanilla JavaScript

Web Audio API

MediaRecorder

Advanced CSS system

🔑 Core Logic
// Recording pipeline
MediaRecorder + Web Audio preprocessing

// Audio enhancement
- High-pass filter
- Notch filters (50Hz / 100Hz)
- Compressor
- Noise gate (AudioWorklet)

// Upload
POST /transcribe

// Rendering
render(data) → updates all UI sections
⚙️ Backend — FastAPI
🧠 AI Stack

Whisper-large-v3 (Groq)

LLaMA-3.3-70B

FFmpeg preprocessing

📦 Pipeline
Audio → Clean → Transcribe → Diarize → LLM → Entities + SOAP → FHIR → Response
🔗 API
POST /transcribe
→ Upload consultation audio

GET /download?session_id=&format=pdf|json
→ Export report

PATCH /consultations/{id}/entities
→ Edit clinical data

GET /search_patient?name=
→ Query FHIR archive
