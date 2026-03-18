🌟 What Does NeuroDoC Do?

<table>
<tr>
<td width="50%">

### 🎙️ You Give It
- Raw audio recording (any format)
- Hindi, English, or Hinglish speech
- Noisy OPD or clinic environment
- A phone recording from any distance

</td>
<td width="50%">

### 📋 It Gives You
- Full **SOAP Note** (AIIMS-style paragraphs)
- **Speaker-labeled dialogue** (Doctor vs Patient)
- **Entities** — diagnosis, ICD-10, meds, tests, vitals
- **5 × FHIR R4 resources** (ABDM-ready)
- **PDF + JSON** download in one click

</td>
</tr>
</table>

---

## 🏗️ How It Works — Full Pipeline

### 🔵 Stage 1 — Audio Cleaning

> **11-step ffmpeg hospital-grade denoising pipeline**

| Step | Name | What Happens |
|------|------|-------------|
| `①` | Normalize | Convert any codec to 32kHz float for precision DSP |
| `②` | 50Hz hum removal | Strip Indian power-line hum and harmonics (100 / 150 Hz) |
| `③` | FFT Spectral Denoise | Adaptive Wiener filter for stationary noise — HVAC, crowd hum |
| `④` | Non-local Means | Non-stationary noise — footsteps, beeps, coughs, paper rustling |
| `⑤` | Click Removal | Phone taps and mic handling noise |
| `⑥` | High-pass 80Hz | Remove sub-bass rumble from trolleys and footsteps |
| `⑦` | Low-pass 8kHz | Keep only the speech intelligibility band |
| `⑧` | Speech EQ | Boost 2.5kHz formants — critical for Hindi consonant clarity |
| `⑨` | Compressor | 8:1 ratio — normalize variable mic distances |
| `⑩` | EBU R128 | Loudness normalize to -14 LUFS (Whisper's tested optimal level) |
| `⑪` | Output | 16kHz 16-bit PCM — Whisper's native format |

<br/>

### 🟡 Stage 2 — Transcription

> **Two-pass Whisper strategy for maximum Indian language accuracy**

| Pass | What Happens |
|------|-------------|
| `Pass 1` | Auto-detect language from full audio — no language lock |
| `Pass 2` | Re-transcribe with confirmed language code — Hindi accuracy jumps significantly |
| `Medical prompt` | 224-token vocabulary primer for Indian drug names, lab terms, dosing shorthand |
| `Output` | Full text plus verbose JSON segments with timestamps and confidence scores |

<br/>

### 🟠 Stage 3 — Speaker Diarization

> **4-stage engine separating Doctor vs Patient with high reliability**

| Stage | Name | What Happens |
|-------|------|-------------|
| `Stage 1` | Rule Scoring | 6 linguistic signal families — vocabulary, sentence structure, question type, Hindi word class, conversation position, turn-taking |
| `Stage 2` | LLM Labeling | 7-rule system prompt sent to LLaMA with full conversation context and rule-based priors |
| `Stage 3` | Consistency Smoothing | Island detection, long-run breaking, backchannel correction — enforces turn-taking physics |
| `Stage 4` | Voice Fingerprinting | Per-session exclusive vocabulary profile boosts accuracy for ambiguous segments |

<br/>

### 🟣 Stage 4 — Correction Engine

> **Two-pass phonetic and contextual correction for Indian clinical terms**

| Pass | Name | What Happens |
|------|------|-------------|
| `Pass A` | Regex Rules | 250+ compiled rules for drug names, diagnoses, Hindi symptoms, dosing notation — instant, zero API cost |
| `Pass B` | LLM Correction | Whole-transcript context-aware correction with speaker labels — fixes garbled words using surrounding clinical context |

<br/>

### 🟢 Stage 5 — Clinical NLP

> **LLaMA-3.3-70b generates the full clinical record**

| Output | What Is Generated |
|--------|------------------|
| `Entities` | Name, age, gender, ICD-10, symptoms, diagnosis, medications with dose + route + freq + duration, tests, vitals, allergies, red flags, follow-up, referral |
| `SOAP Note` | 4 comprehensive paragraphs — S (HPI + PMH + social), O (vitals + exam), A (diagnosis + differentials + ICD-10), P (meds + investigations + lifestyle + follow-up) |
| `FHIR R4` | Patient, Encounter, Observation, Condition, MedicationRequest — all HL7-compliant |

---

## 🧬 FHIR R4 Output Structure

```
📦  FHIR Transaction Bundle  (ABDM-ready export)
│
├── 👤  Patient               identity · name · gender · language · nationality
│
├── 🏥  Encounter             AMB · Ambulatory · SNOMED 11429006 · Consultation
│
├── 📊  Observation           LOINC vital-signs panel · BP · HR · SpO2
│
├── 🔬  Condition             ICD-10 coded · SNOMED severity · confirmed · onset
│
└── 💊  MedicationRequest     RxNorm · full dosage instructions · generic substitution
```

All 5 resources are pushed to **HAPI FHIR public sandbox** automatically on each transcription. Push status is stored in the local DB with a full audit trail.

---

## ⚡ Quickstart

```bash
# 1 — Clone
git clone https://github.com/your-org/neurodoc
cd neurodoc

# 2 — Install Python dependencies
pip install fastapi groq httpx aiosqlite reportlab redis uvicorn python-multipart

# 3 — Install ffmpeg  (required for noise reduction)
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu / Debian

# 4 — Set Groq API key  (free at console.groq.com)
export GROQ_API_KEY=gsk_...

# 5 — Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** — record a consultation, get a full clinical report.

> 💡 **Free tier:** Groq's developer plan includes free Whisper + LLaMA inference — no payment needed to test.

---

## 🔌 API Reference

### 🎙️ Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcribe` | Upload audio, receive full clinical report JSON |
| `GET` | `/download?session_id=&format=pdf` | Download report as PDF or JSON |
| `GET` | `/health` | Service status, model info, DB stats |

### 📂 Consultation History

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/consultations` | List all stored consultations |
| `GET` | `/consultations/search?q=` | Search by patient name, ICD code, or diagnosis |
| `GET` | `/consultations/{sid}/full` | Full report JSON from database |
| `GET` | `/consultations/{sid}/entities` | Fast entity-only lookup |
| `PATCH` | `/consultations/{sid}/entities` | Edit any entity field |

### 🧬 FHIR and Interoperability

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/fhir/Bundle/{sid}` | ABDM-ready transaction Bundle — all 5 resources |
| `POST` | `/consultations/{sid}/push-fhir` | Re-push FHIR to server |
| `GET` | `/consultations/{sid}/fhir` | FHIR resources with push status audit |
| `GET` | `/fhir/Patient/{id}` | Proxy fetch from HAPI FHIR |
| `GET` | `/search_patient?name=` | Search HAPI FHIR by patient name |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | **required** | Groq API key for Whisper and LLaMA |
| `MAX_AUDIO_MB` | `30` | Max upload size in MB |
| `SESSION_TTL_H` | `12` | Session expiry in hours |
| `REDIS_URL` | *(empty)* | Redis URL — enables production-grade session store |
| `FHIR_BASE_URL` | HAPI public | Target FHIR server base URL |
| `FHIR_PUSH` | `1` | Auto-push FHIR resources — set `0` to disable |
| `DB_FILE` | `neurodoc.db` | SQLite database path |
| `RATE_LIMIT_RPM` | `15` | Rate limit per IP per minute |
| `TMP_DIR` | `/tmp/sp_audio` | Temp directory — auto-cleaned after each job |

---

## 🌐 Language Support

| Language | Script | Accuracy | Notes |
|----------|--------|----------|-------|
| 🇮🇳 Hindi | Devanagari | ⭐⭐⭐⭐⭐ | Two-pass Whisper with language lock |
| 🌐 English | Latin | ⭐⭐⭐⭐⭐ | Standard pipeline |
| ✨ Hinglish | Devanagari + Latin | ⭐⭐⭐⭐ | Auto-detected by script analysis |
| 📝 Romanised Hindi | Latin | ⭐⭐⭐⭐ | Labelled as Hindi, no Devanagari required |
| 🕌 Urdu | *(mapped to Hindi)* | ⭐⭐⭐ | Language-mapped to `hi` code |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🎤 **ASR** | Groq Whisper-large-v3 | Two-pass speech recognition |
| 🧠 **LLM** | LLaMA-3.3-70b-versatile | Diarization, NLP, correction |
| ⚡ **Backend** | FastAPI (fully async) | API server and routing |
| 🔊 **Audio** | ffmpeg (11 stages) | Hospital-grade noise reduction |
| 🗄️ **Database** | aiosqlite — SQLite WAL | Zero-config async persistence |
| 💾 **Sessions** | Redis / in-memory | TTL-based session store |
| 📄 **PDF** | ReportLab | Clinical report generation |
| 🏥 **FHIR sandbox** | HAPI FHIR R4 | ABDM-compatible resource store |
| 💻 **Frontend** | Vanilla JS | Zero frameworks, zero build step |

---

## 🗄️ Database Schema

Three tables persist every consultation for full audit and recall:

```sql
-- One row per recording session — full report JSON + patient metadata
consultations    (session_id, patient_name, icd_code, report_json, ...)

-- One row per FHIR resource — 5 per session, push status tracked
fhir_resources   (session_id, resource_type, resource_json, push_status, ...)

-- Every phonetic and LLM correction logged for audit
corrections      (session_id, segment_id, original_text, corrected_text, pass_type, ...)
```

> WAL mode enables concurrent readers — suitable for multi-user clinic deployment.
> Upgrade path: swap `aiosqlite` for `asyncpg` (PostgreSQL) with zero schema changes.

---

## 🔒 Security

| Feature | Detail |
|---------|--------|
| 🚦 Rate limiting | Per-IP sliding window, 15 req/min default |
| ✅ Input validation | Magic-byte audio check, strict UUID regex, file size limits |
| 🛡️ Security headers | `X-Frame-Options`, `X-Content-Type-Options`, `X-XSS-Protection`, `Referrer-Policy` |
| 🌐 CORS | Configurable via `ALLOWED_ORIGINS` environment variable |
| 🔇 No PII logging | Session IDs are opaque hex — audio deleted immediately after processing |

---

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY . .
RUN pip install fastapi groq httpx aiosqlite reportlab redis uvicorn python-multipart
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t neurodoc .
docker run -e GROQ_API_KEY=gsk_... -p 8000:8000 neurodoc
```

### Production Checklist

- [ ] Set `REDIS_URL` for persistent session store across restarts
- [ ] Set `ALLOWED_ORIGINS` to your frontend domain
- [ ] Mount a persistent volume for `DB_FILE`
- [ ] Set `FHIR_BASE_URL` to your ABDM or institutional FHIR endpoint
- [ ] Place behind nginx or Caddy with TLS termination
- [ ] Adjust `RATE_LIMIT_RPM` for your expected clinic load

---

## 📁 Project Structure

```
neurodoc/
├── 📄  main.py           Full backend — FastAPI app, pipeline, all routes
├── 🌐  recorder.html     Frontend — recording UI, SOAP viewer, FHIR explorer
├── 🗄️  neurodoc.db       SQLite database (auto-created on first run)
├── 🔊  /tmp/sp_audio/    Temp audio (auto-deleted after processing)
└── 📋  /tmp/sp_reports/  Generated PDF and JSON reports
```

---

## 🏆 Why We Built This

**90% of Indian OPD consultations go undocumented.** A doctor in a government hospital sees 60–100 patients per day. There is no time — and no expectation — to write notes.

NeuroDoC removes the documentation burden entirely. The doctor speaks naturally. The system listens, understands Hindi and English, separates who said what, and produces a clinical record that meets international standards.

| Problem | NeuroDoC's Solution |
|---------|-------------------|
| Doctor too busy to type | Ambient recording — no interaction needed |
| Hindi and English mixed speech | Two-pass Whisper with Devanagari language lock |
| Noisy OPD environment | 11-stage ffmpeg pipeline optimized for Indian clinics |
| Manual speaker separation | 4-stage engine — rule, LLM, smoothing, fingerprint |
| No EMR integration | 5 FHIR R4 resources with ABDM-ready bundle export |
| Garbled medical terms | 250+ phonetic rules plus context-aware LLM correction |

---

<div align="center">

<br/>

**Built for Indian healthcare · Hindi + English + Hinglish · FHIR R4 · ABDM-ready**

<br/>

[![Stars](https://img.shields.io/github/stars/your-org/neurodoc?style=for-the-badge&color=f5a623&labelColor=0d1829)](.)
[![Forks](https://img.shields.io/github/forks/your-org/neurodoc?style=for-the-badge&color=06c8b4&labelColor=0d1829)](.)
[![Issues](https://img.shields.io/github/issues/your-org/neurodoc?style=for-the-badge&color=ef4444&labelColor=0d1829)](.)

<br/>

*If this helped you, drop a ⭐ — it keeps the project alive.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
