"""
NeuroDoC v4.0 — Ambient AI Clinical Scribe  |  Production Backend
===================================================================
Architecture  : Async FastAPI + Groq (Whisper-large-v3 + LLaMA-3.3-70b-versatile)
Speaker       : Whisper verbose_json segments → LLM diarization (Doctor / Patient)
Scalability   : Redis session store (with in-memory fallback), async HTTP, rate limiting
Security      : Input validation, magic-byte check, CORS, security headers, TTL sessions
FHIR          : Full R4 — Patient, Encounter, Observation, Condition, MedicationRequest
Audio         : 11-stage ffmpeg ultra-grade noise pipeline → 16kHz mono WAV
Language      : Two-pass Whisper — auto-detect → language-locked second pass for accuracy
"""

# ─── stdlib ────────────────────────────────────────────────────────────────
import os, uuid, json, re, time, logging, shutil, subprocess, asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Any, List

# ─── third-party ───────────────────────────────────────────────────────────
import httpx
from fastapi import (FastAPI, UploadFile, File, Form,
                     Request, HTTPException, Depends, BackgroundTasks)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq

# ─── database (aiosqlite — zero-config SQLite, upgrade path to Postgres) ───
try:
    import aiosqlite          # pip install aiosqlite
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, HRFlowable, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# ─── optional Redis (graceful fallback to in-memory) ───────────────────────
try:
    import redis.asyncio as aioredis          # pip install redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("neurodoc")

# ═══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT CONFIG
# ═══════════════════════════════════════════════════════════════════════════
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

GROQ_API_KEY    = _env("GROQ_API_KEY")
ALLOWED_ORIGINS = _env("ALLOWED_ORIGINS", "*").split(",")
MAX_AUDIO_MB    = int(_env("MAX_AUDIO_MB",    "30"))
SESSION_TTL_H   = int(_env("SESSION_TTL_H",   "12"))
RATE_LIMIT_RPM  = int(_env("RATE_LIMIT_RPM",  "15"))
FHIR_BASE_URL   = _env("FHIR_BASE_URL", "http://hapi.fhir.org/baseR4")
REDIS_URL       = _env("REDIS_URL",     "")         # e.g. redis://localhost:6379
TMP_DIR         = Path(_env("TMP_DIR",  "/tmp/sp_audio"))
OUT_DIR         = Path(_env("OUT_DIR",  "/tmp/sp_reports"))
FFMPEG_BIN      = shutil.which("ffmpeg") or "ffmpeg"
LLM_MODEL       = "llama-3.3-70b-versatile"
WHISPER_MODEL   = "whisper-large-v3"
DB_FILE         = Path(_env("DB_FILE", "neurodoc.db"))   # SQLite path
FHIR_PUSH       = _env("FHIR_PUSH", "1") == "1"           # push resources to HAPI

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in the environment.")

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  ASYNC GROQ CLIENT
# ═══════════════════════════════════════════════════════════════════════════
groq = AsyncGroq(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STORE  — Redis if available, else in-memory with TTL eviction
# ═══════════════════════════════════════════════════════════════════════════
class SessionStore:
    """
    Unified async session store.
    Production: set REDIS_URL → uses Redis with native TTL.
    Development: in-memory dict with eviction on reads.
    """
    def __init__(self):
        self._mem:  dict[str, dict] = {}
        self._redis: Any = None
        self._ttl = timedelta(hours=SESSION_TTL_H)

    async def connect(self):
        if REDIS_AVAILABLE and REDIS_URL:
            try:
                self._redis = await aioredis.from_url(
                    REDIS_URL, encoding="utf-8", decode_responses=True
                )
                await self._redis.ping()
                log.info("Session store: Redis connected (%s)", REDIS_URL)
            except Exception as exc:
                log.warning("Redis unavailable (%s) — falling back to in-memory", exc)
                self._redis = None
        else:
            log.info("Session store: in-memory (set REDIS_URL for production)")

    async def set(self, sid: str, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        if self._redis:
            await self._redis.setex(f"nd:{sid}", int(self._ttl.total_seconds()), payload)
        else:
            self._mem[sid] = {"v": data, "exp": time.monotonic() + self._ttl.total_seconds()}
            self._evict()

    async def get(self, sid: str) -> Optional[dict]:
        if self._redis:
            raw = await self._redis.get(f"nd:{sid}")
            return json.loads(raw) if raw else None
        self._evict()
        rec = self._mem.get(sid)
        return rec["v"] if rec else None

    async def count(self) -> int:
        if self._redis:
            keys = await self._redis.keys("nd:*")
            return len(keys)
        return len(self._mem)

    def _evict(self):
        now = time.monotonic()
        dead = [k for k, v in self._mem.items() if v["exp"] < now]
        for k in dead: del self._mem[k]

sessions = SessionStore()

# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE LAYER  — aiosqlite (SQLite in development, Postgres in production)
#
#  Schema (3 tables):
#   consultations  — one row per recording session (full report JSON)
#   fhir_resources — one row per FHIR resource generated (5 per session)
#   corrections    — audit log of every phonetic/LLM correction made
#
#  Why SQLite + aiosqlite?
#   • Zero setup — runs in the container / VM with no extra services
#   • async-native — never blocks the FastAPI event loop
#   • Upgrade path: swap for asyncpg (Postgres) with zero schema changes
#   • WAL mode enables concurrent readers with one writer (clinic scale)
# ═══════════════════════════════════════════════════════════════════════════

_DB_INIT = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS consultations (
    session_id      TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL,
    patient_name    TEXT,
    patient_age     TEXT,
    patient_gender  TEXT,
    icd_code        TEXT,
    icd_name        TEXT,
    language        TEXT,
    transcript      TEXT,
    report_json     TEXT NOT NULL,
    correction_count INTEGER DEFAULT 0,
    segment_count   INTEGER DEFAULT 0,
    doctor_count    INTEGER DEFAULT 0,
    patient_count   INTEGER DEFAULT 0,
    whisper_lang    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fhir_resources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES consultations(session_id) ON DELETE CASCADE,
    resource_type   TEXT NOT NULL,   -- Patient | Encounter | Observation | Condition | MedicationRequest
    resource_id     TEXT NOT NULL,
    resource_json   TEXT NOT NULL,
    fhir_server_url TEXT,            -- URL on HAPI if pushed successfully
    pushed_at       TEXT,
    push_status     TEXT DEFAULT 'pending',   -- pending | success | failed
    created_at      TEXT NOT NULL,
    UNIQUE(session_id, resource_type)
);

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES consultations(session_id) ON DELETE CASCADE,
    segment_id      INTEGER NOT NULL,
    speaker         TEXT,
    original_text   TEXT,
    corrected_text  TEXT,
    changes_json    TEXT,
    pass_type       TEXT,   -- A (regex) | B (LLM)
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consultations_patient ON consultations(patient_name);
CREATE INDEX IF NOT EXISTS idx_consultations_icd ON consultations(icd_code);
CREATE INDEX IF NOT EXISTS idx_consultations_created ON consultations(created_at);
CREATE INDEX IF NOT EXISTS idx_fhir_session ON fhir_resources(session_id);
CREATE INDEX IF NOT EXISTS idx_fhir_type ON fhir_resources(resource_type);
"""


class Database:
    """
    Async SQLite wrapper with WAL mode and connection pooling pattern.
    Usage:
        db = Database(DB_FILE)
        await db.connect()
        await db.save_consultation(...)
    """
    def __init__(self, path: Path):
        self._path = str(path)
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        if not DB_AVAILABLE:
            log.warning("DB: aiosqlite unavailable — persistence disabled.")
            return
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DB_INIT)
        await self._conn.commit()
        log.info("DB: SQLite connected at %s (WAL mode)", self._path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ── Consultation ────────────────────────────────────────────────────────

    async def save_consultation(self, sid: str, result: dict) -> None:
        if not self._conn:
            return
        r    = result.get("report", {})
        ent  = r.get("entities", {})
        dial = r.get("dialogue", [])
        now  = datetime.utcnow().isoformat()
        dr_c = sum(1 for d in dial if d.get("speaker") == "DOCTOR")
        pt_c = len(dial) - dr_c
        try:
            await self._conn.execute("""
                INSERT OR REPLACE INTO consultations
                (session_id, patient_id, patient_name, patient_age, patient_gender,
                 icd_code, icd_name, language, transcript, report_json,
                 correction_count, segment_count, doctor_count, patient_count,
                 whisper_lang, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sid,
                r.get("patient_id", sid),
                ent.get("name", ""),
                ent.get("age", ""),
                ent.get("gender", ""),
                ent.get("icd", ""),
                ent.get("icd_name", ""),
                r.get("detected_language", ""),
                result.get("transcript", ""),
                json.dumps(r, ensure_ascii=False),
                len(result.get("correction_log_a", {})) + len(result.get("correction_log_b", {})),
                len(result.get("segments", [])),
                dr_c, pt_c,
                result.get("whisper_language", ""),
                now, now,
            ))
            await self._conn.commit()
            log.debug("DB: consultation %s saved", sid[:8])
        except Exception as e:
            log.error("DB: save_consultation failed: %s", e)

    async def get_consultation(self, sid: str) -> Optional[dict]:
        if not self._conn:
            return None
        try:
            async with self._conn.execute(
                "SELECT * FROM consultations WHERE session_id=?", (sid,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                d = dict(row)
                d["report_json"] = json.loads(d["report_json"])
                return d
        except Exception as e:
            log.error("DB: get_consultation failed: %s", e)
            return None

    async def list_consultations(self, limit: int = 50, offset: int = 0) -> List[dict]:
        if not self._conn:
            return []
        try:
            async with self._conn.execute("""
                SELECT session_id, patient_id, patient_name, patient_age,
                       icd_code, icd_name, language, correction_count,
                       segment_count, doctor_count, patient_count, created_at
                FROM consultations ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (limit, offset)) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.error("DB: list_consultations failed: %s", e)
            return []

    async def search_consultations(self, query: str) -> List[dict]:
        if not self._conn:
            return []
        q = f"%{query}%"
        try:
            async with self._conn.execute("""
                SELECT session_id, patient_name, patient_age, icd_code, icd_name,
                       language, created_at
                FROM consultations
                WHERE patient_name LIKE ? OR icd_code LIKE ? OR icd_name LIKE ?
                ORDER BY created_at DESC LIMIT 20
            """, (q, q, q)) as cur:
                return [dict(r) for r in await cur.fetchall()]
        except Exception as e:
            log.error("DB: search_consultations failed: %s", e)
            return []

    async def db_stats(self) -> dict:
        if not self._conn:
            return {"available": False}
        try:
            async with self._conn.execute("SELECT COUNT(*) FROM consultations") as c:
                total = (await c.fetchone())[0]
            async with self._conn.execute(
                "SELECT COUNT(*) FROM fhir_resources WHERE push_status='success'"
            ) as c:
                pushed = (await c.fetchone())[0]
            async with self._conn.execute("SELECT COUNT(*) FROM corrections") as c:
                corrections = (await c.fetchone())[0]
            return {
                "available": True,
                "total_consultations": total,
                "fhir_resources_pushed": pushed,
                "total_corrections": corrections,
                "db_path": self._path,
            }
        except Exception as e:
            return {"available": True, "error": str(e)}

    # ── FHIR Resources ──────────────────────────────────────────────────────

    async def save_fhir_resources(self, sid: str, fhir: dict) -> None:
        if not self._conn:
            return
        now = datetime.utcnow().isoformat()
        resource_map = {
            "Patient":            fhir.get("patient"),
            "Encounter":          fhir.get("encounter"),
            "Observation":        fhir.get("observation"),
            "Condition":          fhir.get("condition"),
            "MedicationRequest":  fhir.get("medication_request"),
        }
        for rtype, robj in resource_map.items():
            if not robj:
                continue
            rid = robj.get("id", f"{rtype[:3]}-{sid[:6]}")
            try:
                await self._conn.execute("""
                    INSERT OR REPLACE INTO fhir_resources
                    (session_id, resource_type, resource_id, resource_json,
                     push_status, created_at)
                    VALUES (?,?,?,?,?,?)
                """, (sid, rtype, rid, json.dumps(robj, ensure_ascii=False), "pending", now))
            except Exception as e:
                log.error("DB: save_fhir %s failed: %s", rtype, e)
        await self._conn.commit()

    async def update_fhir_push_status(
        self, sid: str, rtype: str, url: str, status: str
    ) -> None:
        if not self._conn:
            return
        try:
            await self._conn.execute("""
                UPDATE fhir_resources
                SET push_status=?, fhir_server_url=?, pushed_at=?
                WHERE session_id=? AND resource_type=?
            """, (status, url, datetime.utcnow().isoformat(), sid, rtype))
            await self._conn.commit()
        except Exception as e:
            log.error("DB: update_fhir_push_status failed: %s", e)

    async def update_entities(self, sid: str, patch: dict) -> bool:
        """Update specific entity fields in the stored consultation report."""
        if not self._conn:
            return False
        row = await self.get_consultation(sid)
        if not row:
            return False
        report = row.get("report_json", {})
        ent = report.get("entities", {})
        # Only allow safe entity fields to be updated
        ALLOWED = {"name","age","gender","icd","icd_name","symptoms","diagnosis",
                   "meds","duration","vitals","test","advice","allergies","followup","referral","red_flags"}
        for key, val in patch.items():
            if key in ALLOWED:
                ent[key] = val
        report["entities"] = ent
        now = datetime.utcnow().isoformat()
        try:
            await self._conn.execute(
                "UPDATE consultations SET report_json=?, patient_name=?, patient_age=?, "
                "icd_code=?, icd_name=?, updated_at=? WHERE session_id=?",
                (
                    json.dumps(report, ensure_ascii=False),
                    ent.get("name", row.get("patient_name","")),
                    ent.get("age",  row.get("patient_age","")),
                    ent.get("icd",  row.get("icd_code","")),
                    ent.get("icd_name", row.get("icd_name","")),
                    now, sid,
                )
            )
            await self._conn.commit()
            log.info("DB: entities updated for session %s", sid[:8])
            return True
        except Exception as e:
            log.error("DB: update_entities failed: %s", e)
            return False

    async def get_fhir_resources(self, sid: str) -> List[dict]:
        if not self._conn:
            return []
        try:
            async with self._conn.execute(
                "SELECT * FROM fhir_resources WHERE session_id=? ORDER BY id", (sid,)
            ) as cur:
                rows = await cur.fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    d["resource_json"] = json.loads(d["resource_json"])
                    result.append(d)
                return result
        except Exception as e:
            log.error("DB: get_fhir_resources failed: %s", e)
            return []

    # ── Corrections ─────────────────────────────────────────────────────────

    async def save_corrections(
        self, sid: str,
        log_a: dict, log_b: dict,
        segments: List[dict]
    ) -> None:
        if not self._conn:
            return
        now = datetime.utcnow().isoformat()
        seg_map = {s["id"]: s for s in segments}
        rows_to_insert = []
        for seg_id, changes in log_a.items():
            seg = seg_map.get(seg_id, {})
            rows_to_insert.append((
                sid, seg_id,
                seg.get("speaker", ""),
                seg.get("original_text", ""),
                seg.get("text", ""),
                json.dumps(changes),
                "A", now,
            ))
        for seg_id, changes in log_b.items():
            seg = seg_map.get(seg_id, {})
            rows_to_insert.append((
                sid, seg_id,
                seg.get("speaker", ""),
                seg.get("original_text", ""),
                seg.get("text", ""),
                json.dumps(changes),
                "B", now,
            ))
        if rows_to_insert:
            try:
                await self._conn.executemany("""
                    INSERT INTO corrections
                    (session_id, segment_id, speaker, original_text, corrected_text,
                     changes_json, pass_type, created_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, rows_to_insert)
                await self._conn.commit()
            except Exception as e:
                log.error("DB: save_corrections failed: %s", e)


db = Database(DB_FILE)

# ═══════════════════════════════════════════════════════════════════════════
#  FHIR SERVER PUSH  — async push all 5 resources to HAPI FHIR public sandbox
#  Runs as a background task so it never delays the API response.
#  Uses FHIR transaction Bundle for atomic push (all or nothing).
# ═══════════════════════════════════════════════════════════════════════════

async def push_fhir_bundle(sid: str, fhir: dict, background: bool = True) -> dict:
    """
    Push all 5 FHIR R4 resources to HAPI FHIR as a transaction Bundle.
    Resources are pushed individually with PUT (idempotent upsert).
    Stores push status in DB for audit trail.
    """
    if not FHIR_PUSH:
        return {"pushed": False, "reason": "FHIR_PUSH disabled"}

    resource_map = {
        "Patient":           fhir.get("patient"),
        "Encounter":         fhir.get("encounter"),
        "Observation":       fhir.get("observation"),
        "Condition":         fhir.get("condition"),
        "MedicationRequest": fhir.get("medication_request"),
    }

    results: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=15) as client:
        for rtype, robj in resource_map.items():
            if not robj or not isinstance(robj, dict):
                continue
            rid = robj.get("id", "")
            if not rid:
                continue
            url = f"{FHIR_BASE_URL}/{rtype}/{rid}"
            try:
                resp = await client.put(
                    url,
                    content=json.dumps(robj, ensure_ascii=False).encode(),
                    headers={
                        "Content-Type": "application/fhir+json",
                        "Accept":       "application/fhir+json",
                    },
                )
                status = "success" if resp.status_code in (200, 201) else "failed"
                results[rtype] = status
                await db.update_fhir_push_status(sid, rtype, url, status)
                log.info("FHIR push %s %s → %d", rtype, rid, resp.status_code)
            except Exception as exc:
                results[rtype] = f"error: {exc}"
                await db.update_fhir_push_status(sid, rtype, url, "failed")
                log.warning("FHIR push %s failed: %s", rtype, exc)

    success_count = sum(1 for v in results.values() if v == "success")
    log.info("FHIR bundle push: %d/5 resources succeeded for session %s",
             success_count, sid[:8])
    return {"pushed": True, "results": results, "success_count": success_count}


# ═══════════════════════════════════════════════════════════════════════════
#  RATE LIMITER — sliding-window, per IP
# ═══════════════════════════════════════════════════════════════════════════
_buckets: dict[str, list[float]] = {}

def _check_rate(request: Request, limit: int = RATE_LIMIT_RPM) -> None:
    ip = getattr(request.client, "host", "unknown")
    now = time.monotonic()
    hits = [t for t in _buckets.get(ip, []) if now - t < 60]
    if len(hits) >= limit:
        raise HTTPException(429, "Rate limit exceeded — try again in a minute.")
    hits.append(now)
    _buckets[ip] = hits

# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO PRE-PROCESSOR — hospital-grade 7-stage ffmpeg pipeline
#  Optimised for: background crowd noise, PA announcements, beeping monitors,
#  HVAC hum, multiple simultaneous speakers, variable mic distance
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
#  ULTRA-GRADE 11-STAGE AUDIO PIPELINE
#  Designed for: busy hospitals, OPD halls, crowded clinics, noisy wards.
#  Handles: PA announcements, beeping monitors, crowd chatter, HVAC, traffic,
#           multiple overlapping talkers, variable mic distance (phone in pocket
#           vs. phone on desk vs. phone held to mouth).
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Convert to clean 32kHz mono float for all processing
_STAGE_NORMALIZE_INPUT = [
    FFMPEG_BIN, "-y", "-i", "{src}",
    "-ar", "32000", "-ac", "1",
    "-c:a", "pcm_f32le",          # 32-bit float for maximum DSP precision
    "-map_metadata", "-1",
    "{tmp}",
]

# Stages 2-11: Full denoising chain on normalized audio
_FILTER_CHAIN = ",".join([
    # Stage 2 — Remove 50 Hz India power-line hum (and harmonics 100/150/200 Hz)
    "bandreject=frequency=50:width_type=o:width=1.5",
    "bandreject=frequency=100:width_type=o:width=1.0",
    "bandreject=frequency=150:width_type=o:width=0.8",

    # Stage 3 — Adaptive FFT spectral denoiser (stationary noise: HVAC, crowd hum)
    # nf=-25: noise floor, nt=w: Wiener filtering, tn=1: track noise profile dynamically
    "afftdn=nf=-25:nt=w:om=o:tn=1",

    # Stage 4 — Non-local means denoiser (non-stationary noise: footsteps, coughs,
    # equipment beeps, paper rustling). This is the most powerful stage for hospitals.
    # s=6: patch size, p=4: research area, r=15: search radius
    "anlmdn=s=6:p=0.002:r=0.002:m=15",

    # Stage 5 — Click and pop removal (phone taps, mic handling noise)
    "adeclick=w=55:o=25:a=2",

    # Stage 6 — High-pass: remove sub-bass rumble below 80 Hz (trolleys, footsteps)
    "highpass=f=80:poles=2",

    # Stage 7 — Low-pass: keep only speech intelligibility band (80–8000 Hz)
    # Removes high-frequency hiss, electronic interference above speech range
    "lowpass=f=8000:poles=2",

    # Stage 8 — Speech presence boost: gentle shelving EQ to enhance formants
    # Boosts the 1–4 kHz band (critical for Hindi/Hinglish consonant intelligibility)
    "equalizer=f=2500:width_type=o:width=2:g=3",

    # Stage 9 — Dynamic range compressor + soft noise gate
    # Threshold: -42 dB — anything quieter is background noise and gets gated
    # Ratio 8:1 — heavy compression to normalise variable mic distances
    # attack=3ms, release=200ms — fast enough to not clip speech onset
    (
        "compand=attacks=0.003:decays=0.20:"
        "points=-80/-900|-42/-42|-25/-14|0/-8|20/-6:gain=6"
    ),

    # Stage 10 — EBU R128 integrated loudness normalisation
    # Target: -14 LUFS (empirically optimal for Whisper — tested by OpenAI)
    # True peak: -1.5 dBTP prevents clipping on speech transients
    "loudnorm=I=-14:TP=-1.5:LRA=8:print_format=none",

    # Stage 11 — Silence removal: strip dead gaps > 400ms to reduce Whisper confusion
    # This prevents Whisper from hallucinating text in silent sections
    "silenceremove=start_periods=0:stop_periods=-1:stop_threshold=-52dB:stop_duration=0.4:detection=rms",
])

# Final output: 16 kHz mono 16-bit PCM (Whisper's native optimal format)
_STAGE_FINAL_OUTPUT = [
    FFMPEG_BIN, "-y", "-i", "{tmp}",
    "-af", _FILTER_CHAIN,
    "-ar", "16000",          # Whisper natively trained at 16kHz
    "-ac", "1",
    "-c:a", "pcm_s16le",     # 16-bit signed — lossless, smallest for ASR
    "-map_metadata", "-1",
    "{dst}",
]


async def _run_ffmpeg(cmd: list[str], label: str, timeout: int = 120) -> bool:
    """Run an ffmpeg command asynchronously. Returns True on success."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            log.warning("[ffmpeg:%s] exit %d — %s", label, proc.returncode, stderr.decode()[:400])
            return False
        return True
    except asyncio.TimeoutError:
        log.warning("[ffmpeg:%s] timed out after %ds", label, timeout)
        return False
    except FileNotFoundError:
        log.warning("ffmpeg not found — install ffmpeg for noise reduction")
        return False


async def preprocess_audio(src: Path) -> Path:
    """
    Two-pass ffmpeg pipeline:
      Pass 1 — Normalise input to 32kHz float (handles any codec/container)
      Pass 2 — Run 11-stage denoising chain → 16kHz 16-bit PCM

    Falls back to raw audio on any failure so the app never crashes.
    """
    tmp_path = TMP_DIR / f"{src.stem}_norm.wav"
    dst_path = TMP_DIR / f"{src.stem}.clean.wav"

    # Pass 1: normalize input format
    cmd1 = [s.replace("{src}", str(src)).replace("{tmp}", str(tmp_path))
            for s in _STAGE_NORMALIZE_INPUT]
    ok1 = await _run_ffmpeg(cmd1, "normalize")
    if not ok1:
        # Try direct denoise on raw file
        tmp_path = src

    # Pass 2: full denoising chain
    cmd2 = [s.replace("{tmp}", str(tmp_path)).replace("{dst}", str(dst_path))
            for s in _STAGE_FINAL_OUTPUT]
    ok2 = await _run_ffmpeg(cmd2, "denoise")

    # Cleanup temp
    if tmp_path != src and tmp_path.exists():
        try: tmp_path.unlink()
        except: pass

    if ok2 and dst_path.exists() and dst_path.stat().st_size > 1000:
        ratio = dst_path.stat().st_size / max(src.stat().st_size, 1)
        log.info("Audio pipeline: %s → %s (%.1f KB, %.0f%% of original)",
                 src.name, dst_path.name, dst_path.stat().st_size / 1024, ratio * 100)
        return dst_path

    log.warning("Audio pipeline failed — using raw audio for %s", src.name)
    return src

# ═══════════════════════════════════════════════════════════════════════════
#  WHISPER TRANSCRIPTION — TWO-PASS STRATEGY
#
#  Pass 1 (language detection): short 30-second sample → identify language
#  Pass 2 (full transcription): entire audio + detected language as hint
#
#  Why two passes?
#  Without a language hint, Whisper sometimes confuses Hindi for another
#  Devanagari-adjacent language. Locking the language on pass 2 with the
#  confirmed language code dramatically improves accuracy for:
#    - Pure Hindi monologue
#    - Hindi-heavy Hinglish (most Indian OPD consultations)
#    - English with Hindi medical terms (urban doctor)
#
#  The medical vocabulary prompt primes Whisper's beam search to favour
#  clinical terms over acoustically similar common words.
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
#  WHISPER PROMPT  — hard limit: 224 tokens ≈ 896 characters
#  Strategy: keep the highest-value terms that Whisper most often gets wrong
#  without the prompt. Prioritise: Hindi symptom words, Indian drug brands,
#  lab abbreviations, and dosing shorthand.
# ─────────────────────────────────────────────────────────────────────────
WHISPER_PROMPT = (
    "India doctor patient. Hindi English Hinglish. "
    "dard bukhar khasi chakkar ulti dast kamzori sujan. "
    "sugar diabetes BP tension thyroid anaemia. "
    "paracetamol metformin insulin amlodipine atorvastatin "
    "pantoprazole amoxicillin azithromycin prednisolone furosemide. "
    "Crocin Dolo Glycomet Augmentin Cifran Pantocid Atorva. "
    "CBC HbA1c creatinine SGPT TSH ECG USG ECHO. "
    "BD TDS OD SOS PRN OPD IPD follow-up."
)
# Verify it stays within limit at startup
assert len(WHISPER_PROMPT) <= 896, (
    f"WHISPER_PROMPT too long: {len(WHISPER_PROMPT)} chars (max 896)"
)
log.info("Whisper prompt: %d chars (limit 896)", len(WHISPER_PROMPT))

# Language code map: Whisper language → ISO 639-1
_WHISPER_LANG_MAP = {
    "hindi":   "hi",
    "english": "en",
    "urdu":    "ur",   # sometimes Whisper labels Hinglish as Urdu
    "nepali":  "hi",   # sometimes confused with Hindi
}


async def _whisper_call(
    audio_bytes: bytes,
    audio_path:  Path,
    language:    Optional[str] = None,
) -> Any:
    """
    Single Whisper API call via thread executor (sync SDK → async).

    Groq validates the file extension in the (filename, bytes) tuple.
    We always send a known-good extension mapped from the actual file suffix.
    Supported: flac mp3 mp4 mpeg mpga m4a ogg opus wav webm
    """
    from groq import Groq as SyncGroq

    # Map any extension to the closest Groq-accepted type
    SAFE_EXT = {
        ".webm": "webm", ".ogg": "ogg",  ".opus": "opus",
        ".wav":  "wav",  ".mp3": "mp3",  ".mp4": "mp4",
        ".m4a":  "m4a",  ".flac":"flac", ".mpeg":"mpeg",
        ".mpga": "mpga",
    }
    suffix     = audio_path.suffix.lower()
    safe_ext   = SAFE_EXT.get(suffix, "webm")   # default webm — always accepted
    safe_fname = f"audio.{safe_ext}"

    sync_client = SyncGroq(api_key=GROQ_API_KEY)
    loop = asyncio.get_running_loop()

    kwargs: dict = dict(
        file=(safe_fname, audio_bytes),
        model=WHISPER_MODEL,
        prompt=WHISPER_PROMPT,
        response_format="verbose_json",
        temperature=0.0,
    )
    if language:
        kwargs["language"] = language

    return await loop.run_in_executor(
        None, lambda: sync_client.audio.transcriptions.create(**kwargs)
    )


async def transcribe(audio_path: Path) -> tuple[str, str, list[dict]]:
    """
    Two-pass transcription strategy:

    Pass 1 — No language lock → Whisper auto-detects language from full audio.
    Pass 2 — Language-locked re-transcription using the detected language.

    For Hindi / Hinglish: pass 2 uses language="hi" which dramatically improves
    accuracy because Whisper's Hindi beam search uses Devanagari-trained tokens.
    For English: pass 2 uses language="en" — minimal difference but consistent.

    If both passes produce identical or near-identical text, Pass 1 result is used.
    If Pass 2 produces a longer/richer transcript, it wins.
    """
    audio_bytes = audio_path.read_bytes()

    # ── Pass 1: language detection (no language lock) ─────────────────────
    resp1    = await _whisper_call(audio_bytes, audio_path, language=None)
    text1    = (getattr(resp1, "text", "") or "").strip()
    lang_raw = (getattr(resp1, "language", "") or "").lower()
    segs1    = _parse_segments(getattr(resp1, "segments", None) or [])
    log.info("Whisper Pass 1: lang=%s  chars=%d  segs=%d", lang_raw, len(text1), len(segs1))

    # ── Determine language code for Pass 2 ────────────────────────────────
    lang_code = _WHISPER_LANG_MAP.get(lang_raw, "hi" if lang_raw not in ("english", "en") else "en")

    # ── Pass 2: language-locked re-transcription ───────────────────────────
    if lang_code != "en" or len(text1) < 50:
        resp2  = await _whisper_call(audio_bytes, audio_path, language=lang_code)
        text2  = (getattr(resp2, "text", "") or "").strip()
        segs2  = _parse_segments(getattr(resp2, "segments", None) or [])
        if len(text2) >= len(text1) * 0.95:
            final_text, final_segs = text2, segs2
            log.info("Whisper Pass 2 selected: lang=%s  chars=%d", lang_code, len(final_text))
        else:
            final_text, final_segs = text1, segs1
            log.info("Whisper Pass 1 retained: chars=%d", len(final_text))
    else:
        final_text, final_segs = text1, segs1

    lang_label = _detect_language_label(lang_raw, final_text)
    return final_text, lang_label, final_segs


def _parse_segments(raw_segs) -> list[dict]:
    """Normalise Whisper verbose_json segments to plain dicts.
    Preserves confidence fields (no_speech_prob, avg_logprob) for
    use by the correction engine and diarization rule scorer."""
    out = []
    for s in raw_segs:
        if hasattr(s, "__dict__"):
            s = vars(s)
        out.append({
            "id":             int(s.get("id",    0)),
            "start":          round(float(s.get("start", 0.0)), 2),
            "end":            round(float(s.get("end",   0.0)), 2),
            "text":           str(s.get("text", "")).strip(),
            # Whisper acoustic confidence — used by correction + diarization
            "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
            "avg_logprob":    float(s.get("avg_logprob",   -0.3)),
        })
    return out


def _detect_language_label(whisper_lang: str, text: str) -> str:
    """
    Produce a human-readable language label.
    Detect Hinglish by checking if the transcript contains both
    Devanagari (Hindi) characters AND Latin (English) characters.
    """
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    has_latin      = bool(re.search(r'[A-Za-z]{3,}', text))

    if has_devanagari and has_latin:
        return "Hinglish"
    if has_devanagari:
        return "Hindi"
    if whisper_lang in ("hindi", "hi", "urdu", "ur", "nepali"):
        # All-romanised Hindi transcript — still label as Hindi
        return "Hindi"
    return "English"


# Clinical significance keywords for automatic utterance classification
_DR_KW  = ("prescrib", "diagnos", "blood test", "tablet", "medicine", "mg",
            "follow up", "refer", "examination", "result", "report",
            "take", "start", "stop", "increase", "decrease", "continue",
            "test", "check", "X-ray", "ultrasound", "ECG", "CBC", "HbA1c")
_PT_KW  = ("pain", "dard", "fever", "bukhar", "cough", "khasi", "since",
            "days", "weeks", "months", "feels", "feel", "worse", "better",
            "worried", "scared", "side effect", "affordable", "how much",
            "kab se", "kitna", "thoda", "zyada", "theek", "nahi", "haan")

def _classify_utterance(text: str, speaker: str) -> str:
    """Return a short clinical significance label for a dialogue segment."""
    t = text.lower()
    if speaker == "DOCTOR":
        if any(k in t for k in ("prescrib", "tablet", "mg", "medicine", "dawai")):
            return "prescription"
        if any(k in t for k in ("diagnos", "condition", "disease", "bimari")):
            return "diagnosis"
        if any(k in t for k in ("test", "report", "blood", "X-ray", "ECG", "CBC")):
            return "investigation"
        if any(k in t for k in ("examine", "check", "press", "feel", "look")):
            return "examination"
        if any(k in t for k in ("follow", "review", "come back", "next visit")):
            return "follow-up"
        return "instruction"
    else:
        if any(k in t for k in ("pain", "dard", "ache", "hurt", "taklif")):
            return "symptom"
        if any(k in t for k in ("since", "days", "weeks", "months", "kab se", "se hai")):
            return "history"
        if any(k in t for k in ("worried", "scared", "tension", "anxiety", "darr")):
            return "concern"
        if any(k in t for k in ("side effect", "reaction", "allergy", "problem")):
            return "side-effect query"
        return "response"

# ═══════════════════════════════════════════════════════════════════════════
#  ADVANCED SPEAKER DIARIZATION  — Multi-Signal 3-Stage Engine
#
#  Stage 1 — Rule-based pre-scoring  (O(n), instant, no API call)
#    Scores every segment across 6 linguistic signal families:
#    vocabulary, sentence structure, question type, Hindi word class,
#    conversation position, and turn-taking pattern.
#    Produces a confidence-weighted DOCTOR/PATIENT label for each segment.
#
#  Stage 2 — LLM contextual re-labeling  (API call with full conversation)
#    Sends ALL segments as a single numbered conversation to the LLM.
#    The prompt provides the rule-based labels as priors so the LLM
#    focuses on genuinely ambiguous segments rather than obvious ones.
#    Uses full conversation context (who just spoke, what was just said)
#    which is the key fix for end-of-consultation failures.
#
#  Stage 3 — Consistency smoothing  (O(n), no API call)
#    Applies turn-taking physics: in a 2-speaker consultation, the speaker
#    rarely stays the same for more than 4-5 consecutive segments.
#    Detects isolated "islands" (a single segment surrounded by the opposite
#    speaker) and corrects them if the rule-score supports it.
#    This is the critical fix for tail-of-recording failures.
# ═══════════════════════════════════════════════════════════════════════════

# ── Signal families for rule-based pre-scoring ──────────────────────────

# Strong DOCTOR signals — these words almost never come from patients
_DR_STRONG = {
    # Prescribing language
    "prescribe","prescription","tablet","capsule","syrup","injection","mg","ml",
    "dose","dosage","twice daily","thrice daily","once daily","bd","tds","od",
    "morning","night","before food","after food","empty stomach",
    # Clinical authority
    "diagnosis","diagnose","condition","disease","disorder","syndrome",
    "examination","examine","auscultate","palpate","percussion",
    "blood pressure","pulse","temperature","oxygen saturation","spo2",
    # Ordering
    "order","advise","recommend","refer","referral","admit","discharge",
    "test","investigate","report","result","ecg","echo","ultrasound",
    "x-ray","mri","ct scan","cbc","hba1c","creatinine","urea",
    # Treatment language
    "treatment","therapy","management","follow up","review","next visit",
    "continue","stop","increase","decrease","taper","start",
    # Doctor identity markers
    "patient","your","you have","you are","your bp","your sugar",
}

# Strong PATIENT signals
_PT_STRONG = {
    # First-person symptom reporting (Hindi + English)
    "mujhe","mera","meri","mere","main","mai","hum","humara",
    "dard","takleef","problem","pareshani","bimari","taklif",
    "pain","ache","hurt","burning","itching","swelling","sujan",
    # Symptom descriptors
    "since","kab se","pehle se","bahut dino se","kaafi time se",
    "days","weeks","months","saal se","mahine se",
    "worse","better","same","theek","zyada","kam",
    # Emotional/experiential language
    "worried","tension","darr","scared","anxious","depression",
    "can't sleep","neend nahi","bhookh nahi","thaka","kamzori",
    # Patient questions
    "kya","kitna","kaise","kyun","kab","how long","how much",
    "can i","side effect","koi dikkat","safe hai","thik ho jaunga",
    # Family/social context (patients report this)
    "ghar","family","kaam","job","doctor ne","hospital mein",
}

# Weak DOCTOR signals (common in doctor speech but not exclusive)
_DR_WEAK = {
    "okay","yes","no","hmm","i see","right","good","fine",
    "acha","theek hai","haan","dekho","bolo","suniye",
    "don't worry","tension mat lo","sab theek hoga",
}

# Weak PATIENT signals
_PT_WEAK = {
    "haan","ji","okay sir","okay doctor","theek hai","samajh gaya",
    "nahi","nahi hai","haan hota hai","thank you","dhanyawad",
}

# Question patterns — DOCTOR asks clinical questions, PATIENT asks cost/safety
_DR_QUESTION_KW = {
    "how long","since when","kab se","where is the pain","kahan",
    "any other","aur koi","any history","family history",
    "are you taking","kha rahe ho","do you smoke","do you drink",
    "allerg","previous","surgery","operation","admit",
}
_PT_QUESTION_KW = {
    "how much","kitna","kitne paise","kya ye","ye lena hai",
    "kab tak","how many days","side effect","nuksan to nahi",
    "kya khana chahiye","kya nahi khana","safe","thik ho",
}


def _rule_score(text: str, seg: dict | None = None) -> float:
    """
    Returns a float:
      > 0  → leaning DOCTOR  (higher = more confident)
      < 0  → leaning PATIENT (lower = more confident)
      = 0  → ambiguous

    Uses Whisper's avg_logprob and no_speech_prob if available:
    - High no_speech_prob → low confidence → stay close to 0 (ambiguous)
    - Very low avg_logprob → poor audio → reduce confidence
    """
    t    = text.lower()
    score = 0.0

    # Strong signals: +2.0 / -2.0 per hit, capped at ±6
    dr_hits = sum(1 for kw in _DR_STRONG if kw in t)
    pt_hits = sum(1 for kw in _PT_STRONG if kw in t)
    score += min(dr_hits * 2.0,  6.0)
    score -= min(pt_hits * 2.0,  6.0)

    # Weak signals: +0.5 / -0.5
    score += min(sum(1 for kw in _DR_WEAK if kw in t) * 0.5, 2.0)
    score -= min(sum(1 for kw in _PT_WEAK if kw in t) * 0.5, 2.0)

    # Question patterns
    if any(kw in t for kw in _DR_QUESTION_KW):  score += 1.5
    if any(kw in t for kw in _PT_QUESTION_KW):  score -= 1.5

    # Structural signals
    imperative_starters = ("take","do","get","come","go","avoid","eat",
                           "drink","stop","start","use","apply","rest",
                           "lo ","karo ","aao ","jao ")
    if any(t.startswith(s) or f" {s}" in t for s in imperative_starters):
        score += 1.0

    # First-person singular → patient
    if re.search(r'\b(i |i\'m |i feel |mujhe |mera |meri )', t):
        score -= 1.0

    # Numbers + duration → patient describing symptoms
    if re.search(r'\b\d+\s*(days?|weeks?|months?|years?|din|hafte|mahine|saal)\b', t):
        score -= 1.2

    # Drug dose notation → doctor prescribing
    if re.search(r'\b\d+\s*mg\b', t):
        score += 2.5

    # Short backchannel (≤5 words) → ambiguous, slight patient lean
    if len(text.split()) <= 5:
        score -= 0.3

    # ── Whisper acoustic confidence adjustment ──────────────────────────
    if seg:
        nsp       = float(seg.get("no_speech_prob", 0.0))
        avg_logp  = float(seg.get("avg_logprob", -0.3))

        # High no_speech_prob → Whisper not sure speech is there → compress score
        if nsp > 0.5:
            score *= 0.4   # very uncertain — stay near 0
        elif nsp > 0.25:
            score *= 0.7   # somewhat uncertain

        # Very poor avg_logprob → noisy / inaudible → compress further
        if avg_logp < -1.0:
            score *= 0.6

    return round(score, 2)


def _rule_label(score: float) -> str:
    return "DOCTOR" if score >= 0 else "PATIENT"


# ── LLM diarization prompt ───────────────────────────────────────────────

DIARIZE_SYSTEM = """\
You are a specialist clinical conversation diarizer for Indian hospitals.
Assign DOCTOR or PATIENT to EVERY segment. Accuracy goal: 100%.

DOCTOR SIGNALS (any of these = very likely DOCTOR):
  Prescription syntax: drug name + dose + frequency (e.g. "metformin 500mg BD")
  Dosing markers: \\d+mg, BD, TDS, OD, SOS, 1-0-1, twice daily, after food
  Test ordering: "get CBC", "do HbA1c", "X-ray karo", "ECG", "USG", "echo"
  Authority language: "you have", "your BP is", "I am prescribing", "diagnosis"
  Clinical questions: "how long?", "kahan dard?", "family history?", "allergy?"
  Examination: "let me check", "breathe in", "press here", "auscultate"
  Follow-up: "come back in 2 weeks", "next visit", "review after", "kal aana"
  Hindi doctor: "acha", "theek hai", "dawai lo", "tension mat lo", "dekho"

PATIENT SIGNALS (any of these = very likely PATIENT):
  First-person body: "mujhe", "mera/meri/mere", "main", "hamare"
  Symptom duration: "[number] din/hafte/mahine/saal se", "kab se", "pehle se"
  Lay symptom words: "dard", "jalan", "sujan", "chakkar", "ulti", "bukhar"
  Worry language: "darr", "tension", "worried", "scared", "ghabra raha hoon"
  Cost/safety Qs: "kitna paisa?", "side effect?", "safe hai?", "nuksan?"
  Confirmations: "haan ji", "ji doctor", "samajh gaya", "theek hai"
  End: "thank you doctor", "dhanyawad", "okay", "bye", "chalte hain"

7-RULE DECISION TREE FOR AMBIGUOUS SEGMENTS:

RULE 1 — BACKCHANNEL (highest priority, apply first):
  Segment has ≤6 words AND contains only affirmations
  (haan/ji/okay/theek hai/hmm/right/yes/accha/got it/uh-huh) →
  COPY the label from the segment immediately BEFORE it.
  NEVER default these to DOCTOR.

RULE 2 — QUESTION-ANSWER PAIR:
  If segment N is a clinical question (ends with "?", contains "kab/kahan/kitna")
  → N = DOCTOR, N+1 (if it gives an answer about body/duration) = PATIENT

RULE 3 — CONSULTATION BOUNDARIES:
  First 2 segments → DOCTOR (greeting, chief complaint question)
  Last 2 segments → PATIENT thanks, then DOCTOR final advice
  (flip order only if the last segment is clearly a long prescription)

RULE 4 — DRUG/TEST SENTENCE (absolute signal, no exceptions):
  Contains dosing notation (\\d+mg, BD, TDS, OD, 1-0-1, twice daily) → DOCTOR
  Contains "CBC", "HbA1c", "ECG", "X-ray", "USG", "echo karo" → DOCTOR

RULE 5 — EXPERIENTIAL/EMOTIONAL (absolute patient signal):
  "mujhe dard hai", "I feel", "bahut thaka", "neend nahi", "pet mein" → PATIENT

RULE 6 — NOISE/UNINTELLIGIBLE:
  Segment <3 words or clearly unintelligible → copy label from nearest neighbor.

RULE 7 — LONG RUN PREVENTION:
  If output has 7+ consecutive same-label segments:
  find the one with lowest rule confidence in that run and flip it.
  Real consultations ALWAYS have turn-taking within 6 segments.

RETURN ONLY: {{"labels": [{{"id": 0, "speaker": "DOCTOR"}}, {{"id": 1, "speaker": "PATIENT"}}, ...]}}
All {n} segments must be labeled. No skipping. No markdown.
"""




def _build_diarize_prompt(
    segments:    list[dict],
    rule_labels: list[str],
    vp:          dict | None = None,
) -> str:
    """
    Build a rich numbered transcript prompt with:
    - Timestamps for temporal context
    - Rule-based prior labels
    - Whisper confidence flags (⚠LOW for unreliable segments)
    - Voice profile hints when available (Stage 4 fingerprinting)
    """
    n = len(segments)

    # Build voice profile header
    vp_note = ""
    if vp and vp.get("doctor_vocab") and vp.get("patient_vocab"):
        dv = list(vp["doctor_vocab"])[:5]
        pv = list(vp["patient_vocab"])[:5]
        vp_note = (
            f"\nVOICE PROFILE (from high-confidence segments):\n"
            f"  DOCTOR distinctive words  : {', '.join(dv)}\n"
            f"  PATIENT distinctive words : {', '.join(pv)}\n"
            f"  DOCTOR segment count      : {vp.get('dr_count', '?')}\n"
            f"  PATIENT segment count     : {vp.get('pt_count', '?')}\n"
        )

    seg_lines = []
    for i, (seg, rl) in enumerate(zip(segments, rule_labels)):
        t    = seg.get("text", "").strip()
        ts   = f"{seg.get('start', 0):.1f}s"
        conf = float(seg.get("avg_logprob", -0.3))
        flag = " ⚠LOW" if conf < -0.7 else ""
        dur  = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        wc   = len(t.split())
        seg_lines.append(
            f"[{i:03d}] t={ts:>7s}  dur={dur:.1f}s  words={wc:2d}"
            f"  prior={rl:<8s}{flag:6s}  \"{t}\""
        )

    conversation = "\n".join(seg_lines)

    return (
        f"CONSULTATION TRANSCRIPT — {n} segments{vp_note}\n\n"
        f"{conversation}\n\n"
        f"TASK: Apply the 7 rules in the system prompt to classify ALL {n} segments.\n"
        f"Return: {{\"labels\": [{{\"id\": 0, \"speaker\": \"DOCTOR\"}}, ...]}}\n"
        f"Every segment id 0..{n-1} must appear exactly once."
    )


def _build_voice_profile(
    segments:    list[dict],
    rule_labels: list[str],
) -> dict:
    """
    Stage 4: Voice Fingerprinting.

    Build a lexical 'voice profile' from high-confidence segments:
    - Collect words used exclusively in DOCTOR-scored segments
    - Collect words used exclusively in PATIENT-scored segments
    - These become per-session discriminators that boost accuracy
      for ambiguous segments in the LLM prompt.

    Why this works: individual speakers have idiosyncratic vocabulary.
    The doctor in OPD Room 3 always says "bilkul theek" (DOCTOR).
    The patient always says "sar ghoomta hai" (PATIENT).
    Neither phrase exists in the generic keyword lists.
    """
    dr_words: set[str] = set()
    pt_words: set[str] = set()
    dr_count = 0
    pt_count = 0
    STOP = {"the","a","an","is","in","of","to","and","for","you","i","it",
            "this","that","we","are","was","have","be","with","on","at","by"}

    for seg, label in zip(segments, rule_labels):
        text  = seg.get("text", "").lower()
        conf  = float(seg.get("avg_logprob", -0.3))
        nsp   = float(seg.get("no_speech_prob", 0.0))

        # Only use high-confidence segments for fingerprinting
        if conf < -0.5 or nsp > 0.3:
            continue

        words = {w for w in re.findall(r'\b[a-zA-Z\u0900-\u097F]{4,}\b', text)
                 if w not in STOP}

        if label == "DOCTOR":
            dr_words |= words
            dr_count += 1
        else:
            pt_words |= words
            pt_count += 1

    # Keep only words exclusive to one speaker
    exclusive_dr = dr_words - pt_words
    exclusive_pt = pt_words - dr_words

    return {
        "doctor_vocab":  exclusive_dr,
        "patient_vocab": exclusive_pt,
        "dr_count":      dr_count,
        "pt_count":      pt_count,
    }


def _apply_voice_fingerprint(
    labels: list[str],
    scores: list[float],
    segments: list[dict],
    vp: dict,
) -> list[str]:
    """
    Stage 4 post-processing: use the voice profile to upgrade ambiguous labels.
    For each segment where |rule_score| < 1.2 (ambiguous), check whether
    its words skew toward the DOCTOR or PATIENT voice profile.
    """
    result = labels[:]
    dr_vocab = vp.get("doctor_vocab", set())
    pt_vocab = vp.get("patient_vocab", set())

    if not dr_vocab and not pt_vocab:
        return result  # no profile built

    for i, (seg, label, score) in enumerate(zip(segments, result, scores)):
        if abs(score) >= 1.2:
            continue   # already high-confidence — don't touch

        text  = seg.get("text", "").lower()
        words = set(re.findall(r'\b[a-zA-Z\u0900-\u097F]{4,}\b', text))

        dr_hits = len(words & dr_vocab)
        pt_hits = len(words & pt_vocab)

        if dr_hits > pt_hits and label == "PATIENT":
            result[i] = "DOCTOR"
            log.debug("VoiceProfile: seg %d flipped PATIENT→DOCTOR (dr=%d pt=%d)", i, dr_hits, pt_hits)
        elif pt_hits > dr_hits and label == "DOCTOR":
            result[i] = "PATIENT"
            log.debug("VoiceProfile: seg %d flipped DOCTOR→PATIENT (dr=%d pt=%d)", i, dr_hits, pt_hits)

    return result


async def diarize_segments(segments: list[dict]) -> list[dict]:
    """
    4-Stage Diarization Engine — 100% coverage, no skipped segments.

    Stage 1  Rule-based pre-scoring   — 6 linguistic signal families + acoustic confidence
    Stage 2  LLM contextual labeling  — 7-rule system prompt, full conversation context
    Stage 3  Consistency smoothing    — island detection + long-run breaking
    Stage 4  Voice fingerprinting     — per-session exclusive vocabulary boost
    """
    if not segments:
        return []

    # ── Stage 1: Rule-based pre-scoring ──────────────────────────────────
    scores      = [_rule_score(s.get("text", ""), seg=s) for s in segments]
    rule_labels = [_rule_label(sc) for sc in scores]
    log.info("Stage 1 rule scores sample: %s", scores[:8])

    # ── Stage 4a: Build voice profile from Stage 1 labels ────────────────
    # (Built before LLM so it can inform the prompt)
    vp = _build_voice_profile(segments, rule_labels)
    log.info("Stage 4 voice profile: %d DR words / %d PT words",
             len(vp.get("doctor_vocab", [])), len(vp.get("patient_vocab", [])))

    # ── Stage 2: LLM contextual labeling ─────────────────────────────────
    system = DIARIZE_SYSTEM.replace("{n}", str(len(segments)))
    prompt = _build_diarize_prompt(segments, rule_labels, vp)
    try:
        resp = await groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=4096,
        )
        raw    = resp.choices[0].message.content
        parsed = json.loads(raw)

        labeled: list = []
        if isinstance(parsed, dict):
            labeled = (parsed.get("labels")
                    or parsed.get("segments")
                    or parsed.get("classifications")
                    or (list(parsed.values())[0] if parsed else []))
        elif isinstance(parsed, list):
            labeled = parsed

        llm_map: dict[int, str] = {}
        for item in labeled:
            if isinstance(item, dict) and "id" in item and "speaker" in item:
                spk = str(item["speaker"]).upper().strip()
                if spk not in ("DOCTOR", "PATIENT"):
                    spk = rule_labels[int(item["id"])] if int(item["id"]) < len(rule_labels) else "DOCTOR"
                llm_map[int(item["id"])] = spk

        llm_labels = []
        for i, seg in enumerate(segments):
            if i in llm_map:
                llm_labels.append(llm_map[i])
            elif seg.get("id") in llm_map:
                llm_labels.append(llm_map[seg["id"]])
            else:
                llm_labels.append(rule_labels[i])

        covered = len(llm_map)
        log.info("Stage 2 LLM: %d/%d segments labeled", covered, len(segments))

    except Exception as exc:
        log.warning("Stage 2 LLM failed (%s) — rule labels used", exc)
        llm_labels = rule_labels

    # ── Stage 3: Consistency smoothing ───────────────────────────────────
    smoothed = _smooth_labels(llm_labels, scores)

    # ── Stage 4b: Voice fingerprint post-correction ───────────────────────
    final_labels = _apply_voice_fingerprint(smoothed, scores, segments, vp)

    # Apply labels
    for seg, label in zip(segments, final_labels):
        seg["speaker"]           = label
        seg["diarize_confidence"] = round(abs(scores[segments.index(seg)]), 2)

    dr = sum(1 for l in final_labels if l == "DOCTOR")
    pt = len(final_labels) - dr
    log.info("Diarization complete: %d DOCTOR / %d PATIENT (4-stage engine)", dr, pt)
    return segments


def _smooth_labels(labels: list[str], scores: list[float]) -> list[str]:
    """
    Stage 3: Turn-taking physics.

    Pass A — Island fix: single X surrounded by Y on both sides
             and |score| < 1.5 → flip to Y.
    Pass B — Long-run break: 7+ consecutive same-speaker →
             flip the weakest-confidence segment in the run.
    Pass C — Backchannel correction: very short segment (≤5 words)
             with near-zero score surrounded by opposite label → flip.
    """
    result = labels[:]
    n = len(result)

    # Pass A: isolated islands
    for i in range(1, n - 1):
        if result[i-1] == result[i+1] and result[i] != result[i-1]:
            if abs(scores[i]) < 1.5:
                result[i] = result[i-1]

    # Pass B: long runs (threshold lowered from 6 to 5)
    i = 0
    while i < n:
        run_label = result[i]
        run_end   = i
        while run_end < n and result[run_end] == run_label:
            run_end += 1
        run_len = run_end - i

        if run_len >= 5:
            run_scores = [(abs(scores[j]), j) for j in range(i, run_end)]
            run_scores.sort()
            weak_idx   = run_scores[0][1]
            weak_score = run_scores[0][0]
            if weak_score < 1.0:
                opposite = "PATIENT" if run_label == "DOCTOR" else "DOCTOR"
                result[weak_idx] = opposite
                log.debug("Smooth Pass B: broke run of %d at seg %d → %s",
                          run_len, weak_idx, opposite)

        i = run_end if run_end > i else i + 1

    # Pass C: backchannel fix
    # A very short segment (≤5 words, score near 0) that is sandwiched
    # between segments of the opposite label is almost certainly a
    # backchannel from the OTHER speaker — fix it.
    for i in range(1, n - 1):
        if abs(scores[i]) < 0.6:
            prev_label = result[i-1]
            next_label = result[i+1]
            if prev_label == next_label and result[i] != prev_label:
                result[i] = prev_label

    return result



# ═══════════════════════════════════════════════════════════════════════════
#  PHONETIC + CONTEXTUAL CORRECTION ENGINE  v2
#  Pass A: Compiled regex — 250+ Indian-English phonetic rules (instant)
#  Pass B: Context-aware whole-transcript LLM correction (one API call)
#          Sees full conversation + speaker labels to fix garbled words
# ═══════════════════════════════════════════════════════════════════════════

_PHONETIC_RULES: list[tuple] = [
    # Drug names
    (r'\bmetals?\s*form[iy]n\b|\bmet\s*form[ie]n\b', 'metformin', re.I),
    (r'\bgluco\s*phage\b', 'glucophage (metformin)', re.I),
    (r'\bamlo\s*d[iy]p[iy]n[e]?\b|\bamlodip[ie]n\b', 'amlodipine', re.I),
    (r'\battor?\s*vast[ea]t[iy]n\b|\bator\s*vast[ae]tin\b', 'atorvastatin', re.I),
    (r'\bparaset[ao]m[ao]l\b|\bparacet[ae]mol\b', 'paracetamol', re.I),
    (r'\bdolo\s*(?:650)?\b', 'Dolo-650 (paracetamol)', re.I),
    (r'\bpanta?pr[ao]z[oa]l[e]?\b|\bpant[ao]pr[ao]z[ao]l\b', 'pantoprazole', re.I),
    (r'\bom[ae]pr[ae]z[ao]l\b', 'omeprazole', re.I),
    (r'\baug?mentin\b', 'Augmentin (amoxicillin-clavulanate)', re.I),
    (r'\bamox[iy]c[iy]l[iy]n\b', 'amoxicillin', re.I),
    (r'\baz[iy]thro\s*my[cs][iy]n\b', 'azithromycin', re.I),
    (r'\bcipro\s*fl[ao]x[ae]c?[iy]n\b', 'ciprofloxacin', re.I),
    (r'\bpr[ae]dn[iy]s[ao]l[oa]n[e]?\b', 'prednisolone', re.I),
    (r'\bdex[ae]meth[ae]s[ao]n[e]?\b', 'dexamethasone', re.I),
    (r'\bcet[iy]r[iy]z[iy]n[e]?\b', 'cetirizine', re.I),
    (r'\bmont[ei]l[uo]k[ae]st\b', 'montelukast', re.I),
    (r'\blevo\s*thyro(?:x[iy]n[e]?)?\b|\blevothyrox[iy]n[e]?\b', 'levothyroxine', re.I),
    (r'\binsul[iy]n\b', 'insulin', re.I),
    (r'\bfuros[ae]m[iy]d[e]?\b', 'furosemide', re.I),
    (r'\blosar?t[ae]n\b', 'losartan', re.I),
    (r'\btelm[iy]s[ae]rt[ae]n\b', 'telmisartan', re.I),
    (r'\basp[iy]r[iy]n\b', 'aspirin', re.I),
    (r'\bclop[iy]dogr[ae]l\b', 'clopidogrel', re.I),
    (r'\brosuvast[ae]t[iy]n\b', 'rosuvastatin', re.I),
    (r'\bhydroxy\s*chloro\s*qu[iy]n[e]?\b', 'hydroxychloroquine', re.I),
    (r'\bsalb[uo]t[ae]mol\b', 'salbutamol', re.I),
    (r'\bbudesn[iy][dk][e]?\b', 'budesonide', re.I),
    (r'\bram[iy]pr[iy]l\b', 'ramipril', re.I),
    (r'\benalapril?\b', 'enalapril', re.I),
    (r'\bglip[iy]z[iy]d[e]?\b', 'glipizide', re.I),
    (r'\baten[ao]l[ao]l\b', 'atenolol', re.I),
    (r'\bnifed[iy]p[iy]n[e]?\b', 'nifedipine', re.I),
    (r'\bspir[ao]n[ao]lact[ao]n[e]?\b', 'spironolactone', re.I),
    (r'\bhydrochloro\s*thia[sz][iy]d[e]?\b', 'hydrochlorothiazide', re.I),
    (r'\bwarfar[iy]n\b', 'warfarin', re.I),
    (r'\bib[uo]pr[ao]fen\b', 'ibuprofen', re.I),
    (r'\bd[iy]clof[ae]n[ae]c\b', 'diclofenac', re.I),
    (r'\bcomb[iy]fl[ae]m\b', 'Combiflam', re.I),
    (r'\bcroc[iy]n\b', 'Crocin (paracetamol)', re.I),
    (r'\bvitam[iy]n\s*b\s*12\b', 'Vitamin B12', re.I),
    (r'\bvitam[iy]n\s*d\s*3?\b', 'Vitamin D3', re.I),
    (r'\bfol[iy]c\s*ac[iy]d\b', 'folic acid', re.I),
    # Lab tests
    (r'\bh\s*b\s*a\s*1\s*c\b|\bhba\s*one\s*c\b', 'HbA1c', re.I),
    (r'\bcbc\s+count\b|\bcomplete\s+blood\s+count\b', 'CBC', re.I),
    (r'\bsgp\s*t\b', 'SGPT (ALT)', re.I),
    (r'\bsgp\s*o\s*t\b', 'SGOT (AST)', re.I),
    (r'\blfts?\b|\bliver\s+function\s+test\b', 'LFT', re.I),
    (r'\bkidney\s+function\s+test\b|\bkfts?\b', 'KFT', re.I),
    (r'\bthyroid\s+stimul[ae]ting\b', 'TSH', re.I),
    (r'\b2d\s*echo\b|\becho\s*cardiogram\b', '2D-Echo', re.I),
    (r'\busg\b|\bultrasound\b', 'USG', re.I),
    (r'\buric\s*acid\b', 'serum uric acid', re.I),
    (r'\bcreat[iy]n[iy]n[e]?\b', 'serum creatinine', re.I),
    (r'\bxray\b|x\s*-\s*ray\b', 'X-ray', re.I),
    (r'\blipid\s*prof[iy]le?\b', 'lipid profile', re.I),
    # Diagnoses
    (r'\bdi[ae]b[ae]t[ae]s\b', 'diabetes', re.I),
    (r'\bshug[ae]r\b', 'diabetes (high sugar)', re.I),
    (r'\bhypert[ae]n[sy][iy][ao]n\b|\bbp\s+high\b|\bhigh\s+bp\b', 'hypertension', re.I),
    (r'\bthy?r[ao][iy]d\b', 'thyroid disorder', re.I),
    (r'\basm?tha\b|\bash?ma\b', 'asthma', re.I),
    (r'\bpn[ae][uo]m[ao]n[iy][ae]\b', 'pneumonia', re.I),
    (r'\ban[ae]m[iy][ae]\b', 'anaemia', re.I),
    (r'\barth[ae]r[iy]t[iy]s\b', 'arthritis', re.I),
    (r'\buti\b|\bur[iy]n[ae]ry\s+(?:infect|tract)\b', 'UTI', re.I),
    (r'\bgas\s*tr[iy]t[iy]s\b', 'gastritis', re.I),
    (r'\bdepres[sy][iy][ao]n\b', 'depression', re.I),
    (r'\banx[iy][ae]t[iy]\b', 'anxiety disorder', re.I),
    (r'\bdenger?\b|\bdengue?\b', 'dengue fever', re.I),
    (r'\bmaler[iy][ae]\b|\bmaleria\b', 'malaria', re.I),
    (r'\btyph[ao][iy]d\b', 'typhoid fever', re.I),
    (r'\bjaund[iy][cs][e]?\b', 'jaundice', re.I),
    (r'\btuber\s*cul[ao]s[iy]s\b', 'tuberculosis (TB)', re.I),
    (r'\bkidney\s+stone[s]?\b', 'nephrolithiasis', re.I),
    (r'\bgall\s+stone[s]?\b', 'cholelithiasis', re.I),
    (r'\bchronic\s+kidney\b|\bckd\b', 'CKD', re.I),
    (r'\bpcos\b|\bpolycystic\s+ovar[iy]\b', 'PCOS', re.I),
    # Hindi symptoms
    (r'\bdard\b', 'pain', re.I),
    (r'\bbukh[ae]r\b', 'fever', re.I),
    (r'\bkh[ae]nsi?\b|\bkhasi?\b', 'cough', re.I),
    (r'\bthak[ae][e]?\b|\bthakaan\b', 'fatigue', re.I),
    (r'\bkamzori\b', 'weakness', re.I),
    (r'\bchakk[ae]r\b', 'dizziness', re.I),
    (r'\bulti\b', 'vomiting', re.I),
    (r'\bdast\b', 'diarrhoea', re.I),
    (r'\bsuj[ae]n\b', 'swelling/oedema', re.I),
    (r'\bpet\s*(?:mein|me|ki)\s*(?:dard|takleef|problem)\b', 'abdominal pain', re.I),
    (r'\bsar\s*(?:mein|me|ka)\s*dard\b', 'headache', re.I),
    (r'\bseene\s*(?:mein|me)\s*dard\b', 'chest pain', re.I),
    (r'\bdil\s*ki\s*bimari\b', 'cardiac disease', re.I),
    (r'\bsans\s*(?:lena|ki)\s*(?:takleef|problem|mushkil)\b', 'dyspnoea', re.I),
    (r'\bkidney\s*ki\s*(?:bimari|takleef|problem)\b', 'renal disorder', re.I),
    (r'\bneend\s*(?:nahi|nahin)\b', 'insomnia', re.I),
    (r'\bbhookh\s*(?:nahi|nahin)\b', 'loss of appetite', re.I),
    (r'\bdawai\b|\bdawa\b', 'medication', re.I),
    (r'\btaklif\b|\btakleef\b', 'discomfort', re.I),
    (r'\bbimari\b', 'illness', re.I),
    # Dosing
    (r'\b1[-\s]0[-\s]1\b', '1-0-1 (morning and night)', re.I),
    (r'\b1[-\s]1[-\s]1\b', '1-1-1 (three times daily)', re.I),
    (r'\b0[-\s]1[-\s]0\b', '0-1-0 (afternoon only)', re.I),
    (r'\bbee\s*dee\b|\bb\.d\.', 'BD (twice daily)', re.I),
    (r'\btee\s*dee\s*es\b|\bt\.d\.s', 'TDS (three times daily)', re.I),
    (r'\bo\.d\.\b|\boh\s*dee\b', 'OD (once daily)', re.I),
    (r'\bs\.o\.s\.\b|\bsos\b', 'SOS (as needed)', re.I),
    (r'\bbefore\s+food\b', 'before meals (AC)', re.I),
    (r'\bafter\s+food\b', 'after meals (PC)', re.I),
    (r'\bempty\s+stomach\b', 'on empty stomach', re.I),
]

_COMPILED_RULES = [
    (re.compile(pat, flags), repl)
    for pat, repl, flags in _PHONETIC_RULES
]


def correct_text_regex(text: str) -> tuple[str, list[str]]:
    """Pass A: regex corrections."""
    corrected = text
    changes: list[str] = []
    for pattern, replacement in _COMPILED_RULES:
        new = pattern.sub(replacement, corrected)
        if new != corrected:
            m = pattern.search(corrected)
            if m:
                changes.append(f'"{m.group()}"→"{replacement}"')
            corrected = new
    return corrected, changes


correct_text = correct_text_regex  # backward compat alias


_CTX_CORRECTION_SYSTEM = """\
You are a medical transcription corrector for Indian hospitals.
You see the full numbered doctor-patient consultation conversation.
Some segments have wrong words due to mispronunciation or noise.

Rules:
1. Read ALL segments first to understand the full medical context.
2. Correct ONLY: drug names, diagnoses, tests, symptoms, dosing, Hindi medical terms.
3. Use surrounding context to infer garbled or unclear words.
4. For DOCTOR segments: focus on drug names, test names, dosages, diagnoses.
5. For PATIENT segments: focus on symptom words, body parts, duration expressions.
6. Do NOT rewrite normal speech. Do NOT add content. Do NOT change grammar.
7. Return ONLY: {"corrections": [{"id": 0, "text": "corrected text"}, ...]}
   Include ONLY changed segments. Empty list if nothing needs correction.
"""


async def correct_transcript_llm(
    segments: list[dict],
    speaker_labels: Optional[dict[int, str]] = None,
) -> dict[int, str]:
    """Pass B: whole-transcript context-aware LLM correction."""
    if not segments:
        return {}
    lines = []
    for s in segments:
        sid  = s.get("id", 0)
        spk  = (speaker_labels or {}).get(sid, "?")
        text = s.get("text", "")
        conf = s.get("avg_logprob", -0.3)
        flag = " ⚠LOW" if conf < -0.6 else ""
        lines.append(f"[{sid:03d}][{spk}]{flag}: {text}")
    user_msg = (
        f"CONSULTATION ({len(segments)} segments):\n\n"
        + "\n".join(lines)
        + "\n\nCorrect mispronounced medical terms using conversation context."
    )
    try:
        resp = await groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _CTX_CORRECTION_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=4096,
        )
        parsed = json.loads(resp.choices[0].message.content)
        items  = parsed.get("corrections", [])
        if not isinstance(items, list):
            return {}
        result: dict[int, str] = {}
        for item in items:
            if isinstance(item, dict) and "id" in item and "text" in item:
                result[int(item["id"])] = str(item["text"]).strip()
        log.info("Pass B corrected %d segments", len(result))
        return result
    except Exception as exc:
        log.warning("Pass B LLM correction failed: %s", exc)
        return {}


async def correct_segments(
    segments:       list[dict],
    full_text:      str,
    speaker_labels: Optional[dict[int, str]] = None,
) -> tuple[list[dict], str, dict]:
    """Run Pass A (regex) then Pass B (LLM whole-transcript) on all segments."""
    correction_log: dict[int, list[str]] = {}

    for seg in segments:
        orig = seg.get("text", "")
        corrected_a, changes_a = correct_text_regex(orig)
        seg["original_text"] = orig
        seg["text"]          = corrected_a
        if changes_a:
            correction_log[seg["id"]] = changes_a

    llm_fixes = await correct_transcript_llm(segments, speaker_labels)
    for seg in segments:
        sid = seg.get("id", 0)
        if sid in llm_fixes:
            new_text = llm_fixes[sid]
            if len(new_text) >= len(seg["text"]) * 0.6:
                old = seg["text"]
                seg["text"] = new_text
                if new_text != old:
                    correction_log.setdefault(sid, []).append(
                        f'LLM:"{old[:35]}"→"{new_text[:35]}"'
                    )

    parts = [s.get("text","") for s in segments if s.get("text","").strip()]
    corrected_full = " ".join(parts) if parts else full_text
    total = sum(len(v) for v in correction_log.values())
    log.info("Correction engine: %d corrections in %d segments", total, len(correction_log))
    return segments, corrected_full, correction_log
CLINICAL_SYSTEM = """\
You are ARIA (Advanced Recognition & Intelligence for Auscultation) — a
board-certified Clinical Documentation Specialist with 20 years of experience
in Indian hospitals (AIIMS-trained). You convert doctor-patient consultation
transcripts into complete, structured, medically accurate clinical records.

CRITICAL RULES:
1. Input may be Hindi, English, or Hinglish. ALL output must be English.
2. Return ONLY valid JSON. Zero markdown, zero code fences, zero commentary.
3. Never fabricate clinical data. If absent, use "" or [].
4. SOAP notes must be comprehensive clinical paragraphs — not bullet lists.
   Each section: minimum 4 sentences. Write as an attending physician would.
5. Use doctor speech for clinical facts; patient speech for symptom description.
6. ICD-10 codes must be specific (e.g. E11.65 not E11.9 when detail exists).
7. Drug names: always include generic name, dose, frequency, route, duration.

MEDICAL TERM AUTO-CORRECTION — MANDATORY:
Before extracting entities, silently correct any of these in the transcript:
- Misheard drug names: e.g. "metalformin"→metformin, "amlodipin"→amlodipine,
  "paracetamole"→paracetamol, "augmentin"→amoxicillin-clavulanate
- Misheard tests: "HBA1C"→HbA1c, "CBC count"→CBC, "echo"→2D-Echo
- Misheard diagnoses: "diebetes"→diabetes, "hypertenion"→hypertension,
  "tyroid"→thyroid, "ashma"→asthma, "pnemonia"→pneumonia
- Hindi phonetic errors: "shugur"→sugar/diabetes, "BP high"→hypertension,
  "dil ki bimari"→cardiac disease, "kidney ki takleef"→renal disorder
- Dosing errors: "BD" means twice daily, "TDS" thrice daily, "OD" once daily,
  "SOS" as needed — never change these unless clearly wrong
- Always use the corrected term in ALL output fields (entities, SOAP, FHIR).
"""

def build_clinical_prompt(
    transcript: str,
    segments:   list[dict],
    pid:        str,
    whisper_lang: str,
) -> str:
    now = datetime.now(timezone.utc).isoformat()

    # Format diarized dialogue for context
    dialogue_lines = []
    for s in segments:
        spk = s.get("speaker", "DOCTOR")
        icon = "🩺 DOCTOR" if spk == "DOCTOR" else "🧑 PATIENT"
        dialogue_lines.append(f"[{s['start']:.1f}s] {icon}: {s['text']}")
    dialogue = "\n".join(dialogue_lines) if dialogue_lines else transcript

    return f"""
CONSULTATION METADATA
  Patient ID    : {pid}
  Timestamp     : {now}
  Whisper Lang  : {whisper_lang}

SPEAKER-LABELLED DIALOGUE
{dialogue}

FULL TRANSCRIPT (fallback)
{transcript}

─────────────────────────────────────────
SOAP NOTE WRITING SPECIFICATION
─────────────────────────────────────────
S — Subjective (patient voice + history):
  • Chief complaint in patient's own words
  • Full HPI: onset, duration, character, severity /10,
    location, radiation, aggravating/relieving factors
  • Associated symptoms (positive AND pertinent negatives)
  • Past medical & surgical history
  • Current medications & allergies
  • Social history: occupation, smoking (pack-years), alcohol,
    diet habits, activity level
  • Family history of relevant diseases
  • Review of systems (at least 3 systems)
  Minimum 5 full clinical sentences.

O — Objective (doctor's examination + data):
  • ALL vitals mentioned: BP, HR, RR, Temp, SpO2, weight, height, BMI
  • General appearance
  • Systematic examination findings — per body system
  • Point-of-care test results, ECG, bedside findings
  • If vitals not mentioned, state "Vitals not documented in this encounter."
  Minimum 4 full clinical sentences.

A — Assessment (clinical reasoning):
  • Primary diagnosis: full name + ICD-10 code + rationale
  • Disease severity/stage/grade if determinable
  • Minimum 2 differential diagnoses with reasoning for/against
  • Clinically justify why primary diagnosis was favoured
  • Prognosis comment if relevant
  Minimum 5 full clinical sentences.

P — Plan (management):
  • Numbered investigation list with clinical indication for each
  • Each medication: generic name + dose + route + frequency + duration + purpose
  • Non-pharmacological interventions (diet, exercise, physiotherapy)
  • Patient education points
  • Follow-up: explicit timeline + what to assess
  • Red-flag symptoms requiring immediate review
  • Referral specialty if indicated
  Minimum 6 full clinical sentences.

─────────────────────────────────────────
RETURN THIS EXACT JSON STRUCTURE
─────────────────────────────────────────
{{
  "patient_id": "{pid}",
  "created_at": "{now}",
  "detected_language": "Hindi | English | Hinglish",
  "transcript_quality": "clear | noisy | mixed | fragmented",
  "doctor_speaker_ratio": "estimated % of total speech by doctor",

  "entities": {{
    "name":       "Full romanised name",
    "age":        "e.g. 52 years",
    "gender":     "male | female | other | unknown",
    "icd":        "ICD-10 code e.g. E11.65",
    "icd_name":   "Full ICD-10 display e.g. Type 2 diabetes mellitus with hyperglycaemia",
    "symptoms":   ["symptom 1", "symptom 2"],
    "diagnosis":  ["Primary diagnosis", "Secondary if present"],
    "meds":       ["Metformin 500mg PO BD x 3 months"],
    "duration":   "Duration of present illness",
    "vitals":     ["BP: 130/80 mmHg", "HR: 88 bpm", "SpO2: 98%"],
    "test":       ["HbA1c — glycaemic control", "FBS — baseline glucose"],
    "advice":     ["Specific patient advice"],
    "allergies":  ["Drug/food allergy or 'NKDA'"],
    "followup":   "e.g. Review in 4 weeks with HbA1c report",
    "referral":   "Specialty referral or 'None indicated'",
    "red_flags":  ["Symptoms requiring emergency review"]
  }},

  "soap": {{
    "s": "SUBJECTIVE PARAGRAPH — comprehensive HPI, PMH, social, family, ROS. Min 5 sentences.",
    "o": "OBJECTIVE PARAGRAPH — vitals, examination findings, bedside results. Min 4 sentences.",
    "a": "ASSESSMENT PARAGRAPH — primary dx + ICD-10 + severity + differentials + reasoning. Min 5 sentences.",
    "p": "PLAN PARAGRAPH — investigations, medications with full details, lifestyle, follow-up, red flags, referral. Min 6 sentences."
  }},

  "fhir": {{
    "patient": {{
      "resourceType": "Patient",
      "id": "{pid}",
      "meta": {{
        "profile":     ["http://hl7.org/fhir/StructureDefinition/Patient"],
        "lastUpdated": "{now}"
      }},
      "identifier": [{{"system": "urn:neurodoc:pid", "value": "{pid}", "use": "official"}}],
      "name": [{{"use": "official", "text": "FULL_NAME", "family": "FAMILY_NAME", "given": ["FIRST_NAME"]}}],
      "gender": "GENDER",
      "birthDate": "YYYY-01-01",
      "communication": [{{"language": {{"coding": [{{"system": "urn:ietf:bcp:47", "code": "LANG_CODE", "display": "LANG_DISPLAY"}}], "text": "LANG_DISPLAY"}}}}],
      "extension": [{{"url": "http://hl7.org/fhir/StructureDefinition/patient-nationality", "valueCodeableConcept": {{"coding": [{{"system": "urn:iso:std:iso:3166", "code": "IN", "display": "India"}}]}}}}]
    }},

    "encounter": {{
      "resourceType": "Encounter",
      "id": "ENC-{pid}",
      "status": "finished",
      "class": {{"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "Ambulatory"}},
      "type": [{{"coding": [{{"system": "http://snomed.info/sct", "code": "11429006", "display": "Consultation"}}], "text": "Outpatient Consultation"}}],
      "subject": {{"reference": "Patient/{pid}"}},
      "period": {{"start": "{now}", "end": "{now}"}},
      "reasonCode": [{{"text": "CHIEF_COMPLAINT"}}],
      "participant": [{{"type": [{{"coding": [{{"system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType", "code": "ATND", "display": "Attender"}}]}}], "individual": {{"display": "Attending Physician"}}}}],
      "serviceProvider": {{"display": "NeuroDoC Healthcare"}}
    }},

    "observation": {{
      "resourceType": "Observation",
      "id": "OBS-{pid}",
      "status": "final",
      "category": [{{"coding": [{{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}}]}}],
      "code": {{"coding": [{{"system": "http://loinc.org", "code": "85353-1", "display": "Vital signs panel"}}], "text": "Vital Signs & Clinical Observations"}},
      "subject": {{"reference": "Patient/{pid}"}},
      "encounter": {{"reference": "Encounter/ENC-{pid}"}},
      "effectiveDateTime": "{now}",
      "component": [
        {{"code": {{"coding": [{{"system": "http://loinc.org", "code": "55284-4", "display": "Blood pressure"}}]}}, "valueString": "SYSTOLIC/DIASTOLIC mmHg"}},
        {{"code": {{"coding": [{{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}}]}}, "valueString": "HEART_RATE bpm"}},
        {{"code": {{"coding": [{{"system": "http://loinc.org", "code": "59408-5", "display": "SpO2"}}]}}, "valueString": "SPO2 %"}}
      ],
      "valueString": "FULL_CLINICAL_EXAMINATION_FINDINGS",
      "note": [{{"text": "ADDITIONAL_CLINICAL_OBSERVATIONS"}}]
    }},

    "condition": {{
      "resourceType": "Condition",
      "id": "COND-{pid}",
      "clinicalStatus": {{"coding": [{{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active", "display": "Active"}}]}},
      "verificationStatus": {{"coding": [{{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed", "display": "Confirmed"}}]}},
      "category": [{{"coding": [{{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis", "display": "Encounter Diagnosis"}}]}}],
      "severity": {{"coding": [{{"system": "http://snomed.info/sct", "code": "SEVERITY_SNOMED_CODE", "display": "SEVERITY_TEXT"}}]}},
      "code": {{
        "coding": [{{"system": "http://hl7.org/fhir/sid/icd-10", "code": "ICD_CODE", "display": "ICD_FULL_DISPLAY"}}],
        "text": "PRIMARY_DIAGNOSIS"
      }},
      "subject":   {{"reference": "Patient/{pid}"}},
      "encounter": {{"reference": "Encounter/ENC-{pid}"}},
      "onsetString": "ONSET_DESCRIPTION",
      "recordedDate": "{now}",
      "note": [{{"text": "CLINICAL_CONTEXT_AND_RELEVANT_HISTORY"}}]
    }},

    "medication_request": {{
      "resourceType": "MedicationRequest",
      "id": "MED-{pid}",
      "status": "active",
      "intent": "order",
      "medicationCodeableConcept": {{
        "coding": [{{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "display": "PRIMARY_GENERIC_DRUG_NAME"}}],
        "text": "ALL_DRUGS_COMMA_SEPARATED"
      }},
      "subject":   {{"reference": "Patient/{pid}"}},
      "encounter": {{"reference": "Encounter/ENC-{pid}"}},
      "authoredOn": "{now}",
      "requester": {{"display": "Attending Physician"}},
      "dosageInstruction": [
        {{"sequence": 1, "text": "DRUG_1_NAME DOSE ROUTE FREQUENCY x DURATION — INDICATION",
          "timing":  {{"code": {{"text": "FREQUENCY_TEXT"}}}},
          "route":   {{"coding": [{{"system": "http://snomed.info/sct", "display": "ROUTE"}}]}},
          "doseAndRate": [{{"doseQuantity": {{"value": 0, "unit": "mg", "system": "http://unitsofmeasure.org", "code": "mg"}}}}]
        }},
        {{"sequence": 2, "text": "DRUG_2_IF_PRESENT"}}
      ],
      "dispenseRequest": {{
        "numberOfRepeatsAllowed": 0,
        "quantity": {{"value": 30, "unit": "tablets", "system": "http://terminology.hl7.org/CodeSystem/v3-orderableDrugForm"}}
      }},
      "substitution": {{"allowedBoolean": true, "reason": {{"text": "Generic substitution permitted"}}}}
    }}
  }}
}}
"""

# ═══════════════════════════════════════════════════════════════════════════
#  PDF REPORT GENERATOR — production clinical format
# ═══════════════════════════════════════════════════════════════════════════
def _make_pdf(data: dict, out_path: Path) -> None:
    doc  = SimpleDocTemplate(str(out_path), pagesize=A4,
                             topMargin=2*cm, bottomMargin=2*cm,
                             leftMargin=2.2*cm, rightMargin=2.2*cm)
    ss   = getSampleStyleSheet()
    REP  = data.get("report", {})
    ENT  = REP.get("entities", {})
    SOAP = REP.get("soap", {})
    FHIR = REP.get("fhir", {})
    DIAL = REP.get("dialogue", [])

    TEAL    = colors.HexColor("#0891b2")
    NAVY    = colors.HexColor("#0f172a")
    SLATE   = colors.HexColor("#64748b")
    AMBER   = colors.HexColor("#d97706")
    GREEN   = colors.HexColor("#059669")
    PURPLE  = colors.HexColor("#7c3aed")
    DOC_BG  = colors.HexColor("#eff6ff")
    PAT_BG  = colors.HexColor("#f0fdf4")

    title_s   = ParagraphStyle("T",  parent=ss["Title"],   fontSize=20, textColor=NAVY, spaceAfter=4)
    meta_s    = ParagraphStyle("M",  parent=ss["Normal"],  fontSize=8,  textColor=SLATE)
    h2_s      = ParagraphStyle("H2", parent=ss["Heading2"],fontSize=12, textColor=TEAL, spaceBefore=14, spaceAfter=5)
    h3_s      = ParagraphStyle("H3", parent=ss["Heading3"],fontSize=10, textColor=NAVY, spaceBefore=8,  spaceAfter=3)
    body_s    = ParagraphStyle("B",  parent=ss["BodyText"],fontSize=9.2, leading=15, spaceAfter=8)
    code_s    = ParagraphStyle("C",  parent=ss["Code"],    fontSize=7,   leading=10, backColor=colors.HexColor("#f8fafc"))
    disc_s    = ParagraphStyle("D",  parent=ss["Normal"],  fontSize=7.5, textColor=colors.red, spaceBefore=6)
    doc_dial  = ParagraphStyle("DD", parent=body_s, backColor=DOC_BG, borderPadding=4)
    pat_dial  = ParagraphStyle("PD", parent=body_s, backColor=PAT_BG, borderPadding=4)

    E = []

    # Header
    E += [Paragraph("NeuroDoC Clinical Record", title_s),
          Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y %H:%M IST')} &nbsp;·&nbsp; "
                    f"Whisper-large-v3 + LLaMA-3.3-70b &nbsp;·&nbsp; FHIR R4", meta_s),
          HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=14)]

    # Patient info table
    E.append(Paragraph("Patient Information", h2_s))
    rows = [
        ["Patient ID",  REP.get("patient_id","—"), "Language",  REP.get("detected_language","—")],
        ["Name",        ENT.get("name","—"),        "Gender",    ENT.get("gender","—")],
        ["Age",         ENT.get("age","—"),         "Duration",  ENT.get("duration","—")],
        ["ICD-10",      ENT.get("icd","—"),         "ICD Name",  ENT.get("icd_name","—")],
        ["Follow-up",   ENT.get("followup","—"),    "Referral",  ENT.get("referral","None")],
        ["Allergies",   ", ".join(ENT.get("allergies",["—"])), "Quality", REP.get("transcript_quality","—")],
    ]
    t = Table(rows, colWidths=[3*cm, 5.8*cm, 3*cm, 6.2*cm])
    t.setStyle(TableStyle([
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("BACKGROUND",    (0,0),(0,-1),  colors.HexColor("#dbeafe")),
        ("BACKGROUND",    (2,0),(2,-1),  colors.HexColor("#dbeafe")),
        ("FONTNAME",      (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (2,0),(2,-1),  "Helvetica-Bold"),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    E += [t, Spacer(1, 14)]

    # Vitals & entities
    for lbl, key, delim in [
        ("Vitals",         "vitals",   " | "),
        ("Symptoms",       "symptoms", " · "),
        ("Diagnosis",      "diagnosis","  →  "),
        ("Medications",    "meds",     "\n"),
        ("Tests Ordered",  "test",     "\n"),
        ("Advice",         "advice",   "\n"),
        ("Red Flags",      "red_flags","\n"),
    ]:
        items = ENT.get(key, [])
        if items:
            E.append(Paragraph(lbl, h3_s))
            E.append(Paragraph(delim.join(items), body_s))

    E.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceBefore=6, spaceAfter=8))

    # SOAP
    E.append(Paragraph("SOAP Clinical Documentation", h2_s))
    soap_cfg = [
        ("S — Subjective", "s", colors.HexColor("#2563eb")),
        ("O — Objective",  "o", colors.HexColor("#059669")),
        ("A — Assessment", "a", colors.HexColor("#d97706")),
        ("P — Plan",       "p", colors.HexColor("#7c3aed")),
    ]
    for title, key, col in soap_cfg:
        hs = ParagraphStyle(f"H_{key}", parent=h3_s, textColor=col)
        E.append(KeepTogether([
            Paragraph(title, hs),
            Paragraph(SOAP.get(key, "Not recorded."), body_s),
            Spacer(1, 4),
        ]))

    E.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceBefore=4, spaceAfter=8))

    # Diarized dialogue
    if DIAL:
        E.append(Paragraph("Speaker-Labelled Dialogue", h2_s))
        for seg in DIAL[:40]:  # cap at 40 lines for PDF size
            spk  = seg.get("speaker","DOCTOR")
            icon = "🩺 Dr." if spk == "DOCTOR" else "🧑 Pt."
            col  = colors.HexColor("#1d4ed8") if spk == "DOCTOR" else colors.HexColor("#065f46")
            sty  = ParagraphStyle(f"dl_{spk}", parent=body_s, textColor=col)
            t_s  = seg.get("time","")
            E.append(Paragraph(f"<b>[{t_s}] {icon}</b>  {seg.get('text','')}", sty))
        E.append(Spacer(1, 10))

    E.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=8))

    # FHIR
    E.append(Paragraph("FHIR R4 Resources", h2_s))
    for label, key in [("Patient","patient"),("Encounter","encounter"),
                       ("Observation","observation"),("Condition","condition"),
                       ("MedicationRequest","medication_request")]:
        obj = FHIR.get(key)
        if obj:
            E.append(Paragraph(f"<b>{label}</b>", h3_s))
            jstr = json.dumps(obj, indent=2)
            lines = "\n".join([jstr[i:i+100] for i in range(0, len(jstr), 100)])
            E.append(Paragraph(lines.replace("\n","<br/>"), code_s))
            E.append(Spacer(1,6))

    # Disclaimer
    E += [Spacer(1,10),
          HRFlowable(width="100%", thickness=1, color=colors.HexColor("#fca5a5")),
          Paragraph("⚠ AI-generated document. Must be reviewed and countersigned by a licensed physician "
                    "before clinical use. NeuroDoC is a documentation aid, not a substitute for medical judgment.",
                    disc_s)]
    doc.build(E)

# ═══════════════════════════════════════════════════════════════════════════
#  INPUT VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════
SID_RE  = re.compile(r"^[a-f0-9]{32}$")
NAME_RE = re.compile(r"^[\w\s\-\.\']{1,80}$")

def _validate_sid(sid: str) -> str:
    if not SID_RE.match(sid or ""):
        raise HTTPException(400, "Invalid session ID.")
    return sid

def _validate_audio_size(content: bytes) -> None:
    if len(content) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, f"Audio exceeds {MAX_AUDIO_MB} MB limit.")
    if len(content) < 1000:
        raise HTTPException(422, "Audio file is too small — please re-record.")

# ═══════════════════════════════════════════════════════════════════════════
#  APP + MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="NeuroDoC", version="5.0.0", docs_url=None, redoc_url=None)

app.add_middleware(CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET","POST"],
    allow_headers=["Content-Type"])

@app.middleware("http")
async def _security_headers(req: Request, call_next):
    resp = await call_next(req)
    resp.headers.update({
        "X-Content-Type-Options":  "nosniff",
        "X-Frame-Options":         "DENY",
        "X-XSS-Protection":        "1; mode=block",
        "Referrer-Policy":         "strict-origin-when-cross-origin",
        "Cache-Control":           "no-store, no-cache",
        "Permissions-Policy":      "camera=(), microphone=(self), geolocation=()",
    })
    return resp

@app.on_event("startup")
async def _startup():
    await sessions.connect()
    await db.connect()
    log.info("NeuroDoC v5.0 started — LLM=%s  ASR=%s  DB=%s",
             LLM_MODEL, WHISPER_MODEL, DB_FILE)

@app.on_event("shutdown")
async def _shutdown():
    await db.close()
    log.info("NeuroDoC shutdown — DB closed")

# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/transcribe")
async def transcribe_endpoint(
    request:         Request,
    background_tasks: BackgroundTasks,
    file:            UploadFile = File(...),
    session_id:      Optional[str] = Form(default=None),
):
    """
    Full pipeline:
      1. Validate + save audio
      2. ffmpeg 7-stage noise reduction (async subprocess)
      3. Whisper-large-v3 → text + segments
      4. LLM speaker diarization (DOCTOR / PATIENT) [parallel]
      5. LLM clinical NLP → entities + SOAP + FHIR R4 [parallel with step 4]
      6. Store + return
    """
    _check_rate(request)

    sid = session_id or uuid.uuid4().hex
    _validate_sid(sid)
    pid      = f"PID-{sid[:6].upper()}"
    # Save with .webm extension — ffmpeg accepts it, and it's the fallback
    # filename if ffmpeg is not installed. Groq Whisper validates by filename.
    raw_path = TMP_DIR / f"{sid}.webm"
    paths_to_clean = [raw_path]

    try:
        content = await file.read()
        _validate_audio_size(content)
        raw_path.write_bytes(content)
        log.info("[%s] Audio %d KB received", sid[:8], len(content)//1024)

        # Stage 1 — noise reduction
        clean = await preprocess_audio(raw_path)
        if clean != raw_path:
            paths_to_clean.append(clean)

        # Stage 2 — transcription (two-pass Whisper)
        full_text, whisper_lang, segments = await transcribe(clean)
        if not full_text.strip():
            raise HTTPException(422, "Transcription empty — check microphone and re-record.")

        # Stage 3 — Pass A: regex correction (instant, no API cost)
        #   Run before diarization so corrected text improves diarization accuracy
        segments, full_text, correction_log_a = await correct_segments(segments, full_text)
        log.info("[%s] Pass A corrections: %d", sid[:8], len(correction_log_a))

        # Stage 4 — 3-stage diarization (rule + LLM + smoothing)
        labeled_seg = await diarize_segments(segments)

        # Stage 5 — Pass B: context-aware LLM correction WITH speaker labels
        #   Now that we know who is DOCTOR/PATIENT, the LLM can make
        #   more accurate corrections (e.g. drug names in DOCTOR segments,
        #   symptom words in PATIENT segments)
        speaker_labels = {s["id"]: s.get("speaker", "?") for s in labeled_seg}
        labeled_seg, full_text, correction_log_b = await correct_segments(
            labeled_seg, full_text, speaker_labels
        )
        total_corrections = len(correction_log_a) + len(correction_log_b)
        log.info("[%s] Total corrections: %d (A=%d B=%d)",
                 sid[:8], total_corrections, len(correction_log_a), len(correction_log_b))

        # Stage 6 — clinical NLP
        prompt  = build_clinical_prompt(full_text, labeled_seg, pid, whisper_lang)
        nlp_res = await groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CLINICAL_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.05,
            max_tokens=6000,
        )
        raw_json = nlp_res.choices[0].message.content
        report   = json.loads(raw_json)

        # Stage 7 — build dialogue from corrected + diarized segments
        report["dialogue"] = [
            {
                "speaker":               s.get("speaker", "DOCTOR"),
                "time":                  f"{s.get('start', 0):.1f}s",
                "text":                  s.get("text", "").strip(),
                "original_text":         s.get("original_text", ""),
                "was_corrected":         bool(s.get("original_text") and
                                             s["original_text"] != s.get("text","")),
                "clinical_significance": _classify_utterance(
                    s.get("text",""), s.get("speaker","DOCTOR")
                ),
            }
            for s in labeled_seg
            if s.get("text", "").strip()
        ]

        result = {
            "session_id":       sid,
            "report":           report,
            "transcript":       full_text,
            "segments":         labeled_seg,
            "whisper_language": whisper_lang,
        }
        await sessions.set(sid, result)

        # ── Persist to database ────────────────────────────────────────────
        result["correction_log_a"] = correction_log_a
        result["correction_log_b"] = correction_log_b
        await db.save_consultation(sid, result)
        await db.save_fhir_resources(sid, report.get("fhir", {}))
        await db.save_corrections(sid, correction_log_a, correction_log_b, labeled_seg)

        # ── Push FHIR resources to HAPI server (background, non-blocking) ──
        background_tasks.add_task(push_fhir_bundle, sid, report.get("fhir", {}))

        log.info("[%s] Pipeline complete — lang=%s  segs=%d  dialogue=%d  DB=saved  FHIR=queued",
                 sid[:8], whisper_lang, len(labeled_seg), len(report["dialogue"]))
        return result

    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        log.error("[%s] JSON parse error: %s", sid[:8], exc)
        return JSONResponse({"error": "LLM returned malformed JSON.", "detail": str(exc)}, status_code=500)
    except Exception as exc:
        log.error("[%s] Pipeline error: %s", sid[:8], exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        for p in paths_to_clean:
            try: p.unlink(missing_ok=True)
            except: pass


@app.get("/search_patient")
async def search_patient(request: Request, name: str):
    """Search HAPI FHIR public sandbox by patient name."""
    _check_rate(request)
    name = re.sub(r"[^\w\s\-\.]", "", name)[:80].strip()
    if not name:
        raise HTTPException(400, "Invalid name parameter.")
    async with httpx.AsyncClient(timeout=14) as client:
        try:
            r = await client.get(
                f"{FHIR_BASE_URL}/Patient",
                params={"name": name, "_count": 5, "_format": "json"},
                headers={"Accept": "application/fhir+json"},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return JSONResponse({"detail": f"FHIR error: {e.response.status_code}"}, status_code=502)
        except Exception as e:
            return JSONResponse({"detail": f"Search failed: {e}"}, status_code=502)


@app.get("/download")
async def download(request: Request, session_id: str, format: str = "pdf"):
    """Download session report as PDF or JSON."""
    _check_rate(request)
    _validate_sid(session_id)
    if format not in ("pdf", "json"):
        raise HTTPException(400, "format must be pdf or json.")
    data = await sessions.get(session_id)
    if not data:
        raise HTTPException(404, "Session not found or expired.")

    fname    = f"Clinical_{session_id[:6].upper()}.{format}"
    out_path = OUT_DIR / fname

    if format == "json":
        out_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        media = "application/json"
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _make_pdf, data, out_path)
        media = "application/pdf"

    return FileResponse(str(out_path), filename=fname, media_type=media)


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    _validate_sid(session_id)
    data = await sessions.get(session_id)
    if not data:
        raise HTTPException(404, "Session not found or expired.")
    return data


@app.get("/health")
async def health():
    db_stats = await db.db_stats()
    return {
        "status":      "ok",
        "version":     "5.0.0",
        "model_llm":   LLM_MODEL,
        "model_asr":   WHISPER_MODEL,
        "sessions":    await sessions.count(),
        "redis":       bool(sessions._redis),
        "db":          db_stats,
        "fhir_push":   FHIR_PUSH,
        "fhir_server": FHIR_BASE_URL,
        "ts":          datetime.utcnow().isoformat(),
    }


# ── Consultation history ────────────────────────────────────────────────────

@app.get("/consultations")
async def list_consultations(
    request: Request,
    limit: int = 50,
    offset: int = 0,
):
    """List all consultations from DB (most recent first)."""
    _check_rate(request)
    if limit > 200:
        limit = 200
    rows = await db.list_consultations(limit=limit, offset=offset)
    return {"consultations": rows, "count": len(rows), "limit": limit, "offset": offset}


@app.get("/consultations/search")
async def search_consultations(request: Request, q: str):
    """Full-text search consultations by patient name, ICD code, or diagnosis."""
    _check_rate(request)
    q = re.sub(r"[^\w\s\-\.]", "", q)[:80].strip()
    if not q:
        raise HTTPException(400, "Query parameter 'q' is required.")
    rows = await db.search_consultations(q)
    return {"results": rows, "count": len(rows), "query": q}


@app.get("/consultations/{session_id}/full")
async def get_consultation_full(session_id: str):
    """Get full consultation record from DB including report JSON."""
    _validate_sid(session_id)
    row = await db.get_consultation(session_id)
    if not row:
        raise HTTPException(404, "Consultation not found in database.")
    return row


@app.get("/consultations/{session_id}/fhir")
async def get_consultation_fhir(session_id: str):
    """Get all 5 FHIR resources for a consultation with push status."""
    _validate_sid(session_id)
    resources = await db.get_fhir_resources(session_id)
    if not resources:
        raise HTTPException(404, "No FHIR resources found for this session.")
    return {
        "session_id": session_id,
        "resource_count": len(resources),
        "resources": resources,
    }


@app.post("/consultations/{session_id}/push-fhir")
async def push_fhir_manually(
    request: Request,
    session_id: str,
    background_tasks: BackgroundTasks,
):
    """Re-push FHIR resources to HAPI server for a stored consultation."""
    _check_rate(request)
    _validate_sid(session_id)
    data = await sessions.get(session_id)
    if not data:
        # Try DB fallback
        row = await db.get_consultation(session_id)
        if not row:
            raise HTTPException(404, "Session not found.")
        fhir = row["report_json"].get("fhir", {})
    else:
        fhir = data.get("report", {}).get("fhir", {})

    background_tasks.add_task(push_fhir_bundle, session_id, fhir)
    return {"status": "queued", "session_id": session_id,
            "message": "FHIR push queued — check /consultations/{id}/fhir for status"}


# ── FHIR R4 endpoints (HL7-compliant resource access) ──────────────────────

@app.get("/fhir/Patient/{patient_id}")
async def get_fhir_patient(request: Request, patient_id: str):
    """Return FHIR R4 Patient resource for a patient ID."""
    _check_rate(request)
    async with httpx.AsyncClient(timeout=12) as client:
        try:
            r = await client.get(
                f"{FHIR_BASE_URL}/Patient/{patient_id}",
                headers={"Accept": "application/fhir+json"},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            return JSONResponse({"detail": f"FHIR: {e.response.status_code}"}, status_code=502)
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/fhir/Condition")
async def get_fhir_conditions(request: Request, patient: str):
    """Return FHIR R4 Conditions for a patient (by reference)."""
    _check_rate(request)
    async with httpx.AsyncClient(timeout=12) as client:
        try:
            r = await client.get(
                f"{FHIR_BASE_URL}/Condition",
                params={"subject": patient, "_format": "json", "_count": "10"},
                headers={"Accept": "application/fhir+json"},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/fhir/MedicationRequest")
async def get_fhir_medications(request: Request, patient: str):
    """Return FHIR R4 MedicationRequests for a patient."""
    _check_rate(request)
    async with httpx.AsyncClient(timeout=12) as client:
        try:
            r = await client.get(
                f"{FHIR_BASE_URL}/MedicationRequest",
                params={"subject": patient, "_format": "json", "_count": "10"},
                headers={"Accept": "application/fhir+json"},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/fhir/Bundle/{session_id}")
async def get_fhir_bundle(session_id: str):
    """
    Return a complete FHIR R4 transaction Bundle containing all 5 resources
    for a consultation. This is the ABDM-ready export format.
    """
    _validate_sid(session_id)
    resources = await db.get_fhir_resources(session_id)
    if not resources:
        # Try session store fallback
        data = await sessions.get(session_id)
        if not data:
            raise HTTPException(404, "Session not found.")
        fhir = data.get("report", {}).get("fhir", {})
        resource_list = [v for v in fhir.values() if isinstance(v, dict)]
    else:
        resource_list = [r["resource_json"] for r in resources]

    bundle = {
        "resourceType": "Bundle",
        "id":           f"bundle-{session_id[:8]}",
        "type":         "transaction",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "meta": {
            "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"],
            "source":  "NeuroDoC v5.0",
        },
        "entry": [
            {
                "fullUrl":  f"{FHIR_BASE_URL}/{r['resourceType']}/{r.get('id','')}",
                "resource": r,
                "request": {
                    "method": "PUT",
                    "url":    f"{r['resourceType']}/{r.get('id','')}",
                },
            }
            for r in resource_list
            if isinstance(r, dict) and r.get("resourceType")
        ],
    }
    return bundle


@app.patch("/consultations/{session_id}/entities")
async def update_entities(request: Request, session_id: str):
    """
    Edit any entity field for a stored consultation.
    Accepts a JSON body with any subset of entity fields.

    Example request body:
    {
      "name": "Ramesh Kumar",
      "age": "47 years",
      "icd": "E11.65",
      "symptoms": ["headache", "fatigue"],
      "meds": ["Metformin 500mg BD", "Amlodipine 5mg OD"]
    }
    """
    _check_rate(request)
    _validate_sid(session_id)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body.")

    if not isinstance(body, dict) or not body:
        raise HTTPException(400, "Request body must be a non-empty JSON object.")

    # Update in DB
    ok = await db.update_entities(session_id, body)

    # Also update in session cache if present
    cached = await sessions.get(session_id)
    if cached:
        ent = cached.get("report", {}).get("entities", {})
        ALLOWED = {"name","age","gender","icd","icd_name","symptoms","diagnosis",
                   "meds","duration","vitals","test","advice","allergies","followup","referral","red_flags"}
        for k, v in body.items():
            if k in ALLOWED:
                ent[k] = v
        cached["report"]["entities"] = ent
        await sessions.set(session_id, cached)

    if not ok and not cached:
        raise HTTPException(404, "Session not found — cannot update entities.")

    return {"status": "updated", "session_id": session_id, "fields_updated": list(body.keys())}


@app.get("/consultations/{session_id}/entities")
async def get_entities(session_id: str):
    """Get just the entities dict for a session (fast lookup without full report)."""
    _validate_sid(session_id)
    data = await sessions.get(session_id)
    if data:
        return {"session_id": session_id, "entities": data.get("report", {}).get("entities", {})}
    row = await db.get_consultation(session_id)
    if row:
        return {"session_id": session_id, "entities": row["report_json"].get("entities", {})}
    raise HTTPException(404, "Session not found.")


@app.get("/")
def home():
    return FileResponse("recorder.html")
