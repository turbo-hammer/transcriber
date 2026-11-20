import asyncio
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import gigaam
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

MODEL_NAME = "v3_e2e_rnnt"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_DURATION_SECONDS = float(os.getenv("MAX_DURATION_SECONDS", "3600"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


class TranscriptionResponse(BaseModel):
    text: str
    model: str
    duration_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.model = gigaam.load_model(MODEL_NAME)
    app.state.inference_semaphore = asyncio.Semaphore(1)
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(title="GigaAM ASR API", version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ready"}


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise ValueError("Cannot read audio file")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise ValueError("Audio duration is unavailable") from exc
    if not 0 < duration <= MAX_DURATION_SECONDS:
        raise ValueError(f"Duration must be between 0 and {MAX_DURATION_SECONDS:g} seconds")
    return duration


def normalize_wav(source: Path, destination: Path):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(destination)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise ValueError("Invalid or unsupported audio file")


def recognize(model, wav_path: Path, duration: float) -> str:
    if duration <= 25:
        return model.transcribe(str(wav_path))
    if not os.getenv("HF_TOKEN"):
        raise ValueError("Long audio requires HF_TOKEN and access to pyannote/segmentation-3.0")
    segments = model.transcribe_longform(str(wav_path))
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=422, detail="A .wav file is required")

    with tempfile.TemporaryDirectory(prefix="asr-") as temp_dir:
        source = Path(temp_dir) / "uploaded.wav"
        wav = Path(temp_dir) / "normalized.wav"
        size = 0
        try:
            with source.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Upload is too large")
                    out.write(chunk)
        finally:
            await file.close()

        try:
            duration = await run_in_threadpool(probe_duration, source)
            await run_in_threadpool(normalize_wav, source, wav)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=422, detail="Audio processing timeout") from exc

        if duration > 25 and not os.getenv("HF_TOKEN"):
            raise HTTPException(
                status_code=422,
                detail="Audio longer than 25 seconds requires HF_TOKEN and pyannote access",
            )

        async with app.state.inference_semaphore:
            # PyTorch inference is synchronous; keep the event loop responsive.
            text = await run_in_threadpool(recognize, app.state.model, wav, duration)

    return TranscriptionResponse(
        text=text, model=MODEL_NAME, duration_seconds=round(duration, 2)
    )
