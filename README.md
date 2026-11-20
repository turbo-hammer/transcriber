# GigaAM v3 E2E RNN-T HTTP API

CPU-first FastAPI wrapper for https://github.com/salute-developers/GigaAM.

## Start

```sh
docker compose up --build -d
docker compose logs -f asr
```

The first startup downloads model weights. Subsequent starts reuse the named Docker volume.

## Transcribe a WAV file

```sh
curl -fsS -X POST http://localhost:8000/transcribe -F 'file=@recording.wav'
```

Example response:

```json
{"text":"Добрый день.","model":"v3_e2e_rnnt","duration_seconds":2.1}
```

API docs: http://localhost:8000/docs

## Long audio (>25 s)

1. Accept the conditions for https://huggingface.co/pyannote/segmentation-3.0.
2. Create a Hugging Face read token.
3. Put `HF_TOKEN=hf_...` in `.env` (never commit this file).
4. Restart with `docker compose up -d`.

The image installs the GigaAM `longform` dependency extra by default.

## Notes

- `.wav` uploads only; input format converted with ffmpeg to mono PCM at 16 kHz.
- Default upload limit 100 MB and audio length 3600 seconds; change environment variables in compose.yaml.
- Requests run one inference at a time per process.
- Binds host port to 127.0.0.1 for local access. Protect with authentication and HTTPS before external exposure.
- This example has not been smoke-tested with actual model weights in this environment.
