import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from music_generator.agent import root_agent

APP_NAME = "music_generator_api"
GENERATED_DIR = Path(__file__).parent / "music_generator" / "generated_music"

app = FastAPI(title="Music Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session_service = InMemorySessionService()
runner: Runner | None = None


@app.on_event("startup")
async def startup() -> None:
    global runner
    runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)


class GenerateRequest(BaseModel):
    text: str
    mode: str = "clip"


class GenerateResponse(BaseModel):
    session_id: str
    message: str
    emotion_analysis: str
    enhanced_prompt: str
    track_url: str | None
    track_name: str | None
    lyrics: str | None


def _latest_track() -> Path | None:
    if not GENERATED_DIR.exists():
        return None
    files = [p for pat in ("*.wav", "*.mp3") for p in GENERATED_DIR.glob(pat)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text prompt is required.")
    if runner is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    user_id = "web_user"
    session_id = f"s_{int(time.time() * 1000)}"
    mode = req.mode if req.mode in ("clip", "pro") else "clip"
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={"music_mode": mode},
    )

    content = types.Content(role="user", parts=[types.Part(text=req.text.strip())])

    final_message = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_message = event.content.parts[0].text or ""

    state = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    emotion = (state.state.get("emotion_analysis") or "") if state else ""
    prompt = (state.state.get("enhanced_prompt") or "") if state else ""
    lyrics = (state.state.get("last_generated_lyrics") or "") if state else ""

    track = _latest_track()
    return GenerateResponse(
        session_id=session_id,
        message=final_message.strip(),
        emotion_analysis=emotion,
        enhanced_prompt=prompt,
        track_url=f"/api/audio/{track.name}" if track else None,
        track_name=track.name if track else None,
        lyrics=lyrics or None,
    )


@app.get("/api/tracks")
async def tracks() -> list[dict]:
    if not GENERATED_DIR.exists():
        return []
    items = []
    files = [p for pat in ("*.wav", "*.mp3") for p in GENERATED_DIR.glob(pat)]
    for f in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        items.append(
            {
                "name": f.stem.replace("lyria_", "").replace("lyria3_", "").replace("_", " ").strip(),
                "file": f.name,
                "url": f"/api/audio/{f.name}",
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": stat.st_mtime,
            }
        )
    return items


@app.get("/api/audio/{filename}")
async def audio(filename: str) -> FileResponse:
    path = GENERATED_DIR / filename
    if not path.exists() or path.suffix.lower() not in (".wav", ".mp3"):
        raise HTTPException(status_code=404, detail="Track not found.")
    media = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media, filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
