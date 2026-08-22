import base64
import json
import os
import re
import time

from google.adk.tools import ToolContext

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_music")

LYRIA3_MODELS = {
    "clip": "lyria-3-clip-preview",
    "pro": "lyria-3-pro-preview",
}
INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

VERTEX_LOCATIONS = ["us-central1", "us-east4", "europe-west1", "asia-southeast1"]


def _extract_lyria3_output(payload: dict) -> tuple[bytes | None, str]:
    audio = None
    texts = []
    for step in payload.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for block in step.get("content", []):
            btype = block.get("type")
            if btype == "audio" and block.get("data"):
                try:
                    audio = base64.b64decode(block["data"])
                except Exception:
                    pass
            elif btype == "text" and block.get("text"):
                texts.append(block["text"])
    return audio, "\n".join(texts).strip()


def _generate_via_lyria3(prompt: str, mode: str) -> tuple[bytes | None, str, str]:
    import requests

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "", "GOOGLE_API_KEY is not set."

    model = LYRIA3_MODELS[mode]
    try:
        response = requests.post(
            INTERACTIONS_URL,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json={"model": model, "input": prompt},
            timeout=300,
        )
    except Exception as exc:
        return None, "", f"Lyria 3 request failed: {exc}"

    if response.status_code != 200:
        return None, "", f"Lyria 3 API error {response.status_code}: {response.text[:300]}"

    audio, structure = _extract_lyria3_output(response.json())
    if not audio:
        return None, "", "Lyria 3 returned no audio data."
    return audio, structure, model


def _predict_vertex(project_id: str, location: str, prompt: str) -> dict:
    from google.auth.transport.requests import AuthorizedSession
    import google.auth

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/lyria-002:predict"
    )
    body = {"instances": [{"prompt": prompt}], "parameters": {"sampleCount": 1}}
    response = session.post(url, data=json.dumps(body), headers={"Content-Type": "application/json"})
    if response.status_code != 200:
        raise RuntimeError(f"Lyria API error {response.status_code}: {response.text[:300]}")
    return response.json()


def _generate_via_vertex(prompt: str) -> tuple[bytes | None, str, str]:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return None, "", "GOOGLE_CLOUD_PROJECT is not set."

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if location == "global":
        location = "us-central1"
    locations = [location] + [loc for loc in VERTEX_LOCATIONS if loc != location]

    last_error = None
    for loc in locations:
        try:
            result = _predict_vertex(project_id, loc, prompt)
            predictions = result.get("predictions", [])
            if not predictions:
                last_error = "Vertex Lyria returned no predictions."
                continue
            audio_b64 = predictions[0].get("bytesBase64Encoded")
            if not audio_b64:
                last_error = "Vertex Lyria response contained no audio bytes."
                continue
            return base64.b64decode(audio_b64), "", f"lyria-002@{loc}"
        except Exception as exc:
            last_error = str(exc)
            continue
    return None, "", f"All regions failed. Last error: {last_error}"


def generate_music_with_lyria(
    prompt: str,
    tool_context: ToolContext,
    duration_mode: str = "clip",
) -> dict:
    """Generates music from a text prompt using Google Lyria 3.

    Uses the Gemini API Interactions endpoint. Falls back to the Vertex AI
    lyria-002 model if Lyria 3 is unavailable.

    Args:
        prompt: Detailed music description (genre, mood, instruments, tempo).
        tool_context: ADK tool context used to access session state.
        duration_mode: "clip" for a 30-second clip, or "pro" for a
            full-length song with verses, choruses and bridges.

    Returns:
        A dict with 'status', 'file_path', 'engine', 'duration_mode',
        'structure_or_lyrics' or 'error_message'.
    """
    mode = "pro" if str(duration_mode).lower()[:3] in ("pro", "ful", "son") else "clip"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_prompt = re.sub(r"[^a-zA-Z0-9]+", "_", prompt[:40]).strip("_").lower()
    stamp = int(time.time())

    audio, structure, engine = _generate_via_lyria3(prompt, mode)
    ext = "mp3"
    if audio is None:
        fallback_error = engine
        audio, structure, engine = _generate_via_vertex(prompt)
        ext = "wav"
        if audio is None:
            return {
                "status": "error",
                "error_message": f"Lyria 3 failed ({fallback_error}). Vertex fallback failed too: {engine}",
            }

    prefix = "lyria3" if engine.startswith("lyria-3") else "lyria"
    file_name = f"{prefix}_{mode}_{safe_prompt}_{stamp}.{ext}"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    with open(file_path, "wb") as f:
        f.write(audio)

    tool_context.state["last_generated_music_path"] = file_path
    tool_context.state["last_generated_lyrics"] = structure
    return {
        "status": "success",
        "file_path": file_path,
        "engine": engine,
        "duration_mode": mode,
        "structure_or_lyrics": structure[:1500],
        "prompt_used": prompt,
    }
