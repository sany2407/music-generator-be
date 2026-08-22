import base64
import os
import time
from pathlib import Path

from google.adk.tools import ToolContext

COVERS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "generated_music" / "covers"
IMAGE_MODEL = "gemini-2.5-flash-image"
VERTEX_LOCATIONS = ["us-central1", "us-east4", "europe-west1", "asia-southeast1"]


def _build_image_prompt(description: str, style_context: str) -> str:
    parts = [
        "Album cover artwork for a music track.",
        f"Music style and mood: {style_context}.",
        f"Artwork description: {description}.",
        "Square 1:1 composition, striking and professional, no text or lettering in the image.",
    ]
    return " ".join(parts)


def _generate_image_via_vertex(project_id: str, location: str, prompt: str) -> bytes:
    import json

    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/publishers/google/models/{IMAGE_MODEL}:generateContent"
    )
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    response = session.post(url, data=json.dumps(body), headers={"Content-Type": "application/json"})
    if response.status_code != 200:
        raise RuntimeError(f"Image API error {response.status_code}: {response.text[:300]}")
    for part in response.json()["candidates"][0]["content"]["parts"]:
        inline = part.get("inlineData")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"])
    raise RuntimeError("Image model returned no image data.")


def _generate_image(prompt: str) -> tuple[bytes | None, str]:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return None, "GOOGLE_CLOUD_PROJECT is not set."

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if location == "global":
        location = "us-central1"
    locations = [location] + [loc for loc in VERTEX_LOCATIONS if loc != location]

    last_error = None
    for loc in locations:
        try:
            return _generate_image_via_vertex(project_id, loc, prompt), f"{IMAGE_MODEL}@{loc}"
        except Exception as exc:
            last_error = str(exc)
            continue
    return None, f"All regions failed. Last error: {last_error}"


def _upload_cover(data: bytes, object_name: str) -> str | None:
    bucket_name = os.getenv("GCS_BUCKET", "")
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        blob.upload_from_string(data, content_type="image/png")
        return f"https://storage.googleapis.com/{bucket_name}/{object_name}"
    except Exception:
        return None


def generate_album_cover(
    cover_description: str,
    tool_context: ToolContext,
) -> dict:
    """Generates an album cover image matching the music's style.

    Call this AFTER generating the music when the user asks for album art.
    The artwork reflects both the user's visual wishes and the generated
    music's style and mood.

    Args:
        cover_description: What the cover should look like (imagery,
            colors, atmosphere) as wished by the user. May be empty to
            let the style alone drive the artwork.
        tool_context: ADK tool context used to access session state.

    Returns:
        A dict with 'status', 'cover_url' or 'error_message', and
        'image_prompt'.
    """
    enhanced_prompt = tool_context.state.get("enhanced_prompt") or ""
    emotion_analysis = tool_context.state.get("emotion_analysis") or ""
    style_context = f"{emotion_analysis} {enhanced_prompt}".strip() or "instrumental music"

    prompt = _build_image_prompt(cover_description.strip(), style_context)
    data, engine_or_error = _generate_image(prompt)
    if data is None:
        return {
            "status": "error",
            "error_message": f"Cover generation failed ({engine_or_error}). Music is unaffected.",
        }

    track_path = tool_context.state.get("last_generated_music_path")
    if track_path:
        object_name = f"{Path(str(track_path)).stem}_cover.png"
    else:
        object_name = f"cover_{int(time.time())}.png"

    url = _upload_cover(data, object_name)

    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = COVERS_DIR / object_name
    with open(local_path, "wb") as f:
        f.write(data)

    tool_context.state["last_generated_cover_url"] = url or str(local_path)
    return {
        "status": "success",
        "cover_url": url,
        "file_path": str(local_path),
        "engine": engine_or_error,
        "image_prompt": prompt[:500],
    }
