# Music Generator — Google ADK + Gemini + Lyria

An emotion-aware music generation multi-agent system built with **Google ADK (Agent Development Kit)**.

## Agents

| # | Agent | Model | Role |
|---|-------|-------|------|
| 1 | `EmotionAnalyzer` | Gemini 2.0 Flash | Detects emotions, intensity, and valence in the user's text |
| 2 | `PromptEnhancer` | Gemini 2.0 Flash | Converts emotion analysis into a rich Lyria music prompt |
| 3 | `MusicGenerationAgent` | Gemini 2.0 Flash | Calls the Lyria tool to generate the actual music |
| 4 | `music_orchestrator` (root) | Sequential pipeline | Chains: Emotion → Prompt → Music |

## Pipeline

```
User text ──> EmotionAnalyzer ──> PromptEnhancer ──> MusicGenerationAgent ──> WAV file
             (Gemini)            (Gemini)            (Lyria via Vertex AI)
```

Generated audio is saved to `music_generator/generated_music/`.

## Setup

1. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in:
   - `GOOGLE_API_KEY` — Gemini API key (for the agents)
   - `GOOGLE_CLOUD_PROJECT` — GCP project ID with Vertex AI enabled (for Lyria)
   - `GOOGLE_CLOUD_LOCATION` — region supporting Lyria (default `us-central1`)

3. Authenticate for Vertex AI (Lyria):

   ```powershell
   gcloud auth application-default login
   ```

## Run

```powershell
# CLI mode
python main.py "a rainy evening walking alone through old city streets"

# Interactive
python main.py

# ADK Dev UI (optional)
adk web
```
4. Architecture
   ![sany2407/music-generator-be — Main System Architecture](https://datadef.io/api/embed/sany2407music-generator-be-main-fc7a24cd)


- Uses the `lyria-002` model on Vertex AI (`predict` endpoint).
- Returns ~30 second instrumental audio as base64, saved as `.wav`.
- The tool automatically retries across Lyria-supported regions.
