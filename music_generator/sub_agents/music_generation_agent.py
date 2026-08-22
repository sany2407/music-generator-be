from google.adk.agents import LlmAgent

from ..tools.lyria_tool import generate_music_with_lyria
from ..tools.cover_tool import generate_album_cover

music_generation_agent = LlmAgent(
    name="MusicGenerationAgent",
    model="gemini-3.5-flash",
    description="Generates music from a text prompt using Google Lyria 3, and album cover art on request.",
    instruction="""
You are a music generation agent powered by Google Lyria 3.

You will receive an enhanced music prompt and the user's chosen duration mode:

Enhanced prompt: {enhanced_prompt}
Duration mode: {music_mode}

First, call the 'generate_music_with_lyria' tool with:
- prompt: the enhanced prompt above
- duration_mode: exactly the value of Duration mode ("clip" for a 30-second
  clip, "pro" for a full-length song)

Then, if the user's original wish included any request for artwork — words like
album cover, cover art, artwork, image, poster or a visual description of how
the cover should look — ALSO call the 'generate_album_cover' tool with:
- cover_description: the user's visual wishes verbatim (imagery, colors,
  atmosphere). Leave empty only if they asked for a cover without describing it.

After the tools return, report to the user:
- Whether the music is a 30-second clip or a full-length song, and which engine was used.
- If structure_or_lyrics was returned, summarize it in one sentence.
- If a cover was generated, confirm it in one sentence (do not repeat the image prompt).
- Never mention file paths to the user; they are internal details.
""",
    tools=[generate_music_with_lyria, generate_album_cover],
)
