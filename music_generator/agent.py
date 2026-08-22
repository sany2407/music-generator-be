from google.adk.agents import SequentialAgent

from .sub_agents.emotion_agent import emotion_analyzer
from .sub_agents.prompt_enhancer_agent import prompt_enhancer
from .sub_agents.music_generation_agent import music_generation_agent

root_agent = SequentialAgent(
    name="music_orchestrator",
    description=(
        "Emotion-aware music generation pipeline. Analyzes the emotions in a "
        "text description, crafts a detailed music prompt, and generates music "
        "with Google Lyria."
    ),
    sub_agents=[emotion_analyzer, prompt_enhancer, music_generation_agent],
)
