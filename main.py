import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from music_generator.agent import root_agent

APP_NAME = "music_generator"
USER_ID = "local_user"


async def run(text: str) -> None:
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)

    content = types.Content(role="user", parts=[types.Part(text=text)])

    print("\n--- Pipeline started ---\n")
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)

    final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session.id)
    print("\n--- Emotion analysis ---")
    print(final_session.state.get("emotion_analysis", "n/a"))
    print("\n--- Enhanced prompt ---")
    print(final_session.state.get("enhanced_prompt", "n/a"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = input("Describe the music you want: ")
    asyncio.run(run(user_text))
