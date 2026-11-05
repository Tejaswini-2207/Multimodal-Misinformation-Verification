
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are an AI research assistant."
            },
            {
                "role": "user",
                "content": "Explain multimodal misinformation detection in two sentences."
            }
        ],
        temperature=0.2,
        max_completion_tokens=200
    )

    print("\nGROQ CONNECTION SUCCESSFUL!")
    print("\nResponse:")
    print(response.choices[0].message.content)

except Exception as error:
    print("\nGROQ API ERROR:")
    print(error)
