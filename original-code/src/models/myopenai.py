import dotenv
from openai import OpenAI
from string import Template
from pathlib import Path

env_path = Path(__file__).parent.parent.parent.resolve().absolute()/".env"
dotenv.load_dotenv(env_path.as_posix())



client = OpenAI()


def openai_call(prompt):
    return client.responses.create(model="gpt-4o-mini", input=prompt)
