from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

result = model.invoke("Write me a poem about AI and nature")
print(result.content)  # ✅ Only prints the poem text
