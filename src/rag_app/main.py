from fastapi import FastAPI
from pydantic import BaseModel
from anthropic import Anthropic, types
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()
client = Anthropic()

class InputData(BaseModel):
    role: str
    content: str

@app.post("/query")
def read_root(data: InputData) -> types.Message:
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {"role": data.role, "content": data.content}
        ]
    )
    return message

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()