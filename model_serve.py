import os
import re
import requests
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from model.agent_openai import build_agent
from model.utils import load_identiface, encode_image_from_file
import base64
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from fastapi import FastAPI

load_dotenv()
model = load_identiface()
client = OpenAI()
agent = build_agent(model, client)
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    image_path: str | None = None

class QueryResponse(BaseModel):
    output: str

@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):

    if request.image_path:
        encoded_image = encode_image_from_file(request.image_path)
        message = HumanMessage(content=[
            {"type": "text", "text": request.query},
            {"type": "image_url", "image_url": {"url": encoded_image}}
        ])
    else:
        message = HumanMessage(content=[
            {"type": "text", "text": request.query}
        ])

    response = agent.invoke(
        {"input": [message]},
        config={"configurable": {"session_id": request.session_id}}
    )

    return QueryResponse(output=response["output"])