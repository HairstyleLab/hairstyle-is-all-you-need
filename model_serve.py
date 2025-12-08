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
import json
import asyncio
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from queue import Queue
from threading import Thread

load_dotenv()
model = load_identiface()
client = OpenAI()
agent = build_agent(model, client)
app = FastAPI()

# 전역 상태 큐 (세션별로 관리)
status_queues = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    image_path: str | None = None

class QueryResponse(BaseModel):
    output: str
    generated_image: str | None = None  # 생성된 이미지 (base64)

# SSE 스트리밍 엔드포인트
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """SSE를 통한 실시간 상태 스트리밍"""

    async def event_generator():
        # 세션별 큐 생성
        queue = Queue()
        status_queues[request.session_id] = queue

        # 상태 전송 함수
        def send_status(status: str):
            queue.put({"type": "status", "message": status})

        # Agent에 상태 콜백 설정
        agent.status_callback = send_status

        # 백그라운드에서 agent 실행
        def run_agent():
            try:
                send_status("응답 수신 중...")

                if request.image_path:
                    if request.image_path.startswith("data:"):
                        encoded_image = request.image_path
                        send_status("이미지 분석 중...")
                    else:
                        encoded_image = encode_image_from_file(request.image_path)
                        send_status("이미지 분석 중...")

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

                send_status("답변 정리 중...")

                # 생성된 이미지 확인
                generated_image = None
                if hasattr(agent, 'gen_flag') and agent.gen_flag and agent.current_image_base64:
                    generated_image = f"data:image/jpeg;base64,{agent.current_image_base64}"
                    agent.gen_flag = False

                # 최종 응답 전송
                queue.put({
                    "type": "response",
                    "output": response["output"],
                    "generated_image": generated_image
                })

            except Exception as e:
                queue.put({"type": "error", "message": str(e)})
            finally:
                queue.put({"type": "done"})

        # 백그라운드 스레드 시작
        thread = Thread(target=run_agent)
        thread.start()

        # 큐에서 이벤트를 읽어 SSE 형식으로 전송
        while True:
            if not queue.empty():
                event = queue.get()

                if event["type"] == "done":
                    break

                # SSE 형식으로 전송
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.1)

        # 큐 정리
        if request.session_id in status_queues:
            del status_queues[request.session_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):

    if request.image_path:
        # Django에서 이미 base64 인코딩된 이미지를 받으므로 파일 읽기 불필요
        # image_path가 "data:image/jpeg;base64,..." 형식으로 전달됨
        if request.image_path.startswith("data:"):
            # 이미 인코딩된 이미지를 그대로 사용
            encoded_image = request.image_path
            print(f"Base64 인코딩된 이미지 수신 완료 (크기: {len(request.image_path)} bytes)")
        else:
            # 혹시 파일 경로가 전달된 경우 (레거시 지원)
            encoded_image = encode_image_from_file(request.image_path)
            print(f"파일에서 이미지 인코딩 완료: {request.image_path}")

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

    # 생성된 이미지가 있는지 확인
    generated_image = None
    if hasattr(agent, 'gen_flag') and agent.gen_flag and agent.current_image_base64:
        # 이미지 생성이 완료된 경우
        generated_image = f"data:image/jpeg;base64,{agent.current_image_base64}"
        print(f"생성된 이미지를 응답에 포함 (크기: {len(generated_image)} bytes)")
        # 플래그 리셋
        agent.gen_flag = False

    return QueryResponse(
        output=response["output"],
        generated_image=generated_image
    )