import os
import re
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from model.agent import build_agent
from model.utils import load_identiface
import base64
from langchain_core.messages import HumanMessage
from PIL import Image



# 환경 변수 로드
load_dotenv()

# 이미지 경로 설정
file_path = os.path.abspath("images/face1.jpg")

# 이미지 추출 및 표시 함수
def extract_and_display_image(response,image_path=os.path.abspath("images")):
    output = response.get('output', '')
    
    url_pattern = r'https://oaidalleapiprodscus[^\s\)]+\.png[^\s\)]*'
    urls = re.findall(url_pattern, output)
    
    if urls:
        for i, url in enumerate(urls):
            img_response = requests.get(url)
            img = Image.open(BytesIO(img_response.content))
            
            filename = f"generated_hairstyle_{i+1}.png"
            img.save(os.path.join(image_path, filename))
            
            img.show()
        
# 이미지 인코딩 함수
def encode_image_from_file(file_path):
    with open(file_path, "rb") as image_file:
        image_content = image_file.read()
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif file_ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/unknown"
        return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"


model = load_identiface()
agent = build_agent(model)
encoded_image = encode_image_from_file(file_path)

#테스트1
print("\n\n" + "=" * 50)
print("테스트 1: 관련없는 질의")
print("=" * 50)
response = agent.invoke(
    {"input": [HumanMessage(content="오늘 날씨 어때?")]},
    config={"configurable": {"session_id": "test1"}}
)
print("\n응답:")
print(response['output'])

#테스트2
print("\n\n" + "=" * 50)
print("테스트 2: 일반 헤어스타일 추천")
print("=" * 50)
response = agent.invoke(
    {"input": [HumanMessage(content="요즘 유행하는 헤어스타일 추천해줘")]},
    config={"configurable": {"session_id": "test2"}}
)
print("\n응답:")
print(response['output'])

#테스트3
print("\n\n" + "=" * 50)
print("테스트 3: 겨울 헤어스타일 추천")
print("=" * 50)
response = agent.invoke(
    {"input": [HumanMessage(content="겨울에 어울리는 헤어스타일 추천해줘")]},
    config={"configurable": {"session_id": "test3"}}
)
print("\n응답:")
print(response['output'])

#테스트4
print("\n\n" + "=" * 50)
print("테스트 4: 존재하지 않는 헤어스타일")
print("=" * 50)
response = agent.invoke(
    {"input": [HumanMessage(content="애플펜슬 헤어스타일에 대해 알려줘")]},
    config={"configurable": {"session_id": "test4"}}
)
print("\n응답:")
print(response['output'])


#테스트5
print("\n\n" + "=" * 50)
print("테스트 5: 이미지 분석")
response = agent.invoke(
    {"input": [HumanMessage(
        content=[
            {"type": "text", "text": f"이 이미지를 분석해줘. 정밀 분석이 필요하면 이 base64 데이터를 사용해: {encoded_image}"},
            {"type": "image_url", "image_url": {"url": encoded_image}}
        ]
    )]},
    config={"configurable": {"session_id": "test5"}}        
)
print("\n응답:")
print(response['output'])
print("\n\n테스트 완료!")

#테스트6
print("\n\n" + "=" * 50)
print("테스트 6: DALL-E 이미지 생성")
response = agent.invoke(
    {"input": [HumanMessage(
        content=[
            {"type": "text", "text": f"이 이미지에서 헤어스타일만 히피펌으로 바꿔줘. 이 base64 데이터를 사용해: {encoded_image}"},
            {"type": "image_url", "image_url": {"url": encoded_image}}
        ]
    )]},
    config={"configurable": {"session_id": "test6"}}        
)
print("\n응답:")
print(response['output'])
print("\nDALL-E 생성 이미지 추출 중...")
extract_and_display_image(response)



print("\n\n테스트 완료!")
