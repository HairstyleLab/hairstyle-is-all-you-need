import os
import re
import requests
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from model.agent_openai import build_agent
# from model.agent_hf import build_agent
from model.utils import load_identiface
import base64
from langchain_core.messages import HumanMessage
from PIL import Image
# from model.utils import load_hairfastgan, generate_hairstyle
from model.utils import load_identiface, get_face_shape_and_gender

file_path = "images/dw2.jpg"
load_dotenv()

# model = load_hairfastgan()
# result = generate_hairstyle(model, face_img, shape_img, color_img)

model = load_identiface()
client = OpenAI()

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

agent = build_agent(model, client)
encoded_image = encode_image_from_file(file_path)

def make_human_message(input_text,session_id,file_path=None):
    if not file_path:
        response = agent.invoke(
            {"input": [HumanMessage(content=[
                {"type": "text", "text": input_text}
            ])]},
            config={"configurable": {"session_id": session_id}}
        )
        print(response['output'])
        
    else:
        response = agent.invoke(
            {"input": [HumanMessage(content=[
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": encoded_image}}
            ])]},
            config={"configurable": {"session_id": session_id}}
        )
        print(response['output'])

q1 = "이 얼굴에 히피펌 헤어스타일에 애쉬그레이 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성시, 헤어스타일, 헤어컬러 둘 다 명시한 경우)
q2 = "이 얼굴에 히피펌 헤어스타일을 적용한 이미지를 생성해줄래?" # (이미지 생성시, 헤어스타일만 명시한 경우)
q3 = "이 얼굴에 애쉬그레이를 적용한 이미지를 생성해줄래?" # (이미지 생성시, 헤어컬러만 명시한 경우)
q4 = "이 얼굴에 히피펌 헤어스타일에 애쉬그레이 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성 시 이미지 업로드 안한 경우)
q5 = "이 얼굴에 히피 펆 헤어스타일에 애시 그래 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성 시 헤어스타일, 헤어컬러를 살짝 틀린 문자열로 표현한 경우)
q6 = "이 얼굴에 마이쮸펌 헤어스타일에 칙칙한 초코칩 컬러를 적용한 이미지를 생성해줄래?" # (이미지 생성 시 등록된 헤어스타일, 헤어컬러가 아닌 옵션으로 표현한 경우)

# print(make_human_message("이 머리에 어울리는 헤어스타일 추천해줘", session_id="test_session1", file_path=file_path))
print(make_human_message(q5, session_id="test_session2", file_path=file_path))
# print("\n\n테스트 완료!")

